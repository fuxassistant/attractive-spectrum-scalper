import json
import numpy as np
import pytest

from spectrum.cognition.execution import (
    ExecutionReplay, ExecutionConfig, QuoteTape, FEATURE_NAMES,
    _INTEGER_NAMES, _FLOAT_NAMES, execution_contract,
)

NS = 10**9


def tape(seconds, bids=None, asks=None, valid=None, epoch=None, end=None, cutoff=10**6, deadline=10**6):
    times = np.rint(np.asarray(seconds) * NS).astype(np.int64)
    n = len(times)
    bids = np.asarray(bids if bids is not None else [2000.] * n)
    asks = np.asarray(asks if asks is not None else bids + .1)
    return QuoteTape(times, np.rint(bids * NS).astype(np.int64),
                     np.rint(asks * NS).astype(np.int64),
                     np.asarray(valid if valid is not None else [True] * n),
                     np.asarray(epoch if epoch is not None else [1] * n, dtype=np.int32),
                     int((end if end is not None else seconds[-1] + 1) * NS),
                     entry_cutoff_ns=int(cutoff * NS), deadline_ns=int(deadline * NS))


def replay(t, **kwargs):
    return ExecutionReplay(t, ExecutionConfig(plans=((100000, 100000, 10000),), **kwargs))


def enter(r, side=1, qty=1):
    assert r.open_at(0, side, 0, qty)
    r.advance_until(int(.2 * NS))
    assert r.state.side == side
    return r


def test_schema_kernel_indices():
    assert _INTEGER_NAMES[0] == 'side'
    assert _INTEGER_NAMES[11] == 'quote_cursor'
    assert _INTEGER_NAMES[32] == 'last_valid'
    assert _FLOAT_NAMES[4] == 'qty'
    assert _FLOAT_NAMES[16] == 'min_equity'
    assert len(_INTEGER_NAMES) == 33 and len(_FLOAT_NAMES) == 17


@pytest.mark.parametrize('side', [1, -1])
def test_both_sides_fee_reserved_once_and_executable_close(side):
    t = tape([0, .1, .2, .3, .4, .5], [2000, 2000, 2000, 2001, 2001, 2001])
    r = enter(replay(t), side, 2)
    assert r.state.cash == pytest.approx(1499.4)
    assert r.state.equity == pytest.approx(1499.2)
    assert r.request_close(int(.2 * NS))
    r.advance_until(int(.3 * NS))
    assert r.state.side == side  # exactly ready is not eligible
    r.advance_until(int(.4 * NS))
    assert r.state.side == 0
    expected = 1.2 if side > 0 else -2.8
    assert r.state.cash == pytest.approx(1500 + expected)
    assert r.state.last_trade_net == pytest.approx(expected)
    assert r.state.total_fees == pytest.approx(.6)
    assert r.state.exit_reason == 'policy_close'


def test_strict_entry_latency_duplicate_publication_times():
    r = replay(tape([0, .1, .1, .100000001, .2]))
    assert r.open_at(0, 1)
    r.advance_until(100_000_000)
    assert r.state.side == 0
    r.advance_until(100_000_001)
    assert r.state.entry_ns == 100_000_001
    assert r.state.quote_cursor == 4


def test_pending_cannot_be_replaced_or_new_entry_opened():
    r = replay(tape([0, .2, .4, .6]))
    assert r.open_at(0, 1)
    assert not r.open_at(0, -1)
    assert r.state.pending_side == 1
    r.advance_until(int(.2 * NS))
    assert r.request_close(int(.2 * NS))
    snapshot = r.snapshot()
    assert not r.request_close(int(.2 * NS))
    assert not r.open_at(int(.2 * NS), -1)
    assert r.snapshot() == snapshot


def test_entry_and_close_timeout_inclusive():
    r = replay(tape([0, 2., 4.]))
    assert r.open_at(0, 1)
    r.advance_until(2 * NS)
    assert r.state.side == 1
    assert r.request_close(2 * NS)
    r.advance_until(4 * NS)
    assert r.state.exit_ns == 4 * NS
    late = replay(tape([0, 2.000000001, 4]))
    assert late.open_at(0, 1)
    late.advance_until(3 * NS)
    assert late.state.side == 0 and late.state.pending_kind == 0


@pytest.mark.parametrize('bad_kind', ['validity', 'epoch'])
def test_support_loss_censors_no_invented_fill(bad_kind):
    r = replay(tape([0, .2, .4, .6], [2000, 2000, 1000, 3000],
                    valid=[True, True, bad_kind != 'validity', True],
                    epoch=[1, 1, 2 if bad_kind == 'epoch' else 1, 2]))
    enter(r)
    supported_equity = r.state.equity
    r.request_close(int(.2 * NS))
    r.advance_until(NS)
    assert r.state.censored and not r.state.dead
    assert r.state.side == 1 and r.state.exit_ns == 0
    assert r.state.equity == supported_equity
    assert r.state.cash == pytest.approx(1499.7)
    assert r.state.exit_reason == 'support_loss'
    assert r.state.quote_cursor == 3


def test_silence_is_not_support_loss_and_terminal_exposure_censors():
    r = enter(replay(tape([0, .2, 100], end=200)))
    r.advance_until(90 * NS)
    assert not r.state.censored
    assert r.state.side == 1
    r.advance_until(199 * NS)
    assert not r.state.censored
    r.advance_until(200 * NS)
    assert r.state.censored and r.state.exit_reason == 'end_of_tape'
    assert r.state.side == 1 and r.state.trades == 0


def test_reference_horizon_unresolved_is_not_realized_profit():
    r = enter(replay(tape([0, .2, 100], end=200)))
    r.reference_exit(10 * NS)
    assert r.state.censored and r.state.exit_reason == 'horizon_unresolved'
    assert r.state.trades == 0 and r.state.exit_ns == 0


def test_snapshot_clone_serialization_exact_and_shared_tape():
    r = enter(replay(tape([0, .2, .4, .6, .8, 1], [2000, 2000, 2001, 1999, 2001, 2002])))
    r.advance_until(int(.4 * NS))
    snapshot = json.loads(json.dumps(r.snapshot()))
    clone = r.clone()
    restored = ExecutionReplay(r.tape, r.config).restore(snapshot)
    assert clone.tape is r.tape
    assert np.shares_memory(clone.tape.bid, r.tape.bid)
    assert not clone.tape.bid.flags.writeable
    for branch in (r, clone, restored):
        branch.request_close(int(.4 * NS))
        branch.advance_until(NS)
    assert r.snapshot() == clone.snapshot() == restored.snapshot()
    clone.state.cash += 1
    assert clone.state.cash != r.state.cash


def test_snapshot_refuses_other_tape_and_execution_config():
    from dataclasses import replace
    original = enter(replay(tape([0, .2, .4, .6])))
    saved = json.loads(json.dumps(original.snapshot()))
    restored = ExecutionReplay(original.tape, original.config).restore(saved)
    assert json.loads(json.dumps(restored.snapshot())) == saved
    other = ExecutionReplay(replace(original.tape, identity='different-certified-source'), original.config)
    with pytest.raises(ValueError, match='tape differs'):
        other.restore(saved)
    changed_cost = ExecutionReplay(original.tape, replace(original.config, fee_roundtrip_per_oz=.4))
    with pytest.raises(ValueError, match='configuration differs'):
        changed_cost.restore(saved)


def test_prefix_state_invariant_to_future_and_chunk_boundaries():
    a = replay(tape([0, .2, .4, .6, 1], [2000, 2000, 2001, 1999, 1]))
    b = replay(tape([0, .2, .4, .6, 1], [2000, 2000, 2001, 1999, 9999]))
    enter(a)
    enter(b)
    a.advance_until(int(.6 * NS))
    b.advance_until(int(.3 * NS))
    b.advance_until(int(.4 * NS))
    b.advance_until(int(.6 * NS))
    assert a.snapshot() == b.snapshot()
    np.testing.assert_array_equal(a.position_features(), b.position_features())
    assert len(a.position_features()) == len(FEATURE_NAMES)
    assert a.state.position_mfe == pytest.approx(.6)
    assert a.state.position_mae == pytest.approx(-1.4)


@pytest.mark.parametrize('last_valid', [True, False])
def test_flat_fast_forward_matches_incremental_prefix(last_valid):
    t = tape([0, 1, 2, 3, 4], [2000, 2001, 2002, 2003, 2004],
             valid=[True, False, True, False, last_valid], epoch=[1, -1, 2, -1, 3])
    fast, slow = replay(t), replay(t)
    fast.advance_until(4 * NS)
    for k in range(5):
        slow.advance_until(k * NS)
    assert fast.snapshot() == slow.snapshot()
    assert fast.state.cash == fast.state.equity == 1500
    assert fast.state.quote_cursor == 5
    assert fast.state.last_bid_nano == int((2004 if last_valid else 2002) * NS)


def test_death_strictly_greater_than_half_series_base():
    r = enter(replay(tape([0, .2, .4, .6], [2000, 2000, 1250.4, 1250.3])))
    r.advance_until(int(.4 * NS))
    assert r.state.equity == pytest.approx(750)
    assert not r.state.dead
    r.advance_until(int(.6 * NS))
    assert r.state.dead and r.state.exit_reason == 'series_barrier'
    assert r.state.cash == pytest.approx(749.9)


def test_series_base_survives_loss_and_ends_only_on_flat_recovery():
    r = replay(tape([0, .2, .4, .6, .8, 1, 1.2], [2000, 2000, 1999, 1999, 1999, 2002, 2002]))
    enter(r)
    r.request_close(int(.2 * NS))
    r.advance_until(int(.4 * NS))
    assert r.state.series_active and r.state.series_base == 1500
    assert r.open_at(int(.4 * NS), 1)
    r.advance_until(int(.6 * NS))
    assert r.state.series_base == 1500 and r.state.series_id == 1
    r.request_close(int(.8 * NS))
    r.advance_until(NS)
    assert r.state.cash > 1500 and not r.state.series_active


def test_restored_series_debt_and_buffer_survive_second_entry():
    r = replay(tape([0, .2, .4, .6, .8, 1], [2000, 2000, 1999, 1999, 1999, 1998]))
    enter(r)
    r.request_close(int(.2 * NS))
    r.advance_until(int(.4 * NS))
    restored = ExecutionReplay(r.tape, r.config).restore(json.loads(json.dumps(r.snapshot())))
    assert restored.state.series_debt == pytest.approx(1.4)
    for branch in (r, restored):
        assert branch.open_at(int(.4 * NS), -1)
        branch.advance_until(int(.6 * NS))
    assert r.snapshot() == restored.snapshot()
    assert r.state.series_base == 1500
    assert r.state.series_id == 1
    assert r.state.series_debt == pytest.approx(1.8)
    assert r.state.series_risk_buffer == pytest.approx(748.2)


@pytest.mark.parametrize('side', [1, -1])
def test_mechanical_stop_target_and_time_immediate(side):
    for last, reason in [(2002 if side > 0 else 1998, 'take'), (1998 if side > 0 else 2002, 'stop')]:
        r = ExecutionReplay(tape([0, .2, .3], [2000, 2000, last]))
        enter(r, side)
        r.request_close(int(.2 * NS))
        r.advance_until(int(.3 * NS))
        assert r.state.exit_reason == reason
        assert r.state.exit_ns == int(.3 * NS)
    r = ExecutionReplay(tape([0, .2, 120.2], end=130))
    enter(r, side)
    r.reference_exit(121 * NS)
    assert r.state.exit_reason == 'time'


def test_entry_cutoff_does_not_block_exposed_discretionary_management():
    r = enter(replay(tape([0, .2, .4, .6, .8], cutoff=.3)))
    assert r.request_close(int(.4 * NS))
    r.advance_until(int(.6 * NS))
    assert r.state.exit_reason == 'policy_close'
    assert not r.open_at(int(.6 * NS), 1)


def test_mechanical_deadline_and_flat_cash_carried_common_horizon():
    r = enter(replay(tape([0, .2, .4, .6, .8, 1], deadline=.4)))
    r.reference_exit(NS)
    assert r.state.exit_reason == 'session_close'
    assert r.state.exit_ns == int(.4 * NS)
    assert r.state.clock_ns == NS
    assert r.state.cash == r.state.equity == pytest.approx(1499.6)
    assert not r.state.censored


def test_invalid_entry_support_and_epoch_cancels_pending():
    r = replay(tape([0, .2, .4], epoch=[1, 2, 2]))
    assert r.open_at(0, 1)
    r.advance_until(int(.4 * NS))
    assert r.state.side == 0 and not r.state.censored
    assert r.state.cash == 1500


def test_contract_and_reversal_rejected():
    r = enter(replay(tape([0, .2, .4])))
    with pytest.raises(ValueError, match='backwards'):
        r.advance_until(0)
    assert execution_contract()['version'] == r.snapshot()['version']


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('plan_id', [0, 1, 2])
def test_mechanical_reference_matches_legacy_numba_kernel(side, plan_id):
    """Independent frozen v3 engine comparison: prices, exits, fees and equity."""
    from spectrum.player_v3.gym import advance_survival
    from spectrum.player.contract import PLANS
    rng = np.random.default_rng(61019)
    seconds = np.arange(0, 1001) / 1.
    prices = 2000 + np.cumsum(rng.choice([-.1, 0., .1], len(seconds)))
    t = tape(seconds, prices, end=1100)
    r = ExecutionReplay(t)
    assert r.open_at(0, side, plan_id)
    r.reference_exit(1000 * NS)
    s = np.zeros(21, np.int64)
    s[5], s[6], s[7], s[8], s[10], s[13] = side, plan_id, 100_000_000, 2 * NS, 1, 1
    f = np.array([1500., 1500., 1500., 0., 0., 1., 0., 1500., 0.])
    row = advance_survival(1, len(t.quote_t), t.quote_t, t.quote_t, t.bid, t.ask,
                          t.valid, t.epoch, s, f, t.deadline_ns, t.entry_cutoff_ns,
                          -1, .3, np.array(PLANS, np.int64), np.empty(0),
                          np.empty(0), np.empty(0, np.int64))
    assert not r.state.censored
    assert r.state.entry_price_nano == row[5]
    assert r.state.exit_price_nano == row[6]
    assert r.state.exit_ns == row[11]
    assert r.state.closed_reason == row[8]
    assert r.state.last_trade_net == pytest.approx(row[7])
    assert r.state.cash == pytest.approx(f[0])
    assert r.state.max_drawdown == pytest.approx(f[3])
