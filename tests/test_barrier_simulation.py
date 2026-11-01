import numpy as np
import pytest
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from spectrum.execution.replay import QuoteTape
from spectrum.execution.barriers import simulate_tape
from spectrum.execution.barriers import scalar_reference
from spectrum.execution.barriers import SimulationConfig
from spectrum.execution.barriers import NS


def tape(seconds, bids, asks=None, valid=None, epoch=None, deadline=1000):
    t = np.rint(np.array(seconds)*NS).astype(np.int64)
    bid = np.rint(np.array(bids)*1e9).astype(np.int64)
    ask = np.rint((np.array(bids)+.2 if asks is None else np.array(asks))*1e9).astype(np.int64)
    return QuoteTape(t, bid, ask, np.ones(len(t), bool) if valid is None else np.array(valid, bool),
        np.zeros(len(t), np.int64) if epoch is None else np.array(epoch, np.int64),
        int(t[-1]+10*NS), 0, int(deadline*NS), int(deadline*NS))


def cfg(**kwargs):
    return dict(deadline_seconds=3, **kwargs)


def compare(q, origins, risks, config):
    result = simulate_tape(q, np.rint(np.array(origins)*NS).astype(np.int64), risks, config)
    for row in result.itertuples():
        expected = scalar_reference(q, row.t_ns, row.risk_R, row.side, config)
        for key, val in expected.items():
            if isinstance(val, float):
                assert np.isclose(getattr(row, key), val, atol=1e-12, equal_nan=True)
            else:
                assert getattr(row, key) == val, (key, row, expected)
    return result


def test_strict_entry_and_exit_latency_exact_ordinals_and_actual_exit_extrema():
    q = tape([0, .1, .100000001, .2, .3, .300000001, 1, 2, 3, 4, 5],
             [100, 80, 100, 103, 95, 102, 100, 100, 100, 100, 100])
    f = compare(q, [0], [1.], cfg())
    long = f.loc[f.side == 1].iloc[0]
    assert long.entry_ordinal == 2
    assert long.trigger_ordinal == 3 and long.reason == 'TAKE'
    assert long.exit_ordinal == 5  # exactly .3 is not after .2 + .1
    assert np.isclose(long.net, 1.5)
    assert np.isclose(long.mae, -5.5)  # excursion after trigger during latency
    assert np.isclose(long.mfe, 2.5)


def test_short_stop_precedes_later_take_and_fee_charged_once():
    q = tape([0, .11, .2, .31, .5, .7, 1, 2, 3, 4, 5],
             [100, 100, 101, 102, 95, 94, 94, 94, 94, 94, 94])
    f = compare(q, [0], [1.], cfg())
    short = f.loc[f.side == -1].iloc[0]
    assert short.reason == 'STOP' and short.trigger_ordinal == 2
    assert short.exit_ordinal == 3 and np.isclose(short.net, -2.5)
    assert np.isclose(short.mfe, -.5)


def test_timeout_and_initial_cost_can_trigger_immediate_stop():
    q = tape([0, .11, .21, .22, 1, 2, 3, 3.11, 3.22, 4, 5], [100]*11)
    f = compare(q, [0], [10.], cfg())
    assert (f.reason == 'TIME').all()
    assert (f.trigger_ordinal == -1).all() and (f.exit_ordinal == 8).all()
    assert (f.trigger_ns == int(3.11*NS)).all()
    assert np.allclose(f[['net', 'mae', 'mfe']], -.5)
    f = compare(q, [0], [.4], cfg())
    assert (f.reason == 'STOP').all() and (f.trigger_ordinal == f.entry_ordinal).all()


@pytest.mark.parametrize('kind', ['invalid', 'epoch', 'spread'])
def test_support_loss_during_exit_censors_not_zero(kind):
    qargs = {}
    if kind == 'invalid':
        qargs['valid'] = [1, 1, 1, 0, 1, 1, 1]
    if kind == 'epoch':
        qargs['epoch'] = [0, 0, 0, 1, 1, 1, 1]
    if kind == 'spread':
        qargs['asks'] = [100.2, 100.2, 103.2, 103, 102.2, 102.2, 102.2]
    q = tape([0, .11, .2, .25, .31, 4, 5], [100, 100, 103, 103, 102, 102, 102], **qargs)
    f = compare(q, [0], [1.], cfg())
    assert (f.status == 'CENSORED').all()
    assert f.net.isna().all() and (f.exit_ns == 0).all()


def test_invalid_at_origin_and_session_guard_are_causal_exclusions():
    q = tape([0, .11, 1, 2, 3, 4, 5], [100]*7, valid=[0, 1, 1, 1, 1, 1, 1], deadline=10)
    f = compare(q, [0, 3], [1., 1.], cfg())
    assert (f.status == 'INELIGIBLE').all()
    assert list(f.reason) == ['ORIGIN', 'ORIGIN', 'SESSION', 'SESSION']


def test_entry_timeout_is_empty_but_silence_during_position_is_censored():
    q = tape([0, 3, 4, 5], [100]*4)
    f = compare(q, [0], [1.], cfg())
    assert (f.status == 'EMPTY').all()
    q = tape([0, .11, 61, 62, 63], [100]*5)
    f = compare(q, [0], [1.], cfg())
    assert (f.status == 'CENSORED').all() and (f.reason == 'EXIT_TIMEOUT').all()


def test_same_nanosecond_quote_order_and_prefix_invariance():
    q = tape([0, .11, .2, .2, .2, .31, 1, 4, 5], [100, 100, 103, 98, 100, 102, 100, 100, 100])
    f = compare(q, [0], [1.], cfg())
    assert f.loc[f.side == 1, 'trigger_ordinal'].iloc[0] == 2
    changed = tape([0, .11, .2, .2, .2, .31, 1, 4, 5], [100, 100, 103, 98, 100, 102, 500, 1, 500])
    other = simulate_tape(changed, np.array([0], np.int64), [1.], cfg())
    assert f.equals(other)


def test_cannot_exit_after_session_end_on_late_timeout_quote():
    q = tape([0, .11, 1, 2, 20, 21], [100]*6, deadline=10)
    f = compare(q, [0], [10.], cfg())
    assert (f.status == 'CENSORED').all() and (f.reason == 'EXIT_TIMEOUT').all()
    assert (f.exit_ns == 0).all()


def test_clock_timeout_is_independent_of_late_publication_and_includes_path_to_exit():
    q = tape([0, .11, 1, 2, 3.15, 3.20, 3.22, 4, 5], [100, 100, 100, 100, 103, 97, 101, 100, 100])
    f = compare(q, [0], [10.], cfg())
    assert (f.reason == 'TIME').all()
    assert (f.trigger_ns == int(3.11*NS)).all()
    assert (f.trigger_ordinal == -1).all() and (f.exit_ordinal == 6).all()
    long = f.loc[f.side == 1].iloc[0]
    assert np.isclose(long.mae, -3.5) and np.isclose(long.mfe, 2.5)
    assert np.isclose(long.net, .5)


def test_timer_wins_price_barrier_at_exact_deadline():
    q = tape([0, .11, 1, 2, 3.11, 3.22, 4, 5], [100, 100, 100, 100, 104, 104, 104, 104])
    f = compare(q, [0], [1.], cfg())
    assert (f.reason == 'TIME').all() and (f.trigger_ordinal == -1).all()


def test_configuration_rejects_impossible_timing():
    with pytest.raises(ValueError, match='timeout'):
        SimulationConfig.from_dict(dict(latency_ms=2000, pending_timeout_seconds=2))
    with pytest.raises(ValueError, match='nonnegative'):
        SimulationConfig(latency_ms=-10)


def test_support_loss_before_entry_expiry_is_censor_not_empty():
    q = tape([0, .05, 3, 4], [100]*4, valid=[1, 0, 1, 1], epoch=[0, -1, 1, 1])
    f = compare(q, [0], [1.], cfg())
    assert (f.status == 'CENSORED').all() and (f.reason == 'SUPPORT').all()


def test_support_loss_before_exit_expiry_is_not_hidden_by_timeout():
    q = tape([0, .11, .2, .25, 4, 5], [100, 100, 103, 103, 103, 103], valid=[1, 1, 1, 0, 1, 1])
    f = compare(q, [0], [1.], cfg())
    assert (f.status == 'CENSORED').all() and (f.reason == 'SUPPORT').all()


def test_independent_reference_on_irregular_random_path_and_support_boundaries():
    rng = np.random.default_rng(417)
    seconds = np.cumsum(rng.uniform(.01, .7, 900))
    seconds[450:] += 61
    prices = 100+np.cumsum(rng.normal(0, .3, 900))
    valid = np.ones(900, bool)
    valid[[100, 101, 320, 700]] = False
    epoch = np.cumsum(np.r_[False, valid[1:] & ~valid[:-1]])
    q = tape(seconds, prices, asks=prices+rng.uniform(.1, .6, 900), valid=valid, epoch=epoch)
    origins = seconds[np.linspace(0, 870, 40, dtype=int)]
    risks = rng.uniform(.4, 3., len(origins))
    compare(q, origins, risks, cfg(latency_ms=170, fee_roundtrip=.27))


def exact_barrier_nano(risk, fee, multiple, reason):
    """Independent Decimal oracle, not the runtime Fraction implementation."""
    with localcontext() as context:
        context.prec = 80
        amount = Decimal(str(float(risk)))*Decimal(str(multiple))
        level = Decimal(str(fee))+(amount if reason == 'TAKE' else -amount)
        return int((level*Decimal(10**9)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def boundary_tape(side, gross_nano, *, invalid_during_exit=False):
    """Probe one exact native price, then revert before the five-second timer."""
    times = np.array([0, 200_000_000, NS, 1_200_000_000, 2*NS,
                      5_200_000_000, 5_400_000_000], np.int64)
    entry, spread = 4_000_000_000_000, 200_000_000
    bid = np.full(len(times), entry-spread if side == 1 else entry, np.int64)
    ask = bid+spread
    if side == 1:
        bid[2:4] = entry+gross_nano
        ask[2:4] = bid[2:4]+spread
    else:
        ask[2:4] = entry-gross_nano
        bid[2:4] = ask[2:4]-spread
    valid = np.ones(len(times), bool)
    if invalid_during_exit:
        valid[3] = False
    return QuoteTape(times, bid, ask, valid, np.zeros(len(times), np.int64),
                     10*NS, 0, 20*NS, 20*NS)


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('reason', ['TAKE', 'STOP'])
@pytest.mark.parametrize('offset', [-100_000_000, -1, 0, 1, 100_000_000])
@pytest.mark.parametrize('risk,multiple', [
    (3.7, 1.), (12*.1, 1.), (3., 3*.1), (.7000000005, 1.)])
def test_exact_and_adjacent_native_boundaries_match_independent_decimal_oracle(side, reason, offset, risk, multiple):
    config = dict(deadline_seconds=5., gain_multiple=multiple if reason=='TAKE' else 2.,
                  stop_multiple=multiple if reason=='STOP' else 1., fee_roundtrip=.3)
    boundary = exact_barrier_nano(risk, .3, multiple, reason)
    q = boundary_tape(side, boundary+offset)
    result = compare(q, [0], [risk], config)
    row = result.loc[result.side == side].iloc[0]
    hit = offset >= 0 if reason == 'TAKE' else offset <= 0
    assert row.status == 'COMPLETE'
    assert row.reason == (reason if hit else 'TIME')
    assert row.trigger_ns == (NS if hit else 5_200_000_000)
    assert row.trigger_ordinal == (2 if hit else -1)
    assert row.exit_ordinal == (3 if hit else 6)
    expected_net = (boundary+offset)/NS-.3 if hit else -.5
    assert row.net == pytest.approx(expected_net, abs=1e-12)


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('reason', ['TAKE', 'STOP'])
def test_exact_barrier_trigger_does_not_hide_support_loss_during_exit(side, reason):
    risk = 12*.1
    config = dict(deadline_seconds=5., gain_multiple=1., fee_roundtrip=.3)
    q = boundary_tape(side, exact_barrier_nano(risk, .3, 1., reason), invalid_during_exit=True)
    row = compare(q, [0], [risk], config).loc[lambda x: x.side==side].iloc[0]
    assert row.status == 'CENSORED' and row.reason == 'SUPPORT'
    assert row.trigger_ns == NS and row.trigger_ordinal == 2
    assert row.exit_ordinal == -1 and np.isnan(row.net)


@pytest.mark.parametrize('risk', [.7000000005, .7000000015])
def test_half_nanodollar_boundary_uses_ties_to_even(risk):
    take = exact_barrier_nano(risk, .3, 1., 'TAKE')
    assert take == (1_000_000_000 if risk == .7000000005 else 1_000_000_002)
    q = boundary_tape(1, take)
    row = compare(q, [0], [risk], dict(deadline_seconds=5., gain_multiple=1.)).iloc[0]
    assert row.reason == 'TAKE' and row.trigger_ns == NS


@pytest.mark.parametrize('risk,fee', [(1e100, .3), (1.2, 1e100)])
def test_large_finite_barrier_preserves_predicate_without_int64_overflow(risk, fee):
    q = boundary_tape(1, 0)
    f = compare(q, [0], [risk], dict(deadline_seconds=5., fee_roundtrip=fee))
    assert f.reason.eq('TIME' if risk > fee else 'STOP').all()


def test_noninteger_native_prices_are_rejected_instead_of_losing_precision():
    q = boundary_tape(1, 0)
    broken = QuoteTape(q.quote_t, q.bid.astype(float), q.ask.astype(float), q.valid, q.epoch,
                       q.coverage_end_ns, q.entry_start_ns, q.entry_cutoff_ns, q.deadline_ns)
    with pytest.raises(ValueError, match='integer nanodollars'):
        simulate_tape(broken, np.array([0], np.int64), [1.2], dict(deadline_seconds=5.))
    with pytest.raises(ValueError, match='integer nanodollars'):
        scalar_reference(broken, 0, 1.2, 1, dict(deadline_seconds=5.))


def test_portable_fixed_barriers_match_mechanical_engine_at_exact_and_adjacent_prices():
    # Import is restricted to this test: library runtime never imports research.
    from experiments.mechanical_lab.execution import _kernel
    for side in (1, -1):
        for reason in ('TAKE', 'STOP'):
            for risk in (3.7, 12*.1):
                for offset in (-100_000_000, -1, 0, 1, 100_000_000):
                    q = boundary_tape(side, exact_barrier_nano(risk, .3, 1., reason)+offset)
                    ours = simulate_tape(q, np.array([0], np.int64), [risk],
                                         dict(deadline_seconds=5., gain_multiple=1.))
                    ints, values = _kernel(q.quote_t, q.bid, q.ask, q.valid, q.epoch,
                        q.coverage_end_ns, q.entry_start_ns, q.deadline_ns,
                        np.array([0], np.int64), np.array([risk]), .3, 0.,
                        100_000_000, 2*NS, 60*NS, 1., 5*NS, 0, 1., 0., 1.5, 1., .5)
                    for i, row in enumerate(ours.itertuples()):
                        assert row.status == ['INELIGIBLE', 'EMPTY', 'CENSORED', 'COMPLETE'][ints[i, 0]]
                        assert row.reason == ['ORIGIN', 'SESSION', 'RISK', 'ENTRY_TIMEOUT', 'SUPPORT',
                                              'TAKE', 'STOP', 'TIME', 'EXIT_TIMEOUT', 'COVERAGE'][ints[i, 1]]
                        assert [row.decision_ordinal, row.entry_ordinal, row.trigger_ordinal,
                                row.exit_ordinal, row.trigger_ns] == list(ints[i, 2:7])
                        assert [row.net, row.mae, row.mfe] == pytest.approx(values[i, :3], abs=1e-12)
