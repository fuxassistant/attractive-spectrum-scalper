import numpy as np
import pytest

from spectrum.cognition import opportunity as op
from spectrum.cognition.execution import ExecutionReplay, ExecutionConfig, QuoteTape
from spectrum.cognition.signatures import MENU_PLANS, branch, event_context

NS = 10**9
TICK = op.TICK_NANO


def quotes(seconds, mids_ticks, spread=1, valid=None, epoch=None):
    t = np.rint(np.asarray(seconds, float) * NS).astype(np.int64)
    mid = np.asarray(mids_ticks, float) * TICK
    bid = np.rint(mid - spread * TICK / 2).astype(np.int64)
    ask = np.rint(mid + spread * TICK / 2).astype(np.int64)
    n = len(t)
    return (t, bid, ask, np.asarray(valid if valid is not None else [True] * n),
            np.asarray(epoch if epoch is not None else [1] * n, np.int32))


def zigzag(n, offset=.5):
    """One quote per second just after each boundary; mid alternates +-1 tick."""
    return quotes(np.arange(n) + offset, 20000 + (np.arange(n) % 2))


def test_grid_is_as_of_boundary_never_later_quote():
    g = op.second_grid(*quotes([.5, 1.5, 2.5], [10, 20, 30]), 0, 4 * NS)
    assert np.isnan(g['mid'][0])
    assert g['mid'][1] == pytest.approx(10) and g['mid'][2] == pytest.approx(20)
    assert op.origin_index(g, [int(1.999 * NS)])[0] == 1


def test_causal_fields_invariant_to_future_mutation():
    q = zigzag(3000)
    t0 = int(1500.3 * NS)
    a, _ = op.measure(op.second_grid(*q, 0, 3000 * NS), [t0])
    mutated = list(q); mutated[1] = q[1].copy(); mutated[2] = q[2].copy()
    later = q[0] > t0
    mutated[1][later] += 50 * TICK; mutated[2][later] += 50 * TICK
    b, _ = op.measure(op.second_grid(*mutated, 0, 3000 * NS), [t0])
    for name in op.CAUSAL_NAMES:
        assert np.array_equal(a[name], b[name]), name


def test_premium_matches_declared_normal_formula():
    x, y = op.measure(op.second_grid(*zigzag(3000), 0, 3000 * NS), [int(1000.2 * NS)])
    assert x['sigma600_known'][0] and x['sigma600_ticks_per_sqrt_s'][0] == pytest.approx(1.)
    assert x['sigma3600_known'][0] == False  # not enough history: masked, not zero evidence
    assert y['premium_ticks_30'][0] == pytest.approx(np.sqrt(30) * np.sqrt(2 / np.pi))
    assert y['rv_ticks2_30'][0] == pytest.approx(30.)
    assert y['gamma_ticks2_30'][0] == pytest.approx(0.)
    assert y['abs_move_ticks_30'][0] == pytest.approx(0.)  # even number of alternating steps
    assert y['up_ticks_30'][0] == pytest.approx(1.) or y['down_ticks_30'][0] == pytest.approx(1.)


@pytest.mark.parametrize('defect', ['invalid', 'epoch'])
def test_unsupported_step_makes_horizon_immature(defect):
    t, bid, ask, valid, epoch = zigzag(3000)
    i = 1020
    if defect == 'invalid':
        valid = valid.copy(); valid[i] = False
    else:
        epoch = epoch.copy(); epoch[i:] = 2
    _, y = op.measure(op.second_grid(t, bid, ask, valid, epoch, 0, 3000 * NS), [int(1000.2 * NS)])
    assert not y['mature_30'][0] and np.isnan(y['abs_move_ticks_30'][0])
    assert y['mature_30'][0] == y['mature_120'][0] == False


def test_quote_silence_is_supported_and_horizon_past_session_immature():
    t, bid, ask, valid, epoch = zigzag(3000)
    keep = (t < 1005 * NS) | (t > 1015 * NS)
    _, y = op.measure(op.second_grid(t[keep], bid[keep], ask[keep], valid[keep], epoch[keep], 0, 3000 * NS),
                      [int(1000.2 * NS), int(2900.2 * NS)])
    assert y['mature_30'][0]
    assert not y['mature_120'][1]


def test_origin_outside_grid_rejected():
    g = op.second_grid(*zigzag(10), 0, 10 * NS)
    with pytest.raises(ValueError):
        op.measure(g, [10 * NS])


def trend_tape(deadline=10_000):
    seconds = np.arange(0, 400, .1)
    t, bid, ask, valid, epoch = quotes(seconds, 20000 + seconds)  # +1 tick per second
    return QuoteTape(t, bid, ask, valid, epoch, int(401 * NS), 0, int(deadline * NS), int(deadline * NS))


def test_sibling_branches_share_origin_and_do_not_mutate_base():
    tape = trend_tape(); config = ExecutionConfig(plans=MENU_PLANS)
    base = ExecutionReplay(tape, config); base.advance_until(10 * NS)
    before = base.snapshot()
    mid = (tape.bid[100] + tape.ask[100]) / (2 * TICK)
    long = branch(base, tape, 1, 0, 10 * NS, mid)
    short = branch(base, tape, -1, 0, 10 * NS, mid)
    assert base.snapshot() == before
    assert long['status'] == short['status'] == 'COMPLETE'
    assert long['entry_ns'] == short['entry_ns']
    assert long['exit_reason'] == 'take' and short['exit_reason'] == 'stop'
    assert long['net_usd'] == pytest.approx(1.2 - .3, abs=.11)
    assert short['net_usd'] == pytest.approx(-.8 - .3, abs=.11)
    assert long['fees_usd'] == pytest.approx(.3)  # fee charged exactly once per branch
    assert long['gross_usd'] - long['net_usd'] == pytest.approx(.3)
    assert long['entry_cost_ticks'] > 0 and short['entry_cost_ticks'] > 0


def test_plan_beyond_deadline_is_explicitly_excluded():
    tape = trend_tape(deadline=300); base = ExecutionReplay(tape, ExecutionConfig(plans=MENU_PLANS))
    base.advance_until(10 * NS)
    assert branch(base, tape, 1, 3, 10 * NS, 20010.)['status'] == 'EXCLUDED_SESSION_TIME'
    assert branch(base, tape, 1, 0, 10 * NS, 20010.)['status'] == 'COMPLETE'


def test_event_context_counts_only_strictly_earlier_events():
    times = np.array([100, 200, 400], np.int64) * NS
    c = event_context(np.array([400, 401], np.int64) * NS, times)
    assert c['prior_events_300s'].tolist() == [2., 2.]  # 401: event at 100 left the window
    assert c['since_prior_event_s'].tolist() == [200., 1.]
    c = event_context(np.array([50 * NS], np.int64), times)
    assert not c['since_prior_event_known'][0] and c['since_prior_event_s'][0] == 0.


def test_breakeven_side_accuracy_and_unreachable_case():
    from spectrum.cognition.signature_report import breakeven
    assert breakeven(1., -1.) == pytest.approx(.5)
    assert breakeven(1., -3.) == pytest.approx(.75)
    assert breakeven(-.2, -1.2) is None  # even a perfect side call loses after costs


def test_uncovered_event_bank_is_masked_not_a_known_zero():
    c = event_context(np.array([400 * NS], np.int64), np.empty(0, np.int64), bank_known=False)
    assert not c['event_bank_known'][0] and not c['since_prior_event_known'][0]
    c = event_context(np.array([400 * NS], np.int64), np.empty(0, np.int64), bank_known=True)
    assert c['event_bank_known'][0] and c['prior_events_300s'][0] == 0.  # covered quiet day: a real zero
