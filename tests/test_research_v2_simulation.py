import numpy as np
import pandas as pd
import pytest
from spectrum.cognition.execution import QuoteTape
from spectrum.research_v2.simulation import simulate_tape, scalar_reference, align_entries, causal_first_qualifying, NS


def tape(seconds, bids, asks=None, valid=None, epoch=None, deadline=1000):
    t = np.rint(np.array(seconds)*NS).astype(np.int64)
    bid = np.rint(np.array(bids)*1e9).astype(np.int64)
    ask = np.rint((np.array(bids)+.2 if asks is None else np.array(asks))*1e9).astype(np.int64)
    return QuoteTape(t, bid, ask, np.ones(len(t), bool) if valid is None else np.array(valid, bool),
        np.zeros(len(t), np.int64) if epoch is None else np.array(epoch, np.int64),
        int(t[-1]+10*NS), 0, int(deadline*NS), int(deadline*NS))


def config():
    return {'event_query': {'deadline_seconds': 3}, 'execution': {'latency_ms': 100, 'fee_roundtrip': .3}}


def compare(q, origins, risk, take, cfg=None):
    cfg = cfg or config()
    t = np.rint(np.array(origins)*NS).astype(np.int64)
    f = simulate_tape(q, t, risk, take, cfg)
    for i, row in enumerate(f.itertuples()):
        reference = scalar_reference(q, int(row.t_ns), row.risk_R, row.side, int(take[i//2][i % 2]), cfg)
        for key, value in reference.items():
            actual = getattr(row, key)
            if isinstance(value, float):
                assert np.isclose(actual, value, equal_nan=True, atol=1e-12)
            else:
                assert actual == value, (key, actual, value)
    return f


def test_price_gain_does_not_exit_before_qualifying_crossing_and_exit_mae_is_complete():
    q = tape([0, .11, .2, .25, .3, .35, .41, 1, 4, 5], [100, 100, 104, 103, 105, 95, 102, 100, 100, 100])
    f = compare(q, [0], [10.], [[4, -1]])
    long = f.iloc[0]
    assert long.reason == 'TAKE' and long.trigger_ordinal == 4
    assert long.exit_ordinal == 6
    assert np.isclose(long.net, 1.5) and np.isclose(long.mae, -5.5)


def test_stop_can_precede_crossing_and_larger_stop_budget_keeps_same_crossing():
    q = tape([0, .11, .2, .31, .5, .61, 1, 4, 5], [100, 100, 99, 99, 103, 103, 100, 100, 100])
    small = compare(q, [0], [1.], [[4, -1]])
    large = compare(q, [0], [3.], [[4, -1]])
    assert small.iloc[0].reason == 'STOP' and small.iloc[0].trigger_ordinal == 2
    assert large.iloc[0].reason == 'TAKE' and large.iloc[0].trigger_ordinal == 4
    assert np.isclose(large.iloc[0].net, 2.5)


def test_real_timer_wins_same_time_crossing_and_fill_uses_timer_latency():
    q = tape([0, .11, 1, 2, 3.11, 3.20, 3.22, 4, 5], [100, 100, 100, 100, 104, 97, 101, 100, 100])
    f = compare(q, [0], [10.], [[4, -1]])
    assert (f.reason == 'TIME').all()
    assert (f.trigger_ns == 3_110_000_000).all() and (f.trigger_ordinal == -1).all()
    assert (f.exit_ordinal == 6).all()
    assert np.isclose(f.iloc[0].mae, -3.5)


def test_online_seen_line_rule_consumes_first_unprofitable_crossing():
    q = tape([0, .11, .2, .3, .4, .5, .7, 4, 5], [100, 100, 99, 104, 103, 103, 100, 100, 100])
    bank = pd.DataFrame(dict(ordinal=[1, 2, 3, 4], t_ns=q.quote_t[[1, 2, 3, 4]],
        scale=[1, 1, 1, 1], role=[1, 0, 0, 2], support=[2, 2, 2, 2]))
    # Entry event is excluded. The later profitable second lower crossing is
    # already consumed. Only the first upper crossing can qualify LONG.
    result = causal_first_qualifying(q, 1, bank, 2., .3, {1: 2}, 3)
    assert result.tolist() == [4, -1]
    assert causal_first_qualifying(q, 1, bank, 2., .3, {1: 3}, 3).tolist() == [-1, -1]


def test_future_change_after_stop_cannot_change_finished_result():
    first = tape([0, .11, .2, .31, .5, .61, 4, 5], [100, 100, 99, 99, 103, 103, 100, 100])
    second = tape([0, .11, .2, .31, .5, .61, 4, 5], [100, 100, 99, 99, 1000, 1, 1000, 1])
    a = compare(first, [0], [1.], [[4, -1]])
    b = compare(second, [0], [1.], [[5, -1]])
    pd.testing.assert_series_equal(a.iloc[0], b.iloc[0])


@pytest.mark.parametrize('kind', ['invalid', 'epoch'])
def test_pending_support_loss_is_censor_not_empty(kind):
    kwargs = {'valid': [1, 0, 1, 1]} if kind == 'invalid' else {'epoch': [0, 1, 1, 1]}
    q = tape([0, .05, 3, 4], [100]*4, **kwargs)
    a = align_entries(q, [0], config())
    assert a.iloc[0].entry_status == 'CENSORED' and a.iloc[0].entry_reason == 'SUPPORT'
    f = compare(q, [0], [1.], [[-1, -1]])
    assert (f.status == 'CENSORED').all() and f.net.isna().all()


def test_crossing_at_entry_is_rejected_and_session_admission_is_causal():
    q = tape([0, .11, 1, 2, 3, 4, 5], [100]*7, deadline=10)
    with pytest.raises(ValueError, match='strictly after'):
        compare(q, [0], [1.], [[1, -1]])
    f = compare(q, [3], [1.], [[-1, -1]])
    assert (f.status == 'INELIGIBLE').all() and (f.reason == 'SESSION').all()


def test_exit_pending_support_loss_and_fee_are_preserved():
    q = tape([0, .11, .2, .25, 4, 5], [100, 100, 104, 104, 104, 104], valid=[1, 1, 1, 0, 1, 1])
    f = compare(q, [0], [10.], [[2, -1]])
    assert (f.status == 'CENSORED').all() and (f.reason == 'SUPPORT').all()
    assert f.net.isna().all()
    flat = tape([0, .11, .21, .22, 1, 2, 3, 3.11, 3.22, 4], [100]*10)
    f = compare(flat, [0], [.4], [[-1, -1]])
    assert (f.reason == 'STOP').all()
    assert np.allclose(f.net, -.5)  # one spread and one roundtrip fee


def test_independent_irregular_full_quote_reference_with_crossing_candidates():
    rng = np.random.default_rng(927)
    seconds = np.cumsum(rng.uniform(.01, .6, 800))
    seconds[400:] += 61
    prices = 100+np.cumsum(rng.normal(0, .3, 800))
    valid = np.ones(800, bool)
    valid[[70, 250, 251, 600]] = False
    epoch = np.cumsum(np.r_[False, valid[1:] & ~valid[:-1]])
    q = tape(seconds, prices, asks=prices+rng.uniform(.1, .6, 800), valid=valid, epoch=epoch)
    origins = seconds[np.linspace(0, 750, 40, dtype=int)]
    e = np.searchsorted(q.quote_t, np.rint(origins*NS).astype(np.int64)+100_000_000, side='right')
    take = np.minimum(e[:, None]+rng.integers(1, 10, (len(e), 2)), len(seconds)-1)
    take[rng.random(take.shape) < .4] = -1
    compare(q, origins, rng.uniform(.4, 3., len(origins)), take)
