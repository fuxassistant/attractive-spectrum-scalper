import numpy as np
import pytest
from spectrum.cognition.execution import QuoteTape
from spectrum.research.simulation import simulate_tape, scalar_reference, SimulationConfig, NS


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
