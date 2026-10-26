from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from spectrum.research_v6.targets import scan_paths, METRICS, haar_details, cyclic_clock, build_path_targets, learning_view


def tape(t, mid):
    return SimpleNamespace(quote_t=np.asarray(t, np.int64)*10**9,
        bid=np.rint((np.asarray(mid)-.1)*1e9).astype(np.int64),
        ask=np.rint((np.asarray(mid)+.1)*1e9).astype(np.int64))


def test_irregular_quote_integrals_costs_and_risk_order():
    q = tape([0, 1, 3, 6], [100, 102, 99, 103])
    v, s = scan_paths(q.quote_t, q.bid, q.ask, np.array([0]), np.array([3]), np.array([1.]), .3)
    m = dict(zip(METRICS, v[0, 0]))
    assert np.isclose(m['net'], 2.5)
    assert np.isclose(m['area_positive'], 3.)
    assert np.isclose(m['area_negative'], 5.)
    assert m['underwater_seconds'] == 4
    assert np.isclose(m['max_drawdown'], 3.)
    assert m['first_risk_seconds'] == 3
    assert m['risk_before_cross'] == 1
    assert np.isclose(v[0, 0, 0]+v[0, 1, 0], -1.)
    assert np.isclose(s[0, 0, 0], -.5)


def test_same_timestamp_keeps_ordinal_risk_but_no_area():
    q = tape([10, 10, 10], [100, 98, 103])
    v, _ = scan_paths(q.quote_t, q.bid, q.ask, np.array([0]), np.array([2]), np.array([1.]), .3)
    m = dict(zip(METRICS, v[0, 0]))
    assert m['risk_before_cross'] == 1
    assert m['area_negative'] == 0
    assert m['first_risk_seconds'] == 0
    assert np.isclose(m['mae'], -2.5)


def test_endpoint_does_not_determine_shape_and_haar_sign_matters():
    a = np.linspace(-.5, 2.5, 17)[None, :]
    b = a.copy(); b[0, 4:12] -= 3
    assert a[0, -1] == b[0, -1]
    assert not np.allclose(haar_details(a), haar_details(b))
    assert np.allclose(haar_details(-a), -haar_details(a))
    assert np.allclose(haar_details(a)**2, haar_details(-a)**2)
    assert np.allclose((haar_details(a)**2).sum(), ((a[:, 1:]-a[:, 1:].mean())**2).sum())


def test_censor_is_not_none_and_after_horizon_is_not_profile():
    q = tape([0, 1, 2, 4], [100, 101, 102, 103])
    m = pd.DataFrame(dict(ready=[True]*4, entry_ordinal=[0]*4, entry_ns=[0]*4,
        cross_ordinal=[1, 3, -1, -1], cross_ns=np.array([1, 4, 0, 0])*10**9,
        R_event=[1.]*4, support_end_ns=np.array([5, 5, 1, 3])*10**9,
        event_id=['a', 'b', '', '']))
    target = build_path_targets(q, m, horizon_seconds=2)
    assert target.hit.tolist() == [1, 0, -1, 0]
    assert target.profile_known.tolist() == [True, False, False, False]
    assert target.v6_event_id.tolist() == ['a', '', '', '']
    assert np.isnan(target.net_L.iloc[1:]).all()
    view = learning_view(target)
    assert not view['profile_mask'][1:].any()
    assert view['hit_mask'].tolist() == [True, True, False, True]


def test_cyclic_clock_and_future_price_translation_invariance():
    phase = cyclic_clock(np.array([0, 60, 120])*10**9, 60)
    assert np.allclose(phase, [[0, 1]]*3)
    a = tape([0, 1, 2], [100, 101, 102])
    b = tape([0, 1, 2], [10000, 10001, 10002])
    args = (np.array([0]), np.array([2]), np.array([1.]), .3)
    va, sa = scan_paths(a.quote_t, a.bid, a.ask, *args)
    vb, sb = scan_paths(b.quote_t, b.bid, b.ask, *args)
    assert np.allclose(va, vb, equal_nan=True)
    assert np.allclose(sa, sb)


def map_for(q, crosses):
    return pd.DataFrame(dict(t_ns=[q.quote_t[0]]*len(crosses), ready=[True]*len(crosses),
        entry_ordinal=[0]*len(crosses), entry_ns=[q.quote_t[0]]*len(crosses),
        cross_ordinal=crosses, cross_ns=q.quote_t[crosses], R_event=[1.]*len(crosses),
        support_end_ns=[q.quote_t[-1]+10**9]*len(crosses),event_id=[str(x) for x in crosses]))


def test_zero_duration_masks_rates_but_preserves_ordered_hit_and_risk():
    q = tape([10, 10, 10], [100, 98, 103])
    m = build_path_targets(q, map_for(q, [2, 1, 2]))
    assert m.profile_known.all() and not m.velocity_known.any()
    assert np.isnan(m.velocity_R_per_T_L).all()
    assert np.isnan(m.haar_00_L).all()
    assert m.packet_rank.tolist() == [2, 1, 2]
    assert m.risk_before_cross_L.tolist() == [1, 0, 1]
    assert m.target_available_ns.tolist() == [10*10**9]*3


@pytest.mark.parametrize('kind', ['invalid', 'epoch', 'gap'])
def test_supported_endpoints_do_not_hide_internal_breaks(kind):
    q = tape([0, 1, 2, 3], [100, 101, 102, 103])
    q.valid = np.ones(4, bool)
    q.epoch = np.zeros(4, np.int64)
    if kind == 'invalid': q.valid[1] = False
    if kind == 'epoch': q.epoch[1] = 1
    if kind == 'gap': q.quote_t[2:] += 61*10**9
    with pytest.raises(ValueError, match='unsupported quote path'):
        build_path_targets(q, map_for(q, [3]))


def test_future_suffix_cannot_change_an_already_completed_crossing_profile():
    q = tape([0, 1, 2, 3], [100, 101, 102, 103])
    m = map_for(q, [2])
    a = build_path_targets(q, m)
    q.bid[3] += 500*10**9
    q.ask[3] += 500*10**9
    b = build_path_targets(q, m)
    pd.testing.assert_frame_equal(a, b)
