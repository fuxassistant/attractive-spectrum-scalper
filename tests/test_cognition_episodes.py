import numpy as np
import pandas as pd
import pytest
from spectrum.cognition import episodes as e


def source(n=240):
    t = np.arange(n, dtype=np.int64) * e.NS
    d = dict(t_ns=t, sigma600=np.ones(n), sigma60_ratio=np.ones(n), sigma3600_ratio=np.ones(n),
             minutes_since_london=np.arange(n) / 60, spread_ticks=np.ones(n),
             flow_volume_rate_ratio=np.full(n, 2.), flow_volume_rate_known=np.ones(n))
    for w in (5, 15, 60):
        d[f"ret{w}"] = np.zeros(n); d[f"ret{w}_known"] = np.ones(n)
    for w in (5, 30):
        for which in ("low", "high"):
            d[f"flow_pressure{w}_{which}"] = np.full(n, .4)
        d[f"flow_pressure{w}_known"] = np.ones(n)
        d[f"flow_book{w}_pressure"] = np.full(n, -.2); d[f"flow_book{w}_known"] = np.ones(n)
    d["flow_unknown30_fraction"] = np.zeros(n)
    for lvl in (1, 4):
        d[f"flow_depth{lvl}_imbalance"] = np.full(n, -.1); d[f"flow_depth{lvl}_known"] = np.ones(n)
    return pd.DataFrame(d)


def test_prefix_mutation_cannot_change_features_or_episode_identity():
    d = source(); seg = np.zeros(len(d)); a, am = e.features(d, seg)
    altered = d.copy(); altered.loc[101:, altered.columns != "t_ns"] = 900
    b, bm = e.features(altered, seg)
    pd.testing.assert_frame_equal(a.iloc[:101], b.iloc[:101])
    for k in am:
        np.testing.assert_equal(am[k][:101], bm[k][:101])


def test_epoch_and_time_gap_reset_state_and_unknown_pressure_cannot_start():
    d = source(); seg = np.zeros(len(d)); seg[80:] = 1
    x, m = e.features(d, seg)
    assert x.episode_active.iloc[79] == 1 and x.episode_active.iloc[80] == 0
    assert x.episode_active.iloc[94] == 0 and x.episode_active.iloc[95] == 1
    assert np.isnan(x.reaction15.iloc[95])
    g = d.drop(index=[80]).reset_index(drop=True); xx, _ = e.features(g, np.zeros(len(g)))
    assert xx.episode_active.iloc[80] == 0
    d.flow_pressure30_low = -.5; d.flow_pressure30_high = .5
    _, m = e.features(d, np.zeros(len(d)))
    assert not m["episode_start"].any()


def test_ret30_uses_each_return_own_sigma_and_no_future():
    d = source(); d.ret15 = 1.; d.loc[20:, "sigma600"] = 2.
    _, m = e.features(d, np.zeros(len(d)))
    assert m["ret30"][25] == pytest.approx(3 / (2 * np.sqrt(2)))


def test_executable_categories_preserve_time_empty_and_unknown():
    st = [1, 1, 1, 3, 2, 5, 1, 1]
    net = np.asarray([2.1, -1.5, .2, 0., np.nan, np.nan, 2.099, -1.499], np.float32)
    assert e.categories(st, net).tolist() == [0, 1, 2, 3, -1, -1, 2, 2]
    with pytest.raises(ValueError):
        e.categories([1], [np.nan])


def test_episode_weight_normalizes_each_side_and_day_separately():
    f = pd.DataFrame(dict(date=["a"] * 4 + ["b"], side=[1, 1, -1, 1, 1], episode_id=[1, 1, 1, 0, 1], t_ns=range(5)))
    np.testing.assert_equal(e.episode_weights(f), [.5, .5, 1, 1, 1])


def test_account_cannot_overlap_or_skip_unknown_to_get_future_profit():
    f = pd.DataFrame(dict(date=["d"] * 5, t_ns=np.asarray([0, 5, 20, 30, 40]) * e.NS, side=[1] * 5,
                          expected_net=[1.] * 5, status=[1, 1, 3, 2, 1], net=[1., 50., 0., np.nan, 100.],
                          busy_s=[10., 2., 5., np.nan, 5.]))
    tr, censor = e.sequential(f)
    assert tr.index.tolist() == [0, 2] and tr.net.sum() == 1
    assert censor["t_ns"] == str(30 * e.NS)


def test_bootstrap_keeps_zero_days_and_observed_negative_sign():
    lo, hi = e.block_interval(np.r_[np.zeros(10), np.full(10, -2.)])
    assert -2 <= lo <= hi <= 0
