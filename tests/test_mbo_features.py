import numpy as np
import pandas as pd
import pytest

from spectrum.core.features import build_features, NS, MINUTE


def sample(minutes=250):
    start = pd.Timestamp("2026-03-23T00:00:00Z").value
    t = start + np.arange(minutes * 30 + 1) * 2 * NS
    k = np.arange(len(t), dtype=float)
    mid = 3000 + k * .001 + np.sin(k / 31) * .2
    s = pd.DataFrame(dict(t_ns=t, epoch=0, valid=True, age_ns=0,
                          bid=mid-.05, ask=mid+.05, mid=mid, bid_size=10., ask_size=12.))
    path = np.r_[0, np.cumsum(np.abs(np.diff(mid)) / .1)]
    s["c_path_ticks"] = path
    s["c_up_ticks"] = np.r_[0, np.cumsum(np.maximum(np.diff(mid), 0.) / .1)]
    s["c_down_ticks"] = np.r_[0, np.cumsum(np.maximum(-np.diff(mid), 0.) / .1)]
    for name, rate in (("trade_buy", 3), ("trade_sell", 2), ("trade_unknown", 1),
                       ("trade_count", 2), ("add_bid", 4), ("add_ask", 5),
                       ("remove_bid", 2), ("remove_ask", 3), ("matched_fill_remove_bid", 1),
                       ("matched_fill_remove_ask", 1), ("post_trade_add_bid", 1),
                       ("post_trade_add_ask", 2), ("event_count", 10)):
        s["c_"+name] = k * rate
    for depth in (1, 4, 12):
        s[f"depth_bid_{depth}"] = 10. * depth
        s[f"depth_ask_{depth}"] = 12. * depth
    for quantity in (1, 5, 10):
        s[f"sweep_buy_{quantity}"] = quantity * .01
        s[f"sweep_sell_{quantity}"] = quantity * .02
    ix = np.arange(30, len(t), 30)
    bars = pd.DataFrame(dict(t_ns=t[ix], o=mid[ix-30], h=mid[ix]+.3,
                             l=mid[ix]-.3, c=mid[ix], volume=180., coverage=1., epoch=0, valid=True))
    return s, bars


def column(x, manifest, name):
    return x[:, manifest["feature_names"].index(name)]


def test_prefix_invariance_and_common_origin_arms():
    s, b = sample()
    origins, x, m = build_features(s, b)
    end = 200
    cut = int(b.iloc[end-1].t_ns)
    po, px, pm = build_features(s[s.t_ns <= cut], b.iloc[:end])
    np.testing.assert_array_equal(x[:end], px)
    pd.testing.assert_frame_equal(origins.iloc[:end], po)
    assert len(origins) == len(b)
    assert np.isfinite(x).all() and x.dtype == np.float32
    assert set(m["arms"]["ohlc"]) < set(m["arms"]["quotes"]) < set(m["arms"]["trade"]) < set(m["arms"]["depth"])
    assert len(m["arms"]["depth"]) == x.shape[1]


def test_missing_depth_does_not_drop_or_change_ohlc_trade():
    s, b = sample(40)
    o, x, m = build_features(s, b)
    removed = [k for k in s if k.startswith(("depth_", "sweep_"))]
    oo, xx, mm = build_features(s.drop(columns=removed), b)
    assert len(oo) == len(o)
    np.testing.assert_array_equal(x[:, m["arms"]["trade"]], xx[:, mm["arms"]["trade"]])
    assert not column(xx, mm, "depth_12_total_log_known").any()


def test_volume_support_and_full_event_path():
    s, b = sample(10)
    # Quotes at snapshots can be flat while all-event path accumulated movement.
    s[["mid", "bid", "ask"]] = [3000., 2999.95, 3000.05]
    s["c_path_ticks"] = np.arange(len(s)) * 4.
    s["c_up_ticks"] = np.arange(len(s)) * 2.
    s["c_down_ticks"] = np.arange(len(s)) * 2.
    for name in ("trade_buy", "trade_sell", "trade_unknown", "trade_count"):
        s["c_" + name] = 0.
    _, x, m = build_features(s, b)
    assert np.all(column(x, m, "w6s_path_ticks_log") > 0)
    assert not column(x, m, "w6s_path_efficiency").any()
    assert column(x, m, "w6s_path_efficiency_known").all()
    assert not column(x, m, "w6s_ticks_per_100_contracts_known").any()
    assert column(x, m, "w6s_volume_log_known").all()


def test_gap_epoch_and_invalid_snapshot_mask_whole_window():
    s, b = sample(10)
    origin = int(b.iloc[5].t_ns)
    s.loc[s.t_ns == origin-4*NS, "valid"] = False
    _, x, m = build_features(s, b)
    assert column(x, m, "w6s_supported")[5] == 0
    assert column(x, m, "w6s_volume_log_known")[5] == 0
    assert column(x, m, "w6s_supported")[6] == 1
    s = s[s.t_ns != origin - 2*NS]
    _, x, m = build_features(s, b)
    assert column(x, m, "w6s_supported")[5] == 0


def test_counter_regression_rejected_inside_valid_epoch():
    s, b = sample(10)
    s.loc[s.t_ns >= int(b.iloc[5].t_ns), "c_trade_buy"] -= 10000.
    with pytest.raises(ValueError, match="cumulative counter regressed"):
        build_features(s, b)


def test_closed_native_bb_no_developing_or_future_bar():
    s, b = sample(205)
    _, x, m = build_features(s, b)
    known = column(x, m, "bb_m9_middle_distance_scale_known")
    assert not known[:179].any()
    assert known[179:].all()
    b2 = b.copy()
    b2.loc[181:, ["o", "h", "l", "c"]] += 100.
    _, xx, _ = build_features(s, b2)
    np.testing.assert_array_equal(x[:181], xx[:181])


def test_price_translation_and_quantity_scaling():
    s, b = sample(40)
    _, x, m = build_features(s, b)
    ss, bb = s.copy(), b.copy()
    ss[["bid", "ask", "mid"]] += 1000
    bb[["o", "h", "l", "c"]] += 1000
    _, xx, mm = build_features(ss, bb)
    np.testing.assert_allclose(x, xx, rtol=2e-5, atol=2e-5)
    for name in ("trade_buy", "trade_sell", "trade_unknown"):
        s["c_"+name] *= 4
    _, scaled, _ = build_features(s, b)
    np.testing.assert_allclose(column(scaled, m, "w30s_ticks_per_100_contracts"), column(x, m, "w30s_ticks_per_100_contracts") / 4)
    np.testing.assert_allclose(column(scaled, m, "w30s_trade_imbalance"), column(x, m, "w30s_trade_imbalance"))


def test_week_boundary_geometry_is_exact_on_multiday_prefixes():
    # Sparse valid minutes isolate calendar geometry from rolling BB warmup.
    # Two Monday transitions exercise publication of the completed prior week.
    t = pd.date_range("2026-08-03 01:00", periods=16, freq="D", tz="UTC").asi8
    p = 3000. + np.arange(len(t))
    s = pd.DataFrame(dict(t_ns=t, epoch=0, valid=True, age_ns=0,
                          bid=p-.1, ask=p+.1, mid=p))
    b = pd.DataFrame(dict(t_ns=t, o=p-.1, h=p+.2, l=p-.2, c=p,
                          volume=1., coverage=1., epoch=0, valid=True))
    origins, x, manifest = build_features(s, b)
    for end in (8, 10, 15):
        prefix_origins, prefix, _ = build_features(s.iloc[:end], b.iloc[:end])
        np.testing.assert_array_equal(x[:end], prefix)
        pd.testing.assert_frame_equal(origins.iloc[:end], prefix_origins)
    previous_known = column(x, manifest, "previous_week_high_distance_scale_known")
    assert not previous_known[:7].any()
    assert previous_known[7:].all()
    assert column(x, manifest, "previous_week_coverage_minutes")[7] == 7
    # First Monday's current-week high is its own observed high, not Sunday's future high.
    assert column(x, manifest, "week_high_distance_scale")[7] == pytest.approx(-2.)
    assert column(x, manifest, "previous_week_high_distance_scale")[7] == pytest.approx(8.)
