"""Meaningful causality, continuity and directional checks for the lab signals."""
import numpy as np
import pandas as pd
import pytest

from experiments.mechanical_lab.signals import (
    SIGNAL_CATALOG, VETO_CATALOG, accept_mask, build_features, generate_candidates,
)

NS = 1_000_000_000


def bars(n=700, seed=12):
    rng = np.random.default_rng(seed)
    close = 4700 + np.cumsum(rng.normal(0, 0.65, n))
    opening = np.r_[close[0] - 0.1, close[:-1]]
    t = pd.Timestamp("2026-08-03T00:01:00Z").value + np.arange(n, dtype=np.int64) * 60 * NS
    return pd.DataFrame(dict(t_ns=t, available_ns=t + 123456789, o=opening,
                             h=np.maximum(close, opening) + 0.2, l=np.minimum(close, opening) - 0.2,
                             c=close, epoch=np.ones(n, dtype=np.int64), coverage=1.0, valid=True, volume=10.0))


def test_features_and_signals_are_prefix_causal_at_opening_range_boundary():
    source = bars()
    full = build_features(source)
    # Includes the developing opening range and its first completed value.
    for count in (430, 449, 450, 451, 570):
        prefix = build_features(source.iloc[:count])
        pd.testing.assert_frame_equal(prefix, full.iloc[:count])
        a = generate_candidates(prefix)
        b = generate_candidates(full[full.bar_end_ns <= source.t_ns.iloc[count - 1]])
        pd.testing.assert_frame_equal(a, b)


def test_bb_sample_sigma_atr_and_prior_range_have_independent_definitions():
    source = bars()
    f = build_features(source)
    k = 550
    window = source.c.iloc[k - 19:k + 1].to_numpy()
    assert f.bb_upper.iloc[k] == pytest.approx(window.mean() + 2 * window.std(ddof=1))
    assert f.prior_high20.iloc[k] == source.h.iloc[k - 20:k].max()
    tr = [max(source.h.iloc[i] - source.l.iloc[i], abs(source.h.iloc[i] - source.c.iloc[i - 1]),
              abs(source.l.iloc[i] - source.c.iloc[i - 1])) for i in range(k - 13, k + 1)]
    assert f.atr14.iloc[k] == pytest.approx(np.mean(tr))
    assert f.t_ns.iloc[k] == source.available_ns.iloc[k]
    assert f.bar_end_ns.iloc[k] == source.t_ns.iloc[k]


@pytest.mark.parametrize("cause", ["invalid", "epoch", "gap", "partial"])
def test_continuity_break_restarts_warmup(cause):
    source = bars()
    index = 500
    if cause == "invalid":
        source.loc[index, "valid"] = False
    elif cause == "partial":
        source.loc[index, "coverage"] = 0.99
    elif cause == "epoch":
        source.loc[index:, "epoch"] = 2
    else:
        source = source.drop(index).reset_index(drop=True)
    f = build_features(source)
    assert f.ready.iloc[499]
    assert not f.ready.iloc[index:index + 119].any()
    assert f.ready.iloc[index + 120]


def test_all_directional_patterns_are_reflection_symmetric():
    source = bars()
    reflected = source.copy()
    reflected["o"] = 9400 - source.o
    reflected["c"] = 9400 - source.c
    reflected["h"] = 9400 - source.l
    reflected["l"] = 9400 - source.h
    ids = [k for k in SIGNAL_CATALOG if k != "control_clock"]
    a = generate_candidates(build_features(source), ids)
    b = generate_candidates(build_features(reflected), ids)
    a = a[["t_ns", "signal_id", "side"]].sort_values(["t_ns", "signal_id", "side"]).reset_index(drop=True)
    b["side"] *= -1
    b = b[["t_ns", "signal_id", "side"]].sort_values(["t_ns", "signal_id", "side"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_breakout_excludes_trigger_bar_and_uses_information_clock():
    source = bars()
    index = 510
    source.loc[index, "c"] = source.h.iloc[index - 20:index].max() + 3
    source.loc[index, "h"] = source.c.iloc[index] + 0.1
    f = build_features(source)
    c = generate_candidates(f, ["range_breakout"])
    row = c[c.bar_end_ns == source.t_ns.iloc[index]]
    assert row.side.tolist() == [1]
    assert row.t_ns.iloc[0] > source.t_ns.iloc[index]
    assert row.risk_hint.iloc[0] == f.atr14.iloc[index]


def test_control_clock_and_veto_profiles_are_causal_side_aware():
    f = build_features(bars())
    c = generate_candidates(f, ["control_clock"])
    assert (c.minute_utc % 5 == 0).all()
    assert (c.minute_utc >= 420).all()
    assert (c.minute_utc < 1230).all()
    assert len(c) > 30
    for veto in VETO_CATALOG:
        assert accept_mask(c, veto).dtype == bool
        assert len(accept_mask(c, veto)) == len(c)
    opposite = c.copy()
    opposite["side"] = -1
    assert not (accept_mask(c, "trend_aligned") & accept_mask(opposite, "trend_aligned")).any()


def test_reentry_and_failed_breakout_require_close_inside_not_across_whole_range():
    f = build_features(bars()).iloc[[500]].copy()
    f["ready"] = True
    f["prev_c"] = 4690
    f["prev_bb_lower"] = 4695
    f["prev_bb_upper"] = 4705
    f["bb_lower"] = f["prior_low20"] = f["opening_low"] = 4695
    f["bb_upper"] = f["prior_high20"] = f["opening_high"] = 4705
    f["o"] = 4696
    f["l"] = 4690
    f["h"] = 4712
    f["c"] = 4710
    assert generate_candidates(f, ["bb_reentry", "range_failure", "opening_range_failure"]).empty
    f["c"] = 4700
    c = generate_candidates(f, ["bb_reentry", "range_failure", "opening_range_failure"])
    assert len(c) == 3
    assert c.side.tolist() == [1, 1, 1]


def test_input_rejects_noncausal_time_future_columns_and_unknown_knobs():
    source = bars()
    source.loc[10, "available_ns"] = source.t_ns.iloc[10] - 1
    with pytest.raises(ValueError, match="availability"):
        build_features(source)
    source = bars()
    source["future_pnl"] = 0
    with pytest.raises(ValueError, match="forbidden"):
        build_features(source)
    f = build_features(bars())
    with pytest.raises(ValueError, match="Unknown parameters"):
        generate_candidates(f, parameters={"large_candle_fade": {"unknown": 2}})
    with pytest.raises(ValueError, match="finite positive"):
        generate_candidates(f, parameters={"large_candle_fade": {"body_atr": -2}})


def test_decision_window_uses_publication_clock_and_preserves_nanoseconds():
    source = bars(1300)
    source["available_ns"] += 61 * NS
    f = build_features(source)
    expected = (source.available_ns - source.t_ns) / NS
    np.testing.assert_array_equal(f.bar_publication_delay_seconds, expected)
    assert not f.loc[f.minute_utc >= 1230, "ready"].any()
    candidates = generate_candidates(f)
    assert (candidates.t_ns >= pd.Timestamp("2026-08-03T07:00:00Z").value).all()
    assert (candidates.t_ns < pd.Timestamp("2026-08-03T20:30:00Z").value).all()
    assert (candidates.t_ns - candidates.bar_end_ns == 61 * NS + 123456789).all()
    source["available_ns"] = source.available_ns.astype(float)
    with pytest.raises(ValueError, match="integer nanoseconds"):
        build_features(source)
