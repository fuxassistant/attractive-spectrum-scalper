"""Global MBO context: clock identity, support and direction controls."""
import numpy as np
import pandas as pd
import pytest

from experiments.zone_conditional_v1.mbo_features import (
    BASE_CONTEXT_COLUMNS, COUNTERS, MBO_FEATURE_COLUMNS, GlobalMBOContext,
    global_mbo_features,
)

NS = 1_000_000_000
BASE = 1_785_715_200_000_000_000


def fixture():
    i = np.arange(101, dtype=np.int64)
    t = BASE + i * NS
    s = pd.DataFrame(dict(t_ns=t, available_ns=t+NS//10, epoch=3,
                          valid=True, book_valid=True, mid=100.1+i*.1,
                          bid=100+i*.1, ask=100.2+i*.1))
    rates = dict(path_ticks=1, up_ticks=1, down_ticks=0, trade_buy=3, trade_sell=1,
                 trade_unknown=2, trade_count=2, add_bid=10, add_ask=5,
                 remove_bid=4, remove_ask=2, matched_fill_remove_bid=3,
                 matched_fill_remove_ask=1, post_trade_add_bid=2,
                 post_trade_add_ask=1, event_count=1)
    for name in COUNTERS:
        s["c_"+name] = i * rates[name]
    for k in (1, 4, 12):
        s[f"depth_bid_{k}"] = k*20
        s[f"depth_ask_{k}"] = k*10
    q = pd.DataFrame(dict(available_ns=t, epoch=3, valid=True,
                         bid_nano=(100_000_000_000+i*100_000_000),
                         ask_nano=(100_200_000_000+i*100_000_000)))
    c = pd.DataFrame(dict(candidate_id=[4], t_ns=[BASE+60*NS+NS//10],
                          episode_available_ns=[BASE+40*NS+NS//10], side=[1]))
    return s, q, c


def extract(s, q, c, **kwargs):
    return global_mbo_features(c, s, q, selection_available_ns=BASE+50*NS,
                               coverage_end_ns=BASE+101*NS, **kwargs)


def test_exact_counter_deltas_and_baseline_denominators():
    s, q, c = fixture(); r = extract(s, q, c).iloc[0]
    assert r.mbo_now_known == r.mbo_dep_known == 1
    assert r.mbometa_now_window_start_ns == BASE+30*NS
    assert r.mbometa_now_window_end_ns == BASE+60*NS
    assert r.mbometa_now_max_source_available_ns <= c.t_ns.iloc[0]
    assert r.mbo_now_trade_total_log == pytest.approx(np.log1p(180))
    assert r.mbo_now_trade_imbalance == .5
    assert r.mbo_now_trade_unknown_fraction == pytest.approx(1/3)
    assert r.mbo_now_support_matched_fill_log == pytest.approx(np.log1p(90))
    assert r.mbo_now_support_unmatched_remove_log == pytest.approx(np.log1p(30))
    assert r.mbo_now_native_quote_count_log == pytest.approx(np.log1p(30))
    assert r.mbo_now_endpoint_age_seconds == .1
    assert not set(BASE_CONTEXT_COLUMNS) & set(MBO_FEATURE_COLUMNS)
    assert "mbo_now_trade_total_log" in BASE_CONTEXT_COLUMNS
    assert "mbo_now_trade_imbalance" in MBO_FEATURE_COLUMNS


def test_direction_mirrors_aggression_book_and_progress():
    s, q, c = fixture(); a = extract(s, q, c).iloc[0]
    c["side"] = -1; b = extract(s, q, c).iloc[0]
    for name in ("trade_imbalance", "book_net_pressure", "depth_12_toward_imbalance",
                 "toward_progress_ticks", "toward_path_efficiency"):
        assert a["mbo_now_"+name] == pytest.approx(-b["mbo_now_"+name])
    assert a.mbo_now_support_matched_fill_log == b.mbo_now_opposition_matched_fill_log
    assert a.mbo_now_trade_toward_log == b.mbo_now_trade_away_log
    assert a.mbo_now_trade_total_log == b.mbo_now_trade_total_log


def test_historical_preselection_context_is_legal_at_selected_decision():
    s, q, c = fixture(); r = extract(s, q, c).iloc[0]
    assert c.episode_available_ns.iloc[0] < BASE+50*NS
    assert r.mbo_dep_known == 1
    c["t_ns"] = BASE+49*NS
    with pytest.raises(ValueError, match="Route"):
        extract(s, q, c)


def test_prefix_causality_later_values_cannot_repair_or_change_history():
    s, q, c = fixture(); cutoff = c.t_ns.iloc[0]
    a = extract(s, q, c)
    b = extract(s.loc[s.available_ns <= cutoff], q.loc[q.available_ns <= cutoff], c)
    pd.testing.assert_frame_equal(a, b)
    # Even when the current window is clean, a late-publication historical row
    # cannot be used retroactively at its physical endpoint.
    s.loc[40, "available_ns"] = BASE+41*NS
    s.loc[41, "available_ns"] = BASE+41*NS+NS//10
    r = extract(s, q, c).iloc[0]
    assert r.mbometa_dep_window_end_ns == BASE+39*NS


def test_one_nanosecond_release_boundary_is_exact():
    s, q, c = fixture()
    c["t_ns"] = BASE+60*NS+NS//10-1
    before = extract(s, q, c).iloc[0]
    c["t_ns"] += 1
    after = extract(s, q, c).iloc[0]
    assert before.mbometa_now_window_end_ns == BASE+59*NS
    assert after.mbometa_now_window_end_ns == BASE+60*NS
    assert after.mbometa_now_max_source_available_ns == c.t_ns.iloc[0]


@pytest.mark.parametrize("failure", ["invalid", "epoch", "negative_epoch", "missing_second", "counter_reset"])
def test_interior_support_or_counter_failure_cannot_hide_in_endpoints(failure):
    s, q, c = fixture()
    if failure == "invalid":
        q.loc[45, "valid"] = False
    elif failure == "epoch":
        q.loc[45:46, "epoch"] = 4
    elif failure == "negative_epoch":
        q.loc[45, "epoch"] = -1
    elif failure == "missing_second":
        s = s.drop(index=45)
    else:
        s.loc[45, "c_trade_buy"] = 0
    r = extract(s, q, c).iloc[0]
    assert r.mbo_now_known == 0
    assert np.isnan(r.mbo_now_trade_toward_log)
    assert r.mbo_dep_known == 1


def test_unsampled_invalid_native_publication_invalidates_window():
    s, q, c = fixture()
    bad = q.iloc[[45]].copy(); bad["available_ns"] += NS//2; bad["valid"] = False
    q = pd.concat([q, bad]).sort_values("available_ns", kind="stable")
    assert extract(s, q, c).iloc[0].mbo_now_known == 0


def test_gap_and_stale_endpoint_are_unknown_not_zero():
    s, q, c = fixture(); q = q.loc[(q.available_ns <= BASE+31*NS) | (q.available_ns >= BASE+59*NS)]
    r = extract(s, q, c, max_quote_gap_seconds=20).iloc[0]
    assert r.mbo_now_known == 0
    s, q, c = fixture(); s = s.loc[s.t_ns <= BASE+55*NS]
    r = extract(s, q, c).iloc[0]
    assert r.mbometa_now_status == "STALE_PHYSICAL_ENDPOINT"
    assert np.isnan(r.mbo_now_trade_total_log)


def test_no_trades_is_known_zero_quantity_but_undefined_imbalance():
    s, q, c = fixture()
    for name in ("trade_buy", "trade_sell", "trade_unknown", "trade_count"):
        s["c_"+name] = 0
    r = extract(s, q, c).iloc[0]
    assert r.mbo_now_known == 1 and r.mbo_now_trade_total_log == 0
    assert np.isnan(r.mbo_now_trade_imbalance)
    assert np.isnan(r.mbo_now_trade_unknown_fraction)


def test_matched_f_is_not_added_to_trade_volume_and_invalid_partition_rejected():
    s, q, c = fixture(); r = extract(s, q, c).iloc[0]
    assert np.expm1(r.mbo_now_trade_total_log) == pytest.approx(180)
    s["c_matched_fill_remove_bid"] = s.c_remove_bid*2
    assert extract(s, q, c).iloc[0].mbometa_now_status == "MATCHED_EXCEEDS_REMOVAL"


def test_float_clocks_and_future_departure_are_rejected():
    s, q, c = fixture(); c["t_ns"] = c.t_ns.astype(float)
    with pytest.raises(TypeError):
        extract(s, q, c)
    s, q, c = fixture(); c["episode_available_ns"] = c.t_ns+1
    with pytest.raises(ValueError, match="Historical"):
        extract(s, q, c)


def test_query_after_explicit_coverage_is_unknown():
    s, q, c = fixture()
    obj = GlobalMBOContext(s, q, selection_available_ns=BASE,
                          coverage_end_ns=BASE+59*NS)
    x, meta = obj.window(int(c.t_ns.iloc[0]), 1)
    assert x["known"] == 0 and meta["status"] == "AFTER_SOURCE_COVERAGE"
