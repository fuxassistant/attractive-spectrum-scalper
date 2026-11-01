import numpy as np
import pandas as pd
import pytest

from experiments.zone_conditional_v1.targets import build_targets, NS


BASE = 1_787_558_880_123_456_789


def quotes(times, mids, *, valid=None, epochs=None, base=BASE):
    n = len(times)
    clocks = np.rint(np.asarray(times) * NS).astype(np.int64) + base
    prices = np.rint(np.asarray(mids) * NS).astype(np.int64)
    return pd.DataFrame(dict(available_ns=clocks, bid_nano=prices - 100_000_000,
        ask_nano=prices + 100_000_000, valid=np.ones(n, bool) if valid is None else valid,
        epoch=np.zeros(n, np.int64) if epochs is None else np.asarray(epochs, np.int64),
        source_ordinal=np.arange(100, 100 + n, dtype=np.int64)))


def candidates(*, target=102., side=1, risk=1., decision=0, base=BASE):
    return pd.DataFrame(dict(candidate_id=np.array([7], np.int64),
        origin_index=np.array([11], np.int64), t_ns=np.array([base + decision], np.int64),
        side=np.array([side], np.int64), target_nano=np.array([round(target * NS)], np.int64),
        risk_per_oz=[risk], inside=[False], target_available_ns=np.array([base + decision], np.int64)))


def label(q, c=None, *, coverage=400, **kwargs):
    return build_targets(q, candidates() if c is None else c,
                         coverage_end_ns=BASE + int(coverage * NS), **kwargs).iloc[0]


@pytest.mark.parametrize("side,target,path,state", [
    (1, 102, [100, 102, 98], "RETURN_FIRST"),
    (1, 102, [100, 99, 103], "ADVERSE_FIRST"),
    (-1, 98, [100, 98, 102], "RETURN_FIRST"),
    (-1, 98, [100, 101, 97], "ADVERSE_FIRST"),
])
def test_first_event_order_and_symmetric_sides(side, target, path, state):
    row = label(quotes([0, 1, 2], path), candidates(side=side, target=target))
    assert row.state == state and row.label_known
    assert row.first_event_ns == row.label_available_ns == BASE + NS
    assert row.first_event_bin_seconds == 30 and row.censor_ns == -1
    assert row.event_source_ordinal == 101


def test_only_strict_future_publications_and_last_origin_snapshot():
    q = quotes([0, 0, 1], [103, 100, 102])
    row = label(q)
    assert row.origin_mid_x2_nano == 200 * NS
    assert row.origin_source_ordinal == 101 and row.first_event_ns == BASE + NS


@pytest.mark.parametrize("seconds,expected", [(30, 30), (30.000000001, 60),
    (60, 60), (60.000000001, 120), (120, 120), (120.000000001, 300), (300, 300)])
def test_inclusive_bins_and_horizon_endpoint(seconds, expected):
    times = sorted(set([0., *np.arange(20, seconds, 20), seconds]))
    q = quotes(times, [100.] * (len(times) - 1) + [102.])
    row = label(q)
    assert row.state == "RETURN_FIRST" and row.first_event_bin_seconds == expected
    assert row.first_event_ns == BASE + round(seconds * NS)


def test_future_endpoint_hit_and_invalid_data_after_horizon_do_not_leak():
    q = quotes([*range(0, 301, 20), 301], [100] * 16 + [102], valid=[True] * 16 + [False])
    row = label(q)
    assert row.state == "TIMEOUT" and row.label_available_ns == BASE + 300 * NS
    assert row.first_event_ns == row.first_event_bin_seconds == row.censor_ns == -1


@pytest.mark.parametrize("failure,expected", [("invalid", "INVALID_QUOTE"), ("epoch", "EPOCH_CHANGE")])
def test_first_support_failure_censors_and_cannot_be_bridged(failure, expected):
    q = quotes([0, 1, 2, 3], [100, 100, 102, 98])
    if failure == "invalid":
        q.loc[1, "valid"] = False
    else:
        q.loc[1:, "epoch"] = 1
    row = label(q)
    assert row.state == "CENSORED" and row.reason == expected and not row.label_known
    assert row.censor_ns == row.label_available_ns == BASE + NS and row.first_event_ns == -1


def test_stale_gap_censors_at_expiry_not_later_recovery_or_future_hit():
    row = label(quotes([0, 30.000000001], [100, 102]))
    assert row.state == "CENSORED" and row.reason == "STALE_QUOTE"
    assert row.censor_ns == BASE + 30 * NS and row.first_event_ns == -1
    at_limit = label(quotes([0, 30], [100, 102]))
    assert at_limit.state == "RETURN_FIRST"


def test_right_censor_and_prefix_maturity_are_stable_after_known_event():
    q = quotes([0, 20, 40, 50, 60], [100, 100, 102, 98, 97])
    early = label(q, asof_ns=BASE + 39 * NS)
    assert early.state == "CENSORED" and early.reason == "RIGHT_CENSOR"
    assert early.label_available_ns == early.censor_ns == BASE + 39 * NS
    mature = label(q, asof_ns=BASE + 40 * NS)
    full = label(q)
    pd.testing.assert_series_equal(mature, full)
    assert mature.state == "RETURN_FIRST"
    physical_prefix = label(q.loc[q.available_ns <= BASE + 40 * NS], asof_ns=BASE + 40 * NS)
    pd.testing.assert_series_equal(mature, physical_prefix)


def test_timeout_is_unknown_until_horizon_and_coverage_end_is_exclusive():
    q = quotes(range(0, 300, 20), [100] * 15)
    early = label(q, asof_ns=BASE + 300 * NS - 1)
    assert early.state == "CENSORED" and early.reason == "RIGHT_CENSOR"
    assert early.label_available_ns == BASE + 300 * NS - 1
    mature = label(q, asof_ns=BASE + 300 * NS)
    assert mature.state == "TIMEOUT" and mature.label_available_ns == BASE + 300 * NS
    ended = label(q, coverage=300)
    assert ended.state == "CENSORED" and ended.reason == "COVERAGE"
    assert ended.label_available_ns == BASE + 300 * NS


def test_same_time_competing_hits_are_explicitly_ambiguous_despite_source_order():
    q = quotes([0, 1, 1, 2], [100, 102, 99, 100])
    row = label(q)
    assert row.state == "CENSORED" and row.reason == "AMBIGUOUS_SAME_TIME"
    assert row.censor_ns == BASE + NS and row.first_event_ns == -1
    q.iloc[[1, 2]] = q.iloc[[2, 1]].to_numpy()
    other = label(q)
    assert other.state == row.state and other.censor_ns == row.censor_ns


def test_invalid_row_same_time_as_hit_and_invalid_endpoint_censor():
    row = label(quotes([0, 1, 1], [100, 102, 102], valid=[True, True, False]))
    assert row.state == "CENSORED" and row.reason == "INVALID_QUOTE"
    times = list(range(0, 301, 20))
    row = label(quotes(times, [100] * len(times), valid=[True] * (len(times) - 1) + [False]))
    assert row.state == "CENSORED" and row.censor_ns == BASE + 300 * NS


def test_jump_across_zone_counts_return_without_claiming_touch_or_dwell():
    row = label(quotes([0, 1], [100, 110]))
    assert row.state == "RETURN_FIRST" and row.reason == "RETURN_EDGE"


def test_exact_half_nanodollar_midpoint_and_risk_boundary():
    q = quotes([0, 1, 2], [100, 99, 99])
    q.loc[0, "ask_nano"] += 1
    q.loc[1, "ask_nano"] += 2
    row = label(q)
    assert row.origin_mid_x2_nano == 200 * NS + 1
    assert row.adverse_x2_nano == 198 * NS + 1
    assert row.state == "ADVERSE_FIRST" and row.first_event_ns == BASE + 2 * NS


def test_origin_support_and_wrong_geometry_preserve_candidate_row():
    unsupported = label(quotes([0, 1], [100, 102], valid=[False, True]))
    assert unsupported.state == "CENSORED" and unsupported.reason == "ORIGIN_SUPPORT"
    wrong = label(quotes([0, 1], [100, 102]), candidates(target=99))
    assert wrong.state == "CENSORED" and wrong.reason == "TARGET_GEOMETRY"
    assert wrong.candidate_id == 7 and wrong.label_available_ns == BASE


@pytest.mark.parametrize("where,column", [("quotes", "available_ns"), ("quotes", "bid_nano"),
    ("candidates", "t_ns"), ("candidates", "candidate_id"), ("candidates", "target_available_ns")])
def test_float_clocks_prices_and_ids_are_rejected(where, column):
    q = quotes([0, 1], [100, 102]);c = candidates()
    frame = q if where == "quotes" else c
    frame[column] = frame[column].astype(float)
    with pytest.raises(ValueError, match="exact integers"):
        label(q, c)


def test_sort_uniqueness_geometry_availability_and_asof_validation():
    q = quotes([0, 1], [100, 102]);c = candidates()
    with pytest.raises(ValueError, match="sorted"):
        label(q.iloc[::-1], c)
    with pytest.raises(ValueError, match="Unique"):
        build_targets(q, pd.concat([c, c], ignore_index=True), coverage_end_ns=BASE + 400 * NS)
    c.loc[0, "target_available_ns"] = BASE + 1
    with pytest.raises(ValueError, match="later than"):
        label(q, c)
    with pytest.raises(ValueError, match="exact integer"):
        label(q, asof_ns=float(BASE + NS))


def test_asof_before_decision_and_empty_candidates_keep_explicit_types():
    q = quotes([0, 1], [100, 102]);c = candidates()
    row = label(q, asof_ns=BASE - 1)
    assert row.state == "CENSORED" and row.reason == "RIGHT_CENSOR"
    assert row.label_available_ns == BASE - 1 and row.origin_quote_index == -1
    out = build_targets(q, c.iloc[:0], coverage_end_ns=BASE + 400 * NS)
    assert out.empty
    assert out.first_event_ns.dtype == np.int64 and out.label_available_ns.dtype == np.int64
