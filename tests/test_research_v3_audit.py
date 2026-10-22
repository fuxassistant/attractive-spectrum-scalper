"""Counterexamples for independent v3 inference/weight/context diagnostics."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.audit_opportunity_v3 import (
    context_diagnostics, context_starts, event_balanced_metrics, factorial_effects,
    identity_audit, paired_interval, probability_metrics, recompute_train_weights,
    verify_saved_gradients,
)


def roots(n=8):
    return pd.DataFrame(dict(date=["2026-06-01"]*n,
        t_ns=np.arange(n, dtype=np.int64)*5_000_000_000,
        feature_available=np.ones(n, bool), physical_support_available=np.ones(n, bool),
        legacy_segment_token=np.zeros(n, np.int64), physical_segment_token=np.zeros(n, np.int64),
        y_crossing=np.zeros(n, np.int64), first_qualifying_event_id_L=[""]*n,
        first_qualifying_event_id_S=[""]*n))


def test_restored_keeps_missing_clock_position_and_unknown_y_does_not_reset():
    frame = roots()
    frame.loc[3, "feature_available"] = False
    frame.loc[2, "y_crossing"] = -1
    legacy, _ = context_starts(frame, "legacy")
    restored, _ = context_starts(frame, "restored")
    assert legacy.tolist() == [0, 0, 0, 3, 4, 4, 4, 4]
    assert restored.tolist() == [0]*8
    result = context_diagnostics(frame, {"train": ["2026-06-01"]}, receptive_field=8, supplied=8)
    restored_day = result["modes"]["restored"]["by_split"]["train"]
    assert restored_day["supported_clock_tokens"]["maximum"] == 8
    assert restored_day["observed_feature_tokens"]["maximum"] == 7
    assert restored_day["oldest_observed_span_seconds"]["maximum"] == 35
    assert restored_day["fully_observed_RF61_fraction"] == 0
    counts = result["reset_counts"]["train"]
    assert counts["legacy_extra_resets_with_physical_continuity"] == 2
    assert counts["physical_support_boundary_pairs"] == 0
    assert result["missing_X_runs_by_split"]["train"]["clock_occupancy_seconds"]["q50"] == 5


def test_hidden_physical_token_break_resets_supported_history():
    frame = roots()
    frame.loc[4:, ["physical_segment_token", "legacy_segment_token"]] = 1
    for mode in ("legacy", "restored"):
        start, _ = context_starts(frame, mode)
        assert start.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]
    changed = frame.copy()
    changed.loc[6:, "y_crossing"] = 3
    changed.loc[6:, "physical_segment_token"] = 99
    assert np.array_equal(context_starts(frame, "restored")[0][:6], context_starts(changed, "restored")[0][:6])


def test_context_resets_at_date_cadence_and_unsupported_origin():
    frame = roots()
    frame.loc[2, ["physical_support_available", "feature_available"]] = False
    frame.loc[5:, "t_ns"] += 5_000_000_000
    frame.loc[7:, "date"] = "2026-06-02"
    assert context_starts(frame, "restored")[0].tolist() == [0, 0, 2, 3, 3, 5, 5, 7]
    frame.loc[2, "feature_available"] = True
    with pytest.raises(ValueError, match="outside physical"):
        context_diagnostics(frame, {"train": ["2026-06-01"]})


def weighted_frame():
    frame = roots(9)
    frame["y_crossing"] = [0, 1, 1, 1, 2, 3, 3, 1, -1]
    frame["first_qualifying_event_id_L"] = ["", "A", "A", "B", "", "A", "D", "FUTURE", ""]
    frame["first_qualifying_event_id_S"] = ["", "", "", "", "C", "C", "A", "", ""]
    frame.loc[7:, "date"] = "2026-06-02"
    return frame


def test_train_weights_pool_side_memberships_average_both_and_preserve_class_mass():
    frame = weighted_frame()
    weights, audit = recompute_train_weights(frame, frame.date.eq("2026-06-01"), "crossing_id")
    # Pooled A multiplicity is 4: two LONG, one BOTH-LONG, one BOTH-SHORT.
    assert np.allclose(weights[:7], [1, .5, .5, 2, 1, .75, 1.25])
    assert weights[7:].tolist() == [0, 0]
    assert audit["positive_side_memberships"] == 8
    assert audit["distinct_positive_ids"] == 4
    assert audit["multiplicity"]["maximum"] == 4
    for group in audit["by_joint_class"].values():
        assert group["normalized"]["mass"] == pytest.approx(group["root_count"])
    assert audit["overall"]["kish_weight_ESS"] < 7
    assert audit["overall"]["kish_weight_ESS"] != audit["distinct_positive_ids"]


def test_weights_do_not_use_future_ids_or_unknown_unavailable_train():
    frame = weighted_frame()
    first, receipt = recompute_train_weights(frame, frame.date.eq("2026-06-01"), "crossing_id")
    frame.loc[7, "first_qualifying_event_id_L"] = "A"
    frame.loc[8, ["y_crossing", "first_qualifying_event_id_L", "first_qualifying_event_id_S"]] = [3, "A", "A"]
    second, changed = recompute_train_weights(frame, frame.date.eq("2026-06-01"), "crossing_id")
    assert np.array_equal(first, second)
    assert receipt["count_table_sha256"] == changed["count_table_sha256"]
    frame.loc[1, "feature_available"] = False
    unavailable, _ = recompute_train_weights(frame, frame.date.eq("2026-06-01"), "uniform")
    assert unavailable[1] == 0
    assert unavailable.sum() == 6


def test_event_metric_equal_side_id_denominator_not_root_count():
    frame = roots(3)
    frame["y_crossing"] = [1, 1, 0]
    frame["first_qualifying_event_id_L"] = ["A", "B", ""]
    p = np.array([[.1, .7, .1, .1], [.6, .2, .1, .1], [.8, .1, .05, .05]])
    base = event_balanced_metrics(frame, p)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    dup_p = np.concatenate([p, p[[0]]])
    again = event_balanced_metrics(duplicate, dup_p)
    assert base["metrics"] == pytest.approx(again["metrics"])
    assert base["distinct_side_ids"] == 2
    assert again["positive_root_side_labels"] == 3
    assert probability_metrics(frame.y_crossing, p)["joint_nll"] != probability_metrics(duplicate.y_crossing, dup_p)["joint_nll"]
    # Negative root cannot influence this positive-only metric.
    p[2] = [.1, .4, .4, .1]
    assert event_balanced_metrics(frame, p)["metrics"] == pytest.approx(base["metrics"])


def test_both_has_two_side_id_memberships_and_missing_id_fails():
    frame = roots(1)
    frame["y_crossing"] = 3
    frame["first_qualifying_event_id_L"] = "A"
    frame["first_qualifying_event_id_S"] = "A"
    result = event_balanced_metrics(frame, np.array([[.2, .1, .2, .5]]))
    assert result["positive_root_side_labels"] == result["distinct_side_ids"] == 2
    assert result["metrics"]["positive_side_nll"] == pytest.approx((-np.log(.6)-np.log(.7))/2)
    frame["first_qualifying_event_id_S"] = ""
    with pytest.raises(ValueError, match="no crossing identity"):
        event_balanced_metrics(frame, np.array([[.2, .1, .2, .5]]))


def test_identity_overlap_is_explicit_not_an_independence_claim():
    frame = weighted_frame()
    frame.loc[7, "first_qualifying_event_id_L"] = "A"
    result = identity_audit(frame, {"train": ["2026-06-01"], "test": ["2026-06-02"]})
    assert not result["no_cross_split_ids"]
    assert result["cross_split_intersections"]["train__test"]["shared_ids"] == 1


def factorial_rows():
    rows = []
    for seed in (17, 29):
        for date in ("2026-07-14", "2026-07-13"):
            for context, weighting, loss in (("legacy", "uniform", 1.), ("legacy", "crossing_id", .9),
                ("restored", "uniform", .8), ("restored", "crossing_id", .6)):
                rows.append(dict(split="test", date=date, seed=seed, context=context, weighting=weighting,
                    metrics=dict(joint_nll=loss, event_auc=1-loss/2)))
    return rows


def test_factorial_effect_sign_interaction_and_seed_mean_date_denominator():
    result = factorial_effects(factorial_rows(), replications=100)
    item = result["by_split"]["test"]["joint_nll"]
    expected = dict(context_at_uniform=.2, context_at_crossing_id=.3,
        weighting_at_legacy=.1, weighting_at_restored=.2, context_main=.25, weighting_main=.15, interaction=.1)
    for name, value in expected.items():
        effect = item["mean_seed_effect_paired_by_date"][name]
        assert effect["mean"] == pytest.approx(value)
        assert effect["days"] == 2  # two seeds on same date do not double the day denominator.
        assert effect["dates"] == ["2026-07-13", "2026-07-14"]
    assert result["by_split"]["test"]["event_auc"]["by_seed"]["17"]["context_at_uniform"]["mean"] == pytest.approx(.1)


def test_factorial_requires_matched_cells_and_drops_undefined_dates_jointly():
    rows = factorial_rows()
    with pytest.raises(ValueError, match="Missing matched"):
        factorial_effects(rows[:-1], replications=20)
    rows[0]["metrics"]["event_auc"] = None
    result = factorial_effects(rows, replications=20)
    assert result["by_split"]["test"]["event_auc"]["paired_dates"] == 1
    assert result["by_split"]["test"]["joint_nll"]["paired_dates"] == 2
    assert paired_interval([1, -2], ["a", "b"], 100) == paired_interval([1, -2], ["a", "b"], 100)


def test_saved_gradients_are_actual_derivatives_and_train_only():
    frame = roots(4)
    frame["y_crossing"] = [0, 1, 3, -1]
    p = np.tile([.4, .2, .1, .3], (4, 1))
    weights = np.array([2., .5, 0., 0.])
    train = weights > 0
    event = np.array([.6, -.4, -.4, np.nan])
    q = p[:, 1:]/.6
    q[1, 0] -= 1; q[2, 2] -= 1
    direction = np.array([0., np.linalg.norm(q[1]), np.linalg.norm(q[2]), np.nan])
    frame["g_event"], frame["g_direction_norm"] = event, direction
    frame["g_event_training_weighted"] = np.where(train, weights*event, np.nan)
    frame["g_direction_norm_training_weighted"] = np.where(train, weights*direction, np.nan)
    frame["optimization_weight_train_only"] = np.where(train, weights, np.nan)
    assert verify_saved_gradients(frame, p, weights, train)["status"] == "PASS"
    frame.loc[2, "g_event_training_weighted"] = 0.
    with pytest.raises(ValueError, match="nonlocal"):
        verify_saved_gradients(frame, p, weights, train)


@pytest.mark.parametrize("p", [np.array([[.1, .2, .3, .5]]), np.array([[np.nan, .2, .3, .5]]),
    np.array([[-.1, .2, .3, .6]])])
def test_probability_scores_fail_closed(p):
    with pytest.raises(ValueError):
        probability_metrics(np.array([0]), p)
