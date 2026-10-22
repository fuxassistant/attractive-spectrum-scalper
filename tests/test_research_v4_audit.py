"""False-signal meaning, threshold fitting locality, and soft-weight checks."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_opportunity_v4 import (alarm_metrics, empirical_none_threshold,
    false_signal_audit, paired_effects, root_metrics, roughness, signals, soft_weights)


def roots(n):
    return pd.DataFrame(dict(date=["2026-07-13"]*n, t_ns=np.arange(n, dtype=np.int64)*5_000_000_000,
        feature_available=np.ones(n, bool), physical_support_available=np.ones(n, bool),
        physical_segment_token=np.zeros(n, np.int64), y_crossing=np.zeros(n, np.int64),
        first_qualifying_event_id_L=[""]*n, first_qualifying_event_id_S=[""]*n))


def semantic_frame():
    frame = roots(5)
    frame["y_crossing"] = [0, 1, 2, 3, -1]
    frame["first_qualifying_event_id_L"] = ["", "L", "", "BL", ""]
    frame["first_qualifying_event_id_S"] = ["", "", "S", "BS", ""]
    p = np.array([[.1, .6, .2, .1], [.1, .2, .6, .1], [.4, .1, .4, .1],
                  [.2, .3, .3, .2], [.05, .7, .15, .1]])
    return frame, p


def test_none_false_positive_wrong_side_both_unknown_and_inclusive_tie():
    frame, p = semantic_frame()
    event = alarm_metrics(frame, p, .5, "event")
    side = alarm_metrics(frame, p, .5, "selected_side")
    assert (event["alerts"], event["true_positives"], event["none_false_positives"], event["wrong_side_alerts"]) == (4, 3, 1, 0)
    assert (side["alerts"], side["true_positives"], side["none_false_positives"], side["wrong_side_alerts"]) == (4, 2, 1, 1)
    assert side["precision"] == .5
    assert side["none_fpr"] == 1
    assert side["event_opportunity_recall"] == pytest.approx(2/3)
    assert side["conditional_selected_side_recall"] == 1
    assert side["unknown_alerts"] == event["unknown_alerts"] == 1
    assert signals(p)[1][3] == 1  # exact side-score tie chooses LONG.
    assert event["positive_identity"]["covered_positive_side_ids"] == 4
    assert side["positive_identity"]["covered_positive_side_ids"] == 2
    assert side["positive_identity"]["coverage"] == .5


def test_zero_alerts_and_zero_negative_denominator_remain_undefined_not_success():
    frame, p = semantic_frame()
    m = alarm_metrics(frame, p, 1.01, "selected_side")
    assert m["alerts"] == m["false_signals"] == 0
    assert m["precision"] is None
    assert m["event_opportunity_recall"] == 0
    assert m["positive_identity"]["repeat_multiplicity"]["n"] == 0
    choice = empirical_none_threshold(np.array([.4, .8]), np.zeros(2, bool))
    assert choice["threshold"] == 1.01
    assert choice["calibration_none_fpr"] is None
    assert choice["reason"] == "insufficient_negative_denominator"


def test_positive_identity_repeats_count_memberships_not_duration_or_new_ids():
    frame, p = semantic_frame()
    duplicate = pd.concat([frame, frame.iloc[[2, 2]]], ignore_index=True)
    result = alarm_metrics(duplicate, np.concatenate([p, p[[2, 2]]]), .5, "selected_side")
    assert result["positive_identity"]["covered_positive_side_ids"] == 2
    assert result["positive_identity"]["covered_root_side_memberships"] == 4
    assert result["positive_identity"]["repeats_beyond_first"] == 2
    assert result["positive_identity"]["repeat_multiplicity"]["maximum"] == 3


def test_calibration_order_statistic_budget_and_ties_are_conservative():
    scores = np.linspace(.01, .99, 20)
    result = empirical_none_threshold(scores, np.ones(20, bool))
    assert result["allowed_none_alerts"] == result["actual_none_alerts"] == 1
    assert result["threshold"] == np.nextafter(scores[-2], np.inf)
    assert (scores >= result["boundary_score"]).sum() == 2
    scores[-3:] = .99
    tied = empirical_none_threshold(scores, np.ones(20, bool))
    assert tied["actual_none_alerts"] == 0
    assert tied["ties_at_boundary"] == 3
    assert tied["unused_budget_due_to_ties"] == 1


def test_calibration_ignores_positive_unknown_test_scores_and_reuses_frozen_rule():
    scores = np.array([.1, .2, .3, .4, .9, .7])
    none = np.array([True, True, True, True, False, False])
    a = empirical_none_threshold(scores, none, maximum_fpr=.25)
    scores[~none] = [.99, .01]
    b = empirical_none_threshold(scores, none, maximum_fpr=.25)
    assert a == b
    # A frozen threshold can exceed its CAL FPR budget on a shifted TEST distribution.
    test_scores = np.array([.8, .8])
    assert (test_scores >= a["threshold"]).mean() == 1


def test_alarm_and_multisplit_audit_do_not_mutate_feature_admission():
    frame, p = semantic_frame()
    frame.loc[2:, "date"] = "2026-07-14"
    original = frame.copy(deep=True)
    alarm_metrics(frame, p, .5, "event", frame.date.eq("2026-07-13"))
    pd.testing.assert_frame_equal(frame, original)
    split = {"calibration": ["2026-07-13"], "test": ["2026-07-14"]}
    result = false_signal_audit(frame, p, split)
    pd.testing.assert_frame_equal(frame, original)
    assert result["by_split"]["calibration"]["coverage"]["admitted_roots"] == 2
    assert result["by_split"]["test"]["coverage"]["admitted_roots"] == 3
    assert result["by_split"]["test"]["coverage"]["unknown_admitted_roots"] == 1
    assert result["calibrated_none_fpr"]["event"]["test"]["known_roots"] == 2


def test_future_label_mutation_cannot_change_cal_threshold():
    frame, p = semantic_frame()
    frame.loc[2:, "date"] = "2026-07-14"
    split = {"calibration": ["2026-07-13"], "test": ["2026-07-14"]}
    a = false_signal_audit(frame, p, split)
    frame.loc[2:, "y_crossing"] = 0
    p[2:] = [.9, .04, .03, .03]
    b = false_signal_audit(frame, p, split)
    for channel in ("event", "selected_side"):
        assert a["calibrated_none_fpr"][channel]["selection"] == b["calibrated_none_fpr"][channel]["selection"]


def test_physical_roughness_ignores_unknown_nonadmitted_and_cross_epoch_pairs():
    frame = roots(8)
    frame["y_crossing"] = [0, 1, -1, 0, 1, 0, 1, 0]
    frame.loc[5, "feature_available"] = False
    frame.loc[7, "physical_segment_token"] = 1
    p = np.tile([.4, .3, .2, .1], (8, 1))
    result = roughness(frame, p, {"test": ["2026-07-13"]})["test"]
    assert result["adjacent_pairs"] == 2  # only 0->1 and 3->4.
    assert result["target_event_flip_fraction"] == 1
    assert result["event_probability_total_variation_per_step"] == 0


def test_soft_weights_bound_after_normalization_preserve_classes_and_use_train_only():
    frame = roots(100)
    frame["y_crossing"] = 1
    frame["first_qualifying_event_id_L"] = ["frequent"]*90+[f"rare{i}" for i in range(10)]
    train = np.arange(100) < 95
    weights, audit = soft_weights(frame, train, "soft_crossing_id")
    assert weights[train].min() >= .5 and weights[train].max() <= 2
    assert weights[train].sum() == pytest.approx(95)
    assert np.all(weights[95:] == 0)
    assert np.all(weights[90:95] == 2)
    frame.loc[95:, "first_qualifying_event_id_L"] = "frequent"
    other, changed = soft_weights(frame, train, "soft_crossing_id")
    assert np.array_equal(weights, other)
    assert audit["count_table_sha256"] == changed["count_table_sha256"]


def test_pooled_both_membership_and_none_invariance():
    frame = roots(7)
    frame["y_crossing"] = [0, 1, 1, 1, 2, 3, 3]
    frame["first_qualifying_event_id_L"] = ["", "A", "A", "B", "", "A", "D"]
    frame["first_qualifying_event_id_S"] = ["", "", "", "", "C", "C", "A"]
    weights, result = soft_weights(frame, np.ones(7, bool), "soft_crossing_id")
    assert result["positive_side_memberships"] == 8
    assert weights[0] == 1
    assert result["by_joint_class"]["LONG_ONLY"]["raw"]["q50"] == .5  # A pooled multiplicity4.
    assert result["by_joint_class"]["BOTH"]["raw"]["minimum"] == pytest.approx((.5+1/np.sqrt(2))/2)
    assert weights[5:].sum() == pytest.approx(2)
    uniform, _ = soft_weights(frame, np.ones(7, bool), "uniform")
    assert np.all(uniform == 1)


def test_paired_effect_sign_seed_day_denominator_and_missing_cells():
    rows = []
    for seed in (17, 29):
        for date in ("a", "b"):
            for weighting, value in (("uniform", .7), ("soft_crossing_id", .6)):
                rows.append(dict(split="test", date=date, seed=seed, weighting=weighting,
                    metrics=dict(joint_nll=value, event_auc=1-value, event_average_precision=1-value)))
    result = paired_effects(rows, replications=100)["by_split"]["test"]
    for metric in result:
        pooled = result[metric]["mean_seed_effect_paired_by_date"]
        assert pooled["days"] == 2
        assert pooled["mean"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="Missing matched"):
        paired_effects(rows[:-1], replications=10)


def test_conditional_loss_decomposition_and_secondary_ap():
    frame, p = semantic_frame()
    result = root_metrics(frame.y_crossing.to_numpy()[:4], p[:4])
    assert result["event_nll"] == pytest.approx(.25*result["none_nll"]+.75*result["positive_event_nll"])
    assert result["joint_nll"] == pytest.approx(result["event_nll"]+.75*result["direction_nll_given_event"])
    assert 0 <= result["event_average_precision"] <= 1
    assert result["selected_side_average_precision_true_roots"] == 2


def test_nonfinite_probability_fails_before_alarm_counting():
    frame, p = semantic_frame()
    p[-1, 0] = np.nan  # UNKNOWN prediction still must be a valid full-clock probability.
    with pytest.raises(ValueError):
        alarm_metrics(frame, p, .5, "event")
