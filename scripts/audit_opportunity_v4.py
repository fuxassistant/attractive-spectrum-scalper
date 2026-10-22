"""Independent v4 soft-ID comparison and preregistered false-signal diagnostics.

Only a COMPLETE four-fit suite may be inspected. CAL thresholds use negative
scores alone; TEST never selects a model, signal threshold, or trading policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_opportunity_v3 import (array_sha, event_balanced_metrics,
    identity_audit, paired_interval, probability_metrics, read, sha, side_memberships,
    stats, verify_saved_gradients, weight_stats, write_audit)

FIXED_THRESHOLDS = (.35, .5, .65, .8)
MAX_NONE_FPR = .05
SEEDS = (17, 29)
WEIGHTINGS = ("uniform", "soft_crossing_id")
PROTOCOL = dict(version="V4_FALSE_SIGNAL_PROTOCOL_1", fixed_thresholds=list(FIXED_THRESHOLDS),
    event_score="1-p_none", selected_side_score="max(p_long+p_both,p_short+p_both)",
    side_tie="LONG", alert_rule="score >= threshold; float64 computation",
    calibration="On known X-admitted CAL NONE roots only: sort scores descending, k=floor(0.05*N_NONE), threshold=nextafter(sorted_scores[k],+infinity) in float64. This is the lowest representable threshold satisfying the inclusive NONE-alert budget; ties are conservative. No positive score, TEST label, or TEST score participates.",
    zero_negative_calibration="WAIT threshold1.01; no FPR-budget claim",
    event_tp="alert AND y>0", selected_side_tp="alert AND selected side belongs to y; BOTH includes both sides",
    none_false_positive="alert AND y==0", wrong_side="selected-side alert AND y>0 AND selected side does not belong to y",
    primary_recall="Event true alerts / all known positive event roots; selected-side correct alerts / the SAME all-positive-event denominator",
    positive_identity="Equal (side, first qualifying ID) identities on known admitted roots. E alerts cover all true-positive side memberships; selected-side alerts only their correctly selected membership. ID repetition is alerted root-side multiplicity, not duration or independent sample count.",
    unknown="Unknown labels remain in coverage/alert counts, never treated as NONE or scored as false alarms",
    policy_separation="Signal thresholds are descriptive alarm diagnostics; existing after-cost CAL trading selector is unchanged",
    max_calibration_none_fpr=MAX_NONE_FPR)


def signals(probability):
    p = np.asarray(probability, np.float64)
    probability_metrics(np.zeros(len(p), np.int64), p)
    long, short = p[:, 1]+p[:, 3], p[:, 2]+p[:, 3]
    return {"event": 1-p[:, 0], "selected_side": np.maximum(long, short)}, np.where(long >= short, 1, 2)


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def root_metrics(y, p):
    out = probability_metrics(y, p)
    y, p = np.asarray(y), np.asarray(p, np.float64)
    scores, side = signals(p)
    event = y > 0
    ep = np.clip(scores["event"], 1e-12, 1-1e-12)
    chosen_true = (y & side) != 0
    out.update(none_nll=float(-np.log1p(-ep[~event]).mean()) if (~event).any() else None,
        positive_event_nll=float(-np.log(ep[event]).mean()) if event.any() else None,
        event_average_precision=float(average_precision_score(event, scores["event"])) if event.any() else None,
        selected_side_average_precision=float(average_precision_score(chosen_true, scores["selected_side"])) if chosen_true.any() else None,
        selected_side_average_precision_true_roots=int(chosen_true.sum()),
        descriptor_note="NONE/positive-event NLL and average precision are secondary descriptors. Selected-side AP uses chosen-side membership as truth, whose positive denominator depends on the model; opportunity recall elsewhere uses all positive events as a common denominator.")
    return out


def identity_coverage(frame, known, alerts, selected_side, channel):
    rows, sides, ids = side_memberships(frame, known)
    keys = list(zip(sides.tolist(), ids))
    covered = alerts[rows]
    if channel == "selected_side":
        covered &= selected_side[rows] == (1 << sides)
    counts = Counter(key for key, used in zip(keys, covered) if used)
    total = len(set(keys))
    return dict(available_positive_side_ids=total, covered_positive_side_ids=len(counts),
        coverage=ratio(len(counts), total), covered_root_side_memberships=int(covered.sum()),
        repeats_beyond_first=int(covered.sum())-len(counts),
        repeat_multiplicity=stats(list(counts.values())),
        denominator="Distinct (side, ID) among all known admitted positive memberships; BOTH supplies two memberships")


def alarm_metrics(frame, probability, threshold, channel, eligible=None):
    if channel not in {"event", "selected_side"} or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Known alarm channel and finite nonnegative threshold required")
    score, side = signals(probability)
    y = frame.y_crossing.to_numpy()
    admitted = frame.feature_available.to_numpy(bool, copy=True)
    if eligible is not None:
        admitted &= np.asarray(eligible, bool)
    known, unknown = admitted & (y >= 0), admitted & (y < 0)
    if not np.isin(y[known], [0, 1, 2, 3]).all():
        raise ValueError("Malformed known target")
    alert = score[channel] >= threshold
    event, none = known & (y > 0), known & (y == 0)
    chosen_true = event & ((y & side) != 0)
    true_positive = alert & (event if channel == "event" else chosen_true)
    false_none = alert & none
    wrong_side = alert & event & ~chosen_true if channel == "selected_side" else np.zeros(len(y), bool)
    n_alerts, n_event, n_none = int((alert & known).sum()), int(event.sum()), int(none.sum())
    tp, fp, wrong = int(true_positive.sum()), int(false_none.sum()), int(wrong_side.sum())
    if tp+fp+wrong != n_alerts:
        raise ValueError("Alarm categories do not partition known alerts")
    return dict(channel=channel, threshold=float(threshold), known_roots=int(known.sum()),
        unknown_roots=int(unknown.sum()), known_event_roots=n_event, known_none_roots=n_none,
        alerts=n_alerts, true_positives=tp, none_false_positives=fp, wrong_side_alerts=wrong,
        false_signals=fp+wrong, unknown_alerts=int((alert & unknown).sum()),
        precision=ratio(tp, n_alerts), event_opportunity_recall=ratio(tp, n_event),
        none_fpr=ratio(fp, n_none), alerts_per_1000_known=ratio(n_alerts*1000, int(known.sum())),
        wrong_side_fraction_of_alerts=ratio(wrong, n_alerts), wrong_side_rate_on_event_roots=ratio(wrong, n_event),
        selected_side_true_roots=int(chosen_true.sum()) if channel == "selected_side" else None,
        conditional_selected_side_recall=ratio(tp, int(chosen_true.sum())) if channel == "selected_side" else None,
        positive_identity=identity_coverage(frame, known, alert, side, channel))


def empirical_none_threshold(scores, none, maximum_fpr=MAX_NONE_FPR):
    scores, none = np.asarray(scores, np.float64), np.asarray(none, bool)
    if scores.ndim != 1 or scores.shape != none.shape or not np.isfinite(scores).all() or not 0 <= maximum_fpr < 1:
        raise ValueError("Finite scores and an aligned negative mask required")
    selected = np.sort(scores[none])[::-1]
    if not len(selected):
        return dict(threshold=1.01, known_none_roots=0, allowed_none_alerts=0,
                    actual_none_alerts=0, calibration_none_fpr=None, reason="insufficient_negative_denominator",
                    budget_count=0, actual_calibration_fpr=None, source_cohort_sha256=array_sha(np.flatnonzero(none)))
    budget = int(np.floor(maximum_fpr*len(selected)))
    boundary = float(selected[budget])
    threshold = float(np.nextafter(np.float64(boundary), np.float64(np.inf)))
    count = int((selected >= threshold).sum())
    if count > budget:
        raise ValueError("Conservative inclusive negative budget violated")
    return dict(threshold=threshold, known_none_roots=len(selected), allowed_none_alerts=budget,
        actual_none_alerts=count, calibration_none_fpr=count/len(selected), boundary_score=boundary,
        ties_at_boundary=int((selected == boundary).sum()), unused_budget_due_to_ties=budget-count,
        calibration_negative_indices_sha256=array_sha(np.flatnonzero(none)),
        budget_count=budget, actual_calibration_fpr=count/len(selected), source_cohort_sha256=array_sha(np.flatnonzero(none)),
        reason="lowest_float64_threshold_with_inclusive_negative_budget")


def false_signal_audit(frame, probability, splits):
    score, _ = signals(probability)
    admitted = frame.feature_available.to_numpy(bool)
    known = admitted & (frame.y_crossing.to_numpy() >= 0)
    by_split = {}
    for role, dates in splits.items():
        mask = frame.date.isin(dates).to_numpy()
        by_split[role] = dict(coverage=dict(all_roots=int(mask.sum()), admitted_roots=int((mask & admitted).sum()),
            known_admitted_roots=int((mask & known).sum()), unknown_admitted_roots=int((mask & admitted & ~known).sum()),
            unavailable_roots=int((mask & ~admitted).sum()), known_fraction_of_admitted=ratio(int((mask & known).sum()), int((mask & admitted).sum()))),
            event_fixed=[alarm_metrics(frame, probability, threshold, "event", mask) for threshold in FIXED_THRESHOLDS],
            selected_side_fixed=[alarm_metrics(frame, probability, threshold, "selected_side", mask) for threshold in FIXED_THRESHOLDS])
    calibration = frame.date.isin(splits["calibration"]).to_numpy()
    test = frame.date.isin(splits["test"]).to_numpy()
    negative = calibration & admitted & frame.y_crossing.eq(0).to_numpy()
    selected = {}
    for channel in ("event", "selected_side"):
        rule = empirical_none_threshold(score[channel], negative)
        selected[channel] = dict(selection=rule,
            calibration=alarm_metrics(frame, probability, rule["threshold"], channel, calibration),
            test=alarm_metrics(frame, probability, rule["threshold"], channel, test))
    return dict(protocol=PROTOCOL, by_split=by_split, calibrated_none_fpr=selected)


def roughness(frame, probability, splits, cadence_seconds=5):
    score, side = signals(probability)
    y = frame.y_crossing.to_numpy()
    known = frame.feature_available.to_numpy(bool) & (y >= 0)
    support = frame.physical_support_available.to_numpy(bool)
    token, t, date = frame.physical_segment_token.to_numpy(), frame.t_ns.to_numpy(), frame.date.to_numpy()
    adjacent = np.r_[False, known[1:] & known[:-1] & support[1:] & support[:-1] &
        (token[1:] == token[:-1]) & (date[1:] == date[:-1]) & (np.diff(t) == int(cadence_seconds*1e9))]
    out = {}
    for role, dates in splits.items():
        mask = adjacent & frame.date.isin(dates).to_numpy()
        indices = np.flatnonzero(mask)
        def change(values):
            return float(np.abs(np.asarray(values, float)[indices]-np.asarray(values, float)[indices-1]).mean()) if len(indices) else None
        out[role] = dict(adjacent_pairs=len(indices), current_root_indices_sha256=array_sha(indices),
            event_probability_total_variation_per_step=change(score["event"]),
            selected_side_probability_total_variation_per_step=change(score["selected_side"]),
            selected_direction_switch_fraction=float((side[indices] != side[indices-1]).mean()) if len(indices) else None,
            target_event_flip_fraction=change(y > 0), target_long_flip_fraction=change(np.isin(y, [1, 3])),
            target_short_flip_fraction=change(np.isin(y, [2, 3])), cadence_seconds=cadence_seconds,
            definition="Same physical-support token/date, regular cadence, both endpoints admitted and known. Common pair cohort across all arms; unknown roots break scored adjacency, never become negative labels.")
    return out


def soft_weights(frame, train_mask, scheme, alpha=.5, lower=.5, upper=2.):
    if scheme not in WEIGHTINGS or not (0 < alpha <= 1 and 0 < lower < 1 < upper):
        raise ValueError("Unsupported soft identity weight recipe")
    selected = np.asarray(train_mask, bool) & frame.feature_available.to_numpy(bool) & (frame.y_crossing.to_numpy() >= 0)
    if not selected.any():
        raise ValueError("Nonempty admitted known TRAIN required")
    rows, _, ids = side_memberships(frame, selected)
    counts = Counter(ids)
    raw = np.ones(len(frame), float)
    if scheme == "soft_crossing_id":
        sums = np.bincount(rows, weights=[counts[key]**(-alpha) for key in ids], minlength=len(frame))
        memberships = np.bincount(rows, minlength=len(frame))
        raw[memberships > 0] = sums[memberships > 0]/memberships[memberships > 0]
    y, full, classes = frame.y_crossing.to_numpy(), np.zeros(len(frame), float), {}
    for value, name in enumerate(("NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH")):
        mask = selected & (y == value)
        r = raw[mask]
        if not len(r):
            classes[name] = dict(roots=0)
            continue
        if value == 0 or scheme == "uniform":
            weights, scale = np.ones(len(r)), 1.
        else:
            # Independent monotone root solve, deliberately different iteration
            # count from training; compare numerical weights, not bitwise solver history.
            left, right = 0., upper/float(r.min())
            for _ in range(120):
                middle = (left+right)/2
                if np.clip(middle*r, lower, upper).sum() < len(r):
                    left = middle
                else:
                    right = middle
            scale = (left+right)/2
            weights = np.clip(scale*r, lower, upper)
        full[mask] = weights
        if np.any(weights < lower) or np.any(weights > upper) or not np.isclose(weights.sum(), len(r), rtol=0, atol=1e-8):
            raise ValueError("Clipped within-class mass not preserved")
        classes[name] = dict(roots=len(r), raw=weight_stats(r), normalized=weight_stats(weights),
            scale=scale, lower_bound_count=int((weights == lower).sum()), upper_bound_count=int((weights == upper).sum()),
            mass_residual=float(weights.sum()-len(r)))
    count_json = json.dumps(sorted(counts.items()), separators=(",", ":"))
    import hashlib
    return full, dict(scheme=scheme, alpha=alpha, minimum=lower, maximum=upper,
        train_root_indices_sha256=array_sha(np.flatnonzero(selected)),
        count_table_sha256=hashlib.sha256(count_json.encode()).hexdigest(),
        positive_side_memberships=len(ids), unique_raw_ids=len(counts), by_joint_class=classes,
        overall=weight_stats(full[selected]), definition="TRAIN-only pooled side-membership counts; inverse square-root, BOTH mean; solve scale inside clip to preserve each class mass; NONE remains one")


def paired_effects(day_rows, replications=20000):
    indexed = {(r["split"], r["date"], int(r["seed"]), r["weighting"]): r["metrics"] for r in day_rows}
    if len(indexed) != len(day_rows):
        raise ValueError("Duplicate comparison cells")
    output = {}
    for role in sorted({r["split"] for r in day_rows}):
        dates = sorted({r["date"] for r in day_rows if r["split"] == role})
        metric_names = sorted(set.intersection(*[set(r["metrics"]) for r in day_rows if r["split"] == role]))
        for date in dates:
            if any((role, date, seed, scheme) not in indexed for seed in SEEDS for scheme in WEIGHTINGS):
                raise ValueError("Missing matched date/seed/weight cell")
        output[role] = {}
        for metric in metric_names:
            valid = [d for d in dates if all(indexed[role, d, s, w][metric] is not None and
                np.isfinite(indexed[role, d, s, w][metric]) for s in SEEDS for w in WEIGHTINGS)]
            if not valid:
                output[role][metric] = dict(paired_dates=0)
                continue
            effects = {seed: np.array([indexed[role, date, seed, "soft_crossing_id"][metric]-
                indexed[role, date, seed, "uniform"][metric] for date in valid])*(1 if metric in {"event_auc", "event_average_precision", "selected_side_average_precision"} else -1)
                for seed in SEEDS}
            output[role][metric] = dict(paired_dates=len(valid),
                by_seed={str(s): paired_interval(effects[s], valid, replications) for s in SEEDS},
                mean_seed_effect_paired_by_date=paired_interval(np.mean(list(effects.values()), axis=0), valid, replications))
    return dict(by_split=output, sign="Soft minus uniform utility; +AUC and -NLL/Brier. Positive improves the metric.",
        interval="Descriptive equal-day paired bootstrap, 20,000 replicates seed20261023; average seeds within a date first. No independence, multiplicity correction, or TEST selection claim.")


def validate_recipe(config):
    if config.get("schema") != "scientific_opportunity_pipeline_v4" or config["data"]["context_mode"] != "restored":
        raise ValueError("V4 restored context required")
    if config["training"]["id_weighting"] != dict(alpha=.5, min_weight=.5, max_weight=2.):
        raise ValueError("Weight parameters differ from preregistered recipe")
    if config["experiment"]["seeds"] != list(SEEDS) or config["experiment"]["weightings"] != list(WEIGHTINGS):
        raise ValueError("Four fixed comparison arms required")
    protocol = config["experiment"]["false_signal_protocol"]
    calibration = protocol["calibration_none_fpr"]
    if protocol["fixed_thresholds"] != list(FIXED_THRESHOLDS) or protocol["side_tie_break"] != "LONG" or calibration["budget"] != MAX_NONE_FPR or calibration["score_dtype"] != "float64":
        raise ValueError("False-signal preregistration differs from audit protocol")
    expected = [day for role in ("train", "validation", "calibration", "test") for day in config["splits"][role]]
    if expected != sorted(set(expected)) or expected != config["data"]["dates"]:
        raise ValueError("Disjoint chronological complete split dates required")


def load_shared_dataset(config, suite, sources):
    directory = Path(suite["dataset_root"])
    path = directory/"manifest.json"
    manifest = read(path)
    digest = sha(path)
    sources[str(path)] = digest
    if manifest.get("status") != "COMPLETE" or digest != suite["dataset_manifest_sha256"]:
        raise ValueError("Dataset is incomplete or changed")
    if manifest["splits"] != config["splits"] or manifest["dates"] != config["data"]["dates"] or directory != Path(config["data"]["root"]):
        raise ValueError("Configured shared data/cohorts differ")
    frames = []
    for day in manifest["days"]:
        if day["status"] != "COMPLETE":
            raise ValueError("Incomplete dataset day")
        for name in (day["roots_file"], day["x_file"]):
            actual = sha(directory/name)
            if actual != day["files"][name]:
                raise ValueError("Shared dataset payload changed")
            sources[str(directory/name)] = actual
        frames.append(pd.read_parquet(directory/day["roots_file"]))
    frame = pd.concat(frames, ignore_index=True)
    if np.any(np.diff(frame.t_ns.to_numpy()) <= 0):
        raise ValueError("Unique chronological roots required")
    return frame, manifest, digest


def uniform_v3_replay(directory, prediction, summary, history, seed, manifest_sha, sources):
    previous = directory/f"tcn_micro_restored_uniform_seed{seed}"
    if not (previous/"receipt.json").exists():
        return dict(status="UNAVAILABLE", reason="No previous uniform run receipt")
    receipt = read(previous/"receipt.json")
    if receipt.get("status") != "COMPLETE" or receipt["dataset_manifest_sha256"] != manifest_sha:
        return dict(status="NOT_COMPARABLE", reason="Previous run incomplete or bound to different data")
    for name in ("predictions.parquet", "summary.json", "history.json", "config.json"):
        digest = sha(previous/name)
        if digest != receipt["outputs"][name]:
            raise ValueError("Previous immutable uniform payload changed")
        sources[str(previous/name)] = digest
    old = pd.read_parquet(previous/"predictions.parquet", columns=["root_index", "p_none", "p_long", "p_short", "p_both"])
    columns = ["p_none", "p_long", "p_short", "p_both"]
    if not np.array_equal(old.root_index, prediction.root_index):
        raise ValueError("Previous uniform roots differ")
    exact = np.array_equal(old[columns].to_numpy(), prediction[columns].to_numpy())
    old_summary, old_history = read(previous/"summary.json"), read(previous/"history.json")
    same_keys = ("initial_state_sha256", "train_metric_probe_sha256", "gradient_probe_sha256", "best_update")
    same = {key: old_summary["training"][key] == summary["training"][key] for key in same_keys}
    prefixes = {int(r["update"]): r["sampled_root_stream_sha256"] for r in history}
    prior_prefixes = {int(r["update"]): r["sampled_root_stream_sha256"] for r in old_history}
    common = sorted(prefixes.keys() & prior_prefixes.keys())
    same["sampled_root_prefixes"] = all(prefixes[u] == prior_prefixes[u] for u in common)
    return dict(status="EXACT_REPLAY" if exact and all(same.values()) else "DIFFERS", previous_path=str(previous),
        predictions_bitwise_equal=exact, maximum_probability_difference=float(np.abs(old[columns].to_numpy()-prediction[columns].to_numpy()).max()),
        training_checks=same, compared_sampling_updates=common,
        definition="Fresh V4 uniform fit compared with immutable V3 restored-uniform predictions, selected update, initialization, probes, and common sampled TRAIN prefixes; no model reused")


def audit_run(directory, config, frame, manifest_sha, sources, expected_weights, v3_runs):
    receipt = read(directory/"receipt.json")
    if receipt.get("status") != "COMPLETE" or receipt["dataset_manifest_sha256"] != manifest_sha:
        raise ValueError("Incomplete run or changed dataset binding")
    for name, expected in receipt["outputs"].items():
        digest = sha(directory/name)
        if digest != expected:
            raise ValueError("Changed run output: "+str(directory/name))
        sources[str(directory/name)] = digest
    sources[str(directory/"receipt.json")] = sha(directory/"receipt.json")
    for path, expected in receipt["sources"].items():
        if sources.get(path) != expected:
            digest = sha(path)
            if digest != expected:
                raise ValueError("Bound fitting source changed: "+path)
            sources[path] = digest
    saved = read(directory/"config.json")
    reference = json.loads(json.dumps(config))
    normalized = json.loads(json.dumps(saved))
    for value in (reference, normalized):
        value.pop("run_id", None); value.pop("seed", None)
        value["training"].pop("weighting", None)
    if normalized != reference:
        raise ValueError("Unplanned configuration change")
    seed, scheme = int(saved["seed"]), saved["training"]["weighting"]
    expected_id = f"tcn_micro_restored_{scheme}_seed{seed}"
    if saved["run_id"] != expected_id or directory.name != expected_id:
        raise ValueError("Run identity/configuration mismatch")
    summary, history = read(directory/"summary.json"), read(directory/"history.json")
    prediction = pd.read_parquet(directory/"predictions.parquet")
    if not np.array_equal(prediction.root_index, np.arange(len(frame))):
        raise ValueError("Prediction root order differs")
    for name in ("date", "t_ns", "y_crossing", "feature_available", "physical_support_available", "physical_segment_token", "legacy_segment_token"):
        if not np.array_equal(prediction[name], frame[name]):
            raise ValueError("Prediction data/target/support differs: "+name)
    p = prediction[["p_none", "p_long", "p_short", "p_both"]].to_numpy(float)
    signals(p)
    known = frame.feature_available.to_numpy(bool) & (frame.y_crossing.to_numpy() >= 0)
    train = known & frame.date.isin(config["splits"]["train"]).to_numpy()
    weights, independent = expected_weights[scheme]
    saved_weights, weight_receipt = pd.read_parquet(directory/"train_weights.parquet"), read(directory/"weighting.json")
    if not np.array_equal(saved_weights.root_index, np.flatnonzero(train)) or not np.allclose(saved_weights.weight, weights[train], rtol=0, atol=1e-11):
        raise ValueError("Independent soft weights differ")
    for key, expected in (("train_root_indices_sha256", independent["train_root_indices_sha256"]),
        ("crossing_id_count_table_sha256", independent["count_table_sha256"]),
        ("train_weights_sha256", array_sha(saved_weights.weight.to_numpy(), "<f8"))):
        if weight_receipt[key] != expected:
            raise ValueError("Weight artifact identity mismatch")
    gradient_check = verify_saved_gradients(prediction, p, weights, train)
    gradient = read(directory/"gradient_artifact.json")
    economics = read(directory/"economic_evaluation.json")
    y = frame.y_crossing.to_numpy()
    prior = (np.bincount(y[train], minlength=4)+.5)/(train.sum()+2.)
    pooled, by_day, day_rows = {}, [], []
    metric_names = ("joint_nll", "event_nll", "event_auc", "direction_nll_given_event", "event_brier",
        "none_nll", "positive_event_nll", "event_average_precision", "selected_side_average_precision")
    for role, dates in config["splits"].items():
        mask = known & frame.date.isin(dates).to_numpy()
        model = root_metrics(y[mask], p[mask])
        for key in metric_names[:5]:
            a, b = model[key], summary["evaluation"][role]["model"][key]
            if not (a is None and b is None) and not np.isclose(a, b, rtol=0, atol=1e-10):
                raise ValueError("Independent model scores differ")
        pooled[role] = dict(model=model, train_prior=root_metrics(y[mask], np.tile(prior, (mask.sum(), 1))),
            event_balanced_positive=event_balanced_metrics(frame.loc[mask].reset_index(drop=True), p[mask]))
        for date in dates:
            daily = known & frame.date.eq(date).to_numpy()
            metrics = root_metrics(y[daily], p[daily])
            ids = event_balanced_metrics(frame.loc[daily].reset_index(drop=True), p[daily])
            by_day.append(dict(date=date, split=role, model=metrics, event_balanced_positive=ids))
            flatten = {key: metrics[key] for key in metric_names}
            flatten.update({"event_id_"+key: value for key, value in ids["metrics"].items()})
            day_rows.append(dict(date=date, split=role, seed=seed, weighting=scheme, metrics=flatten))
    training = {key: value for key, value in summary["training"].items()
                if key not in {"gradients_before", "gradients_selected", "train_metric_probe_indices"}}
    replay = uniform_v3_replay(v3_runs, prediction, summary, history, seed, manifest_sha, sources) if scheme == "uniform" else None
    result = dict(run_id=expected_id, seed=seed, weighting=scheme, pooled=pooled, by_day=by_day,
        training=training, weighting_artifact=weight_receipt, uniform_v3_replay=replay,
        false_signals=false_signal_audit(frame, p, config["splits"]), roughness=roughness(frame, p, config["splits"], config["data"]["cadence_seconds"]),
        gradients=dict(independent_check=gradient_check, before=gradient["before"], selected=gradient["selected"],
            unweighted_full_clock_groups=gradient["full_clock_groups"], weighted_train_groups=gradient["weighted_train_groups"]),
        gradient_clip_fraction_by_validation_interval=[r["gradient_clip_fraction"] for r in history if r.get("gradient_clip_fraction") is not None],
        economic_policy=economics["summary"], calibration_grid=economics["calibration_grid"],
        scaler_sha256=summary["scaler_sha256"], cohorts=summary["cohorts"], source_identity=receipt["sources"],
        sampling_prefixes={str(r["update"]): r["sampled_root_stream_sha256"] for r in history})
    return result, day_rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/research/opportunity_pipeline_v4.json")
    parser.add_argument("--suite", default="reports/scientific_pipeline_v4/suite_summary.json")
    parser.add_argument("--v3-runs", default="data/derived/scientific_pipeline_v3/runs")
    parser.add_argument("--output", default="reports/scientific_pipeline_v4/statistical_audit.json")
    args = parser.parse_args()
    if not Path(args.suite).exists() or read(args.suite).get("status") != "COMPLETE":
        raise SystemExit("Four-fit suite is not COMPLETE; no fitted-result audit performed")
    config, suite = read(args.config), read(args.suite)
    validate_recipe(config)
    expected = {f"tcn_micro_restored_{weight}_seed{seed}" for seed in SEEDS for weight in WEIGHTINGS}
    if len(suite["runs"]) != 4 or {r["run_id"] for r in suite["runs"]} != expected:
        raise ValueError("Exactly four prespecified runs required")
    if suite["config_sha256"] != sha(args.config):
        raise ValueError("Suite configuration changed")
    sources = {str(path): sha(path) for path in (args.config, args.suite, Path(__file__), Path("scripts/audit_opportunity_v3.py"))}
    frame, manifest, manifest_sha = load_shared_dataset(config, suite, sources)
    train = frame.date.isin(config["splits"]["train"]).to_numpy()
    weights = {scheme: soft_weights(frame, train, scheme) for scheme in WEIGHTINGS}
    identities = identity_audit(frame, config["splits"])
    if not identities["no_cross_split_ids"]:
        raise ValueError("Crossing IDs cross split boundaries")
    old_path = Path("configs/research/opportunity_pipeline_v3.json")
    old = read(old_path); sources[str(old_path)] = sha(old_path)
    unchanged = {name: old[name] == config[name] for name in ("data", "splits", "event_query", "execution", "risk", "model", "loss", "policy")}
    # V3 base configuration used legacy; V4 fixes its already-tested restored arm.
    old_data = dict(old["data"]); old_data["context_mode"] = "restored"
    unchanged["data"] = old_data == config["data"]
    if not all(unchanged.values()):
        raise ValueError("Source/features/target/loss/execution recipe changed")
    runs, days = {}, []
    for run_id in sorted(expected):
        run, rows = audit_run(Path(suite["runs_root"])/run_id, config, frame, manifest_sha, sources, weights, Path(args.v3_runs))
        runs[run_id] = run; days.extend(rows)
    values = list(runs.values())
    for key in ("scaler_sha256", "cohorts", "source_identity"):
        if any(value[key] != values[0][key] for value in values[1:]):
            raise ValueError("Unmatched comparison arms: "+key)
    for seed in SEEDS:
        pair = [value for value in values if value["seed"] == seed]
        for key in ("initial_state_sha256", "train_metric_probe_sha256", "gradient_probe_sha256"):
            if pair[0]["training"][key] != pair[1]["training"][key]:
                raise ValueError("Unmatched initialization or probe")
        common = set(pair[0]["sampling_prefixes"]) & set(pair[1]["sampling_prefixes"])
        if any(pair[0]["sampling_prefixes"][u] != pair[1]["sampling_prefixes"][u] for u in common):
            raise ValueError("Unmatched TRAIN sample order")
    for role in config["splits"]:
        if len({r["roughness"][role]["current_root_indices_sha256"] for r in values}) != 1:
            raise ValueError("Unmatched physical adjacency cohorts")
    write_audit(args.output, dict(status="COMPLETE", source_hashes=sources, protocol=PROTOCOL,
        fixed_recipe_checks=unchanged, shared_feature_names=manifest["groups"]["micro"],
        train_weight_audit={name: receipt for name, (_, receipt) in weights.items()}, identity_overlap=identities,
        matched_scaler_cohorts_initialization_probes_sampling_and_physical_pairs=True,
        runs=runs, paired=paired_effects(days),
        limitations=["Fixed alarms and CAL NONE-FPR thresholds are geometric classification diagnostics, not trading policies.",
            "Known-label cohorts are legitimate scoring cohorts but cannot prefilter live/full-clock execution; UNKNOWN coverage is separate.",
            "Positive-ID diagnostics omit negative roots and cannot establish false-positive control or economic value.",
            "Weight-concentration ESS and crossing multiplicity are not independent sample sizes; day intervals are descriptive.",
            "June/July dates had historical research exposure and V3 evaluation; V4 is a controlled same-block follow-up, not a fresh external holdout."] ))


if __name__ == "__main__":
    main()
