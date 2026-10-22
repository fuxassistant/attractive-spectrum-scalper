"""Descriptive paired day audit after the immutable v2 model suite is COMPLETE.

No model, checkpoint, threshold, or risk selection occurs here. The fixed
bootstrap samples whole TEST days, keeps each model/prior comparison paired,
and weights days equally. Ten adjacent market days do not establish a robust
population confidence interval; the interval is a descriptive sensitivity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum.research.training import metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, q10=None, q50=None, q90=None)
    q = np.quantile(values, [.1, .5, .9])
    return dict(n=len(values), mean=float(values.mean()), q10=float(q[0]), q50=float(q[1]), q90=float(q[2]))


def bootstrap_days(delta, seed=20261022, replications=20000):
    delta = np.asarray(delta, float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Finite nonempty paired day deltas required")
    sampled = np.random.default_rng(seed).integers(0, len(delta), size=(replications, len(delta)))
    means = delta[sampled].mean(1)
    ci = np.quantile(means, [.025, .975])
    return dict(unit="whole day", paired=True, weighting="equal days", days=len(delta),
        seed=seed, replications=replications, mean_improvement=float(delta.mean()),
        percentile_95_interval=[float(ci[0]), float(ci[1])], positive_days=int((delta > 0).sum()),
        bootstrap_fraction_positive=float((means > 0).mean()),
        inference="descriptive only; adjacent days may be dependent; only ten TEST days; no correction for historical exposure or model-selection uncertainty")


def evaluated(y, p):
    out = metrics(y, p)
    positive = y > 0
    out["direction_accuracy_given_event"] = (float((p[positive, 1:].argmax(1)+1 == y[positive]).mean())
                                              if positive.any() else None)
    out["both_fraction_known"] = float((y == 3).mean()) if len(y) else None
    out["both_fraction_events"] = float((y[positive] == 3).mean()) if positive.any() else None
    return out


def calibration_verdict(economics, minimum):
    rows = []
    for candidate in economics["calibration_grid"]:
        if candidate["threshold"] > 1:
            continue
        if not candidate["completed"]:
            verdict = "censored_incomplete"
        elif candidate["trades"] < minimum:
            verdict = "complete_too_few_trades"
        elif candidate["net"] <= 0:
            verdict = "complete_sufficient_trades_nonpositive"
        else:
            verdict = "eligible_complete_positive_sufficient_trades"
        rows.append(dict(plan=candidate["plan"], threshold=candidate["threshold"],
            completed=candidate["completed"], closed_cash=candidate["net"], trades=candidate["trades"],
            censored=candidate["censored"], halt=candidate["halt"], verdict=verdict,
            cash_scope="whole calibration projection" if candidate["completed"] else "partial prefix before unknown outcome"))
    return dict(minimum_trades=minimum, nonwait_candidates=len(rows),
        completed=sum(r["completed"] for r in rows), censored=sum(not r["completed"] for r in rows),
        positive_closed_cash_including_partial=sum(r["closed_cash"] > 0 for r in rows),
        positive_complete=sum(r["completed"] and r["closed_cash"] > 0 for r in rows),
        minimum_trade_count_met_including_partial=sum(r["trades"] >= minimum for r in rows),
        eligible=sum(r["verdict"] == "eligible_complete_positive_sufficient_trades" for r in rows),
        mutually_exclusive_verdict_counts={name: sum(r["verdict"] == name for r in rows)
            for name in ("censored_incomplete", "complete_too_few_trades", "complete_sufficient_trades_nonpositive", "eligible_complete_positive_sufficient_trades")},
        candidates=rows,
        interpretation="WAIT can result from incomplete support or insufficient sample size. A censored prefix cash value is not a full-period profit/loss verdict; no future outcome is substituted with zero.")


def derivative_check(prediction, p, config):
    if config["loss"]["formulation"] != "factorized" or config["loss"]["event_loss"] != "bce":
        return dict(status="not_applicable", reason="Analytic check is specific to the frozen BCE plus conditional CE loss")
    y = prediction.y_crossing.to_numpy()
    known = y >= 0
    expected_event = 1-p[:, 0]-(y > 0)
    q = p[:, 1:]/p[:, 1:].sum(1, keepdims=True)
    q[np.arange(len(q)), np.maximum(y-1, 0)] -= 1
    expected_direction = np.linalg.norm(q, axis=1)*(y > 0)
    e = float(np.abs(prediction.g_event.to_numpy()[known]-expected_event[known]).max())
    d = float(np.abs(prediction.g_direction_norm.to_numpy()[known]-expected_direction[known]).max())
    unknown_nan = bool(prediction.loc[~known, ["g_event", "g_direction_norm"]].isna().all().all())
    if max(e, d) > 2e-6 or not unknown_nan:
        raise ValueError("Saved per-root gradient does not match factorized BCE/CE derivative")
    return dict(status="PASS", all_known_roots=int(known.sum()),
        max_abs_event_derivative_error=e, max_abs_direction_norm_error=d,
        unknown_saved_gradients_are_nan=unknown_nan,
        definition="Recompute unreduced BCE gradient pE-yE and conditional CE softmax-minus-onehot L2 from saved probabilities; NONE direction gradient is zero. Float32 probability roundtrip tolerance 2e-6.")


def head_difficulty(prediction, p, test):
    y = prediction.y_crossing.to_numpy()
    event_probability = 1-p[:, 0]
    event_true = np.where(y > 0, event_probability, 1-event_probability)
    positive = test & (y > 0)
    direction_true = np.full(len(y), np.nan)
    ids = np.flatnonzero(positive)
    direction_true[ids] = np.clip(p[ids, y[ids]]/np.maximum(event_probability[ids], 1e-12), 0, 1)
    def groups(true_p, admitted, mass):
        mass = np.asarray(mass, float)
        total = float(mass[admitted].sum())
        masks = {"hard_true_head_probability_lt_0.4": admitted & (true_p < .4),
                 "intermediate_true_head_probability_0.4_to_0.8": admitted & (true_p >= .4) & (true_p < .8),
                 "easy_true_head_probability_ge_0.8": admitted & (true_p >= .8)}
        return {name: dict(count=int(mask.sum()), fraction_head_examples=float(mask.sum()/admitted.sum()),
            mean_true_head_probability=float(true_p[mask].mean()) if mask.any() else None,
            gradient_mass_fraction=float(mass[mask].sum()/total) if total else None)
                for name, mask in masks.items()}
    return dict(scope="TEST known feature-available roots; diagnostics only, no loss or fit changed",
        event_definition="E true probability is pE for positive Y, 1-pE for NONE; pE=1-p_none; mass is absolute saved dL/dz_event",
        direction_definition="D true probability is p_true_joint/pE only for positive Y (three-way LONG_ONLY/SHORT_ONLY/BOTH); mass is saved conditional-direction gradient L2. Tiny float32 roundtrip deviations are clipped to [0,1].",
        event=groups(event_true, test, np.abs(prediction.g_event.to_numpy())),
        direction_given_event=groups(direction_true, positive, prediction.g_direction_norm.to_numpy()),
        caution="Existing joint-probability difficulty groups remain unchanged. Joint-hard does not imply conditional-direction-hard; separate head probabilities are needed to discuss focal loss.")


def audit_run(directory, expected_dates, sources):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    summary = read(directory / "summary.json")
    if receipt["status"] != "COMPLETE" or summary["status"] != "COMPLETE":
        raise ValueError("Cannot audit an unfinished run")
    for name in ("config.json", "summary.json", "predictions.parquet", "gradient_artifact.json", "economic_evaluation.json"):
        path = directory/name
        actual = sha(path)
        if receipt["outputs"].get(name) != actual:
            raise ValueError("Consumed run payload hash mismatch: " + str(path))
        sources[str(path)] = actual
    sources[str(directory/"receipt.json")] = sha(directory/"receipt.json")
    config = read(directory/"config.json")
    if config["splits"]["test"] != expected_dates:
        raise ValueError("Run TEST dates differ from the frozen suite")
    prediction = pd.read_parquet(directory/"predictions.parquet")
    p = prediction[["p_none", "p_long", "p_short", "p_both"]].to_numpy(float)
    y = prediction.y_crossing.to_numpy()
    available = prediction.feature_available.to_numpy() & (y >= 0)
    train = available & prediction["split"].eq("train").to_numpy()
    validation = available & prediction["split"].eq("validation").to_numpy()
    prior = (np.bincount(y[train], minlength=4)+.5)/(int(train.sum())+2.)
    validation_prior = (np.bincount(y[validation], minlength=4)+.5)/(int(validation.sum())+2.)
    test = available & prediction["split"].eq("test").to_numpy()
    model, baseline = evaluated(y[test], p[test]), evaluated(y[test], np.tile(prior, (test.sum(), 1)))
    validation_baseline = evaluated(y[test], np.tile(validation_prior, (test.sum(), 1)))
    for name, value in (("model", model), ("prior", baseline)):
        saved = summary["evaluation"]["test"][name]
        for metric in ("joint_nll", "event_nll", "event_brier", "direction_nll_given_event"):
            if not np.isclose(value[metric], saved[metric], rtol=0, atol=1e-11):
                raise ValueError("Independent TEST metric differs from saved summary: " + metric)
    by_day = []
    for date in expected_dates:
        keep = test & prediction.date.eq(date).to_numpy()
        ym, pm = y[keep], p[keep]
        m, b = evaluated(ym, pm), evaluated(ym, np.tile(prior, (len(ym), 1)))
        vb = evaluated(ym, np.tile(validation_prior, (len(ym), 1)))
        by_day.append(dict(date=date, roots=len(ym), model=m, prior=b, validation_prior_posthoc=vb,
            joint_nll_improvement=b["joint_nll"]-m["joint_nll"],
            event_nll_improvement=b["event_nll"]-m["event_nll"],
            direction_nll_improvement_given_event=b["direction_nll_given_event"]-m["direction_nll_given_event"],
            joint_nll_improvement_vs_validation_prior_posthoc=vb["joint_nll"]-m["joint_nll"],
            event_nll_improvement_vs_validation_prior_posthoc=vb["event_nll"]-m["event_nll"],
            direction_nll_improvement_vs_validation_prior_posthoc=vb["direction_nll_given_event"]-m["direction_nll_given_event"],
            both_count=int((ym == 3).sum()), both_fraction_known=float((ym == 3).mean()),
            both_fraction_events=float((ym[ym > 0] == 3).mean()) if (ym > 0).any() else None))
    artifact = read(directory/"gradient_artifact.json")
    economics = read(directory/"economic_evaluation.json")
    trainprobe = summary["training"]["gradients_selected"]
    def prevalence(mask):
        values = y[mask]
        return dict(roots=len(values), state_counts=np.bincount(values, minlength=4).tolist(),
                    event_fraction=float((values > 0).mean()), both_fraction=float((values == 3).mean()))
    return dict(run_id=summary["run_id"], family=summary["feature_family"],
        train_prior=prior.tolist(), train_prior_definition="(TRAIN known feature-available state_count + 0.5)/(TRAIN roots + 2)",
        cohort_test_sha256=summary["cohorts"]["test"]["root_indices_sha256"],
        pooled_test=dict(model=model, prior=baseline,
            joint_nll_improvement=baseline["joint_nll"]-model["joint_nll"],
            event_nll_improvement=baseline["event_nll"]-model["event_nll"],
            direction_nll_improvement_given_event=baseline["direction_nll_given_event"]-model["direction_nll_given_event"]),
        equal_day_joint_nll_bootstrap=bootstrap_days([row["joint_nll_improvement"] for row in by_day]),
        validation_prior_posthoc_diagnostic=dict(prior=validation_prior.tolist(),
            fitted_from="Only known feature-available VALIDATION labels; additive 0.5 per class; no TEST fitting",
            role="Posthoc diagnostic of prevalence/intercept adaptation, not the prespecified benchmark and never used for model/checkpoint/risk selection",
            test=validation_baseline,
            pooled_joint_nll_improvement=validation_baseline["joint_nll"]-model["joint_nll"],
            pooled_event_nll_improvement=validation_baseline["event_nll"]-model["event_nll"],
            pooled_direction_nll_improvement_given_event=validation_baseline["direction_nll_given_event"]-model["direction_nll_given_event"],
            equal_day_joint_nll_bootstrap=bootstrap_days([row["joint_nll_improvement_vs_validation_prior_posthoc"] for row in by_day]),
            event_auc_note="A constant prior has event AUC 0.5 on a two-class cohort. Model AUC measures ranking and is not improved merely by shifting a constant intercept; changing conditional populations can still change AUC."),
        by_day=by_day,
        gradients=dict(full_clock_groups=artifact["full_clock_groups"],
            independent_saved_probability_derivative_check=derivative_check(prediction, p, config),
            test_head_difficulty=head_difficulty(prediction, p, test),
            selected_train_probe={k: trainprobe[k] for k in ("n", "root_indices_sha256", "event_encoder_gradient_l2", "direction_encoder_gradient_l2", "cosine_event_direction", "analytic_bce_max_abs_error")},
            mass_definition="Actual unreduced configured dL_i/dlogit: absolute event gradient and direction L2 mass. Difficulty partitions use true joint probability; event/class groups overlap those partitions.",
            stored_artifact_sha256=sources[str(directory/"gradient_artifact.json")]),
        temporal=artifact["temporal"],
        train_test_prevalence=dict(train=prevalence(train), validation=prevalence(validation), test=prevalence(test),
            event_percentage_point_change=100*float((y[test] > 0).mean()-(y[train] > 0).mean()),
            cohort="feature_available and known fixed crossing labels"),
        calibration_selection_audit=calibration_verdict(economics, config["policy"]["minimum_calibration_trades"]),
        selected_policy=economics["summary"]["selected"],
        test_policy=economics["summary"]["test"],
        test_policy_caution="A censored projection is partial; future zero daily placeholders are unprocessed, not observed zero returns")


def risk_weeks(dataset, dates, config, sources):
    roots, executions = [], []
    for date in dates:
        for suffix, target in (("roots", roots), ("execution", executions)):
            path = Path(dataset)/(date+"_"+suffix+".parquet")
            sources[str(path)] = sha(path)
            target.append(pd.read_parquet(path))
    frame = pd.concat(roots, ignore_index=True)
    execution = pd.concat(executions, ignore_index=True)
    t, y = frame.t_ns.to_numpy(), frame.y_crossing.to_numpy()
    common = frame.feature_available.to_numpy() & (y >= 0)
    plans = {}
    for multiple in config["risk"]["cost_multipliers"]:
        name = f"cost{multiple:g}"
        rows = execution.loc[execution.plan == name].sort_values(["t_ns", "side"], ascending=[True, False])
        if not np.array_equal(rows.t_ns.to_numpy(), np.repeat(t, 2)) or not np.array_equal(rows.side.to_numpy(), np.tile([1, -1], len(t))):
            raise ValueError("TEST risk execution alignment mismatch")
        plans[name] = {col: rows[col].to_numpy().reshape(len(frame), 2)
                       for col in ("status", "reason", "net", "entry_ns", "exit_ns", "R_trade", "G_event")}
        common &= (plans[name]["status"] == "COMPLETE").all(1)
    blocks = {"test_all": dates, "test_week_1": dates[:5], "test_week_2": dates[5:]}
    out = {}
    for block, block_dates in blocks.items():
        mask = common & frame.date.isin(block_dates).to_numpy()
        by_plan = {}
        for name, a in plans.items():
            net = a["net"][mask]
            retention = {}
            for si, side in enumerate(("LONG", "SHORT")):
                selected = mask & ((y & (1 << si)) != 0)
                v = a["net"][selected, si]
                retention[side] = dict(roots=len(v), net=stats(v),
                    positive_fraction=float((v > 0).mean()) if len(v) else None,
                    fixed_gain_retention_fraction=float((v >= a["G_event"][selected, si]).mean()) if len(v) else None,
                    take_fraction=float((a["reason"][selected, si] == "TAKE").mean()) if len(v) else None)
            by_plan[name] = dict(side_trials=int(mask.sum()*2), net_all_sides=stats(net),
                holding_seconds=stats((a["exit_ns"][mask]-a["entry_ns"][mask])/1e9),
                R_trade=stats(a["R_trade"][mask]),
                cost_R_trade=stats((frame.decision_spread.to_numpy()[mask, None]+config["execution"]["fee_roundtrip"])/a["R_trade"][mask]),
                reasons={reason: int((a["reason"][mask] == reason).sum()) for reason in ("TAKE", "STOP", "TIME")},
                conditional_true_crossing=retention)
        out[block] = dict(dates=block_dates, common_roots=int(mask.sum()),
            common_root_clock_sha256=hashlib.sha256(np.asarray(t[mask], dtype="<i8").tobytes()).hexdigest(), plans=by_plan)
    return dict(kind="oracle paired independent trial diagnostic, not account/policy/equity",
        cohort="feature_available AND known fixed crossing AND both sides COMPLETE in all plans; identical roots for cost2/4/6",
        event="Fixed G_event; only stop R_trade changes. Positive retention means actual exit net > 0 conditional on true LONG/SHORT crossing label.",
        units="USD per reference ounce; decision-cost ratio is a proxy based on origin spread, not entry spread", blocks=out)


def crossing_multiplicity(dataset, config, sources):
    columns = ["date", "t_ns", "feature_available", "y_crossing",
               "first_qualifying_event_id_L", "first_qualifying_event_id_S"]
    by_split = {}
    for split, dates in config["splits"].items():
        frames, banks = [], []
        for date in dates:
            path = Path(dataset)/(date+"_roots.parquet")
            sources[str(path)] = sha(path)
            frames.append(pd.read_parquet(path, columns=columns))
            bank_path = Path(dataset)/(date+"_bank.parquet")
            sources[str(bank_path)] = sha(bank_path)
            banks.append(pd.read_parquet(bank_path, columns=["event_id", "scale", "role"]))
        frame = pd.concat(frames, ignore_index=True)
        bank = pd.concat(banks, ignore_index=True).drop_duplicates()
        if bank.event_id.duplicated().any():
            raise ValueError("A saved crossing event ID maps to conflicting scale/line identities")
        bank = bank.set_index("event_id")
        y = frame.y_crossing.to_numpy()
        admitted = frame.feature_available.to_numpy() & (y >= 0)
        sides = {}
        pooled = []
        for si, side in enumerate(("L", "S")):
            mask = admitted & ((y & (1 << si)) != 0)
            ids = frame.loc[mask, "first_qualifying_event_id_"+side]
            if ids.isna().any() or ids.eq("").any():
                raise ValueError("Positive fixed crossing label has no saved event identity")
            pooled.extend(ids.astype(str).tolist())
            sides[side] = ids.astype(str).value_counts()
        def distribution(counts):
            n = int(counts.sum())
            located = bank.reindex(counts.index)
            if located.isna().any().any():
                raise ValueError("Positive saved first qualifying ID is absent from the saved crossing bank")
            scale_line = []
            for scale in config["event_query"]["scales"]:
                for role, name in enumerate(("lower", "middle", "upper")):
                    selected = located.scale.eq(scale) & located.role.eq(role)
                    root_sides = int(counts[selected.to_numpy()].sum())
                    unique = int(selected.sum())
                    scale_line.append(dict(scale=int(scale), line=name, role=role,
                        positive_root_side_labels=root_sides, unique_saved_event_ids=unique,
                        fraction_positive_labels=root_sides/n if n else None,
                        fraction_unique_ids=unique/len(counts) if len(counts) else None))
            return dict(positive_root_side_labels=n, unique_saved_event_ids=len(counts),
                positive_labels_per_id_median=float(counts.median()) if len(counts) else None,
                positive_labels_per_id_p90=float(counts.quantile(.9)) if len(counts) else None,
                positive_labels_per_id_max=int(counts.max()) if len(counts) else None,
                largest_single_id_fraction_of_positive_labels=float(counts.max()/n) if n else None,
                scale_line_distribution=scale_line)
        by_split[split] = dict(dates=dates, admitted_known_roots=int(admitted.sum()),
            pooled_sides=distribution(pd.Series(pooled, dtype=str).value_counts()),
            by_side={side: distribution(counts) for side, counts in sides.items()})
    return dict(definition="On feature-available known roots, count each positive LONG and SHORT label against its saved first_qualifying_event_id. BOTH contributes one root-side label to each side. Per-ID quantiles weight distinct saved IDs equally; top-1 share divides the largest ID multiplicity by positive root-side labels. Scale/line is joined by exact ID to saved crossing banks: role 0 lower, 1 middle, 2 upper; it describes the first qualifying witness, not every future crossed line.",
        training_usage="The frozen baseline samples uniform roots. Saved event IDs are not used for loss weighting, cooldown, or sampling.",
        witness_tie_rule="The saved target chooses the smallest qualifying publication ordinal; same-ordinal ties follow ascending scale then line role. Witness-scale shares are not an ablation of larger-scale information.",
        limitation="Saved IDs describe repeated assignment to a crossing. Unique ID counts are not effective sample sizes or counts of statistically independent events; different crossings may share market paths.",
        by_split=by_split)


def actual_context_history(dataset, config, sources, receptive_field):
    frames = []
    for date in config["data"]["dates"]:
        path = Path(dataset)/(date+"_roots.parquet")
        sources[str(path)] = sha(path)
        frames.append(pd.read_parquet(path, columns=["date", "t_ns", "feature_available", "segment_token", "y_crossing"]))
    frame = pd.concat(frames, ignore_index=True)
    t, dates = frame.t_ns.to_numpy(), frame.date.to_numpy()
    available, token = frame.feature_available.to_numpy(), frame.segment_token.to_numpy()
    step = int(round(config["data"]["cadence_seconds"]*1e9))
    causes = dict(date_change=np.r_[True, dates[1:] != dates[:-1]],
        origin_clock_gap=np.r_[False, (np.diff(t) != step) & (dates[1:] == dates[:-1])],
        segment_token_change=np.r_[False, token[1:] != token[:-1]],
        unavailable_current_or_previous=np.r_[False, ~available[1:] | ~available[:-1]])
    reset = np.logical_or.reduce(list(causes.values()))
    start = np.maximum.accumulate(np.where(reset, np.arange(len(frame)), 0))
    real = np.arange(len(frame))-start+1
    supplied = int(config["data"]["sequence_length"])
    cap = min(supplied, int(receptive_field))
    effective = np.minimum(real, cap)
    elapsed = (effective-1)*config["data"]["cadence_seconds"]
    known = available & (frame.y_crossing.to_numpy() >= 0)
    by_split = {}
    for role, role_dates in {"all": config["data"]["dates"], **config["splits"]}.items():
        day = np.isin(dates, role_dates)
        selected = day & known
        by_split[role] = dict(known_feature_roots=int(selected.sum()),
            effective_tokens=stats(effective[selected]), detailed_history_seconds=stats(elapsed[selected]),
            full_receptive_field_fraction=float((real[selected] >= cap).mean()),
            full_supplied_window_fraction=float((real[selected] >= supplied).mean()),
            effective_token_histogram=np.bincount(effective[selected], minlength=cap+1)[1:].tolist(),
            sequence_resets_on_known_roots=int((selected & reset).sum()),
            reset_causes_on_all_role_roots={name: int((day & mask).sum()) for name, mask in causes.items()},
            token_changes_between_two_available_regular_same_day_roots=int((day & available &
                np.r_[False, available[:-1]] & causes["segment_token_change"] &
                ~causes["date_change"] & ~causes["origin_clock_gap"]).sum()))
    return dict(definition="Reproduce FeatureStore.segment_starts exactly on every origin before filtering known Y: reset on date/cadence gap, segment-token change, or unavailable current/previous X. Real tokens=i-start+1; actual TCN tokens=min(real,supplied64,RF61); elapsed=(actual_tokens-1)*5s. Unknown labels do not reset history.",
        supplied_tokens=supplied, architecture_receptive_field=int(receptive_field),
        maximum_detailed_history_seconds=(cap-1)*config["data"]["cadence_seconds"],
        by_split=by_split,
        reset_counts_caution="Cause counts overlap. Token changes combine source capture/epoch/1s gap/admission and quote epoch/invalid/60s gap; these are not automatically physical quote support failures.",
        source_code_findings=[
            "features.asof_source starts a source segment after any endpoint step !=1s, capture/epoch change, or inadmissible source row (including recovery).",
            "full_history.generate_features admission additionally requires available_ns-t_ns<=1s, even if the canonical second's book is valid.",
            "finalize_market._second_rows creates one-second rows and publishes them at max(endpoint,next raw receive time); an ordinary quiet interval can therefore produce late rows and an admission-driven sequence reset without proving a quote support loss.",
            "This is a conservative representation boundary. Five minutes is the maximum receptive field, not the typical supplied detailed context. No frozen features, segments, runs or fits are changed by this audit."])


def source_fragmentation_witness(dataset, config, sources):
    """Reproduce one publication-delay reset with a 15-second quote slice."""
    root = Path(config["data"]["source_root"])
    key = "member_007/capture_017008"
    market = root/"market"/key
    features = root/"features"/key/"features.parquet"
    seconds = market/"seconds.parquet"
    manifest = market/"manifest.json"
    for path in (features, seconds, manifest):
        sources[str(path)] = sha(path)
    published = read(manifest)
    if sources[str(seconds)] != published["payloads"]["seconds.parquet"]:
        raise ValueError("Witness seconds differ from their canonical manifest")
    origins = np.array([1778482915000000000, 1778482920000000000], dtype=np.int64)
    start, end = 1778482910000000000, 1778482925000000000
    f = pd.read_parquet(features, columns=["t_ns", "available_ns", "epoch", "admissible", "scheduled_closed", "source_day_degraded"],
                        filters=[("t_ns", ">=", start), ("t_ns", "<=", end)])
    s = pd.read_parquet(seconds, columns=["t_ns", "available_ns", "epoch", "valid", "book_valid"],
                        filters=[("t_ns", ">=", start), ("t_ns", "<=", end)])
    q = pd.read_parquet(market/"quotes.parquet", columns=["available_ns", "epoch", "bid_nano", "ask_nano", "valid"],
                        filters=[("available_ns", ">=", start), ("available_ns", "<=", end)])
    original = pd.read_parquet(Path(dataset)/"2026-05-11_roots.parquet", columns=["t_ns", "feature_available", "segment_token"])
    observed = original.set_index("t_ns").loc[origins]
    if not observed.feature_available.all() or observed.segment_token.nunique() != 2:
        raise ValueError("Witness no longer changes segment between two available origins")
    late_ns = 1778482916000000000
    late, second = f.set_index("t_ns").loc[late_ns], s.set_index("t_ns").loc[late_ns]
    lag = (int(late.available_ns)-late_ns)/1e9
    qt = q.available_ns.to_numpy(np.int64)
    max_gap = float(np.diff(qt).max()/1e9)
    supported = bool(q.valid.all() and (q.ask_nano > q.bid_nano).all() and (q.bid_nano > 0).all()
                     and q.epoch.nunique() == 1 and max_gap <= config["execution"]["max_quote_gap_seconds"])
    if not (lag > 1 and not late.admissible and not late.scheduled_closed and not late.source_day_degraded
            and second.valid and second.book_valid and supported):
        raise ValueError("Publication-delay support witness did not reproduce")
    # Hash exactly the consumed quote slice, not the full quote payload.
    h = hashlib.sha256()
    for column in q:
        h.update(column.encode())
        h.update(q[column].to_numpy().tobytes())
    return dict(status="REPRODUCED", date="2026-05-11", origins_ns=[str(t) for t in origins],
        observed_segment_tokens=observed.segment_token.astype(int).tolist(),
        late_second=dict(endpoint_ns=str(late_ns), publication_ns=str(int(late.available_ns)),
            publication_delay_seconds=lag, feature_admissible=bool(late.admissible),
            canonical_second_valid=bool(second.valid), canonical_book_valid=bool(second.book_valid),
            epoch=str(int(late.epoch)), scheduled_closed=bool(late.scheduled_closed),
            source_day_degraded=bool(late.source_day_degraded)),
        bounded_quote_support=dict(start_ns=str(start), end_ns=str(end), rows=len(q),
            all_valid_positive_spread_single_epoch=supported, max_publication_gap_seconds=max_gap,
            support_gap_limit_seconds=config["execution"]["max_quote_gap_seconds"],
            first_publication_ns=str(int(qt[0])), last_publication_ns=str(int(qt[-1])),
            consumed_slice_sha256=h.hexdigest(), full_quote_sha256_declared_by_manifest=published["payloads"]["quotes.parquet"],
            full_quote_payload_rehashed=False),
        conclusion="This exact source-admission reset happens while the bounded canonical quote slice remains supported: a valid second published 1.215s late becomes inadmissible under the source 1s publication-delay rule. It demonstrates conservative fragmentation, not a physical quote support break.",
        generalization_limit="One verified witness; the share of all resets attributable to this mechanism was not measured. Do not describe it as the dominant cause without a full attribution audit.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", default="reports/scientific_pipeline_v2/suite_summary.json")
    parser.add_argument("--output", default="reports/scientific_pipeline_v2/statistical_audit.json")
    args = parser.parse_args()
    suite = read(args.suite)
    if suite.get("status") != "COMPLETE":
        raise SystemExit("Suite is not COMPLETE; no TEST audit performed and no output written")
    sources = {str(Path(__file__)): sha(__file__), args.suite: sha(args.suite)}
    dataset = Path(suite["dataset_root"])
    manifest = dataset/"manifest.json"
    sources[str(manifest)] = sha(manifest)
    if sources[str(manifest)] != suite["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed after the suite")
    primary = Path(suite["runs_root"])/suite["primary_run_id"]
    confirmation = Path(suite["runs_root"])/suite["confirmation_run_id"]
    config = read(primary/"config.json")
    dates = config["splits"]["test"]
    if len(dates) != 10:
        raise ValueError("This prespecified audit expects ten TEST days")
    runs = {role: audit_run(path, dates, sources) for role, path in (("primary", primary), ("confirmation", confirmation))}
    if runs["primary"]["cohort_test_sha256"] != runs["confirmation"]["cohort_test_sha256"]:
        raise ValueError("Primary and confirmation TEST cohorts differ")
    risk = risk_weeks(dataset, dates, config, sources)
    multiplicity = crossing_multiplicity(dataset, config, sources)
    history = actual_context_history(dataset, config, sources,
        read(primary/"summary.json")["training"]["architecture_receptive_field"])
    history["bounded_source_witness"] = source_fragmentation_witness(dataset, config, sources)
    for path in ("spectrum/research_v2/features.py", "spectrum/research_v2/learning.py", "spectrum/full_history.py", "spectrum/finalize_market.py"):
        sources[path] = sha(path)
    dataset_manifest = read(manifest)
    published = {name: digest for day in dataset_manifest["days"] for name, digest in day["files"].items()}
    for path, digest in sources.items():
        consumed = Path(path)
        if consumed.parent == dataset and consumed.name != "manifest.json" and published.get(consumed.name) != digest:
            raise ValueError("Consumed dataset payload differs from its manifest: " + path)
    for directory in (primary, confirmation):
        if read(directory/"receipt.json")["dataset_manifest_sha256"] != suite["dataset_manifest_sha256"]:
            raise ValueError("Run and audit dataset manifest differ")
    result = dict(status="COMPLETE", source_hashes=sources,
        purpose="Descriptive audit of prespecified primary and replication; no outcome-based selection",
        improvement_sign="Positive means TRAIN-prior NLL minus model NLL: lower model loss is better",
        definitions=dict(pooled="Each admitted known root has equal weight", equal_day="Each of ten TEST days has equal weight; all rows within a sampled day stay together",
            both="State 3: both LONG and SHORT fixed crossing events can occur within the horizon; neither side is a mutually exclusive binary complement",
            direction="Conditional three-way LONG_ONLY/SHORT_ONLY/BOTH; scored only where fixed crossing event is positive",
            temporal="Existing artifact differences across same-support adjacent known roots; target flip and probability total variation are not loss gradients"),
        runs=runs, paired_risk=risk, crossing_multiplicity=multiplicity, actual_context_history=history,
        caveats=["Ten adjacent days and a single GCM6 contract do not prove robustness across regimes/contracts.",
                 "These dates are new relative to scientific-v1 but some were inspected in older repository experiments.",
                 "Bootstrap is descriptive, not a significance certificate; no independent-day or stationarity claim.",
                 "Classification NLL, crossing retention, and after-cost sequential policy are distinct claims."])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="COMPLETE", output=args.output,
                         primary=runs["primary"]["run_id"], confirmation=runs["confirmation"]["run_id"])))


if __name__ == "__main__":
    main()
