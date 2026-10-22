"""Independent descriptive audit of the fixed context x weighting factorial.

Fitted-result analysis requires all eight fits and causal verification COMPLETE.
--metadata-only reads shared input/label identities and TRAIN weight recipes,
never model predictions. Neither mode selects an arm, seed, or trading policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONTEXTS = ("legacy", "restored")
WEIGHTINGS = ("uniform", "crossing_id")
STATES = ("NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH")
ID_COLUMNS = ("first_qualifying_event_id_L", "first_qualifying_event_id_S")
ROOT_METRICS = ("joint_nll", "event_nll", "event_auc", "direction_nll_given_event", "event_brier")
ID_METRICS = ("positive_side_nll", "positive_side_brier", "positive_joint_nll", "positive_event_nll", "positive_direction_nll")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_audit(path, result):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=result["status"], output=str(output), runs=len(result.get("runs", {})), sha256=sha(output))))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(values, dtype="<i8"):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, q10=None, q50=None, q90=None, maximum=None)
    q = np.quantile(values, [.1, .5, .9])
    return dict(n=len(values), mean=float(values.mean()), q10=float(q[0]), q50=float(q[1]),
                q90=float(q[2]), maximum=float(values.max()))


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    if (p.shape != (len(y), 4) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1), 1., rtol=0, atol=1e-6) or not np.isin(y, [0, 1, 2, 3]).all()):
        raise ValueError("Known labels and finite normalized joint probabilities required")
    if not len(y):
        return dict(n=0, **{name: None for name in ROOT_METRICS})
    p = np.clip(p, 1e-12, 1)
    event = y > 0
    ep = np.clip(1-p[:, 0], 1e-12, 1-1e-12)
    bce = -np.where(event, np.log(ep), np.log1p(-ep))
    direction = -np.log(p[np.flatnonzero(event), y[event]]/ep[event])
    return dict(n=len(y), joint_nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
        event_nll=float(bce.mean()), event_auc=float(roc_auc_score(event, ep)) if np.unique(event).size == 2 else None,
        direction_nll_given_event=float(direction.mean()) if len(direction) else None,
        event_brier=float(np.square(ep-event).mean()),
        state_counts=np.bincount(y, minlength=4).tolist(), event_fraction=float(event.mean()),
        direction_accuracy_given_event=float((p[event, 1:].argmax(1)+1 == y[event]).mean()) if event.any() else None)


def side_memberships(frame, admitted):
    """Return row index, side code, and ID for each positive root-side label."""
    y = frame.y_crossing.to_numpy()
    rows, sides, identities = [], [], []
    for side, column in enumerate(ID_COLUMNS):
        selected = np.flatnonzero(np.asarray(admitted, bool) & ((y & (1 << side)) != 0) & (y >= 0))
        ids = frame[column].to_numpy()[selected]
        if any(not isinstance(v, str) or not v.strip() for v in ids):
            raise ValueError("Positive admitted side has no crossing identity")
        rows.extend(selected.tolist())
        sides.extend([side]*len(selected))
        identities.extend(ids.tolist())
    return np.asarray(rows, np.int64), np.asarray(sides, np.int64), identities


def event_balanced_metrics(frame, p):
    """Positive-only diagnostic: equal (side, ID) weight, mean roots within ID."""
    y, p = frame.y_crossing.to_numpy(), np.asarray(p, float)
    probability_metrics(y, p)  # Same fail-closed probability/label validation.
    rows, sides, identities = side_memberships(frame, y >= 0)
    if not len(rows):
        return dict(positive_root_side_labels=0, distinct_side_ids=0,
                    metrics={name: None for name in ID_METRICS}, by_side={})
    keys = pd.MultiIndex.from_arrays([sides, identities])
    code, distinct = pd.factorize(keys, sort=True)
    count = np.bincount(code)
    side_probability = np.where(sides == 0, p[rows, 1]+p[rows, 3], p[rows, 2]+p[rows, 3])
    ep = np.clip(1-p[rows, 0], 1e-12, 1)
    true = np.clip(p[rows, y[rows]], 1e-12, 1)
    scores = dict(positive_side_nll=-np.log(np.clip(side_probability, 1e-12, 1)),
        positive_side_brier=np.square(1-side_probability), positive_joint_nll=-np.log(true),
        positive_event_nll=-np.log(ep), positive_direction_nll=-np.log(true/ep))
    per_id = {name: np.bincount(code, weights=value)/count for name, value in scores.items()}
    group_side = np.asarray(distinct.get_level_values(0), int)
    by_side = {}
    for side, name in enumerate(("LONG", "SHORT")):
        selected = group_side == side
        by_side[name] = dict(distinct_side_ids=int(selected.sum()), positive_root_side_labels=int((sides == side).sum()),
            metrics={metric: float(value[selected].mean()) if selected.any() else None for metric, value in per_id.items()})
    return dict(positive_root_side_labels=len(rows), distinct_side_ids=len(distinct),
        metrics={name: float(value.mean()) for name, value in per_id.items()}, by_side=by_side)


def context_starts(frame, mode, cadence_seconds=5):
    if mode not in CONTEXTS:
        raise ValueError("Unknown context mode")
    t, date = frame.t_ns.to_numpy(), frame.date.to_numpy()
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Unique chronological roots required")
    if mode == "legacy":
        ok, token = frame.feature_available.to_numpy(), frame.legacy_segment_token.to_numpy()
    else:
        ok, token = frame.physical_support_available.to_numpy(), frame.physical_segment_token.to_numpy()
    reset = np.r_[True, (np.diff(t) != int(round(cadence_seconds*1e9))) | (date[1:] != date[:-1]) |
                  (token[1:] != token[:-1]) | ~ok[1:] | ~ok[:-1]]
    return np.maximum.accumulate(np.where(reset, np.arange(len(t)), 0)), reset


def context_diagnostics(frame, splits, cadence_seconds=5, receptive_field=61, supplied=64):
    available = frame.feature_available.to_numpy(bool)
    supported = frame.physical_support_available.to_numpy(bool)
    if np.any(available & ~supported):
        raise ValueError("Feature admission outside physical support")
    n, index = len(frame), np.arange(len(frame))
    known = available & (frame.y_crossing.to_numpy() >= 0)
    cumulative = np.r_[0, np.cumsum(available)]
    next_observed = np.minimum.accumulate(np.where(available, index, n)[::-1])[::-1]
    modes, arrays = {}, {}
    for mode in CONTEXTS:
        start, reset = context_starts(frame, mode, cadence_seconds)
        real = index-start+1
        clock = np.where(supported, np.minimum(real, min(receptive_field, supplied)), 0)
        left = index-clock+1
        observed = cumulative[index+1]-cumulative[left]
        oldest = next_observed[np.minimum(left, n-1)]
        oldest_span = np.where(observed > 0, (index-oldest)*cadence_seconds, 0)
        fraction = np.divide(observed, clock, out=np.zeros(n, float), where=clock > 0)
        def describe(mask):
            selected = known & np.asarray(mask, bool)
            if not selected.any():
                return dict(known_feature_roots=0)
            return dict(known_feature_roots=int(selected.sum()),
                supported_clock_tokens=stats(clock[selected]), observed_feature_tokens=stats(observed[selected]),
                supported_span_seconds=stats((clock[selected]-1)*cadence_seconds),
                oldest_observed_span_seconds=stats(oldest_span[selected]),
                observed_fraction_of_supported_tokens=stats(fraction[selected]),
                observed_fraction_of_RF_slots=stats(observed[selected]/min(receptive_field, supplied)),
                full_RF61_clock_fraction=float((clock[selected] == min(receptive_field, supplied)).mean()),
                fully_observed_RF61_fraction=float((observed[selected] == min(receptive_field, supplied)).mean()),
                full_supplied64_clock_fraction=float((real[selected] >= supplied).mean()),
                clock_token_histogram=np.bincount(clock[selected], minlength=min(receptive_field, supplied)+1)[1:].tolist())
        modes[mode] = dict(by_split={role: describe(frame.date.isin(dates)) for role, dates in splits.items()},
            by_day={str(date): describe(frame.date.eq(date)) for date in frame.date.unique()})
        arrays[mode] = (start, reset)
    t, date = frame.t_ns.to_numpy(), frame.date.to_numpy()
    same_clock = np.r_[False, (np.diff(t) == int(round(cadence_seconds*1e9))) & (date[1:] == date[:-1])]
    legacy_reset, physical_reset = arrays["legacy"][1], arrays["restored"][1]
    source_only = same_clock & legacy_reset & ~physical_reset
    actual_physical = same_clock & physical_reset
    physical_token_change = np.r_[False, np.diff(frame.physical_segment_token.to_numpy()) != 0]
    missing = supported & ~available
    missing_start = missing & (np.r_[True, ~missing[:-1]] | physical_reset)
    missing_group = np.cumsum(missing_start)
    missing_runs = []
    missing_positions = np.flatnonzero(missing)
    groups = np.split(missing_positions, np.flatnonzero(np.diff(missing_group[missing]) != 0)+1) if missing.any() else []
    for positions in groups:
        first, last = int(positions[0]), int(positions[-1])
        missing_runs.append(dict(date=str(date[first]), clock_positions=len(positions),
            clock_occupancy_seconds=len(positions)*cadence_seconds,
            first_to_last_seconds=(last-first)*cadence_seconds,
            first_ns=str(t[first]), last_ns=str(t[last])))
    reset_counts = {}
    missing_by_split = {}
    for role, dates in {"all": list(frame.date.unique()), **splits}.items():
        mask = frame.date.isin(dates).to_numpy()
        reset_counts[role] = dict(same_day_regular_adjacent_pairs=int((mask & same_clock).sum()),
            physical_support_boundary_pairs=int((mask & actual_physical).sum()),
            physical_token_change_pairs=int((mask & same_clock & physical_token_change).sum()),
            physical_unavailable_endpoint_pairs=int((mask & same_clock & (~supported | np.r_[False, ~supported[:-1]])).sum()),
            legacy_extra_resets_with_physical_continuity=int((mask & source_only).sum()),
            legacy_extra_resets_between_two_admitted_origins=int((mask & source_only & available & np.r_[False, available[:-1]]).sum()),
            missing_X_inside_physical_support=int((mask & supported & ~available).sum()),
            physical_unavailable_roots=int((mask & ~supported).sum()))
        runs = [row for row in missing_runs if row["date"] in dates]
        missing_by_split[role] = dict(runs=len(runs), clock_positions=stats([row["clock_positions"] for row in runs]),
            clock_occupancy_seconds=stats([row["clock_occupancy_seconds"] for row in runs]),
            longest_examples=sorted(runs, key=lambda row: (-row["clock_positions"], row["first_ns"]))[:3])
    return dict(modes=modes, reset_counts=reset_counts, missing_X_runs_by_split=missing_by_split,
        missing_run_definition="Consecutive missing-X roots inside physical support, bounded at physical/date/cadence reset. Occupancy is positions times 5s; first-to-last timestamp span is (positions-1) times 5s. These are absent observations, not physical support gaps.",
        receptive_field=receptive_field, supplied_tokens=supplied,
        definition="On known admitted evaluation roots, RF61 counts actual supported 5s clock positions. Missing X positions are present in restored context but have zero values/masks. Observed-token fraction divides admitted X positions by supported clock positions; observed/RF also includes padding in its denominator. Unknown Y never resets context.",
        physical_definition="Saved full-publication quote token includes invalid, epoch and >60s gap transitions, even between two individually available origins. Day/cadence reset is counted separately from same-day physical boundary pairs.",
        source_only_definition="A same-day regular pair reset by legacy but not by restored support. This is a known observation/admission-only history reset, not a missing physical quote event.")


def weight_stats(weights):
    w = np.asarray(weights, float)
    total = float(w.sum())
    return dict(**stats(w), mass=total, minimum=float(w.min()) if len(w) else None,
        q99=float(np.quantile(w, .99)) if len(w) else None,
        kish_weight_ESS=total*total/float(np.square(w).sum()) if np.square(w).sum() else 0.)


def recompute_train_weights(frame, train_mask, scheme):
    if scheme not in WEIGHTINGS:
        raise ValueError("Unknown weighting scheme")
    selected = np.asarray(train_mask, bool) & frame.feature_available.to_numpy() & (frame.y_crossing.to_numpy() >= 0)
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("Nonempty known TRAIN cohort required")
    rows, sides, identities = side_memberships(frame, selected)
    counts = Counter(identities)
    raw = np.ones(len(frame), float)
    if scheme == "crossing_id":
        inverse_sum = np.bincount(rows, weights=[1/counts[k] for k in identities], minlength=len(frame))
        memberships = np.bincount(rows, minlength=len(frame))
        positive = memberships > 0
        raw[positive] = inverse_sum[positive]/memberships[positive]
    full = np.zeros(len(frame), float)
    classes = {}
    y = frame.y_crossing.to_numpy()
    for value, name in enumerate(STATES):
        mask = selected & (y == value)
        if mask.any():
            full[mask] = raw[mask]*float(mask.sum()/raw[mask].sum())
        classes[name] = dict(root_count=int(mask.sum()), raw=weight_stats(raw[mask]), normalized=weight_stats(full[mask]))
    receipt = dict(scheme=scheme, train_roots=len(indices), by_joint_class=classes,
        overall=weight_stats(full[indices]), train_root_indices_sha256=array_sha(indices),
        train_weights_sha256=array_sha(full[indices], "<f8"),
        count_table_sha256=hashlib.sha256(json.dumps(sorted(counts.items()), separators=(",", ":")).encode()).hexdigest(),
        distinct_positive_ids=len(counts), positive_side_memberships=len(identities),
        multiplicity=stats(list(counts.values())),
        ESS_definition="Kish (sum weights)^2/sum squared weights, a weight-concentration diagnostic; not an independent-event count or time-series effective sample size")
    return full, receipt


def identity_audit(frame, splits):
    known = frame.feature_available.to_numpy() & (frame.y_crossing.to_numpy() >= 0)
    by_split, sets = {}, {}
    for role, dates in splits.items():
        rows, sides, identities = side_memberships(frame, known & frame.date.isin(dates).to_numpy())
        counts = Counter(identities)
        sets[role] = set(counts)
        by_split[role] = dict(known_feature_roots=int((known & frame.date.isin(dates).to_numpy()).sum()),
            positive_root_side_labels=len(rows), unique_saved_ids=len(counts), multiplicity=stats(list(counts.values())),
            largest_id_share=max(counts.values())/len(rows) if len(rows) else None,
            unique_side_id_pairs=len(set(zip(sides.tolist(), identities))))
    overlap = {a+"__"+b: dict(shared_ids=len(sets[a] & sets[b]),
        shared_ids_sha256=hashlib.sha256(json.dumps(sorted(sets[a] & sets[b])).encode()).hexdigest())
        for a, b in itertools.combinations(splits, 2)}
    return dict(by_split=by_split, cross_split_intersections=overlap,
        no_cross_split_ids=all(v["shared_ids"] == 0 for v in overlap.values()),
        caveat="Saved crossing IDs may share market paths. Unique IDs and multiplicity are not counts of statistically independent events.")


def paired_interval(values, dates, replications=20000, seed=20261023):
    values = np.asarray(values, float)
    if values.shape != (len(dates),) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired date effects required")
    sampled = np.random.default_rng(seed).integers(0, len(values), size=(replications, len(values)))
    means = values[sampled].mean(1)
    return dict(days=len(values), dates=list(dates), mean=float(values.mean()),
        percentile_95_interval=np.quantile(means, [.025, .975]).tolist(),
        positive_days=int((values > 0).sum()), seed=seed, replications=replications,
        by_day=[dict(date=str(date), effect=float(value)) for date, value in zip(dates, values)])


def factorial_effects(day_rows, seeds=(17, 29), replications=20000):
    """Positive effects improve score; interaction is extra joint benefit."""
    indexed = {}
    for row in day_rows:
        key = row["split"], row["date"], int(row["seed"]), row["context"], row["weighting"]
        if key in indexed:
            raise ValueError("Duplicate factorial cell")
        indexed[key] = row["metrics"]
    output = {}
    for split in sorted({row["split"] for row in day_rows}):
        dates = sorted({row["date"] for row in day_rows if row["split"] == split})
        metric_names = sorted(set.intersection(*[set(row["metrics"]) for row in day_rows if row["split"] == split]))
        result = {}
        for metric in metric_names:
            by_seed, effects_for_seed = {}, {}
            valid_dates = []
            for date in dates:
                keys = [(split, date, seed, context, weighting) for seed in seeds for context in CONTEXTS for weighting in WEIGHTINGS]
                if any(key not in indexed for key in keys):
                    raise ValueError("Missing matched factorial cell")
                values = [indexed[key][metric] for key in keys]
                if all(value is not None and np.isfinite(value) for value in values):
                    valid_dates.append(date)
            if not valid_dates:
                result[metric] = dict(paired_dates=0, reason="metric undefined on every matched day")
                continue
            for seed in seeds:
                sign = 1. if metric == "event_auc" else -1.
                lu, lc, ru, rc = [sign*np.asarray([indexed[(split, date, seed, context, weighting)][metric] for date in valid_dates])
                    for context, weighting in (("legacy", "uniform"), ("legacy", "crossing_id"), ("restored", "uniform"), ("restored", "crossing_id"))]
                effects = dict(context_at_uniform=ru-lu, context_at_crossing_id=rc-lc,
                    weighting_at_legacy=lc-lu, weighting_at_restored=rc-ru,
                    context_main=(ru+rc-lu-lc)/2, weighting_main=(lc+rc-lu-ru)/2,
                    interaction=rc-ru-lc+lu)
                effects_for_seed[seed] = effects
                by_seed[str(seed)] = {name: paired_interval(value, valid_dates, replications) for name, value in effects.items()}
            average = {name: paired_interval(np.mean([effects_for_seed[s][name] for s in seeds], axis=0), valid_dates, replications)
                       for name in effects_for_seed[seeds[0]]}
            result[metric] = dict(paired_dates=len(valid_dates), by_seed=by_seed, mean_seed_effect_paired_by_date=average)
        output[split] = result
    return dict(by_split=output,
        sign="Positive means lower NLL/Brier or higher event AUC. Main effects average the two settings of the other factor. Interaction=(restored crossing_id-restored uniform)-(legacy crossing_id-legacy uniform) on utility, where utility=-loss or +AUC.",
        intervals="Descriptive equal-day paired percentile bootstrap. Same dates are resampled for all arms. Seed-mean intervals average seeds within each date before resampling; the same day is never counted as two independent days. No multiple-comparison or independent-day claim.",
        selection="All eight prespecified arms are reported; no TEST winner selection")


def verify_saved_gradients(prediction, p, weights, train):
    y = prediction.y_crossing.to_numpy()
    known = y >= 0
    event = 1-p[:, 0]-(y > 0)
    q = p[:, 1:]/p[:, 1:].sum(1, keepdims=True)
    q[np.arange(len(q)), np.maximum(y-1, 0)] -= 1
    direction = np.linalg.norm(q, axis=1)*(y > 0)
    errors = {}
    for column, expected, mask in (("g_event", event, known), ("g_direction_norm", direction, known),
        ("g_event_training_weighted", event*weights, train),
        ("g_direction_norm_training_weighted", direction*weights, train)):
        actual = prediction[column].to_numpy()
        error = float(np.abs(actual[mask]-expected[mask]).max()) if mask.any() else 0.
        tolerance = 2e-6*max(1., float(np.max(weights))) if "weighted" in column else 2e-6
        if not np.isfinite(error) or error > tolerance or not np.isnan(actual[~mask]).all():
            raise ValueError("Malformed or nonlocal saved gradient: " + column)
        errors[column] = error
    saved = prediction.optimization_weight_train_only.to_numpy()
    if not np.allclose(saved[train], weights[train], rtol=0, atol=1e-12) or not np.isnan(saved[~train]).all():
        raise ValueError("Optimization weights are wrong or extend outside TRAIN")
    return dict(status="PASS", maximum_absolute_errors=errors,
                definition="Independently reconstruct pE-yE and conditional softmax-minus-onehot norm from saved joint probabilities; weighted gradients equal fixed TRAIN weight times derivative; non-TRAIN weighted fields must be NaN")


def audit_run(directory, frame, config, manifest_sha, sources, expected_weights):
    receipt = read(directory/"receipt.json")
    if receipt.get("status") != "COMPLETE" or receipt.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Run is incomplete or bound to another dataset")
    for name in ("config.json", "summary.json", "predictions.parquet", "weighting.json", "train_weights.parquet", "gradient_artifact.json", "economic_evaluation.json", "history.json"):
        path = directory/name
        digest = sha(path)
        if receipt["outputs"].get(name) != digest:
            raise ValueError("Changed run payload: " + str(path))
        sources[str(path)] = digest
    sources[str(directory/"receipt.json")] = sha(directory/"receipt.json")
    saved_config, summary = read(directory/"config.json"), read(directory/"summary.json")
    context, weighting, seed = saved_config["data"]["context_mode"], saved_config["training"]["weighting"], saved_config["seed"]
    expected_id = f"tcn_micro_{context}_{weighting}_seed{seed}"
    if saved_config["run_id"] != expected_id or summary["run_id"] != expected_id or directory.name != expected_id:
        raise ValueError("Run ID does not match factorial configuration")
    normalized = json.loads(json.dumps(saved_config))
    reference = json.loads(json.dumps(config))
    for c in (normalized, reference):
        c.pop("run_id", None); c.pop("seed", None)
        c["data"].pop("context_mode", None); c["training"].pop("weighting", None)
    if normalized != reference:
        raise ValueError("An unplanned configuration changed between arms")
    prediction = pd.read_parquet(directory/"predictions.parquet")
    if not np.array_equal(prediction.root_index, np.arange(len(frame))):
        raise ValueError("Prediction root order changed")
    parity_columns = ["date", "t_ns", "feature_available", "y_crossing", "legacy_segment_token", "physical_segment_token", "physical_support_available"]
    for name in parity_columns:
        if not np.array_equal(prediction[name], frame[name]):
            raise ValueError("Prediction cohort/target/support differs: " + name)
    p = prediction[["p_none", "p_long", "p_short", "p_both"]].to_numpy(float)
    probability_metrics(np.zeros(len(p), np.int64), p)  # Validate UNKNOWN/full-clock predictions too.
    y = frame.y_crossing.to_numpy()
    known = frame.feature_available.to_numpy() & (y >= 0)
    train = known & frame.date.isin(config["splits"]["train"]).to_numpy()
    weights, weight_receipt = expected_weights[weighting]
    saved_weights = pd.read_parquet(directory/"train_weights.parquet")
    ids = np.flatnonzero(train)
    if not np.array_equal(saved_weights.root_index, ids) or not np.allclose(saved_weights.weight, weights[ids], rtol=0, atol=1e-12):
        raise ValueError("Saved TRAIN weights differ from independent reconstruction")
    published_weights = read(directory/"weighting.json")
    for saved, expected in (("train_root_indices_sha256", "train_root_indices_sha256"),
        ("crossing_id_count_table_sha256", "count_table_sha256"), ("train_weights_sha256", "train_weights_sha256")):
        if published_weights[saved] != weight_receipt[expected]:
            raise ValueError("TRAIN weight recipe identity differs")
    gradient_check = verify_saved_gradients(prediction, p, weights, train)
    artifact = read(directory/"gradient_artifact.json")
    economics, history = read(directory/"economic_evaluation.json"), read(directory/"history.json")
    prior = (np.bincount(y[train], minlength=4)+.5)/(train.sum()+2.)
    pooled, by_day, factorial_rows = {}, [], []
    for role, dates in config["splits"].items():
        selected = known & frame.date.isin(dates).to_numpy()
        model = probability_metrics(y[selected], p[selected])
        baseline = probability_metrics(y[selected], np.tile(prior, (selected.sum(), 1)))
        for metric in ROOT_METRICS:
            saved = summary["evaluation"][role]["model"].get(metric)
            if not (saved is None and model[metric] is None) and not np.isclose(saved, model[metric], rtol=0, atol=1e-10):
                raise ValueError("Independent probability score differs from run summary")
        balanced = event_balanced_metrics(frame.loc[selected].reset_index(drop=True), p[selected])
        pooled[role] = dict(model=model, train_prior=baseline, event_balanced_positive=balanced)
        for date in dates:
            mask = known & frame.date.eq(date).to_numpy()
            point = probability_metrics(y[mask], p[mask])
            event_id = event_balanced_metrics(frame.loc[mask].reset_index(drop=True), p[mask])
            by_day.append(dict(date=date, split=role, model=point, event_balanced_positive=event_id))
            flattened = {name: point[name] for name in ROOT_METRICS}
            flattened.update({"event_id_"+name: value for name, value in event_id["metrics"].items()})
            factorial_rows.append(dict(date=date, split=role, seed=seed, context=context, weighting=weighting, metrics=flattened))
    clips = [row["gradient_clip_fraction"] for row in history if row.get("gradient_clip_fraction") is not None]
    result = dict(run_id=expected_id, seed=seed, context=context, weighting=weighting, pooled=pooled, by_day=by_day,
        training=summary["training"], context_artifact=summary["context"], weighting_artifact=published_weights,
        gradients=dict(independent_check=gradient_check,
            unweighted_full_clock_groups=artifact["full_clock_groups"], weighted_train_groups=artifact["weighted_train_groups"],
            before=artifact.get("before"), selected=artifact.get("selected"),
            distinction="Actual unweighted and weighted TRAIN gradients at the same selected parameters; weighted VAL/CAL/TEST gradients are intentionally undefined"),
        gradient_clip_fraction_by_validation_interval=clips, temporal=artifact["temporal"],
        economic_policy=economics["summary"], calibration_grid=economics["calibration_grid"],
        scaler_sha256=summary["scaler_sha256"], cohorts=summary["cohorts"],
        source_identity=receipt["sources"])
    return result, factorial_rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", default="reports/scientific_pipeline_v3/suite_summary.json")
    parser.add_argument("--config", default="configs/research/opportunity_pipeline_v3.json")
    parser.add_argument("--causal-proof", default="reports/scientific_pipeline_v3/causal_verification.json")
    parser.add_argument("--output", default="reports/scientific_pipeline_v3/statistical_audit.json")
    parser.add_argument("--metadata-only", action="store_true", help="Shared dataset context and TRAIN weights only; never read fitted results")
    args = parser.parse_args()
    if not args.metadata_only and (not Path(args.suite).exists() or read(args.suite).get("status") != "COMPLETE"):
        raise SystemExit("Eight-fit suite is not COMPLETE; no fitted-result audit or output performed")
    config = read(args.config)
    suite = None if args.metadata_only else read(args.suite)
    seeds = tuple(config["experiment"]["seeds"])
    expected = {f"tcn_micro_{c}_{w}_seed{s}" for s in seeds for c in CONTEXTS for w in WEIGHTINGS}
    if not args.metadata_only and (len(expected) != 8 or {row["run_id"] for row in suite["runs"]} != expected or len(suite["runs"]) != 8):
        raise ValueError("Exactly eight prespecified factorial runs required")
    proof = read(args.causal_proof)
    if proof.get("status") not in {"COMPLETE", "PASS"} or proof.get("config_sha256") != sha(args.config):
        raise ValueError("Completed independent causal/parity verification required")
    sources = {p: sha(p) for p in (args.config, args.causal_proof, str(Path(__file__)))}
    if not args.metadata_only:
        sources[args.suite] = sha(args.suite)
    dataset = Path(config["data"]["root"] if args.metadata_only else suite["dataset_root"])
    manifest_path = dataset/"manifest.json"
    manifest = read(manifest_path)
    manifest_sha = sha(manifest_path)
    sources[str(manifest_path)] = manifest_sha
    if manifest.get("status") != "COMPLETE" or (not args.metadata_only and suite["dataset_manifest_sha256"] != manifest_sha):
        raise ValueError("Dataset is incomplete or changed after fitting")
    if manifest["splits"] != config["splits"] or manifest["dates"] != config["data"]["dates"]:
        raise ValueError("Dataset time blocks differ from configuration")
    frames, parity = [], []
    for day in manifest["days"]:
        if day["status"] != "COMPLETE":
            raise ValueError("Incomplete source day")
        for name in (day["roots_file"], day["x_file"]):
            path = dataset/name
            digest = sha(path)
            if digest != day["files"][name]:
                raise ValueError("Changed shared dataset payload: " + str(path))
            sources[str(path)] = digest
        frame = pd.read_parquet(dataset/day["roots_file"])
        frames.append(frame)
        history = day["feature_receipt"]["v3_history"]
        checks = {name: history[name] for name in ("v2_values_unchanged", "v2_current_origin_admission_unchanged", "v2_legacy_tokens_reproduced", "no_forward_fill")}
        if not all(checks.values()) or history["extra_feature_channels"] != 0:
            raise ValueError("V3 changed more than history semantics")
        parity.append(dict(date=day["date"], **checks, extra_feature_channels=0,
                           shared_X_sha256=day["files"][day["x_file"]], shared_roots_sha256=day["files"][day["roots_file"]]))
    frame = pd.concat(frames, ignore_index=True)
    if np.any(np.diff(frame.t_ns.to_numpy()) <= 0):
        raise ValueError("Dataset roots are not strictly chronological")
    if not np.array_equal(frame.segment_token, frame.legacy_segment_token):
        raise ValueError("Legacy compatibility alias differs")
    identities = identity_audit(frame, config["splits"])
    if not identities["no_cross_split_ids"]:
        raise ValueError("Crossing identity crosses configured split boundary")
    train_mask = frame.date.isin(config["splits"]["train"]).to_numpy()
    expected_weights = {scheme: recompute_train_weights(frame, train_mask, scheme) for scheme in WEIGHTINGS}
    contexts = context_diagnostics(frame, config["splits"], config["data"]["cadence_seconds"], 61, config["data"]["sequence_length"])
    if args.metadata_only:
        write_audit(args.output, dict(status="DATA_DIAGNOSTICS_ONLY_PENDING_FITS", source_hashes=sources,
            execution_scope="No run summaries, probabilities, or fitted TEST results read. Shared data context, crossing-ID identity counts, and TRAIN-only deterministic weights audited.",
            context_availability=contexts, identity_overlap=identities,
            train_weight_audit={name: receipt for name, (_, receipt) in expected_weights.items()},
            parity=dict(daily=parity, independent_causal_verification=dict(path=args.causal_proof,
                sha256=sources[args.causal_proof], evidence=proof))))
        return
    runs, rows = {}, []
    for run_id in sorted(expected):
        result, day_rows = audit_run(Path(suite["runs_root"])/run_id, frame, config, manifest_sha, sources, expected_weights)
        runs[run_id] = result
        rows.extend(day_rows)
    values = list(runs.values())
    for key in ("scaler_sha256", "cohorts", "source_identity"):
        if any(value[key] != values[0][key] for value in values[1:]):
            raise ValueError("Unmatched factorial arms: " + key)
    for result in values:
        for role in config["splits"]:
            independent = contexts["modes"][result["context"]]["by_split"][role]
            saved = result["context_artifact"]["by_split"][role]
            for expected_name, saved_name in (("supported_clock_tokens", "supported_clock_rows"),
                ("observed_feature_tokens", "observed_feature_rows"),
                ("oldest_observed_span_seconds", "oldest_observed_feature_span_seconds")):
                if any(not np.isclose(independent[expected_name][key], saved[saved_name][key], rtol=0, atol=1e-10)
                       for key in ("n", "mean", "q10", "q50", "q90")):
                    raise ValueError("Independent history audit differs from the learner")
    for seed in seeds:
        matching = [value for value in values if value["seed"] == seed]
        for key in ("initial_state_sha256", "train_metric_probe_sha256", "gradient_probe_sha256"):
            if len({value["training"][key] for value in matching}) != 1:
                raise ValueError("Unmatched seed initialization or probe: " + key)
    old_config_path = Path("configs/research/opportunity_pipeline_v2.json")
    old = read(old_config_path)
    sources[str(old_config_path)] = sha(old_config_path)
    fixed = {name: old[name] == config[name] for name in ("event_query", "execution", "risk", "loss")}
    if not all(fixed.values()):
        raise ValueError("V2 target, execution, risk or loss recipe changed")
    result = dict(status="COMPLETE", source_hashes=sources,
        design="Prespecified 2x2 context x weighting factorial at seeds17/29; checkpoint choice uses unweighted VAL NLL, policy uses CAL only; audit selects no TEST winner",
        context_availability=contexts, identity_overlap=identities,
        train_weight_audit={name: receipt for name, (_, receipt) in expected_weights.items()},
        parity=dict(fixed_v2_recipes=fixed, shared_input_feature_names=manifest["groups"]["micro"],
            all_arms_same_scaler_cohort_and_source=True, independent_context_matches_learner=True,
            within_seed_initialization_and_probes_matched=True, daily=parity,
            independent_causal_verification=dict(path=args.causal_proof, sha256=sources[args.causal_proof], evidence=proof)),
        runs=runs, factorial=factorial_effects(rows, seeds),
        event_balanced_definition="Among known feature-available positive side memberships, group by (LONG or SHORT, first qualifying ID). First average each score across that group's roots, then average groups equally. BOTH supplies two memberships. Overall denominator is distinct side-ID groups, not roots or raw ID alone. NEGATIVE roots are excluded; this cannot measure false positives, full-population calibration, or economic edge. Uniform-root metrics remain primary.",
        limitations=["Kish weight ESS quantifies weight concentration, not independent financial observations.",
            "Class-normalized inverse multiplicity preserves joint-class mass but does not exactly equalize crossing IDs.",
            "Day intervals are descriptive; adjacent days and shared market events are dependent, and dates were historically exposed.",
            "Context, training weights, BCE/CE derivatives, crossing outcomes, and after-cost policy returns are separate claims."])
    write_audit(args.output, result)


if __name__ == "__main__":
    main()
