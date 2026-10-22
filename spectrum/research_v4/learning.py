"""V4: bounded soft crossing-ID weights on the frozen restored V3 context.

The orchestration loop below is a bounded copy of frozen V3 train_run, with
only configuration validation, TRAIN weight construction and run-stage name
adapted. All data, model, original loss, gradient and prediction helpers are
explicit imports. No global monkeypatch or change to V3 code is performed.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from spectrum.research.losses import classification_loss
from spectrum.research.models import build_model
from spectrum.research.training import metrics
from spectrum.research_v2 import learning as v2
from spectrum.research_v3 import learning as v3
from spectrum.research_v3.learning import (
    EVENT_ID_COLUMNS, STATE_NAMES, FeatureStore, load_dataset, fit_scaler,
    split_indices, sha, _save, _array_sha, _state_sha, _weight_summary,
    _batch, predict, evaluate, weighted_classification_loss, logit_gradients,
    gradient_diagnostic, context_diagnostic, _prediction_frame,
)


FIXED_WEIGHT_PARAMETERS = dict(alpha=0.5, min_weight=0.5, max_weight=2.0)


def validate_weight_parameters(parameters):
    if parameters != FIXED_WEIGHT_PARAMETERS:
        raise ValueError("V4 prespecifies alpha=0.5 and final bounds [0.5,2.0]")
    if any(isinstance(value, bool) or not np.isfinite(value) for value in parameters.values()):
        raise ValueError("Finite numeric weighting parameters required")


def bounded_class_normalize(raw, lower=0.5, upper=2.0):
    """Solve sum clip(a*r,lower,upper)=N without any later renormalization.

    The positive multiplier is found by monotone float64 bisection. Its upper
    bracket makes every root reach the upper cap. Final clipping preserves
    the bounds exactly; class-mass residual is separately checked and saved.
    """
    r = np.asarray(raw, np.float64)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("Nonempty finite positive raw class weights required")
    if not np.isfinite(lower) or not np.isfinite(upper) or not (0 < lower < 1 < upper):
        raise ValueError("Final bounds must strictly contain unit class mean")
    lo, hi = 0., float(upper / r.min())
    if not np.isfinite(hi):
        raise ValueError("Raw weights exceed stable normalization range")
    for _ in range(100):
        middle = (lo + hi) / 2
        if np.clip(middle * r, lower, upper).sum() < len(r):
            lo = middle
        else:
            hi = middle
    multiplier = (lo + hi) / 2
    weights = np.clip(multiplier * r, lower, upper)
    residual = float(weights.sum() - len(r))
    if abs(residual) > max(1e-10, 1e-12 * len(r)):
        raise ValueError("Bounded class normalization failed its mass equation")
    return weights, multiplier, residual


def compute_train_weights(store, train_ids, scheme="uniform", parameters=None):
    """Fit soft weights from known TRAIN side-ID memberships only.

    Counts pool sides, including BOTH. Side raw weight is m(ID)^(-alpha);
    BOTH averages the two side raw weights. Positive joint classes solve
    their own clipped mass equation. NONE remains exactly 1. The returned
    full-clock vector is zero outside the supplied TRAIN cohort.
    """
    parameters = dict(FIXED_WEIGHT_PARAMETERS if parameters is None else parameters)
    validate_weight_parameters(parameters)
    if scheme not in {"uniform", "soft_crossing_id"}:
        raise ValueError("Unsupported V4 weighting scheme")
    frame, ids = store.frame, np.asarray(train_ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or np.any(ids < 0) or np.any(ids >= len(frame)) or len(np.unique(ids)) != len(ids)):
        raise ValueError("Unique nonempty integer TRAIN root indices required")
    ids = np.sort(ids)
    y = frame.y_crossing.to_numpy()[ids]
    if np.any(y < 0) or not frame.feature_available.to_numpy()[ids].all():
        raise ValueError("Weight fitting requires the known feature-available TRAIN cohort")
    memberships, side_ids = [], {}
    for side, column in EVENT_ID_COLUMNS.items():
        if column not in frame:
            raise ValueError("Missing TRAIN crossing identity column: " + column)
        mask = (y == side) | (y == 3)
        values = frame[column].to_numpy()[ids][mask]
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Every positive TRAIN side requires a nonempty crossing ID")
        memberships.extend(values.tolist())
        side_ids[side] = (mask, values)
    multiplicity = Counter(memberships)
    side_raw = np.zeros((len(ids), 2), np.float64)
    for side, (mask, values) in side_ids.items():
        side_raw[mask, side - 1] = [multiplicity[value] ** -parameters["alpha"] for value in values]
    raw = np.ones(len(ids), np.float64)
    if scheme == "soft_crossing_id":
        raw[y == 1] = side_raw[y == 1, 0]
        raw[y == 2] = side_raw[y == 2, 1]
        raw[y == 3] = side_raw[y == 3].mean(axis=1)
    weights = np.ones(len(ids), np.float64)
    normalizers, classes = {}, {}
    lower, upper = parameters["min_weight"], parameters["max_weight"]
    for value, name in enumerate(STATE_NAMES):
        mask = y == value
        normalizer, residual = (1., 0.) if mask.any() else (None, 0.)
        if mask.any() and value > 0 and scheme == "soft_crossing_id":
            weights[mask], normalizer, residual = bounded_class_normalize(raw[mask], lower, upper)
        normalizers[name] = normalizer
        n = int(mask.sum())
        below, above = int((weights[mask] == lower).sum()), int((weights[mask] == upper).sum())
        classes[name] = dict(original_root_count=n, raw=_weight_summary(raw[mask]),
            normalized=_weight_summary(weights[mask]), class_mass_residual=residual,
            lower_bound_count=below, upper_bound_count=above,
            lower_bound_fraction=below/n if n else None, upper_bound_fraction=above/n if n else None)
    if np.any(weights < lower) or np.any(weights > upper) or not np.all(weights[y == 0] == 1.):
        raise ValueError("Final V4 weights violate bounds or NONE invariance")
    full = np.zeros(len(frame), np.float64)
    full[ids] = weights
    counts = np.asarray(list(multiplicity.values()), np.int64)
    count_digest = hashlib.sha256(json.dumps(sorted(multiplicity.items()), separators=(",", ":")).encode()).hexdigest()
    receipt = dict(scheme=scheme, parameters=parameters, fitted_on="known feature-available TRAIN roots only",
        train_root_indices_sha256=_array_sha(ids, "<i8"), train_root_count=len(ids),
        train_weights_sha256=_array_sha(weights, "<f8"), full_clock_weight_vector_sha256=_array_sha(full, "<f8"),
        raw_formula="NONE=1; L=m(idL)^(-alpha); S=m(idS)^(-alpha); BOTH=mean(L_raw,S_raw)",
        multiplicity_definition="All positive known-TRAIN (root,side) memberships pooled across sides, including BOTH",
        normalization="For each positive joint class solve sum clip(a*raw,0.5,2.0)=original root count; NONE=1; no later renormalization",
        solver=dict(method="monotone float64 bisection", iterations=100,
                    lower_bracket=0., upper_bracket_definition="max_weight/min(raw class weights)",
                    final_bounds_after_normalization=[lower, upper], post_clip_renormalization=False),
        class_normalizers=normalizers, by_joint_class=classes, overall=_weight_summary(weights),
        unique_positive_crossing_ids=len(multiplicity), positive_side_memberships=len(memberships),
        crossing_id_count_table_sha256=count_digest,
        id_multiplicity=dict(minimum=int(counts.min()) if len(counts) else None,
            q50=float(np.quantile(counts, .5)) if len(counts) else None,
            q90=float(np.quantile(counts, .9)) if len(counts) else None,
            maximum=int(counts.max()) if len(counts) else None),
        loss_denominator="Number of known roots in batch; not sum of weights",
        caveats=["Class masses are preserved; actual gradient masses may change with prediction errors.",
                 "Soft capped identity weights change the TRAIN measure and do not guarantee uniform-root calibration.",
                 "No held-out identity is counted and no identity is a model feature; evaluation remains unweighted."])
    return full, receipt


def _validate_config(store, config):
    if config.get("schema") != "scientific_opportunity_pipeline_v4":
        raise ValueError("Unsupported V4 configuration schema")
    if config["data"].get("context_mode") != "restored":
        raise ValueError("V4 fixes restored context")
    if config["training"].get("weighting") not in {"uniform", "soft_crossing_id"}:
        raise ValueError("Unsupported V4 weighting scheme")
    validate_weight_parameters(config["training"]["id_weighting"])
    previous = copy.deepcopy(config)
    previous["schema"] = "scientific_opportunity_pipeline_v3"
    previous["training"]["weighting"] = "uniform"
    v3._validate_config(store, previous)


def train_run(store, config, out):
    """Fit one V4 arm using the frozen V3 orchestration; caller finalizes its receipt."""
    _validate_config(store, config)
    tr = config["training"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    owned = ("checkpoint.pt", "history.json", "scaler.json", "predictions.parquet", "summary.json",
             "gradient_artifact.json", "weighting.json", "train_weights.parquet")
    if any((out / name).exists() for name in owned):
        raise ValueError("Training outputs already exist; use a new immutable run directory")
    if (out / "config.json").exists():
        if v2._json(out / "config.json") != config:
            raise ValueError("Existing run configuration differs")
    else:
        _save(out / "config.json", config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(tr["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = tr.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = split_indices(store, config)
    if any(not len(ids) for ids in split.values()):
        raise ValueError("Every split requires a nonempty common known cohort")
    scaler = fit_scaler(store, np.flatnonzero(store.frame.date.isin(config["splits"]["train"]).to_numpy()))
    _save(out / "scaler.json", scaler)
    train_weights, weight_receipt = compute_train_weights(store, split["train"], tr["weighting"], tr["id_weighting"])
    weight_frame = store.frame.iloc[split["train"]][["date", "t_ns", "y_crossing", *EVENT_ID_COLUMNS.values()]].copy()
    weight_frame.insert(0, "root_index", split["train"])
    weight_frame["weight"] = train_weights[split["train"]]
    weight_frame.to_parquet(out / "train_weights.parquet", index=False)
    weight_receipt["train_weights_parquet_sha256"] = sha(out / "train_weights.parquet")
    _save(out / "weighting.json", weight_receipt)
    columns = store.groups[config["data"]["family"]]
    input_dim = 2 * len(columns) + 1
    model = build_model(input_dim, config["model"]).to(device)
    initial_state_sha256 = _state_sha(model.state_dict())
    _batch(store, split["train"][:1], config, scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    train_probe = np.sort(np.random.default_rng(config["seed"] + 410).choice(
        split["train"], min(tr["train_metric_probe"], len(split["train"])), replace=False))
    gradient_probe = np.sort(np.random.default_rng(config["seed"] + 411).choice(
        split["train"], min(tr["gradient_probe"], len(split["train"])), replace=False))
    y = store.frame.y_crossing.to_numpy(np.int64)
    prior = (np.bincount(y[split["train"]], minlength=4) + .5) / (len(split["train"]) + 2.)
    gradient_before = gradient_diagnostic(store, model, config, scaler, gradient_probe, train_weights)
    initial_val = evaluate(store, model, config, scaler, split["validation"])
    initial_train = evaluate(store, model, config, scaler, train_probe)
    history = [dict(update=0, examples_seen=0, cycles_completed=0, optimization_loss=None, unweighted_batch_loss=None,
        train_probe_nll=initial_train["joint_nll"], validation_nll=initial_val["joint_nll"],
        train_probe_accuracy=initial_train["accuracy"], validation_accuracy=initial_val["accuracy"],
        gradient_norm_mean=None, gradient_norm_max=None, gradient_clip_fraction=None,
        sampled_root_stream_sha256=hashlib.sha256().hexdigest(),
        stage="untrained_network_diagnostic_and_checkpoint_candidate")]
    best, best_update, stale = initial_val["joint_nll"], 0, 0
    best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
    rng = np.random.default_rng(config["seed"])
    order, offset, cycles, examples = rng.permutation(split["train"]), 0, 0, 0
    interval_losses, interval_unweighted, interval_norms = [], [], []
    sampled_root_stream = hashlib.sha256()
    started = time.perf_counter()
    for update in range(1, tr["max_updates"] + 1):
        if offset == len(order):
            cycles += 1
            order, offset = rng.permutation(split["train"]), 0
        ids = order[offset:offset + tr["batch_size"]]
        offset += len(ids)
        sampled_root_stream.update(np.asarray(ids, dtype="<i8").tobytes())
        model.train()
        xb = torch.as_tensor(_batch(store, ids, config, scaler), device=device)
        yy = torch.as_tensor(y[ids], device=device)
        ww = torch.as_tensor(train_weights[ids], dtype=xb.dtype, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = weighted_classification_loss(logits, yy, ww, config["loss"])
        unweighted = classification_loss(logits, yy, config["loss"])
        if not torch.isfinite(loss) or not torch.isfinite(unweighted):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True))
        optimizer.step()
        examples += len(ids)
        interval_losses.append((float(loss.detach()), len(ids)))
        interval_unweighted.append((float(unweighted.detach()), len(ids)))
        interval_norms.append(norm)
        if update % tr["validate_every"] and update != tr["max_updates"]:
            continue
        validation = evaluate(store, model, config, scaler, split["validation"])
        train_metric = evaluate(store, model, config, scaler, train_probe)
        row = dict(update=update, examples_seen=examples, cycles_completed=cycles + int(offset == len(order)),
            optimization_loss=sum(v * n for v, n in interval_losses) / sum(n for _, n in interval_losses),
            unweighted_batch_loss=sum(v * n for v, n in interval_unweighted) / sum(n for _, n in interval_unweighted),
            train_probe_nll=train_metric["joint_nll"], validation_nll=validation["joint_nll"],
            train_probe_accuracy=train_metric["accuracy"], validation_accuracy=validation["accuracy"],
            gradient_norm_mean=float(np.mean(interval_norms)), gradient_norm_max=float(np.max(interval_norms)),
            gradient_clip_fraction=float(np.mean(np.asarray(interval_norms) > tr["gradient_clip"])),
            sampled_root_stream_sha256=sampled_root_stream.hexdigest(), stage="trained")
        history.append(row)
        if validation["joint_nll"] < best - tr["min_delta"]:
            best, best_update, stale = validation["joint_nll"], update, 0
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
        else:
            stale += 1
        print(json.dumps({**row, "stage": "fit_v4", "run": config["run_id"]}), flush=True)
        interval_losses, interval_unweighted, interval_norms = [], [], []
        if stale >= tr["patience_checks"]:
            break
    model.load_state_dict(best_state)
    gradient_selected = gradient_diagnostic(store, model, config, scaler, gradient_probe, train_weights)
    probability, logits = predict(store, model, config, scaler, return_logits=True)
    evaluation = {role: dict(model=metrics(y[ids], probability[ids]), prior=metrics(y[ids], np.tile(prior, (len(ids), 1))))
                  for role, ids in split.items()}
    predictions, gradient_artifact = _prediction_frame(store, probability, logits, config, train_weights)
    gradient_artifact.update(before=gradient_before, selected=gradient_selected)
    cohorts = {role: dict(dates=config["splits"][role], known_feature_available_roots=len(ids),
        root_indices_sha256=_array_sha(ids, "<i8"), state_counts=np.bincount(y[ids], minlength=4).tolist()) for role, ids in split.items()}
    context = context_diagnostic(store, config, split, model.receptive_field)
    training = dict(best_update=best_update, selected_checkpoint="untrained_network" if best_update == 0 else "trained_network",
        best_validation_nll=best, updates_completed=update, examples_seen=examples, validation_checks=len(history)-1,
        seconds=time.perf_counter()-started, train_metric_probe_indices=train_probe.tolist(),
        initial_state_sha256=initial_state_sha256, sampled_root_stream_sha256=sampled_root_stream.hexdigest(),
        train_metric_probe_sha256=_array_sha(train_probe, "<i8"), gradient_probe_sha256=_array_sha(gradient_probe, "<i8"),
        parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
        architecture_receptive_field=model.receptive_field, supplied_tokens=config["data"]["sequence_length"],
        effective_tokens=context["effective_receptive_field"], theoretical_max_history_seconds=context["maximum_span_seconds"],
        gradients_before=gradient_before, gradients_selected=gradient_selected,
        selection="Unweighted validation joint NLL only; initial network eligible; prior is separate",
        beats_train_prior_on_validation=best < evaluation["validation"]["prior"]["joint_nll"],
        sampling="Uniform TRAIN-root permutations; identical order by seed across all arms",
        weighting=tr["weighting"], loss_denominator="Known-root batch count, not weight sum; no predicted event gate")
    summary = dict(run_id=config["run_id"], status="TRAINING_ARTIFACTS_WRITTEN_REQUIRES_RUN_FINALIZATION",
        context_mode=config["data"]["context_mode"], weighting=tr["weighting"], feature_family=config["data"]["family"],
        feature_names=columns, input_channels=input_dim, training=training, evaluation=evaluation, cohorts=cohorts,
        context=context, weight_diagnostics=weight_receipt, weighting_sha256=sha(out / "weighting.json"),
        scaler_sha256=sha(out / "scaler.json"), source_receipt=store.source_receipt,
        memory=dict(raw_shape=list(store.raw.shape), raw_bytes=store.raw.nbytes, materialized_full_sequence_tensor=False,
                    one_input_batch_bytes=tr["batch_size"] * config["data"]["sequence_length"] * input_dim * 4),
        caveats=["Restored context supplies supported clock positions; missing observations remain masked, never forward-filled.",
                 "Crossing IDs only weight TRAIN optimization and never become X or weighted validation/test metrics.",
                 "Class-mass preservation does not guarantee uniform-root probability calibration or economic value.",
                 "Original feature values and current-origin admission are identical across the context arms."])
    torch.save(dict(state_dict=best_state, input_dim=input_dim, model_config=config["model"], loss_config=config["loss"],
        seed=config["seed"], best_update=best_update, scaler_sha256=summary["scaler_sha256"], feature_names=columns,
        initial_state_sha256=initial_state_sha256,
        context_mode=config["data"]["context_mode"], clock_flag_definition=context["clock_flag_definition"],
        weighting=tr["weighting"], train_weights_sha256=weight_receipt["train_weights_sha256"]), out / "checkpoint.pt")
    predictions.to_parquet(out / "predictions.parquet", index=False)
    _save(out / "history.json", history)
    _save(out / "gradient_artifact.json", gradient_artifact)
    _save(out / "summary.json", summary)
    return dict(summary=summary, predictions=predictions, history=history, gradient_artifact=gradient_artifact)
