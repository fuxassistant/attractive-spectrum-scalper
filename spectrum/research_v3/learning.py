"""Matched context and crossing-ID weighting experiments.

Future crossing identities are optimization metadata, never model inputs.
Weights are fitted on known TRAIN roots only. Uniform root sampling and
unweighted validation joint NLL remain fixed across the experiment matrix.
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
from torch.nn import functional as F

from spectrum.research.losses import classification_loss, joint_probabilities
from spectrum.research.models import build_model
from spectrum.research.training import metrics
from spectrum.research_v2 import learning as v2


EVENT_ID_COLUMNS = {1: "first_qualifying_event_id_L", 2: "first_qualifying_event_id_S"}
STATE_NAMES = ("NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH")
fit_scaler = v2.fit_scaler
split_indices = v2.split_indices
sha = v2.sha
_save = v2._save


class FeatureStore(v2.FeatureStore):
    """One NxF matrix, two causal window definitions, identical input schema.

    The last channel denotes a supported clock position, including positions
    with no admitted X. Those positions have zero values AND zero feature
    masks. No forward fill, target-based reset, or new age channel is used.
    """

    def __init__(self, frame, raw, feature_names, groups, *, source_receipt=None):
        required = {"legacy_segment_token", "physical_segment_token", "physical_support_available"}
        if not required <= set(frame):
            raise ValueError("Missing v3 context columns: " + str(sorted(required - set(frame))))
        f = frame.copy()
        if "segment_token" in f and not np.array_equal(f.segment_token, f.legacy_segment_token):
            raise ValueError("Compatibility segment_token must equal legacy_segment_token")
        f["segment_token"] = f.legacy_segment_token
        super().__init__(f, raw, feature_names, groups, source_receipt=source_receipt)
        if f.physical_support_available.dtype != bool:
            raise ValueError("physical_support_available must be boolean")
        for column in ("legacy_segment_token", "physical_segment_token"):
            if not np.issubdtype(f[column].dtype, np.integer):
                raise ValueError("Integer support tokens required")
        if (f.feature_available & ~f.physical_support_available).any():
            raise ValueError("Admitted current X requires physical support")
        causal = ("source_endpoint_ns", "source_available_ns")
        if all(column in f for column in causal):
            admitted = f.feature_available.to_numpy()
            endpoint, available, origin = (f[column].to_numpy()[admitted] for column in (*causal, "t_ns"))
            if np.any(endpoint > available) or np.any(available > origin):
                raise ValueError("Admitted X cannot precede its source endpoint/publication")

    def segment_starts(self, cadence_seconds, context_mode="legacy"):
        if context_mode not in {"legacy", "restored"}:
            raise ValueError("Unsupported context mode")
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
            raise ValueError("Positive finite cadence required")
        step = int(round(float(cadence_seconds) * 1e9))
        if step <= 0:
            raise ValueError("Positive nanosecond cadence required")
        key = (context_mode, step)
        if key not in self._segments:
            f = self.frame
            t, date = f.t_ns.to_numpy(), f.date.to_numpy()
            if context_mode == "legacy":
                ok, token = f.feature_available.to_numpy(), f.legacy_segment_token.to_numpy()
            else:
                ok, token = f.physical_support_available.to_numpy(), f.physical_segment_token.to_numpy()
            reset = np.ones(len(f), bool)
            reset[1:] = ((np.diff(t) != step) | (date[1:] != date[:-1]) |
                         (token[1:] != token[:-1]) | ~ok[1:] | ~ok[:-1])
            self._segments[key] = np.maximum.accumulate(np.where(reset, np.arange(len(f)), 0))
        return self._segments[key]

    def batch(self, root_indices, *, family, length, cadence_seconds, scaler, context_mode="legacy"):
        ids = np.asarray(root_indices)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= len(self.frame))):
            raise ValueError("Valid integer origin indices required")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Positive integer window length required")
        if family not in self._columns or scaler["feature_names"] != self.feature_names:
            raise ValueError("Family or shared scaler schema mismatch")
        columns = self._columns[family]
        idx = ids[:, None] - np.arange(length - 1, -1, -1)
        starts = self.segment_starts(cadence_seconds, context_mode)
        safe = np.maximum(idx, 0)
        support = self.frame.physical_support_available.to_numpy()
        clock = (idx >= starts[ids, None]) & (idx >= 0) & support[safe] & support[ids, None]
        values = self.raw[safe[..., None], columns]
        observed = np.isfinite(values) & self.frame.feature_available.to_numpy()[safe, None] & clock[..., None]
        mean = np.asarray(scaler["mean"], np.float32)[columns]
        scale = np.asarray(scaler["scale"], np.float32)[columns]
        scaled = np.where(observed, (values - mean) / scale, 0.)
        scaled = np.clip(scaled, -scaler["cap"], scaler["cap"]).astype(np.float32)
        result = np.concatenate((scaled, observed.astype(np.float32), clock[..., None].astype(np.float32)), axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite lazy batch after normalization")
        return result


def load_dataset(root):
    """Reuse v2 COMPLETE/payload checks, then require the explicit v3 contract."""
    old = v2.load_dataset(root)
    return FeatureStore(old.frame, old.raw, old.feature_names, old.groups, source_receipt=old.source_receipt)


def _validate_config(store, config):
    if config.get("schema") != "scientific_opportunity_pipeline_v3":
        raise ValueError("Unsupported v3 configuration schema")
    if config["data"].get("context_mode") not in {"legacy", "restored"}:
        raise ValueError("Unsupported context mode")
    if config["training"].get("weighting") not in {"uniform", "crossing_id"}:
        raise ValueError("Unsupported weighting scheme")
    if config["loss"].get("formulation", "factorized") != "factorized" or config["loss"].get("event_loss", "bce") != "bce":
        raise ValueError("V3 experiment fixes factorized event BCE plus conditional CE")
    old = copy.deepcopy(config)
    old["schema"] = "scientific_opportunity_pipeline_v2"
    old["training"]["weighting"] = "uniform_roots"
    v2._validate_config(store, old)


def _batch(store, ids, config, scaler):
    data = config["data"]
    return store.batch(ids, family=data["family"], length=data["sequence_length"],
                       cadence_seconds=data["cadence_seconds"], scaler=scaler, context_mode=data["context_mode"])


@torch.no_grad()
def predict(store, model, config, scaler, root_indices=None, *, return_logits=False):
    model.eval()
    ids = np.arange(len(store.frame)) if root_indices is None else np.asarray(root_indices)
    device = next(model.parameters()).device
    p, z = [], []
    for start in range(0, len(ids), config["training"]["batch_size"]):
        batch_ids = ids[start:start + config["training"]["batch_size"]]
        logits = model(torch.as_tensor(_batch(store, batch_ids, config, scaler), device=device))
        probabilities = joint_probabilities(logits, config["loss"].get("formulation", "factorized"))
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite prediction")
        p.append(probabilities.cpu().numpy())
        if return_logits:
            z.append(logits.cpu().numpy())
    probabilities = np.concatenate(p) if p else np.empty((0, 4), np.float32)
    if return_logits:
        return probabilities, np.concatenate(z) if z else np.empty((0, 4), np.float32)
    return probabilities


def evaluate(store, model, config, scaler, root_indices):
    """Always unweighted root metrics; no future identity lookup."""
    ids = np.asarray(root_indices)
    return metrics(store.frame.y_crossing.to_numpy()[ids], predict(store, model, config, scaler, ids))


def _array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + "\0" + str(value.dtype) + "\0" + str(tuple(value.shape)) + "\0").encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _weight_summary(weights):
    w = np.asarray(weights, np.float64)
    total = float(w.sum())
    squares = float(np.square(w).sum())
    return dict(count=len(w), mass=total, mean=float(w.mean()) if len(w) else None,
                minimum=float(w.min()) if len(w) else None,
                q50=float(np.quantile(w, .5)) if len(w) else None,
                q90=float(np.quantile(w, .9)) if len(w) else None,
                q99=float(np.quantile(w, .99)) if len(w) else None,
                maximum=float(w.max()) if len(w) else None,
                kish_effective_sample_size=total * total / squares if squares else 0.)


def compute_train_weights(store, train_ids, scheme="uniform"):
    """Return full-clock weights (zero outside TRAIN) and a compact receipt.

    Multiplicity counts every used positive (TRAIN root, side) membership,
    including BOTH. The counter is common across sides. NONE raw weight is 1;
    one-sided roots use inverse multiplicity and BOTH averages its two inverse
    multiplicities. Each observed joint class is normalized to its original
    root count. This preserves class masses, not exact per-event equality.
    """
    if scheme not in {"uniform", "crossing_id"}:
        raise ValueError("Unsupported weighting scheme")
    frame = store.frame
    ids = np.asarray(train_ids)
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
    inverse = np.zeros((len(ids), 2), np.float64)
    for side, (mask, values) in side_ids.items():
        inverse[mask, side - 1] = [1. / multiplicity[v] for v in values]
    raw = np.ones(len(ids), np.float64)
    if scheme == "crossing_id":
        raw[y == 1] = inverse[y == 1, 0]
        raw[y == 2] = inverse[y == 2, 1]
        raw[y == 3] = inverse[y == 3].mean(axis=1)
    weights = raw.copy()
    normalizers = {}
    classes = {}
    for value, name in enumerate(STATE_NAMES):
        mask = y == value
        normalizer = float(mask.sum() / raw[mask].sum()) if mask.any() else None
        if mask.any():
            weights[mask] *= normalizer
        normalizers[name] = normalizer
        classes[name] = dict(original_root_count=int(mask.sum()), raw=_weight_summary(raw[mask]),
                             normalized=_weight_summary(weights[mask]))
    full = np.zeros(len(frame), np.float64)
    full[ids] = weights
    counts = np.asarray(list(multiplicity.values()), dtype=np.int64)
    count_digest = hashlib.sha256(json.dumps(sorted(multiplicity.items()), separators=(",", ":")).encode()).hexdigest()
    receipt = dict(scheme=scheme, fitted_on="known feature-available TRAIN roots only",
        train_root_indices_sha256=_array_sha(ids, "<i8"), train_root_count=len(ids),
        train_weights_sha256=_array_sha(weights, "<f8"),
        full_clock_weight_vector_sha256=_array_sha(full, "<f8"),
        raw_formula="NONE=1; L=1/m(idL); S=1/m(idS); BOTH=mean(1/m(idL),1/m(idS))",
        multiplicity_definition="All positive known-TRAIN (root,side) memberships pooled across sides, including BOTH",
        normalization="Within each joint class, sum(weights)=original root count",
        class_normalizers=normalizers, by_joint_class=classes, overall=_weight_summary(weights),
        unique_positive_crossing_ids=len(multiplicity), positive_side_memberships=len(memberships),
        crossing_id_count_table_sha256=count_digest,
        id_multiplicity=dict(minimum=int(counts.min()) if len(counts) else None,
                             q50=float(np.quantile(counts, .5)) if len(counts) else None,
                             q90=float(np.quantile(counts, .9)) if len(counts) else None,
                             maximum=int(counts.max()) if len(counts) else None),
        loss_denominator="Number of known roots in batch; not sum of weights",
        caveats=["Class normalization preserves class masses but does not exactly equalize distinct crossing IDs.",
                 "Weights change the TRAIN optimization measure; weighted BCE is not assumed calibrated to uniform roots.",
                 "No validation, calibration, or test identity contributes to a weight; all evaluation metrics are unweighted."])
    return full, receipt


def weighted_classification_loss(logits, labels, weights, loss_config, reduction="mean"):
    """Actual weighted per-root loss with the original known-root denominator."""
    if loss_config.get("formulation", "factorized") != "factorized" or loss_config.get("event_loss", "bce") != "bce":
        raise ValueError("V3 experiment fixes factorized event BCE plus conditional CE")
    y = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    w = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if w.shape != y.shape or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("Finite nonnegative per-root weights matching labels required")
    if (w[y >= 0] <= 0).any():
        raise ValueError("Known optimization roots require positive weights")
    per_root = classification_loss(logits, y, loss_config, reduction="none") * w
    if reduction == "none":
        return per_root
    if reduction == "sum":
        return per_root.sum()
    if reduction == "mean":
        return per_root.sum() / (y >= 0).sum().clamp_min(1)
    raise ValueError("Unsupported loss reduction")


def logit_gradients(logits, labels, loss_config, weights=None):
    """Autograd derivatives and analytic BCE/CE parity, before batch reduction."""
    z = torch.as_tensor(logits).detach().clone().requires_grad_(True)
    y = torch.as_tensor(labels, dtype=torch.long, device=z.device)
    w = torch.ones(len(y), dtype=z.dtype, device=z.device) if weights is None else torch.as_tensor(weights, dtype=z.dtype, device=z.device)
    losses = weighted_classification_loss(z, y, w, loss_config, reduction="none")
    gradient, = torch.autograd.grad(losses.sum(), z)
    expected = torch.zeros_like(z)
    expected[:, 0] = z[:, 0].sigmoid() - (y > 0).to(z.dtype)
    expected[:, 1:] = (z[:, 1:].softmax(1) - F.one_hot((y - 1).clamp_min(0), 3).to(z.dtype)) * (y > 0)[:, None]
    expected *= w[:, None]
    expected[y < 0] = 0.
    error = float((expected - gradient).detach().abs().max()) if len(y) else 0.
    if not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite loss-logit gradient")
    return gradient.detach().cpu().numpy(), error


def _gradient_groups(y, p, gradient):
    """Retain joint bins and add distinct E and conditional D difficulty bins."""
    groups = v2._gradient_groups(y, p, gradient)
    known, positive = y >= 0, y > 0
    event_probability = p[:, 1:].sum(axis=1)
    p_true_e = np.where(positive, event_probability, p[:, 0])
    p_true_d = np.full(len(y), np.nan)
    p_true_d[positive] = p[np.flatnonzero(positive), y[positive]] / np.maximum(event_probability[positive], 1e-15)
    event_abs, direction_l2 = np.abs(gradient[:, 0]), np.linalg.norm(gradient[:, 1:], axis=1)
    for head, probability, eligible, magnitude in (("event", p_true_e, known, event_abs),
                                                   ("conditional_direction", p_true_d, positive, direction_l2)):
        bins = {}
        total = float(magnitude[eligible].sum())
        for name, low, high in (("hard", 0., .4), ("intermediate", .4, .8), ("easy", .8, np.inf)):
            mask = eligible & (probability >= low) & (probability < high)
            bins[name] = dict(count=int(mask.sum()), fraction_eligible=float(mask.sum() / max(1, eligible.sum())),
                absolute_gradient_mass=float(magnitude[mask].sum()),
                gradient_mass_fraction=float(magnitude[mask].sum() / total) if total else 0.,
                mean_true_head_probability=float(probability[mask].mean()) if mask.any() else None)
        groups[head + "_difficulty"] = dict(eligible_roots=int(eligible.sum()), bins=bins)
    return groups


def gradient_diagnostic(store, model, config, scaler, root_indices, train_weights):
    """Same fixed TRAIN probe, actual unweighted and weighted encoder losses."""
    model.eval()
    ids = np.asarray(root_indices)
    device = next(model.parameters()).device
    x = torch.as_tensor(_batch(store, ids, config, scaler), device=device)
    y = torch.as_tensor(store.frame.y_crossing.to_numpy()[ids], dtype=torch.long, device=device)
    weights = torch.as_tensor(train_weights[ids], dtype=x.dtype, device=device)
    z = model(x)
    params = [p for name, p in model.named_parameters() if not name.startswith("output.")]
    known = (y >= 0).sum().clamp_min(1)
    ce = F.cross_entropy(z[:, 1:], (y - 1).clamp_min(0), reduction="none") * (y > 0)

    def gradvec(loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)])

    yy, zz = y.detach().cpu().numpy(), z.detach().cpu().numpy()
    probability = joint_probabilities(z).detach().cpu().numpy()
    result = {}
    for name, w in (("unweighted", torch.ones_like(weights)), ("weighted", weights)):
        total = weighted_classification_loss(z, y, w, config["loss"])
        direction = (ce * w).sum() / known
        ge, gd = gradvec(total - direction), gradvec(direction)
        ne, nd = float(ge.norm()), float(gd.norm())
        if not torch.isfinite(ge).all() or not torch.isfinite(gd).all():
            raise ValueError("Nonfinite shared encoder gradient")
        actual, error = logit_gradients(zz, yy, config["loss"], w.detach().cpu().numpy())
        result[name] = dict(event_encoder_gradient_l2=ne, direction_encoder_gradient_l2=nd,
            cosine_event_direction=float(torch.dot(ge, gd) / (ge.norm() * gd.norm())) if ne > 1e-12 and nd > 1e-12 else None,
            loss=float(total.detach()), analytic_bce_max_abs_error=error, groups=_gradient_groups(yy, probability, actual))
    return dict(root_indices=ids.tolist(), root_indices_sha256=_array_sha(ids, "<i8"), n=len(ids),
        train_probe_weights=_weight_summary(weights.cpu().numpy()),
        denominator="Common known-root count for weighted and unweighted losses; not weight sum",
        shared_encoder_definition="Output layer excluded; event BCE and conditional CE gradients measured separately",
        all_finite=True, **result)


def context_diagnostic(store, config, splits, receptive_field):
    """Available clock span and observed X are distinct, with no model fitting."""
    length = min(config["data"]["sequence_length"], receptive_field or config["data"]["sequence_length"])
    cadence, mode = config["data"]["cadence_seconds"], config["data"]["context_mode"]
    starts = store.segment_starts(cadence, mode)
    available = store.frame.feature_available.to_numpy()
    support = store.frame.physical_support_available.to_numpy()

    def stats(a):
        return dict(n=len(a), mean=float(a.mean()), q10=float(np.quantile(a, .1)),
                    q50=float(np.quantile(a, .5)), q90=float(np.quantile(a, .9)))

    by_split = {}
    for role, ids in splits.items():
        records = []
        for start in range(0, len(ids), 8192):
            batch_ids = ids[start:start + 8192]
            idx = batch_ids[:, None] - np.arange(length - 1, -1, -1)
            safe = np.maximum(idx, 0)
            clock = (idx >= starts[batch_ids, None]) & (idx >= 0) & support[safe] & support[batch_ids, None]
            observed = clock & available[safe]
            oldest_age = np.max(np.where(observed, batch_ids[:, None] - idx, 0), axis=1) * cadence
            records.append(np.column_stack((clock.sum(1), observed.sum(1), oldest_age)))
        a = np.concatenate(records)
        by_split[role] = dict(known_feature_roots=len(ids), supported_clock_rows=stats(a[:, 0]),
            observed_feature_rows=stats(a[:, 1]), oldest_observed_feature_span_seconds=stats(a[:, 2]),
            full_supported_clock_receptive_field_fraction=float((a[:, 0] == length).mean()),
            fully_observed_receptive_field_fraction=float((a[:, 1] == length).mean()))
    return dict(context_mode=mode, effective_receptive_field=length, maximum_span_seconds=(length - 1) * cadence,
        clock_flag_definition="Supported historical clock position, including missing X; zero for padding or physical non-support",
        missing_x_definition="Values and feature masks zero; no forward fill and no added age channels",
        by_split=by_split)


def _prediction_frame(store, probability, logits, config, train_weights):
    columns = ["date", "t_ns", "feature_available", "legacy_segment_token", "physical_segment_token",
               "physical_support_available", "y_crossing"]
    columns += [k for k in ("source_endpoint_ns", "source_available_ns", "source_age_seconds") if k in store.frame]
    frame = store.frame[columns].copy()
    frame.insert(0, "root_index", np.arange(len(frame), dtype=np.int64))
    frame["split"] = [next(role for role, days in config["splits"].items() if date in days) for date in frame.date]
    mode = config["data"]["context_mode"]
    frame["segment_token"] = frame["legacy_segment_token" if mode == "legacy" else "physical_segment_token"]
    for j, name in enumerate(("none", "long", "short", "both")):
        frame["p_" + name] = probability[:, j]
    y = frame.y_crossing.to_numpy()
    gradient = np.empty_like(logits)
    errors = []
    for start in range(0, len(y), 65536):
        g, error = logit_gradients(logits[start:start + 65536], y[start:start + 65536], config["loss"])
        gradient[start:start + 65536] = g
        errors.append(error)
    frame["g_event"] = np.where(y >= 0, gradient[:, 0], np.nan)
    frame["g_direction_norm"] = np.where(y >= 0, np.linalg.norm(gradient[:, 1:], axis=1), np.nan)
    train = train_weights > 0
    frame["optimization_weight_train_only"] = np.where(train, train_weights, np.nan)
    frame["g_event_training_weighted"] = np.where(train, train_weights * gradient[:, 0], np.nan)
    frame["g_direction_norm_training_weighted"] = np.where(train, train_weights * np.linalg.norm(gradient[:, 1:], axis=1), np.nan)
    frame["z_event"] = logits[:, 0]
    target = {"event": (y > 0).astype(float), "long": np.isin(y, [1, 3]).astype(float), "short": np.isin(y, [2, 3]).astype(float)}
    scores = {"event": 1 - probability[:, 0], "long": probability[:, 1] + probability[:, 3], "short": probability[:, 2] + probability[:, 3]}
    # Common physical adjacency keeps temporal comparisons matched across the
    # two context modes. Context-specific adjacency is reported separately.
    ids = np.arange(len(frame))
    physical_starts = store.segment_starts(config["data"]["cadence_seconds"], "restored")
    adjacent = np.zeros(len(frame), bool)
    adjacent[1:] = ((physical_starts[1:] < ids[1:]) & frame.feature_available.to_numpy()[1:] &
                   frame.feature_available.to_numpy()[:-1] & (y[1:] >= 0) & (y[:-1] >= 0))
    mode_starts = store.segment_starts(config["data"]["cadence_seconds"], mode)
    frame["context_adjacent_previous"] = mode_starts < ids
    frame["common_physical_adjacent_known"] = adjacent
    temporal = {}
    for name in target:
        frame["target_" + name] = np.where(y >= 0, target[name], np.nan)
        frame["p_" + name + "_marginal"] = scores[name]
        delta_y, delta_p = np.full(len(y), np.nan), np.full(len(y), np.nan)
        delta_y[1:], delta_p[1:] = np.diff(target[name]), np.diff(scores[name])
        delta_y[~adjacent], delta_p[~adjacent] = np.nan, np.nan
        frame["delta_target_" + name], frame["delta_probability_" + name] = delta_y, delta_p
        temporal[name] = {}
        for role in config["splits"]:
            mask = adjacent & frame["split"].eq(role).to_numpy()
            temporal[name][role] = dict(adjacent_known_pairs=int(mask.sum()),
                target_flip_fraction=float(np.abs(delta_y[mask]).mean()) if mask.any() else None,
                probability_mean_absolute_step=float(np.abs(delta_p[mask]).mean()) if mask.any() else None,
                cadence_seconds=config["data"]["cadence_seconds"])
    groups = {role: _gradient_groups(y[m], probability[m], gradient[m]) for role in config["splits"]
              for m in [frame["split"].eq(role).to_numpy() & frame.feature_available.to_numpy()]}
    weighted = gradient[train] * train_weights[train, None]
    artifact = dict(per_root_gradient_definition="Unreduced actual BCE+conditional-CE derivatives; NaN for unknown saved labels",
        weighted_gradient_definition="TRAIN-only fixed identity weight times the unreduced derivative; no weighted VAL/CAL/TEST score",
        temporal_definition="Binary E/L/S changes on common physical adjacent known admitted roots; identical pair cohort across context modes",
        analytic_bce_max_abs_error=max(errors), full_clock_groups=groups,
        weighted_train_groups=_gradient_groups(y[train], probability[train], weighted), temporal=temporal)
    return frame, artifact


def train_run(store, config, out):
    """Fit one matched arm; caller binds sources and completes the run receipt."""
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
    train_weights, weight_receipt = compute_train_weights(store, split["train"], tr["weighting"])
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
        print(json.dumps({**row, "stage": "fit_v3", "run": config["run_id"]}), flush=True)
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
