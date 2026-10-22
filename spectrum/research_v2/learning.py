"""Lazy, source-bound crossing classification on a common chronological cohort.

Only N x F source values and batch-sized B x T x F windows are materialized.
Labels never define window resets or scaler admission. Economic replay is a
separate consumer of the saved full-clock predictions.
"""
from __future__ import annotations

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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


class FeatureStore:
    """In-memory N x F values with lazy family projection and causal windows."""

    def __init__(self, frame, raw, feature_names, groups, *, source_receipt=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.raw = np.asarray(raw)
        self.feature_names = list(feature_names)
        self.groups = {k: list(v) for k, v in groups.items()}
        self.source_receipt = source_receipt or {}
        required = {"date", "t_ns", "feature_available", "segment_token", "y_crossing"}
        if not required <= set(self.frame):
            raise ValueError("Missing origin columns: " + str(sorted(required - set(self.frame))))
        if self.raw.dtype != np.float32 or self.raw.ndim != 2 or self.raw.shape != (len(frame), len(feature_names)):
            raise ValueError("Expected aligned raw float32[N,F] feature matrix")
        if not len(frame) or len(set(feature_names)) != len(feature_names):
            raise ValueError("Nonempty rows and unique feature names required")
        if self.frame.feature_available.dtype != bool:
            raise ValueError("feature_available must be boolean")
        if not np.issubdtype(self.frame.t_ns.dtype, np.integer) or not np.issubdtype(self.frame.segment_token.dtype, np.integer):
            raise ValueError("Integer nanosecond clock and support segment required")
        if not self.frame.t_ns.is_monotonic_increasing or self.frame.t_ns.duplicated().any():
            raise ValueError("Strict chronological unique origins required")
        if not np.issubdtype(self.frame.y_crossing.dtype, np.integer) or not self.frame.y_crossing.isin([-1, 0, 1, 2, 3]).all():
            raise ValueError("y_crossing must be integer UNKNOWN/NONE/LONG/SHORT/BOTH")
        for family, names in self.groups.items():
            if not names or len(names) != len(set(names)) or not set(names) <= set(feature_names):
                raise ValueError("Invalid feature family: " + family)
        self._columns = {k: np.array([self.feature_names.index(n) for n in names], dtype=np.int64)
                         for k, names in self.groups.items()}
        self._segments = {}

    def segment_starts(self, cadence_seconds):
        """O(N) starts; support/date/missing-X resets, never unknown labels."""
        step = int(round(float(cadence_seconds) * 1e9))
        if not np.isfinite(cadence_seconds) or step <= 0:
            raise ValueError("Positive finite cadence required")
        if step not in self._segments:
            f = self.frame
            t, date = f.t_ns.to_numpy(), f.date.to_numpy()
            ok, token = f.feature_available.to_numpy(), f.segment_token.to_numpy()
            reset = np.ones(len(f), dtype=bool)
            reset[1:] = ((np.diff(t) != step) | (date[1:] != date[:-1]) |
                         (token[1:] != token[:-1]) | ~ok[1:] | ~ok[:-1])
            self._segments[step] = np.maximum.accumulate(np.where(reset, np.arange(len(f)), 0))
        return self._segments[step]

    def batch(self, root_indices, *, family, length, cadence_seconds, scaler):
        """Build only [B,T,2F+1], including observed-feature and real-row flags."""
        ids = np.asarray(root_indices)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= len(self.frame)):
            raise ValueError("Valid integer origin indices required")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Positive integer window length required")
        if family not in self._columns or scaler["feature_names"] != self.feature_names:
            raise ValueError("Family or shared scaler schema mismatch")
        columns = self._columns[family]
        idx = ids[:, None] - np.arange(length - 1, -1, -1)
        starts = self.segment_starts(cadence_seconds)
        real = (idx >= starts[ids, None]) & (idx >= 0) & self.frame.feature_available.to_numpy()[ids, None]
        safe = np.maximum(idx, 0)
        values = self.raw[safe[..., None], columns]
        finite = np.isfinite(values)
        mean = np.asarray(scaler["mean"], np.float32)[columns]
        scale = np.asarray(scaler["scale"], np.float32)[columns]
        scaled = np.where(finite, (values - mean) / scale, 0.)
        scaled = np.clip(scaled, -scaler["cap"], scaler["cap"]).astype(np.float32)
        result = np.concatenate((scaled, finite.astype(np.float32), real[..., None].astype(np.float32)), axis=-1)
        result[~real] = 0.
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite lazy batch after normalization")
        return result


def load_dataset(root):
    """Verify manifest and every declared daily payload before joining N x F."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Dataset manifest is not COMPLETE")
    names, groups, days = manifest["feature_names"], manifest["groups"], manifest["days"]
    dates = [d["date"] for d in days]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Unique chronological manifest dates required")
    frames, arrays = [], []
    hashes = {str(manifest_path.resolve()): sha(manifest_path)}
    for day in days:
        for key in ("roots_file", "x_file"):
            if day[key] not in day["files"]:
                raise ValueError("Daily consumed payload missing SHA: " + day[key])
        for filename, expected in day["files"].items():
            path = root / filename
            if path.resolve().parent != root.resolve():
                raise ValueError("Daily files must be direct dataset children")
            observed = sha(path)
            if observed != expected:
                raise ValueError("Dataset payload SHA mismatch: " + str(path))
            hashes[str(path.resolve())] = observed
        frame = pd.read_parquet(root / day["roots_file"])
        if not frame.date.eq(day["date"]).all():
            raise ValueError("Daily root date mismatch")
        values = np.load(root / day["x_file"], mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (len(frame), len(names)):
            raise ValueError("Daily X shape/dtype mismatch")
        frames.append(frame)
        arrays.append(values)
    return FeatureStore(pd.concat(frames, ignore_index=True), np.concatenate(arrays, axis=0), names, groups,
                        source_receipt=dict(manifest_sha256=hashes[str(manifest_path.resolve())],
                                            source_sha256=hashes, dates=dates,
                                            memory_layout="Only N x F values; B x T x F windows built per batch"))


def fit_scaler(store, train_rows, chunk_size=8192):
    """One shared full-schema scaler on causal TRAIN rows, independent of Y.

    Stable merged population moments avoid retaining a float64 copy of N x F.
    All feature families slice these exact same means, scales and cap.
    """
    ids = np.unique(np.asarray(train_rows, dtype=np.int64))
    if not len(ids) or np.any(ids < 0) or np.any(ids >= len(store.frame)):
        raise ValueError("Nonempty valid scaler training roots required")
    ids = ids[store.frame.feature_available.to_numpy()[ids]]
    if not len(ids):
        raise ValueError("No causal TRAIN feature rows")
    size = len(store.feature_names)
    count, mean, m2 = np.zeros(size, np.int64), np.zeros(size), np.zeros(size)
    for start in range(0, len(ids), chunk_size):
        x = store.raw[ids[start:start + chunk_size]].astype(np.float64)
        good = np.isfinite(x)
        n = good.sum(axis=0)
        batch_mean = np.divide(np.where(good, x, 0.).sum(axis=0), n, out=np.zeros(size), where=n > 0)
        batch_m2 = np.where(good, x - batch_mean, 0.) ** 2
        batch_m2 = batch_m2.sum(axis=0)
        total, delta = count + n, batch_mean - mean
        mean += np.divide(delta * n, total, out=np.zeros(size), where=total > 0)
        m2 += batch_m2 + np.divide(delta ** 2 * count * n, total, out=np.zeros(size), where=total > 0)
        count = total
    std = np.sqrt(np.divide(m2, count, out=np.zeros(size), where=count > 0))
    scale = np.where(std > 1e-8, std, 1.)
    return dict(feature_names=store.feature_names, mean=mean.tolist(), scale=scale.tolist(), cap=10.,
                finite_count=count.tolist(), all_missing_columns=np.flatnonzero(count == 0).tolist(),
                constant_columns=np.flatnonzero((count > 0) & (std <= 1e-8)).tolist(),
                train_feature_row_count=len(ids), train_feature_indices_sha256=_ids_sha(ids),
                method="all unique causal TRAIN rows; label-independent; merged population moments; cap10")


def split_indices(store, config):
    """One cohort shared by every family/architecture; labels do not change X."""
    f = store.frame
    known = f.feature_available.to_numpy() & (f.y_crossing.to_numpy() >= 0)
    return {role: np.flatnonzero(f.date.isin(dates).to_numpy() & known)
            for role, dates in config["splits"].items()}


def _validate_config(store, config):
    if config.get("schema") != "scientific_opportunity_pipeline_v2":
        raise ValueError("Unsupported v2 configuration schema")
    if set(config["splits"]) != {"train", "validation", "calibration", "test"}:
        raise ValueError("Four chronological split roles required")
    dates = [d for role in ("train", "validation", "calibration", "test") for d in config["splits"][role]]
    if any(not ds for ds in config["splits"].values()) or dates != sorted(set(dates)):
        raise ValueError("Nonempty disjoint chronological splits required")
    if set(dates) != set(store.frame.date.unique()):
        raise ValueError("Splits must exactly cover loaded dates")
    if config["data"]["family"] not in store.groups:
        raise ValueError("Unsupported feature family")
    if config["loss"].get("formulation", "factorized") != "factorized":
        raise ValueError("V2 named event/direction diagnostics currently require factorized logits")
    tr = config["training"]
    for key in ("batch_size", "max_updates", "validate_every", "patience_checks", "cpu_threads", "train_metric_probe", "gradient_probe"):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError("Positive training integer required: " + key)
    for key in ("learning_rate", "gradient_clip", "weight_decay", "min_delta"):
        if not np.isfinite(tr[key]) or tr[key] < 0 or (key in ("learning_rate", "gradient_clip") and tr[key] == 0):
            raise ValueError("Invalid finite training scalar: " + key)
    if tr.get("weighting") != "uniform_roots":
        raise ValueError("Only uniform-root baseline weighting implemented")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("Nonnegative integer seed required")


def _batch(store, ids, config, scaler):
    return store.batch(ids, family=config["data"]["family"], length=config["data"]["sequence_length"],
                       cadence_seconds=config["data"]["cadence_seconds"], scaler=scaler)


@torch.no_grad()
def predict(store, model, config, scaler, root_indices=None, *, return_logits=False):
    model.eval()
    ids = np.arange(len(store.frame)) if root_indices is None else np.asarray(root_indices)
    device = next(model.parameters()).device
    batch_size = config["training"]["batch_size"]
    probability, saved_logits = [], []
    for start in range(0, len(ids), batch_size):
        x = _batch(store, ids[start:start + batch_size], config, scaler)
        z = model(torch.as_tensor(x, device=device))
        p = joint_probabilities(z, config["loss"].get("formulation", "factorized"))
        if not torch.isfinite(p).all() or not torch.isfinite(z).all():
            raise ValueError("Nonfinite prediction")
        probability.append(p.cpu().numpy())
        if return_logits:
            saved_logits.append(z.cpu().numpy())
    p = np.concatenate(probability) if probability else np.empty((0, 4), np.float32)
    if return_logits:
        return p, np.concatenate(saved_logits) if saved_logits else np.empty((0, 4), np.float32)
    return p


def evaluate(store, model, config, scaler, root_indices):
    ids = np.asarray(root_indices)
    return metrics(store.frame.y_crossing.to_numpy()[ids], predict(store, model, config, scaler, ids))


def logit_gradients(logits, labels, loss_config):
    """Actual unreduced per-root dL/dlogits; UNKNOWN is masked to exact zero."""
    z = torch.as_tensor(logits).detach().clone().requires_grad_(True)
    y = torch.as_tensor(labels, dtype=torch.long, device=z.device)
    per_root = classification_loss(z, y, loss_config, reduction="none")
    gradient, = torch.autograd.grad(per_root.sum(), z)
    error = None
    if loss_config.get("event_loss", "bce") == "bce":
        expected = torch.zeros_like(z)
        expected[:, 0] = z[:, 0].sigmoid() - (y > 0).to(z.dtype)
        q = z[:, 1:].softmax(1)
        expected[:, 1:] = (q - F.one_hot((y - 1).clamp_min(0), 3).to(z.dtype)) * (y > 0)[:, None]
        expected[y < 0] = 0.
        error = float((expected - gradient).detach().abs().max()) if len(y) else 0.
    if not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite loss-logit gradient")
    return gradient.detach().cpu().numpy(), error


def _gradient_groups(y, p, gradient):
    known = y >= 0
    true_probability = np.full(len(y), np.nan)
    true_probability[known] = p[np.flatnonzero(known), y[known]]
    masks = {"event_negative": y == 0, "event_positive": y > 0,
             "hard_p_true_lt_0.4": known & (true_probability < .4),
             "intermediate_p_true_0.4_to_0.8": known & (true_probability >= .4) & (true_probability < .8),
             "easy_p_true_ge_0.8": known & (true_probability >= .8)}
    masks.update({name: y == value for value, name in enumerate(("NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH"))})
    abs_event = np.abs(gradient[:, 0])
    direction_norm = np.linalg.norm(gradient[:, 1:], axis=1)
    mass_e, mass_d = abs_event[known].sum(), direction_norm[known].sum()
    out = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        out[name] = dict(count=n, fraction_known=n / max(1, int(known.sum())),
                         mean_true_joint_probability=float(true_probability[mask].mean()) if n else None,
                         mean_abs_unreduced_event_gradient=float(abs_event[mask].mean()) if n else None,
                         event_absolute_gradient_mass_fraction=float(abs_event[mask].sum() / mass_e) if mass_e else 0.,
                         mean_unreduced_direction_gradient_l2=float(direction_norm[mask].mean()) if n else None,
                         direction_l2_mass_fraction=float(direction_norm[mask].sum() / mass_d) if mass_d else 0.)
    return out


def gradient_diagnostic(store, model, config, scaler, root_indices):
    """Actual selected-loss encoder gradients on frozen TRAIN root IDs."""
    model.eval()
    ids = np.asarray(root_indices)
    device = next(model.parameters()).device
    x = torch.as_tensor(_batch(store, ids, config, scaler), device=device)
    y = torch.as_tensor(store.frame.y_crossing.to_numpy()[ids], dtype=torch.long, device=device)
    z = model(x)
    total = classification_loss(z, y, config["loss"])
    known_count = (y >= 0).sum().clamp_min(1)
    direction = (F.cross_entropy(z[:, 1:], (y - 1).clamp_min(0), reduction="none") * (y > 0)).sum() / known_count
    event = total - direction
    params = [v for name, v in model.named_parameters() if not name.startswith("output.")]
    def gradvec(loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(v) if g is None else g).flatten() for v, g in zip(params, grads)])
    ge, gd = gradvec(event), gradvec(direction)
    ne, nd = float(ge.norm()), float(gd.norm())
    cosine = float(torch.dot(ge, gd) / (ge.norm() * gd.norm())) if ne > 1e-12 and nd > 1e-12 else None
    if not torch.isfinite(ge).all() or not torch.isfinite(gd).all():
        raise ValueError("Nonfinite shared-encoder gradient")
    zz = z.detach().cpu().numpy()
    yy = y.detach().cpu().numpy()
    p = joint_probabilities(z).detach().cpu().numpy()
    actual, error = logit_gradients(zz, yy, config["loss"])
    return dict(root_indices=ids.tolist(), root_indices_sha256=_ids_sha(ids), n=len(ids),
                event_encoder_gradient_l2=ne, direction_encoder_gradient_l2=nd,
                cosine_event_direction=cosine, loss=float(total.detach()),
                loss_gradient_semantics="Actual configured event loss vs conditional CE; shared known-root denominator; output layer excluded",
                analytic_bce_max_abs_error=error, groups=_gradient_groups(yy, p, actual),
                all_finite=True)


def _prediction_frame(store, probability, logits, config):
    frame = store.frame[["date", "t_ns", "feature_available", "segment_token", "y_crossing"]].copy()
    frame.insert(0, "root_index", np.arange(len(frame), dtype=np.int64))
    frame["split"] = [next(role for role, days in config["splits"].items() if date in days) for date in frame.date]
    for j, name in enumerate(("none", "long", "short", "both")):
        frame["p_" + name] = probability[:, j]
    y = frame.y_crossing.to_numpy()
    all_gradient = np.empty_like(logits)
    errors = []
    for start in range(0, len(y), 65536):
        chunk, error = logit_gradients(logits[start:start + 65536], y[start:start + 65536], config["loss"])
        all_gradient[start:start + 65536] = chunk
        if error is not None:
            errors.append(error)
    frame["g_event"] = np.where(y >= 0, all_gradient[:, 0], np.nan)
    frame["g_direction_norm"] = np.where(y >= 0, np.linalg.norm(all_gradient[:, 1:], axis=1), np.nan)
    frame["z_event"] = logits[:, 0]
    target = {"event": (y > 0).astype(float), "long": np.isin(y, [1, 3]).astype(float),
              "short": np.isin(y, [2, 3]).astype(float)}
    scores = {"event": 1 - probability[:, 0], "long": probability[:, 1] + probability[:, 3],
              "short": probability[:, 2] + probability[:, 3]}
    step = int(round(config["data"]["cadence_seconds"] * 1e9))
    contiguous = np.zeros(len(frame), bool)
    contiguous[1:] = ((np.diff(frame.t_ns.to_numpy()) == step) &
                      (frame.date.to_numpy()[1:] == frame.date.to_numpy()[:-1]) &
                      (frame.segment_token.to_numpy()[1:] == frame.segment_token.to_numpy()[:-1]) &
                      frame.feature_available.to_numpy()[1:] & frame.feature_available.to_numpy()[:-1] &
                      (y[1:] >= 0) & (y[:-1] >= 0))
    temporal = {}
    for name in target:
        observed = target[name].copy()
        observed[y < 0] = np.nan
        frame["target_" + name] = observed
        frame["p_" + name + "_marginal"] = scores[name]
        delta_y, delta_p = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        delta_y[1:] = np.diff(target[name])
        delta_p[1:] = np.diff(scores[name])
        delta_y[~contiguous] = np.nan
        delta_p[~contiguous] = np.nan
        frame["delta_target_" + name] = delta_y
        frame["delta_probability_" + name] = delta_p
        temporal[name] = {}
        for role in config["splits"]:
            mask = contiguous & frame["split"].eq(role).to_numpy()
            temporal[name][role] = dict(adjacent_known_pairs=int(mask.sum()),
                target_flip_fraction=float(np.abs(delta_y[mask]).mean()) if mask.any() else None,
                probability_mean_absolute_step=float(np.abs(delta_p[mask]).mean()) if mask.any() else None,
                cadence_seconds=config["data"]["cadence_seconds"])
    groups = {role: _gradient_groups(y[mask], probability[mask], all_gradient[mask])
              for role in config["splits"]
              for mask in [frame["split"].eq(role).to_numpy() & frame.feature_available.to_numpy()]}
    artifact = dict(per_root_gradient_definition="Unreduced actual dL_i/dz_i; stored NaN for unknown Y although the masked optimization gradient is zero",
        direction_gradient_definition="L2 norm across three conditional direction logits; zero for NONE",
        temporal_definition="Differences of E/L/S indicators and corresponding probabilities on same-support adjacent known roots; not gradients of a categorical code",
        distinction="Temporal target changes, dloss/dlogit, and shared-encoder gradients are separate quantities",
        analytic_bce_max_abs_error=max(errors) if errors else None, full_clock_groups=groups, temporal=temporal)
    return frame, artifact


def train_run(store, config, out):
    """Train one frozen experiment and save artifacts; caller finalizes receipt.

    Returns dict(summary, predictions, history, gradient_artifact). Checkpoint
    selection sees only validation. Predictions retain all roots, including
    unknown labels and missing X, so downstream account replay can fail closed.
    """
    _validate_config(store, config)
    tr = config["training"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    owned = ("checkpoint.pt", "history.json", "scaler.json", "predictions.parquet", "summary.json", "gradient_artifact.json")
    if any((out / name).exists() for name in owned):
        raise ValueError("Training outputs already exist; use a new immutable run directory")
    if (out / "config.json").exists():
        if _json(out / "config.json") != config:
            raise ValueError("Existing run configuration differs from requested training")
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
    train_feature_rows = np.flatnonzero(store.frame.date.isin(config["splits"]["train"]).to_numpy())
    scaler = fit_scaler(store, train_feature_rows)
    _save(out / "scaler.json", scaler)
    cols = store.groups[config["data"]["family"]]
    model = build_model(2 * len(cols) + 1, config["model"]).to(device)
    # Validate lazy window/scaler compatibility before the first optimization.
    _batch(store, split["train"][:1], config, scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    probe_rng = np.random.default_rng(config["seed"] + 410)
    train_probe = np.sort(probe_rng.choice(split["train"], min(tr["train_metric_probe"], len(split["train"])), replace=False))
    gradient_probe = np.sort(np.random.default_rng(config["seed"] + 411).choice(
        split["train"], min(tr["gradient_probe"], len(split["train"])), replace=False))
    y = store.frame.y_crossing.to_numpy(np.int64)
    prior = (np.bincount(y[split["train"]], minlength=4) + .5) / (len(split["train"]) + 2.)
    gradient_before = gradient_diagnostic(store, model, config, scaler, gradient_probe)
    first_validation = evaluate(store, model, config, scaler, split["validation"])
    first_train = evaluate(store, model, config, scaler, train_probe)
    history = [dict(update=0, examples_seen=0, cycles_completed=0, optimization_loss=None,
                    train_probe_nll=first_train["joint_nll"], validation_nll=first_validation["joint_nll"],
                    train_probe_accuracy=first_train["accuracy"], validation_accuracy=first_validation["accuracy"],
                    gradient_norm_mean=None, gradient_norm_max=None, gradient_clip_fraction=None,
                    stage="untrained_network_diagnostic_and_checkpoint_candidate")]
    best, best_update, stale = first_validation["joint_nll"], 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(config["seed"])
    order, offset, cycles, examples = rng.permutation(split["train"]), 0, 0, 0
    interval_losses, interval_norms = [], []
    started = time.perf_counter()
    for update in range(1, tr["max_updates"] + 1):
        if offset == len(order):
            cycles += 1
            order, offset = rng.permutation(split["train"]), 0
        ids = order[offset:offset + tr["batch_size"]]
        offset += len(ids)
        model.train()
        xb = torch.as_tensor(_batch(store, ids, config, scaler), device=device)
        yy = torch.as_tensor(y[ids], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = classification_loss(model(xb), yy, config["loss"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True))
        optimizer.step()
        examples += len(ids)
        interval_losses.append((float(loss.detach()), len(ids)))
        interval_norms.append(norm)
        if update % tr["validate_every"] and update != tr["max_updates"]:
            continue
        validation = evaluate(store, model, config, scaler, split["validation"])
        train_metric = evaluate(store, model, config, scaler, train_probe)
        row = dict(update=update, examples_seen=examples, cycles_completed=cycles + int(offset == len(order)),
                   optimization_loss=sum(v * n for v, n in interval_losses) / sum(n for _, n in interval_losses),
                   train_probe_nll=train_metric["joint_nll"], validation_nll=validation["joint_nll"],
                   train_probe_accuracy=train_metric["accuracy"], validation_accuracy=validation["accuracy"],
                   gradient_norm_mean=float(np.mean(interval_norms)), gradient_norm_max=float(np.max(interval_norms)),
                   gradient_clip_fraction=float(np.mean(np.asarray(interval_norms) > tr["gradient_clip"])), stage="trained")
        history.append(row)
        if validation["joint_nll"] < best - tr["min_delta"]:
            best, best_update, stale = validation["joint_nll"], update, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(json.dumps({**row, "stage": "fit_v2", "run": config["run_id"]}), flush=True)
        interval_losses, interval_norms = [], []
        if stale >= tr["patience_checks"]:
            break
    model.load_state_dict(best_state)
    gradient_selected = gradient_diagnostic(store, model, config, scaler, gradient_probe)
    probability, logits = predict(store, model, config, scaler, return_logits=True)
    evaluation = {role: dict(model=metrics(y[ids], probability[ids]), prior=metrics(y[ids], np.tile(prior, (len(ids), 1))))
                  for role, ids in split.items()}
    predictions, gradient_artifact = _prediction_frame(store, probability, logits, config)
    gradient_artifact.update(before=gradient_before, selected=gradient_selected)
    cohort = {role: dict(dates=config["splits"][role], known_feature_available_roots=len(ids),
                         root_indices_sha256=_ids_sha(ids), state_counts=np.bincount(y[ids], minlength=4).tolist())
              for role, ids in split.items()}
    receptive = model.receptive_field
    effective = min(config["data"]["sequence_length"], receptive or config["data"]["sequence_length"])
    training = dict(best_update=best_update, selected_checkpoint="untrained_network" if best_update == 0 else "trained_network",
        best_validation_nll=best, updates_completed=update, examples_seen=examples,
        validation_checks=len(history) - 1, seconds=time.perf_counter() - started,
        train_metric_probe_indices=train_probe.tolist(), train_metric_probe_sha256=_ids_sha(train_probe),
        gradient_probe_sha256=_ids_sha(gradient_probe), parameter_count=sum(v.numel() for v in model.parameters()),
        device=str(device), architecture_receptive_field=receptive, supplied_tokens=config["data"]["sequence_length"],
        effective_tokens=effective, detailed_history_seconds=(effective - 1) * config["data"]["cadence_seconds"],
        gradients_before=gradient_before, gradients_selected=gradient_selected,
        selection="Only validation joint NLL; min_delta; initial untrained network is an explicit candidate; prior is a separate benchmark",
        beats_train_prior_on_validation=best < evaluation["validation"]["prior"]["joint_nll"],
        loss_weighting="uniform roots; shared known-root denominator; no predicted event gate")
    summary = dict(run_id=config["run_id"], status="TRAINING_ARTIFACTS_WRITTEN_REQUIRES_RUN_FINALIZATION",
        feature_family=config["data"]["family"], feature_names=cols, input_channels=2 * len(cols) + 1,
        training=training, evaluation=evaluation, cohorts=cohort,
        scaler_sha256=sha(out / "scaler.json"), source_receipt=store.source_receipt,
        memory=dict(raw_shape=list(store.raw.shape), raw_bytes=store.raw.nbytes,
                    materialized_full_sequence_tensor=False,
                    one_input_batch_bytes=tr["batch_size"] * config["data"]["sequence_length"] * (2 * len(cols) + 1) * 4),
        caveats=["Class labels are discontinuous observations; temporal differences are not loss derivatives.",
                 "Long London context is represented only by supplied causal aggregates; detailed sequence extent is explicit.",
                 "Calibration/test labels never choose a checkpoint; unknown future outcomes remain on the prediction clock."])
    torch.save(dict(state_dict=best_state, input_dim=2 * len(cols) + 1, model_config=config["model"],
                    loss_config=config["loss"], seed=config["seed"], best_update=best_update,
                    scaler_sha256=summary["scaler_sha256"], feature_names=cols), out / "checkpoint.pt")
    predictions.to_parquet(out / "predictions.parquet", index=False)
    _save(out / "history.json", history)
    _save(out / "gradient_artifact.json", gradient_artifact)
    _save(out / "summary.json", summary)
    return dict(summary=summary, predictions=predictions, history=history, gradient_artifact=gradient_artifact)
