"""V5 auxiliary filter; no base model or future metadata enters this learner.

The caller supplies detached causal features and hierarchical teacher labels.
Five sigmoid outputs are CONDITIONAL probabilities, not a five-class simplex.
The original crossing target and the frozen base's loss remain unchanged.
"""
from __future__ import annotations

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
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


HEAD_NAMES = ("event", "chosen_correct_given_event", "safe_cost2_given_correct",
              "safe_cost4_given_correct", "safe_cost6_given_correct")
HEAD_WEIGHTS = (1., 1., 1. / 3., 1. / 3., 1. / 3.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def _array_sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _indices(ids, n, role):
    a = np.asarray(ids)
    if (a.ndim != 1 or not len(a) or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a >= n) or len(np.unique(a)) != len(a)):
        raise ValueError(role + " requires nonempty unique in-range integer indices")
    return np.sort(a.astype(np.int64))


def validate_labels(labels):
    """Reject invalid hierarchy instead of silently training a different task."""
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Labels must be N x 5 with values -1, 0, 1")
    if np.any((y[:, 1] >= 0) & (y[:, 0] != 1)):
        raise ValueError("Chosen-side labels are conditional on event=1")
    if np.any((y[:, 2:] >= 0) & ((y[:, 0] != 1) | (y[:, 1] != 1))[:, None]):
        raise ValueError("Risk labels are conditional on event=chosen-correct=1")
    return y.astype(np.float32, copy=False)


def fit_scaler(x, train_ids):
    """Feature-only statistics over ALL admitted TRAIN rows, independent of Y.

    Missing values are mean-imputed after standardization (zero); callers must
    provide any observation masks in X. No channels or forward fill are added.
    """
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not x.shape[1] or np.isinf(x).any():
        raise ValueError("Two-dimensional finite-or-NaN features required")
    ids = _indices(train_ids, len(x), "Scaler TRAIN")
    values = x[ids].astype(np.float64)
    finite = np.isfinite(values)
    count = finite.sum(0)
    mean = np.where(finite, values, 0.).sum(0) / np.maximum(count, 1)
    variance = np.where(finite, (values - mean) ** 2, 0.).sum(0) / np.maximum(count, 1)
    std = np.sqrt(variance)
    std[(std < 1e-8) | (count == 0)] = 1.
    return dict(mean=mean.tolist(), std=std.tolist(), finite_count=count.tolist(), cap=10.,
                fitted_rows=len(ids), fitted_root_indices_sha256=_array_sha(ids, "<i8"),
                fit_rule="All supplied admitted TRAIN rows, independent of every teacher mask",
                missing_rule="Mean imputation after standardization; caller-supplied masks unchanged in schema")


def transform(x, scaler):
    x = np.asarray(x, np.float32)
    mean, std = np.asarray(scaler["mean"]), np.asarray(scaler["std"])
    if x.ndim != 2 or x.shape[1] != len(mean) or np.isinf(x).any():
        raise ValueError("Feature/scaler shape mismatch or infinite feature")
    z = (x.astype(np.float64) - mean) / std
    return np.clip(np.where(np.isfinite(z), z, 0.), -scaler["cap"], scaler["cap"]).astype(np.float32)


class FilterMLP(nn.Module):
    """One small auxiliary backbone, five independent sigmoid logits."""
    def __init__(self, input_dim, hidden=32, layers=2, dropout=.05):
        super().__init__()
        blocks, width = [], input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)])
            width = hidden
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(width, 5)

    def forward(self, x):
        return self.head(self.encoder(x))


def masked_bce(logits, labels, reduction="mean"):
    """Sum masked heads per root; mean divides by E-known roots, not head count.

    `none` returns the UNREDUCED per-root objective, unknown roots exactly zero.
    Risk heads retain their fixed 1/3 coefficient even when another risk label
    is missing. There is no predicted-probability gate or per-head reweighting.
    """
    if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape != labels.shape:
        raise ValueError("Logits and labels must have matching N x 5 shape")
    known = labels >= 0
    values = F.binary_cross_entropy_with_logits(logits, labels.clamp_min(0), reduction="none")
    pieces = values * known.to(logits.dtype) * logits.new_tensor(HEAD_WEIGHTS)
    roots = pieces.sum(1)
    if reduction == "none":
        return roots
    if reduction != "mean":
        raise ValueError("Reduction must be mean or none")
    return roots.sum() / known[:, 0].sum().clamp_min(1)


def logit_gradients(logits, labels):
    """Actual unreduced dL/dz, checked against masked weighted sigmoid minus Y."""
    z = torch.tensor(np.asarray(logits), dtype=torch.float64, requires_grad=True)
    y = torch.as_tensor(validate_labels(labels), dtype=torch.float64)
    actual = torch.autograd.grad(masked_bce(z, y, "none").sum(), z)[0]
    analytical = (z.detach().sigmoid() - y.clamp_min(0)) * (y >= 0) * z.new_tensor(HEAD_WEIGHTS)
    error = float((actual - analytical).abs().max()) if len(z) else 0.
    return actual.numpy(), error


def probability_metrics(labels, probability, logits=None):
    """Per-head conditional metrics plus the exact shared-root composite NLL."""
    y, p = validate_labels(labels), np.asarray(probability, np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite N x 5 probabilities in [0,1] required")
    if logits is None:
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        losses = -(np.maximum(y, 0) * np.log(pc) + (1 - np.maximum(y, 0)) * np.log1p(-pc))
    else:
        z = np.asarray(logits, np.float64)
        if z.shape != y.shape or not np.isfinite(z).all():
            raise ValueError("Finite matching logits required")
        losses = np.logaddexp(0., z) - np.maximum(y, 0) * z
    known = y >= 0
    heads = {}
    for j, name in enumerate(HEAD_NAMES):
        use = known[:, j]
        n = int(use.sum())
        if not n:
            heads[name] = dict(n=0, positive=0, negative=0, nll=None, brier=None, auc=None,
                               accuracy=None, mean_probability=None, reliability=[])
            continue
        yy, pp = y[use, j], p[use, j]
        bins = []
        for k in range(10):
            mask = (pp >= k / 10) & ((pp < (k + 1) / 10) if k < 9 else (pp <= 1))
            bins.append(dict(lower=k / 10, upper=(k + 1) / 10, count=int(mask.sum()),
                             probability=float(pp[mask].mean()) if mask.any() else None,
                             frequency=float(yy[mask].mean()) if mask.any() else None))
        heads[name] = dict(n=n, positive=int(yy.sum()), negative=int(n - yy.sum()),
            nll=float(losses[use, j].mean()), brier=float(((pp - yy) ** 2).mean()),
            auc=float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
            accuracy=float(((pp >= .5) == yy).mean()), mean_probability=float(pp.mean()), reliability=bins)
    n_roots = int(known[:, 0].sum())
    composite = float((losses * known * np.asarray(HEAD_WEIGHTS)).sum() / n_roots) if n_roots else None
    return dict(n=n_roots, composite_nll=composite, heads=heads)


@torch.no_grad()
def predict(model, x, scaler, batch_size=2048, root_indices=None, return_logits=False):
    model.eval()
    ids = np.arange(len(x)) if root_indices is None else np.asarray(root_indices, np.int64)
    device = next(model.parameters()).device
    logits = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(transform(x[ids[start:start + batch_size]], scaler), device=device)
        logits.append(model(batch).cpu().numpy())
    z = np.concatenate(logits) if logits else np.empty((0, 5), np.float32)
    # Float64 sigmoid retains information in saturated float32 model logits.
    p = np.exp(-np.logaddexp(0., -z.astype(np.float64)))
    return (p, z) if return_logits else p


def evaluate(model, x, labels, scaler, ids, batch_size=2048):
    p, z = predict(model, x, scaler, batch_size, ids, return_logits=True)
    return probability_metrics(labels[ids], p, z)


def _gradient_groups(probability, labels, gradient):
    groups = {}
    for j, head in enumerate(HEAD_NAMES):
        known = labels[:, j] >= 0
        true_p = np.where(labels[:, j] == 1, probability[:, j], 1 - probability[:, j])
        masks = dict(all=known, positive=known & (labels[:, j] == 1), negative=known & (labels[:, j] == 0),
                     hard=known & (true_p < .4), intermediate=known & (true_p >= .4) & (true_p < .8),
                     easy=known & (true_p >= .8))
        total = float(np.abs(gradient[known, j]).sum())
        groups[head] = {}
        for name, mask in masks.items():
            mass = float(np.abs(gradient[mask, j]).sum())
            groups[head][name] = dict(n=int(mask.sum()), absolute_logit_gradient_mass=mass,
                fraction_of_head_gradient_mass=mass / total if total else None,
                mean_absolute_logit_gradient=float(np.abs(gradient[mask, j]).mean()) if mask.any() else None)
    return groups


def gradient_diagnostic(model, x, labels, scaler, ids):
    model.eval()
    device = next(model.parameters()).device
    xx = torch.as_tensor(transform(x[ids], scaler), device=device)
    yy = torch.as_tensor(labels[ids], device=device)
    z = model(xx)
    denominator = int((yy[:, 0] >= 0).sum())
    pieces = F.binary_cross_entropy_with_logits(z, yy.clamp_min(0), reduction="none")
    pieces = pieces * (yy >= 0) * z.new_tensor(HEAD_WEIGHTS)
    trunk_parameters = list(model.encoder.parameters())
    vectors, norms = [], {}
    for j, name in enumerate(HEAD_NAMES):
        gradients = torch.autograd.grad(pieces[:, j].sum() / max(1, denominator), trunk_parameters,
                                        retain_graph=True, allow_unused=False)
        vector = torch.cat([g.detach().reshape(-1) for g in gradients]).double()
        vectors.append(vector)
        norms[name] = float(vector.norm())
    cosine = []
    for a in vectors:
        row = []
        for b in vectors:
            norm = float(a.norm() * b.norm())
            row.append(float(torch.dot(a, b)) / norm if norm else None)
        cosine.append(row)
    raw_z = z.detach().cpu().numpy()
    p = np.exp(-np.logaddexp(0., -raw_z.astype(np.float64)))
    g, error = logit_gradients(raw_z, labels[ids])
    return dict(root_indices=np.asarray(ids).tolist(), root_indices_sha256=_array_sha(ids, "<i8"),
        common_known_root_denominator=denominator, encoder_gradient_norm=norms,
        encoder_head_cosine=cosine, head_order=list(HEAD_NAMES), actual_vs_analytical_max_abs_error=error,
        groups=_gradient_groups(p, labels[ids], g),
        difficulty_definition="Conditional head true-label probability: hard<0.4, intermediate[0.4,0.8), easy>=0.8",
        logit_gradient_definition="Unreduced masked dL/dz including risk coefficient 1/3; encoder gradients divide by E-known roots")


def _filter_config(config):
    c = dict(config["filter"])
    for key, default in dict(device="auto", train_metric_probe=2048, gradient_probe=256).items():
        c.setdefault(key, default)
    for key in ("hidden", "layers", "batch_size", "max_updates", "validate_every", "patience_checks",
                "cpu_threads", "train_metric_probe", "gradient_probe"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError("Positive filter integer required: " + key)
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("Positive finite filter scalar required: " + key)
    if (not np.isfinite([c["weight_decay"], c["min_delta"], c["dropout"]]).all()
            or min(c["weight_decay"], c["min_delta"], c["dropout"]) < 0 or c["dropout"] >= 1):
        raise ValueError("Invalid filter regularization parameters")
    if tuple(c.get("head_names", HEAD_NAMES)) != HEAD_NAMES:
        raise ValueError("Fixed V5 conditional head order required")
    if c.get("base_updates", 0) != 0:
        raise ValueError("V5 auxiliary learner cannot update the frozen base")
    return c


def train_filter(x, labels, train_ids, val_ids, config, out, seed):
    """Fit a new auxiliary filter; return full-N sigmoid predictions and receipts.

    train_ids and val_ids include every feature-admitted root in each role.
    TRAIN scaler uses all train_ids, then optimization/evaluation excludes
    E-unknown roots. Only validation composite NLL selects a checkpoint.
    No CAL/TEST outcome aggregate or policy selection occurs in this function.
    """
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or not x.shape[1]:
        raise ValueError("x must be a float32 N x F detached numpy array")
    if np.isinf(x).any():
        raise ValueError("Infinite features are invalid")
    y = validate_labels(labels)
    if len(y) != len(x):
        raise ValueError("Feature/label root alignment mismatch")
    train_all, val_all = _indices(train_ids, len(x), "TRAIN"), _indices(val_ids, len(x), "VAL")
    if train_all[-1] >= val_all[0]:
        raise ValueError("TRAIN and VAL indices must be disjoint and chronological")
    train = train_all[y[train_all, 0] >= 0]
    validation = val_all[y[val_all, 0] >= 0]
    if not len(train) or not len(validation):
        raise ValueError("Nonempty E-known TRAIN and VAL cohorts required")
    tr = _filter_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    owned = ("checkpoint.pt", "predictions.parquet", "history.json", "summary.json", "scaler.json", "gradient_artifact.json")
    if any((out / name).exists() for name in owned):
        raise ValueError("Immutable training outputs already exist; use a new run directory")
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run configuration differs")
    else:
        _save(out / "config.json", config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(tr["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = tr["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = fit_scaler(x, train_all)
    _save(out / "scaler.json", scaler)
    model = FilterMLP(x.shape[1], tr["hidden"], tr["layers"], tr["dropout"]).to(device)
    initial_sha = _state_sha(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    train_probe = np.sort(np.random.default_rng(seed + 410).choice(train, min(tr["train_metric_probe"], len(train)), replace=False))
    grad_probe = np.sort(np.random.default_rng(seed + 411).choice(train, min(tr["gradient_probe"], len(train)), replace=False))
    prior = np.array([(y[train, j][y[train, j] >= 0].sum() + .5) / ((y[train, j] >= 0).sum() + 1.) for j in range(5)])
    before = gradient_diagnostic(model, x, y, scaler, grad_probe)
    initial_val = evaluate(model, x, y, scaler, validation)
    initial_train = evaluate(model, x, y, scaler, train_probe)
    stream = hashlib.sha256()
    history = [dict(update=0, examples_seen=0, cycles_completed=0, optimization_loss=None,
        train_probe_nll=initial_train["composite_nll"], validation_nll=initial_val["composite_nll"],
        train_probe_heads=initial_train["heads"], validation_heads=initial_val["heads"],
        gradient_norm_mean=None, gradient_norm_max=None, gradient_clip_fraction=None,
        sampled_root_stream_sha256=stream.hexdigest(), stage="untrained_network_diagnostic_and_checkpoint_candidate")]
    best, best_update, stale = initial_val["composite_nll"], 0, 0
    best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
    rng = np.random.default_rng(seed)
    order, offset, cycles, examples = rng.permutation(train), 0, 0, 0
    losses, norms = [], []
    started = time.perf_counter()
    for update in range(1, tr["max_updates"] + 1):
        if offset == len(order):
            cycles += 1
            order, offset = rng.permutation(train), 0
        ids = order[offset:offset + tr["batch_size"]]
        offset += len(ids)
        stream.update(np.asarray(ids, dtype="<i8").tobytes())
        model.train()
        xx = torch.as_tensor(transform(x[ids], scaler), device=device)
        yy = torch.as_tensor(y[ids], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_bce(model(xx), yy)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite filter training loss")
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True))
        optimizer.step()
        examples += len(ids)
        losses.append((float(loss.detach()), len(ids)))
        norms.append(norm)
        if update % tr["validate_every"] and update != tr["max_updates"]:
            continue
        val_metric = evaluate(model, x, y, scaler, validation)
        train_metric = evaluate(model, x, y, scaler, train_probe)
        row = dict(update=update, examples_seen=examples, cycles_completed=cycles + int(offset == len(order)),
            optimization_loss=sum(v * n for v, n in losses) / sum(n for _, n in losses),
            train_probe_nll=train_metric["composite_nll"], validation_nll=val_metric["composite_nll"],
            train_probe_heads=train_metric["heads"], validation_heads=val_metric["heads"],
            gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=float(np.max(norms)),
            gradient_clip_fraction=float(np.mean(np.asarray(norms) > tr["gradient_clip"])),
            sampled_root_stream_sha256=stream.hexdigest(), stage="trained")
        history.append(row)
        if val_metric["composite_nll"] < best - tr["min_delta"]:
            best, best_update, stale = val_metric["composite_nll"], update, 0
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
        else:
            stale += 1
        print(json.dumps(dict(stage="fit_filter_v5", seed=seed, update=update, validation_nll=row["validation_nll"],
                              train_probe_nll=row["train_probe_nll"], best_update=best_update,
                              gradient_clip_fraction=row["gradient_clip_fraction"])), flush=True)
        losses, norms = [], []
        if stale >= tr["patience_checks"]:
            break
    model.load_state_dict(best_state)
    selected = gradient_diagnostic(model, x, y, scaler, grad_probe)
    probability, logits = predict(model, x, scaler, return_logits=True)
    gradient, error = logit_gradients(logits, y)
    prediction_frame = pd.DataFrame({"root_index": np.arange(len(x), dtype=np.int64)})
    for j, name in enumerate(HEAD_NAMES):
        for prefix, values in (("p", probability), ("z", logits), ("y", y), ("g", gradient)):
            prediction_frame[prefix + "_" + name] = values[:, j]
    evaluation = {}
    for role, ids in (("train", train), ("validation", validation)):
        evaluation[role] = dict(model=probability_metrics(y[ids], probability[ids], logits[ids]),
                                prior=probability_metrics(y[ids], np.tile(prior, (len(ids), 1))))
    artifact = dict(before=before, selected=selected, head_order=list(HEAD_NAMES),
        head_loss_coefficients=list(HEAD_WEIGHTS), full_clock_actual_vs_analytical_max_abs_error=error,
        train_groups=_gradient_groups(probability[train], y[train], gradient[train]),
        validation_groups=_gradient_groups(probability[validation], y[validation], gradient[validation]),
        caveat="Teacher masks gate only loss and diagnostics, never inference or admission; risk-safe is not realized profit")
    cohorts = {}
    for role, all_ids, ids in (("train", train_all, train), ("validation", val_all, validation)):
        cohorts[role] = dict(admitted_roots=len(all_ids), known_event_roots=len(ids),
            admitted_root_indices_sha256=_array_sha(all_ids, "<i8"), known_root_indices_sha256=_array_sha(ids, "<i8"),
            known_per_head=(y[ids] >= 0).sum(0).tolist(), positive_per_head=(y[ids] == 1).sum(0).tolist())
    training = dict(best_update=best_update, selected_checkpoint="untrained_network" if best_update == 0 else "trained_network",
        best_validation_nll=best, updates_completed=update, examples_seen=examples, validation_checks=len(history)-1,
        seconds=time.perf_counter()-started, initial_state_sha256=initial_sha, selected_state_sha256=_state_sha(best_state),
        sampled_root_stream_sha256=stream.hexdigest(), train_metric_probe_indices=train_probe.tolist(),
        train_metric_probe_sha256=_array_sha(train_probe, "<i8"), gradient_probe_sha256=_array_sha(grad_probe, "<i8"),
        parameter_count=sum(p.numel() for p in model.parameters()), device=str(device), base_updates=0,
        selection="Unweighted validation masked composite NLL; initial network eligible; TRAIN prior separate",
        sampling="Uniform permutations of admitted E-known TRAIN roots",
        loss_denominator="E-known root count; conditional head masks; coefficients 1,1,1/3,1/3,1/3")
    summary = dict(status="TRAINING_ARTIFACTS_WRITTEN_REQUIRES_RUN_FINALIZATION", seed=seed,
        input_channels=x.shape[1], head_names=list(HEAD_NAMES), head_loss_coefficients=list(HEAD_WEIGHTS),
        training=training, evaluation=evaluation, cohorts=cohorts, train_prior=prior.tolist(),
        scaler_sha256=sha(out / "scaler.json"), learner_source_sha256=sha(__file__),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        caveats=["Conditional sigmoid outputs are not five mutually exclusive classes.",
                 "Pre-crossing risk safety does not imply profitable latency-delayed execution.",
                 "Caller verifies frozen feature lineage, temporal splits, maturity and full-clock admission.",
                 "No CAL/TEST outcomes are used for scaling, optimization, early stopping or model selection."])
    torch.save(dict(state_dict=best_state, input_dim=x.shape[1], filter_config=tr, seed=seed,
        head_names=list(HEAD_NAMES), head_loss_coefficients=list(HEAD_WEIGHTS), best_update=best_update,
        scaler_sha256=summary["scaler_sha256"], learner_source_sha256=summary["learner_source_sha256"],
        initial_state_sha256=initial_sha), out / "checkpoint.pt")
    prediction_frame.to_parquet(out / "predictions.parquet", index=False)
    _save(out / "history.json", history)
    _save(out / "gradient_artifact.json", artifact)
    _save(out / "summary.json", summary)
    return dict(predictions=probability, history=history, gradient=artifact, gradient_artifact=artifact, summary=summary)
