"""Immutable causal feature admission and train-only sequence preprocessing.

Saved crossing outcomes are diagnostics. ``execution_profit`` is supplied by
the independent execution engine; its crossing comparator remains amplitude.
No target, future event ID, maturity time, or support end can enter ``x_raw``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = ROOT / "data/derived/crossing_map_target_v1"
BB_ROOT = ROOT / "data/derived/bb_crossing_bank_v1"
NS = 1_000_000_000

# Intentionally explicit. The source parquet interleaves future targets and X.
RICH_FEATURES = (
    "log_sigma", "sigma60_ratio", "sigma3600_ratio", "minutes_since_london",
    "spread_ticks", "ret5", "ret15", "ret60", "pressure5", "pressure30",
    "unknown30", "book5", "book30", "depth1", "depth4", "volume_rate",
    "pressure_change5", "pressure_change15", "book_change15", "depth_change5",
    "depth_change15", "ret300", "range60", "range300", "unknown5", "depth12",
    "volume5", "volume30", "volume60", "volume300", "pressure60", "unknown60",
    "book60", "pressure300", "unknown300", "book300", "day_age_minutes",
    "context_age_seconds", "day_coverage", "day_from_open", "day_from_high",
    "day_from_low", "day_vwap", "day_range", "day_log_volume", "previous_known",
    "previous_age_days", "previous_coverage", "previous_from_open",
    "previous_from_close", "previous_from_high", "previous_from_low",
    "previous_vwap", "previous_range", "combined_vwap", "combined_range",
    "volume_vs_previous", "stft_price_b0", "stft_price_b1", "stft_price_b2",
    "stft_price_entropy", "stft_volume_b0", "stft_volume_b1", "stft_volume_b2",
    "stft_volume_entropy",
)
GEOMETRY_FEATURES = tuple(
    f"bb{s}_{kind}" for s in (1, 3, 6, 9)
    for kind in ("position", "log_half_ticks", "spread_half", "phase")
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verified(path, expected, hashes):
    observed = _sha(path)
    if observed != expected:
        raise ValueError(f"Source SHA mismatch: {path}")
    hashes[str(Path(path).resolve())] = observed


def _join_features(roots, source, names):
    """Left admission keeps every clock root; no label-dependent filtering."""
    keys = ["date", "t_ns"]
    if roots.duplicated(keys).any() or source.duplicated(keys).any():
        raise ValueError("Duplicate origin keys")
    missing = set(names) - set(source.columns)
    if missing:
        raise ValueError(f"Missing allowlisted columns: {sorted(missing)}")
    selected = source[keys + list(names)].copy()
    selected["feature_available"] = True
    frame = roots.merge(selected, on=keys, how="left", validate="one_to_one", sort=False)
    frame["feature_available"] = frame.feature_available.eq(True)
    values = frame[list(names)].to_numpy(dtype=np.float32)
    # Present but nonfinite features retain an explicit missing channel later.
    values[~np.isfinite(values)] = np.nan
    return frame, values


def _source_paths(data):
    """Explicit overrides are honored; relative paths follow the run's cwd."""
    return (Path(data.get("roots_dir", MAP_ROOT)),
            Path(data.get("feature_roots_dir", BB_ROOT)))


def load_market(config: dict) -> dict:
    """Load all requested clock roots, causal X, and a mature crossing label.

    Admission is ``frame.feature_available``. A finite feature is observed; a
    nonfinite feature is missing. Unknown labels never control feature windows.
    Source payload hashes are checked against the original completed receipts.
    """
    from scripts.crossing_map_analyze import target_frame

    data = config["data"]
    map_root, bb_root = _source_paths(data)
    target = config.get("target", {})
    dates = sorted(data["dates"])
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Nonempty unique dates required")
    family = data.get("feature_family", "rich")
    if family not in ("geometry", "rich"):
        raise ValueError("feature_family must be geometry or rich")
    names = list(GEOMETRY_FEATURES if family == "geometry" else RICH_FEATURES + GEOMETRY_FEATURES)
    line_family = target.get("family", "BB")
    scales = target.get("scales", [1, 3, 6])
    if line_family not in ("BB", "VWAP") or not scales or len(scales) != len(set(scales)) or not set(scales) <= {1, 3, 5, 6, 9}:
        raise ValueError("Unsupported crossing family or scales")
    deadline = float(target.get("deadline_seconds", 300))
    gain = float(target.get("gain_multiple", 2))
    if deadline <= 0 or not np.isfinite(deadline) or gain <= 0 or not np.isfinite(gain):
        raise ValueError("Positive finite deadline and gain required")
    kind = target.get("kind", "execution_profit")
    comparator = {"execution_profit": "amplitude", "crossing_amplitude": "amplitude",
                  "amplitude": "amplitude", "safe": "safe", "raw": "raw"}.get(kind)
    if comparator is None:
        raise ValueError(f"Unsupported target kind: {kind}")

    map_manifest = _json(map_root / "manifest.json")
    bb_manifest = _json(bb_root / "manifest.json")
    if map_manifest.get("status") != "COMPLETE" or bb_manifest.get("status") != "COMPLETE":
        raise ValueError("Unpublished source dataset")
    map_days = {r["date"]: r for r in map_manifest["days"]}
    bb_days = {r["date"]: r for r in bb_manifest["days"]}
    hashes = {str((p / "manifest.json").resolve()): _sha(p / "manifest.json") for p in (map_root, bb_root)}
    all_roots, all_source, all_labels, provenance = [], [], [], []
    for date in dates:
        if date not in map_days or date not in bb_days:
            raise ValueError(f"Date missing from completed source: {date}")
        mr, br = map_days[date], bb_days[date]
        for directory, receipt in ((map_root, mr), (bb_root, br)):
            path = directory / f"{date}.json"
            if _json(path) != receipt:
                raise ValueError(f"Daily receipt differs from manifest: {path}")
            hashes[str(path.resolve())] = _sha(path)
        for suffix in ("roots", "map"):
            path = map_root / f"{date}_{suffix}.parquet"
            _verified(path, mr["files"][path.name], hashes)
        bp = bb_root / f"{date}_roots.parquet"
        _verified(bp, br["files"]["roots"], hashes)
        roots = pd.read_parquet(map_root / f"{date}_roots.parquet").sort_values("t_ns").reset_index(drop=True)
        source = pd.read_parquet(bp, columns=["date", "t_ns"] + names)
        cells = pd.read_parquet(map_root / f"{date}_map.parquet")
        cells = cells.loc[cells.family == line_family]
        labels = target_frame(cells, roots, scales, deadline, gain)
        if not np.array_equal(labels.t_ns, roots.t_ns):
            raise ValueError("Crossing target alignment changed")
        all_roots.append(roots)
        all_source.append(source)
        all_labels.append(labels[f"{comparator}_state"].to_numpy(np.int64))
        provenance.append(dict(date=date, source_feature_roots=len(source), clock_roots=len(roots),
                               old_feature_source_sha256=br["source_sha256"],
                               market_manifest_sha256=br["market_manifest_sha256"]))
    roots = pd.concat(all_roots, ignore_index=True)
    source = pd.concat(all_source, ignore_index=True)
    frame, raw = _join_features(roots, source, names)
    if not frame.t_ns.is_monotonic_increasing:
        raise ValueError("Origins must be chronological")
    finite = np.isfinite(raw)
    receipts = dict(
        source_paths=dict(roots_dir=str(map_root.resolve()), feature_roots_dir=str(bb_root.resolve())),
        source_sha256=hashes, days=provenance, feature_allowlist=names,
        feature_allowlist_sha256=hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        target_implementation_sha256=_sha(ROOT / "scripts/crossing_map_analyze.py"),
        feature_source="BB root explicit causal allowlist; rich day is compressed London context; STFT uses M1 midpoint/volume proxies",
        quote_hash_scope="Inherited quote lineage in completed source receipts; this loader directly rechecks consumed parquet and receipt hashes, not raw quote payloads",
        total_roots=len(frame), feature_available=int(frame.feature_available.sum()),
        missing_feature_origins=int((~frame.feature_available).sum()),
        nonfinite_per_feature={n: int((~finite[:, i]).sum()) for i, n in enumerate(names)},
        removed_roots=0, crossing_kind=comparator,
        label_semantics="Mature joint states NONE=0,LONG_ONLY=1,SHORT_ONLY=2,BOTH=3,UNKNOWN=-1; crossing marks are not an executable exit policy",
    )
    return dict(frame=frame, x_raw=raw, feature_names=names,
                y_crossing=np.concatenate(all_labels), receipts=receipts)


def build_sequence_indices(frame: pd.DataFrame, length: int, cadence_seconds=30) -> np.ndarray:
    """Past-only windows; reset on date/gap/missing-X, never on unknown Y."""
    if not isinstance(length, (int, np.integer)) or length <= 0 or cadence_seconds <= 0:
        raise ValueError("Positive sequence length and cadence required")
    times = frame.t_ns.to_numpy(np.int64)
    dates = frame.date.to_numpy()
    available = frame.feature_available.to_numpy(bool)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Strict chronological origin order required")
    indices = np.full((len(frame), length), -1, dtype=np.int64)
    begin = 0
    step = int(cadence_seconds * NS)
    for i in range(len(frame)):
        if not available[i]:
            begin = i + 1
            continue
        if i == 0 or dates[i] != dates[i - 1] or times[i] - times[i - 1] != step:
            begin = i
        start = max(begin, i - length + 1)
        indices[i, -(i - start + 1):] = np.arange(start, i + 1)
    return indices


def prepare_sequences(x_raw: np.ndarray, seqidx: np.ndarray, train_root_indices) -> tuple[np.ndarray, dict]:
    """Scale unique observed TRAIN-window rows, then add finite/padding masks.

    Columns are [F standardized values, F observed flags, one real-row flag].
    All-missing and constant training channels use scale 1; NaN values become
    zero only together with observed=0. Validation/test cannot fit the scaler.
    """
    raw = np.asarray(x_raw, dtype=np.float64)
    idx = np.asarray(seqidx)
    train = np.asarray(train_root_indices)
    if raw.ndim != 2 or idx.ndim != 2 or len(idx) != len(raw) or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("Expected aligned [N,F] values and integer [N,T] indices")
    if np.any(idx < -1) or np.any(idx >= len(raw)) or np.any(idx > np.arange(len(raw))[:, None]):
        raise ValueError("Sequence indices must be past-only or -1 padding")
    if train.ndim != 1 or not len(train) or not np.issubdtype(train.dtype, np.integer) or np.any(train < 0) or np.any(train >= len(raw)):
        raise ValueError("Nonempty integer training origin indices required")
    unique = np.unique(idx[train])
    unique = unique[unique >= 0]
    if not len(unique):
        raise ValueError("No observed training feature rows")
    values = raw[unique]
    known = np.isfinite(values)
    count = known.sum(axis=0)
    mean = np.divide(np.where(known, values, 0).sum(axis=0), count,
                     out=np.zeros(raw.shape[1]), where=count > 0)
    centered = np.where(known, values - mean, 0)
    variance = np.divide((centered * centered).sum(axis=0), count,
                         out=np.zeros(raw.shape[1]), where=count > 0)
    std = np.sqrt(variance)
    scale = np.where(std > 1e-8, std, 1.)
    finite = np.isfinite(raw)
    normalized = np.where(finite, (raw - mean) / scale, 0.)
    clipped = np.abs(normalized) > 10
    normalized = np.clip(normalized, -10, 10).astype(np.float32)
    real = idx >= 0
    safe = np.maximum(idx, 0)
    output = np.concatenate((normalized[safe], finite[safe].astype(np.float32), real[..., None].astype(np.float32)), axis=-1)
    output[~real] = 0.
    scaler = dict(method="unique_train_window_rows_mean_population_std", mean=mean.tolist(), scale=scale.tolist(),
                  finite_count=count.tolist(), all_missing_columns=np.flatnonzero(count == 0).tolist(),
                  constant_columns=np.flatnonzero((count > 0) & (std <= 1e-8)).tolist(), cap=10.,
                  train_root_count=len(np.unique(train)), train_feature_row_indices=unique.tolist(),
                  train_feature_rows=len(unique), total_nonfinite=int((~finite).sum()),
                  clipped_value_count=int(clipped.sum()), clipped_train_value_count=int(clipped[unique].sum()),
                  output_channels="F scaled values, F finite masks, one real-row padding flag")
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite normalized sequence")
    return output, scaler
