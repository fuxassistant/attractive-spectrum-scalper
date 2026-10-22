"""Causal 1s MBO features and London context on a common origin clock.

The three feature families share admission and source segments. Missing members
of a family remain NaN for the downstream finite-mask adapter; they never
silently select a different cohort. No target or future event metadata enters X.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.cognition import flow_context as fc
from spectrum.playground.data import MonthDay

NS = 1_000_000_000
SCALES = (1, 3, 6)
SOURCE_METADATA = ("t_ns", "available_ns", "epoch", "admissible", "source_day_degraded")
PREPARED_CONTEXT_FILES = (
    "x.npy", "bar_t.npy", "decision_t.npy", "decision_mid.npy", "cuts.npy",
    "execution_valid.npy", "execution_segment.npy",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bind(path, hashes, expected=None):
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Consumed source hash mismatch: {path}")
    hashes[str(Path(path).resolve())] = actual
    return actual


def transform_kind(name):
    """Magnitude compression is explicit and leaves normalized ratios intact."""
    if any(x in name for x in ("fraction", "entropy", "efficiency", "imbalance")):
        return "identity"
    if name.startswith(("depth_", "trade_", "add_", "remove_", "matched_fill_remove_", "event_count_", "path_")) or name.endswith("_power"):
        return "signed_log1p"
    return "identity"


def transform_values(values, names):
    out = np.asarray(values, float).copy()
    mappings = []
    for i, name in enumerate(names):
        operation = transform_kind(name)
        output_name = "mbo__" + name + ("__signed_log1p" if operation == "signed_log1p" else "")
        if operation == "signed_log1p":
            out[:, i] = np.sign(out[:, i]) * np.log1p(np.abs(out[:, i]))
        mappings.append(dict(source=name, output=output_name, transform=operation))
    return out, mappings


def asof_source(frame, origins_ns, names, max_age_seconds=2.):
    """Latest fully published endpoint, with explicit support and feature masks.

    Prefix-maximum publication prevents an out-of-order proof from backfilling
    unavailable earlier observations. Age is measured from the endpoint.
    """
    t = np.asarray(origins_ns, np.int64)
    endpoints = frame.t_ns.to_numpy(np.int64)
    if len(frame) == 0 or np.any(np.diff(endpoints) <= 0):
        raise ValueError("Nonempty strictly ordered source endpoints required")
    if not np.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("Positive finite source maximum age required")
    published = np.maximum.accumulate(frame.available_ns.to_numpy(np.int64))
    idx = np.minimum(np.searchsorted(endpoints, t, side="right"),
                     np.searchsorted(published, t, side="right")) - 1
    safe = np.maximum(idx, 0)
    age = (t - endpoints[safe]) / NS
    source_good = frame.admissible.to_numpy(bool) & ~frame.source_day_degraded.to_numpy(bool)
    ready = (idx >= 0) & source_good[safe] & (age >= 0) & (age <= max_age_seconds)
    source_part = frame.source_part.to_numpy(np.int64) if "source_part" in frame else np.zeros(len(frame), np.int64)
    epoch = frame.epoch.to_numpy(np.int64)
    boundary = np.r_[True, (np.diff(endpoints) != NS) | (epoch[1:] != epoch[:-1]) |
                     (source_part[1:] != source_part[:-1]) | ~source_good[1:] | ~source_good[:-1]]
    tokens = np.cumsum(boundary).astype(np.int64)
    values = frame[list(names)].to_numpy(float)[safe]
    masks = frame[[n + "__known" for n in names]].to_numpy(bool)[safe]
    if np.any(masks & ~np.isfinite(values)):
        raise ValueError("Known source feature is nonfinite")
    values = np.where(masks & ready[:, None], values, np.nan)
    return values, dict(ready=ready, row_index=idx, age_seconds=np.where(idx >= 0, age, np.nan),
                        endpoint_ns=np.where(idx >= 0, endpoints[safe], -1),
                        publication_ns=np.where(idx >= 0, published[safe], -1),
                        source_segment=np.where(ready, tokens[safe], -1))


def quote_state(q, origins_ns, max_gap_seconds=60.):
    """Quote support tokens retain every intervening gap/invalid transition."""
    t = np.asarray(origins_ns, np.int64)
    qt = np.asarray(q.quote_t, np.int64)
    if len(qt) == 0 or np.any(np.diff(qt) < 0):
        raise ValueError("Ordered nonempty quote publications required")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Positive finite maximum quote gap required")
    good = np.asarray(q.valid, bool) & (q.epoch >= 0) & (q.bid > 0) & (q.ask > q.bid)
    gap = int(round(max_gap_seconds * NS))
    boundary = np.r_[True, ~good[1:] | ~good[:-1] | (q.epoch[1:] != q.epoch[:-1]) | (np.diff(qt) > gap)]
    tokens = np.cumsum(boundary).astype(np.int64)
    idx = np.searchsorted(qt, t, side="right") - 1
    safe = np.maximum(idx, 0)
    ready = (idx >= 0) & good[safe] & (t - qt[safe] <= gap)
    return dict(ready=ready, ordinal=idx, segment=np.where(ready, tokens[safe], -1))


def common_segments(origins, ready, source_segment, quote_segment, cadence_seconds):
    """Origin tokens change on either underlying continuity boundary."""
    t = np.asarray(origins, np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Strictly chronological origins required")
    boundary = np.r_[True, ~ready[1:] | ~ready[:-1] | (source_segment[1:] != source_segment[:-1]) |
                     (quote_segment[1:] != quote_segment[:-1]) | (np.diff(t) != int(round(cadence_seconds * NS)))]
    tokens = np.cumsum(boundary).astype(np.int64)
    return np.where(ready, tokens, -1)


def supported_source_at_origin(q, origins, source_proof, max_gap_seconds=60.):
    """A fresh row from before support loss is still stale in the new book.

    Query both publication clocks in one quote scan so the potentially large
    publication-token array is allocated only once.
    """
    t = np.asarray(origins, np.int64)
    states = quote_state(q, np.r_[t, source_proof["publication_ns"]], max_gap_seconds)
    n = len(t)
    same = states["ready"][:n] & states["ready"][n:] & (states["segment"][:n] == states["segment"][n:])
    ready = source_proof["ready"] & same
    return ready, states["segment"][:n]


def geometry_features(scene, origins_ns, london_ns):
    """Read only named causal geometry fields; future metadata is ignored."""
    t = np.asarray(origins_ns, np.int64)
    if not np.array_equal(scene.t_ns.to_numpy(np.int64), t):
        raise ValueError("Scene and origin clocks differ")
    mid = scene.mid.to_numpy(float)
    spread = scene.decision_spread.to_numpy(float)
    values, names = [], []
    for scale in SCALES:
        middle = scene[f"BB{scale}_middle"].to_numpy(float)
        half = scene[f"BB{scale}_half"].to_numpy(float)
        support = scene[f"BB{scale}_support"].to_numpy(np.int64)
        good = (support >= 0) & np.isfinite(mid) & np.isfinite(middle) & np.isfinite(half) & (half > 0)
        vals = ((mid - middle) / half, np.log1p(half / .1), spread / half,
                (t % (scale * 60 * NS)) / (scale * 60 * NS))
        for suffix, value in zip(("position", "log_half_ticks", "spread_half", "phase"), vals):
            values.append(np.where(good, value, np.nan))
            names.append(f"bb{scale}_{suffix}")
    values.extend(((t - london_ns) / (60 * NS), spread / .1))
    names.extend(("minutes_since_london", "spread_ticks"))
    return np.column_stack(values), names


def _bind_prepared_context(day, hashes):
    _bind(day.path / "manifest.json", hashes)
    for name in PREPARED_CONTEXT_FILES:
        _bind(day.path / name, hashes, day.meta["files"][name]["sha256"])


def _selected_capture(catalog, overlay, date, symbol):
    selected = [r for r in catalog["captures"] if r.get("status") == "COMPLETE" and
                r["entry"]["selection"]["date"] == date and r["entry"]["selection"]["symbol"] == symbol]
    if len(selected) != 1:
        return None
    rec = selected[0]
    audit = overlay.get(rec["key"])
    selection = rec["entry"]["selection"]
    if not audit or not audit.get("audit_eligible") or selection.get("source_condition") != "available" or selection.get("context_only"):
        return None
    return rec


def build_features(day, q, origins_ns, scene: pd.DataFrame, config):
    """Build shared X; hashes cover actual consumed prepared and 1s payloads.

    ``q`` is supplied by the caller. Previous-day context uses prepared arrays
    through ``fc.minute_tape`` and never accesses ``MonthDay.quote_t/bid/ask``.
    Exact trade-price VWAP is not reconstructed by this adapter.
    """
    data = config["data"]
    source_root = Path(data["source_root"])
    market_root = Path(data["market_root"])
    overlay_path = Path(data["admission_overlay"])
    hashes = {}
    _bind(source_root / "manifest.json", hashes)
    _bind(overlay_path, hashes)
    catalog = _json(source_root / "manifest.json")
    overlay_doc = _json(overlay_path)
    overlay = {r["key"]: r for r in overlay_doc["captures"]}
    if catalog.get("status") != "COMPLETE":
        raise ValueError("Unpublished full MBO source")
    date, symbol = day.meta["date"], day.meta["symbol"]
    current = _selected_capture(catalog, overlay, date, symbol)
    if current is None:
        raise ValueError(f"Current day/contract is not admitted by overlay: {date}/{symbol}")
    t = np.asarray(origins_ns, np.int64)
    if t.ndim != 1 or not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Nonempty chronological origin vector required")
    capture_map = {r["key"]: r for r in catalog["captures"] if r.get("status") == "COMPLETE"}
    parts, raw_names, consumed = [], None, []
    for number, part in enumerate(day.meta["quote_parts"]):
        rec = capture_map.get(part["key"])
        if rec is None or rec["entry"]["selection"]["symbol"] != symbol:
            raise ValueError("Prepared session source or contract is not canonical")
        feature_root = Path(rec["features"])
        manifest_path = feature_root / "manifest.json"
        feature_path = feature_root / "features.parquet"
        _bind(manifest_path, hashes, rec["feature_manifest_sha256"])
        manifest = _json(manifest_path)
        if manifest.get("status") != "COMPLETE":
            raise ValueError("Unpublished 1s features")
        digest = _bind(feature_path, hashes, rec["feature_sha256"])
        if digest != manifest["feature_sha256"] or digest != overlay[part["key"]]["feature_sha256"]:
            raise ValueError("Feature payload disagrees with admission or feature receipt")
        _bind(Path(rec["market"]) / "manifest.json", hashes, manifest["market_manifest_sha256"])
        columns = pq.ParquetFile(feature_path).schema_arrow.names
        names = [n for n in columns if n + "__known" in columns]
        if len(names) != 182 or len(names) != manifest["features"]:
            raise ValueError("Expected frozen 182-value source schema")
        if raw_names is not None and names != raw_names:
            raise ValueError("Source feature schema changed between captures")
        raw_names = names
        selected_columns = list(SOURCE_METADATA) + names + [n + "__known" for n in names]
        frame = pq.read_table(feature_path, columns=selected_columns,
                              filters=[("t_ns", ">=", int(part["start_ns"])), ("t_ns", "<", int(part["end_ns"]))]).to_pandas()
        frame["source_part"] = number
        parts.append(frame)
        consumed.append(dict(key=part["key"], rows=len(frame), feature_sha256=digest))
    source = pd.concat(parts, ignore_index=True).sort_values("t_ns").reset_index(drop=True)
    age_limit = float(data.get("max_feature_age_seconds", 2.))
    raw, asof = asof_source(source, t, raw_names, age_limit)
    raw, mappings = transform_values(raw, raw_names)
    ready, quote_tokens = supported_source_at_origin(q, t, asof, config["execution"].get("max_quote_gap_seconds", 60.))
    tokens = common_segments(t, ready, asof["source_segment"], quote_tokens, data["cadence_seconds"])
    geometry, geometry_names = geometry_features(scene, t, int(day.meta["london_ns"]))

    _bind_prepared_context(day, hashes)
    current_tape = fc.minute_tape(day)
    previous_date = str((pd.Timestamp(date) - pd.offsets.BDay()).date())
    previous_path = market_root / previous_date[:7] / previous_date
    previous_capture = _selected_capture(catalog, overlay, previous_date, symbol)
    previous_tape = None
    previous_status = "NOT_ADMITTED_OR_CONTRACT_CHANGED"
    if previous_capture is not None and (previous_path / "manifest.json").exists():
        previous_day = MonthDay(previous_path, verify=False)
        if previous_day.meta["date"] != previous_date or previous_day.meta["symbol"] != symbol:
            raise ValueError("Previous prepared identity disagrees with catalog")
        _bind_prepared_context(previous_day, hashes)
        previous_tape = fc.minute_tape(previous_day)
        previous_status = "ADMITTED_SAME_CONTRACT; per-context coverage and publication masks still apply"
    half = scene.BB1_half.to_numpy(float)
    price_ticks = scene.mid.to_numpy(float) / .1
    sigma = (half / .1) / np.sqrt(600.)
    context, clocks = fc.context_features(current_tape, previous_tape, t, price_ticks, sigma)
    if np.any(clocks["latest_available_ns"] > t) or np.any(clocks["latest_endpoint_ns"] > t):
        raise ValueError("Future London context publication")
    context_columns = list(fc.DAY) + list(fc.SPECTRUM)
    context_names = ["context__" + n for n in context_columns]
    micro_indices = [i for i, name in enumerate(raw_names) if not name.startswith("fft_")]
    spectral_indices = [i for i, name in enumerate(raw_names) if name.startswith("fft_")]
    order = micro_indices + spectral_indices
    raw_names_out = [mappings[i]["output"] for i in order]
    names = geometry_names + raw_names_out + context_names
    values = np.c_[geometry, raw[:, order], context[context_columns].to_numpy(float)]
    values[~np.isfinite(values)] = np.nan
    groups = dict(geometry=geometry_names, micro=geometry_names + [mappings[i]["output"] for i in micro_indices],
                  rich=names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names")
    receipts = dict(
        status="CAUSAL_FEATURES_BUILT", date=date, symbol=symbol,
        source_sha256=hashes,
        implementation_sha256={str(Path(p).resolve()): _sha(p) for p in (__file__, fc.__file__)},
        consumed_captures=consumed, source_values=len(raw_names), feature_names=names, groups=groups,
        transforms=mappings, geometry_transform="position=(mid-middle)/half; log_half_ticks=log1p(half/.1); spread_half=spread/half; native phase from causal clock",
        context_normalization="fc sigma=(causal BB1 half/.1)/sqrt(600); price_ticks=mid/.1. Therefore price-distance denominator is causal BB1 half in ticks, not fitted future volatility.",
        context_semantics="Current London prefix and immediately previous admitted same-contract London-to-close summaries; combined VWAP includes current overnight. M1 midpoint/volume VWAP proxy, not exact T-price VWAP. STFT 128 completed M1 endpoints with original masks.",
        previous_date=previous_date, previous_source_status=previous_status,
        previous_known_roots=int((context.previous_known > .5).sum()),
        origins=len(t), feature_available=int(ready.sum()), unavailable=int((~ready).sum()),
        admission="Common across geometry/micro/rich: latest 1s source admitted, not source_day_degraded, endpoint and publication <= origin, endpoint age within limit, quote supported both at source publication and origin with identical continuity token. Never requires all feature-family masks.",
        max_feature_age_seconds=age_limit,
        max_observed_admitted_age_seconds=float(np.max(asof["age_seconds"][ready])) if ready.any() else None,
        max_asof_lookahead_ns=int(max(0, np.max(asof["publication_ns"] - t))),
        max_context_lookahead_ns=int(max(0, np.max(clocks["latest_available_ns"] - t))),
        missing_fraction={n: float((~np.isfinite(values[:, i])).mean()) for i, n in enumerate(names)},
        segments="Combined source capture/epoch/1s gap/admission and quote epoch/invalid/gap support tokens; no crossing outcome or future support-end used",
        quote_lineage="Caller supplies full quote tape and owns quote-payload verification; this function does not load quotes",
        no_seconds_payload_read=True,
    )
    return dict(x_raw=values.astype(np.float32), feature_names=names, groups=groups,
                feature_available=ready, segment_token=tokens, receipts=receipts)
