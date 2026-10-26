"""Reusable formulas extracted from spectrum/research_v2/features.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd

NS = 1_000_000_000


SCALES = (1, 3, 6)


SOURCE_METADATA = ("t_ns", "available_ns", "epoch", "admissible", "source_day_degraded")


PREPARED_CONTEXT_FILES = (
    "x.npy", "bar_t.npy", "decision_t.npy", "decision_mid.npy", "cuts.npy",
    "execution_valid.npy", "execution_segment.npy",
)


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

from spectrum.core.storage import sha as _sha
