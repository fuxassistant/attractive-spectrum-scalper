"""Causal geometric competing first passage on native midpoint publications.

RETURN_FIRST means that the frozen near edge was reached or crossed, including
a jump across the entire zone. It does not require an observed dwell, and is
neither an executable fill nor a profit label. ADVERSE_FIRST uses the midpoint
known at the decision minus the toward side times one frozen risk distance.

Prices are compared in doubled nanodollars; clocks remain integer nanoseconds.
Risk is rounded once to the nearest nanodollar (NumPy ties-to-even). The future
window is (decision, decision + horizon], so an endpoint hit precedes TIMEOUT.
Only an uninterrupted supported path can mature to TIMEOUT. CENSORED is an
observation mask, never a fourth market class or an invented failure.

Rows sharing one publication time are intentionally treated as simultaneous,
even when source_ordinal preserves their source order. Opposite hits within
that group are censored as AMBIGUOUS_SAME_TIME; any invalid/reset row in the
group also prevents a known first-passage label. No intra-clock order is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

NS = 1_000_000_000
STATES = ("RETURN_FIRST", "ADVERSE_FIRST", "TIMEOUT", "CENSORED")
REASONS = ("RETURN_EDGE", "ADVERSE_BOUNDARY", "HORIZON", "RIGHT_CENSOR",
           "COVERAGE", "ORIGIN_SUPPORT", "TARGET_GEOMETRY", "INVALID_QUOTE",
           "EPOCH_CHANGE", "STALE_QUOTE", "AMBIGUOUS_SAME_TIME")
TIME_BINS_SECONDS = (30, 60, 120, 300)
_MAX = np.iinfo(np.int64).max


def _exact_int(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} requires exact integer nanoseconds")
    value = int(value)
    if value < (1 if positive else 0) or value > _MAX:
        raise ValueError(f"{name} is outside the supported integer clock range")
    return value


def _integer_column(frame, name):
    if name not in frame or frame[name].dtype.kind not in "iu":
        raise ValueError(f"{name} requires exact integers")
    if len(frame) and frame[name].dtype.kind == "u" and frame[name].max() > _MAX:
        raise ValueError(f"{name} exceeds signed int64")
    return frame[name].to_numpy(dtype=np.int64)


def _duration_ns(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    duration = int(round(float(value) * NS))
    if not 0 < duration < _MAX // 4:
        raise ValueError(f"{name} is outside the supported duration range")
    return duration


@njit(cache=True)
def _first_passage(qt, mid2, good, epoch, origins, sides, targets, risks,
                   coverage, asof, duration, gap):
    # state, reason, origin index, event index, event time, label availability,
    # censor time, terminal observed index, origin midpoint*2, adverse price*2.
    result = np.full((len(origins), 10), -1, np.int64)
    visible = np.searchsorted(qt, asof, side="right")
    for i in range(len(origins)):
        t = origins[i]
        end = t + duration
        result[i, 0] = 3
        result[i, 1] = 3 if asof < coverage else 4
        result[i, 5] = min(asof, coverage)
        result[i, 6] = result[i, 5]
        if t > asof or t >= coverage:
            continue
        d = np.searchsorted(qt, t, side="right") - 1
        result[i, 2] = d
        result[i, 7] = d
        if d < 0 or not good[d] or t - qt[d] > gap:
            result[i, 1], result[i, 5], result[i, 6] = 5, t, t
            continue
        origin = mid2[d]
        adverse = origin - sides[i] * 2 * risks[i]
        target = 2 * targets[i]
        result[i, 8], result[i, 9] = origin, adverse
        if sides[i] * (target - origin) <= 0:
            result[i, 1], result[i, 5], result[i, 6] = 6, t, t
            continue
        token = epoch[d]
        previous = d
        j = d + 1
        while True:
            expiry = qt[previous] + gap
            boundary = min(end, expiry, asof, coverage)
            if j >= visible or qt[j] > boundary or qt[j] >= coverage:
                if end <= asof and end < coverage and end <= expiry:
                    result[i, 0], result[i, 1] = 2, 2
                    result[i, 5], result[i, 6] = end, -1
                elif expiry <= asof and expiry < coverage and expiry < end:
                    result[i, 1], result[i, 5], result[i, 6] = 9, expiry, expiry
                else:
                    result[i, 1] = 3 if asof < coverage else 4
                    result[i, 5] = min(asof, coverage)
                    result[i, 6] = result[i, 5]
                break
            stamp = qt[j]
            k = j
            return_index = -1
            adverse_index = -1
            support_reason = -1
            while k < visible and qt[k] == stamp:
                if not good[k]:
                    if support_reason < 0:
                        support_reason = 7
                elif epoch[k] != token:
                    if support_reason < 0:
                        support_reason = 8
                else:
                    if sides[i] * (mid2[k] - target) >= 0 and return_index < 0:
                        return_index = k
                    if sides[i] * (mid2[k] - adverse) <= 0 and adverse_index < 0:
                        adverse_index = k
                k += 1
            result[i, 7] = k - 1
            if support_reason >= 0:
                result[i, 1], result[i, 5], result[i, 6] = support_reason, stamp, stamp
                break
            if return_index >= 0 and adverse_index >= 0:
                result[i, 1], result[i, 5], result[i, 6] = 10, stamp, stamp
                break
            if return_index >= 0 or adverse_index >= 0:
                state = 0 if return_index >= 0 else 1
                event_index = return_index if state == 0 else adverse_index
                result[i, 0], result[i, 1], result[i, 3] = state, state, event_index
                result[i, 4], result[i, 5], result[i, 6] = stamp, stamp, -1
                break
            previous = k - 1
            j = k
    return result


def build_targets(quotes, candidates, *, coverage_end_ns, asof_ns=None,
                  horizon_seconds=300, max_quote_age_seconds=30):
    """Return one target row per input candidate, preserving input order/IDs.

    Quotes require integer available_ns/bid_nano/ask_nano/epoch and bool valid;
    optional source_ordinal is preserved for lineage. Quotes must be sorted by
    publication time, and coverage_end_ns is exclusive. asof_ns is inclusive;
    omitting it means the full recorded coverage is known. Quotes beyond asof
    cannot affect a target, even when the caller passes the complete tape.

    Candidates require candidate_id/origin_index/t_ns/side/target_nano and a
    positive finite risk_per_oz. Available geometry clocks, when supplied, must
    be integer and no later than t_ns. This function accepts outside candidates
    only; it never drops unsupported candidates or uses a later path to admit
    them. decision_mid_per_oz is deliberately not trusted: origin midpoint is
    reconstructed from the last publication at or before the decision.

    label_known is false for every CENSORED row. first_event_ns and
    first_event_bin_seconds are -1 unless a supported competing event occurred.
    TIMEOUT matures at the horizon; CENSORED availability is the first support
    failure, asof cutoff or coverage end. No latency, spread cost or execution
    claim is embedded in these geometric targets.
    """
    coverage = _exact_int(coverage_end_ns, "coverage_end_ns", positive=True)
    asof = coverage if asof_ns is None else _exact_int(asof_ns, "asof_ns")
    duration = _duration_ns(horizon_seconds, "horizon_seconds")
    gap = _duration_ns(max_quote_age_seconds, "max_quote_age_seconds")
    if duration > TIME_BINS_SECONDS[-1] * NS:
        raise ValueError("This target contract supports horizons up to 300 seconds")
    qt = _integer_column(quotes, "available_ns")
    bid = _integer_column(quotes, "bid_nano")
    ask = _integer_column(quotes, "ask_nano")
    epoch = _integer_column(quotes, "epoch")
    if "valid" not in quotes or quotes.valid.dtype.kind != "b":
        raise ValueError("valid requires bool values without missingness")
    if not len(qt) or np.any(qt[1:] < qt[:-1]) or qt[0] < 0:
        raise ValueError("Nonempty sorted publication tape required")
    if qt[-1] >= coverage or qt[-1] > _MAX - gap:
        raise ValueError("Exclusive coverage must follow all publications without clock overflow")
    if "t_ns" in quotes:
        physical = _integer_column(quotes, "t_ns")
        if np.any(qt < physical):
            raise ValueError("Publication cannot precede source time")
    ordinal = (_integer_column(quotes, "source_ordinal") if "source_ordinal" in quotes
               else np.arange(len(qt), dtype=np.int64))
    ids = _integer_column(candidates, "candidate_id")
    origin_ids = _integer_column(candidates, "origin_index")
    origins = _integer_column(candidates, "t_ns")
    sides = _integer_column(candidates, "side")
    targets = _integer_column(candidates, "target_nano")
    if len(np.unique(ids)) != len(ids) or len(np.unique(origin_ids)) != len(origin_ids):
        raise ValueError("Unique candidate_id and origin_index required")
    if np.any(origins < 0) or np.any(origins > _MAX - duration):
        raise ValueError("Decision clock is outside the supported range")
    if not np.isin(sides, [-1, 1]).all() or np.any(targets <= 0) or np.any(targets > _MAX // 4):
        raise ValueError("Sides must be +/-1 and target prices positive within range")
    if "inside" in candidates and candidates.inside.any():
        raise ValueError("Only outside candidates belong to this target contract")
    for name in ("zone_available_ns", "target_available_ns", "episode_available_ns"):
        if name in candidates and np.any(_integer_column(candidates, name) > origins):
            raise ValueError(f"{name} is later than the decision")
    if "risk_per_oz" not in candidates:
        raise ValueError("risk_per_oz required")
    risk = candidates.risk_per_oz.to_numpy(dtype=float)
    if not np.isfinite(risk).all() or np.any(risk <= 0) or np.any(risk >= (_MAX // 8) / NS):
        raise ValueError("risk_per_oz must be finite and positive within range")
    risk_nano = np.rint(risk * NS).astype(np.int64)
    if np.any(risk_nano <= 0):
        raise ValueError("risk_per_oz must round to at least one nanodollar")
    good = (quotes.valid.to_numpy() & (epoch >= 0) & (bid > 0) & (ask > bid)
            & (bid <= _MAX // 4) & (ask <= _MAX // 4))
    mid2 = np.zeros(len(qt), np.int64)
    mid2[good] = bid[good] + ask[good]
    z = _first_passage(qt, mid2, good, epoch, origins, sides, targets,
                       risk_nano, coverage, asof, duration, gap)
    out = candidates[["candidate_id", "origin_index", "t_ns", "side", "target_nano", "risk_per_oz"]].reset_index(drop=True).copy()
    out["state"] = np.asarray(STATES)[z[:, 0]]
    out["reason"] = np.asarray(REASONS)[z[:, 1]]
    out["label_known"] = out.state.ne("CENSORED")
    out["event_observed"] = out.state.isin(["RETURN_FIRST", "ADVERSE_FIRST"])
    out["horizon_ns"] = origins + duration
    out["horizon_seconds"] = duration / NS
    out["origin_mid_x2_nano"] = z[:, 8]
    out["adverse_x2_nano"] = z[:, 9]
    out["risk_nano"] = risk_nano
    out["first_event_ns"] = z[:, 4]
    out["label_available_ns"] = z[:, 5]
    out["censor_ns"] = z[:, 6]
    out["observed_until_ns"] = z[:, 5]
    out["observed_duration_ns"] = np.maximum(0, z[:, 5] - origins)
    elapsed = np.where(z[:, 4] >= 0, z[:, 4] - origins, -1)
    out["first_event_elapsed_ns"] = elapsed
    bins = np.asarray(TIME_BINS_SECONDS, np.int64)
    bin_index = np.searchsorted(bins * NS, np.maximum(0, elapsed), side="left")
    out["first_event_bin_index"] = np.where(elapsed >= 0, bin_index, -1).astype(np.int64)
    out["first_event_bin_seconds"] = np.where(elapsed >= 0, bins[np.minimum(bin_index, len(bins) - 1)], -1)
    for key, column in (("origin", 2), ("event", 3), ("terminal", 7)):
        indices = z[:, column]
        out[key + "_quote_index"] = indices
        out[key + "_quote_ns"] = np.where(indices >= 0, qt[np.maximum(indices, 0)], -1)
        out[key + "_source_ordinal"] = np.where(indices >= 0, ordinal[np.maximum(indices, 0)], -1)
    return out
