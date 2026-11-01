"""Event-origin game with frozen price targets and actual BBO consequences.

The caller chooses causal episodes, zones, origins, risk, and one absolute target.
This adapter adds no entry signal and never scans future outcomes for admission.
Both sides retain the same decision-time gross target distance: the signal side
aims at its frozen zone edge, and the other side aims at a mirrored control edge.
An unconditional market entry that overruns its target is still charged and
immediately requests a latency-bearing exit. It is never erased as a bad fill.
"""
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numba import njit

from experiments.mechanical_lab.execution import (
    ExecutionConfig, QuoteTape, NS, TICK_SIZE, STATUSES,
    REASONS as BASE_REASONS, _kernel, _good,
)
from experiments.mechanical_lab.portfolio import AccountConfig, account

REASONS = (*BASE_REASONS, "TARGET_GEOMETRY")
REQUIRED = ("candidate_id", "origin_index", "t_ns", "side", "target_nano", "risk_per_oz")
ACCOUNT_REQUIRED = ("date", "regime", "time_regime")


def validate_candidates(candidates):
    missing = set(REQUIRED) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate fields: {sorted(missing)}")
    if candidates.empty:
        return
    for col in ("candidate_id", "origin_index", "t_ns", "side", "target_nano"):
        if candidates[col].dtype.kind not in "iu":
            raise ValueError(f"{col} requires exact integers")
    if candidates.candidate_id.duplicated().any() or candidates.origin_index.duplicated().any():
        raise ValueError("Each candidate and origin index must be unique")
    if not candidates.side.isin([-1, 1]).all():
        raise ValueError("Signal side must be +1 or -1")
    if (candidates.target_nano <= 0).any():
        raise ValueError("GC target price must be positive")
    if "deadline_ns" in candidates:
        if candidates.deadline_ns.dtype.kind not in "iu" or (candidates.deadline_ns <= 0).any():
            raise ValueError("deadline_ns requires positive exact integer nanoseconds")
    for col in ("zone_available_ns", "target_available_ns", "episode_available_ns", "deadline_available_ns"):
        if col in candidates:
            if candidates[col].dtype.kind not in "iu":
                raise ValueError(f"{col} requires exact integer nanoseconds")
            if (candidates[col] > candidates.t_ns).any():
                raise ValueError(f"{col} is later than the decision; target must be causal")


@njit(cache=True)
def _zone_trials(qt, bid, ask, valid, epoch, coverage, start, end,
                 origins, sides, zone_targets, risks, deadlines, fee, slip,
                 latency, pending, gap, duration):
    ints = np.full((2 * len(origins), 8), -1, np.int64)
    values = np.full((2 * len(origins), 7), np.nan)
    targets = np.zeros(2 * len(origins), np.int64)
    signal_net = np.full(2 * len(origins), np.nan)
    entry_net = np.full(2 * len(origins), np.nan)
    overruns = np.zeros(2 * len(origins), np.bool_)
    effective_holds = np.full(2 * len(origins), np.nan)
    for i in range(len(origins)):
        t, risk = origins[i], risks[i]
        d = np.searchsorted(qt, t, side="right") - 1
        e = np.searchsorted(qt, t + latency, side="right")
        known_duration = duration
        effective_duration = duration
        if deadlines[i] > 0:
            known_duration = min(duration, max(0, deadlines[i] - t))
            if e < len(qt):
                effective_duration = min(duration, max(0, deadlines[i] - qt[e]))
        # Geometry is decided using the last available publication, never e.
        has_origin = d >= 0 and _good(d, bid, ask, valid, epoch) and t - qt[d] <= gap
        distance = 0
        if has_origin:
            entry_at_signal = ask[d] if sides[i] == 1 else bid[d]
            distance = sides[i] * (zone_targets[i] - entry_at_signal)
        for si in range(2):
            r, side = 2 * i + si, 1 if si == 0 else -1
            if has_origin:
                target = zone_targets[i] if side == sides[i] else ((ask[d] if side == 1 else bid[d]) + side * distance)
                targets[r] = target
                signal_net[r] = distance / 1e9 - fee - 2 * slip
            else:
                target = zone_targets[i]
                targets[r] = target
            # Eligibility may use the known session/deadline clock, never the
            # eventual fill-dependent reduction of holding duration.
            if t < start or t + 2 * pending + known_duration >= end:
                ints[r, 0], ints[r, 1], ints[r, 2], ints[r, 7] = 0, 1, d, 0
                continue
            if has_origin and distance <= 0:
                # Only a decision-time geometric impossibility is ineligible.
                # Latency overruns below ALWAYS remain filled consequences.
                ints[r, 0], ints[r, 1], ints[r, 2], ints[r, 7] = 0, 10, d, 0
                continue
            gain = 1.
            overrun = False
            if e < len(qt) and np.isfinite(risk) and risk > 0:
                entry_nano = ask[e] if side == 1 else bid[e]
                gross_distance = side * (target - entry_nano) / 1e9
                entry_net[r] = gross_distance - fee - 2 * slip
                gain = entry_net[r] / risk
                overrun = gross_distance - slip <= 1e-12
                # Negative target threshold forces a close request at entry;
                # the verified engine still applies support, stop priority,
                # quote timeout, and exit latency before returning any PnL.
                if overrun:
                    gain = -1e100
            z, v = _kernel(qt, bid, ask, valid, epoch, coverage, start, end,
                origins[i:i+1], risks[i:i+1], fee, slip, latency, pending, gap,
                gain, effective_duration, 0, 1., 0., 1.5, 1., .5)
            ints[r], values[r] = z[si], v[si]
            if ints[r, 3] >= 0:
                overruns[r] = overrun
                effective_holds[r] = effective_duration / NS
            else:
                # An unfilled/ineligible row does not know an entry price.
                entry_net[r] = np.nan
    return ints, values, targets, signal_net, entry_net, overruns, effective_holds


def simulate_zone_pairs(tape: QuoteTape, candidates, *, holding_seconds=300., config=None):
    """Return portfolio-compatible LONG/SHORT consequences per causal origin.

    Required: candidate_id, origin_index, t_ns, side, target_nano, risk_per_oz.
    Optional metadata is copied unchanged except side/target_nano, which become
    the executed side and its frozen target; originals are signal_side and
    signal_target_nano. Risk is used exactly as supplied; there is no hidden ATR
    rescaling, clipping, future target update, or outcome-dependent admission.
    Optional deadline_ns is a frozen absolute close-request deadline; holding
    time becomes min(TTL, max(0, deadline-entry_time)). An already expired
    pending entry still fills and requests an immediate costed exit.

    The target is a quote-price trigger (long bid>=target; short ask<=target),
    followed by a market exit with the same latency as entry. It is not a
    guaranteed limit fill. TARGET_OVERRUN is descriptive, never an exclusion.
    """
    validate_candidates(candidates)
    if not np.isfinite(holding_seconds) or holding_seconds <= 0:
        raise ValueError("Positive finite holding duration required")
    c = ExecutionConfig() if config is None else ExecutionConfig(**config) if isinstance(config, Mapping) else config
    if not isinstance(c, ExecutionConfig):
        raise TypeError("ExecutionConfig or field mapping required")
    t = candidates.t_ns.to_numpy(np.int64)
    sides = candidates.side.to_numpy(np.int64)
    original_targets = candidates.target_nano.to_numpy(np.int64)
    risk = candidates.risk_per_oz.to_numpy(float)
    deadlines = candidates.deadline_ns.to_numpy(np.int64) if "deadline_ns" in candidates else np.zeros(len(candidates),np.int64)
    z, v, targets, signal_net, entry_net, overruns, effective_holds = _zone_trials(
        tape.quote_t, tape.bid, tape.ask, tape.valid, tape.epoch,
        int(tape.coverage_end_ns), int(tape.entry_start_ns), int(tape.deadline_ns),
        t, sides, original_targets, risk, deadlines, c.fee_roundtrip_per_oz,
        c.slippage_ticks_per_fill * TICK_SIZE, int(round(c.latency_ms * 1e6)),
        int(round(c.pending_timeout_seconds * NS)), int(round(c.max_quote_gap_seconds * NS)),
        int(round(holding_seconds * NS)))
    out = candidates.iloc[np.repeat(np.arange(len(candidates)), 2)].reset_index(drop=True).copy()
    out["signal_side"] = np.repeat(sides, 2)
    out["signal_target_nano"] = np.repeat(original_targets, 2)
    out["side"] = np.tile([1, -1], len(candidates))
    out["signal_ns"] = np.repeat(t, 2)
    out["target_nano"] = targets
    out["target_mode"] = np.where(out.side == out.signal_side, "signal_zone_edge", "mirrored_distance_control")
    out["plan"] = f"zone_target_{holding_seconds:g}s"
    out["holding_limit_seconds"] = float(holding_seconds)
    out["effective_holding_limit_seconds"] = effective_holds
    out["risk_per_oz"] = np.repeat(risk, 2)
    out["target_expected_net_at_signal"] = signal_net
    out["target_net_at_entry"] = entry_net
    out["target_overrun_at_entry"] = overruns
    out["status"] = np.asarray(STATUSES)[z[:, 0]]
    out["reason"] = np.asarray(REASONS)[z[:, 1]]
    for index, key in enumerate(("net_per_oz", "mae_per_oz", "mfe_per_oz", "max_drawdown_per_oz",
                                  "entry_price", "exit_price", "final_stop_net_per_oz")):
        out[key] = v[:, index]
    out["stop_updates"] = z[:, 7]
    for k, name in enumerate(("decision", "entry", "trigger", "exit"), 2):
        out[name + "_ordinal"] = z[:, k]
        out[name + "_ns"] = np.where(z[:, k] >= 0, tape.quote_t[np.maximum(z[:, k], 0)], 0)
    out["trigger_ns"] = np.maximum(z[:, 6], 0)
    # Failed pending exits remain SUPPORT/EXIT_TIMEOUT/COVERAGE, so censoring
    # cannot be hidden by the informative target-overrun annotation.
    closed_overrun = overruns & out.status.eq("COMPLETE").to_numpy()
    out.loc[closed_overrun, "reason"] = "TARGET_OVERRUN"
    out["deadline_expired_at_entry"] = (np.repeat(deadlines,2) > 0) & out.entry_ns.gt(0).to_numpy() & (out.entry_ns.to_numpy() >= np.repeat(deadlines,2))
    out.loc[out.deadline_expired_at_entry & out.status.eq("COMPLETE"), "reason"] = "DEADLINE_EXPIRED_AT_ENTRY"
    return out


def run_account(candidates, paired, *, direction="toward", seed=0, sizing="fixed", config=None):
    """Run one-position capital accounting, reusing the verified policy clock.

    toward/away/coin choose signal/reverse/random side at the SAME offered
    origins. Busy positions or pending entry timeouts prevent another entry.
    Filled censoring halts the account; no partial unknown equity is called a
    complete curve. Fresh calls start fresh capital, as in the original lab.
    """
    validate_candidates(candidates)
    missing = set(ACCOUNT_REQUIRED) - set(candidates.columns)
    if missing:
        raise ValueError(f"Account requires candidate fields: {sorted(missing)}")
    aliases = {"toward": "signal", "away": "reverse", "coin": "coin",
               "signal": "signal", "reverse": "reverse"}
    if direction not in aliases:
        raise ValueError("Direction must be toward, away, or coin")
    if paired.duplicated(["origin_index", "side"]).any():
        raise ValueError("One outcome per origin and side required; choose one TTL/execution condition")
    expected = {(int(i), side) for i in candidates.origin_index for side in (1, -1)}
    supplied = set(zip(paired.origin_index.astype(int), paired.side.astype(int)))
    if not expected.issubset(supplied):
        raise ValueError("Missing counterfactual outcome; never filter failed or censored trials")
    stats, trades = account(candidates, paired, sizing=sizing, direction=aliases[direction], seed=seed,
                            config=config or AccountConfig())
    stats["direction_control"] = direction
    stats["entry_contract"] = "Unconditional market; passed target causes immediate costed exit request"
    if len(trades):
        if np.any(trades.signal_ns.to_numpy()[1:] <= trades.exit_ns.to_numpy()[:-1]):
            raise AssertionError("One-position chronological accounting violated")
        keep = [x for x in paired.columns if x not in trades.columns or x in ("origin_index", "side")]
        trades = trades.merge(paired[keep], on=["origin_index", "side"], how="left", validate="one_to_one")
        if len(trades) != stats["trades"]:
            raise AssertionError("Account trade identity lost during metadata merge")
    return stats, trades
