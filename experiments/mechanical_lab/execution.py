"""Independent mechanical trials with symmetric publication-clock taker latency.

Only the signal time and its frozen risk hint are inputs at admission. Quotes
after that time are consequences, not features. Every barrier requests an order;
none is a guaranteed limit/stop fill. Prices and outcomes are USD per ounce; one
standard GC contract is 100 ounces. This experiment assumes executable BBO size
and does not model queues, depth consumption, margin, or market impact.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numba import njit

NS = 1_000_000_000
TICK_SIZE = .1
CONTRACT_OZ = 100.
STATUSES = ("INELIGIBLE", "EMPTY", "CENSORED", "COMPLETE")
REASONS = ("ORIGIN", "SESSION", "RISK", "ENTRY_TIMEOUT", "SUPPORT",
           "TAKE", "STOP", "TIME", "EXIT_TIMEOUT", "COVERAGE")


@dataclass(frozen=True)
class ExecutionConfig:
    fee_roundtrip_per_oz: float = .30
    latency_ms: float = 100.
    pending_timeout_seconds: float = 2.
    max_quote_gap_seconds: float = 30.
    slippage_ticks_per_fill: float = 0.

    def __post_init__(self):
        vals = np.array(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError("Execution parameters must be finite and nonnegative")
        if self.max_quote_gap_seconds <= 0 or self.pending_timeout_seconds <= self.latency_ms / 1000:
            raise ValueError("Positive quote support and timeout greater than latency required")


@dataclass(frozen=True)
class ExitPlan:
    name: str = "atr_rr"
    risk_mode: str = "atr"  # atr uses the decision's risk hint; fixed ignores it.
    fixed_stop_ticks: float = 12.
    risk_multiple: float = 1.
    min_risk_per_oz: float = .8
    max_risk_per_oz: float = 6.
    take_r: float = 1.5
    max_hold_seconds: float = 600.
    stop_mode: str = "fixed"  # fixed, breakeven, ratchet, chandelier
    be_trigger_r: float = 1.
    be_lock_r: float = 0.
    trail_start_r: float = 1.5
    trail_distance_r: float = 1.
    trail_step_r: float = .5

    def __post_init__(self):
        if not self.name or self.risk_mode not in ("atr", "fixed"):
            raise ValueError("Named plan with atr/fixed risk required")
        if self.stop_mode not in ("fixed", "breakeven", "ratchet", "chandelier"):
            raise ValueError("Unknown stop mode")
        vals = [v for k, v in self.__dict__.items() if k not in ("name", "risk_mode", "stop_mode")]
        if not np.isfinite(np.asarray(vals, float)).all():
            raise ValueError("Finite plan parameters required")
        if min(self.fixed_stop_ticks, self.risk_multiple, self.min_risk_per_oz,
               self.take_r, self.max_hold_seconds, self.be_trigger_r,
               self.trail_start_r, self.trail_distance_r, self.trail_step_r) <= 0:
            raise ValueError("Risk, distances, thresholds, and holding duration must be positive")
        if self.max_risk_per_oz < self.min_risk_per_oz:
            raise ValueError("Risk maximum must not be below minimum")
        if not 0 <= self.be_lock_r < self.be_trigger_r:
            raise ValueError("Break-even lock must be nonnegative and below activation")


EXIT_PLANS = {
    "fixed_rr": ExitPlan(name="fixed_rr", risk_mode="fixed", fixed_stop_ticks=12., max_hold_seconds=300.),
    "atr_rr": ExitPlan(),
    "wide_rr": ExitPlan(name="wide_rr", risk_multiple=1.5, min_risk_per_oz=1.2,
                        max_risk_per_oz=8., take_r=2.5, max_hold_seconds=900.),
    "breakeven": ExitPlan(name="breakeven", take_r=2., stop_mode="breakeven"),
    "ratchet": ExitPlan(name="ratchet", risk_multiple=1.2, max_risk_per_oz=8.,
                        take_r=3., max_hold_seconds=900., stop_mode="ratchet"),
    "chandelier": ExitPlan(name="chandelier", take_r=3., max_hold_seconds=900.,
                           stop_mode="chandelier"),
    "quick_time": ExitPlan(name="quick_time", risk_multiple=.8, max_risk_per_oz=5.,
                           take_r=1., max_hold_seconds=120.),
}


def resolve_plan(plan):
    if isinstance(plan, str):
        return EXIT_PLANS[plan]
    if isinstance(plan, ExitPlan):
        return plan
    if isinstance(plan, Mapping):
        return ExitPlan(**plan)
    raise TypeError("Use a named plan, ExitPlan, or plan field mapping")


@dataclass(frozen=True)
class QuoteTape:
    quote_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray
    epoch: np.ndarray
    coverage_end_ns: int
    entry_start_ns: int
    deadline_ns: int

    def __post_init__(self):
        n = len(self.quote_t)
        for name, dtype in (("quote_t", np.int64), ("bid", np.int64), ("ask", np.int64),
                            ("valid", bool), ("epoch", np.int64)):
            a = np.asarray(getattr(self, name), dtype=dtype)
            if a.shape != (n,):
                raise ValueError("Quote arrays must be aligned and one dimensional")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not n or np.any(self.quote_t[1:] < self.quote_t[:-1]):
            raise ValueError("Nonempty monotonically published quote tape required")
        if self.coverage_end_ns <= self.quote_t[-1]:
            raise ValueError("Coverage end is exclusive and must follow the last publication")
        if self.deadline_ns <= self.entry_start_ns:
            raise ValueError("Session deadline must follow entry start")

    @classmethod
    def from_frame(cls, frame, *, coverage_end_ns, entry_start_ns=None, deadline_ns=None):
        clock = "available_ns" if "available_ns" in frame else "t_ns"
        qt = frame[clock].to_numpy(dtype=np.int64)
        if not len(qt):
            raise ValueError("Nonempty quote tape required")
        if clock == "available_ns" and "t_ns" in frame and np.any(qt < frame.t_ns.to_numpy(dtype=np.int64)):
            raise ValueError("Quote availability cannot precede its source timestamp")
        return cls(qt, frame.bid_nano.to_numpy(dtype=np.int64), frame.ask_nano.to_numpy(dtype=np.int64),
                   frame.valid.to_numpy(dtype=bool), frame.epoch.to_numpy(dtype=np.int64),
                   int(coverage_end_ns), int(qt[0] if entry_start_ns is None else entry_start_ns),
                   int(coverage_end_ns if deadline_ns is None else deadline_ns))


@njit(cache=True)
def _good(j, bid, ask, valid, epoch):
    return valid[j] and epoch[j] >= 0 and bid[j] > 0 and ask[j] > bid[j]


@njit(cache=True)
def _step(j, qt, bid, ask, valid, epoch, token, gap):
    return (_good(j, bid, ask, valid, epoch) and epoch[j] == token
            and (j == 0 or qt[j] - qt[j-1] <= gap))


@njit(cache=True)
def _mark(side, bid, ask, entry_nano, j, fee, slip):
    # Subtract exact nano prices before converting to float: subtracting two
    # approximately $4,000 floats can miss an exact one-tick barrier.
    exit_nano = bid[j] if side == 1 else ask[j]
    return side * (exit_nano - entry_nano) / 1e9 - fee - 2 * slip


@njit(cache=True)
def _kernel(qt, bid, ask, valid, epoch, coverage, start, end, origins, risks,
            fee, slip, latency, pending, gap, take_r, duration, stop_mode,
            be_trigger_r, be_lock_r, trail_start_r, trail_distance_r, trail_step_r):
    # ints: status, reason, decision index, entry index, trigger index, exit
    # index, request ns, number of stop changes. values: net, min, max, maxDD,
    # entry price, exit price, final stop (all monetary fields USD / ounce).
    ints = np.full((2 * len(origins), 8), -1, np.int64)
    values = np.full((2 * len(origins), 7), np.nan)
    for i in range(len(origins)):
        t = origins[i]
        risk = risks[i]
        d = np.searchsorted(qt, t, side="right") - 1
        reason = -1
        if t < start or t + 2 * pending + duration >= end:
            reason = 1
        elif not np.isfinite(risk) or risk <= 0:
            reason = 2
        elif d < 0 or not _good(d, bid, ask, valid, epoch) or t - qt[d] > gap:
            reason = 0
        for si in range(2):
            r = 2 * i + si
            ints[r, 0] = 0
            ints[r, 1] = reason
            ints[r, 2] = d
            ints[r, 7] = 0
        if reason >= 0:
            continue
        e = np.searchsorted(qt, t + latency, side="right")
        scan_end = min(e, np.searchsorted(qt, t + pending, side="right") - 1)
        status = 3
        for j in range(d + 1, scan_end + 1):
            if not _step(j, qt, bid, ask, valid, epoch, epoch[d], gap):
                status, reason = 2, 4
                break
        if status == 3 and (e == len(qt) or qt[e] > t + pending):
            status, reason = 1, 3
            if coverage <= t + pending:
                status, reason = 2, 9
            elif t + pending - qt[scan_end] > gap:
                status, reason = 2, 4
        if status != 3:
            for si in range(2):
                ints[2 * i + si, 0] = status
                ints[2 * i + si, 1] = reason
            continue
        for si in range(2):
            r = 2 * i + si
            side = 1 if si == 0 else -1
            entry_nano = ask[e] if side == 1 else bid[e]
            entry = entry_nano / 1e9 + side * slip
            ints[r, 0] = 2
            ints[r, 1] = 9
            ints[r, 3] = e
            values[r, 4] = entry
            stop = -risk
            low, high, peak, dd = np.inf, -np.inf, 0., 0.
            last = e - 1
            trigger, request_ns, reason = -1, -1, 9
            due = qt[e] + duration
            for j in range(e, len(qt)):
                # A clock timer wins against a publication at exactly its time.
                if qt[j] >= due:
                    if last >= e and due - qt[last] > gap:
                        reason = 4
                    else:
                        request_ns, reason = due, 7
                    break
                if qt[j] >= end:
                    reason = 1
                    break
                if not _step(j, qt, bid, ask, valid, epoch, epoch[e], gap):
                    reason = 4
                    break
                mark = _mark(side, bid, ask, entry_nano, j, fee, slip)
                low, high = min(low, mark), max(high, mark)
                peak = max(peak, mark)
                dd = max(dd, peak - mark)
                last = j
                # Existing stop first, then take. Tightening becomes available
                # after this observation; it never fills at an earlier price.
                if mark <= stop + 1e-12:
                    trigger, request_ns, reason = j, qt[j], 6
                    break
                if mark >= take_r * risk - 1e-12:
                    trigger, request_ns, reason = j, qt[j], 5
                    break
                new_stop = stop
                if stop_mode in (1, 2) and mark >= be_trigger_r * risk - 1e-12:
                    new_stop = max(new_stop, be_lock_r * risk)
                if stop_mode == 2 and peak >= trail_start_r * risk - 1e-12:
                    lock = np.floor((peak / risk - trail_distance_r) / trail_step_r + 1e-12) * trail_step_r * risk
                    new_stop = max(new_stop, lock)
                if stop_mode == 3 and peak >= trail_start_r * risk - 1e-12:
                    new_stop = max(new_stop, peak - trail_distance_r * risk)
                if new_stop > stop + 1e-12:
                    ints[r, 7] += 1
                    stop = new_stop
            if request_ns < 0 and reason == 9 and coverage > due and last >= e:
                if due - qt[last] <= gap:
                    request_ns, reason = due, 7
                else:
                    reason = 4
            ints[r, 1] = reason
            values[r, 6] = stop
            if last >= e:
                values[r, 1], values[r, 2], values[r, 3] = low, high, dd
            if request_ns < 0:
                continue
            ints[r, 4], ints[r, 6] = trigger, request_ns
            x = np.searchsorted(qt, request_ns + latency, side="right")
            scan_end = min(x, np.searchsorted(qt, request_ns + pending, side="right") - 1)
            supported = True
            for j in range(last + 1, scan_end + 1):
                if not _step(j, qt, bid, ask, valid, epoch, epoch[e], gap):
                    supported, ints[r, 1] = False, 4
                    break
                if qt[j] >= end:
                    supported, ints[r, 1] = False, 1
                    break
                mark = _mark(side, bid, ask, entry_nano, j, fee, slip)
                low, high = min(low, mark), max(high, mark)
                peak = max(peak, mark)
                dd = max(dd, peak - mark)
            values[r, 1], values[r, 2], values[r, 3] = low, high, dd
            if not supported:
                continue
            if x == len(qt) or qt[x] > request_ns + pending:
                ints[r, 1] = 8 if coverage > request_ns + pending else 9
                if coverage > request_ns + pending and request_ns + pending - qt[scan_end] > gap:
                    ints[r, 1] = 4
                continue
            ints[r, 0], ints[r, 5] = 3, x
            values[r, 0] = _mark(side, bid, ask, entry_nano, x, fee, slip)
            values[r, 5] = bid[x] / 1e9 - slip if side == 1 else ask[x] / 1e9 + slip
    return ints, values


def simulate_pairs(tape, origins_ns, risk_hint_per_oz, exit_plan="atr_rr", config=None):
    """Paired LONG/SHORT trials at each supplied causal origin, in input order.

    Rows are independent counterfactuals, not a sequential account. CENSORED is
    never a zero return. Stop/target thresholds are *net liquidation* PnL: the
    initial stop is -R after round-trip fees and adverse slippage; break-even is
    zero on that same scale. A stop request can still fill at a loss after latency.
    ATR risk is frozen at the origin and clipped; fixed risk is independent of it.
    """
    p = resolve_plan(exit_plan)
    c = ExecutionConfig() if config is None else (ExecutionConfig(**config) if isinstance(config, Mapping) else config)
    if not isinstance(c, ExecutionConfig):
        raise TypeError("ExecutionConfig or field mapping required")
    t = np.asarray(origins_ns)
    hint = np.asarray(risk_hint_per_oz, dtype=float)
    if t.ndim != 1 or t.dtype.kind not in "iu" or hint.shape != t.shape:
        raise ValueError("One integer-nanosecond origin and risk hint per trial required")
    t = t.astype(np.int64, copy=False)
    raw_risk = np.full(len(t), p.fixed_stop_ticks * TICK_SIZE) if p.risk_mode == "fixed" else hint * p.risk_multiple
    risk = np.clip(raw_risk, p.min_risk_per_oz, p.max_risk_per_oz)
    # Invalid ATR is not repaired by clipping or rounding.
    risk[~np.isfinite(raw_risk) | (raw_risk <= 0)] = np.nan
    z, v = _kernel(tape.quote_t, tape.bid, tape.ask, tape.valid, tape.epoch,
                   int(tape.coverage_end_ns), int(tape.entry_start_ns), int(tape.deadline_ns),
                   t, risk, c.fee_roundtrip_per_oz, c.slippage_ticks_per_fill * TICK_SIZE,
                   int(round(c.latency_ms * 1e6)), int(round(c.pending_timeout_seconds * NS)),
                   int(round(c.max_quote_gap_seconds * NS)), p.take_r,
                   int(round(p.max_hold_seconds * NS)),
                   ("fixed", "breakeven", "ratchet", "chandelier").index(p.stop_mode),
                   p.be_trigger_r, p.be_lock_r, p.trail_start_r, p.trail_distance_r, p.trail_step_r)
    out = pd.DataFrame({"origin_index": np.repeat(np.arange(len(t)), 2),
                        "signal_ns": np.repeat(t, 2), "side": np.tile([1, -1], len(t)),
                        "plan": p.name, "risk_per_oz": np.repeat(risk, 2),
                        "status": np.asarray(STATUSES)[z[:, 0]],
                        "reason": np.asarray(REASONS)[z[:, 1]],
                        "net_per_oz": v[:, 0], "mae_per_oz": v[:, 1], "mfe_per_oz": v[:, 2],
                        "max_drawdown_per_oz": v[:, 3], "entry_price": v[:, 4], "exit_price": v[:, 5],
                        "final_stop_net_per_oz": v[:, 6], "stop_updates": z[:, 7]})
    for k, name in enumerate(("decision", "entry", "trigger", "exit"), 2):
        out[name + "_ordinal"] = z[:, k]
        out[name + "_ns"] = np.where(z[:, k] >= 0, tape.quote_t[np.maximum(z[:, k], 0)], 0)
    out["trigger_ns"] = np.maximum(z[:, 6], 0)
    return out
