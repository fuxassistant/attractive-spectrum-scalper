"""Serializable exact-BBO option replay for HOLD/CLOSE research.

Entry and discretionary CLOSE fill at the first supported publication strictly
after intent + latency and no later than intent + timeout. Fees are reserved
once, on entry. Existing v3 mechanical stop/target/time/session/death protections
execute immediately at the triggering supported publication; they have priority
over a pending discretionary CLOSE. This deliberate asymmetry is a research
contract, not a promise of live execution. Quote silence is not support loss.
An invalid publication or segment change while exposed censors without a fill.
No re-entry is performed by reference_exit. A close's cash is carried unchanged
to the common branch horizon. Arrays are read-only shared views, never cloned.
"""
from dataclasses import dataclass, asdict, fields
import copy
import numpy as np
from numba import njit
from spectrum.player.contract import PLANS, NS

VERSION = "CAUSAL_HOLD_CLOSE_BBO_V1"
MONEY_TOLERANCE = 1e-8
REASONS = {0: "none", 1: "take", 2: "stop", 3: "time", 4: "session_close",
           5: "series_barrier", 6: "support_loss", 7: "policy_close",
           8: "end_of_tape", 9: "horizon_unresolved"}


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 1500.
    fee_roundtrip_per_oz: float = .30
    latency_ns: int = 100_000_000
    pending_timeout_ns: int = 2 * NS
    loss_limit_fraction: float = .5
    plans: tuple = PLANS

    def __post_init__(self):
        if self.initial_capital <= 0 or self.fee_roundtrip_per_oz < 0:
            raise ValueError("Positive capital and nonnegative fees required")
        if not 0 <= self.latency_ns < self.pending_timeout_ns:
            raise ValueError("Timeout must exceed nonnegative latency")
        if self.loss_limit_fraction != .5:
            raise ValueError("Series death contract is fixed at strictly >50% loss")
        if not self.plans or any(len(p) != 3 or min(p) <= 0 for p in self.plans):
            raise ValueError("Plans are positive (take ticks, stop ticks, max seconds)")


@dataclass
class PositionState:
    side: int = 0
    entry_price_nano: int = 0
    entry_ns: int = 0
    plan_id: int = -1
    pending_kind: int = 0  # 0 none, 1 entry, 2 discretionary close
    pending_side: int = 0
    pending_plan: int = -1
    pending_ready_ns: int = 0
    pending_expires_ns: int = 0
    pending_epoch: int = -1
    pending_reason: int = 0
    quote_cursor: int = 0
    quote_epoch: int = -1
    clock_ns: int = 0
    last_quote_ns: int = 0
    entry_index: int = -1
    series_active: int = 0
    series_start_ns: int = 0
    series_id: int = 0
    trades: int = 0
    termination: int = 0  # 0 live, 1 ruined, 2 censored
    censor_ns: int = 0
    ruin_ns: int = 0
    closed_reason: int = 0
    exit_ns: int = 0
    exit_price_nano: int = 0
    exit_side: int = 0
    target_distance_nano: int = 0
    stop_distance_nano: int = 0
    max_hold_ns: int = 0
    last_bid_nano: int = 0
    last_ask_nano: int = 0
    last_valid: int = 0
    cash: float = 1500.
    equity: float = 1500.
    peak_equity: float = 1500.
    max_drawdown: float = 0.
    qty: float = 0.
    position_mfe: float = 0.
    position_mae: float = 0.
    series_base: float = 0.
    series_min_equity: float = 1500.
    max_series_loss_fraction: float = 0.
    last_trade_net: float = 0.
    last_trade_gross: float = 0.
    pending_qty: float = 0.
    initial_capital: float = 1500.
    total_fees: float = 0.
    last_trade_qty: float = 0.
    min_equity: float = 1500.

    @property
    def censored(self): return self.termination == 2

    @property
    def dead(self): return self.termination == 1

    @property
    def exit_reason(self): return REASONS[self.closed_reason]

    @property
    def series_debt(self):
        return max(0., self.series_base - self.equity) if self.series_active else 0.

    @property
    def series_risk_buffer(self):
        return self.equity - .5 * self.series_base if self.series_active else .5 * self.cash

    @property
    def time_left_ns(self):
        return max(0, self.entry_ns + self.max_hold_ns - self.clock_ns) if self.side else 0


_INTEGER_NAMES = tuple(f.name for f in fields(PositionState) if f.type is int)
_FLOAT_NAMES = tuple(f.name for f in fields(PositionState) if f.type is float)


@dataclass(frozen=True)
class QuoteTape:
    quote_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray
    epoch: np.ndarray
    coverage_end_ns: int
    entry_start_ns: int = 0
    entry_cutoff_ns: int = np.iinfo(np.int64).max
    deadline_ns: int = np.iinfo(np.int64).max
    identity: str = "synthetic"

    def __post_init__(self):
        n = len(self.quote_t)
        if not n or any(len(getattr(self, key)) != n for key in ("bid", "ask", "valid", "epoch")):
            raise ValueError("Nonempty equally sized certified quote columns required")
        if self.coverage_end_ns <= int(self.quote_t[-1]):
            raise ValueError("Coverage end is exclusive and must follow final publication")
        for name in ("quote_t", "bid", "ask", "valid", "epoch"):
            view = np.asarray(getattr(self, name)).view()
            view.setflags(write=False)
            object.__setattr__(self, name, view)

    @classmethod
    def from_day(cls, day):
        return cls(day.quote_t, day.bid, day.ask, day.valid, day.epoch,
                   int(day.meta["end_ns"]), int(day.meta["london_ns"]),
                   int(day.meta["ny_close_ns"]), int(day.deadline),
                   str(day.path.resolve()))


@njit(cache=True)
def _flat_prefix(hi, qt, bid, ask, valid, epoch, s):
    """No exposure or pending intent: only final causal quote facts can matter."""
    lo = s[11]
    if hi > lo:
        j = hi - 1
        s[11] = hi
        s[13] = qt[j]
        s[14] = qt[j]
        s[12] = epoch[j]
        s[32] = int(valid[j] and bid[j] > 0 and ask[j] >= bid[j])
        # Usually one step; preserve latest supported price across an invalid
        # suffix without scanning the already consumed prefix.
        while j >= lo:
            if valid[j] and bid[j] > 0 and ask[j] >= bid[j]:
                s[30] = bid[j]
                s[31] = ask[j]
                break
            j -= 1


@njit(cache=True)
def _advance(hi, qt, bid, ask, valid, epoch, s, f, plans, fee, cutoff, deadline):
    """Single linear quote scan. State indices are schema-locked by tests."""
    if s[0] == 0 and s[4] == 0:
        _flat_prefix(hi, qt, bid, ask, valid, epoch, s)
        return
    for j in range(s[11], hi):
        now = qt[j]
        supported = valid[j] and bid[j] > 0 and ask[j] >= bid[j]
        old_epoch = s[12]
        s[11] = j + 1
        s[13] = now
        s[14] = now
        s[12] = epoch[j]
        s[32] = int(supported)
        # Retain last supported price and equity when support is lost.
        if s[0] != 0 and (not supported or epoch[j] != old_epoch):
            s[20] = 2
            s[21] = now
            s[23] = 6
            s[4] = 0
            return
        if supported:
            s[30] = bid[j]
            s[31] = ask[j]
        if s[4] != 0 and (now > s[8] or epoch[j] != s[9] or not supported):
            s[4] = 0
        if s[4] == 1:
            if now >= cutoff or now >= deadline:
                s[4] = 0
            elif now > s[7] and supported:
                if not s[16]:
                    s[16] = 1
                    s[17] = now
                    s[18] += 1
                    f[7] = f[0]
                    f[8] = f[0]
                s[0] = s[5]
                s[1] = ask[j] if s[0] > 0 else bid[j]
                s[2] = now
                s[3] = s[6]
                s[15] = j
                s[27] = int(round(plans[s[3], 0] * 100_000_000))
                s[28] = int(round(plans[s[3], 1] * 100_000_000))
                s[29] = int(round(plans[s[3], 2] * NS))
                f[4] = f[12]
                f[5] = 0.
                f[6] = 0.
                f[0] -= fee * f[4]
                f[14] += fee * f[4]
                s[4] = 0
                s[23] = 0
                s[24] = 0
                s[25] = 0
                s[26] = 0
        if s[0] != 0:
            px = bid[j] if s[0] > 0 else ask[j]
            gross = s[0] * (px - s[1]) / 1e9 * f[4]
            f[1] = f[0] + gross
            net = gross - fee * f[4]
            f[5] = max(f[5], net)
            f[6] = min(f[6], net)
            f[2] = max(f[2], f[1])
            f[3] = max(f[3], f[2] - f[1])
            f[16] = min(f[16], f[1])
            f[8] = min(f[8], f[1])
            f[9] = max(f[9], max(0., (f[7] - f[1]) / f[7]))
            reason = 0
            if f[1] < .5 * f[7] - MONEY_TOLERANCE:
                reason = 5
                s[20] = 1
                s[22] = now
            elif now >= deadline:
                reason = 4
            elif s[0] * (px - s[1]) <= -s[28]:
                reason = 2
            elif s[0] * (px - s[1]) >= s[27]:
                reason = 1
            elif now - s[2] >= s[29]:
                reason = 3
            elif s[4] == 2 and now > s[7] and supported:
                reason = 7
            if reason:
                f[0] += gross
                f[1] = f[0]
                f[10] = net
                f[11] = gross
                f[15] = f[4]
                s[23] = reason
                s[24] = now
                s[25] = px
                s[26] = s[0]
                s[19] += 1
                s[0] = 0
                f[4] = 0.
                s[4] = 0
                if f[0] >= f[7] - MONEY_TOLERANCE and s[20] == 0:
                    s[16] = 0
                if s[20]:
                    return
                _flat_prefix(hi, qt, bid, ask, valid, epoch, s)
                return


FEATURE_NAMES = (
    "position_side", "size_oz", "cash_return", "liquidation_return", "unrealized_usd",
    "age_fraction", "time_left_fraction", "target_usd_per_oz", "stop_usd_per_oz",
    "target_remaining_usd_per_oz", "stop_remaining_usd_per_oz", "position_mfe_usd",
    "position_mae_usd", "drawdown_fraction", "series_active", "series_base_return",
    "series_debt_fraction", "series_risk_buffer_fraction", "pending_entry", "pending_close",
    "pending_time_to_ready_seconds", "pending_time_to_expiry_seconds", "spread_usd",
    "quote_age_seconds", "supported", "dead", "censored", "plan_id", "fee_paid_usd",
    "session_time_left_seconds")


class ExecutionReplay:
    """Mutable small state over an immutable certified tape; clone is O(1)."""
    feature_names = FEATURE_NAMES

    def __init__(self, day_or_tape, config=None, state=None):
        self.tape = day_or_tape if isinstance(day_or_tape, QuoteTape) else QuoteTape.from_day(day_or_tape)
        self.config = config or ExecutionConfig()
        self.plans = np.asarray(self.config.plans, dtype=np.float64)
        capital = self.config.initial_capital
        self.state = copy.deepcopy(state) if state is not None else PositionState(
            cash=capital, equity=capital, peak_equity=capital, initial_capital=capital,
            series_min_equity=capital, min_equity=capital)

    def snapshot(self):
        # Decimal strings preserve integer nanoseconds across JSON/JS consumers.
        state = asdict(self.state)
        for key in _INTEGER_NAMES:
            state[key] = str(state[key])
        return dict(version=VERSION, tape_identity=self.tape.identity,
                    config=asdict(self.config), state=state)

    def restore(self, snapshot):
        if snapshot["version"] != VERSION or snapshot["tape_identity"] != self.tape.identity:
            raise ValueError("Snapshot contract or tape differs")
        config = dict(snapshot["config"])
        config["plans"] = tuple(tuple(p) for p in config["plans"])
        if ExecutionConfig(**config) != self.config:
            raise ValueError("Snapshot execution configuration differs")
        values = dict(snapshot["state"])
        for key in _INTEGER_NAMES:
            values[key] = int(values[key])
        self.state = PositionState(**values)
        if not 0 <= self.state.quote_cursor <= len(self.tape.quote_t):
            raise ValueError("Snapshot cursor outside tape")
        return self

    def clone(self):
        return ExecutionReplay(self.tape, self.config, self.state)

    def advance_until(self, t_ns):
        t_ns = int(t_ns)
        state = self.state
        if t_ns < state.clock_ns:
            raise ValueError("Replay clock cannot move backwards")
        if state.termination:
            return state
        hi = int(np.searchsorted(self.tape.quote_t, min(t_ns, self.tape.coverage_end_ns - 1), side="right"))
        integers = np.array([getattr(state, n) for n in _INTEGER_NAMES], dtype=np.int64)
        numbers = np.array([getattr(state, n) for n in _FLOAT_NAMES], dtype=np.float64)
        _advance(hi, self.tape.quote_t, self.tape.bid, self.tape.ask, self.tape.valid,
                 self.tape.epoch, integers, numbers, self.plans,
                 self.config.fee_roundtrip_per_oz, self.tape.entry_cutoff_ns, self.tape.deadline_ns)
        for name, value in zip(_INTEGER_NAMES, integers): setattr(state, name, int(value))
        for name, value in zip(_FLOAT_NAMES, numbers): setattr(state, name, float(value))
        if not state.termination:
            state.clock_ns = min(t_ns, self.tape.coverage_end_ns)
            if state.pending_kind and state.clock_ns > state.pending_expires_ns:
                state.pending_kind = 0
            if t_ns >= self.tape.coverage_end_ns:
                state.pending_kind = 0
                if state.side:
                    self._censor(8, self.tape.coverage_end_ns)
        return state

    def open_at(self, t_ns, side, plan_id=0, qty=1.):
        self.advance_until(t_ns)
        s = self.state
        if side not in (-1, 1) or not 0 <= plan_id < len(self.plans) or not np.isfinite(qty) or qty <= 0:
            raise ValueError("Side, plan or quantity invalid")
        if (s.termination or s.side or s.pending_kind or not s.last_valid
                or not self.tape.entry_start_ns <= int(t_ns) < min(self.tape.entry_cutoff_ns, self.tape.deadline_ns)
                or int(t_ns) >= self.tape.coverage_end_ns):
            return False
        self._pending(1, int(t_ns), side, plan_id, qty, 0)
        return True

    def request_close(self, t_ns):
        self.advance_until(t_ns)
        s = self.state
        if s.termination or not s.side or s.pending_kind:
            return False
        self._pending(2, int(t_ns), s.side, s.plan_id, s.qty, 7)
        return True

    def _pending(self, kind, t_ns, side, plan_id, qty, reason):
        s = self.state
        s.pending_kind, s.pending_side, s.pending_plan = kind, side, plan_id
        s.pending_ready_ns = t_ns + self.config.latency_ns
        s.pending_expires_ns = t_ns + self.config.pending_timeout_ns
        s.pending_epoch, s.pending_reason, s.pending_qty = s.quote_epoch, reason, qty

    def _censor(self, reason, now):
        self.state.termination = 2
        self.state.censor_ns = int(now)
        self.state.closed_reason = reason
        self.state.pending_kind = 0

    def reference_exit(self, horizon_ns=None):
        """Advance a branch; unresolved exposure has no usable terminal target."""
        horizon = self.tape.coverage_end_ns if horizon_ns is None else int(horizon_ns)
        self.advance_until(horizon)
        if self.state.side and not self.state.termination:
            self._censor(9, horizon)
        return self.state

    def position_features(self):
        s = self.state
        initial = s.initial_capital
        per_oz = s.side * ((s.last_bid_nano if s.side > 0 else s.last_ask_nano) - s.entry_price_nano) / 1e9 if s.side else 0.
        age = max(0, s.clock_ns - s.entry_ns) / s.max_hold_ns if s.side and s.max_hold_ns else 0.
        values = (
            s.side, s.qty, s.cash / initial - 1, s.equity / initial - 1, s.equity - s.cash,
            age, s.time_left_ns / s.max_hold_ns if s.max_hold_ns else 0.,
            s.target_distance_nano / 1e9, s.stop_distance_nano / 1e9,
            s.target_distance_nano / 1e9 - per_oz, s.stop_distance_nano / 1e9 + per_oz,
            s.position_mfe, s.position_mae, s.max_drawdown / initial,
            s.series_active, s.series_base / initial - 1 if s.series_active else 0.,
            s.series_debt / s.series_base if s.series_active else 0.,
            s.series_risk_buffer / (s.series_base if s.series_active else initial),
            s.pending_kind == 1, s.pending_kind == 2,
            max(0, s.pending_ready_ns - s.clock_ns) / NS if s.pending_kind else 0.,
            max(0, s.pending_expires_ns - s.clock_ns) / NS if s.pending_kind else 0.,
            (s.last_ask_nano - s.last_bid_nano) / 1e9 if s.last_valid else 0.,
            max(0, s.clock_ns - s.last_quote_ns) / NS, s.last_valid, s.dead, s.censored,
            s.plan_id, s.total_fees, max(0, self.tape.deadline_ns - s.clock_ns) / NS)
        return np.asarray(values, dtype=np.float32)


def execution_contract():
    return dict(version=VERSION, initial_capital=1500., initial_qty_oz=1.,
        fee="0.30 USD per oz roundtrip, reserved once at entry",
        entry_and_policy_close="First supported quote strictly after intent+100ms, within intent+2s inclusive",
        protections="Death, daily deadline, stop, target, max duration: immediate triggering supported BBO; priority in that order",
        support="Invalid publication or epoch change while exposed censors without fill; observed silence remains supported",
        risk="Death strictly below 0.5*capital-before-series-first-entry, 1e-8 USD numerical tolerance; flat recovery alone resets series",
        valuation="Executable liquidation side, fees already included; all intervening publications scanned",
        paired_horizon="Same fixed calendar horizon; flat cash carried; unresolved exposure censored; no reentry",
        arrays="Immutable shared views; certified source masks consumed without repeated source analysis",
        full_state_fields=list(_INTEGER_NAMES + _FLOAT_NAMES), feature_names=list(FEATURE_NAMES))
