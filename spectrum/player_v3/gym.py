"""Exact-BBO replay with capital-relative martingale-series risk.

A series begins at its first executed entry, before the fee. Its base remains
fixed across losses, partial wins and sessions, until flat cash recovers that
base after costs. Only a loss strictly greater than half THAT series' base
kills the population. This is an oz quote proxy, not exchange GC margin.
"""
from dataclasses import dataclass, asdict, field
import numpy as np
from numba import njit
from spectrum.player.contract import Contract, NS, PLANS

RISK_VERSION = 'MARTINGALE_SERIES_RISK_V1'
MONEY_TOLERANCE = 1e-8  # dollars; float accumulation noise, not a risk buffer.
REASONS = {1: 'take', 2: 'stop', 3: 'time', 4: 'session_close',
           5: 'series_barrier', 6: 'support_loss'}
SERIES_STATUSES = {0: 'active', 1: 'recovered', 2: 'ruined', 3: 'censored'}
ACCOUNT_FEATURES = (
    'position_side', 'cash_change_over_initial_half', 'liquidation_change_over_initial_half',
    'lifetime_drawdown_over_initial_half', 'last_trade_net_over_5', 'population_ruined',
    'lifetime_trades_over_100_capped_5', 'series_active',
    'active_series_base_over_initial_minus_1', 'active_series_loss_fraction',
    'active_series_trade_count_over_100')


def risk_contract():
    return dict(version=RISK_VERSION, loss_limit_fraction=.5, money_tolerance_usd=MONEY_TOLERANCE,
                definition='One cumulative martingale series relative to its own capital before first executed entry.',
                start='First actual fill while no series is active; freeze cash before reserving the entry fee.',
                valuation='Liquidation bid/ask equity, including every series fee and current unrealized PnL.',
                death='Strictly more than 50% series-base loss: equity < 0.5*series_start_equity - money_tolerance_usd.',
                exact_half='Alive at exactly half the series base; no fixed initial-capital or peak-trailing barrier.',
                recovery='Only flat realized cash >= series_start_equity - money_tolerance_usd ends a series.',
                carry='Base, losses, partial wins and trade count persist across waits and session/day boundaries.',
                completed_series='Counts only supported flat recoveries; dead, censored and still-active series remain distinct.',
                support_censor='No certified death or recovery from a close after missing support.',
                account_features=list(ACCOUNT_FEATURES),
                objective_risk='0.25 * maximum supported series loss fraction; lifetime dollar drawdown is reported separately.')


@dataclass
class CapitalState:
    initial_capital: float = 200.
    cash: float = 200.
    peak: float = 200.
    max_drawdown: float = 0.
    ruined: bool = False
    ruin_ns: int = 0
    censored: bool = False
    censor_ns: int = 0
    trades: int = 0
    wins: int = 0
    last_trade_net: float = 0.
    loss_streak: int = 0
    risk_version: str = RISK_VERSION
    series_id: int = 0
    series_start_ns: int = 0
    series_start_equity: float = 0.
    series_active: bool = False
    completed_series: int = 0
    series_min_equity: float = 0.
    series_trade_count: int = 0
    max_series_loss_fraction: float = 0.
    series_history: list = field(default_factory=list)

    def dict(self):
        result = asdict(self)
        result['ruin_ns'] = str(result['ruin_ns'])
        result['censor_ns'] = str(result['censor_ns'])
        result['series_start_ns'] = str(result['series_start_ns'])
        return result


@njit(cache=True)
def advance_survival(lo, hi, t, source, bid, ask, valid, epoch, s, f,
                     deadline, cutoff, gap, fee_per_oz, plans, trace, floor_trace, series_trace):
    # s: original 15 fields, death_ns, series_active/id/start_ns/completed/trades.
    # f: cash, liquidation equity, lifetime peak, lifetime max DD,
    #    previous trade net, position size, series base/minimum, max series loss.
    trade = np.full(24, -1., dtype=np.float64)
    if s[0] == 0 and s[5] == 0:
        if hi > lo:
            s[9] = t[hi - 1]
            s[10] = epoch[hi - 1]
        if len(trace):
            trace[lo:hi] = f[1]
            floor_trace[lo:hi] = .5 * f[6] if s[16] or s[12] else np.nan
            series_trace[lo:hi] = s[17] if s[16] or s[12] else 0
        return trade
    for j in range(lo, hi):
        now = t[j]
        supported = valid[j] and bid[j] > 0 and ask[j] >= bid[j] and (gap < 0 or 0 <= now - source[j] <= gap)
        if s[0] != 0 and (not supported or epoch[j] != s[10] or (gap >= 0 and s[9] > 0 and now - s[9] > gap)):
            s[11] = 1
            if gap < 0:
                # Certified segments: stop at the first unsupported observation.
                # Preserve the open position and last supported equity; no invented
                # recovery fill, realized PnL, barrier crossing or terminal reward.
                s[9] = now
                return trade
        s[9] = now
        s[10] = epoch[j]
        if s[5] != 0:
            if now > s[8] or now >= cutoff or now >= deadline or epoch[j] != s[13] or s[12]:
                s[5] = 0
            elif now > s[7] and supported:
                if not s[16]:
                    s[16] = 1
                    s[17] += 1
                    s[18] = now
                    s[20] = 0
                    f[6] = f[0]  # first FILL cash, before this entry's fee
                    f[7] = f[0]
                s[20] += 1
                s[0] = s[5]
                s[4] = s[6]
                s[1] = ask[j] if s[0] > 0 else bid[j]
                s[2] = now
                s[3] = j
                s[5] = 0
                s[11] = 0
                f[0] -= fee_per_oz * f[5]
        if s[0] != 0 and supported:
            px = bid[j] if s[0] > 0 else ask[j]
            gross = s[0] * (px - s[1]) / 1e9 * f[5]
            fee = fee_per_oz * f[5]
            f[1] = f[0] + gross
            f[2] = max(f[2], f[1])
            f[3] = max(f[3], f[2] - f[1])
            floor = .5 * f[6]
            if not s[11]:
                f[7] = min(f[7], f[1])
                f[8] = max(f[8], max(0., (f[6] - f[1]) / f[6]))
                if f[1] < floor - MONEY_TOLERANCE:
                    s[12] = 1
                    if not s[15]:
                        s[15] = now
            reason = 0
            p = plans[s[4]]
            if s[11]:
                reason = 6
            elif s[12]:
                reason = 5
            elif now >= deadline:
                reason = 4
            elif s[0] * (px - s[1]) <= -p[1] * 100_000_000:
                reason = 2
            elif s[0] * (px - s[1]) >= p[0] * 100_000_000:
                reason = 1
            elif now - s[2] >= p[2] * NS:
                reason = 3
            if reason:
                trade[0] = s[14]
                trade[1] = s[3]
                trade[2] = j
                trade[3] = s[0]
                trade[4] = s[4]
                trade[5] = s[1]
                trade[6] = px
                trade[7] = gross - fee
                trade[8] = reason
                trade[9] = s[11]
                trade[10] = s[2]
                trade[11] = now
                trade[12] = gross
                trade[13] = f[5]
                trade[14] = fee
                f[0] += gross
                f[4] = gross - fee
                status = 0
                if reason == 6:
                    status = 3
                elif s[12]:
                    status = 2
                    s[16] = 0
                elif f[0] >= f[6] - MONEY_TOLERANCE:
                    status = 1
                    s[16] = 0
                    s[19] += 1
                trade[15] = s[17]
                trade[16] = f[6]
                trade[17] = floor
                trade[18] = f[0] - f[6]
                trade[19] = status
                trade[20] = s[18]
                trade[21] = s[20]
                trade[22] = f[7]
                trade[23] = max(0., (f[6] - f[0]) / f[6])
                s[0] = 0
                s[11] = 0
                if len(trace):
                    trace[j] = f[1]
                    floor_trace[j] = floor
                    series_trace[j] = s[17]
                if reason == 6:
                    # The recovery quote is a diagnostic close, not proof of
                    # the lost interval's execution or barrier ordering.
                    return trade
        if s[0] == 0:
            f[1] = f[0]
        f[2] = max(f[2], f[1])
        f[3] = max(f[3], f[2] - f[1])
        if len(trace):
            trace[j] = f[1]
            # Preserve the just-closed trade's active barrier at its exit quote;
            # later flat quotes have no barrier until a new actual fill.
            if trade[2] != j:
                floor_trace[j] = .5 * f[6] if s[16] or s[12] else np.nan
                series_trace[j] = s[17] if s[16] or s[12] else 0
    return trade


class SurvivalGym:
    action_names = ('WAIT', 'LONG/fast', 'LONG/medium', 'LONG/wide',
                    'SHORT/fast', 'SHORT/medium', 'SHORT/wide')
    account_dim = 11

    def __init__(self, day, contract=None, record=False, capital_state=None):
        self.day = day
        self.c = contract or Contract(initial_capital=200., daily_drawdown_stop=float('inf'))
        self.record = record
        self.finalized_support = day.meta.get('support_contract') == 'FINALIZED_SEGMENTS_V2'
        self.initial_state = capital_state or CapitalState(
            initial_capital=float(self.c.initial_capital), cash=float(self.c.initial_capital),
            peak=float(self.c.initial_capital))
        if self.initial_state.initial_capital <= 0:
            raise ValueError('Initial capital must be positive')
        if self.initial_state.censored:
            raise ValueError('Cannot continue a support-censored population')
        if self.initial_state.risk_version != RISK_VERSION:
            raise ValueError('Capital state uses a different risk contract')
        if self.initial_state.series_active and self.initial_state.series_start_equity <= 0:
            raise ValueError('An active series needs its positive frozen starting equity')
        self.plans = np.asarray(PLANS, dtype=np.int64)
        self.reset()

    def reset(self):
        state = self.initial_state
        self.start_cash = float(state.cash)
        self.i = 0
        self.s = np.zeros(21, dtype=np.int64)
        self.s[12] = int(state.ruined)
        self.s[15] = int(state.ruin_ns)
        self.s[16:21] = [state.series_active, state.series_id, state.series_start_ns,
                         state.completed_series, state.series_trade_count]
        series_min = state.series_min_equity
        if state.series_active and series_min == 0:
            series_min = min(state.cash, state.series_start_equity)
        historical_risk = state.max_series_loss_fraction
        if state.series_active:
            historical_risk = max(historical_risk, max(0., (state.series_start_equity - series_min) / state.series_start_equity))
        self.f = np.asarray([state.cash, state.cash, state.peak, state.max_drawdown,
                             state.last_trade_net, self.c.size_oz, state.series_start_equity,
                             series_min, historical_risk], dtype=float)
        self.trades = []
        self.equity = [float(state.cash)]
        self.actions = []
        self.cancelled = 0
        self.support_halted = False
        self.censor_ns = 0
        self.trace = np.full(len(self.day.quote_t), np.nan) if self.record else np.empty(0)
        self.floor_trace = np.full(len(self.day.quote_t), np.nan) if self.record else np.empty(0)
        self.trace_series_floor = self.floor_trace
        self.trace_series_id = np.zeros(len(self.day.quote_t), dtype=np.int64) if self.record else np.empty(0, dtype=np.int64)
        self.series_history = {int(row['series_id']): dict(row) for row in state.series_history}
        self.q = int(self.day.cuts[0])
        self.done = False
        if self.q:
            self.s[9] = self.day.quote_t[self.q - 1]
            self.s[10] = self.day.epoch[self.q - 1]
        if self.record:
            self.trace[:self.q] = state.cash
            if state.series_active or state.ruined:
                self.floor_trace[:self.q] = self.barrier
                self.trace_series_id[:self.q] = state.series_id
        return self.observation(), {'action_mask': self.mask()}

    def account(self):
        scale = self.initial_state.initial_capital * .5
        active = bool(self.s[16])
        fraction = max(0., (self.f[6] - self.f[1]) / self.f[6]) if active else 0.
        return np.asarray([self.s[0], (self.f[0] - self.initial_state.initial_capital) / scale,
                           (self.f[1] - self.initial_state.initial_capital) / scale,
                           (self.f[2] - self.f[1]) / scale, self.f[4] / 5., self.s[12],
                           min((self.initial_state.trades + len(self.trades)) / 100., 5.),
                           active, self.f[6] / self.initial_state.initial_capital - 1. if active else 0.,
                           fraction, self.s[20] / 100. if active else 0.], dtype=np.float32)

    @property
    def barrier(self):
        return .5 * self.f[6] if self.s[16] or self.s[12] else float('nan')

    def risk_snapshot(self):
        active = bool(self.s[16])
        status = ('censored' if self.support_halted or (self.done and self.s[0]) else
                  'ruined' if self.s[12] else 'active' if active else 'recovered' if self.s[17] else 'none')
        base = float(self.f[6])
        return dict(risk_version=RISK_VERSION, series_id=int(self.s[17]), series_active=active,
                    series_start_ns=str(int(self.s[18])), series_start_equity=base,
                    series_base=base, series_floor=.5 * base if active or self.s[12] else None,
                    series_status=status, series_cumulative_net=float(self.f[1] - base) if self.s[17] else 0.,
                    series_loss_fraction=max(0., (base - self.f[1]) / base) if base > 0 else 0.,
                    series_min_equity=float(self.f[7]), series_trade_count=int(self.s[20]),
                    completed_series=int(self.s[19]), max_series_loss_fraction=float(self.f[8]),
                    liquidation_equity=float(self.f[1]), money_tolerance_usd=MONEY_TOLERANCE)

    def _update_series_history(self):
        if not self.s[17]:
            return
        snapshot = self.risk_snapshot()
        row = dict(self.series_history.get(int(self.s[17]), {}), **snapshot)
        row['series_floor'] = .5 * self.f[6]
        if self.trades and self.trades[-1]['series_id'] == self.s[17]:
            row['last_trade_exit_ns'] = str(self.trades[-1]['exit_ns'])
        if snapshot['series_status'] in ('recovered', 'ruined', 'censored'):
            if snapshot['series_status'] == 'censored':
                observed_end = self.censor_ns or (int(self.day.quote_t[self.q - 1]) if self.q else 0)
                row['end_ns'] = str(observed_end)
            else:
                row['end_ns'] = str(self.s[15]) if self.s[12] else row.get('last_trade_exit_ns')
        else:
            row['end_ns'] = None
        self.series_history[int(self.s[17])] = row

    def mask(self):
        m = np.zeros(7, dtype=bool)
        m[0] = True
        if self.done or self.support_halted or self.s[0] or self.s[5] or self.s[12] or self.f[0] <= 0 or not self.day.can_enter[self.i]:
            return m
        t = int(self.day.decision_t[self.i])
        j = int(self.day.cuts[self.i]) - 1
        if (j < 0 or not self.day.valid[j] or self.day.bid[j] <= 0 or self.day.ask[j] < self.day.bid[j]
                or (not self.finalized_support and (not 0 <= t - int(self.day.source_t[j]) <= self.c.support_gap_ns
                or not 0 <= t - int(self.day.quote_t[j]) <= self.c.support_gap_ns))):
            return m
        if t >= int(self.day.meta['ny_close_ns']):
            return m
        for p, plan in enumerate(PLANS):
            if t + self.c.latency_ns + self.c.pending_timeout_ns + plan[2] * NS < self.day.deadline:
                m[1 + p] = m[4 + p] = True
        return m

    def observation(self):
        return {'market': self.day.x[self.i], 'account': self.account()}

    def step(self, action, size_oz=None):
        if self.done:
            raise RuntimeError('Episode ended')
        action = int(action)
        if action < 0 or action >= 7 or not self.mask()[action]:
            raise ValueError('Illegal action')
        before = self.f[1]
        self.actions.append(action)
        if action:
            size = self.c.size_oz if size_oz is None else float(size_oz)
            if not np.isfinite(size) or size <= 0:
                raise ValueError('Position size must be finite and positive')
            self.f[5] = size
            self.s[5] = 1 if action <= 3 else -1
            self.s[6] = (action - 1) % 3
            self.s[7] = int(self.day.decision_t[self.i]) + self.c.latency_ns
            self.s[8] = self.s[7] + self.c.pending_timeout_ns
            self.s[13] = self.s[10]
            self.s[14] = self.i
        pending = self.s[5]
        hi = int(self.day.cuts[self.i + 1]) if self.i + 1 < len(self.day.x) else len(self.day.quote_t)
        row = advance_survival(
            self.q, hi, self.day.quote_t, self.day.source_t, self.day.bid, self.day.ask,
            self.day.valid, self.day.epoch, self.s, self.f, self.day.deadline,
            int(self.day.meta['ny_close_ns']), -1 if self.finalized_support else self.c.support_gap_ns,
            self.c.fee_roundtrip, self.plans, self.trace, self.floor_trace, self.trace_series_id)
        if self.finalized_support and self.s[11]:
            self.support_halted = True
            self.censor_ns = int(self.s[9])
            hi = int(np.searchsorted(self.day.quote_t, self.censor_ns, side='left'))
        if row[0] >= 0:
            names = ('decision_index', 'entry_index', 'exit_index', 'side', 'plan', 'entry_nano',
                     'exit_nano', 'net', 'reason_code', 'censored', 'entry_ns', 'exit_ns',
                     'gross', 'size_oz', 'fee', 'series_id', 'series_start_equity', 'series_floor',
                     'series_cumulative_net', 'series_status_code', 'series_start_ns',
                     'series_trade_count', 'series_min_equity', 'series_loss_fraction')
            floating = ('net', 'gross', 'size_oz', 'fee', 'series_start_equity', 'series_floor',
                        'series_cumulative_net', 'series_min_equity', 'series_loss_fraction')
            item = {n: float(v) if n in floating else int(v) for n, v in zip(names, row)}
            item['entry_ns'] = int(self.day.quote_t[item['entry_index']])
            item['exit_ns'] = int(self.day.quote_t[item['exit_index']])
            item['entry_nano'] = int(self.day.ask[item['entry_index']] if item['side'] > 0 else self.day.bid[item['entry_index']])
            item['exit_nano'] = int(self.day.bid[item['exit_index']] if item['side'] > 0 else self.day.ask[item['exit_index']])
            item['reason'] = REASONS[item['reason_code']]
            item['risk_version'] = RISK_VERSION
            item['series_start_ns'] = int(self.s[18])
            item['series_status'] = SERIES_STATUSES[item['series_status_code']]
            item['series_recovered'] = item['series_status'] == 'recovered'
            item['series_active'] = bool(self.s[16])
            item['series_base'] = item['series_start_equity']
            item['series_net'] = item['series_cumulative_net']
            item['equity_after'] = float(self.f[0])
            item['equity_before'] = float(self.f[0] - item['net'])
            self.trades.append(item)
            if item['censored']:
                self.support_halted = True
                self.censor_ns = item['exit_ns']
                hi = item['exit_index'] + 1
        if pending and not self.s[5] and not self.s[0] and row[0] < 0:
            self.cancelled += 1
        self.q = hi
        self.equity.append(float(self.f[1]))
        self.i += 1
        self.done = self.i >= len(self.day.x) or self.support_halted
        self._update_series_history()
        info = {'action_mask': np.asarray([True] + [False] * 6) if self.done else self.mask()}
        if self.done:
            info.update(self.metrics())
        return None if self.done else self.observation(), float(self.f[1] - before), self.done, False, info

    def capital_state(self):
        self._update_series_history()
        streak = self.initial_state.loss_streak
        for item in self.trades:
            streak = streak + 1 if item['net'] < 0 else 0
        return CapitalState(
            initial_capital=self.initial_state.initial_capital, cash=float(self.f[0]),
            peak=float(self.f[2]), max_drawdown=float(self.f[3]), ruined=bool(self.s[12]),
            ruin_ns=int(self.s[15]), censored=not self.metrics()['complete'],
            censor_ns=int(self.censor_ns or (self.day.quote_t[self.q - 1] if self.s[0] and self.q else 0)),
            trades=self.initial_state.trades + len(self.trades),
            wins=self.initial_state.wins + sum(x['net'] > 0 for x in self.trades),
            last_trade_net=float(self.f[4]), loss_streak=streak,
            series_id=int(self.s[17]), series_start_ns=int(self.s[18]), series_start_equity=float(self.f[6]),
            series_active=bool(self.s[16]), completed_series=int(self.s[19]),
            series_min_equity=float(self.f[7]), series_trade_count=int(self.s[20]),
            max_series_loss_fraction=float(self.f[8]),
            series_history=[dict(row) for _, row in sorted(self.series_history.items())])

    def metrics(self):
        censored = sum(x['censored'] for x in self.trades) + int(self.s[0] != 0)
        return dict(net=float(self.f[1] - self.start_cash), total_net=float(self.f[1] - self.initial_state.initial_capital),
                    final_equity=float(self.f[1]), realized=float(self.f[0] - self.start_cash),
                    max_drawdown=float(self.f[3]), trades=len(self.trades), censored=censored,
                    risk_version=RISK_VERSION, max_series_loss_fraction=float(self.f[8]),
                    completed_series=int(self.s[19]), series_active=bool(self.s[16]),
                    complete=not bool(censored), cancelled=self.cancelled, risk_halted=bool(self.s[12]),
                    ruined=bool(self.s[12]), ruin_ns=str(int(self.s[15])),
                    support_halted=self.support_halted, censor_ns=str(self.censor_ns),
                    observed_below_barrier=bool(self.f[6] > 0 and self.f[1] < .5 * self.f[6] - MONEY_TOLERANCE),
                    risk=self.risk_snapshot(),
                    open_terminal=bool(self.s[0]), wins=sum(x['net'] > 0 for x in self.trades),
                    action_counts=np.bincount(self.actions, minlength=7).tolist())


def eligible_intervals(day):
    """Observed legal origin intervals; gaps are not silently counted as life."""
    start = np.asarray(day.decision_t, dtype=np.int64)
    end = np.r_[start[1:], min(day.deadline, int(day.meta['ny_close_ns']))]
    end = np.minimum(end, start + 60 * NS)
    end = np.minimum(end, min(day.deadline, int(day.meta['ny_close_ns'])))
    ok = np.asarray(day.can_enter, dtype=bool) & (end > start)
    return start[ok], end[ok]


def population_metrics(days, state, day_metrics):
    scheduled = 0.
    survived = 0.
    for day in days:
        start, end = eligible_intervals(day)
        scheduled += float(np.maximum(0, end - start).sum()) / NS
        stop_ns = state.ruin_ns if state.ruined else state.censor_ns if state.censored else 0
        clipped = np.minimum(end, stop_ns) if stop_ns else end
        survived += float(np.maximum(0, clipped - start).sum()) / NS
    survival = survived / scheduled if scheduled else 1.
    complete = not state.censored and all(m['complete'] for m in day_metrics)
    # Negative account cash is retained economically. Epsilon is only a finite
    # logarithmic utility floor and is disclosed separately from final cash.
    log_growth = float(np.log(max(state.cash / state.initial_capital, 1e-8)))
    score = (log_growth + .25 * (survival - 1.) - .25 * state.max_series_loss_fraction) if complete else None
    return dict(initial_capital=state.initial_capital, final_equity=state.cash,
                net=state.cash - state.initial_capital, max_drawdown=state.max_drawdown,
                risk_version=RISK_VERSION, max_series_loss_fraction=state.max_series_loss_fraction,
                completed_series=state.completed_series, series_active=state.series_active,
                trades=state.trades, wins=state.wins, ruined=state.ruined,
                ruin_ns=str(state.ruin_ns), complete=complete, censored=state.censored,
                censor_ns=str(state.censor_ns),
                survival_fraction=survival, lifetime_eligible_seconds=survived,
                scheduled_eligible_seconds=scheduled, lifetime_right_censored=not state.ruined,
                lifetime_censor_reason='market_support' if state.censored else 'observation_end' if not state.ruined else None,
                log_growth=log_growth, score=score,
                utility_equity_floor_ratio=1e-8)


def evaluate_policy(days, policy, contract=None, record=False, record_decisions=False):
    """Run one population through ordered days; policy returns action or (action,size).

    Policy may implement reset() and on_trade(trade). The same state carries
    across session closes. Support-censoring stops further population replay.
    """
    c = contract or Contract(initial_capital=200., daily_drawdown_stop=float('inf'))
    state = CapitalState(initial_capital=c.initial_capital, cash=c.initial_capital, peak=c.initial_capital)
    if hasattr(policy, 'reset'):
        policy.reset()
    daily, trades, decisions, envs = [], [], [], []
    for day in days:
        if state.ruined or state.censored:
            break
        env = SurvivalGym(day, c, record=record, capital_state=state)
        seen = 0
        while not env.done:
            i = env.i
            risk_before = env.risk_snapshot()
            choice = policy(env)
            action, size = choice if isinstance(choice, tuple) else (choice, c.size_oz)
            _, reward, _, _, _ = env.step(action, size_oz=size)
            if record or record_decisions:
                decisions.append(dict(date=day.meta.get('date', ''), t_ns=int(day.decision_t[i]),
                                      action=int(action), size_oz=float(size), equity=float(env.f[1]), reward=reward,
                                      **{'risk_' + k: v for k, v in risk_before.items()},
                                      **{'risk_' + k + '_after': v for k, v in env.risk_snapshot().items()}))
            for item in env.trades[seen:]:
                item = dict(item, date=day.meta.get('date', ''))
                trades.append(item)
                if hasattr(policy, 'on_trade'):
                    policy.on_trade(item)
            seen = len(env.trades)
        daily.append(dict(env.metrics(), date=day.meta.get('date', '')))
        state = env.capital_state()
        if record:
            envs.append(env)
    result = population_metrics(days, state, daily)
    result.update(days=daily, trade_rows=trades, decisions=decisions, capital_state=state.dict(),
                  series_rows=state.series_history)
    if record:
        result['environments'] = envs
    return result
