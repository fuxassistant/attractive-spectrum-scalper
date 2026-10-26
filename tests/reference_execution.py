"""Reusable formulas extracted from spectrum/player_v3/gym.py; see docs/MIGRATION.json."""
import numpy as np
from numba import njit
from spectrum.core.schema import NS
MONEY_TOLERANCE = 1e-8

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
