"""Direction-free opportunity: realised movement against a causal baseline.

A one-second as-of midpoint grid is built from certified quotes. Causal fields
use the grid boundary at or before the origin; future fields (Y) are stored
separately and are mature only when every second of the horizon is supported
in the origin's execution segment. The virtual straddle and gamma residuals are
knowledge targets, never rewards and never tradable PnL.

    premium_h = sigma * sqrt(h) * sqrt(2/pi)     (normal increments, ticks)
    straddle_h = |P(t+h) - P(t)| - premium_h      (ticks)
    gamma_h = sum dP^2 - sigma^2 * h              (ticks^2, variance residual)
"""
import numpy as np
from spectrum.player.contract import NS

TICK_NANO = 100_000_000  # GC: $0.10 per oz
HORIZONS = (30, 120, 600, 1800)
SIGMA_WINDOWS = (60, 600, 3600)
BASELINE_WINDOW = 600
MIN_SUPPORT = .8
SQRT_2_PI = np.sqrt(2 / np.pi)
CAUSAL_NAMES = tuple(f'sigma{w}_ticks_per_sqrt_s' for w in SIGMA_WINDOWS) + tuple(
    f'sigma{w}_known' for w in SIGMA_WINDOWS) + ('spread_ticks', 'mid_known')
FUTURE_NAMES = tuple(f'{n}_{h}' for h in HORIZONS for n in (
    'mature', 'move_ticks', 'abs_move_ticks', 'rv_ticks2', 'up_ticks', 'down_ticks',
    'premium_ticks', 'straddle_ticks', 'gamma_ticks2'))


def second_grid(quote_t, bid, ask, valid, epoch, start_ns, end_ns):
    """As-of state at each second boundary start+k (k=0..n-1), half-open session.

    A boundary is supported when the latest publication at or before it is a
    valid book. Quote silence is not support loss. Mid is from the latest valid
    publication, so it is unknown before the first valid book.
    """
    n = int((int(end_ns) - int(start_ns)) // NS)
    boundary = int(start_ns) + np.arange(n, dtype=np.int64) * NS
    last = np.searchsorted(quote_t, boundary, side='right') - 1
    ok = np.asarray(valid, bool) & (bid > 0) & (ask >= bid)
    good = np.flatnonzero(ok)
    last_good = np.searchsorted(quote_t[good], boundary, side='right') - 1
    supported = (last >= 0) & (last_good >= 0)
    supported[supported] &= ok[last[supported]]
    j = good[np.clip(last_good, 0, None)] if len(good) else np.zeros(n, np.int64)
    mid = np.where(supported, (bid[j] + ask[j]) / (2 * TICK_NANO), np.nan) if len(good) else np.full(n, np.nan)
    spread = np.where(supported, (ask[j] - bid[j]) / TICK_NANO, np.nan) if len(good) else np.full(n, np.nan)
    seg = np.where(supported, np.asarray(epoch)[j] if len(good) else -1, -1).astype(np.int64)
    step = np.r_[False, supported[1:] & supported[:-1] & (seg[1:] == seg[:-1])]
    d = np.where(step, np.diff(np.r_[np.nan, mid]), 0.)
    return dict(start_ns=int(start_ns), boundary=boundary, mid=mid, spread=spread, segment=seg,
                step=step, d2_cum=np.r_[0., np.cumsum(d * d)], step_cum=np.r_[0, np.cumsum(step)])


def origin_index(grid, t_ns):
    """Latest boundary at or before t: never a later second."""
    return ((np.asarray(t_ns, np.int64) - grid['start_ns']) // NS).astype(np.int64)


def causal(grid, k):
    """Per-second variance rate over trailing windows ending at boundary k."""
    k = np.asarray(k, np.int64); out = {}
    for w in SIGMA_WINDOWS:
        lo = np.clip(k - w + 1, 0, None); hi = k + 1
        steps = grid['step_cum'][hi] - grid['step_cum'][lo]
        ss = grid['d2_cum'][hi] - grid['d2_cum'][lo]
        known = (k - w + 1 >= 0) & (steps >= MIN_SUPPORT * w)
        out[f'sigma{w}_ticks_per_sqrt_s'] = np.where(known, np.sqrt(ss / np.maximum(steps, 1)), 0.)
        out[f'sigma{w}_known'] = known
    known = ~np.isnan(grid['mid'][k])
    out['spread_ticks'] = np.where(known, grid['spread'][k], 0.)
    out['mid_known'] = known
    return out


def future(grid, k, sigma, sigma_known):
    """Y only. A horizon is mature iff all its one-second steps are supported."""
    k = np.asarray(k, np.int64); n = len(grid['mid']); out = {}
    for h in HORIZONS:
        end = k + h
        inside = end < n
        e = np.clip(end, 0, n - 1)
        steps = np.where(inside, grid['step_cum'][e + 1] - grid['step_cum'][np.clip(k + 1, 0, n)], 0)
        mature = inside & (steps == h) & ~np.isnan(grid['mid'][k])
        move = np.where(mature, grid['mid'][e] - grid['mid'][k], np.nan)
        rv = np.where(mature, grid['d2_cum'][e + 1] - grid['d2_cum'][np.clip(k + 1, 0, n)], np.nan)
        up = np.full(len(k), np.nan); down = np.full(len(k), np.nan)
        for i in np.flatnonzero(mature):
            path = grid['mid'][k[i]:k[i] + h + 1] - grid['mid'][k[i]]
            up[i] = path.max(); down[i] = -path.min()
        premium = np.where(sigma_known, sigma * np.sqrt(h) * SQRT_2_PI, np.nan)
        out.update({f'mature_{h}': mature, f'move_ticks_{h}': move, f'abs_move_ticks_{h}': np.abs(move),
                    f'rv_ticks2_{h}': rv, f'up_ticks_{h}': up, f'down_ticks_{h}': down,
                    f'premium_ticks_{h}': premium, f'straddle_ticks_{h}': np.abs(move) - premium,
                    f'gamma_ticks2_{h}': rv - np.where(sigma_known, sigma * sigma * h, np.nan)})
    return out


def measure(grid, t_ns):
    """Causal X and future Y for origins; the caller keeps them in separate columns."""
    k = origin_index(grid, t_ns)
    if np.any(k < 0) or np.any(k >= len(grid['mid'])):
        raise ValueError('Origin outside the second grid')
    x = causal(grid, k)
    y = future(grid, k, x[f'sigma{BASELINE_WINDOW}_ticks_per_sqrt_s'], x[f'sigma{BASELINE_WINDOW}_known'])
    return x, y
