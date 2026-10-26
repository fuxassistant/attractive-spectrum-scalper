"""Reusable formulas extracted from spectrum/cognition/opportunity.py; see docs/MIGRATION.json."""
import numpy as np
from spectrum.core.schema import NS

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
