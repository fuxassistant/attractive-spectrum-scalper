"""Reusable formulas extracted from spectrum/cognition/opportunity.py; see docs/MIGRATION.json."""
import numpy as np
from spectrum.core.schema import NS
from spectrum.features.families.opportunity import second_grid, origin_index, causal

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
