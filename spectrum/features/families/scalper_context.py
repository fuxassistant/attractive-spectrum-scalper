"""Causal price/flow context formerly consumed by the continuous scalper."""
import numpy as np
import pandas as pd
from spectrum.core.schema import NS
from . import opportunity as op
from .signature import flow_family
RET_WINDOWS = (5, 15, 60, 300)
FLOW_WINDOWS = (5, 30, 60, 300)

def features(day, q, grid, ks, t_dec, seconds):
    x = op.causal(grid, ks)
    sig = x['sigma600_ticks_per_sqrt_s']; known = x['sigma600_known'] & (sig > 0)
    out = dict(spread_ticks=x['spread_ticks'], sigma600=sig, sigma_known=known.astype(np.int8),
               sigma60_ratio=np.where(known & x['sigma60_known'], x['sigma60_ticks_per_sqrt_s'] / np.maximum(sig, 1e-9), 0.),
               sigma3600_ratio=np.where(known & x['sigma3600_known'], x['sigma3600_ticks_per_sqrt_s'] / np.maximum(sig, 1e-9), 0.),
               minutes_since_london=(t_dec - int(day.meta['london_ns'])) / (60 * NS))
    mid, seg = grid['mid'], grid['segment']
    for w in RET_WINDOWS:
        lo = np.clip(ks - w, 0, None)
        ok = known & (ks - w >= 0) & ~np.isnan(mid[ks]) & ~np.isnan(mid[lo]) & (seg[lo] == seg[ks])
        out[f'ret{w}'] = np.where(ok, (mid[ks] - mid[lo]) / np.maximum(sig * np.sqrt(w), 1e-9), 0.)
        out[f'ret{w}_known'] = ok.astype(np.int8)
    series = pd.Series(mid)
    for w in (60, 300):
        hi_, lo_ = series.rolling(w + 1, min_periods=w + 1).max().to_numpy(), series.rolling(w + 1, min_periods=w + 1).min().to_numpy()
        rng = hi_[ks] - lo_[ks]; ok = np.isfinite(rng) & (rng > 0)
        out[f'range{w}_pos'] = np.where(ok, (mid[ks] - lo_[ks]) / np.where(ok, rng, 1), .5)
        out[f'range{w}_known'] = ok.astype(np.int8)
    f = flow_family(seconds, t_dec, windows=FLOW_WINDOWS, rate=(30, 300))
    for k, v in f.items():
        out[k] = v.astype(np.int8) if v.dtype == bool else v
    return out


DIRECTIONAL = ['ret5', 'ret15', 'ret60', 'ret300'] + [f'flow_book{w}_pressure' for w in (5, 30, 60, 300)] + \
              [f'flow_depth{d}_imbalance' for d in (1, 4, 12)]

SYMMETRIC = ['spread_ticks', 'sigma600', 'sigma60_ratio', 'sigma3600_ratio', 'minutes_since_london', 'flow_volume_rate_ratio'] + \
            [f'flow_log_volume{w}' for w in (5, 30, 60, 300)] + [f'flow_unknown{w}_fraction' for w in (5, 30, 60, 300)] + \
            ['ret5_known', 'ret60_known', 'ret300_known', 'range60_known', 'range300_known']

def side_features(f, side):
    s = float(side); x = {c: f[c].to_numpy(np.float32) for c in SYMMETRIC}
    for c in DIRECTIONAL:
        x[c] = s * f[c].to_numpy(np.float32)
    for w in (5, 30, 60, 300):
        lo, hi = f[f'flow_pressure{w}_low'].to_numpy(np.float32), f[f'flow_pressure{w}_high'].to_numpy(np.float32)
        x[f'pressure{w}_low'], x[f'pressure{w}_high'] = (lo, hi) if side > 0 else (-hi, -lo)
    for w in (60, 300):
        pos = f[f'range{w}_pos'].to_numpy(np.float32); x[f'range{w}'] = pos if side > 0 else 1 - pos
    return pd.DataFrame(x)
