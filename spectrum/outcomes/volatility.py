"""Reusable formulas extracted from spectrum/cognition/volsense.py; see docs/MIGRATION.json."""
import numpy as np
from . import race as rc
from spectrum.features.families.volatility import _cs

VERSION = 'VOLSENSE_V1'


GRID_G = rc.GRID_G


RHO = 2.


EPS = .05


SIG_W = (10, 30, 60, 300, 1800)


RANGE_W = (60, 300, 1800)


ER_W = (60, 300, 1800)


VOL_W = (10, 60, 300, 1800)


STFT_WIN, STFT_HOP = 1800, 60


STFT_BANDS = ((2, 10), (10, 60), (60, 300), (300, 1800))


EVENTS = (('us_data', 'America/New_York', '08:30'), ('ny_open', 'America/New_York', '09:30'), ('comex_close', 'America/New_York', '13:30'),
          ('lbma_am', 'Europe/London', '10:30'), ('lbma_pm', 'Europe/London', '15:00'))


RECENT_W = 1800


FAMILIES = {
    'vol': ['log_sigma'] + [f'sig{w}_r' for w in SIG_W] + ['sig3600_r', 'jump300'] + [f'range{w}' for w in RANGE_W] + ['sig300_acc'],
    'stft': [f'act_b{j}' for j in range(len(STFT_BANDS))] + ['act_slope'] + [f'vr_b{j}' for j in range(len(STFT_BANDS))],
    'eff': [f'er{w}' for w in ER_W] + ['flicker300', 'chop300', 'ac10'],
    'flow': [f'vol{w}' for w in VOL_W] + ['vol60_r', 'vol300_r', 'book60', 'book60_r', 'ease300', 'visc300', 'imb60', 'imb300',
                                          'absorb300', 'withdraw60', 'bookp60'],
    'book': ['spread', 'spread300', 'wide300', 'dimb1', 'dimb4', 'depth1', 'depth4', 'depth12', 'depth4_chg'],
    'time': ['london_min', 'close_min'] + [f'ev_{e[0]}' for e in EVENTS] + ['weekday'],
    'day': ['today_range', 'night_sigma'],
    'recent': [f'recent{G}' for G in GRID_G] + ['recent_n'],
}


FEATURES = [c for cols in FAMILIES.values() for c in cols]


def labels(b, mid=None):
    """Opportunity strength per side and scale for every second with a known sigma600 (NaN elsewhere)."""
    mid = b['mid'] if mid is None else mid
    n = len(mid); k = np.flatnonzero(np.asarray(b['sigma_known'], bool) & np.isfinite(mid) & (b['sigma'] > 0))
    out = {}
    for G in GRID_G:
        e = rc.excursion(mid, b['seg'], k, b['sigma'][k], G)
        RL, RS, H = (np.full(n, np.nan) for _ in range(3))
        RL[k], RS[k], H[k] = e['R_long'], e['R_short'], e['H']
        out[G] = dict(R_long=RL, R_short=RS, H=H)
    return out


def recent(lab, n):
    """Share of opportunities among decisions whose full horizon ended within the last RECENT_W seconds."""
    out = {}; counts = None
    for G, L in lab.items():
        _, R = rc.strength(L['R_long'], L['R_short'], RHO); ok = np.isfinite(R)
        j = np.flatnonzero(ok); r = j + L['H'][j]; order = np.argsort(r, kind='stable'); r = r[order]
        y = (R[j][order] >= RHO).astype(float); cy = _cs(y)
        t = np.arange(n); hi = np.searchsorted(r, t, side='right'); lo = np.searchsorted(r, t - RECENT_W, side='right')
        cnt = hi - lo; s = cy[hi] - cy[lo]
        out[f'recent{G}'] = np.where(cnt >= 30, s / np.maximum(cnt, 1), np.nan)
        if counts is None:
            counts = cnt
    out['recent_n'] = np.log1p(counts) if counts is not None else np.zeros(n)
    return out
