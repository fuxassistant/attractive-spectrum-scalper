"""Scalper volatility detector: causal features and opportunity labels for every second (VOLSENSE_V1).

Label (Y): at scale G (ticks) the opportunity is that some side moves 2G before its own -G stop within
H = clip(2 (G / sigma600)^2, 20 s, 15 min) (race.excursion). Features (X) at second i use data up to i only:
  vol      log sigma600; sigma on 10..3600 s relative to sigma600; realised / bipower variation (jumps);
           normalised high-low range; acceleration of sigma300
  stft     trailing 30-minute frames every minute (value valid from the frame end): share of activity |dmid|
           power in period bands, activity spectral slope, variance ratio of dmid per band
  eff      efficiency |move| / path, direction-change share, path / range, lag-1 autocorrelation of 10 s moves
  flow     trade and book event rates and their acceleration, path per unit volume (ease), volume per tick of
           range (viscosity), |aggressor imbalance|, absorption (one-sided aggression without progress),
           liquidity withdrawal (removes / adds), |book pressure|
  book     spread now / mean / wide share, |depth imbalance|, total depth and its change
  time     minutes since London open and to NY close, signed minutes to US data 08:30 ET, NY open 09:30 ET,
           COMEX close 13:30 ET, LBMA auctions 10:30 and 15:00 London (DST-aware), weekday
  day      today's range so far, overnight sigma (previous-day context is joined at training time)
  recent   share of opportunities among decisions whose whole horizon has already elapsed (last 30 min)
Every feature is a deterministic function of the per-second base data up to its second; tests check that
changing the future leaves the past untouched.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from . import race as rc

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


def _cs(x):
    return np.r_[0., np.cumsum(x)]


def _ws(c, w):
    """Trailing sum over (i - w, i] from a cumulative array of length n + 1."""
    n = len(c) - 1; i = np.arange(n)
    return c[i + 1] - c[np.clip(i + 1 - w, 0, None)]


def _shift(x, k):
    out = np.full(len(x), np.nan); out[k:] = x[:-k]; return out


def stft_features(d, n):
    """Trailing-window spectra every STFT_HOP seconds; returns (per-second arrays, frame ends)."""
    out = {c: np.full(n, np.nan) for c in FAMILIES['stft']}
    if n < STFT_WIN:
        return out, np.zeros(0, np.int64)
    win = sliding_window_view(d, STFT_WIN)[::STFT_HOP]            # frame j covers [j*HOP, j*HOP + WIN)
    ends = np.arange(len(win)) * STFT_HOP + STFT_WIN - 1
    hann = np.hanning(STFT_WIN); fr = np.fft.rfftfreq(STFT_WIN, 1.)
    act = np.abs(win); act = act - act.mean(1, keepdims=True)
    pa = np.abs(np.fft.rfft(act * hann, axis=1)) ** 2
    pdir = np.abs(np.fft.rfft(win * hann, axis=1)) ** 2; flat = np.sum((win * hann) ** 2, 1)
    masks = [(fr > 1 / hi) & (fr <= 1 / lo) for lo, hi in STFT_BANDS]
    band_pow = np.stack([pa[:, m].sum(1) for m in masks], 1); tot = band_pow.sum(1, keepdims=True)
    frames = {f'act_b{j}': np.where(tot[:, 0] > 0, band_pow[:, j] / np.maximum(tot[:, 0], 1e-12), np.nan) for j in range(len(masks))}
    sel = (fr >= 1 / 1800) & (fr <= 1 / 10); lf = np.log(fr[sel])
    lp = np.log(pa[:, sel] + 1e-12); lfc = lf - lf.mean()
    frames['act_slope'] = ((lp - lp.mean(1, keepdims=True)) * lfc).sum(1) / (lfc ** 2).sum()
    for j, m in enumerate(masks):
        frames[f'vr_b{j}'] = np.where(flat > 0, np.log(pdir[:, m].mean(1) / np.maximum(flat, 1e-12) + 1e-6), np.nan)
    quiet = (np.abs(win) > 0).sum(1) < 100
    idx = np.searchsorted(ends, np.arange(n), side='right') - 1; has = idx >= 0; ic = np.clip(idx, 0, None)
    for c, v in frames.items():
        v = np.where(quiet, np.nan, v); out[c] = np.where(has, v[ic], np.nan)
    return out, ends


def event_seconds(date, start_ns):
    out = {}
    for name, tz, hm in EVENTS:
        ts = pd.Timestamp(f'{date} {hm}', tz=tz).tz_convert('UTC').value
        out[name] = (ts - int(start_ns)) / 1e9
    return out


def features(b, m, mid=None):
    """All causal features for every second of the day file. `mid` replaces the market mid (null paths)."""
    mid = b['mid'] if mid is None else mid
    n = len(mid); i = np.arange(n); f = {}
    step = np.asarray(b['step'], bool)
    d = np.where(step, np.diff(mid, prepend=np.nan), 0.); d = np.nan_to_num(d); ad = np.abs(d)
    cstep, cd2, cad = _cs(step.astype(float)), _cs(d * d), _cs(ad)
    s600 = np.asarray(b['sigma'], float); s600k = np.asarray(b['sigma_known'], bool)
    sig = {}
    for w in SIG_W:
        cnt = _ws(cstep, w); sig[w] = np.where(cnt >= .8 * w, np.sqrt(_ws(cd2, w) / np.maximum(cnt, 1)), np.nan)
    f['log_sigma'] = np.where(s600k, np.log(s600 + EPS), np.nan)
    for w in SIG_W:
        f[f'sig{w}_r'] = np.log((sig[w] + EPS) / (s600 + EPS))
    f['sig3600_r'] = np.where(b['sigma3600_known'], np.log((b['sigma3600'] + EPS) / (s600 + EPS)), np.nan)
    pair = ad * np.r_[0., ad[:-1]] * (step & np.r_[False, step[:-1]])
    cnt300 = _ws(cstep, 300)
    rv = _ws(cd2, 300) / np.maximum(cnt300, 1); bv = (np.pi / 2) * _ws(_cs(pair), 300) / np.maximum(cnt300, 1)
    f['jump300'] = np.where(cnt300 >= 240, np.log((rv + EPS ** 2) / (bv + EPS ** 2)), np.nan)
    ms = pd.Series(mid); rng = {}
    for w in RANGE_W:
        hi = ms.rolling(w, min_periods=int(.8 * w)).max().to_numpy(); lo = ms.rolling(w, min_periods=int(.8 * w)).min().to_numpy()
        rng[w] = hi - lo; f[f'range{w}'] = rng[w] / (s600 * np.sqrt(w) + EPS)
    f['sig300_acc'] = np.log(sig[300] + EPS) - np.log(_shift(sig[300], 300) + EPS)
    st, _ = stft_features(d, n); f.update(st)
    mf = ms.ffill().to_numpy()
    path = {w: _ws(cad, w) for w in set(ER_W) | {300}}
    for w in ER_W:
        f[f'er{w}'] = np.abs(mf - _shift(mf, w)) / (path[w] + EPS)
    nz = d != 0; sgn = np.sign(d)
    last = pd.Series(np.where(nz, sgn, np.nan)).ffill().shift(1).to_numpy()
    change = nz & np.isfinite(last) & (sgn != last)
    f['flicker300'] = _ws(_cs(change.astype(float)), 300) / np.maximum(_ws(_cs(nz.astype(float)), 300), 1)
    f['chop300'] = path[300] / (rng[300] + .5)
    nb = n // 10; x10 = d[:nb * 10].reshape(nb, 10).sum(1)
    ac = pd.Series(x10).rolling(180, min_periods=120).corr(pd.Series(x10).shift(1)).to_numpy()
    jb = (i + 1) // 10 - 1; okb = (jb >= 0) & (jb < nb)
    f['ac10'] = np.where(okb, ac[np.clip(jb, 0, nb - 1)], np.nan)
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']; cvol = _cs(vol)
    adds, rems = b['add_bid'] + b['add_ask'], b['remove_bid'] + b['remove_ask']; cbook = _cs(adds + rems)
    V = {w: _ws(cvol, w) for w in set(VOL_W) | {3600}}
    for w in VOL_W:
        f[f'vol{w}'] = np.log(V[w] / w + 1e-3)
    base = np.log(V[3600] / 3600 + 1e-3)
    f['vol60_r'], f['vol300_r'] = f['vol60'] - base, f['vol300'] - base
    f['book60'] = np.log(_ws(cbook, 60) / 60 + 1e-3); f['book60_r'] = f['book60'] - np.log(_ws(cbook, 3600) / 3600 + 1e-3)
    f['ease300'] = np.log((path[300] + EPS) / (V[300] + 1))
    f['visc300'] = np.log((V[300] + 1) / (rng[300] + .5))
    cb, cs_ = _cs(b['trade_buy']), _cs(b['trade_sell'])
    for w in (60, 300):
        f[f'imb{w}'] = np.abs(_ws(cb, w) - _ws(cs_, w)) / (V[w] + 1)
    f['absorb300'] = f['imb300'] * np.log1p(V[300]) / (1 + np.abs(mf - _shift(mf, 300)) / (s600 * np.sqrt(300) + EPS))
    f['withdraw60'] = np.log((_ws(_cs(rems), 60) + 1) / (_ws(_cs(adds), 60) + 1))
    press = (b['add_bid'] - b['remove_bid']) - (b['add_ask'] - b['remove_ask'])
    f['bookp60'] = np.abs(_ws(_cs(press), 60)) / (_ws(cbook, 60) + 1)
    sp = pd.Series(b['spread'])
    f['spread'] = b['spread']; f['spread300'] = sp.rolling(300, min_periods=200).mean().to_numpy()
    f['wide300'] = (sp >= 4).astype(float).where(sp.notna()).rolling(300, min_periods=200).mean().to_numpy()
    f['dimb1'], f['dimb4'] = np.abs(b['depth1']), np.abs(b['depth4'])
    for lvl in (1, 4, 12):
        f[f'depth{lvl}'] = np.log1p(b[f'depth{lvl}_total'])
    f['depth4_chg'] = f['depth4'] - _shift(f['depth4'], 60)
    f['london_min'] = (i - m['london']) / 60.; f['close_min'] = (m['ny_close'] - i) / 60.
    for name, es in event_seconds(m['date'], m['start_ns']).items():
        f[f'ev_{name}'] = np.clip((i - es) / 60., -120, 120)
    f['weekday'] = np.full(n, float(pd.Timestamp(m['date']).dayofweek))
    run_hi = np.full(n, np.nan); run_lo = np.full(n, np.nan)
    for a0, a1 in ((0, min(m['london'], n)), (m['london'], n)):
        if a1 > a0:
            run_hi[a0:a1] = np.fmax.accumulate(np.nan_to_num(mid[a0:a1], nan=-np.inf))
            run_lo[a0:a1] = np.fmin.accumulate(np.nan_to_num(mid[a0:a1], nan=np.inf))
    f['today_range'] = np.log1p(np.where(np.isfinite(run_hi - run_lo), run_hi - run_lo, 0.))
    night = np.where(s600k[:m['london']], np.log(s600[:m['london']] + EPS), np.nan) if m['london'] > 0 else np.zeros(0)
    cn = np.cumsum(np.nan_to_num(night)); kn = np.cumsum(np.isfinite(night))
    ns = np.full(n, np.nan)
    if len(night):
        upto = np.minimum(i, m['london'] - 1); ns = np.where(kn[upto] > 0, cn[upto] / np.maximum(kn[upto], 1), np.nan)
    f['night_sigma'] = ns
    return f


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


def day_stats(b, m, mid=None):
    """Session summary used as the next day's context."""
    mid = b['mid'] if mid is None else mid
    a, z = m['london'], m['ny_close']; s = b['sigma'][a:z]; k = np.asarray(b['sigma_known'][a:z], bool)
    mm = mid[a:z]
    return dict(date=m['date'], sess_sigma=float(np.nanmean(np.log(s[k] + EPS))) if k.any() else np.nan,
                sess_range=float(np.log1p(np.nanmax(mm) - np.nanmin(mm))) if np.isfinite(mm).any() else np.nan,
                sess_vol=float(np.log(np.mean((b['trade_buy'] + b['trade_sell'] + b['trade_unknown'])[a:z]) + 1e-3)))
