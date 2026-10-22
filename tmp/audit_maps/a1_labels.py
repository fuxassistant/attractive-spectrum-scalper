"""Audit: censoring with fmax, side bug in impulse, H clipping, bands immediate exits."""
import sys, numpy as np, time
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
from sklearn.metrics import roc_auc_score
date = sys.argv[1] if len(sys.argv) > 1 else '2025-11-18'
t0 = time.time(); b, m = base_day(date); print('load', round(time.time()-t0, 1), 's', m)
k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
sig = b['sigma'][k]; print('n', len(k), 'sigma600 median', np.median(sig).round(3), 'spread median', np.nanmedian(b['spread'][k]))
seg = b['seg']; print('segments in session', np.unique(seg[k]), 'seg changes in file', int(np.sum(seg[1:] != seg[:-1])))
print('last finite mid idx', np.flatnonzero(np.isfinite(b['mid']))[-1], 'ny_close', m['ny_close'], 'n', m['n'])
for G in rc.GRID_G:
    H = rc.horizon(sig, G)
    e = rc.excursion(b['mid'], seg, k, sig, G); L, S = e['R_long'], e['R_short']
    one = np.isfinite(L) ^ np.isfinite(S); both = ~np.isfinite(L) & ~np.isfinite(S)
    R = np.fmax(L, S); lab0 = one & (R < 2)
    u = rc.impulse(L, S); side_true = np.where(np.isfinite(L) & ~np.isfinite(S), 1, np.where(np.isfinite(S) & ~np.isfinite(L), -1, 0))
    wrong = (side_true == 1) & (u < 0)
    print(f'G{G}: H clip max {np.mean(H==rc.H_MAX):.3f} min {np.mean(H==rc.H_MIN):.3f} medH {np.median(H)}; one-side-NaN {one.mean():.4f} both {both.mean():.4f}; '
          f'labelled 0 while a side censored {lab0.sum()} ; impulse sign wrong (long finite, short NaN, u<0) {wrong.sum()} of {((side_true==1)&(np.abs(np.nan_to_num(u))>0)).sum() + wrong.sum()}')
    # where in time are the one-sided rows
    if one.any():
        tt = (k[one] - m['london']) / 3600 + 8
        print('    one-side rows UTC hours range', tt.min().round(3), tt.max().round(3), 'count', one.sum())
# H clip mechanical effect on null: label rate vs sigma for G=40 on the null
null = rc.sign_flipped(b['mid'], seg, np.random.default_rng(1))
for G in (27, 40):
    e = rc.excursion(null, seg, k, sig, G); R = np.fmax(e['R_long'], e['R_short']); H = rc.horizon(sig, G); ok = np.isfinite(R)
    y = (R >= 2)[ok]
    print(f'null G{G}: rate clipped {y[(H==900)[ok]].mean():.3f} unclipped {y[(H<900)[ok]].mean():.3f}; AUC of log_sigma on null label {roc_auc_score(y, np.log(sig[ok])):.3f}')
    e = rc.excursion(b['mid'], seg, k, sig, G); R = np.fmax(e['R_long'], e['R_short']); ok = np.isfinite(R); y = (R >= 2)[ok]
    print(f'real G{G}: AUC of log_sigma {roc_auc_score(y, np.log(sig[ok])):.3f}')
# bands: immediate exits
vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
bands = bd.vwap_bands(b['mid'], vol)
for W in bd.WINDOWS:
    vw, half = bands[W]; z = (b['mid'][k] - vw[k]) / half[k]
    f = bd.future(b['mid'], seg, k, vw[k], half[k], W // 4); E = f['E']; ok = np.isfinite(E) & np.isfinite(z)
    out = np.abs(z) >= 1
    ex = ok & (E != 0)
    print(f'W{W}: |z|>=1 at start {np.mean(out[ok]):.3f}; exits {np.mean(E[ok]!=0):.3f}; exits with start already outside {np.mean(out[ex]):.3f}; '
          f'exit side = sign(z) among those {np.mean(np.sign(z[ex & out]) == E[ex & out]):.3f}; AUC exit_side by z alone {roc_auc_score(E[ex] > 0, z[ex]):.3f}; '
          f'AUC excluding already-outside {roc_auc_score(E[ex & ~out] > 0, z[ex & ~out]):.3f}; median half {np.nanmedian(half[k]):.1f} ticks')
