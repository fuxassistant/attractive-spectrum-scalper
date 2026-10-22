"""Independent brute-force recomputation of race / excursion / tug / bands future on random rows; fill proxy; VR."""
import numpy as np, json
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
date = '2025-11-18'
b, m = base_day(date); mid, seg = b['mid'], b['seg']
k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
sig = b['sigma'][k]
rng = np.random.default_rng(5); pick = rng.choice(len(k), 3000, replace=False)
bad = 0
for G in rc.GRID_G:
    r = rc.race(mid, seg, k, sig, G); e = rc.excursion(mid, seg, k, sig, G)
    for i in pick:
        t = k[i]; H = int(np.clip(round(2 * (G / sig[i]) ** 2), 20, 900)); w = 0; RL = RS = None
        path = []
        for j in range(1, H + 1):
            if t + j >= len(mid) or not np.isfinite(mid[t + j]) or seg[t + j] != seg[t]: w = None; break
            path.append(mid[t + j] - mid[t])
        p = np.array(path)
        if w is not None:
            up = np.flatnonzero(p >= G); dn = np.flatnonzero(p <= -G)
            fu = up[0] if len(up) else 10**9; fd = dn[0] if len(dn) else 10**9
            w = 1 if fu < fd else (-1 if fd < fu else 0)
            # excursion
            RL = (p[:fd].max(initial=0.) if fd < 10**9 else p.max(initial=0.)) / G
            RS = (-(p[:fu].min(initial=0.)) if fu < 10**9 else -p.min(initial=0.)) / G
        exp_w = rc.CENSORED if w is None else w
        if r['winner'][i] != exp_w: bad += 1
        if w is not None and (not np.isclose(e['R_long'][i], RL) or not np.isclose(e['R_short'][i], RS)): bad += 1
print('independent race/excursion mismatches:', bad)
# tug: brute force forward definition check on random series
u = rng.choice([-1, 0, 0, 0, .5, 1], 2000).astype(float); tau = 30
s = rc.tug_of_war(u, tau, 5.); lam = 2 ** (-1 / tau); lo = 2 ** (-5 / tau); nxt = 0; ref = np.zeros(len(u))
for t in range(len(u) - 1, -1, -1):
    l = lam if (u[t] == 0 or u[t] * nxt >= 0) else lo; nxt = l * nxt + (1 - lam) * u[t]; ref[t] = nxt
print('tug max diff', np.abs(s - ref).max())
# bands future brute force
vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']; bands = bd.vwap_bands(mid, vol)
W = 600; vw, half = bands[W]; Hh = W // 4
f = bd.future(mid, seg, k, vw[k], half[k], Hh); badb = 0
for i in pick[:500]:
    t = k[i]; p = mid[t + 1:t + Hh + 1]
    if t + Hh >= len(mid) or not np.isfinite(p).all() or (seg[t + 1:t + Hh + 1] != seg[t]).any(): continue
    O = np.mean((p - vw[t]) / half[t]); D = np.mean((p - mid[t]) / half[t])
    ex = np.flatnonzero((p >= vw[t] + half[t]) | (p <= vw[t] - half[t])); E = 0 if not len(ex) else (1 if p[ex[0]] >= vw[t] + half[t] else -1)
    badb += not (np.isclose(O, f['O'][i]) and np.isclose(D, f['D'][i]) and E == f['E'][i])
# brute-force vwap at one point
t = k[1000]; lo_ = t - W + 1; ww = vol[lo_:t + 1] + bd.FLOOR * np.maximum(np.cumsum(vol)[lo_:t + 1] / (np.arange(lo_, t + 1) + 1), 1e-3)
vref = np.sum(ww * mid[lo_:t + 1]) / ww.sum(); sref = np.sqrt(np.sum(ww * (mid[lo_:t + 1] - vref) ** 2) / ww.sum())
print('bands future mismatches', badb, '; vwap diff', vw[t] - vref, 'half diff', half[t] - max(2 * sref, 2))
# passive-fill proxy: long wins where the mid never dipped >= 1.5 ticks before winning
for G in rc.GRID_G:
    r = rc.race(mid, seg, k, sig, G); won = r['winner'] == 1
    print(f'G{G}: long wins {won.mean():.3f}; of them adverse excursion <1.5 ticks (bid at mid-1.5 likely never filled) {np.mean(r["mae"][won] < 1.5):.3f}, <1 tick {np.mean(r["mae"][won] < 1):.3f}')
