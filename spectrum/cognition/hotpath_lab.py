"""Prequential lab for the hot-path instrument (HOTPATH_LAB_V1).

Instrument = ridge regression of z_h on a feature family, refitted every 15 minutes from London open
to NY close on rows whose outcome is already known (t' + h <= T). Knobs ("control panel"):
  family    path | flow | levels | path+flow | all | drift (window mean of z only)
  window    hot (previous London open -> T) | today (today's file start -> T) | london (today's London -> T)
            | global (all earlier days + today so far, uniform weights)
  half-life none | 2 h | 6 h (exponential forgetting, local windows only)
  alpha     ridge strength on the standardised (correlation) scale
Cross-products are accumulated per 15-minute bucket of label-known time, so every knob is a slice.
Decisions in [T_j, T_j+1) use the fit at T_j. Per (config, scale, day, 2-hour block) the lab keeps
sufficient statistics for IC and out-of-sample R^2 against the zero (martingale) forecast, and a
causal "conviction" trade test: sign(prediction) when |prediction| exceeds the previous day's
90th percentile for the same config, measured in ticks of the realised mid move.
"""
from pathlib import Path
import itertools
import json
import time
import numpy as np
import pandas as pd
from spectrum.player.contract import NS
from .hotpath import OUTPUT as DATA, SCALES, PATH, FLOW, LEVELS

FEATS = PATH + FLOW + LEVELS
FAMILY_IDX = dict(path=PATH, flow=FLOW, levels=LEVELS, path_flow=PATH + FLOW, all=FEATS)
WINDOWS = ('hot', 'today', 'london', 'global')
HALF_LIVES = (None, 2., 6.)
ALPHAS = (.03, .3, 3.)
BUCKET = 900


def configs():
    out = []
    for fam in list(FAMILY_IDX) + ['drift']:
        for win in WINDOWS:
            for hl in (HALF_LIVES if win != 'global' else (None,)):
                for a in (ALPHAS if fam != 'drift' else (None,)):
                    out.append((fam, win, hl, a))
    return out


def load(dates, data=DATA):
    return [pd.read_parquet(Path(data) / f'{d}.parquet') for d in dates]


def cross(Z, w, b, nb):
    """Per-bucket weighted cross-products of Z (rows x p) into (nb, p, p)."""
    out = np.zeros((nb, Z.shape[1], Z.shape[1]))
    for j in np.unique(b):
        m = b == j; Zm = Z[m]
        out[j] = (Zm * w[m, None]).T @ Zm
    return out


def solve(S, idx, alpha):
    """Ridge on the standardised scale from accumulated [1, x, y] cross-products. Returns (mu, sd, beta, mu_y)."""
    p = S.shape[0]; s0 = S[0, 0]
    if s0 <= 0:
        return None
    sel = np.r_[0, np.asarray(idx) + 1, p - 1]
    M = S[np.ix_(sel, sel)] / s0
    mu = M[0, 1:-1]; my = M[0, -1]
    C = M[1:-1, 1:-1] - np.outer(mu, mu); cy = M[1:-1, -1] - mu * my
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12)); R = C / np.outer(sd, sd); r = cy / sd
    beta = np.linalg.solve(R + alpha * np.eye(len(r)), r)
    return mu, sd, beta, my


def run_day(prev, today, global_S, h, cfgs, q90_prev):
    """Prequential predictions for one day and one scale; returns aggregates, today's full cross-products, q90."""
    L = int(today.k[today.minutes_london >= 0].min()) if (today.minutes_london >= 0).any() else None
    t_lon = int(today.t_ns[today.minutes_london >= 0].min())
    close = int(today.t_ns[today.session].max()) + NS
    pv = prev[prev.minutes_london >= 0]
    rows = pd.concat([pv.assign(_today=False), today.assign(_today=True)], ignore_index=True)
    y = rows[f'z{h}'].to_numpy(float); known = np.isfinite(y)
    t = rows.t_ns.to_numpy(np.int64); tau = t + h * NS
    X = rows[FEATS].to_numpy(float)
    Z = np.c_[np.ones(len(rows)), X, np.nan_to_num(y)]
    b = np.floor((tau - t_lon) / (BUCKET * NS)).astype(np.int64)
    bmin = b.min(); b = b - bmin; nb = int(b.max()) + 1
    j_refit = np.arange(0, int(np.ceil((close - t_lon) / (BUCKET * NS))))
    T = t_lon + j_refit * BUCKET * NS; jb = j_refit - bmin           # buckets < jb are usable at T
    wins = dict(hot=np.ones(len(rows), bool), today=rows._today.to_numpy(), london=rows._today.to_numpy() & (t >= t_lon))
    cum = {}
    for win, mwin in wins.items():
        for hl in HALF_LIVES:
            w = np.where(known & mwin, 1., 0.)
            if hl is not None:
                w = w * np.exp2((t - close) / (hl * 3600 * NS))
            C = np.cumsum(cross(Z, w, b, nb), 0)
            cum[(win, hl)] = np.concatenate([np.zeros((1,) + C.shape[1:]), C])
    tmask = rows._today.to_numpy() & known
    today_full = cross(Z, np.where(tmask, 1., 0.), b, nb)
    cum[('global', None)] = np.concatenate([np.zeros((1,) + today_full.shape[1:]), np.cumsum(today_full, 0)]) + global_S[None]
    dec = np.flatnonzero(rows._today.to_numpy() & rows.session.to_numpy() & known)
    jd = np.clip(np.searchsorted(T, t[dec], side='right') - 1, 0, len(T) - 1)
    hour = ((t[dec] - t_lon) // (7200 * NS)).astype(int)
    yd = y[dec]; move = rows[f'move{h}'].to_numpy(float)[dec]; Xd = X[dec]
    agg = []; q90_new = {}
    for cfg in cfgs:
        fam, win, hl, a = cfg; key = (win, hl)
        pred = np.zeros(len(dec)); fitted = np.zeros(len(dec), bool)
        idx = None if fam == 'drift' else [FEATS.index(c) for c in FAMILY_IDX[fam]]
        for j in np.unique(jd):
            pos = int(np.clip(jb[j], 0, nb)); S = cum[key][pos]
            m = jd == j
            if cum[(win, None)][pos][0, 0] < 200:      # too few resolved rows in the window
                continue
            if fam == 'drift':
                if S[0, 0] > 0:
                    pred[m] = S[0, -1] / S[0, 0]; fitted[m] = True
                continue
            fit = solve(S, idx, a)
            if fit is None:
                continue
            mu, sd, beta, my = fit
            pred[m] = ((Xd[m][:, idx] - mu) / sd) @ beta; fitted[m] = True
        q = q90_prev.get(cfg); q90_new[cfg] = float(np.quantile(np.abs(pred[fitted]), .9)) if fitted.any() else None
        conv = fitted & (np.abs(pred) > q) if q is not None else np.zeros(len(dec), bool)
        for hb in np.unique(hour):
            m = fitted & (hour == hb); c = conv & (hour == hb)
            agg.append(dict(fam=fam, win=win, hl=hl if hl is not None else 0., alpha=a if a is not None else 0., h=h, hour=int(hb),
                            n=int(m.sum()), syy=float((yd[m] ** 2).sum()), spp=float((pred[m] ** 2).sum()), spy=float((pred[m] * yd[m]).sum()),
                            sp=float(pred[m].sum()), sy=float(yd[m].sum()), conv_n=int(c.sum()),
                            conv_ticks=float((np.sign(pred[c]) * move[c]).sum()), conv_z=float((np.sign(pred[c]) * yd[c]).sum())))
    full = np.zeros_like(global_S)
    for j in range(nb):
        full += today_full[j]
    return agg, full, q90_new


def run(dates, scales=SCALES, cfgs=None, log=print):
    cfgs = cfgs or configs(); frames = load(dates); out = []
    for h in scales:
        G = np.zeros((len(FEATS) + 2, len(FEATS) + 2)); q90 = {}; t0 = time.perf_counter()
        for i in range(1, len(frames)):
            agg, full, q90 = run_day(frames[i - 1], frames[i], G, h, cfgs, q90)
            for r in agg:
                r['date'] = dates[i]
            out += agg; G = G + full
        log(json.dumps(dict(h=h, days=len(frames) - 1, seconds=round(time.perf_counter() - t0, 1))))
    return pd.DataFrame(out)


def score(agg, by=('fam', 'win', 'hl', 'alpha', 'h')):
    """Per-day IC and OOS R^2, then day-clustered means and t; conviction trades in ticks."""
    by = list(by)
    d = agg.groupby(by + ['date'], dropna=False)[['n', 'syy', 'spp', 'spy', 'sp', 'sy', 'conv_n', 'conv_ticks', 'conv_z']].sum().reset_index()
    cov = d.spy / d.n - d.sp / d.n * d.sy / d.n
    vp = d.spp / d.n - (d.sp / d.n) ** 2; vy = d.syy / d.n - (d.sy / d.n) ** 2
    d['ic'] = cov / np.sqrt(np.maximum(vp * vy, 1e-18))
    d['r2'] = 1 - (d.syy - 2 * d.spy + d.spp) / d.syy
    rows = []
    for key, g in d.groupby(by, dropna=False):
        ic = g.ic.dropna(); r2 = g.r2
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(days=len(g), ic=float(ic.mean()), ic_t=float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 else np.nan,
                 ic_pos=float((ic > 0).mean()), r2=float(r2.mean()), r2_pooled=float(1 - (g.syy - 2 * g.spy + g.spp).sum() / g.syy.sum()),
                 conv_n=int(g.conv_n.sum()), conv_ticks=float(g.conv_ticks.sum() / max(g.conv_n.sum(), 1)),
                 conv_z=float(g.conv_z.sum() / max(g.conv_n.sum(), 1)))
        rows.append(r)
    return pd.DataFrame(rows)
