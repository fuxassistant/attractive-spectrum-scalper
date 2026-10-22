"""VWAP-band representation of the future against the race / opportunity representation, market against null.

Scales: windows 240 / 600 / 1200 / 2800 s, horizons H = W / 4 = 60 / 150 / 300 / 700 s (about the median race
horizons of G = 12 / 18 / 27 / 40). For each scale: O (sleeve occupation), D (future displacement x time),
V (volatility in band units), E (band exit and its side) from bands frozen at t (spectrum.cognition.bands).
Every A day is computed on the market path and on its sign-flipped null (same volume, same |increments|).
1. base rates and smoothness on the test week;
2. predictability out of sample (train: other A days, every 10th second; test: the test week), market and null,
   with the causal volatility-sense features plus band features (position z, band width / sigma sqrt(H),
   VWAP slope over 60 s): sign O, sign D, exit, exit side given exit, V;
3. link to the opportunity label (some side moves 2G before its -G stop) at the matching G.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
sys.path.insert(0, str(Path(__file__).parent))
import opportunity_sense as osn                       # noqa: E402
from spectrum.cognition import race as rc             # noqa: E402
from spectrum.cognition import bands as bd            # noqa: E402
from spectrum.cognition.hotpath import base_day       # noqa: E402

OUT = Path('reports/race_v1')
H = {W: W // 4 for W in bd.WINDOWS}
MATCH = dict(zip(bd.WINDOWS, rc.GRID_G))
SEED = 7
PAGE_STEP = 10


def one_path(b, m, k, mid):
    bp = dict(b, mid=mid); f = osn.features(bp, m)
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
    bands = bd.vwap_bands(mid, vol)
    timew = bd.vwap_bands(mid, np.ones_like(vol))      # equal weights per second: no volume-price coupling
    out = dict(Xv=np.stack([np.asarray(f[c], float)[k] for c in osn.FEATS], 1))
    xb = []
    sig = np.maximum(b['sigma'][k], 1e-6)
    for W in bd.WINDOWS:
        vw, half = bands[W]
        z = (mid[k] - vw[k]) / half[k]
        width = np.log(half[k] / (sig * np.sqrt(H[W])))
        slope = (vw[k] - vw[np.clip(k - 60, 0, None)]) / half[k]
        xb += [z, width, slope]
        fut = bd.future(mid, b['seg'], k, vw[k], half[k], H[W])
        for q in 'ODVE':
            out[f'{q}{W}'] = fut[q].astype(np.float32)
        tvw, thalf = timew[W]
        tf = bd.future(mid, b['seg'], k, tvw[k], thalf[k], H[W])
        out[f'tz{W}'] = ((mid[k] - tvw[k]) / thalf[k]).astype(np.float32); out[f'tE{W}'] = tf['E'].astype(np.float32)
        out[f'vw{W}'] = vw[k].astype(np.float32); out[f'half{W}'] = half[k].astype(np.float32)
        G = MATCH[W]; e = rc.excursion(mid, b['seg'], k, b['sigma'][k], G)
        out[f'RL{W}'] = e['R_long'].astype(np.float32); out[f'RS{W}'] = e['R_short'].astype(np.float32)
    out['Xb'] = np.stack(xb, 1)
    out['X'] = np.nan_to_num(np.c_[out['Xv'], out['Xb']]).astype(np.float32)
    out['mid'] = mid[k].astype(np.float32)
    return out


def day(date):
    b, m = base_day(date)
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
    null = rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(51 * 100000 + int(date.replace('-', '')) % 100000))
    return dict(date=date, t=(k - k[0]), real=one_path(b, m, k, b['mid']), null=one_path(b, m, k, null))


def hysteresis_flips(s, band=.05):
    state = 0; flips = 0
    for v in s:
        if not np.isfinite(v):
            continue
        if v > band and state != 1:
            flips += state == -1; state = 1
        elif v < -band and state != -1:
            flips += state == 1; state = -1
    return flips


def acf(x, lag):
    a, c = x[lag:], x[:-lag]; m = np.isfinite(a) & np.isfinite(c)
    return float(np.corrcoef(a[m], c[m])[0, 1]) if m.sum() > 10 else np.nan


def clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=500, l2_regularization=1., random_state=SEED)


def reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=500, l2_regularization=1., random_state=SEED)


def predict_block(train, test, kind, W):
    """AUC / R^2 of each question on the test week, model trained on the other days of the same kind."""
    zi = len(osn.FEATS) + 3 * bd.WINDOWS.index(W)
    def cat(ds, stride, key):
        return np.concatenate([d[kind][key][::stride] for d in ds])
    Xtr, Xte = cat(train, 10, 'X'), cat(test, 1, 'X')
    res = {}
    for name, fn in (('sign_O', lambda d: d[f'O{W}']), ('sign_D', lambda d: d[f'D{W}'])):
        ytr, yte = cat(train, 10, f'{name[-1]}{W}'), cat(test, 1, f'{name[-1]}{W}')
        a, b_ = np.isfinite(ytr) & (ytr != 0), np.isfinite(yte) & (yte != 0)
        p = clf().fit(Xtr[a], ytr[a] > 0).predict_proba(Xte[b_])[:, 1]
        res[name] = float(roc_auc_score(yte[b_] > 0, p))
        if name == 'sign_O':
            pz = clf().fit(Xtr[a][:, [zi]], ytr[a] > 0).predict_proba(Xte[b_][:, [zi]])[:, 1]
            res['sign_O_z_only'] = float(roc_auc_score(yte[b_] > 0, pz))
    ytr, yte = cat(train, 10, f'E{W}'), cat(test, 1, f'E{W}')
    a, b_ = np.isfinite(ytr), np.isfinite(yte)
    p = clf().fit(Xtr[a], ytr[a] != 0).predict_proba(Xte[b_])[:, 1]; res['exit'] = float(roc_auc_score(yte[b_] != 0, p))
    a2, b2 = a & (ytr != 0), b_ & (yte != 0)
    p = clf().fit(Xtr[a2], ytr[a2] > 0).predict_proba(Xte[b2])[:, 1]; res['exit_side'] = float(roc_auc_score(yte[b2] > 0, p))
    ytr, yte = cat(train, 10, f'V{W}'), cat(test, 1, f'V{W}')
    a, b_ = np.isfinite(ytr), np.isfinite(yte)
    res['V_r2'] = float(r2_score(yte[b_], reg().fit(Xtr[a], ytr[a]).predict(Xte[b_])))
    R = np.fmax(cat(test, 1, f'RL{W}'), cat(test, 1, f'RS{W}')); ok = np.isfinite(R)
    Rtr = np.fmax(cat(train, 10, f'RL{W}'), cat(train, 10, f'RS{W}')); oktr = np.isfinite(Rtr)
    p = clf().fit(Xtr[oktr], Rtr[oktr] >= 2).predict_proba(Xte[ok])[:, 1]; res['opportunity'] = float(roc_auc_score(R[ok] >= 2, p))
    return res


def main():
    dates = osn.dates_of(('2025-10', '2025-11', '2025-12'))
    with ProcessPoolExecutor(max_workers=6) as pool:
        days = list(pool.map(day, dates))
    test = [d for d in days if d['date'] in osn.TEST]; train = [d for d in days if d['date'] not in osn.TEST]
    res = dict(windows=list(bd.WINDOWS), horizons=H, k=bd.K, match=MATCH, descr={}, smooth={}, predict={}, link={})
    for W in bd.WINDOWS:
        for kind in ('real', 'null'):
            O, D, V, E = (np.concatenate([d[kind][f'{q}{W}'] for d in test]) for q in 'ODVE')
            z = O - D
            ins = np.isfinite(E)
            res['descr'][f'{W}_{kind}'] = dict(exit=float(np.mean(E[ins] != 0)), exit_up=float(np.mean(E[ins] == 1)), exit_dn=float(np.mean(E[ins] == -1)),
                                               inside_share=float(ins.mean()),
                                               abs_O=float(np.nanmean(np.abs(O))), abs_D=float(np.nanmean(np.abs(D))), abs_z=float(np.nanmean(np.abs(z))),
                                               V=float(np.nanmean(V)), half_ticks=float(np.nanmedian(np.concatenate([d[kind][f'half{W}'] for d in test]))),
                                               share_O_from_z=float(np.nanvar(z) / np.nanvar(O)))
            tz = np.concatenate([d[kind][f'tz{W}'] for d in test]); tE = np.concatenate([d[kind][f'tE{W}'] for d in test]); tins = np.isfinite(tE)
            res['descr'][f'{W}_{kind}'].update(abs_z_time=float(np.nanmean(np.abs(tz))), exit_time=float(np.mean(tE[tins] != 0)))
            rows = []
            for d in test:
                hours = len(d['t']) / 3600; o, dd, e = d[kind][f'O{W}'], d[kind][f'D{W}'], d[kind][f'E{W}']
                rows.append(dict(O_flips=hysteresis_flips(o) / hours, D_flips=hysteresis_flips(dd) / hours, E_flips=float(np.nansum(np.abs(np.diff(e)) > 0) / hours),
                                 O_acf30=acf(o, 30), D_acf30=acf(dd, 30)))
            res['smooth'][f'{W}_{kind}'] = pd.DataFrame(rows).mean().to_dict()
            res['predict'][f'{W}_{kind}'] = predict_block(train, test, kind, W)
            print(W, kind, json.dumps(res['predict'][f'{W}_{kind}']), flush=True)
        # link to the opportunity label on the market path
        E = np.concatenate([d['real'][f'E{W}'] for d in test]); D = np.concatenate([d['real'][f'D{W}'] for d in test])
        RL = np.concatenate([d['real'][f'RL{W}'] for d in test]); RS = np.concatenate([d['real'][f'RS{W}'] for d in test])
        side, R = rc.strength(RL, RS); ok = np.isfinite(R) & np.isfinite(E); opp = R >= 2
        res['link'][str(W)] = dict(opp_if_exit=float(opp[ok & (E != 0)].mean()), opp_if_inside=float(opp[ok & (E == 0)].mean()),
                                   exit_side_agrees=float((side[ok & opp & (E != 0)] == E[ok & opp & (E != 0)]).mean()),
                                   D_sign_agrees=float((np.sign(D[ok & opp]) == side[ok & opp]).mean()))
    (OUT / 'bands_compare.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf8')
    page = dict(windows=list(bd.WINDOWS), horizons=H, step=PAGE_STEP, summary=res, days=[])
    for d in test:
        sub = slice(None, None, PAGE_STEP); dd = dict(date=d['date'], t=d['t'][sub].tolist(), mid=np.round(d['real']['mid'][sub] / 10, 2).tolist(), S={})
        for W in bd.WINDOWS:
            r, n = d['real'], d['null']
            def ints(x, s):
                return [None if not np.isfinite(v) else int(round(v * s)) for v in x[sub]]
            dd['S'][str(W)] = dict(vw=ints(r[f'vw{W}'] / 10, 100), half=ints(r[f'half{W}'] / 10, 100), O=ints(r[f'O{W}'], 100), D=ints(r[f'D{W}'], 100),
                                   V=ints(r[f'V{W}'], 100), E=ints(r[f'E{W}'], 1), nO=ints(n[f'O{W}'], 100), nE=ints(n[f'E{W}'], 1))
        page['days'].append(dd)
    (OUT / 'bands_data.json').write_text(json.dumps(page, separators=(',', ':'), default=float), encoding='utf8')


if __name__ == '__main__':
    main()
