"""Diagnostics of the volatility detector: how hard the task is, what the gradient carries, where the model is stable.

Same data and split as volsense_train (train + valid = 2025-10 .. 2026-06, test = 2026-07 .. 2026-09-01, amendment 1).
1. Hindsight quantities per row and scale G (Y side, never features): realised sigma over the label horizon
   (t, t + H] relative to sigma600 at t, efficiency of the future move |m(t+H) - m(t)| / path, horizon clipped.
   Oracles trained on train + valid with these hindsight inputs give the ceiling of a "volatility sense":
     oracle_vol    log(sigma_future / sigma600), clipped flag, log sigma600
     oracle_trend  oracle_vol + future efficiency
2. Forecastability of volatility itself: R^2 of log(sigma_future / sigma600) on the test months from the causal
   features (full) and from the sigma baseline.
3. Brier decomposition (reliability, resolution, uncertainty) of the detector, the baseline and the oracles; the
   log-loss gradient p - y: its spread, its predictable share (resolution / uncertainty), its mean by segment.
4. Stability by market fragment: AUC and calibration gap of the detector by hour since London, sigma600 decile,
   event window (+-15 min around US data and NY open), spread, weekday; AUC per day.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, log_loss, r2_score
sys.path.insert(0, str(Path(__file__).parent))
import volsense_train as vt                           # noqa: E402
from spectrum.cognition import volsense as vs         # noqa: E402
from spectrum.cognition import race as rc             # noqa: E402
from spectrum.cognition.hotpath import base_day       # noqa: E402

OUT = Path('reports/volsense_v1'); DATA = Path('data/derived/volsense_v1')
SEED = 7


def hindsight_day(date, kind='real'):
    """Future realised sigma and efficiency over each row's own horizon, for every scale (market or null path)."""
    f = pd.read_parquet(DATA / kind / f'{date}.parquet', columns=['t_ns', 'sec_london', 'sigma600'] + [f'H{G}' for G in vs.GRID_G])
    b, m = base_day(date)
    mid = b['mid'] if kind == 'real' else rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(61 * 100000 + int(date.replace('-', '')) % 100000))
    k = (f.sec_london.to_numpy() + m['london']).astype(np.int64); n = len(mid)
    step = np.asarray(b['step'], bool); d = np.where(step, np.diff(mid, prepend=np.nan), 0.); d = np.nan_to_num(d)
    c2, c1, cs = np.r_[0., np.cumsum(d * d)], np.r_[0., np.cumsum(np.abs(d))], np.r_[0, np.cumsum(step)]
    mf = pd.Series(mid).ffill().to_numpy()
    out = {'date': date, 't_ns': f.t_ns.to_numpy()}
    for G in vs.GRID_G:
        H = np.nan_to_num(f[f'H{G}'].to_numpy()).astype(np.int64); e = np.clip(k + H, 0, n - 1); ok = (k + H < n) & (H > 0)
        steps = cs[e + 1] - cs[k + 1]
        sig_f = np.where(ok & (steps > .8 * H), np.sqrt((c2[e + 1] - c2[k + 1]) / np.maximum(steps, 1)), np.nan)
        path = c1[e + 1] - c1[k + 1]
        out[f'vsur{G}'] = np.log((sig_f + vs.EPS) / (f.sigma600.to_numpy() + vs.EPS)).astype(np.float32)
        out[f'eff{G}'] = np.where(ok & (path > 0), np.abs(mf[e] - mf[k]) / np.maximum(path, 1e-9), np.nan).astype(np.float32)
        out[f'cap{G}'] = (H >= 900).astype(np.float32)
    return out


def brier_parts(y, p, bins=20):
    q = np.clip((p * bins).astype(int), 0, bins - 1); ybar = y.mean(); N = len(y)
    rel = res = 0.
    for b in range(bins):
        m = q == b
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2; res += m.sum() * (y[m].mean() - ybar) ** 2
    return dict(reliability=rel / N, resolution=res / N, uncertainty=float(ybar * (1 - ybar)), brier=float(np.mean((p - y) ** 2)))


def seg_table(name, key, y, p, base):
    rows = []
    for v in sorted(pd.unique(key)):
        m = key == v
        if m.sum() < 2000 or not (0 < y[m].mean() < 1):
            continue
        rows.append(dict(segment=name, value=str(v), n=int(m.sum()), rate=float(y[m].mean()), pred=float(p[m].mean()),
                         auc=float(roc_auc_score(y[m], p[m])), gap=float(p[m].mean() - y[m].mean()),
                         bss=float(1 - np.mean((p[m] - y[m]) ** 2) / np.mean((base - y[m]) ** 2))))
    return rows


def main():
    t0 = time.perf_counter()
    df = vt.load('real'); df = df.reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        hs = list(pool.map(hindsight_day, sorted(df.date.unique())))
    h = pd.concat([pd.DataFrame(x) for x in hs], ignore_index=True)
    df = df.merge(h, on=['date', 't_ns'], how='left', validate='one_to_one')
    print('hindsight', round(time.perf_counter() - t0, 1), flush=True)
    summ = json.loads((OUT / 'volsense_summary.json').read_text(encoding='utf8'))
    fin = df.month.isin(vt.TRAIN + vt.VALID).to_numpy(); te = df.month.isin(vt.TEST).to_numpy()
    res = dict(scales={}, note='oracles use hindsight inputs; they bound what any volatility sense can reach')
    for G in vs.GRID_G:
        cfg = summ['scales'][str(G)]['chosen']; r = res['scales'][str(G)] = {}
        lab = df[f'y{G}'].notna().to_numpy()
        tr_m = fin & lab & (df.sec_london.to_numpy() % 10 == 0); te_m = te & lab
        ytr, yte = df.loc[tr_m, f'y{G}'].to_numpy().astype(int), df.loc[te_m, f'y{G}'].to_numpy().astype(int)
        base = ytr.mean()
        sets = dict(sigma=vt.SIGMA, full=vt.FULL, oracle_vol=[f'vsur{G}', f'cap{G}', 'log_sigma'], oracle_trend=[f'vsur{G}', f'cap{G}', 'log_sigma', f'eff{G}'],
                    oracle_trend_full=vt.FULL + [f'vsur{G}', f'eff{G}'])
        preds = {}
        for name, cols in sets.items():
            mdl = vt.model(cfg).fit(df.loc[tr_m, cols].to_numpy(np.float32), ytr)
            p = mdl.predict_proba(df.loc[te_m, cols].to_numpy(np.float32))[:, 1]; preds[name] = p
            r[name] = dict(auc=float(roc_auc_score(yte, p)), skill=float(1 - log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)) / log_loss(yte, np.full(len(yte), base))),
                           **brier_parts(yte.astype(float), p))
        # volatility itself: log(sigma_future / sigma600) on the label horizon
        v = df[f'vsur{G}'].to_numpy(); vtr = tr_m & np.isfinite(v); vte = te_m & np.isfinite(v)
        for name, cols in (('vol_r2_sigma', vt.SIGMA), ('vol_r2_full', vt.FULL)):
            reg = HistGradientBoostingRegressor(learning_rate=.05, max_iter=300, early_stopping=False, random_state=SEED, **vt.CONFIGS[cfg])
            reg.fit(df.loc[vtr, cols].to_numpy(np.float32), v[vtr]); pv = reg.predict(df.loc[vte, cols].to_numpy(np.float32))
            ls = df.loc[vte, 'log_sigma'].to_numpy()                       # level log(sigma_future) = change + log sigma600
            r[name] = dict(change=float(r2_score(v[vte], pv)), level=float(r2_score(v[vte] + ls, pv + ls)))
        # gradient of the log-loss with respect to the logit: p - y
        p = preds['full']; g = p - yte
        r['gradient'] = dict(mean=float(g.mean()), std=float(g.std()), mean_abs=float(np.abs(g).mean()),
                             share_explainable=float(r['full']['resolution'] / r['full']['uncertainty']),
                             share_explainable_oracle_vol=float(r['oracle_vol']['resolution'] / r['oracle_vol']['uncertainty']),
                             hist=np.histogram(g, bins=np.linspace(-1, 1, 41))[0].tolist())
        # stability by market fragment
        rows = df.loc[te_m]
        hour = (rows.sec_london.to_numpy() // 3600).astype(int)
        sdec = pd.qcut(rows.sigma600.rank(method='first'), 10, labels=False).to_numpy()
        ev = np.where((np.abs(rows.ev_us_data) <= 15) | (np.abs(rows.ev_ny_open) <= 15), 'около событий США', 'остальное')
        spr = np.select([rows.spread <= 2, rows.spread <= 3], ['≤2', '3'], '≥4')
        wd = rows.weekday.astype(int).to_numpy()
        seg = []
        for name, key in (('час от Лондона', hour), ('дециль σ600', sdec), ('события', ev), ('спред', spr), ('день недели', wd)):
            seg += seg_table(name, key, yte, p, base)
        r['segments'] = seg
        r['per_day'] = [dict(date=dte, auc=float(roc_auc_score(yte[(rows.date == dte).to_numpy()], p[(rows.date == dte).to_numpy()])),
                             rate=float(yte[(rows.date == dte).to_numpy()].mean()))
                        for dte in sorted(rows.date.unique()) if 0 < yte[(rows.date == dte).to_numpy()].mean() < 1]
        print(G, json.dumps({k: (round(v['auc'], 4) if isinstance(v, dict) and 'auc' in v else v) for k, v in r.items() if k not in ('segments', 'per_day', 'gradient')}), flush=True)
    (OUT / 'volsense_diag.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print('done', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
