"""Attempt 4: nonlinear machines that describe the hot path (kNN analogs, local boosting) vs a global booster.

Prequential: refit at fixed times from today's London open; training rows are hot-path rows whose outcome is
known (t' + h <= T); session-only rows (overnight scales differ). Unbiased day statistic: mean over ALL of the
day's resolved session decisions of pred * z (unfitted decisions contribute 0), and of sign(pred) * move ticks.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

NS = 10**9
STATE = ['ret10', 'ret30', 'ret60', 'ret120', 'ret300', 'ret600', 'ret1800', 'imb30', 'imb120', 'imb600', 'depth1', 'book30',
         'vol_rate', 'd_vwap_today', 'd_avwap', 'va_pos', 'd_win_hi', 'd_win_lo', 'd_today_hi', 'd_today_lo', 'd_poc']
CLIP = 4.


def prep(f):
    X = f[STATE].to_numpy(float)
    return np.clip(X, -CLIP * 3, CLIP * 3)


def run(months, h, refit_s, out):
    fs = sorted(p for p in glob.glob('data/derived/hotpath_v1/2*.parquet') if os.path.basename(p)[:7] in months)
    frames = [pd.read_parquet(p) for p in fs]; rows = []
    glob_X, glob_y = [], []
    for i in range(1, len(frames)):
        prev, today = frames[i - 1], frames[i]
        pool = pd.concat([prev[prev.session], today[today.session]], ignore_index=True)
        y = pool[f'z{h}'].to_numpy(float); ok = np.isfinite(y)
        X = prep(pool); t = pool.t_ns.to_numpy(np.int64); is_today = np.r_[np.zeros(prev.session.sum(), bool), np.ones(today.session.sum(), bool)]
        dec = np.flatnonzero(is_today & ok)
        t_lon = int(today.t_ns[today.minutes_london >= 0].min())
        preds = {k: np.zeros(len(dec)) for k in ('knn50', 'knn400', 'gbm_local', 'gbm_global')}
        T = t_lon + np.arange(0, 15 * 3600, refit_s) * NS
        jd = np.searchsorted(T, t[dec], side='right') - 1
        # global booster: fitted once per day on all earlier days' session rows (expanding)
        g = None
        if len(glob_X) >= 5:
            GX, Gy = np.concatenate(glob_X), np.concatenate(glob_y)
            take = np.random.default_rng(i).choice(len(GX), min(len(GX), 120_000), replace=False)
            g = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=500,
                                              l2_regularization=1., random_state=0).fit(GX[take], Gy[take])
            preds['gbm_global'] = g.predict(X[dec])
        for j in np.unique(jd):
            m = jd == j; Tj = T[j]
            tr = ok & (t + h * NS <= Tj)
            if tr.sum() < 500:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xs, Xd = (X[tr] - mu) / sd, (X[dec[m]] - mu) / sd
            yt = np.clip(y[tr], -5, 5)
            for k in (50, 400):
                preds[f'knn{k}'][m] = KNeighborsRegressor(n_neighbors=k).fit(Xs, yt).predict(Xd) - yt.mean()
            preds['gbm_local'][m] = HistGradientBoostingRegressor(max_depth=2, learning_rate=.05, max_iter=60, min_samples_leaf=300,
                                                                  l2_regularization=1., random_state=0).fit(Xs, yt).predict(Xd) - yt.mean()
        z = y[dec]; mv = pool[f'move{h}'].to_numpy(float)[dec]
        for k, p in preds.items():
            rows.append(dict(date=today.date.iloc[0], model=k, h=h, n=len(dec), py=float((p * z).mean()), sy=float((np.sign(p) * z).mean()),
                             ticks=float((np.sign(p) * mv).mean()), p_sd=float(p.std())))
        tm = today[today.session & today[f'z{h}'].notna()]
        glob_X.append(prep(tm)); glob_y.append(np.clip(tm[f'z{h}'].to_numpy(float), -5, 5))
        print(json.dumps(dict(date=today.date.iloc[0], h=h)), flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)


if __name__ == '__main__':
    months = sys.argv[1].split(','); h = int(sys.argv[2]); out = sys.argv[3]
    run(months, h, 3600, out)
