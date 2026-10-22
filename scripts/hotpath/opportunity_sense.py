"""Volatility sense: can a causal model foresee the new opportunity (a move of RHO * G before a -G stop)?

Label (Y, hindsight): opportunity at scale G0 = some side's favourable move before its own -G0 stop, within
H = clip(2 (G0 / sigma600)^2, 20 s, 15 min), reaches RHO * G0 (RHO = 2). Side = the side that did.
Features (X, causal, side-free, no direction): sigma600 level and its ratios to sigma60 / sigma3600, spread,
mid direction changes and path / range over 300 s, normalised |return| over 60 / 300 s, normalised range
over 300 s, trade and book activity rate (60 s against 30 min), minutes since London open.
Model: gradient boosting per scale, fitted on every A day except the test week (every 10th second),
evaluated on every second of the test week against the training base rate. Loss: binary log-loss; its
gradient with respect to the logit is p - y, which the page draws as the training signal.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day
from spectrum.cognition.path_model import dates_of

TEST = ('2025-11-17', '2025-11-18', '2025-11-19', '2025-11-20', '2025-11-21')
RHO = 2.
GRADES = (1.5, 2., 3.)
OUT = Path('reports/race_v1')
FEATS = ['log_sigma', 'sig60_ratio', 'sig3600_ratio', 'sig3600_known', 'spread', 'flicker300', 'chop300', 'absret60', 'absret300',
         'range300', 'trade_rate', 'book_rate', 'minutes_london']
SEED = 7


def roll_sum(x, w):
    c = np.r_[0., np.cumsum(x)]; i = np.arange(len(x)); return c[i + 1] - c[np.clip(i + 1 - w, 0, None)]


def features(b, m):
    mid = pd.Series(b['mid']); d = mid.diff(); sig = np.maximum(b['sigma'], 1e-6)
    sgn = np.sign(d.where(d != 0)).ffill()
    flips = (sgn != sgn.shift()).astype(float).where(d != 0, 0.).rolling(300, min_periods=200).sum()
    path = d.abs().rolling(300, min_periods=200).sum()
    hi, lo = mid.rolling(300, min_periods=200).max(), mid.rolling(300, min_periods=200).min()
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
    book = b['add_bid'] + b['add_ask'] + b['remove_bid'] + b['remove_ask']
    f = dict(log_sigma=np.log(sig), sig60_ratio=np.log(np.maximum(b['sigma60'], 1e-6) / sig),
             sig3600_ratio=np.where(b['sigma3600_known'], np.log(sig / np.maximum(b['sigma3600'], 1e-6)), 0.),
             sig3600_known=b['sigma3600_known'].astype(float), spread=b['spread'], flicker300=flips.to_numpy(),
             chop300=(path / (hi - lo).where(hi > lo)).to_numpy(),
             absret60=(mid - mid.shift(60)).abs().to_numpy() / (sig * np.sqrt(60)),
             absret300=(mid - mid.shift(300)).abs().to_numpy() / (sig * np.sqrt(300)),
             range300=(hi - lo).to_numpy() / (sig * np.sqrt(300)),
             trade_rate=np.log((roll_sum(vol, 60) / 60 + .01) / (roll_sum(vol, 1800) / 1800 + .01)),
             book_rate=np.log((roll_sum(book, 60) / 60 + .01) / (roll_sum(book, 1800) / 1800 + .01)),
             minutes_london=(np.arange(len(mid)) - m['london']) / 60.)
    return f


def day(date):
    b, m = base_day(date)
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
    f = features(b, m)
    out = dict(date=date, k=k, mid=b['mid'][k], X=np.stack([np.asarray(f[c], float)[k] for c in FEATS], 1).astype(np.float32))
    for G in rc.GRID_G:
        e = rc.excursion(b['mid'], b['seg'], k, b['sigma'][k], G)
        out[f'L{G}'] = e['R_long'].astype(np.float32); out[f'S{G}'] = e['R_short'].astype(np.float32)
    return out


def main():
    dates = dates_of(('2025-10', '2025-11', '2025-12'))
    with ProcessPoolExecutor(max_workers=6) as pool:
        days = list(pool.map(day, dates))
    train = [d for d in days if d['date'] not in TEST]; test = [d for d in days if d['date'] in TEST]
    res = dict(rho=RHO, features=FEATS, train_days=len(train), test_days=[d['date'] for d in test], scales={})
    page = dict(grid=list(rc.GRID_G), rho=RHO, grades=list(GRADES), step=5, days=[{'date': d['date'], 't': (d['k'][::5] - d['k'][0] + 0).tolist()} for d in test])
    for d, pd_ in zip(test, page['days']):
        pd_['t'] = ((d['k'][::5]) - d['k'][0]).tolist(); pd_['t0'] = int(d['k'][0]); pd_['mid'] = np.round(d['mid'][::5] / 10, 2).tolist(); pd_['G'] = {}
    for G in rc.GRID_G:
        def xy(ds, stride):
            X = np.concatenate([d['X'][::stride] for d in ds]); R = np.concatenate([np.fmax(d[f'L{G}'], d[f'S{G}'])[::stride] for d in ds])
            ok = np.isfinite(R); return np.nan_to_num(X[ok]), (R[ok] >= RHO).astype(int)
        Xtr, ytr = xy(train, 10)
        mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=250, min_samples_leaf=500, l2_regularization=1., random_state=SEED).fit(Xtr, ytr)
        base = float(ytr.mean())
        Xte, yte = xy(test, 1)
        p = mdl.predict_proba(Xte)[:, 1]
        imp = permutation_importance(mdl, Xte[::20], yte[::20], scoring='neg_log_loss', n_repeats=3, random_state=SEED)
        dec = pd.DataFrame(dict(p=p, y=yte)).assign(q=lambda x: pd.qcut(x.p.rank(method='first'), 10, labels=False)).groupby('q').agg(p=('p', 'mean'), y=('y', 'mean'))
        perday = []
        for d in test:
            R = np.fmax(d[f'L{G}'], d[f'S{G}']); ok = np.isfinite(R); y = (R[ok] >= RHO).astype(int); pp = mdl.predict_proba(np.nan_to_num(d['X'][ok]))[:, 1]
            perday.append(dict(date=d['date'], rate=float(y.mean()), ll_base=float(log_loss(y, np.full(len(y), base), labels=[0, 1])), ll_model=float(log_loss(y, pp, labels=[0, 1])),
                               auc=float(roc_auc_score(y, pp)) if 0 < y.mean() < 1 else None))
        res['scales'][str(G)] = dict(base_rate_train=base, test_rate=float(yte.mean()), ll_base=float(log_loss(yte, np.full(len(yte), base))), ll_model=float(log_loss(yte, p)),
                                     auc=float(roc_auc_score(yte, p)), deciles=dec.round(4).reset_index().to_dict('records'), per_day=perday,
                                     importance={c: float(v) for c, v in sorted(zip(FEATS, imp.importances_mean), key=lambda z: -z[1])})
        for d, pd_ in zip(test, page['days']):
            L, S = d[f'L{G}'], d[f'S{G}']; R = np.fmax(L, S); side = np.where(L >= S, 1, -1)
            grade = np.select([R >= 3, R >= 2, R >= 1.5], [3, 2, 1], 0)
            pp = mdl.predict_proba(np.nan_to_num(d['X']))[:, 1]
            sub = slice(None, None, 5)
            pd_['G'][str(G)] = dict(code=''.join('abcdefg'[3 + s * g] for s, g in zip(side[sub], grade[sub])),
                                    y=''.join('x' if not np.isfinite(r) else ('1' if r >= RHO else '0') for r in R[sub]),
                                    p=[int(round(v * 1000)) for v in pp[sub]], base=round(base, 4))
        print(G, json.dumps({k: v for k, v in res['scales'][str(G)].items() if k in ('base_rate_train', 'test_rate', 'll_base', 'll_model', 'auc')}), flush=True)
    (OUT / 'opportunity_sense.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    page['summary'] = {G: {k: v for k, v in s.items() if k != 'per_day'} | {'per_day': s['per_day']} for G, s in res['scales'].items()}
    (OUT / 'sense_data.json').write_text(json.dumps(page, separators=(',', ':')), encoding='utf8')


if __name__ == '__main__':
    main()
