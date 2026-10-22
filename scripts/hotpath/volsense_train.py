"""Train and evaluate the scalper volatility detector exactly as in reports/volsense_v1/PREREGISTRATION_RU.md.

Selection and family ablations on the validation months, one final read of the test months with models refitted
on train + validation; the same pipeline on the sign-flipped null path (train and test on null data).
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from spectrum.cognition import volsense as vs
from spectrum.cognition import race as rc

DATA = Path('data/derived/volsense_v1'); OUT = Path('reports/volsense_v1')
TRAIN = ('2025-10', '2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04')
VALID = ('2026-05', '2026-06')
TEST = ('2026-07', '2026-08', '2026-09')
CONFIGS = {'shallow': dict(max_depth=3, min_samples_leaf=2000), 'deep': dict(max_depth=6, min_samples_leaf=300)}
SIGMA = ['log_sigma', 'sig60_r', 'sig3600_r', 'london_min']
CONTEXT = ['prev_sigma', 'prev_range', 'prev_vol', 'range_vs_prev', 'night_vs_prev']
FULL = vs.FEATURES + CONTEXT
FAMILY = {k: list(v) for k, v in vs.FAMILIES.items()}; FAMILY['day'] = FAMILY['day'] + CONTEXT
PAGE_WEEK = ('2026-07-13', '2026-07-14', '2026-07-15', '2026-07-16', '2026-07-17')
FROZEN_SIGMA = .3
SEED = 7


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load(kind):
    days = sorted(p.stem for p in (DATA / kind).glob('2*.parquet'))
    stats = {d: json.loads((DATA / 'stats' / f'{d}_{kind}.json').read_text(encoding='utf8')) for d in days}
    frames = []
    for i, d in enumerate(days):
        f = pd.read_parquet(DATA / kind / f'{d}.parquet')
        prev = stats[days[i - 1]] if i else None
        f['prev_sigma'] = np.float32(prev['sess_sigma'] if prev else np.nan)
        f['prev_range'] = np.float32(prev['sess_range'] if prev else np.nan)
        f['prev_vol'] = np.float32(prev['sess_vol'] if prev else np.nan)
        f['range_vs_prev'] = (f.today_range - f.prev_range).astype(np.float32)
        f['night_vs_prev'] = (f.night_sigma - f.prev_sigma).astype(np.float32)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df['month'] = df.date.str[:7]
    frozen = df.sigma600.to_numpy() < FROZEN_SIGMA                     # amendment 1: frozen holiday tails out
    for G in vs.GRID_G:
        _, R = rc.strength(df[f'RL{G}'].to_numpy(), df[f'RS{G}'].to_numpy(), vs.RHO)   # amendment 1: open outcomes stay NaN
        df[f'y{G}'] = np.where(np.isfinite(R) & ~frozen, (R >= vs.RHO).astype(np.float32), np.nan)
    return df


def model(cfg):
    return HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, early_stopping=False, random_state=SEED, **CONFIGS[cfg])


def xy(df, cols, G, stride):
    m = df[f'y{G}'].notna().to_numpy() & (df.sec_london.to_numpy() % stride == 0)
    return df.loc[m, cols].to_numpy(np.float32), df.loc[m, f'y{G}'].to_numpy().astype(int), m


def scores(y, p, base):
    p = np.clip(p, 1e-6, 1 - 1e-6); ll = log_loss(y, p, labels=[0, 1]); llb = log_loss(y, np.full(len(y), base), labels=[0, 1])
    return dict(auc=float(roc_auc_score(y, p)), ll=float(ll), ll_base=float(llb), skill=float(1 - ll / llb), rate=float(y.mean()), n=int(len(y)))


def per_day(df_rows, y, p, q=None):
    d = pd.DataFrame(dict(date=df_rows.date.to_numpy(), y=y, p=np.clip(p, 1e-6, 1 - 1e-6)))
    d['ll'] = -(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p))
    if q is not None:
        d['llq'] = -(d.y * np.log(np.clip(q, 1e-6, 1 - 1e-6)) + (1 - d.y) * np.log(1 - np.clip(q, 1e-6, 1 - 1e-6)))
    return d


def t_of(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real, null = load('real'), load('null'); log('loaded', len(real), len(null))
    tr, va, te = (real.month.isin(s).to_numpy() for s in (TRAIN, VALID, TEST))
    ntr, nva, nte = (null.month.isin(s).to_numpy() for s in (TRAIN, VALID, TEST))
    res = dict(version='VOLSENSE_TRAIN_V1', splits=dict(train=TRAIN, valid=VALID, test=TEST), configs=CONFIGS, scales={})
    page = dict(grid=list(vs.GRID_G), days=[])
    for G in vs.GRID_G:
        r = res['scales'][str(G)] = {}
        # 1. selection on validation
        Xtr, ytr, _ = xy(real[tr], FULL, G, 10); Xva, yva, mva = xy(real[va], FULL, G, 5); base_tr = float(ytr.mean())
        sel = {}
        for cfg in CONFIGS:
            p = model(cfg).fit(Xtr, ytr).predict_proba(Xva)[:, 1]; sel[cfg] = scores(yva, p, base_tr)
        chosen = min(sel, key=lambda c: sel[c]['ll']); r['selection'] = sel; r['chosen'] = chosen
        log(G, 'selection', {c: round(v['ll'], 5) for c, v in sel.items()}, '->', chosen)
        # 2. family ablations on validation (training every 20 s)
        sets = {'sigma': SIGMA, 'full': FULL}
        for fam, cols in FAMILY.items():
            sets[f'only_{fam}'] = list(dict.fromkeys(SIGMA + cols)); sets[f'drop_{fam}'] = [c for c in FULL if c not in cols]
        abl = {}
        for name, cols in sets.items():
            Xa, ya, _ = xy(real[tr], cols, G, 20); Xv, yv, _ = xy(real[va], cols, G, 5)
            abl[name] = scores(yv, model(chosen).fit(Xa, ya).predict_proba(Xv)[:, 1], float(ya.mean()))
        r['ablation_valid'] = abl; log(G, 'ablation', {k: round(v['auc'], 4) for k, v in abl.items()})
        # 3. final: refit on train + valid, read the test months once
        fin = tr | va
        Xf, yf, _ = xy(real[fin], FULL, G, 10); Xs, ys, _ = xy(real[fin], SIGMA, G, 10); base = float(yf.mean())
        mf, ms = model(chosen).fit(Xf, yf), model(chosen).fit(Xs, ys)
        Xt, yt, mt = xy(real[te], FULL, G, 5); Xts, _, _ = xy(real[te], SIGMA, G, 5)
        pf, ps = mf.predict_proba(Xt)[:, 1], ms.predict_proba(Xts)[:, 1]
        rows = real[te][mt]
        r['test_full'], r['test_sigma'] = scores(yt, pf, base), scores(yt, ps, base)
        pdx = per_day(rows, yt, pf, ps); dd = pdx.groupby('date')[['ll', 'llq']].sum()
        r['V1_t'] = t_of(dd.llq - dd.ll); r['V1_days_better'] = float(((dd.llq - dd.ll) > 0).mean())
        q = pd.qcut(pd.Series(pf).rank(method='first'), 10, labels=False).to_numpy()
        dec = pd.DataFrame(dict(q=q, p=pf, y=yt)).groupby('q').agg(p=('p', 'mean'), y=('y', 'mean'))
        r['deciles'] = dec.round(4).reset_index().to_dict('records'); r['V2_lift'] = float(dec.y.iloc[-1] / yt.mean())
        r['V3_month_auc'] = {mo: float(roc_auc_score(yt[(rows.month == mo).to_numpy()], pf[(rows.month == mo).to_numpy()]))
                             for mo in sorted(rows.month.unique()) if rows[rows.month == mo].date.nunique() >= 5 and 0 < yt[(rows.month == mo).to_numpy()].mean() < 1}
        day_auc_real = {dte: roc_auc_score(g.y, g.p) for dte, g in pdx.groupby('date') if 0 < g.y.mean() < 1}
        # permutation importance by family on a test subsample
        rng = np.random.default_rng(SEED); sub = rng.choice(len(yt), min(60000, len(yt)), replace=False)
        ll0 = log_loss(yt[sub], np.clip(mf.predict_proba(Xt[sub])[:, 1], 1e-6, 1 - 1e-6))
        imp = {}
        for fam, cols in FAMILY.items():
            Xp = Xt[sub].copy(); idx = [FULL.index(c) for c in cols]; perm = rng.permutation(len(sub)); Xp[:, idx] = Xp[perm][:, idx]
            imp[fam] = float(log_loss(yt[sub], np.clip(mf.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)) - ll0)
        r['importance_test'] = imp
        # 4. null path: same pipeline on null data
        Xn, yn, _ = xy(null[ntr | nva], FULL, G, 10); Xnt, ynt, mnt = xy(null[nte], FULL, G, 5)
        pn = model(chosen).fit(Xn, yn).predict_proba(Xnt)[:, 1]; r['test_null'] = scores(ynt, pn, float(yn.mean()))
        Xns, _, _ = xy(null[ntr | nva], SIGMA, G, 10); Xnts, _, _ = xy(null[nte], SIGMA, G, 5)
        pns = model(chosen).fit(Xns, yn).predict_proba(Xnts)[:, 1]; r['test_null_sigma'] = scores(ynt, pns, float(yn.mean()))
        pnd = per_day(null[nte][mnt], ynt, pn, pns)
        gain_real = (dd.llq - dd.ll); gnd = pnd.groupby('date')[['ll', 'llq']].sum(); gain_null = (gnd.llq - gnd.ll)
        both = gain_real.index.intersection(gain_null.index)
        r['V1_did_t'] = t_of((gain_real[both] - gain_null[both]).to_numpy()); r['V1_null_t'] = t_of(gain_null.to_numpy())
        Hc = rows[f'H{G}'].to_numpy() >= 900
        r['auc_capped'] = float(roc_auc_score(yt[Hc], pf[Hc])) if 0 < yt[Hc].mean() < 1 and Hc.sum() > 1000 else None
        r['auc_uncapped'] = float(roc_auc_score(yt[~Hc], pf[~Hc])) if 0 < yt[~Hc].mean() < 1 else None
        r['capped_share'] = float(Hc.mean())
        day_auc_null = {dte: roc_auc_score(g.y, g.p) for dte, g in pnd.groupby('date') if 0 < g.y.mean() < 1}
        common = sorted(set(day_auc_real) & set(day_auc_null))
        r['V4_t'] = t_of([day_auc_real[d_] - day_auc_null[d_] for d_ in common]); r['V4_days'] = len(common)
        r['valid_quantiles'] = {q_: float(v) for q_, v in zip(('q30', 'q80', 'q95'), np.quantile(mf.predict_proba(Xva)[:, 1], (.3, .8, .95)))}
        log(G, 'test', json.dumps({k: r[k] for k in ('test_full', 'test_sigma', 'test_null', 'test_null_sigma', 'V1_t', 'V1_null_t', 'V1_did_t', 'V2_lift',
                                                   'V3_month_auc', 'V4_t', 'auc_capped', 'auc_uncapped', 'capped_share')}))
        # page data for one test week: every row of the day (labels may be missing near the close)
        for d_ in PAGE_WEEK:
            g0 = real[real.date.to_numpy() == d_]
            if not len(g0):
                continue
            ent = next((x for x in page['days'] if x['date'] == d_), None)
            if ent is None:
                ent = dict(date=d_, t=g0.sec_london.astype(int).tolist(), london_utc=int(((int(g0.t_ns.iloc[0]) - int(g0.sec_london.iloc[0]) * 10**9) // 10**9) % 86400),
                           mid=np.round(g0.mid.to_numpy() / 10, 2).tolist(), G={})
                page['days'].append(ent)
            pa = mf.predict_proba(g0[FULL].to_numpy(np.float32))[:, 1]; psa = ms.predict_proba(g0[SIGMA].to_numpy(np.float32))[:, 1]
            yy = g0[f'y{G}'].to_numpy()
            ent['G'][str(G)] = dict(p=[int(round(v * 1000)) for v in pa], ps=[int(round(v * 1000)) for v in psa],
                                    y=''.join('x' if not np.isfinite(v) else str(int(v)) for v in yy), q=r['valid_quantiles'], base=round(base, 4))
        (OUT / 'volsense_summary.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    page['summary'] = res
    page['days'] = sorted(page['days'], key=lambda x: x['date'])
    (OUT / 'volsense_page.json').write_text(json.dumps(page, separators=(',', ':')), encoding='utf8')
    log('done')


if __name__ == '__main__':
    main()
