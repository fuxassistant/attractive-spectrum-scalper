"""Difference-in-differences: does the full-vs-sigma gain exist on the null too? G=40 clipping. Censored-side labels."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/hotpath')
import volsense_train as vt
from sklearn.metrics import roc_auc_score
from spectrum.cognition import volsense as vs
real, null = vt.load('real'), vt.load('null')
def t_of(x): x = np.asarray(x); return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
for G in (12, 40):
    for kind, df in (('real', real), ('null', null)):
        fin = df.month.isin(vt.TRAIN + vt.VALID).to_numpy(); te = df.month.isin(vt.TEST).to_numpy()
        cfg = 'shallow'
        out = {}
        for name, cols in (('full', vt.FULL), ('sigma', vt.SIGMA)):
            X, y, _ = vt.xy(df[fin], cols, G, 20); Xt, yt, mt = vt.xy(df[te], cols, G, 5)
            p = np.clip(vt.model(cfg).fit(X, y).predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6); out[name] = p
        rows = df[te][mt]
        ll = lambda p: -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
        dd = pd.DataFrame(dict(date=rows.date.to_numpy(), a=ll(out['sigma']) - ll(out['full']))).groupby('date').a.sum()
        print(f'G{G} {kind}: AUC full {roc_auc_score(yt, out["full"]):.4f} sigma {roc_auc_score(yt, out["sigma"]):.4f}; per-day-sum t(full better) {t_of(dd):.2f}, days better {np.mean(dd > 0):.2f}', flush=True)
        if G == 40:
            H = rows['H40'].to_numpy(); c = H >= 900
            print(f'   G40 {kind}: share H clipped {c.mean():.3f}; rate clipped {yt[c].mean():.3f} unclipped {yt[~c].mean():.3f}; AUC full within clipped {roc_auc_score(yt[c], out["full"][c]):.3f} unclipped {roc_auc_score(yt[~c], out["full"][~c]):.3f}; '
                  f'AUC log_sigma alone {roc_auc_score(yt, rows.log_sigma.fillna(-9)):.3f}; AUC of clip indicator {roc_auc_score(yt, (~c).astype(float)):.3f}', flush=True)
# one-side censored rows
for G in vs.GRID_G:
    L, S = real[f'RL{G}'].to_numpy(), real[f'RS{G}'].to_numpy(); one = np.isfinite(L) ^ np.isfinite(S); y = real[f'y{G}'].to_numpy()
    print(f'G{G}: one-side-censored rows {one.mean():.4f}, their y rate {np.nanmean(y[one]):.3f} vs others {np.nanmean(y[~one]):.3f}; median minutes to close {np.median(real.close_min[one]):.1f}')
