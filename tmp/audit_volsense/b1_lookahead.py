"""Perturb the future of a real day after T; every feature and recent() at <= T must stay. Also compare with stored parquet."""
import numpy as np, pandas as pd
from spectrum.cognition import volsense as vs, race as rc
from spectrum.cognition.hotpath import base_day
date = '2026-07-14'
b, m = base_day(date)
f0 = vs.features(b, m); lab0 = vs.labels(b); f0.update(vs.recent(lab0, m['n']))
rng = np.random.default_rng(0)
for T in (m['london'] + 3600, m['london'] + 6 * 3600 + 1234):
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    nfut = m['n'] - T - 1
    b2['mid'][T + 1:] = b2['mid'][T + 1:] + np.cumsum(rng.normal(0, 3, nfut))
    for k in ('spread', 'trade_buy', 'trade_sell', 'trade_unknown', 'add_bid', 'add_ask', 'remove_bid', 'remove_ask', 'depth1', 'depth4',
              'depth1_total', 'depth4_total', 'depth12_total', 'sigma', 'sigma60', 'sigma3600'):
        b2[k][T + 1:] = b2[k][T + 1:] * rng.uniform(.3, 3, nfut)
    f1 = vs.features(b2, m); lab1 = vs.labels(b2); f1.update(vs.recent(lab1, m['n']))
    bad = [c for c in vs.FEATURES if not np.allclose(np.asarray(f0[c], float)[:T + 1], np.asarray(f1[c], float)[:T + 1], equal_nan=True)]
    print('T', T, 'features changed at <=T:', bad)
# stored parquet equals recomputation
p = pd.read_parquet(f'data/derived/volsense_v1/real/{date}.parquet')
k = p.sec_london.to_numpy() + m['london']
worst = max((np.nanmax(np.abs(np.asarray(f0[c], float)[k].astype(np.float32) - p[c].to_numpy())) if np.isfinite(p[c]).any() else 0., c) for c in vs.FEATURES)
print('max |recomputed - stored| over features:', worst)
for G in vs.GRID_G:
    print(G, 'label equal', np.allclose(lab0[G]['R_long'][k].astype(np.float32), p[f'RL{G}'], equal_nan=True))
# fmax one-side-censored rows in stored data (all days quick count on a sample of days)
import glob
cnt = {G: [0, 0] for G in vs.GRID_G}
for fn in sorted(glob.glob('data/derived/volsense_v1/real/*.parquet'))[::5]:
    q = pd.read_parquet(fn, columns=[f'R{s}{G}' for G in vs.GRID_G for s in 'LS'])
    for G in vs.GRID_G:
        L, S = q[f'RL{G}'].to_numpy(), q[f'RS{G}'].to_numpy(); one = np.isfinite(L) ^ np.isfinite(S)
        cnt[G][0] += one.sum(); cnt[G][1] += len(L)
print('one-side-censored rows (every 5th day):', {G: f'{a}/{b_}' for G, (a, b_) in cnt.items()})
