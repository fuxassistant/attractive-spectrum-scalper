"""How much of the opportunity label is future volatility: P(y | realised sigma ratio) on the market and the null path.

For every row of volsense_v1 (all 236 days, uncapped horizons only) the hindsight ratio r = sigma over (t, t + H] /
sigma600 at t. The null path has the same increment sizes per second, so r is identical on both paths; any gap
between the two curves is path shape (signs), not volatility. A Brownian reference (continuous path, horizon
T = 2 r^2 in units of G) is simulated for comparison.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from volsense_diag import hindsight_day, DATA, OUT     # noqa: E402
from spectrum.cognition import volsense as vs          # noqa: E402
from spectrum.cognition import race as rc              # noqa: E402

EDGES = np.round(np.arange(-1., 1.01, .1), 2)
FROZEN_SIGMA = .3


def day(date):
    out = {}
    for kind in ('real', 'null'):
        f = pd.read_parquet(DATA / kind / f'{date}.parquet', columns=['t_ns', 'sigma600'] + [f'{s}{G}' for G in vs.GRID_G for s in ('RL', 'RS')])
        h = hindsight_day(date, kind)
        for G in vs.GRID_G:
            _, R = rc.strength(f[f'RL{G}'].to_numpy(), f[f'RS{G}'].to_numpy(), vs.RHO)
            ok = np.isfinite(R) & (f.sigma600.to_numpy() >= FROZEN_SIGMA) & (h[f'cap{G}'] == 0) & np.isfinite(h[f'vsur{G}'])
            b = np.digitize(h[f'vsur{G}'][ok], EDGES); y = (R[ok] >= vs.RHO).astype(float)
            out[(kind, G)] = (np.bincount(b, minlength=len(EDGES) + 1), np.bincount(b, weights=y, minlength=len(EDGES) + 1))
    return out


def brownian(rs, n=40000, steps=1500, seed=1):
    rng = np.random.default_rng(seed); p = []
    for r in rs:
        T = 2 * r * r; W = np.cumsum(rng.standard_normal((n, steps)) * np.sqrt(T / steps), axis=1)
        def side(X):
            t2 = np.where((X >= 2).any(1), (X >= 2).argmax(1), steps); t1 = np.where((X <= -1).any(1), (X <= -1).argmax(1), steps)
            return (t2 < t1) & (t2 < steps)
        p.append(float((side(W) | side(-W)).mean()))
    return p


def main():
    days = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))['days']
    acc = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for o in pool.map(day, days):
            for k, (n, s) in o.items():
                a = acc.setdefault(k, [np.zeros(len(EDGES) + 1), np.zeros(len(EDGES) + 1)]); a[0] += n; a[1] += s
    centers = np.r_[EDGES[0] - .05, (EDGES[:-1] + EDGES[1:]) / 2, EDGES[-1] + .05]
    res = dict(log_r=centers.round(3).tolist(), curves={})
    for (kind, G), (n, s) in acc.items():
        res['curves'][f'{kind}{G}'] = dict(n=n.astype(int).tolist(), rate=np.where(n >= 500, s / np.maximum(n, 1), np.nan).round(4).tolist())
    res['brownian'] = brownian(np.exp(centers))
    (OUT / 'volsense_curve.json').write_text(json.dumps(res, indent=1).replace('NaN', 'null'), encoding='utf8')
    for G in vs.GRID_G:
        print(G, 'real', res['curves'][f'real{G}']['rate'][::2], '\n   null', res['curves'][f'null{G}']['rate'][::2])
    print('bm', np.round(res['brownian'][::2], 3).tolist())


if __name__ == '__main__':
    main()
