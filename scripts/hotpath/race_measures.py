"""Which measure of the race magnitude G gives the smoothest opportunity and direction lines?

Horizons stay as in race_week: H = clip(2 (G0 / sigma600)^2, 20 s, 15 min) for G0 in 12/18/27/40.
The barrier becomes G(t) = G0 * m(t). The raw state behind m(t) is causal, but m is normalised to a median of 1
over the whole week (a hindsight constant, acceptable only because the barrier defines labels, never features)
so every measure has the same typical G and no measure wins by simply being larger; clipped to [0.5, 2]:
  fixed    1
  slowvol  sigma3600 / sigma600       (barrier wider when the last minutes are calmer than the hour)
  fastvol  sigma60 / sigma600         (wider when the last minute is hotter than ten)
  spread   current spread
  chop     path length / range of the mid over the last 300 s
  flicker  direction changes of the mid over the last 300 s
Smoothness functions on the per-second lines (direction = dominance (long - short) / 2, opportunity = max(.,0)):
  flips/h of the winner, mean run (s), total variation of opportunity per second, share of direction
  power at periods < 60 s, autocorrelation at 30 s of direction and of opportunity.
Every measure is also raced on two sign-flipped null paths with the same barrier schedule: what the
market adds beyond geometry is the real / null ratio.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day

DAYS = ('2025-11-17', '2025-11-18', '2025-11-19', '2025-11-20', '2025-11-21')
MEASURES = ('fixed', 'slowvol', 'fastvol', 'spread', 'chop', 'flicker')
SEEDS = (21, 22)
OUT = Path('reports/race_v1')


def state(b):
    mid = pd.Series(b['mid']); d = mid.diff()
    path = d.abs().rolling(300, min_periods=200).sum(); rng = mid.rolling(300, min_periods=200).max() - mid.rolling(300, min_periods=200).min()
    sgn = np.sign(d.where(d != 0)).ffill()
    flips = (sgn != sgn.shift()).astype(float).where(d != 0, 0.).rolling(300, min_periods=200).sum()
    sig = np.maximum(b['sigma'], 1e-6)
    return dict(fixed=np.ones(len(mid)),
                slowvol=np.where(b['sigma3600_known'], b['sigma3600'] / sig, np.nan),
                fastvol=b['sigma60'] / sig, spread=b['spread'],
                chop=(path / rng.where(rng > 0)).to_numpy(), flicker=flips.to_numpy())


def acf(x, lag):
    a, c = x[lag:], x[:-lag]; m = np.isfinite(a) & np.isfinite(c)
    return float(np.corrcoef(a[m], c[m])[0, 1]) if m.sum() > 10 and a[m].std() > 0 and c[m].std() > 0 else np.nan


def smooth(r):
    w = np.where(r['winner'] == rc.CENSORED, 0, r['winner'])
    rl = rc.runs(w); dom = np.nan_to_num(r['dominance']); opp = np.nan_to_num(r['opp'])
    f = np.fft.rfft(dom - dom.mean()); p = np.abs(f) ** 2; fr = np.fft.rfftfreq(len(dom), 1.)
    hours = len(w) / 3600
    won = np.isin(r['winner'], (1, -1)); valid = r['winner'] != rc.CENSORED
    return dict(flips_h=float(np.sum(w[1:] != w[:-1]) / hours), run_s=float(rl.mean()) if len(rl) else 0.,
                tv_opp=float(np.mean(np.abs(np.diff(opp)))), hf_dir=float(p[fr > 1 / 60].sum() / p[1:].sum()),
                acf30_dir=acf(dom, 30), acf30_opp=acf(opp, 30),
                resolved=float(won[valid].mean()), clean=float(r['clean'][won].mean()) if won.any() else np.nan,
                opp_share=float((opp > 0).mean()))


def main():
    days = []
    for date in DAYS:
        b, m = base_day(date)
        k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
        days.append((date, b, k, {name: v[k] for name, v in state(b).items()}))
    med = {name: float(np.nanmedian(np.concatenate([d[3][name] for d in days]))) for name in MEASURES}
    rows = []
    for date, b, k, raw in days:
        nulls = [rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(s * 100000 + int(date.replace('-', '')) % 100000)) for s in SEEDS]
        for G0 in rc.GRID_G:
            for name in MEASURES:
                mult = np.clip(np.nan_to_num(raw[name] / med[name], nan=1.), .5, 2.)
                bar = G0 * mult
                r = rc.race(b['mid'], b['seg'], k, b['sigma'][k], G0, barrier=bar)
                base = dict(date=date, G0=G0, measure=name, G_med=float(np.median(bar)),
                            need=float(np.median((bar + rc.C_LOSS) / (2 * bar - rc.C_WIN + rc.C_LOSS))))
                rows.append(dict(base, variant='real', **smooth(r)))
                for s, nm in zip(SEEDS, nulls):
                    rows.append(dict(base, variant='null', **smooth(rc.race(nm, b['seg'], k, b['sigma'][k], G0, barrier=bar))))
        print(date, flush=True)
    pd.DataFrame(rows).to_parquet(OUT / 'race_measures.parquet', index=False)
    (OUT / 'race_measures_norm.json').write_text(json.dumps(med, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
