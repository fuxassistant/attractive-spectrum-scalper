"""Build the volatility-detector dataset for every trading day (market path and sign-flipped null path).

Rows: every 5th second of the London open .. NY close session with a known sigma600. Columns: identity
(date, t_ns, seconds since London open, mid in ticks, sigma600), all causal features of spectrum.cognition.
volsense, and for each scale G the opportunity strengths R_long / R_short and the horizon H.
The null path keeps the 1-second increment sizes, volume, book and spread of the day and flips the signs
of the increments at random (deterministic seed per date). Day statistics feed the next day's context.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.cognition import volsense as vs
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day, trading_days

OUT = Path('data/derived/volsense_v1')
MONTHS = ('2025-10', '2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06', '2026-07', '2026-08', '2026-09')
STEP = 5


def frame(b, m, mid, k):
    f = vs.features(b, m, mid); lab = vs.labels(b, mid); f.update(vs.recent(lab, m['n']))
    cols = dict(date=m['date'], t_ns=m['start_ns'] + k.astype(np.int64) * 10**9, sec_london=(k - m['london']).astype(np.int32),
                mid=mid[k].astype(np.float32), sigma600=b['sigma'][k].astype(np.float32))
    for c in vs.FEATURES:
        cols[c] = np.asarray(f[c], float)[k].astype(np.float32)
    for G in vs.GRID_G:
        cols[f'RL{G}'] = lab[G]['R_long'][k].astype(np.float32); cols[f'RS{G}'] = lab[G]['R_short'][k].astype(np.float32)
        cols[f'H{G}'] = lab[G]['H'][k].astype(np.float32)
    return pd.DataFrame(cols)


def build(date):
    t0 = time.perf_counter()
    if (OUT / 'real' / f'{date}.parquet').exists() and (OUT / 'null' / f'{date}.parquet').exists():
        return date, 'exists'
    try:
        b, m = base_day(date)
    except Exception as e:
        return date, f'skip: {e!r}'[:160]
    k = np.arange(m['london'], m['ny_close'])
    k = k[np.asarray(b['sigma_known'], bool)[k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
    k = k[(k - m['london']) % STEP == 0]
    if len(k) < 1000:
        return date, f'skip: {len(k)} session rows'
    null = rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(61 * 100000 + int(date.replace('-', '')) % 100000))
    for kind, mid in (('real', b['mid']), ('null', null)):
        frame(b, m, mid, k).to_parquet(OUT / kind / f'{date}.parquet', index=False)
        save_json(OUT / 'stats' / f'{date}_{kind}.json', vs.day_stats(b, m, mid))
    return date, f'{len(k)} rows {time.perf_counter() - t0:.1f}s'


def main(workers=6, months=MONTHS):
    for sub in ('real', 'null', 'stats'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    days = trading_days(months)
    code = {p: sha(Path('spectrum/cognition') / p) for p in ('volsense.py', 'race.py', 'hotpath.py')}
    save_json(OUT / 'protocol.json', dict(version=vs.VERSION, step=STEP, features=vs.FEATURES, families=vs.FAMILIES, grid=list(vs.GRID_G),
                                          rho=vs.RHO, months=list(months), code=code, semantics=vs.__doc__.strip()))
    done, skipped = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for d, msg in pool.map(build, days):
            print(d, msg, flush=True); (skipped if msg.startswith('skip') else done).append(d)
    save_json(OUT / 'manifest.json', dict(version=vs.VERSION, days=sorted(done), skipped=skipped, code=code))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); main(a.workers)
