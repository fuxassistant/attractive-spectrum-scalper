"""Decompose every completed trade: net = -fee + entry half-spread + exit half-spread + mid move."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import Quotes
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE

rows = []
for path in sorted(Path(sys.argv[2] if len(sys.argv) > 2 else 'data/derived/passive_execution_v5').glob('2*.parquet')):
    date = path.stem
    f = pd.read_parquet(path, filters=[('variant', 'in', ['TT', 'PQ_30']), ('status', '==', 'COMPLETE')])
    if not len(f):
        continue
    q = Quotes(QuoteTape.from_day(MonthDay(Path('data/derived/basket_multimonth_v1') / date[:7] / date)))
    def mid(t):
        j = np.searchsorted(q.good_t, t, side='right') - 1
        g = q.good[np.clip(j, 0, None)]
        return (q.bid[g] + q.ask[g]) / 2
    e_mid, x_mid = mid(f.entry_ns.to_numpy(np.int64)), mid(f.exit_ns.to_numpy(np.int64))
    side = f.side.to_numpy(); entry = f.entry_px.to_numpy(float)
    exit_px = entry + side * (f.net_usd.to_numpy() + .30) * 1e9
    f = f.assign(entry_half=side * (e_mid - entry) / 1e9, exit_half=side * (exit_px - x_mid) / 1e9,
                 mid_move=side * (x_mid - e_mid) / 1e9, block='A' if date < '2026-01' else 'B')
    rows.append(f[['variant', 'plan_id', 'block', 'net_usd', 'entry_half', 'exit_half', 'mid_move', 'exit_reason']])
    QUOTE_CACHE.clear()
d = pd.concat(rows)
d['check'] = d.entry_half + d.exit_half + d.mid_move - .30 - d.net_usd
pd.set_option('display.width', 200)
print('identity max error', float(d.check.abs().max()))
print(d.groupby(['block', 'variant', 'plan_id'])[['net_usd', 'entry_half', 'exit_half', 'mid_move']].mean().round(3))
print(d.groupby(['block', 'variant', 'exit_reason'])[['mid_move', 'exit_half']].agg(['mean', 'size']).round(3))
