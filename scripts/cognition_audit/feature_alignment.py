"""Alignment audit: representation features vs independent recomputation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import day_tapes, captures_by_date, Quotes
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay
from spectrum.player.contract import NS

date = sys.argv[2]
x = pd.read_parquet(f'data/derived/opportunity_signatures_v1/{date}/x.parquet')
rep = pd.read_parquet(f'data/derived/representation_v1/{date}.parquet')
f = x.merge(rep, on='origin_id')
day = MonthDay(Path('data/derived/basket_multimonth_v1') / date[:7] / date)
q = Quotes(QuoteTape.from_day(day))
start = int(day.meta['start_ns'])
member = captures_by_date(months=(date[:7],))[date][0]['member_key']
for _, tape, _ in day_tapes(member, [date]):
    pass
T, B, S = ord('T'), ord('B'), ord('A')
rows = []
for r in f.sample(60, random_state=1).itertuples():
    t = int(r.intent_ns); k = (t - start) // NS; b0 = start + k * NS  # grid boundary used by the features
    def mid_at(ts):
        j = np.searchsorted(q.good_t, ts, side='right') - 1
        g = q.good[j]; return (q.bid[g] + q.ask[g]) / 2e8
    ret = (mid_at(b0) - mid_at(b0 - 300 * NS)) / (r.sigma600_ticks_per_sqrt_s * np.sqrt(300))
    lo, hi = np.searchsorted(tape['ts'], [t - 300 * NS, t])
    a = tape['action'][lo:hi]; s = tape['side'][lo:hi]; z = tape['size'][lo:hi]
    buy, sell = z[(a == T) & (s == B)].sum(), z[(a == T) & (s == S)].sum()
    unk = z[(a == T) & (s != B) & (s != S)].sum(); v = buy + sell + unk
    rows.append(dict(ret_feature=r.price_ret300_sigma, ret_independent=ret,
                     pressure_low_feature=r.flow_pressure300_low, pressure_low_independent=(buy - sell - unk) / max(v, 1),
                     volume_feature=np.expm1(r.flow_log_volume300), volume_independent=v))
d = pd.DataFrame(rows)
print(d.round(3).head(8).to_string())
print('corr ret', round(d.ret_feature.corr(d.ret_independent), 4), 'max abs diff', round((d.ret_feature - d.ret_independent).abs().max(), 4))
print('corr pressure', round(d.pressure_low_feature.corr(d.pressure_low_independent), 4),
      'corr volume', round(d.volume_feature.corr(d.volume_independent), 4),
      'median volume ratio', round((d.volume_feature / d.volume_independent).median(), 4))
