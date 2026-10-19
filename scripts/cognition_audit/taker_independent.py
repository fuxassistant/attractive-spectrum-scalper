"""Independent re-implementation checks on one real day.

1. Taker branches: plain Python loop over quotes with the kernel rules,
   compared with saved signature nets.
2. P&L decomposition of taker trades into fee, half-spreads and mid move.
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import day_tapes, captures_by_date, Quotes, LATENCY
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.signatures import MENU_PLANS
from spectrum.cognition import queue as qu
from spectrum.playground.data import MonthDay
from spectrum.player.contract import NS

date = sys.argv[2]
day = MonthDay(Path('data/derived/basket_multimonth_v1') / date[:7] / date)
q = Quotes(QuoteTape.from_day(day))
br = pd.read_parquet(f'data/derived/opportunity_signatures_v1/{date}/branches.parquet')
x = pd.read_parquet(f'data/derived/opportunity_signatures_v1/{date}/x.parquet')
br = br.merge(x[['origin_id', 'intent_ns']], on='origin_id')

# ---- 1. taker -------------------------------------------------------------
def naive_taker(intent, side, plan):
    take, stop, hold = MENU_PLANS[plan]
    t, bid, ask, ok, ep = q.t, q.bid, q.ask, q.ok, q.epoch
    j = np.searchsorted(t, intent, side='right') - 1
    if j < 0 or not ok[j]:
        return 'NOT_ELIGIBLE', None, None
    j = np.searchsorted(t, intent + LATENCY, side='right')
    while j < len(t) and t[j] <= intent + 2 * NS and not ok[j]:
        j += 1
    if j >= len(t) or t[j] > intent + 2 * NS:
        return 'NOT_FILLED', None, None
    entry = ask[j] if side > 0 else bid[j]; t0 = t[j]; e0 = ep[j]; horizon = intent + (hold + 5) * NS
    mid0 = (bid[j] + ask[j]) / 2
    k = j
    while k < len(t) and t[k] < horizon:
        if not ok[k] or ep[k] != e0:
            return 'CENSORED', None, None
        px = bid[k] if side > 0 else ask[k]; mv = side * (px - entry)
        if t[k] >= q.deadline or mv <= -stop * 1e8 or mv >= take * 1e8 or t[k] - t0 >= hold * NS:
            mid1 = (bid[k] + ask[k]) / 2
            net = side * (px - entry) / 1e9 - .30
            return 'COMPLETE', net, dict(fee=-.30, entry_half=-side * (entry - mid0) / 1e9 if side > 0 else (entry - mid0) / 1e9,
                                        exit_half=side * (px - mid1) / 1e9, mid_move=side * (mid1 - mid0) / 1e9)
        k += 1
    return 'CENSORED', None, None

sample = br.sample(1500, random_state=7)
rows = []
for r in sample.itertuples():
    st, net, dec = naive_taker(int(r.intent_ns), int(r.side), int(r.plan_id))
    rows.append(dict(status=st, status_sig=r.status, net=net, net_sig=getattr(r, 'net_usd', np.nan), **(dec or {})))
f = pd.DataFrame(rows)
both = f[(f.status == 'COMPLETE') & (f.status_sig == 'COMPLETE')]
print('TAKER independent: status agree', (f.status == f.status_sig).mean().round(4), 'n', len(f),
      'max |net diff|', float((both.net - both.net_sig).abs().max()))
print(f[f.status != f.status_sig][['status', 'status_sig']].value_counts().head())
print('decomposition mean $:', both[['fee', 'entry_half', 'exit_half', 'mid_move']].mean().round(4).to_dict(), 'net', round(both.net.mean(), 4))

