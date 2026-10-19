"""Audit: trades vs reconstructed BBO, spread, depth, from the raw MBO tape.

Pure-Python L3 replay (independent of the numba kernel and of quotes.parquet).
For every T record we compare its price with the book's best bid/ask built
from records strictly before it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import day_tapes, captures_by_date
from spectrum.playground.data import MonthDay
from spectrum.cognition.execution import QuoteTape

dates = sys.argv[2:]
A, C, M, R, T, F, B, S = (ord(c) for c in 'ACMRTFBA')
for date in dates:
    member = captures_by_date(months=(date[:7],))[date][0]['member_key']
    day = MonthDay(Path('data/derived/basket_multimonth_v1') / date[:7] / date)
    lo, hi = int(day.meta['london_ns']), int(day.meta['ny_close_ns'])
    for d, t, _ in day_tapes(member, [date]):
        orders = {}; bid = defaultdict(int); ask = defaultdict(int)
        stats = defaultdict(int); spreads = []; l1 = []; trade_vol = defaultdict(int)
        ts, act, side, price, size, oid = (t[k].tolist() for k in ('ts', 'action', 'side', 'price', 'size', 'order_id'))
        next_sample = lo
        for i in range(len(ts)):
            a = act[i]
            if a == R:
                orders.clear(); bid.clear(); ask.clear(); continue
            if lo <= ts[i] < hi and (a == T or ts[i] >= next_sample):
                bb = max((p for p, z in bid.items() if z > 0), default=None)
                ba = min((p for p, z in ask.items() if z > 0), default=None)
                if a == T and bb is not None and ba is not None and ba > bb:
                    p = price[i]; v = size[i]
                    where = ('inside' if bb < p < ba else 'at_bid' if p == bb else 'at_ask' if p == ba
                             else 'below_bid' if p < bb else 'above_ask')
                    stats[where] += 1; trade_vol[where] += v
                if ts[i] >= next_sample and bb is not None and ba is not None:
                    spreads.append((ba - bb) / 1e8); l1.append((bid[bb], ask[ba])); next_sample += 10**9
            if a == A:
                orders[oid[i]] = (side[i], price[i], size[i]); (bid if side[i] == B else ask)[price[i]] += size[i]
            elif a == C and oid[i] in orders:
                sd, p, z = orders[oid[i]]; r = min(size[i], z); (bid if sd == B else ask)[p] -= r
                if z - r == 0: orders.pop(oid[i])
                else: orders[oid[i]] = (sd, p, z - r)
            elif a == M and oid[i] in orders:
                sd, p, z = orders[oid[i]]; (bid if sd == B else ask)[p] -= z
                if size[i] > 0:
                    orders[oid[i]] = (side[i], price[i], size[i]); (bid if side[i] == B else ask)[price[i]] += size[i]
                else:
                    orders.pop(oid[i])
            elif a == M and size[i] > 0:
                orders[oid[i]] = (side[i], price[i], size[i]); (bid if side[i] == B else ask)[price[i]] += size[i]
        n = sum(stats.values()); vol = sum(trade_vol.values())
        sp = np.array(spreads); l1a = np.array(l1)
        q = QuoteTape.from_day(day); ok = np.asarray(q.valid, bool); qt = q.quote_t
        sel = ok & (qt >= lo) & (qt < hi)
        print(json.dumps(dict(date=date, symbol=day.meta['symbol'], trades=n, volume=vol,
                              trade_location={k: round(v / n, 4) for k, v in stats.items()},
                              volume_location={k: round(v / vol, 4) for k, v in trade_vol.items()},
                              raw_spread_ticks_p50=float(np.median(sp)), raw_spread_p90=float(np.percentile(sp, 90)),
                              raw_spread_1tick_share=float((sp <= 1.0001).mean()),
                              l1_bid_p50=float(np.median(l1a[:, 0])), l1_ask_p50=float(np.median(l1a[:, 1])),
                              quotes_spread_p50=float(np.median((q.ask[sel] - q.bid[sel]) / 1e8)))), flush=True)
