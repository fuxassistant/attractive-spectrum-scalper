"""Explicit FIFO (deques of [id, size, pending] per level) vs kernel v5 on a real day."""
import sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import day_tapes, captures_by_date, Quotes, LATENCY
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition import queue as qu
from spectrum.playground.data import MonthDay
from spectrum.player.contract import NS

date, n_orig, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
day = MonthDay(Path('data/derived/basket_multimonth_v1') / date[:7] / date); q = Quotes(QuoteTape.from_day(day))
x = pd.read_parquet(f'data/derived/opportunity_signatures_v1/{date}/x.parquet')
member = captures_by_date(months=(date[:7],))[date][0]['member_key']
for _, tape, _ in day_tapes(member, [date]):
    pass
probes = []
for r in x.sample(n_orig, random_state=seed).itertuples():
    d = q.asof(int(r.intent_ns))
    if d >= 0:
        for side in (1, -1):
            probes.append((int(r.intent_ns) + LATENCY, side, int(q.bid[d] if side > 0 else q.ask[d]), int(r.intent_ns) + LATENCY + 30 * NS))
st, fill, kind, _ = qu.probe(tape, *zip(*probes))
A, C, M, R, T, F, B, S = (ord(c) for c in 'ACMRTFBA')
ts, act, sd, pr, sz, oid = (tape[k].tolist() for k in ('ts', 'action', 'side', 'price', 'size', 'order_id'))
ME = -1


def brute(place, side, price, end):
    levels = defaultdict(deque); where = {}; placed = False
    own = B if side > 0 else S
    def remaining_ahead():
        lv = levels[(own, price)]; total = 0
        for e in lv:
            if e[0] == ME:
                return total
            total += max(e[1] - e[2], 0)
        return total
    for i in range(len(ts)):
        if not placed and ts[i] >= place:
            levels[(own, price)].append([ME, 1, 0]); placed = True
        a = act[i]
        if placed and ts[i] > end:
            return 'EXPIRED', 0
        if a == R:
            if placed:
                return 'CENSORED', ts[i]
            levels.clear(); where.clear(); continue
        if a == T:
            if placed and ((side > 0 and pr[i] < price) or (side < 0 and pr[i] > price)):
                return 'FILLED', ts[i]
            continue
        o = oid[i]
        if a == F:
            if placed and sd[i] == own and ((side > 0 and pr[i] < price) or (side < 0 and pr[i] > price)):
                return 'FILLED', ts[i]  # executed at a price worse than ours
            if o not in where or where[o] != (sd[i], pr[i]):
                continue  # unknown order or stale book level: no queue evidence
            lvl = where[o]; e = next(e for e in levels[lvl] if e[0] == o)
            disp = min(sz[i], max(e[1] - e[2], 0)); excess = sz[i] - disp
            if placed and lvl[0] == own:
                bp = lvl[1]
                if bp == price:
                    ids = [z[0] for z in levels[lvl]]
                    behind = ids.index(o) > ids.index(ME)
                    e[2] += disp
                    if (behind or excess > 0) and remaining_ahead() <= 0:
                        return 'FILLED', ts[i]
                    continue
            e[2] += disp
            continue
        if a == A:
            levels[(sd[i], pr[i])].append([o, sz[i], 0]); where[o] = (sd[i], pr[i])
        elif a == C and o in where:
            lv = levels[where[o]]; e = next(e for e in lv if e[0] == o)
            r = min(sz[i], e[1]); e[1] -= r; e[2] -= min(r, e[2])
            if e[1] == 0:
                lv.remove(e); where.pop(o)
        elif a == M:
            if o in where:
                lv = levels[where[o]]; e = next(e for e in lv if e[0] == o)
                if sz[i] <= 0:
                    lv.remove(e); where.pop(o); continue
                if pr[i] == where[o][1] and sz[i] <= e[1] - e[2]:
                    e[1] = sz[i]; e[2] = 0
                else:
                    lv.remove(e); levels[(sd[i], pr[i])].append([o, sz[i], 0]); where[o] = (sd[i], pr[i])
            elif sz[i] > 0:
                levels[(sd[i], pr[i])].append([o, sz[i], 0]); where[o] = (sd[i], pr[i])
    return 'CENSORED', 0

agree = 0; diffs = []; names = {1: 'FILLED', 2: 'EXPIRED', 3: 'CENSORED', 4: 'UNPLACED'}
for k, p in enumerate(probes):
    s2, f2 = brute(*p); s1 = names[int(st[k])]
    ok = s1 == s2 and (s1 != 'FILLED' or int(fill[k]) == f2)
    agree += ok
    if not ok:
        diffs.append((k, s1, int(fill[k]), qu.KIND[int(kind[k])], s2, f2))
print(date, 'kernel v5 vs explicit FIFO agree', agree, 'of', len(probes), diffs[:6], flush=True)
