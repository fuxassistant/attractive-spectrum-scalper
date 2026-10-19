"""Parity: scalper_live.trade with the taker take must equal the precomputed scalper labels.

Usage: python scripts/cognition_audit/scalper_live_parity.py 2025-10-02
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spectrum.cognition import scalper as sc, scalper_live as sl, scalper_policy as sp
from spectrum.cognition.passive import Quotes, day_tapes, captures_by_date
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay

d = sys.argv[1]
f = pd.read_parquet(f'data/derived/scalper_v1/{d}.parquet')
q = Quotes(QuoteTape.from_day(MonthDay(Path('data/derived/basket_multimonth_v1') / d[:7] / d)))
for _, tape, _ in day_tapes(captures_by_date(months=(d[:7],))[d][0]['member_key'], [d]):
    pass
plans = {p[0]: p for p in sp.ADAPTIVE}; bad = n = 0
for i in np.random.default_rng(1).choice(len(f), 40, replace=False):
    for mode in sc.MODES:
        for s in (1, -1):
            for name in plans:
                key = f'{mode}_{sc.SIDES[s]}_{name}'; lab_st = int(f[key + '_st'].iat[i])
                if lab_st not in (sc.COMPLETE, sc.EMPTY):
                    continue
                st, net, _ = sl.trade(q, tape, int(f.t_ns.iat[i]), mode, s, plans[name], float(f.sigma600.iat[i]), False)
                n += 1; bad += (st != lab_st) or abs(net - f[key + '_net'].iat[i]) > 1e-5
print(d, 'taker-take parity mismatches', bad, 'of', n)
