"""Reusable formulas extracted from spectrum/lab.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.features.contract import rolling_valid

def flow(c):
    for col in ('trade_buy','trade_sell','trade_unknown','add_bid','add_ask','remove_bid','remove_ask','matched_fill_remove_bid','matched_fill_remove_ask','event_count'):
        d=c.delta('c_'+col)
        for n in (1,5,30):
            v=pd.Series(d).rolling(n,min_periods=n).sum().to_numpy()
            c.add(f'{col}_{n}s',v,rolling_valid(c.edge,n)&(v>=0),'count' if col=='event_count' else 'contract')
