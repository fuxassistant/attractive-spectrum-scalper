"""Reusable formulas extracted from spectrum/lab.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.features.contract import rolling_valid

def motion(c):
    f=c.df;mid=f.mid.to_numpy();bid=f.bid.to_numpy();ask=f.ask.to_numpy()
    c.add('spread_ticks',(ask-bid)/.1,c.admitted,'tick')
    for n in (1,5,15,30,60,128):
        good=rolling_valid(c.edge,n)
        previous=np.r_[np.full(n,np.nan),mid[:-n]] if len(mid)>n else np.full(len(mid),np.nan)
        change=(mid-previous)/.1
        c.add(f'displacement_{n}s',change,good,'tick')
        delta=c.delta('c_path_ticks')
        path=pd.Series(delta).rolling(n,min_periods=n).sum().to_numpy()
        c.add(f'path_{n}s',path,good,'tick')
        c.add(f'efficiency_{n}s',np.divide(abs(change),path,out=np.zeros(len(mid)),where=path>0),good&(path>0),'ratio')
    denom=(f.bid_size+f.ask_size).to_numpy()
    c.add('imbalance_l1',np.divide((f.bid_size-f.ask_size).to_numpy(),denom,out=np.zeros(len(f)),where=denom>0),c.admitted&(denom>0),'ratio')
    for side in ('bid','ask'):
        for n in (1,4,12):c.add(f'depth_{side}_{n}',f[f'depth_{side}_{n}'],c.admitted,'contract')
