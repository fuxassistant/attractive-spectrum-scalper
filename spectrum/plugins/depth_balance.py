"""Example plugin using the existing tape, without another MBO owner."""
import numpy as np
from spectrum.lab import FeaturePlugin

def calculate(context):
    f=context.df
    bid=f.depth_bid_12.to_numpy();ask=f.depth_ask_12.to_numpy();total=bid+ask
    value=np.divide(bid-ask,total,out=np.zeros(len(f)),where=total>0)
    context.add('depth_balance_12',value,context.supported&(total>0),'ratio')

PLUGIN=FeaturePlugin('depth_balance','1',calculate,0)
