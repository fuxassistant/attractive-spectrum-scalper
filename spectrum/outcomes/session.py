"""Reusable formulas extracted from scripts/bb_crossing_session.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from numba import njit
from spectrum.crossings import bank as bb

@njit(cache=True)
def all_marks(bid,ask,k,blocks,entries,crosses,known,breaks):
    values=np.full((len(entries),6),np.nan)
    for i in range(len(entries)):
        if not known[i]:continue
        e=entries[i];x=crosses[i]
        if x>=breaks[e]:raise ValueError('Crossing after lost quote support')
        ext=bb.range_extrema(bid,ask,k,blocks,e,x)
        values[i,0]=(bid[x]-ask[e])/1e9-.3;values[i,1]=(bid[e]-ask[x])/1e9-.3
        values[i,2]=(ext[0]-ask[e])/1e9-.3;values[i,3]=(bid[e]+ext[3])/1e9-.3
        values[i,4]=(-ext[1]-ask[e])/1e9-.3;values[i,5]=(bid[e]-ext[2])/1e9-.3
    return values


def lookup(p,t,limit,ready,key):
    ix=np.searchsorted(p.t_ns.to_numpy(),t,side='right');safe=np.minimum(ix,max(0,len(p)-1))
    if not len(p):raise ValueError('Empty scale bank')
    event_t=p.t_ns.to_numpy()[safe];known=ready&(ix<len(p))&(event_t<limit)
    return pd.DataFrame({key+'_known':known,key+'_y':np.where(known,p.label.to_numpy()[safe],-1),
        key+'_id':np.where(known,p.packet_id.to_numpy()[safe],''),key+'_event_ns':np.where(known,event_t,0),
        key+'_ordinal':np.where(known,p.ordinal.to_numpy()[safe],-1),key+'_mask':np.where(known,p.line_mask.to_numpy()[safe],0),
        key+'_delay_s':np.where(known,(event_t-t)/bb.NS,np.nan)})
