"""First crossing packet of sampled 1 s bands (bid OR ask rule); extracted from scripts/vwap_week_report.py."""
import numpy as np
from spectrum.core.schema import NS
from spectrum.crossings.bank import crossing_rows
from spectrum.leakage import future_output

@future_output()
def future_first_crossing(f,name):
    t=f.t_ns.to_numpy();m=f[name].to_numpy();h=f[name+'_half'].to_numpy()
    support=np.where(np.isfinite(m)&np.isfinite(h)&(h>0),f.segment,-1).astype(np.int64)
    row,role,_=crossing_rows(t,f.bid.to_numpy(float),f.ask.to_numpy(float),m,h,support)
    event=np.unique(row);bits=np.zeros(len(f),np.int64)
    np.bitwise_or.at(bits,row,1<<role)
    end=np.r_[np.flatnonzero(support[1:]!=support[:-1])+1,len(f)]
    ends=np.repeat(end,np.diff(np.r_[0,end]))
    origins=np.arange(0,len(f),30);chosen=np.searchsorted(event,origins,side='right')
    has=chosen<len(event);eventrow=event[np.minimum(chosen,max(0,len(event)-1))] if len(event) else np.full(len(origins),len(f))
    resolved=has&(eventrow<ends[origins])&(eventrow-origins<=1800)&(support[origins]>=0)
    known_none=(origins+1800<ends[origins])&(support[origins]>=0)
    y=np.where(known_none,8,-1);y[resolved]=bits[eventrow[resolved]]
    delay=np.where(resolved,eventrow-origins,np.nan)
    return y,delay,dict(line_crossings=len(row),packets=len(event),supported_seconds=int((support>=0).sum()))
