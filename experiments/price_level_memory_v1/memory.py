"""Absolute-price memory from native quote publications, with explicit support.

Midpoint contacts are not executions. A jump across a bin is not a touch.
Rejection/passage describes a hysteresis-band exit observed later than contact;
it is never backdated to that contact and is not a profitable-trade label.
"""
import numpy as np
import pandas as pd
from numba import njit

NS=1_000_000_000
TICK=100_000_000
FIELDS=('mid_dwell_s','bid_dwell_s','ask_dwell_s','arrivals','present_at_restart',
        'pass_up','pass_down','reject_up','reject_down','jump_over','censored',
        'active','observed_age_s','age_interval_known')

@njit(cache=True)
def _snapshots(qt,bid,ask,good,epoch,queries,base,bins,width,margin,gap):
    values=np.zeros((len(queries),bins,len(FIELDS)),np.float64)
    stats=np.zeros((len(queries),2),np.float64)
    dwell=np.zeros((bins,3),np.float64)
    counts=np.zeros((bins,8),np.int64)
    active=np.zeros(bins,np.int8);incoming=np.zeros(bins,np.int8)
    last=np.full(bins,-1,np.int64)
    j=-1;cursor=qt[0];known_s=0.;last_unknown=qt[0];had_support=False
    for qi in range(len(queries)):
        end=queries[qi]
        while j+1<len(qt) and qt[j+1]<=end:
            k=j+1;t=qt[k]
            if j>=0 and good[j]:
                until=min(t,qt[j]+gap)
                dt=max(0,until-cursor)/NS
                if dt>0:
                    mid=(bid[j]+ask[j])//2
                    dwell[mid//width-base,0]+=dt
                    dwell[bid[j]//width-base,1]+=dt
                    dwell[ask[j]//width-base,2]+=dt
                    known_s+=dt
                    last[mid//width-base]=until
                if t>qt[j]+gap:
                    for b in range(bins):
                        if active[b]:counts[b,7]+=1;active[b]=0
                    last_unknown=t;had_support=False
            elif j>=0:
                last_unknown=t;had_support=False
            continuous=j>=0 and good[j] and good[k] and epoch[j]==epoch[k] and t-qt[j]<=gap and had_support
            if not good[k] or (j>=0 and epoch[j]!=epoch[k]):
                for b in range(bins):
                    if active[b]:counts[b,7]+=1;active[b]=0
                last_unknown=t;had_support=False
                continuous=False
            if good[k]:
                mid=(bid[k]+ask[k])//2;b=mid//width-base
                if continuous:
                    previous=(bid[j]+ask[j])//2;old=previous//width-base
                    if old!=b:
                        for crossed in range(min(old,b)+1,max(old,b)):
                            counts[crossed,6]+=1
                    # At most neighbouring bins remain active with margin<width.
                    for a in range(max(0,old-1),min(bins,old+2)):
                        if not active[a]:continue
                        lo=(base+a)*width;hi=lo+width
                        direction=1 if mid>=hi+margin else -1 if mid<lo-margin else 0
                        if direction:
                            if incoming[a]==1:
                                counts[a,2 if direction==1 else 5]+=1
                            elif incoming[a]==-1:
                                counts[a,3 if direction==-1 else 4]+=1
                            active[a]=0
                if not active[b]:
                    active[b]=1
                    if continuous:
                        counts[b,0]+=1
                        incoming[b]=1 if previous<(base+b)*width else -1 if previous>=(base+b+1)*width else 0
                    else:
                        counts[b,1]+=1;incoming[b]=0
                last[b]=t;had_support=True
            j=k;cursor=t
        if j>=0 and good[j]:
            until=min(end,qt[j]+gap)
            dt=max(0,until-cursor)/NS
            if dt>0:
                mid=(bid[j]+ask[j])//2
                dwell[mid//width-base,0]+=dt
                dwell[bid[j]//width-base,1]+=dt
                dwell[ask[j]//width-base,2]+=dt
                known_s+=dt
                # Supported persistent quotes mean the midpoint remains there.
                last[mid//width-base]=until
            if end>qt[j]+gap:
                for b in range(bins):
                    if active[b]:counts[b,7]+=1;active[b]=0
                last_unknown=end;had_support=False
        elif end>cursor:
            last_unknown=end;had_support=False
        cursor=end
        for b in range(bins):
            values[qi,b,0:3]=dwell[b]
            values[qi,b,3:11]=counts[b]
            values[qi,b,11]=active[b]
            values[qi,b,12]=(end-last[b])/NS if last[b]>=0 else np.nan
            values[qi,b,13]=1 if last[b]>=last_unknown else 0
        stats[qi,0]=known_s;stats[qi,1]=max(0,end-qt[0])/NS
    return values,stats

def build_memory(quotes,query_ns,bin_ticks=5,hysteresis_ticks=2,max_gap_seconds=30,radius_ticks=300):
    """Return a causal absolute-price viewport per asof, for one route.

    The storage extent may cover the supplied tape, but outputs at any asof are
    a prefix function, including its extent: the viewport is centered on the
    last valid observed midpoint, never the eventual day's range. Memory outside
    the viewport is retained. Empty cells mean no OBSERVED contacts, not proof
    of absence across gaps or before the start of this tape.
    Prices/boundaries remain integer nano coordinates; clocks are int64.
    """
    required={'available_ns','bid_nano','ask_nano','valid','epoch'}
    if not required<=set(quotes):raise ValueError('Native quote columns required')
    if any(quotes[name].dtype.kind not in 'iu' for name in ('available_ns','bid_nano','ask_nano','epoch')):
        raise ValueError('Integer nanosecond clocks, nano prices and epochs required')
    if 't_ns' in quotes and (quotes.t_ns.dtype.kind not in 'iu' or (quotes.available_ns<quotes.t_ns).any()):
        raise ValueError('Publication cannot precede integer source clock')
    query=np.asarray(query_ns)
    if query.dtype.kind not in 'iu' or query.ndim!=1 or not len(query) or np.any(np.diff(query)<0):raise ValueError('Nonempty sorted integer query clocks required')
    if bin_ticks<=0 or int(bin_ticks)!=bin_ticks or not 0<=hysteresis_ticks<bin_ticks or max_gap_seconds<=0:raise ValueError('Invalid map geometry/support')
    if radius_ticks<bin_ticks or int(radius_ticks)!=radius_ticks:raise ValueError('Invalid viewport radius')
    q=quotes.loc[quotes.available_ns<=int(query[-1])].copy()
    qt=q.available_ns.to_numpy(np.int64);bid=q.bid_nano.to_numpy(np.int64);ask=q.ask_nano.to_numpy(np.int64);ep=q.epoch.to_numpy(np.int64)
    if not len(qt) or np.any(np.diff(qt)<0):raise ValueError('Nonempty publication-order quote tape required')
    good=q.valid.to_numpy(bool)&(bid>0)&(ask>bid)&(ep>=0)
    if not good.any():raise ValueError('No supported quotes')
    width=int(bin_ticks*TICK)
    base=int(bid[good].min()//width)-1;lastbin=int(ask[good].max()//width)+1
    n=lastbin-base+1
    values,stats=_snapshots(qt,bid,ask,good,ep,query.astype(np.int64),base,n,width,int(hysteresis_ticks*TICK),int(max_gap_seconds*NS))
    frames=[]
    supported_indices=np.flatnonzero(good)
    radius=int(np.ceil(radius_ticks/bin_ticks))
    for i,t in enumerate(query):
        f=pd.DataFrame(values[i],columns=FIELDS)
        f['price_bin_nano']=(np.arange(n,dtype=np.int64)+base)*width
        anchor_index=np.searchsorted(qt[supported_indices],t,'right')-1
        if anchor_index<0:raise ValueError('Query precedes first supported quote')
        anchor_index=supported_indices[anchor_index]
        anchor_mid=(bid[anchor_index]+ask[anchor_index])//2
        viewport=(np.arange(-radius,radius+1,dtype=np.int64)+anchor_mid//width)*width
        f=f.set_index('price_bin_nano').reindex(viewport).rename_axis('price_bin_nano').reset_index()
        for field in FIELDS:
            if field!='observed_age_s':f[field]=f[field].fillna(0)
        current_index=np.searchsorted(qt,t,'right')-1
        f['anchor_mid']=anchor_mid/1e9
        f['current_quote_known']=bool(good[current_index] and t-qt[current_index]<=int(max_gap_seconds*NS))
        f['price_low']=f.price_bin_nano/1e9;f['price_high']=(f.price_bin_nano+width)/1e9
        f['query_ns']=int(t);f['history_observed_seconds']=stats[i,0]
        f['history_elapsed_seconds']=stats[i,1]
        f['history_missing_seconds']=max(0.,stats[i,1]-stats[i,0])
        f['history_coverage']=stats[i,0]/stats[i,1] if stats[i,1]>0 else 0.
        for name in FIELDS[3:12]:f[name]=f[name].astype('int64')
        f['age_interval_known']=f.age_interval_known.astype(bool)
        f['no_contact_in_observed_history']=f.observed_age_s.isna()
        frames.append(f)
    return pd.concat(frames,ignore_index=True)
