"""Exact native first-passage bid targets, shared by both opposite actions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit

NS=1_000_000_000
STATES=['NONE','UP_CLEAN','UP_DIRTY','DOWN_CLEAN','DOWN_DIRTY']


@njit(cache=True)
def scan(qt,bid,good,epoch,origins,thresholds,horizon,coverage,gap,clean_fraction):
    # status (1 known,0 censored), class, origin/exit quote idx, exit decision,
    # event direction, reason (0 known,1 origin,2 discontinuity,3 coverage).
    ints=np.full((len(origins),7),-1,np.int64)
    vals=np.full((len(origins),4),np.nan)
    for k in range(len(origins)):
        t=origins[k];due=t+horizon;threshold=thresholds[k]
        a=np.searchsorted(qt,t,'right')-1
        ints[k,0]=0;ints[k,6]=1;ints[k,2]=a
        if a<0 or not good[a] or t-qt[a]>gap:continue
        start=bid[a];low=0;high=0;last=a;sign=0;exit_time=due;valid=True
        for j in range(a+1,len(qt)):
            if qt[j]>due:break
            if not good[j] or epoch[j]!=epoch[a] or qt[j]-qt[j-1]>gap:
                valid=False;ints[k,6]=2;break
            delta=bid[j]-start;low=min(low,delta);high=max(high,delta);last=j
            if delta>=threshold:sign=1;exit_time=qt[j];break
            if delta<=-threshold:sign=-1;exit_time=qt[j];break
        if not valid:continue
        if sign==0 and (due>=coverage or due-qt[last]>gap):
            ints[k,6]=3;continue
        state=0
        if sign==1:state=1 if -low<=clean_fraction*threshold+1e-9 else 2
        if sign==-1:state=3 if high<=clean_fraction*threshold+1e-9 else 4
        ints[k,0]=1;ints[k,1]=state;ints[k,3]=last;ints[k,4]=exit_time;ints[k,5]=sign;ints[k,6]=0
        vals[k,0]=(bid[last]-start)/NS;vals[k,1]=low/NS;vals[k,2]=high/NS;vals[k,3]=(exit_time-t)/NS
    return ints,vals


def make_targets(quotes,features,spec,*,coverage,config):
    qt=quotes.available_ns.to_numpy(np.int64);bid=quotes.bid_nano.to_numpy(np.int64)
    ask=quotes.ask_nano.to_numpy(np.int64);epoch=quotes.epoch.to_numpy(np.int64)
    good=quotes.valid.to_numpy(bool)&(epoch>=0)&(bid>0)&(ask>bid)
    if not len(qt) or np.any(np.diff(qt)<0):raise ValueError('Ordered native publication tape required')
    x=np.full(len(features),spec['usd_per_oz'])
    if spec['threshold']=='max_fixed_risk':x=np.maximum(x,features.risk_per_oz.to_numpy(float))
    # Round threshold UP to a whole exchange tick, using only origin X.
    ticks=np.ceil(x/.1-1e-10).astype(np.int64);nano=ticks*100_000_000
    z,v=scan(qt,bid,good,epoch,features.t_ns.to_numpy(np.int64),nano,int(spec['seconds']*NS),int(coverage),
             int(config['post_cost']['quote_gap_seconds']*NS),config['clean_loss_max_favorable_x'])
    f=features[['candidate_id','t_ns','date','side']].rename(columns={'side':'zone_side'}).copy().reset_index(drop=True)
    f['target']=spec['id'];f['threshold_per_oz']=nano/NS;f['known']=z[:,0]==1;f['class_index']=z[:,1]
    f['origin_ordinal']=z[:,2];f['exit_ordinal']=z[:,3];f['exit_ns']=z[:,4];f['event_sign']=z[:,5];f['censor_reason']=z[:,6]
    f['horizon_ns']=f.t_ns+spec['seconds']*NS;f['release_ns']=f.horizon_ns
    f['bid_delta_per_oz']=v[:,0];f['bid_low_per_oz']=v[:,1];f['bid_high_per_oz']=v[:,2];f['seconds']=v[:,3]
    f['state']=[STATES[i] if k else 'UNKNOWN' for i,k in zip(z[:,1],f.known)]
    return f


def post_cost_pairs(quotes,targets,config,*,coverage):
    """Fixed bid-derived decisions; costs never alter targets or label rewards."""
    qt=quotes.available_ns.to_numpy(np.int64);bid=quotes.bid_nano.to_numpy(np.int64);ask=quotes.ask_nano.to_numpy(np.int64)
    epoch=quotes.epoch.to_numpy(np.int64);good=quotes.valid.to_numpy(bool)&(epoch>=0)&(bid>0)&(ask>bid)
    gap=int(config['post_cost']['quote_gap_seconds']*NS);pending=int(config['post_cost']['pending_seconds']*NS)
    bad=np.r_[0,np.cumsum(~good)];broken=np.r_[0,np.cumsum(np.r_[False,(np.diff(qt)>gap)|(np.diff(epoch)!=0)])]
    rows=[]
    for r in targets.itertuples():
        for mode in ('base','stress'):
            ec=config['post_cost'][mode];lat=int(ec['latency_ms']*1e6);slip=ec['slippage_ticks_per_fill']*.1
            a=int(r.origin_ordinal);e=int(np.searchsorted(qt,r.t_ns+lat,'right'))
            base=dict(candidate_id=r.candidate_id,target=r.target,mode=mode,date=r.date,known=False,entry_ns=-1,exit_ns=-1)
            if r.known and a>=0 and e<len(qt) and qt[e]<=r.t_ns+pending:
                request=max(int(r.exit_ns),int(qt[e]));ex=int(np.searchsorted(qt,request+lat,'right'))
                supported=ex<len(qt) and qt[ex]<=request+pending and qt[ex]<coverage
                if supported:
                    supported=(bad[ex+1]==bad[a] and broken[ex+1]==broken[a+1] and epoch[a]==epoch[ex])
                if supported:
                    base.update(known=True,entry_ns=int(qt[e]),exit_ns=int(qt[ex]))
            for side in (1,-1):
                row={**base,'side':side,'net_per_oz':np.nan}
                if base['known']:
                    entry=ask[e] if side==1 else bid[e];exitp=bid[ex] if side==1 else ask[ex]
                    row['net_per_oz']=side*(int(exitp)-int(entry))/NS-ec['fee_per_oz']-2*slip
                rows.append(row)
    return pd.DataFrame(rows)
