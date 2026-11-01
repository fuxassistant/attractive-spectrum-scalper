"""Native-quote episode ledger at immutable absolute price addresses.

Completed episodes are outcomes known at end_ns, never attributes of an earlier
arrival. Event rows are immutable, with their actual availability. A short edge
excursion that returns before departure confirmation is not a failed breakout.
"""
import numpy as np
import pandas as pd
from numba import njit

NS=1_000_000_000
TICK=100_000_000
EVENTS={1:'ARRIVAL',2:'PRESENT_ON_RESTART',3:'EDGE_EXIT',4:'EDGE_RETURN',
        5:'GATE_START',6:'GATE_ABORT',7:'DEPARTURE_CONFIRMED',8:'CENSORED',9:'JUMP_OVER'}
OUTCOMES={1:'PASS_UP',2:'PASS_DOWN',3:'REJECT_UP',4:'REJECT_DOWN',5:'UNKNOWN_ENTRY',6:'CENSORED'}
ECOLS=['available_ns','event_code','zone_low_nano','episode_id','side','mid_nano','duration_ns','excursion_nano','source_ordinal']
PCOLS=['episode_id','zone_low_nano','epoch','arrival_ns','end_ns','arrival_ordinal','end_ordinal',
       'entry_side','exit_side','outcome_code','core_dwell_ns','arrival_mid_nano','end_mid_nano',
       'path_nano','previous_departure_ns','previous_departure_mid_nano','approach_path_nano','arrival_index','previous_departure_index']


@njit(cache=True)
def _ledger(qt,bid,ask,valid,epoch,ordinal,base,n,width,pad,confirm,gap,end):
    ev=[]
    ep=[]
    active=np.zeros(n,np.bool_);ids=np.zeros(n,np.int64);beg=np.zeros(n,np.int64)
    starti=np.zeros(n,np.int64);sides=np.zeros(n,np.int64);dwell=np.zeros(n,np.int64)
    startp=np.zeros(n,np.int64);startpath=np.zeros(n,np.int64);entryord=np.zeros(n,np.int64)
    in_core=np.zeros(n,np.bool_);edge=np.full(n,-1,np.int64);edge_peak=np.zeros(n,np.int64)
    pending=np.full(n,-1,np.int64);pending_side=np.zeros(n,np.int64)
    lastdep=np.full(n,-1,np.int64);lastdepp=np.zeros(n,np.int64);lastdeppath=np.zeros(n,np.int64)
    lastdepi=np.full(n,-1,np.int64);lastdepepoch=np.full(n,-1,np.int64)
    prevdep=np.full(n,-1,np.int64);prevdepp=np.zeros(n,np.int64);approach=np.zeros(n,np.int64);prevdepi=np.full(n,-1,np.int64)
    alist=np.zeros(n,np.int64);ac=0;nextid=0;globalpath=np.int64(0)
    previous_good=False;previous_mid=np.int64(0);previous_t=qt[0];previous_epoch=np.int64(-1)
    for k in range(len(qt)+1):
        terminal=k==len(qt)
        t=end if terminal else qt[k]
        good=False if terminal else valid[k] and bid[k]>0 and ask[k]>bid[k] and epoch[k]>=0
        until=min(t,previous_t+gap)
        continuous=k>0 and previous_good and good and epoch[k]==previous_epoch and t-previous_t<=gap if not terminal else False
        i=0
        while i<ac:
            b=alist[i];lo=(base+b)*width
            due=pending[b]+confirm if pending[b]>=0 else np.int64(9223372036854775807)
            # A known clock deadline precedes the publication at the same time.
            if previous_good and due<=until:
                if in_core[b]:dwell[b]+=max(0,due-previous_t)
                direction=pending_side[b]
                outcome=(1 if direction==1 else 2) if sides[b]==direction else ((3 if direction==1 else 4) if sides[b]!=0 else 5)
                ep.append((ids[b],lo,previous_epoch,beg[b],due,entryord[b],ordinal[k-1],sides[b],direction,outcome,dwell[b],startp[b],previous_mid,globalpath-startpath[b],prevdep[b],prevdepp[b],approach[b],starti[b],prevdepi[b]))
                ev.append((due,7,lo,ids[b],direction,previous_mid,due-beg[b],edge_peak[b],ordinal[k-1]))
                lastdep[b]=due;lastdepp[b]=previous_mid;lastdeppath[b]=globalpath;lastdepi[b]=k-1;lastdepepoch[b]=previous_epoch
                active[b]=False;pending[b]=-1;ac-=1;alist[i]=alist[ac]
                continue
            if previous_good and in_core[b]:dwell[b]+=max(0,until-previous_t)
            if terminal or not continuous:
                censor_at=until if previous_good and t-previous_t>gap else t
                ep.append((ids[b],lo,previous_epoch,beg[b],censor_at,entryord[b],ordinal[max(0,k-1)],sides[b],0,6,dwell[b],startp[b],previous_mid,globalpath-startpath[b],prevdep[b],prevdepp[b],approach[b],starti[b],prevdepi[b]))
                ev.append((censor_at,8,lo,ids[b],0,previous_mid,censor_at-beg[b],edge_peak[b],ordinal[max(0,k-1)]))
                active[b]=False;pending[b]=-1;ac-=1;alist[i]=alist[ac]
                continue
            i+=1
        if terminal:break
        if good:
            mid=(bid[k]+ask[k])//2;current=mid//width-base
            if continuous:
                globalpath+=abs(mid-previous_mid)
                old=previous_mid//width-base
                for crossed in range(min(old,current)+1,max(old,current)):
                    ev.append((t,9,(base+crossed)*width,-1,1 if mid>previous_mid else -1,mid,0,0,ordinal[k]))
            else:
                # Prior departures must not establish returns across unknown time.
                lastdepepoch[:]=-1
            for i in range(ac):
                b=alist[i];lo=(base+b)*width;hi=lo+width;inside=lo<=mid<hi
                side=1 if mid>=hi else -1 if mid<lo else 0
                distance=mid-hi if mid>=hi else lo-mid if mid<lo else 0
                if in_core[b] and not inside:
                    edge[b]=t;edge_peak[b]=distance
                    ev.append((t,3,lo,ids[b],side,mid,0,distance,ordinal[k]))
                elif not in_core[b] and inside:
                    ev.append((t,4,lo,ids[b],-1 if previous_mid<lo else 1,mid,t-edge[b],edge_peak[b],ordinal[k]))
                    edge[b]=-1;edge_peak[b]=0
                elif not inside:edge_peak[b]=max(edge_peak[b],distance)
                in_core[b]=inside
                far=(mid>=hi+pad) or (mid<lo-pad)
                if far:
                    if pending[b]<0 or pending_side[b]!=side:
                        if pending[b]>=0:ev.append((t,6,lo,ids[b],pending_side[b],mid,t-pending[b],edge_peak[b],ordinal[k]))
                        pending[b]=t;pending_side[b]=side
                        ev.append((t,5,lo,ids[b],side,mid,0,distance,ordinal[k]))
                elif pending[b]>=0:
                    ev.append((t,6,lo,ids[b],pending_side[b],mid,t-pending[b],edge_peak[b],ordinal[k]))
                    pending[b]=-1
            b=current;lo=(base+b)*width
            if not active[b]:
                nextid+=1;active[b]=True;ids[b]=nextid;beg[b]=t;starti[b]=k
                startp[b]=mid;startpath[b]=globalpath;entryord[b]=ordinal[k]
                sides[b]=(1 if previous_mid<lo else -1 if previous_mid>=lo+width else 0) if continuous else 0
                dwell[b]=0;in_core[b]=True;edge[b]=-1;edge_peak[b]=0;pending[b]=-1
                alist[ac]=b;ac+=1
                has_previous=continuous and lastdep[b]>=0 and lastdepepoch[b]==epoch[k]
                prevdep[b]=lastdep[b] if has_previous else -1
                prevdepp[b]=lastdepp[b] if has_previous else 0
                prevdepi[b]=lastdepi[b] if has_previous else -1
                approach[b]=globalpath-lastdeppath[b] if has_previous else 0
                ev.append((t,1 if continuous else 2,lo,ids[b],sides[b],mid,t-lastdep[b] if has_previous else -1,approach[b],ordinal[k]))
            previous_mid=mid
        previous_good=good;previous_t=t;previous_epoch=epoch[k]
    return ev,ep


def build_episodes(quotes,*,end_ns=None,bin_ticks=10,departure_ticks=10,confirmation_seconds=3,max_quote_age_seconds=30):
    required=['available_ns','bid_nano','ask_nano','epoch','source_ordinal']
    if any(c not in quotes or quotes[c].dtype.kind not in 'iu' for c in required):raise ValueError('Integer native clocks/prices/epochs/ordinals required')
    if quotes.empty or np.any(np.diff(quotes.available_ns.to_numpy())<0):raise ValueError('Nonempty sorted quote tape required')
    if bin_ticks<=0 or int(bin_ticks)!=bin_ticks or departure_ticks<=0 or confirmation_seconds<=0 or max_quote_age_seconds<confirmation_seconds:raise ValueError('Invalid geometry/timing')
    if end_ns is not None and (not isinstance(end_ns,(int,np.integer)) or isinstance(end_ns,(bool,np.bool_))):raise ValueError('Integer end_ns required')
    end=int(quotes.available_ns.iloc[-1]) if end_ns is None else int(end_ns)
    q=quotes.loc[quotes.available_ns<=end]
    qt=q.available_ns.to_numpy(np.int64);bid=q.bid_nano.to_numpy(np.int64);ask=q.ask_nano.to_numpy(np.int64)
    valid=q.valid.to_numpy(bool);epoch=q.epoch.to_numpy(np.int64);ordinal=q.source_ordinal.to_numpy(np.int64)
    good=valid&(bid>0)&(ask>bid)&(epoch>=0)
    if not good.any():raise ValueError('No supported quotes')
    width=int(bin_ticks*TICK);mid=(bid+ask)//2;base=int(mid[good].min()//width);n=int(mid[good].max()//width)-base+1
    events,episodes=_ledger(qt,bid,ask,valid,epoch,ordinal,base,n,width,int(departure_ticks*TICK),int(confirmation_seconds*NS),int(max_quote_age_seconds*NS),end)
    events=pd.DataFrame(events,columns=ECOLS,dtype='int64');episodes=pd.DataFrame(episodes,columns=PCOLS,dtype='int64')
    events['event_type']=events.event_code.map(EVENTS)
    episodes['outcome']=episodes.outcome_code.map(OUTCOMES)
    episodes['completed']=episodes.outcome_code.ne(6)
    episodes['zone_high_nano']=episodes.zone_low_nano+width
    episodes['core_dwell_seconds']=episodes.core_dwell_ns/NS
    episodes['duration_seconds']=(episodes.end_ns-episodes.arrival_ns)/NS
    episodes['has_previous_departure']=episodes.previous_departure_ns.ge(0)
    episodes['absence_seconds']=np.where(episodes.has_previous_departure,(episodes.arrival_ns-episodes.previous_departure_ns)/NS,np.nan)
    episodes['approach_efficiency']=np.divide((episodes.arrival_mid_nano-episodes.previous_departure_mid_nano).abs(),episodes.approach_path_nano,out=np.full(len(episodes),np.nan),where=episodes.has_previous_departure & episodes.approach_path_nano.gt(0))
    episodes['return_available_ns']=episodes.arrival_ns.where(episodes.has_previous_departure,-1)
    # Each return's excursion is only in its already observed absence interval.
    peaks=np.full(len(episodes),np.nan)
    for i,r in enumerate(episodes.itertuples()):
        if r.has_previous_departure:
            part=mid[r.previous_departure_index:r.arrival_index+1]
            peaks[i]=max(0,r.zone_low_nano-int(part.min()),int(part.max())-r.zone_high_nano)/TICK
    episodes['absence_peak_distance_ticks']=peaks
    events=events.sort_values(['available_ns','source_ordinal','event_code'],kind='stable').reset_index(drop=True)
    episodes=episodes.sort_values(['end_ns','episode_id'],kind='stable').reset_index(drop=True)
    return events,episodes


@njit(cache=True)
def _occupancy(qt,mid,good,epoch,base,n,width,start,end,step,gap):
    rows=(end-start)//step
    # Integrate the native clock exactly. Summing float seconds can make a
    # fully occupied bin exceed 1 after normalization and fabricate missingness.
    dwell=np.zeros((rows,n),np.int64);support=np.zeros(rows,np.int64)
    token=np.full(rows,-1,np.int64);mixed=np.zeros(rows,np.bool_)
    for i in range(len(qt)):
        if not good[i]:continue
        a=max(start,qt[i]);b=min(end,qt[i]+gap,qt[i+1] if i+1<len(qt) else end)
        while a<b:
            r=(a-start)//step
            if r>=rows:break
            boundary=min(b,start+(r+1)*step);dt=boundary-a
            dwell[r,mid[i]//width-base]+=dt;support[r]+=dt
            if token[r]<0:token[r]=epoch[i]
            elif token[r]!=epoch[i]:mixed[r]=True
            a=boundary
    return dwell,support,token,mixed


def occupancy_grid(quotes,*,start_ns,end_ns,bin_ticks=10,sample_seconds=5,max_quote_age_seconds=30):
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in (start_ns,end_ns)):raise ValueError('Integer interval clocks required')
    step=int(sample_seconds*NS)
    start=int(start_ns)//step*step;end=int(end_ns)//step*step
    quotes=quotes.loc[quotes.available_ns<=end]
    qt=quotes.available_ns.to_numpy(np.int64);mid=(quotes.bid_nano.to_numpy(np.int64)+quotes.ask_nano.to_numpy(np.int64))//2
    good=quotes.valid.to_numpy(bool)&(quotes.epoch.to_numpy()>=0)&(quotes.bid_nano.to_numpy()>0)&(quotes.ask_nano.to_numpy()>quotes.bid_nano.to_numpy())
    width=int(bin_ticks*TICK);base=int(mid[good].min()//width);n=int(mid[good].max()//width)-base+1
    dwell,support,epoch,mixed=_occupancy(qt,mid,good,quotes.epoch.to_numpy(np.int64),base,n,width,start,end,step,int(max_quote_age_seconds*NS))
    known=(support==step)&~mixed
    # A zero-duration invalid publication is still a support discontinuity.
    bad_bins=(qt[~good]-start)//step
    bad_bins=bad_bins[(bad_bins>=0)&(bad_bins<len(known))]
    known[bad_bins]=False
    return dict(times_ns=np.arange(start+step,end+1,step,dtype=np.int64),occupancy=dwell/step,
        known=known,epoch=epoch,price_bin_nano=(np.arange(n,dtype=np.int64)+base)*width,support_seconds=support/NS)
