"""Causal, latency-aware basket martingale on the certified quote tape.

All positions are liquidated before the daily data boundary. Losing series carry
their original capital base across baskets and days. No invented gap fills.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from numba import njit, prange

NS = 10**9
FIELDS = ('cash', 'equity', 'peak', 'max_drawdown', 'series_base', 'series_min',
          'max_series_loss', 'series_active', 'ruined', 'censored', 'orders',
          'closed_baskets', 'recovered_series', 'max_oz', 'fees', 'turnover', 'baskets')
EVENT_FIELDS = ('quote_index', 'kind', 'side', 'price', 'quantity', 'total_oz',
                'average_entry', 'cash', 'equity', 'series_base', 'basket_net')
# kind: 1 initial entry, 2 add, 3 target, 4 basket stop, 5 time/session,
#       6 series death, 7 unsupported observation (no execution).


@dataclass(frozen=True)
class BasketConfig:
    mode: str = 'fade'
    lookback: int = 60
    deviation: float = 1.
    er_max: float = .35
    vol_max: float = 1.5
    step_scale: float = 2.
    step_growth: float = 1.5
    target_scale: float = 2.
    multiplier: float = 1.5
    max_orders: int = 4
    cap_oz: float = 12.
    max_hold_minutes: int = 120
    basket_stop_fraction: float = .15
    cooldown_minutes: int = 10

    def __post_init__(self):
        if self.mode not in ('long', 'short', 'trend', 'fade'):
            raise ValueError('Unknown direction rule')
        if not all(np.isfinite(v) for k,v in asdict(self).items() if k!='mode'):
            raise ValueError('Nonfinite configuration')
        if not 2<=self.lookback<=240 or not 1<=self.max_orders<=8:
            raise ValueError('Invalid lookback/order count')
        if min(self.deviation,self.er_max,self.vol_max)<0 or min(self.step_scale,self.target_scale)<=0:
            raise ValueError('Invalid signal/grid distances')
        if not 1<=self.multiplier<=2.5 or not 1<=self.step_growth<=3 or not 1<=self.cap_oz<=32:
            raise ValueError('Invalid exposure bound')
        if not 0<self.basket_stop_fraction<=.5 or not 1<=self.max_hold_minutes<=840 or self.cooldown_minutes<1:
            raise ValueError('Invalid lifetime/stop bound')

    def array(self):
        return np.array([self.step_scale,self.step_growth,self.target_scale,self.multiplier,
            self.max_orders,self.cap_oz,self.max_hold_minutes*60*NS,
            self.basket_stop_fraction,self.cooldown_minutes*60*NS],dtype=np.float64)


def initial_state():
    a=np.zeros(len(FIELDS));a[:3]=1500.;return a


def signals(day, config):
    """Only completed/available M1 candles. No backfill or future normalization.

    Scale = sqrt(5) * mean M1 high-low range over last 60 known contiguous
    minutes, floored at $0.50. It is a range scale, not Wilder ATR.
    """
    x=np.asarray(day.x);n=len(x);known=x[:,8]>.5
    close=pd.Series(np.cumsum(x[:,3].astype(float))*.1)
    span=config.lookback
    ranges=pd.Series(x[:,1].astype(float)*.1).where(known)
    scale=(ranges.rolling(60,min_periods=60).mean()*np.sqrt(5)).clip(lower=.5)
    vol=scale/scale.rolling(180,min_periods=60).mean()
    er=close.diff(span).abs()/close.diff().abs().rolling(span).sum().replace(0,np.nan)
    center=close.rolling(span,min_periods=span).mean()
    dev=(close-center)/scale
    contiguous=np.r_[True,np.diff(np.asarray(day.bar_t))==60*NS]
    supported=pd.Series(known & contiguous).rolling(max(span+1,120),min_periods=max(span+1,120)).sum()==max(span+1,120)
    ok=np.asarray(day.can_enter)&supported.to_numpy()&(er.to_numpy()<=config.er_max)&(vol.to_numpy()<=config.vol_max)
    if config.mode=='long': side=np.where(dev<=-config.deviation,1,0)
    elif config.mode=='short': side=np.where(dev>=config.deviation,-1,0)
    elif config.mode=='fade': side=np.where(dev>=config.deviation,-1,np.where(dev<=-config.deviation,1,0))
    else: side=np.where(dev>=config.deviation,1,np.where(dev<=-config.deviation,-1,0))
    side=np.where(ok,side,0).astype(np.int8)
    return side,np.nan_to_num(scale.to_numpy(),nan=.5)


@njit(cache=True)
def replay_day(t,bid,ask,valid,segment,dt,side_signal,scale,p,state,london,cutoff,deadline,
               record=False,latency=100_000_000,fee=.30):
    """Mutates account state, returns exact integer event clocks + compact traces.

    Market orders fill at first supported publication strictly after latency,
    never at an intended grid/target level. Add/target/stop intents expire after
    2 seconds. Death barrier is checked at every supported quote before intents.
    """
    s=state.copy();limit=100000 if record else 0
    clocks=np.zeros(limit,np.int64);events=np.zeros((limit,11));count=0
    trace=np.full((len(dt),4),np.nan) if record else np.empty((0,4))
    pos=0;qty=0.;cost=0.;last_entry=0.;orders=0;entry_time=0;epoch=-1
    basket_base=0.;basket_fee=0.;basket_scale=1.;next_step=0.;next_allowed=0
    pending=0;ready=0;expiry=0;pending_size=0.;pending_side=0
    k=0;j=0;last_equity=s[1]
    if s[8] or s[9]:return s,clocks[:0],events[:0],trace
    while j<len(t):
        # Flat intervals do not need a scan of millions of irrelevant quotes.
        if pos==0 and pending==0:
            while k<len(dt):
                if record:trace[k]=np.array([s[1],0.,s[4]*.5 if s[7] else np.nan,0.])
                if side_signal[k] and dt[k]>=next_allowed and london<=dt[k]<min(cutoff,deadline):
                    pending=1;pending_side=int(side_signal[k]);pending_size=1.
                    ready=dt[k]+latency;expiry=dt[k]+2*NS;basket_scale=scale[k]
                    k+=1;break
                k+=1
            if pending==0:break
            j=max(j,np.searchsorted(t,ready,side='right'))
            if j>=len(t):break
        now=t[j];b=bid[j]/1e9;a=ask[j]/1e9
        supported=valid[j] and b>0 and a>=b
        if pos and (not supported or segment[j]!=epoch):
            s[9]=1
            if record:
                clocks[count]=now;events[count]=np.array([j,7,pos,0.,0.,qty,cost/qty,s[0],last_equity,s[4],np.nan]);count+=1
            break
        reason=0;action=0;fill=0.;filled=0.;basket_net=0.
        if pos:
            mark=b if pos>0 else a
            s[1]=s[0]+pos*(qty*mark-cost)
            s[2]=max(s[2],s[1]);s[3]=max(s[3],s[2]-s[1]);s[5]=min(s[5],s[1])
            s[6]=max(s[6],max(0.,(s[4]-s[1])/s[4]))
            if s[1]<.5*s[4]-1e-8:reason=6;s[8]=1
            elif now>=deadline:reason=5
        if reason:
            pending=0
        elif pending and (now>expiry or (pending in (1,2) and now>=min(cutoff,deadline))):
            pending=0
        elif pending and now>ready and supported:
            if pending in (1,2):
                if not pos:
                    if not s[7]:s[7]=1;s[4]=s[0];s[5]=s[0]
                    pos=pending_side;epoch=segment[j];entry_time=now;basket_base=s[0];basket_fee=0.;orders=0;cost=0.;qty=0.
                    next_step=max(.5,p[0]*basket_scale);s[16]+=1
                fill=a if pos>0 else b;filled=pending_size
                cost+=filled*fill;qty+=filled;orders+=1;last_entry=fill
                s[0]-=fee*filled;basket_fee+=fee*filled;s[14]+=fee*filled;s[15]+=fill*filled;s[10]+=1;s[13]=max(s[13],qty)
                s[1]=s[0]+pos*(qty*(b if pos>0 else a)-cost)
                action=pending
                if action==2:next_step*=p[1]
                s[5]=min(s[5],s[1]);s[6]=max(s[6],max(0.,(s[4]-s[1])/s[4]));s[3]=max(s[3],s[2]-s[1])
                if record:
                    if count>=limit:raise ValueError('Event buffer exhausted')
                    clocks[count]=now;events[count]=np.array([j,action,pos,fill,filled,qty,cost/qty,s[0],s[1],s[4],0.]);count+=1
                if s[1]<.5*s[4]-1e-8:reason=6;s[8]=1
            else:reason=pending
            pending=0
        if reason and pos:
            fill=b if pos>0 else a
            s[0]+=pos*(qty*fill-cost);s[1]=s[0];s[15]+=qty*fill;s[11]+=1
            basket_net=s[0]-basket_base
            if record:
                if count>=limit:raise ValueError('Event buffer exhausted')
                clocks[count]=now;events[count]=np.array([j,reason,pos,fill,qty,0.,cost/qty,s[0],s[1],s[4],basket_net]);count+=1
            if s[0]>=s[4]-1e-8 and not s[8]:s[7]=0;s[12]+=1
            pos=0;qty=0.;cost=0.;pending=0;next_allowed=now+int(p[8])
        if pos and pending==0:
            # Target includes all reserved commissions. Add triggers use the
            # executable entry side so the initial spread cannot trigger adds.
            net=s[1]-basket_base
            if net>=max(.5,p[2]*basket_scale)*qty:pending=3
            elif net<=-p[7]*basket_base:pending=4
            elif now-entry_time>=p[6]:pending=5
            elif now<cutoff and orders<int(p[4]) and qty<p[5]-1e-8:
                px=a if pos>0 else b
                if pos*(px-last_entry)<=-next_step:
                    pending=2;pending_size=min(p[3]**orders,p[5]-qty);pending_side=pos
            if pending:ready=now+latency;expiry=now+2*NS
        last_equity=s[1]
        while k<len(dt) and dt[k]<=now:
            # Snapshot at/before publication clock; never feed this trace as X.
            if record:trace[k]=np.array([s[1],qty,s[4]*.5 if s[7] else np.nan,float(pos)])
            k+=1
        if s[8]:break
        j+=1
    if pos and not s[8]:s[9]=1  # unresolved end, no fabricated last-price fill
    if record and not s[9]:
        while k<len(dt):trace[k]=np.array([s[1],0.,s[4]*.5 if s[7] else np.nan,0.]);k+=1
    return s,clocks[:count],events[:count],trace


@njit(parallel=True,cache=True)
def batch_day(t,bid,ask,valid,segment,dt,signals_,scales,params,states,london,cutoff,deadline):
    out=states.copy()
    for i in prange(len(states)):
        out[i]=replay_day(t,bid,ask,valid,segment,dt,signals_[i],scales[i],params[i],states[i],london,cutoff,deadline)[0]
    return out


def summary(s):
    d={k:float(v) for k,v in zip(FIELDS,s)}
    d.update(complete=not bool(s[9]),alive=not bool(s[8] or s[9]),net=float(s[1]-1500),initial_capital=1500.)
    return d
