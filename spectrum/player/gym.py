"""Minute decisions, publication-by-publication execution, explicit support loss.

Gym-style reset/step API. No external Gymnasium dependency. Prices remain int64
nanodollars until PnL. A plan is immutable after entry; WAIT while invested means
the executor continues that plan, not that the position disappears.
"""
import numpy as np
from numba import njit
from .contract import Contract,NS,PLANS

# Integer state: side, entry price/time/index, plan, pending side/plan/due/expiry,
# last publication time/epoch, censored, halted, pending epoch, entry decision.
# Float state: cash PnL, liquidation PnL, peak, max drawdown, last trade PnL.
REASONS={1:'take',2:'stop',3:'time',4:'session_close',5:'risk_stop',6:'support_loss'}

@njit(cache=True)
def advance(lo,hi,t,source,bid,ask,valid,epoch,s,f,deadline,cutoff,latency,
            timeout,gap,fee,size,risk,plans,trace):
    trade=np.full(13,-1.,dtype=np.float64)
    for j in range(lo,hi):
        now=t[j]
        supported=valid[j] and bid[j]>0 and ask[j]>=bid[j] and now-source[j]<=gap
        if s[0]!=0 and (not supported or epoch[j]!=s[10] or (s[9]>0 and now-s[9]>gap)):
            s[11]=1
        s[9]=now;s[10]=epoch[j]
        if s[5]!=0:
            if now>s[8] or now>=cutoff or now>=deadline or epoch[j]!=s[13]:
                s[5]=0
            elif now>s[7] and supported:
                s[0]=s[5];s[4]=s[6];s[1]=ask[j] if s[0]>0 else bid[j]
                s[2]=now;s[3]=j;s[5]=0;s[11]=0
                f[0]-=fee
        if s[0]!=0 and supported:
            px=bid[j] if s[0]>0 else ask[j]
            gross=s[0]*(px-s[1])/1e9*size
            f[1]=f[0]+gross;f[2]=max(f[2],f[1]);f[3]=max(f[3],f[2]-f[1])
            reason=0;p=plans[s[4]]
            if s[11]:reason=6
            elif f[2]-f[1]>=risk:reason=5
            elif now>=deadline:reason=4
            elif s[0]*(px-s[1])<=-p[1]*100_000_000:reason=2
            elif s[0]*(px-s[1])>=p[0]*100_000_000:reason=1
            elif now-s[2]>=p[2]*NS:reason=3
            if reason:
                trade[0]=s[14];trade[1]=s[3];trade[2]=j;trade[3]=s[0];trade[4]=s[4]
                trade[5]=s[1];trade[6]=px;trade[7]=gross-fee;trade[8]=reason
                trade[9]=s[11];trade[10]=s[2];trade[11]=now;trade[12]=gross
                f[0]+=gross;f[4]=gross-fee;s[0]=0;s[11]=0
        if s[0]==0:f[1]=f[0]
        f[2]=max(f[2],f[1]);f[3]=max(f[3],f[2]-f[1])
        if f[2]-f[1]>=risk:s[12]=1;s[5]=0
        if len(trace):trace[j]=f[1]
    return trade


class TradingGym:
    action_names=('WAIT','LONG/fast','LONG/medium','LONG/wide','SHORT/fast','SHORT/medium','SHORT/wide')
    account_dim=7
    def __init__(self,day,contract=None,record=False):
        self.day=day;self.c=contract or Contract();self.record=record
        self.plans=np.asarray(PLANS,dtype=np.int64)
        self.reset()

    def reset(self):
        self.i=0;self.s=np.zeros(15,dtype=np.int64);self.f=np.zeros(5)
        self.trades=[];self.equity=[0.];self.actions=[];self.cancelled=0
        self.trace=np.full(len(self.day.quote_t),np.nan) if self.record else np.empty(0)
        self.q=int(self.day.cuts[0]);self.done=False
        if self.q:
            self.s[9]=self.day.quote_t[self.q-1];self.s[10]=self.day.epoch[self.q-1]
        if self.record:self.trace[:self.q]=0.
        return self.observation(),{'action_mask':self.mask()}

    def account(self):
        return np.asarray([self.s[0],self.f[0]/30,self.f[1]/30,(self.f[2]-self.f[1])/30,
                           self.f[4]/5,self.s[12],min(len(self.trades)/100,5)],dtype=np.float32)

    def mask(self):
        m=np.zeros(7,dtype=bool);m[0]=True
        if self.done or self.s[0] or self.s[5] or self.s[12] or not self.day.can_enter[self.i]:return m
        t=int(self.day.decision_t[self.i])
        for p,plan in enumerate(PLANS):
            if t+self.c.pending_timeout_ns+plan[2]*NS<self.day.deadline:m[1+p]=m[4+p]=True
        return m

    def observation(self):return {'market':self.day.x[self.i], 'account':self.account()}

    def step(self,action):
        if self.done:raise RuntimeError('Episode ended')
        action=int(action)
        if action<0 or action>=7 or not self.mask()[action]:raise ValueError('Illegal action')
        before=self.f[1];self.actions.append(action)
        if action:
            self.s[5]=1 if action<=3 else -1;self.s[6]=(action-1)%3
            self.s[7]=int(self.day.decision_t[self.i])+self.c.latency_ns
            self.s[8]=self.s[7]+self.c.pending_timeout_ns;self.s[13]=self.s[10];self.s[14]=self.i
        pending=self.s[5]
        hi=int(self.day.cuts[self.i+1]) if self.i+1<len(self.day.x) else len(self.day.quote_t)
        row=advance(self.q,hi,self.day.quote_t,self.day.source_t,self.day.bid,self.day.ask,
                    self.day.valid,self.day.epoch,self.s,self.f,self.day.deadline,int(self.day.meta['ny_close_ns']),
                    self.c.latency_ns,self.c.pending_timeout_ns,self.c.support_gap_ns,self.c.fee_roundtrip,
                    self.c.size_oz,self.c.daily_drawdown_stop,self.plans,self.trace)
        if row[0]>=0:
            names=('decision_index','entry_index','exit_index','side','plan','entry_nano','exit_nano','net','reason_code','censored','entry_ns','exit_ns','gross')
            item={n:(float(v) if n in ('net','gross') else int(v)) for n,v in zip(names,row)}
            # float64 cannot exactly represent nanosecond clocks. Recover from source arrays.
            item['entry_ns']=int(self.day.quote_t[item['entry_index']]);item['exit_ns']=int(self.day.quote_t[item['exit_index']])
            item['reason']=REASONS[item['reason_code']];self.trades.append(item)
        if pending and not self.s[5] and not self.s[0] and row[0]<0:self.cancelled+=1
        self.q=hi;self.equity.append(float(self.f[1]));self.i+=1;self.done=self.i>=len(self.day.x)
        info={'action_mask':np.asarray([True]+[False]*6) if self.done else self.mask()}
        if self.done:info.update(self.metrics())
        return (None if self.done else self.observation()),float(self.f[1]-before),self.done,False,info

    def metrics(self):
        censored=sum(x['censored'] for x in self.trades)+int(self.s[0]!=0)
        return dict(net=float(self.f[1]),realized=float(self.f[0]),max_drawdown=float(self.f[3]),
                    trades=len(self.trades),censored=censored,complete=not bool(censored),
                    cancelled=self.cancelled,risk_halted=bool(self.s[12]),open_terminal=bool(self.s[0]),
                    wins=sum(x['net']>0 for x in self.trades),action_counts=np.bincount(self.actions,minlength=7).tolist())
