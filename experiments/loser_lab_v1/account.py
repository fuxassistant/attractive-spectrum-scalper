"""Paper deposit experiments. Position sizing is NEVER a model criterion."""
from __future__ import annotations
import numpy as np
import pandas as pd


def replay(decisions,targets,post,config,*,mode='bid',style='fixed',reverse=False,side_override=None):
    cfg=config['risk'];cash=float(cfg['initial_deposit_usd']);peak=cash;mult=1;busy=-1;logs=[]
    counts=dict(offered=int(decisions.accepted.sum()),overlap=0,unaffordable=0,unknown=0,complete=True,bankrupt=False,depleted80=False)
    lookup=targets.set_index('candidate_id');rlookup=post.set_index(['candidate_id','mode','side']) if mode!='bid' else None
    if lookup.index.duplicated().any():raise ValueError('One target per candidate')
    for i,d in enumerate(decisions.sort_values(['t_ns','candidate_id']).itertuples()):
        if not d.accepted:continue
        if d.t_ns<=busy:counts['overlap']+=1;continue
        side=int(side_override[d.candidate_id] if side_override is not None else d.side)
        if reverse:side=-side
        if side not in (-1,1):raise ValueError('Accepted action requires a side')
        t=lookup.loc[d.candidate_id];planned=t.threshold_per_oz*cfg['contract_oz']*mult
        if cash<planned:
            counts['unaffordable']+=1
            # No scale-down hindsight or invented fractional contracts.
            continue
        if not t.known:counts['unknown']+=1;counts['complete']=False;break
        unit=side*t.bid_delta_per_oz;exit_ns=int(t.exit_ns)
        if mode!='bid':
            r=rlookup.loc[(d.candidate_id,mode,side)]
            if not r.known:counts['unknown']+=1;counts['complete']=False;break
            unit=float(r.net_per_oz);exit_ns=int(r.exit_ns)
        previous=cash;pnl=unit*cfg['contract_oz']*mult;cash+=pnl;peak=max(peak,cash);busy=exit_ns
        counts['depleted80']|=cash<=(1-cfg['depletion_fraction'])*cfg['initial_deposit_usd']
        counts['bankrupt']|=cash<=0
        logs.append(dict(candidate_id=d.candidate_id,date=d.date,t_ns=d.t_ns,exit_ns=exit_ns,side=side,multiplier=mult,
                         equity_before=previous,equity_usd=cash,pnl_usd=pnl,planned_risk_usd=planned,
                         event_reward=-side*int(t.event_sign),unit_bid_pnl=side*t.bid_delta_per_oz))
        if style=='after_loss':mult=min(cfg['max_multiplier'],mult*2) if pnl<0 else 1
        elif style=='after_gain':mult=min(cfg['max_multiplier'],mult*2) if pnl>0 else 1
        elif style!='fixed':raise ValueError('Unknown risk style')
        if cash<=0:break
    frame=pd.DataFrame(logs)
    risk=float(frame.planned_risk_usd.sum()) if len(frame) else 0.
    out=dict(**counts,trades=len(frame),net_usd=cash-cfg['initial_deposit_usd'],terminal_usd=cash,
             total_planned_risk_usd=risk,pnl_per_planned_risk=(cash-cfg['initial_deposit_usd'])/risk if risk else None,
             min_closed_equity_usd=float(min(cfg['initial_deposit_usd'],frame.equity_usd.min())) if len(frame) else cash,
             max_multiplier_used=int(frame.multiplier.max()) if len(frame) else 0)
    return out,frame
