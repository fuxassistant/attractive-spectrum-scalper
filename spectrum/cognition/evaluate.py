"""Repeat learned HOLD/CLOSE decisions on each saved option, no reset-equity sum."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.player.contract import NS
from .execution import ExecutionReplay, ExecutionConfig
from .dataset import OFFSETS, features, outcome
from .learn import load_predictor, SEEDS


def bootstrap_days(frame,column,seed=61021):
    """Moving blocks of 5 observed days, sensitivity diagnostic, not iid proof."""
    daily=frame.groupby('date')[column].mean().dropna().to_numpy()
    if not len(daily):return None
    rng=np.random.default_rng(seed);n=len(daily);length=min(5,n);samples=[]
    for _ in range(2000):
        starts=rng.integers(0,n-length+1,size=int(np.ceil(n/length)))
        values=np.concatenate([daily[s:s+length] for s in starts])[:n];samples.append(values.mean())
    return dict(mean=float(daily.mean()),lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),
                days=n,block_observed_days=length,meaning='Exploratory dependent-day block interval; not corrected for prior project selection')


def replay_option(day,root,controller,predictor=None,trace=False,market_override=None):
    r=ExecutionReplay(day,config=ExecutionConfig());r.open_at(root['intent_ns'],root['side'],root['plan_id'])
    r.advance_until(root['intent_ns']+3*NS)
    decisions=[];count=0;closes=0
    for age in OFFSETS:
        now=root['entry_ns']+age*NS
        if now>=root['horizon_ns']:break
        r.advance_until(now)
        if r.state.censored or r.state.dead or not r.state.side:break
        advantage=None;close=False
        if controller=='close_first':close=True
        elif controller=='age60':close=age>=60
        elif predictor is not None:
            x,_=features(day,r)
            if market_override is not None:x[:len(market_override)]=market_override
            advantage=float(predictor(x)[0]);close=advantage<0
        count+=1;closes+=int(close)
        if trace:decisions.append(dict(seconds=(now-root['entry_ns'])/NS,advantage=advantage,action='CLOSE' if close else 'HOLD',
            equity=r.state.equity,unrealized=r.state.equity-r.state.cash,stop_distance=float(r.position_features()[10]),target_distance=float(r.position_features()[9])))
        if close:r.request_close(now)
    r.reference_exit(root['horizon_ns']);result=outcome(r)
    if trace:
        lo=int(np.searchsorted(day.quote_t,root['intent_ns']-30*NS));hi=int(np.searchsorted(day.quote_t,root['horizon_ns'],side='right'))
        # Keep last quote per second for picture only; execution scans all quotes.
        indices=np.arange(lo,hi);seconds=day.quote_t[indices]//NS
        keep=np.r_[seconds[1:]!=seconds[:-1],True] if len(indices) else np.empty(0,bool)
        indices=indices[keep]
        if len(indices)>800:indices=indices[np.linspace(0,len(indices)-1,800).astype(int)]
        picture=dict(root_id=root['root_id'],date=root['date'],side=root['side'],plan_id=root['plan_id'],controller=controller,
            entry_price=root['entry_price_nano']/1e9,exit_seconds=(result['exit_ns']-root['entry_ns'])/NS,
            net=result['net'],reference_net=root['reference']['net'],decisions=decisions,
            price=[dict(seconds=(int(day.quote_t[j])-root['entry_ns'])/NS,bid=float(day.bid[j]/1e9),ask=float(day.ask[j]/1e9)) for j in indices])
    else:picture=None
    return result,count,closes,picture


def evaluate(data='data/derived/hold_close_v1',output='reports/hold_close_v1'):
    torch.set_num_threads(2);started=time.perf_counter();root=Path(data);out=Path(output)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    predictors={}
    for variant in ('full','account_only'):
        for seed in SEEDS:
            name=f'{variant}_{seed}';predictors[name]=load_predictor(out/(name+'.pt'))[0]
    predictors['permuted_targets_61021']=load_predictor(out/'permuted_targets_61021.pt')[0]
    # Controls + all seeds on the chronological test; validation was used only for loss selection.
    controllers=['fixed_hold','close_first','age60']+list(predictors)
    rows=[];cases=[];per_day=[]
    for dm in manifest['days']:
        if dm['split']!='test':continue
        roots=json.loads((root/dm['date']/'roots.json').read_text(encoding='utf8'))
        if not roots:continue
        day=MonthDay(roots[0]['source_path']);daily=[]
        # First test day and first matching root per plan/side with >=30s survival:
        # predetermined sampling criterion, no ranking by model profit.
        selected=set()
        for rr in roots:
            ref=rr['reference']
            eligible_case=len(per_day)<2 and ref['exit_ns']-rr['entry_ns']>=30*NS and (rr['side'],rr['plan_id']) not in selected
            if eligible_case:selected.add((rr['side'],rr['plan_id']))
            for controller in controllers:
                trace=eligible_case and controller=='full_61021'
                result,n,closes,picture=replay_option(day,rr,controller,predictors.get(controller),trace=trace)
                valid=not (result['censored'] or result['open'] or ref['censored'] or ref['open'])
                row=dict(root_id=rr['root_id'],origin_id=rr['origin_id'],date=rr['date'],split='test',side=rr['side'],plan_id=rr['plan_id'],
                    controller=controller,net=result['net'] if not result['censored'] and not result['open'] else None,
                    reference_net=ref['net'] if not ref['censored'] and not ref['open'] else None,
                    paired_known=valid,delta_vs_hold=result['net']-ref['net'] if valid else None,
                    censored=result['censored'],dead=result['dead'],decision_count=n,close_count=closes,
                    equity_min=result['min_equity'],equity_max=result['max_equity'],exit_reason=result['exit_reason'],
                    exit_ns=result['exit_ns'])
                if controller=='fixed_hold' and valid and abs(row['delta_vs_hold'])>1e-8:raise ValueError('Saved reference replay drift')
                daily.append(row)
                if picture:cases.append(picture)
        rows.extend(daily);per_day.append(dict(date=dm['date'],roots=len(roots),seconds=time.perf_counter()-started))
        print(json.dumps(per_day[-1]),flush=True);QUOTE_CACHE.clear()
    frame=pd.DataFrame(rows);frame.to_parquet(out/'sequential_options.parquet',index=False)
    reports=[]
    for name,g in frame.groupby('controller',sort=False):
        good=g.loc[g.paired_known]
        reports.append(dict(controller=name,roots=len(g),known_comparisons=len(good),censored=int(g.censored.sum()),
            dead=int(g.dead.sum()),mean_net=float(good.net.mean()),mean_delta_vs_hold=float(good.delta_vs_hold.mean()),
            median_net=float(good.net.median()),net_p05=float(good.net.quantile(.05)),
            roots_with_close=int((g.close_count>0).sum()),zero_decision_roots=int((g.decision_count==0).sum()),
            equity_min=float(g.equity_min.min()),equity_max=float(g.equity_max.max()),
            day_block_interval=bootstrap_days(good,'delta_vs_hold')))
    save_json(out/'sequential_summary.json',reports);save_json(out/'cases.json',cases)
    save_json(out/'evaluation_receipt.json',dict(status='COMPLETE',seconds=time.perf_counter()-started,days=per_day,
        source_sha256=sha(Path(__file__)),controllers=controllers,source_manifest_sha256=sha(root/'manifest.json'),
        scope='Repeated HOLD/CLOSE decisions on identical independent-reset options. No overlapping root PnL summed into a capital curve.',
        limitation='Q target uses fixed-plan continuation; repeated selected policy is separately tested, not relabelled as its own Q.',
        data_disposition='August chronological technical test on previously researched history; no test-based checkpoint choice.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/derived/hold_close_v1');p.add_argument('--output',default='reports/hold_close_v1')
    a=p.parse_args();evaluate(a.data,a.output)
