"""Frozen multi-day episode-memory and price-zone game audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT,source_for,digest,dump
from experiments.mechanical_lab.signals import build_features
from experiments.mechanical_lab.execution import QuoteTape,ExecutionConfig
from experiments.mechanical_lab.portfolio import prepare_outcomes,account
from .episodes import build_episodes,occupancy_grid,NS
from .candidates import make_candidates,select_policy
from .game import simulate_zone_pairs,run_account

HERE=Path(__file__).resolve().parent
POLICIES=['return_frequent','return_sticky','return_permeable','return_sticky_spectral','return_sticky_interval','inside_now','wait_outside']


def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,(np.bool_,)):return bool(obj)
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
    return obj


def prepare_day(date,split,index,cfg,destination):
    begun=time.perf_counter();out=Path(destination)/'days'/date;out.mkdir(parents=True,exist_ok=False)
    source=source_for(date,cfg['route']);p=Path(source['market']);p=p if p.is_absolute() else ROOT/p
    mm=json.loads((p/'manifest.json').read_text())
    if mm['status']!='COMPLETE' or digest(p/'manifest.json')!=source['market_manifest_sha256']:raise ValueError('Canonical manifest mismatch')
    for name in ('bars.parquet','quotes.parquet'):
        if digest(p/name)!=mm['payloads'][name]:raise ValueError('Canonical payload mismatch')
    q=pd.read_parquet(p/'quotes.parquet');f=build_features(pd.read_parquet(p/'bars.parquet'))
    selection=int(source['entry']['selection']['selection_available_ns']);f['ready'] &= f.t_ns>=selection
    day=int(pd.Timestamp(date,tz='UTC').value);end=int(mm['input']['end_ns'])+1
    ev,ep=build_episodes(q,end_ns=end,**cfg['geometry'])
    grid=occupancy_grid(q,start_ns=day,end_ns=day+24*3600*NS,bin_ticks=cfg['geometry']['bin_ticks'],sample_seconds=cfg['spectral']['sample_seconds'])
    c,parents,tokens,spec=make_candidates(q,f,ev,ep,grid,cfg,index,date)
    for frame in (ev,ep,c,parents,tokens,spec):frame['date']=date;frame['split']=split
    tape=QuoteTape.from_frame(q,coverage_end_ns=end,entry_start_ns=max(selection,day+7*3600*NS),deadline_ns=day+(20*3600+30*60)*NS)
    outside=c.loc[~c.inside].drop(columns=['deadline_ns'])
    main=simulate_zone_pairs(tape,outside,holding_seconds=cfg['holding_seconds'],config=cfg['execution'])
    stressconfig={**cfg['execution'],**cfg['stress']}
    stressed=simulate_zone_pairs(tape,outside,holding_seconds=cfg['holding_seconds'],config=stressconfig)
    pc=c.loc[c.parent_episode_id.ge(0)].copy()
    parent_outcomes=simulate_zone_pairs(tape,pc,holding_seconds=cfg['parent_comparison']['deadline_seconds_from_arrival'],config=cfg['execution'])
    parent_stress=simulate_zone_pairs(tape,pc,holding_seconds=cfg['parent_comparison']['deadline_seconds_from_arrival'],config=stressconfig)
    # Return trials are separate from visit completion. No-return at horizon is
    # only assigned with uninterrupted support through that deadline.
    returns=ep.loc[ep.has_previous_departure,['zone_low_nano','previous_departure_ns','arrival_ns','absence_seconds','approach_efficiency','absence_peak_distance_ticks']].copy()
    returns=returns.rename(columns={'arrival_ns':'available_ns'})
    departures=ev.loc[ev.event_type=='DEPARTURE_CONFIRMED'].copy()
    by_departure={(int(r.zone_low_nano),int(r.previous_departure_ns)):r for r in returns.itertuples()}
    qt=q.available_ns.to_numpy(np.int64);qgood=q.valid.to_numpy(bool)&q.epoch.ge(0).to_numpy()&(q.ask_nano>q.bid_nano).to_numpy()
    bad=np.r_[0,np.cumsum(~qgood)];epochs=q.epoch.to_numpy();longgap=np.r_[0,np.cumsum(np.r_[False,np.diff(qt)>30*NS])]
    bad_times=np.sort(np.r_[qt[~qgood],qt[:-1][np.diff(qt)>30*NS]+30*NS,qt[1:][epochs[1:]!=epochs[:-1]]])
    trials=[]
    for d in departures.itertuples():
        deadline=int(d.available_ns)+300*NS
        r=by_departure.get((int(d.zone_low_nano),int(d.available_ns)))
        returned=r is not None and r.available_ns<=deadline
        maturity=int(r.available_ns) if returned else deadline
        a=int(np.searchsorted(qt,int(d.available_ns),'right')-1);b=int(np.searchsorted(qt,maturity,'right')-1)
        supported=a>=0 and b>=a and maturity<=end and maturity-qt[b]<=30*NS and bad[b+1]-bad[a]==0 and longgap[b+1]-longgap[a+1]==0 and epochs[a]==epochs[b]
        censor_pos=int(np.searchsorted(bad_times,int(d.available_ns),'right'))
        first_bad=int(bad_times[censor_pos]) if censor_pos<len(bad_times) else end
        availability=maturity if supported else min(maturity,end,first_bad)
        trials.append(dict(episode_id=d.episode_id,zone_low_nano=d.zone_low_nano,departure_ns=d.available_ns,available_ns=availability,
            status='RETURN' if returned and supported else 'NO_RETURN_300S' if supported else 'CENSORED',
            seconds_to_return=r.absence_seconds if returned and supported else np.nan,
            approach_efficiency=r.approach_efficiency if returned and supported else np.nan,
            absence_peak_distance_ticks=r.absence_peak_distance_ticks if returned and supported else np.nan,date=date,split=split))
    tables={'events':ev,'episodes':ep,'candidates':c,'parents':parents,'history_tokens':tokens,'spectral':spec,'returns':returns,
            'return_trials':pd.DataFrame(trials),'outcomes':main,'stress_outcomes':stressed,'parent_outcomes':parent_outcomes,'parent_stress_outcomes':parent_stress}
    for name,frame in tables.items():frame.to_parquet(out/f'{name}.parquet',index=False)
    receipt=dict(status='COMPLETE',date=date,split=split,source=str(p),source_manifest_sha256=digest(p/'manifest.json'),
        source_payloads={name:mm['payloads'][name] for name in ('bars.parquet','quotes.parquet')},quote_rows=len(q),
        events=len(ev),episodes=len(ep),completed_episodes=int(ep.completed.sum()),outside_candidates=len(outside),
        inside_candidates=int(c.inside.sum()),parent_contacts=len(parents),history_tokens=len(tokens),
        event_counts=ev.event_type.value_counts().to_dict(),outcome_counts=ep.outcome.value_counts().to_dict(),
        grid_full_support_fraction=float(grid['known'].mean()),elapsed_seconds=time.perf_counter()-begun,
        artifact_sha256={p.name:digest(p) for p in sorted(out.glob('*.parquet'))})
    dump(out/'receipt.json',clean(receipt));return receipt


def assemble(destination,cfg,receipts):
    out=Path(destination);frames={}
    for name in ('candidates','parents','history_tokens','spectral','episodes','return_trials','outcomes','stress_outcomes','parent_outcomes','parent_stress_outcomes'):
        frames[name]=pd.concat([pd.read_parquet(out/'days'/r['date']/f'{name}.parquet') for r in sorted(receipts,key=lambda x:x['date'])],ignore_index=True)
    c=frames['candidates'];reports=[];trades=[];coinrows=[]
    for split in cfg['splits']:
        cc=c.loc[c.split==split]
        if cc.empty:continue
        dates=set(cc.date)
        for name in POLICIES:
            chosen=select_policy(cc,name)
            parent=name in ('inside_now','wait_outside')
            if not parent:chosen=chosen.drop(columns=['deadline_ns'])
            outcomes=frames['parent_outcomes' if parent else 'outcomes'];outcomes=outcomes.loc[outcomes.date.isin(dates)]
            stress=frames['parent_stress_outcomes' if parent else 'stress_outcomes'];stress=stress.loc[stress.date.isin(dates)]
            lookup=prepare_outcomes(outcomes)
            for direction in ('toward','away'):
                summary,log=run_account(chosen,outcomes,direction=direction)
                row=dict(split=split,policy=name,direction=direction,stress=False,**summary);reports.append(row)
                if len(log):log['split']=split;log['policy']=name;log['direction']=direction;log['stress']=False;trades.append(log)
            summary,log=run_account(chosen,stress,direction='toward')
            reports.append(dict(split=split,policy=name,direction='toward',stress=True,**summary))
            if len(log):log['split']=split;log['policy']=name;log['direction']='toward';log['stress']=True;trades.append(log)
            seeds=cfg['coin_seeds'] if name in ('return_sticky','return_sticky_spectral','return_sticky_interval') else 39
            coins=[];coin_complete=[]
            for seed in range(1,seeds+1):
                s,_=account(chosen,lookup,direction='coin',seed=seed);coins.append(s['net_usd'])
                coin_complete.append(bool(s['capital_curve_complete']))
                coinrows.append(dict(split=split,policy=name,seed=seed,net_usd=s['net_usd'],trades=s['trades'],complete=s['capital_curve_complete']))
            valid_null=all(coin_complete)
            for row in reports[-3:]:row.update(coin_seeds=seeds,coin_complete_runs=sum(coin_complete),coin_distribution_complete=valid_null,
                coin_reference_execution='base',coin_median_usd=float(np.median(coins)) if valid_null else None,
                coin_p95_usd=float(np.quantile(coins,.95)) if valid_null else None)
    pd.DataFrame(reports).to_parquet(out/'policy_summary.parquet',index=False)
    pd.DataFrame(coinrows).to_parquet(out/'coin_controls.parquet',index=False)
    pd.concat(trades,ignore_index=True).to_parquet(out/'account_trades.parquet',index=False) if trades else pd.DataFrame().to_parquet(out/'account_trades.parquet',index=False)
    # Per-parent diagnostic includes WAIT=0, without pretending overlapping
    # hypothetical parents constitute one executable capital curve.
    parent_rows=[]
    lookup=prepare_outcomes(frames['parent_outcomes'])
    cmap=c.set_index('candidate_id')
    for p in frames['parents'].itertuples():
        row=dict(date=p.date,split=p.split,episode_id=p.episode_id,inside_pnl_usd=0.,outside_pnl_usd=0.,inside_status='WAIT',outside_status='WAIT')
        for arm,cid in (('inside',p.inside_candidate_id),('outside',p.outside_candidate_id)):
            if cid<0:continue
            candidate=cmap.loc[cid];r=lookup[(int(candidate.origin_index),int(candidate.side))]
            row[arm+'_status']=r.status
            row[arm+'_pnl_usd']=r.net_per_oz*100 if r.status=='COMPLETE' else np.nan if r.status=='CENSORED' else 0.
        parent_rows.append(row)
    pd.DataFrame(parent_rows).to_parquet(out/'parent_comparison.parquet',index=False)
    for name in ('candidates','spectral','episodes','return_trials','history_tokens'):frames[name].to_parquet(out/f'{name}.parquet',index=False)
    return reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='research_results/zone_episode_lab/verified_v1');parser.add_argument('--workers',type=int,default=2);parser.add_argument('--dates',nargs='*');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',cfg)
    dates=[]
    for split,(a,b) in cfg['splits'].items():dates.extend((str(d.date()),split) for d in pd.bdate_range(a,b))
    if args.dates:dates=[x for x in dates if x[0] in args.dates]
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [HERE/n for n in ('episodes.py','candidates.py','spectral.py','game.py','run.py','config.json')]}
    started=time.perf_counter();receipts=[]
    dump(out/'manifest.json',dict(status='RUNNING',config_sha256=digest(HERE/'config.json'),implementation_sha256=source_hashes,dates=dates))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(prepare_day,date,split,i+1,cfg,str(out)) for i,(date,split) in enumerate(dates)]
            for job in as_completed(jobs):
                receipt=job.result();receipts.append(receipt)
                print(json.dumps({k:receipt[k] for k in ('date','quote_rows','episodes','outside_candidates','inside_candidates','parent_contacts','elapsed_seconds')}),flush=True)
        reports=assemble(out,cfg,receipts)
    except Exception as exc:
        dump(out/'manifest.json',dict(status='FAILED',error=repr(exc),implementation_sha256=source_hashes,completed_days=[r['date'] for r in receipts]))
        raise
    summary=dict(status='COMPLETE_DIAGNOSTIC_ONLY',elapsed_seconds=time.perf_counter()-started,days=len(receipts),
        quote_rows=sum(r['quote_rows'] for r in receipts),episodes=sum(r['episodes'] for r in receipts),
        outside_candidates=sum(r['outside_candidates'] for r in receipts),parent_contacts=sum(r['parent_contacts'] for r in receipts),
        implementation_sha256=source_hashes,config_sha256=digest(HERE/'config.json'),receipts=receipts,
        economic_admission='NOT_ESTABLISHED: all dates previously inspected; frozen diagnostic hypotheses only',
        artifacts={p.name:digest(p) for p in sorted(out.glob('*.parquet'))})
    dump(out/'manifest.json',clean(summary));dump(out/'policy_summary.json',clean(reports))
    print(json.dumps({k:summary[k] for k in ('status','days','quote_rows','episodes','outside_candidates','parent_contacts','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
