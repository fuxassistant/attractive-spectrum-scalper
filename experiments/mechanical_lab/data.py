"""Read verified immutable canonical MBO state, never historical experiment PnL."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'data/derived/full_mbo_v1'
NS=1_000_000_000

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def dates_for(config):
    out=[]
    for split,(a,b) in config['splits'].items():
        out.extend((str(d.date()),split) for d in pd.bdate_range(a,b))
    return out

def source_for(date,route):
    catalog=json.loads((CATALOG/'manifest.json').read_text())
    rows=[x for x in catalog['captures'] if x['status']=='COMPLETE'
          and x['entry']['selection']['date']==date and x['entry']['route']==route]
    if len(rows)!=1: raise ValueError(f'Expected one COMPLETE route/date: {date}')
    return rows[0]

def prepare_day(date,split,index,config,destination,stress=False):
    from .signals import build_features,generate_candidates
    from .execution import QuoteTape,ExecutionConfig,simulate_pairs,EXIT_PLANS
    started=time.perf_counter()
    out=Path(destination)/'days'/date
    out.mkdir(parents=True,exist_ok=False)
    source=source_for(date,config['route'])
    p=Path(source['market'])
    if not p.is_absolute(): p=ROOT/p
    mp=p/'manifest.json'
    mm=json.loads(mp.read_text())
    if mm['status']!='COMPLETE' or digest(mp)!=source['market_manifest_sha256']:
        raise ValueError('Canonical manifest identity mismatch')
    for name in ('bars.parquet','quotes.parquet'):
        if digest(p/name)!=mm['payloads'][name]: raise ValueError(f'Source hash mismatch {date}/{name}')
    bars=pd.read_parquet(p/'bars.parquet')
    features=build_features(bars)
    recipe=config.get('recipe')
    candidates=generate_candidates(features,
        signal_ids=list(dict.fromkeys([recipe['entry']['id'],'control_clock'])) if recipe else None,
        parameters={recipe['entry']['id']:recipe['entry'].get('parameters',{})} if recipe else None)
    selection_ns=int(source['entry']['selection']['selection_available_ns'])
    candidates=candidates.loc[candidates.t_ns>=selection_ns].copy()
    q=pq.read_table(p/'quotes.parquet',columns=['available_ns','bid_nano','ask_nano','valid','epoch']).to_pandas()
    # Never skip invalid publications while taking as-of market state.
    times=q.available_ns.to_numpy(np.int64)
    ix=np.searchsorted(times,candidates.t_ns.to_numpy(np.int64),'right')-1
    if (ix<0).any(): raise ValueError('Missing origin prefix')
    candidates['spread_per_oz']=(q.ask_nano.to_numpy()[ix]-q.bid_nano.to_numpy()[ix])/1e9
    candidates['quote_age_seconds']=(candidates.t_ns.to_numpy(np.int64)-times[ix])/NS
    candidates['quote_valid']=q.valid.to_numpy()[ix]
    candidates['quote_epoch']=q.epoch.to_numpy()[ix]
    candidates['bid_per_oz']=q.bid_nano.to_numpy()[ix]/1e9
    candidates['ask_per_oz']=q.ask_nano.to_numpy()[ix]/1e9
    candidates['cost_over_atr']=(candidates.spread_per_oz+config['execution']['fee_roundtrip_per_oz'])/candidates.risk_hint
    quality=config['origin_quality']
    mask=(candidates.quote_valid & candidates.quote_epoch.ge(0)&candidates.quote_epoch.eq(candidates.epoch)
        & candidates.bid_per_oz.gt(0) & candidates.spread_per_oz.gt(0)
        & candidates.spread_per_oz.le(quality['max_spread_per_oz'])
        & candidates.quote_age_seconds.le(config['execution']['max_quote_gap_seconds'])
        & candidates.bar_publication_delay_seconds.le(quality['max_bar_publication_delay_seconds'])
        & candidates.cost_over_atr.le(quality['max_cost_over_atr']))
    candidates['origin_admitted']=mask
    # Keep rejected signals in audit; replay only causally supported origins.
    all_candidates=candidates.copy()
    candidates=candidates.loc[mask].copy().sort_values(['t_ns','signal_id','side'])
    origins=candidates[['t_ns','risk_hint']].drop_duplicates('t_ns').sort_values('t_ns').reset_index(drop=True)
    if candidates.groupby('t_ns').risk_hint.nunique().gt(1).any(): raise ValueError('Origin risk differs across signals')
    origins['origin_index']=np.arange(len(origins))+index*10000
    candidates=candidates.merge(origins[['t_ns','origin_index']],on='t_ns',validate='many_to_one')
    candidates['candidate_id']=np.arange(len(candidates))+index*100000
    candidates['date']=date; candidates['split']=split
    candidates.to_parquet(out/'candidates.parquet',index=False)
    all_candidates.to_parquet(out/'all_signals.parquet',index=False)
    features.to_parquet(out/'features.parquet',index=False)
    day_ns=int(pd.Timestamp(date,tz='UTC').value)
    tape=QuoteTape.from_frame(q,coverage_end_ns=int(mm['input']['end_ns'])+1,
        entry_start_ns=max(day_ns+7*3600*NS,selection_ns),deadline_ns=day_ns+(20*3600+30*60)*NS)
    execution=dict(config['execution'])
    if stress: execution.update(config['stress'])
    ec=ExecutionConfig(**execution)
    frames=[]
    plans={recipe['exit']['id']:dict(EXIT_PLANS[recipe['exit']['id']].__dict__,**recipe['exit'].get('parameters',{}))} if recipe else EXIT_PLANS
    for plan_id,plan in plans.items():
        f=simulate_pairs(tape,origins.t_ns.to_numpy(np.int64),origins.risk_hint.to_numpy(float),exit_plan=plan,config=ec)
        f['origin_index']=origins.origin_index.to_numpy()[f.origin_index.to_numpy(int)]
        f['plan']=plan_id;f['date']=date;f['split']=split
        frames.append(f)
    outcomes=pd.concat(frames,ignore_index=True)
    outcomes.to_parquet(out/'outcomes.parquet',index=False)
    receipt=dict(status='COMPLETE',date=date,split=split,route=config['route'],symbol=config['symbol'],
        source=str(p),manifest_sha256=digest(mp),payloads={k:mm['payloads'][k] for k in ('bars.parquet','quotes.parquet')},
        selection_available_ns=selection_ns,quote_rows=len(q),bars=len(bars),ready_bars=int(features.ready.sum()),
        proposed=len(all_candidates),quality_rejected=int((~mask).sum()),candidates=len(candidates),origins=len(origins),
        outcome_statuses={str(k):int(v) for k,v in outcomes.status.value_counts().items()},
        elapsed_seconds=time.perf_counter()-started,
        artifact_sha256={f.name:digest(f) for f in sorted(out.glob('*.parquet'))})
    dump(out/'receipt.json',receipt)
    return receipt
