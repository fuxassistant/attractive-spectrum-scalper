"""Hash-checked 25-day extension; reuse the old causal clock without its fits."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT, digest
from experiments.mechanical_lab.signals import build_features
from experiments.zone_episode_lab.episodes import build_episodes, occupancy_grid, NS
from experiments.zone_episode_lab.candidates import make_candidates
from experiments.zone_conditional_v1.features import make_features
from experiments.zone_conditional_v1.mbo_features import global_mbo_features, MBO_FEATURE_COLUMNS, BASE_CONTEXT_COLUMNS
from experiments.hypothesis_atlas_v1.engine import decision_cohort, causal_paths
from experiments.loser_lab_v1.targets import make_targets, post_cost_pairs

HERE = Path(__file__).resolve().parent


def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple,np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,float)): return float(obj) if np.isfinite(obj) else None
    return obj


def dump(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def check(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def prepare_day(source, index, config, destination, reuse=None):
    started=time.perf_counter(); selection=source['entry']['selection']; date=selection['date']
    out=Path(destination)/'days'/date; receipt_path=out/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['config_sha256'] != digest(Path(destination)/'config.json'):
            raise ValueError('Resume configuration changed')
        for name, sha in receipt['artifacts'].items(): check(out/name, sha)
        for name, sha in receipt['source_hashes'].items(): check(Path(name), sha)
        return receipt
    out.mkdir(parents=True, exist_ok=True)
    market=Path(source['market']); hashes={str(market/'manifest.json'):check(market/'manifest.json',source['market_manifest_sha256'])}
    mm=json.loads((market/'manifest.json').read_text()); assert mm['status']=='COMPLETE'
    for name in ('quotes.parquet','bars.parquet','seconds.parquet'):
        hashes[str(market/name)]=check(market/name,mm['payloads'][name])
    q=pd.read_parquet(market/'quotes.parquet'); coverage=int(mm['input']['end_ns'])+1
    selection_ns=int(selection['selection_available_ns']); day=int(pd.Timestamp(date,tz='UTC').value)
    if reuse is None:
        cfg=json.loads((ROOT/'experiments/zone_episode_lab/config.json').read_text())
        bars=build_features(pd.read_parquet(market/'bars.parquet')); bars['ready'] &= bars.t_ns>=selection_ns
        ev, ep=build_episodes(q,end_ns=coverage,**cfg['geometry'])
        grid=occupancy_grid(q,start_ns=day,end_ns=day+86400*NS,bin_ticks=cfg['geometry']['bin_ticks'],sample_seconds=cfg['spectral']['sample_seconds'])
        c, _, tokens, spec=make_candidates(q,bars,ev,ep,grid,cfg,index,date)
        c=decision_cohort(c.loc[~c.inside],seconds=300)
        for table in (ev,tokens,spec): table['date']=date
        f=make_features(c,bars,ev,tokens,spec)
        mbo=global_mbo_features(c,pd.read_parquet(market/'seconds.parquet'),q,selection_available_ns=selection_ns,coverage_end_ns=coverage,window_seconds=30)
        renamed={k:'x_mbo_'+k for k in MBO_FEATURE_COLUMNS}; renamed.update({k:'x_base_'+k for k in BASE_CONTEXT_COLUMNS})
        f=f.merge(mbo.rename(columns=renamed),on='candidate_id',validate='one_to_one')
        f=f.merge(causal_paths(q,f),on='candidate_id',validate='one_to_one')
        counts=dict(quote_rows=len(q),episodes=len(ep),events=len(ev))
    else:
        atlas=ROOT/config['old_atlas']; am=json.loads((atlas/'manifest.json').read_text(encoding='utf-8'))
        hashes[str(atlas/'observations.parquet')]=check(atlas/'observations.parquet',am['artifacts']['observations.parquet'])
        f=pd.read_parquet(atlas/'observations.parquet',filters=[('date','==',date)])
        # No learned atlas flags, old outcome, matching-bin or placebo information
        # may enter the new fit. Keep only explicitly whitelisted model inputs.
        counts=dict(quote_rows=len(q),reused_origin_rows=len(f))
    f['symbol']=selection['symbol']; f['capture_key']=source['key']; f['route']=json.dumps(selection['route'])
    f['source_population']='additional' if reuse is None else 'original'; f['source_selection_ns']=selection_ns
    avail=[c for c in ('bar_context_available_ns','memory_available_ns','spectrum_available_ns','past_max_available_ns','mbo_now_available_ns') if c in f]
    for col in avail:
        if (f[col]>f.t_ns).any(): raise AssertionError(f'Future input {col}')
    if (f.t_ns<selection_ns).any(): raise AssertionError('Unselected contract')
    ys=[]; costs=[]
    for spec in config['targets']:
        y=make_targets(q,f,spec,coverage=coverage,config=config); ys.append(y)
        costs.append(post_cost_pairs(q,y,config,coverage=coverage))
    tables={'features':f,'targets':pd.concat(ys,ignore_index=True),'post_cost':pd.concat(costs,ignore_index=True)}
    for name,table in tables.items(): table.to_parquet(out/(name+'.parquet'),index=False)
    r=dict(date=date,symbol=selection['symbol'],capture_key=source['key'],source=str(market),selection_ns=selection_ns,
           coverage_end_ns=coverage,rows=len(f),known_targets=int(tables['targets'].known.sum()),target_rows=len(tables['targets']),
           source_hashes=hashes,config_sha256=digest(Path(destination)/'config.json'),**counts,
           artifacts={p.name:digest(p) for p in out.glob('*.parquet')},elapsed_seconds=time.perf_counter()-started)
    dump(receipt_path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--dates',nargs='*'); args=ap.parse_args(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    config=json.loads((HERE/'config.json').read_text()); frozen=out/'config.json'
    if frozen.exists() and json.loads(frozen.read_text())!=config: raise ValueError('Immutable run configuration changed')
    dump(frozen,config)
    catalog=ROOT/'data/derived/full_mbo_v1'; manifest=json.loads((catalog/'manifest.json').read_text())
    admission={c['key']:c for c in json.loads((catalog/'training_catalog.json').read_text())['captures']}
    added=[str(d.date()) for d in pd.bdate_range(config['additional_start'],config['additional_end'])]
    old=pd.read_parquet(ROOT/config['old_atlas']/'observations.parquet',columns=['date']).date.unique().tolist()
    wanted=added+sorted(old); jobs=[]
    for i,date in enumerate(wanted):
        if args.dates and date not in args.dates: continue
        captures=[c for c in manifest['captures'] if c['status']=='COMPLETE' and c['entry']['selection']['date']==date and admission.get(c['key'],{}).get('eligible')]
        if len(captures)!=1: raise ValueError(f'Ambiguous or inadmissible date {date}')
        jobs.append((captures[0],1001+i,config,str(out),True if date in old else None))
    dump(out/'inventory.json',dict(added_dates=added,original_dates=sorted(old),source_gaps=manifest['source_gaps'],
        catalog_sha256=digest(catalog/'manifest.json'),admission_sha256=digest(catalog/'training_catalog.json'),
        selection_rule='All 25 weekdays of latest five complete calendar weeks before the incomplete July27 week; no outcome selection.',
        study_omission='July27-30 available but outside the selected five weeks; July31 missing. July3 ends at17:00 UTC.',
        sources=[dict(date=j[0]['entry']['selection']['date'],key=j[0]['key'],symbol=j[0]['entry']['selection']['symbol']) for j in jobs]))
    receipts=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare_day,*job) for job in jobs]
        for fut in as_completed(futures):
            r=fut.result(); receipts.append(r); print(json.dumps({k:r[k] for k in ('date','symbol','rows','known_targets','target_rows','elapsed_seconds')}),flush=True)
    if args.dates: return
    receipts.sort(key=lambda r:r['date'])
    for name in ('features','targets','post_cost'):
        frame=pd.concat([pd.read_parquet(out/'days'/r['date']/(name+'.parquet')) for r in receipts],ignore_index=True)
        frame.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'manifest.json',dict(status='COMPLETE_RESEARCH_INPUT',days=len(receipts),added_days=len(added),receipts=receipts,
        implementation={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*.py')},
        artifacts={p.name:digest(p) for p in out.glob('*.parquet')}))


if __name__=='__main__': main()
