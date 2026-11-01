"""Reproducible v2 week: verified file handoffs, overlap workers, CME stores.

Run prepare once, then run NAME --workers N. Every destination is immutable.
The 3606s overlap is an explicit dependency, never an independent daily reset.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'feature_week_v1'))
from extract_context import source_for,CATALOG,ROUTE
from profile import exact_moments,write_frame
import numpy as np
import pandas as pd
import psutil
from spectrum.io import save_json,sha
from spectrum.core import StreamConfig
from spectrum.core.features import build_features
from spectrum.core.schema import COUNTERS
from spectrum.market.source import DbnSource
from spectrum.market.ingest import PACKED,packed_records
from spectrum.market.continuity import verified_handoff
from spectrum.features.pipeline import compute,implementation_identity
from spectrum.representation.representation import make_fast,make_minutes,ledger_releases,NS
from spectrum.representation.session import session_bounds,session_context
from spectrum.representation.session_store import publish_session_store,open_session_store
from spectrum.representation.zones import zone_and_event_tables

OUT=ROOT/'data/derived/feature_week_v2'
PREP=OUT/'prepared'
DATES=('2026-08-20','2026-08-21','2026-08-23','2026-08-24','2026-08-25','2026-08-26','2026-08-27','2026-08-28')
WEEK=DATES[3:]
FAMILIES=('motion','flow','spectral','depth_balance','volatility_vwap_proxy','volatility_vwap_exact','spectral_activity')
HALO=3606*NS


def raw_for(date):
    for root in (OUT/'raw',ROOT/'data/derived/feature_week_v1/raw',ROOT/'data/derived/market_representation_v1/raw'):
        if (root/date/'manifest.json').exists():return root/date
    raise FileNotFoundError(date)


def read(path):return json.loads(Path(path).read_text())


def prefix_records(item,entry):
    offset=0
    for block in DbnSource(item['archive'],item['member']):
        ids=np.flatnonzero((block['publisher_id']==ROUTE[0])&(block['instrument_id']==ROUTE[1])&(block['channel_id']==ROUTE[2]))
        ids=ids[(ids+offset>=entry['first_ordinal'])&(ids+offset<=entry['last_ordinal'])]
        p=np.empty(len(ids),PACKED);p['source_ordinal']=ids+offset;p['record']=block[ids]
        yield from packed_records(p,ROUTE)
        offset+=len(block)


def prepare():
    PREP.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    save_json(PREP/'manifest.json',dict(status='RUNNING',pid=os.getpid()))
    seconds={};bars={};sources={};maps={};offsets={};proofs={};checks={}
    config=read(CATALOG/'run_config.json')
    try:
        for date in DATES:
            s=source_for(date);market=Path(s['market']);raw=raw_for(date)
            mm,rm=read(market/'manifest.json'),read(raw/'manifest.json')
            if mm['status']!='COMPLETE' or rm['status']!='COMPLETE' or rm['source']['partition_sha256']!=s['entry']['sha256']:
                raise ValueError('Incomplete or unrelated source '+date)
            if sha(market/'manifest.json')!=s['market_manifest_sha256']:raise ValueError('Catalog receipt drift')
            for base,manifest in ((market,mm['payloads']),(raw,rm['outputs'])):
                for name,digest in manifest.items():
                    if sha(base/name)!=digest:raise ValueError('Source drift '+str(base/name))
                    checks[str(base/name)]=digest
            seconds[date]=pd.read_parquet(market/'seconds.parquet');bars[date]=pd.read_parquet(market/'bars.parquet')
            for f in (seconds[date],bars[date]):f['source_epoch']=f.epoch
            seconds[date]['ledger_complete']=True
            sources[date]=s
            maps[date]={int(x):int(x) for x in seconds[date].epoch.unique()}
            offsets[date]={int(x):{ 'c_'+n:0. for n in COUNTERS} for x in seconds[date].epoch.unique()}
        for prev,date in zip(DATES,DATES[1:]):
            boundary=int(pd.Timestamp(date,tz='UTC').value)
            if boundary-int(seconds[prev].t_ns.iloc[-1])>30*NS:continue
            s=sources[date];item=next(x for x in config['members'] if x['key']==s['entry']['member_key'])
            with ZipFile(item['archive']) as archive,archive.open(item['member']) as stream:
                h=hashlib.file_digest(stream,'sha256').hexdigest()
            if h!=item['sha256']:raise ValueError('Original DBN member drift')
            cp=Path(sources[prev]['market'])/'checkpoint.json'
            fragment=int(Path(s['market']).name.split('_')[-1])+1
            iterator=prefix_records(item,s['entry'])
            try:
                hand=verified_handoff(read(cp),iterator,next_config=StreamConfig(*ROUTE,snapshot_ns=NS,fragment=fragment),
                                      boundary_ns=boundary,prefix_end_ns=boundary+65*NS)
            finally:iterator.close()
            proof=hand.proof;old,new=proof['previous_epoch'],proof['next_epoch']
            logical=maps[prev][old];prior_offset=offsets[prev][old]
            maps[date][new]=logical
            offsets[date][new]={n:prior_offset[n]+proof['counter_offsets'][n] for n in prior_offset}
            for f in (seconds[date],bars[date]):
                f.loc[f.source_epoch==new,'epoch']=logical
            for name,value in offsets[date][new].items():seconds[date].loc[seconds[date].source_epoch==new,name]+=value
            for kind,patch,bank in (('seconds',hand.seconds,seconds),('bars',hand.bars,bars)):
                patch['source_epoch']=np.where(patch.t_ns<=boundary,old,new)
                patch['epoch']=logical
                if kind=='seconds':
                    for name,value in prior_offset.items():patch[name]+=value
                    patch['ledger_complete']=patch.t_ns!=boundary
                older=patch[patch.t_ns<=boundary];newer=patch[patch.t_ns>boundary]
                bank[prev]=pd.concat([bank[prev][~bank[prev].t_ns.isin(older.t_ns)],older],ignore_index=True).sort_values('t_ns').reset_index(drop=True)
                bank[date]=pd.concat([bank[date][~bank[date].t_ns.isin(newer.t_ns)],newer],ignore_index=True).sort_values('t_ns').reset_index(drop=True)
            proof.update(previous_date=prev,date=date,previous_checkpoint_sha256=sha(cp),next_member_sha256=h,
                         previous_partition_sha256=sources[prev]['entry']['sha256'],next_partition_sha256=s['entry']['sha256'],logical_epoch=logical)
            proofs[date]=proof
            print(json.dumps(dict(stage='handoff',date=date,bootstrap=proof['bootstrap_records'],recovered=proof['endpoint_recovered'])),flush=True)
        outputs={}
        for i,date in enumerate(DATES):
            dest=PREP/date;dest.mkdir()
            s=seconds[date];q=pd.read_parquet(Path(sources[date]['market'])/'quotes.parquet',columns=['t_ns','available_ns'])
            # A verified next capture supplies a real publication frontier.
            if i+1<len(DATES) and DATES[i+1] in proofs:
                nxt=DATES[i+1]
                if proofs[nxt]['previous_date']==date:
                    q2=pd.read_parquet(Path(sources[nxt]['market'])/'quotes.parquet',columns=['t_ns','available_ns'],
                                       filters=[('t_ns','<=',proofs[nxt]['boundary_ns']+NS)])
                    q=pd.concat([q,q2],ignore_index=True)
            release=ledger_releases(s,q)
            s['ledger_release_ns']=release
            s['trade_complete']=release<np.iinfo(np.int64).max
            s=exact_moments(s,pd.read_parquet(raw_for(date)/'trades.parquet'),release)
            if (release==np.iinfo(np.int64).max).any():raise ValueError('Unreleased endpoint')
            files={n:write_frame(f,dest/n) for n,f in [('seconds.parquet',s),('bars.parquet',bars[date])]}
            outputs[date]=files
        result=dict(status='COMPLETE',version='WEEK_V2_PREPARATION',dates=DATES,files=outputs,handoffs=proofs,
                    epoch_maps=maps,counter_offsets=offsets,source_hashes=checks,elapsed_seconds=time.perf_counter()-start,
                    partial_ledger_policy='UTC tail price/T complete; old sparse level tail absent and explicitly masked',
                    calendar_holidays='NOT_COVERED; ordinary-session fixture only')
        save_json(PREP/'manifest.json',result);return result
    except Exception as exc:
        save_json(PREP/'manifest.json',dict(status='FAILED',error=repr(exc)));raise


def window_raw(date,name,start,end):
    parts=[]
    for d in DATES:
        lo=int(pd.Timestamp(d,tz='UTC').value);hi=lo+86400*NS
        if hi<start or lo>end:continue
        parts.append(pd.read_parquet(raw_for(d)/(name+'.parquet'),filters=[('t_ns','>=',start),('t_ns','<=',end)]))
    return pd.concat(parts,ignore_index=True)


def generate_day(date,run):
    start=time.perf_counter();dest=Path(run)/'days'/date;dest.mkdir(parents=True,exist_ok=False)
    save_json(dest/'manifest.json',dict(status='RUNNING',pid=os.getpid()))
    f=pd.read_parquet(PREP/date/'seconds.parquet');first=int(f.t_ns.iloc[0]);end=int(f.t_ns.iloc[-1])
    i=DATES.index(date)
    if i:
        left=pd.read_parquet(PREP/DATES[i-1]/'seconds.parquet',filters=[('t_ns','>=',first-HALO)])
        f=pd.concat([left,f],ignore_index=True)
    if not f.t_ns.is_unique or not f.t_ns.is_monotonic_increasing:raise ValueError('Overlap clock')
    # The sparse interaction collector starts fresh in each UTC file. Its
    # two-second refill memory is not restored by a canonical book checkpoint.
    since_midnight=f.t_ns.to_numpy()%(86400*NS)
    f['ledger_complete'] &= ~((since_midnight>0)&(since_midnight<=2*NS))
    lo=int(f.t_ns.iloc[0])-NS
    levels=window_raw(date,'levels_1s',lo,end)
    trades=window_raw(date,'trades',lo,end-1)
    fast=make_fast(f,levels,trades)
    del levels,trades;gc.collect();prepared=time.perf_counter()
    c=compute(f,FAMILIES);calculated=time.perf_counter()
    columns={k:f[k].to_numpy() for k in ('t_ns','available_ns','epoch')}
    columns.update(admissible=c.admitted,scheduled_closed=c.closed)
    for name in c.values:columns[name]=c.values[name];columns[name+'__known']=c.masks[name]
    features=pd.DataFrame(columns)
    payloads={n:write_frame(data[data.t_ns>=first].reset_index(drop=True),dest/n)
              for n,data in [('features.parquet',features),('fast.parquet',fast)]}
    result=dict(status='COMPLETE',date=date,rows=int((f.t_ns>=first).sum()),features=len(c.values),families=FAMILIES,
                units=c.units,files=payloads,implementation_sha256=implementation_identity(),halo_seconds=HALO/NS,
                seconds=dict(prepare=prepared-start,formulas=calculated-prepared,total=time.perf_counter()-start),
                working_set_peak=getattr(psutil.Process().memory_info(),'peak_wset',None))
    save_json(dest/'manifest.json',result);return result


def assemble(run):
    start=time.perf_counter();run=Path(run);prep=read(PREP/'manifest.json')
    fast=pd.concat([pd.read_parquet(run/'days'/d/'fast.parquet') for d in DATES],ignore_index=True)
    trades=pd.concat([pd.read_parquet(raw_for(d)/'trades.parquet') for d in DATES],ignore_index=True)
    seconds=pd.concat([pd.read_parquet(PREP/d/'seconds.parquet') for d in DATES],ignore_index=True)
    bars=pd.concat([pd.read_parquet(PREP/d/'bars.parquet') for d in DATES],ignore_index=True)
    bars['source_available_ns']=bars.available_ns
    indices=np.searchsorted(seconds.t_ns.to_numpy(),bars.t_ns.to_numpy(),side='right')-1
    dependency=np.where(indices>=0,seconds.available_ns.to_numpy()[np.maximum(indices,0)],0)
    bars['available_ns']=np.maximum.accumulate(np.maximum(bars.available_ns,dependency))
    origins,values,manifest=build_features(seconds,bars,snapshot_ns=NS)
    origins=origins.rename(columns={n:'origin_'+n for n in origins if n in manifest['feature_names']})
    core=pd.concat([origins.reset_index(drop=True),pd.DataFrame(values,columns=manifest['feature_names'])],axis=1)
    for n in manifest['feature_names']:
        if n.startswith('previous_week_'):core[n]=0.
    manifest['previous_week_policy']='Unknown: preceding ISO week is incomplete in this fixture'
    write_frame(core,run/'core_m1.parquet');save_json(run/'core_m1_manifest.json',manifest)
    del seconds,origins,values;gc.collect()
    minutes=make_minutes(fast,trades,bars);write_frame(minutes,run/'minutes.parquet')
    inventory=[]
    for i,date in enumerate(WEEK):
        previous='2026-08-21' if i==0 else WEEK[i-1]
        lo,hi=session_bounds(date);context=session_context(minutes,trades,date,previous)
        local=fast[(fast.t_ns>lo-900*NS)&(fast.t_ns<=hi)].copy()
        levels=window_raw(date,'levels_1s',lo,hi)
        zones,events=zone_and_event_tables(local,context,levels,date,'GCZ6',start_ns=lo,end_ns=hi)
        del levels
        pieces=[];quotes=[]
        for d in DATES:
            dlo=int(pd.Timestamp(d,tz='UTC').value)
            if dlo+86400*NS<lo or dlo>hi:continue
            pieces.append(pd.read_parquet(run/'days'/d/'features.parquet',filters=[('t_ns','>=',lo-900*NS),('t_ns','<=',hi)]))
            path=Path(source_for(d)['market'])/'quotes.parquet'
            q=dict(path=str(path),sha256=sha(path),epoch_map=prep['epoch_maps'][d])
            if d in prep['handoffs']:
                p=prep['handoffs'][d];q.update(skip_bootstrap_through_ordinal=p['bootstrap_last_source_ordinal'],
                    handoff_version=p['version'],book_sha256=p['book_sha256'],accepted_available_ns=p['accepted_available_ns'])
            quotes.append(q)
        features=pd.concat(pieces,ignore_index=True);del pieces
        tables=dict(fast=local,context=context,zones=zones,events=events,features=features,
                    core_features=core[(core.t_ns>=lo-900*NS)&(core.t_ns<=hi)].copy())
        dest=run/'stores'/date
        publish_session_store(dest,date=date,previous_date=previous,symbol='GCZ6',tables=tables,quote_sources=quotes,
            source_identity=dict(profile='FEATURE_WEEK_V2',preparation_sha256=sha(PREP/'manifest.json'),
                                 day_receipts={d:sha(run/'days'/d/'manifest.json') for d in DATES}))
        inventory.append(dict(date=date,context=context.attrs['context_contract'],fast_rows=len(local),
                              features=len(features),zones=len(zones),events=len(events)))
        del local,context,zones,events,features,tables;gc.collect()
        print(json.dumps(dict(stage='session',date=date)),flush=True)
    return dict(elapsed_seconds=time.perf_counter()-start,inventory=inventory)


def run_profile(name,workers,days_only=False):
    if read(PREP/'manifest.json')['status']!='COMPLETE':raise ValueError('Preparation incomplete')
    run=OUT/name;run.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    save_json(run/'manifest.json',dict(status='RUNNING',workers=workers,pid=os.getpid()))
    try:
        results=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job in as_completed([pool.submit(generate_day,d,str(run)) for d in DATES]):
                r=job.result();results.append(r)
                print(json.dumps(dict(stage='day',date=r['date'],features=r['features'],seconds=r['seconds'])),flush=True)
        day_seconds=time.perf_counter()-start
        context=None if days_only else assemble(run)
        result=dict(status='COMPLETE',version='FEATURE_WEEK_V2',workers=workers,dates=DATES,target_week=WEEK,
                    day_seconds=day_seconds,context=context,elapsed_seconds=time.perf_counter()-start,
                    days=sorted(results,key=lambda r:r['date']),preparation_sha256=sha(PREP/'manifest.json'))
        save_json(run/'manifest.json',result);return result
    except Exception as exc:
        save_json(run/'manifest.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','assemble']);p.add_argument('--name',default='parallel2')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--days-only',action='store_true');a=p.parse_args()
    if a.action=='prepare':r=prepare()
    elif a.action=='assemble':r=assemble(OUT/a.name)
    else:r=run_profile(a.name,a.workers,a.days_only)
    print(json.dumps({k:v for k,v in r.items() if k in ('status','elapsed_seconds','day_seconds')}),flush=True)
