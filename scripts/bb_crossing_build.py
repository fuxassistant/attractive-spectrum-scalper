"""Reproducible full-publication BB bank and paired first-crossing trade profiles."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from scripts.flow_context_coverage import sha, save

DATA=Path('data/derived/bb_crossing_bank_v1')
REPORT=Path('reports/bb_crossing_bank_v1')
SOURCE=Path('data/derived/flow_context_v2')
MARKET=Path('data/derived/basket_multimonth_v1')


def identity():
    return {str(p):sha(p) for p in [Path('contracts/bb_crossing_bank_v1.json'),Path(bb.__file__),Path(__file__)]}


def profiles(q, bank, tape, roots):
    t=roots.t_ns.to_numpy(np.int64); k=tape['k']; blocks=bb.extremum_blocks(q.bid,q.ask,k)
    br=bb.execution_breaks(q.quote_t,q.valid,q.epoch); frames=[]
    for s in bb.SCALES:
        p=bb.packets(bank,s)
        ints, vals=bb.trade_profile(q.quote_t,q.bid,q.ask,q.valid,q.epoch,br,k,blocks,t,
            p.t_ns.to_numpy(np.int64),p.ordinal.to_numpy(np.int64),q.deadline_ns)
        f=pd.DataFrame(dict(t_ns=t,scale=s,entry_ordinal=ints[:,0],exit_ordinal=ints[:,1],
            cross_ordinal=ints[:,2],execution_status=ints[:,3],request_ns=ints[:,4]))
        e=ints[:,0]; x=ints[:,1]; c=ints[:,2]; eok=e>=0; cok=c>=0
        f['entry_ns']=np.where(eok,q.quote_t[np.maximum(e,0)],0)
        f['exit_ns']=np.where(x>=0,q.quote_t[np.maximum(x,0)],0)
        f['cross_ns']=np.where(cok,q.quote_t[np.maximum(c,0)],0)
        eg=np.searchsorted(tape['t'],f.entry_ns,side='right')-1
        g=tape['geometries'][s]
        origin_ready=np.isfinite(roots[f'bb{s}_position'])
        entry_ready=(eg>=0)&(g['support'][np.maximum(eg,0)]>=0)
        ge=bb.support_ends(tape['t'],g['support'],q.coverage_end_ns)[np.maximum(eg,0)]
        geometric_break=eok&(~entry_ready | (f.request_ns.to_numpy()>=ge))
        f.loc[geometric_break,'execution_status']=-1
        f.loc[~origin_ready,'execution_status']=-2
        f['decision_event_id']=roots[f'm{s}_id'].to_numpy()
        f['decision_event_before_entry']=(roots[f'm{s}_event_ns'].to_numpy()>0)&(roots[f'm{s}_event_ns'].to_numpy()<=f.entry_ns.to_numpy())&eok
        id_map=p.set_index('ordinal').packet_id
        f['cross_id']=pd.Series(c).map(id_map).fillna('')
        f['cross_after_decision_s']=np.where(cok,(f.cross_ns-t)/bb.NS,np.nan)
        f['cross_after_entry_s']=np.where(cok&eok,(f.cross_ns-f.entry_ns)/bb.NS,np.nan)
        f['entry_ask']=np.where(eok,q.ask[np.maximum(e,0)]/1e9,np.nan)
        f['entry_bid']=np.where(eok,q.bid[np.maximum(e,0)]/1e9,np.nan)
        fields=['mark_L','mark_S','net_L','net_S','cross_mae_L','cross_mae_S','exit_mae_L','exit_mae_S',
                'cross_mfe_L','cross_mfe_S','exit_mfe_L','exit_mfe_S']
        for j,name in enumerate(fields): f[name]=vals[:,j]
        # A price mark after lost geometry is not a known strategy endpoint.
        f.loc[geometric_break, ['mark_L','mark_S','net_L','net_S']]=np.nan
        f.loc[f.execution_status!=1,['net_L','net_S']]=np.nan
        f['termination']=np.where(f.execution_status==-2,'NO_ORIGIN_GEOMETRY',
            np.where(f.execution_status<0,'CENSORED',np.where(f.execution_status==0,'EMPTY',np.where(cok,'CROSS','TIME_OR_SESSION'))))
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def build_date(date, receipt, ids, checked):
    started=time.perf_counter(); source=SOURCE/(date+'.parquet')
    if sha(source)!=receipt['sha256']: raise ValueError('Changed source roots')
    day=MonthDay(MARKET/date[:7]/date,verify=True)
    for part in day.meta['quote_parts']:
        path=Path(part['root'])/'quotes.parquet'
        expected=part['quote_file']['sha256']
        if str(path) not in checked:
            if sha(path)!=expected: raise ValueError('Changed canonical quote '+str(path))
            checked[str(path)]=expected
    q=QuoteTape.from_day(day)
    source_id=bb.source_identity(day.meta)
    bank,tape=bb.build_bank(q,source_id)
    old=pd.read_parquet(source); old=old.loc[(old['mode']=='T')&(old.side>0)].reset_index(drop=True)
    target=bb.make_targets(bank,tape,old.t_ns.to_numpy(),q.deadline_ns,bb.contract()['target_horizon_seconds'])
    target['date']=date; target['symbol']=day.meta['symbol']
    for s in bb.SCALES:
        half=np.expm1(target[f'bb{s}_log_half_ticks'])*.1
        target[f'bb{s}_spread_half']=old.spread_ticks.to_numpy()*.1/half
    features=[c for c in old if c not in ('mode','side','status','category','net','busy_s','t_ns','date')]
    target=pd.concat([target,old[features]],axis=1)
    prof=profiles(q,bank,tape,target); prof['date']=date
    sensitivity={}
    if date in bb.contract()['sensitivity_dates']:
        bank0,tape0=bb.build_bank(q,bb.source_identity(day.meta,0),0)
        target0=bb.make_targets(bank0,tape0,target.t_ns.to_numpy(),q.deadline_ns)
        for key in ['m1','m3','m6','m9','global']:
            both=(target[key+'_y']>=0)&(target0[key+'_y']>=0)
            sensitivity[key]=dict(n=int(both.sum()),class_agreement=float((target.loc[both,key+'_y'].to_numpy()==target0.loc[both,key+'_y'].to_numpy()).mean()),
                exact_publication_agreement=float((target.loc[both,key+'_ordinal'].to_numpy()==target0.loc[both,key+'_ordinal'].to_numpy()).mean()))
        del bank0,tape0,target0
    if date in bb.contract()['visual_dates']:
        times=np.arange(q.entry_start_ns,min(q.entry_cutoff_ns,q.deadline_ns),5*bb.NS,dtype=np.int64)
        j=np.searchsorted(tape['t'],times,side='right')-1
        scene=pd.DataFrame(dict(t_ns=times,mid=tape['price'][j]))
        for s,g in tape['geometries'].items():
            scene[f'm{s}']=g['middle'][j];scene[f'h{s}']=g['half'][j]
        scene.to_parquet(DATA/(date+'_scene.parquet'),index=False)
    paths={}
    for suffix,frame in [('bank',bank),('roots',target),('profiles',prof)]:
        path=DATA/(date+'_'+suffix+'.parquet');frame.to_parquet(path,index=False);paths[suffix]=sha(path)
    counts={}
    for key in ['m1','m3','m6','m9','global']:
        y=target[key+'_y'];ids0=target.loc[target[key+'_event_ns']>0,key+'_id'];freq=ids0.value_counts()
        counts[key]=dict(known=int((y>=0).sum()),censored=int((y<0).sum()),classes={str(k):int(v) for k,v in y.value_counts().items()},
            distinct_next_events=len(freq),event_queries=len(ids0),max_queries_per_event=int(freq.max()) if len(freq) else 0)
    result=dict(date=date,source_id=source_id,geometry_definition=bb.geometry_definition(),identity=ids,source_sha256=sha(source),market_manifest_sha256=sha(day.path/'manifest.json'),
        files=paths,quotes=len(q.quote_t),reduced_quotes=len(tape['k']),events=len(bank),roots=len(target),counts=counts,
        causes={str(k):int(v) for k,v in bank.cause.value_counts().items()},sensitivity_ddof0=sensitivity,
        execution={str(k):int(v) for k,v in prof.execution_status.value_counts().items()},
        seconds=round(time.perf_counter()-started,2))
    save(DATA/(date+'.json'),result)
    print(json.dumps({k:result[k] for k in ('date','quotes','reduced_quotes','events','roots','seconds')}),flush=True)
    del q,day,bank,tape,old,target,prof;QUOTE_CACHE.clear();gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--date');args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'manifest.json').read_text());ids=identity();results=[];checked={}
    checkpoint=DATA/'checked_quotes.json'
    if checkpoint.exists(): checked=json.loads(checkpoint.read_text())
    for r in source['days']:
        date=r['date']
        if args.date and date!=args.date:continue
        cached=DATA/(date+'.json')
        if cached.exists():
            result=json.loads(cached.read_text())
            if result['identity']!=ids:raise ValueError('Changed build identity; preserve prior build before rerun')
            for suffix,expected in result['files'].items():
                if sha(DATA/(date+'_'+suffix+'.parquet'))!=expected:raise ValueError('Changed cache')
        else:
            result=build_date(date,r,ids,checked);save(checkpoint,checked)
        results.append(result)
    if not args.date:
        manifest=dict(status='COMPLETE',identity=ids,source_manifest_sha256=sha(SOURCE/'manifest.json'),
            days=results,excluded=source['excluded'],verified_quote_sha256=checked)
        save(DATA/'manifest.json',manifest)
        save(REPORT/'data_receipt.json',dict(status='COMPLETE',identity=ids,manifest_sha256=sha(DATA/'manifest.json'),
            days=len(results),roots=sum(r['roots'] for r in results),events=sum(r['events'] for r in results),
            quotes=sum(r['quotes'] for r in results),quote_files_hash_verified=len(checked),
            daily=[{k:r[k] for k in ('date','roots','events','counts','causes','execution','sensitivity_ddof0')} for r in results]))


if __name__=='__main__':main()
