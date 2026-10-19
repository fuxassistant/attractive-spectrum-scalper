"""Prepare compact, resumable basket inputs, sharing audited canonical quotes."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time,shutil
import numpy as np
import pandas as pd
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from spectrum.player_v3.data import minute_values,BASE_NAMES
from .data import load_session,verify_source,VERSION,MonthDay

DEFAULT='data/derived/basket_multimonth_v1'


def prepare_one(root,receipts,reference_path,date):
    dest=Path(root)/date
    if (dest/'manifest.json').exists():
        d=MonthDay(dest,verify=True)
        return dict(date=date,manifest_sha256=sha(dest/'manifest.json'),diagnostics=read_json(dest/'diagnostics.json'),reused=True)
    if dest.exists():raise ValueError('Unpublished partial directory must be reviewed: '+str(dest))
    choice=next(r['entry']['selection'] for r in receipts if r['entry']['selection']['date']==date)
    cur=load_session(receipts,date,choice['symbol'],np.load(reference_path,mmap_mode='r'))
    if cur is None:raise ValueError('No supported aggregate session '+date)
    s,b,f,q=(cur[n] for n in ('seconds','bars','features','quotes'))
    if np.any(np.diff(b.t_ns.to_numpy())<=0):raise ValueError('Duplicate/unsorted minute endpoints')
    if np.any(np.diff(q['quote_t'])<0):raise ValueError('Quote ordering regression')
    x,dt,legal=minute_values(b,f,cur['start'],cur['end'])
    legal&=(dt>=cur['london'])&(dt<cur['ny'])&(dt>=int(choice['selection_available_ns']))
    ix=np.searchsorted(q['quote_t'],dt,side='right')-1
    mid=np.zeros(len(dt));known=ix>=0
    mid[known]=(q['bid'][ix[known]]/1e9+q['ask'][ix[known]]/1e9)/2
    arrays=dict(x=x,decision_t=dt,bar_t=b.t_ns.to_numpy(np.int64),can_enter=legal,
        decision_mid=mid,cuts=ix+1,execution_valid=q['valid'],execution_segment=q['epoch'].astype(np.int32))
    if shutil.disk_usage(Path(root).parent).free<10*1024**3+sum(a.nbytes for a in arrays.values()):raise RuntimeError('DISK_RESERVE_REACHED')
    active=(q['quote_t']>=cur['london'])&(q['quote_t']<cur['ny']);qt=q['quote_t'][active]
    gaps=np.diff(qt);large=np.flatnonzero(gaps>5*10**9)
    diag=dict(quotes=len(qt),invalid_publications=int((~q['valid'][active]).sum()),
        invalid_times=[str(int(t)) for t in qt[~q['valid'][active]]],
        max_pause_seconds=float(gaps.max()/1e9) if len(gaps) else None,pauses_over_5s=len(large),
        long_pauses=[dict(start_ns=str(int(qt[i])),end_ns=str(int(qt[i+1])),seconds=float(gaps[i]/1e9)) for i in large],
        legal_minute_origins=int(legal.sum()),expected_activity_minutes=int((cur['ny']-cur['london'])/(60*10**9)),
        observed_activity_minutes=len(np.unique(qt//(60*10**9))),source_condition=choice.get('source_condition'),symbol=choice['symbol'],
        quote_order=True,mode='Observed quote silence is diagnostic; support transitions come from precomputed validity/segments')
    dest.mkdir(parents=True)
    for name,value in arrays.items():np.save(dest/(name+'.npy'),value)
    save_json(dest/'diagnostics.json',diag)
    meta=dict(status='COMPLETE',backing=VERSION,input_layout='BASKET_M1_ONLY_V1',support_contract='FINALIZED_SEGMENTS_V2',
        date=date,symbol=choice['symbol'],initial_capital=1500,quote_parts=cur['quote_parts'],
        start_ns=str(cur['start']),end_ns=str(cur['end']),london_ns=str(cur['london']),ny_close_ns=str(cur['ny']),
        liquidation_deadline_ns=str(cur['end']-30*10**9),feature_names=BASE_NAMES,rows=len(x),quotes=len(q['quote_t']),
        files={n+'.npy':dict(bytes=(dest/(n+'.npy')).stat().st_size,sha256=sha(dest/(n+'.npy'))) for n in arrays})
    save_json(dest/'manifest.json',meta)
    return dict(date=date,manifest_sha256=sha(dest/'manifest.json'),diagnostics=diag,reused=False)


def prepare(market='data/derived/full_mbo_v1',output=DEFAULT,workers=3):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    manifest=read_json(Path(market)/'manifest.json');config=read_json(Path(market)/'run_config.json')
    receipts=sorted([r for r in manifest['captures'] if r['status']=='COMPLETE'],key=lambda r:(r['entry']['selection']['date'],r['key']))
    verification=root/'source_verification.json'
    if not verification.exists():
        print('Verifying raw members and canonical payloads once',flush=True)
        verify_source(receipts,config,root)
    else:
        seal=read_json(root/'preparation_contract.json')
        if seal['source_manifest_sha256']!=sha(Path(market)/'manifest.json') or seal['verification_sha256']!=sha(verification):raise ValueError('Source verification drift')
    if sha(config['reference']['path'])!=config['reference']['sha256']:raise ValueError('Reference changed')
    dates=sorted({r['entry']['selection']['date'] for r in receipts if not r['entry']['selection']['context_only'] and pd.Timestamp(r['entry']['selection']['date']).weekday()<5})
    months=sorted({d[:7] for d in dates})
    save_json(root/'preparation_contract.json',dict(source_manifest_sha256=sha(Path(market)/'manifest.json'),verification_sha256=sha(verification),
        date_count=len(dates),months=months,workers=workers,initial_capital=1500,layout='BASKET_M1_ONLY_V1',reserve_gib=10,
        policy='No duplicated quote tape; aggregate clocks and support masks materialized once. Existing August mixed-memory data retained.'))
    bymonth={month:[] for month in months};failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(prepare_one,root/date[:7],receipts,config['reference']['path'],date):date for date in dates}
        for future in as_completed(jobs):
            date=jobs[future]
            try:
                row=future.result();bymonth[date[:7]].append(row)
                print(f'Prepared {date} ({sum(map(len,bymonth.values()))}/{len(dates)})',flush=True)
            except Exception as exc:
                failures.append(dict(date=date,error=str(exc)));print(f'FAILED {date}: {exc}',flush=True)
            save_json(root/'progress.json',dict(status='PREPARING',prepared=sum(map(len,bymonth.values())),expected=len(dates),failures=failures,seconds=time.perf_counter()-started))
    catalogs=[]
    for month,rows in bymonth.items():
        rows.sort(key=lambda r:r['date']);expected=[str(d.date()) for d in pd.bdate_range(month+'-01',pd.Timestamp(month+'-01')+pd.offsets.MonthEnd())]
        absent=[d for d in expected if d not in {r['date'] for r in rows}]
        exceptions=[dict(date=r['entry']['selection']['date'],key=r['key'],reasons=r.get('exclusion_reasons')) for r in manifest['captures'] if r['status']!='COMPLETE' and r['entry']['selection']['date'].startswith(month)]
        output_manifest=dict(status='COMPLETE',month=month,coverage='AVAILABLE_DAYS_ONLY' if absent else 'ALL_WEEKDAYS_PRESENT',missing_weekdays=absent,
            source_exclusions=exceptions,days=rows,initial_capital=1500,split=dict(train=[],validation=[],holdout=[r['date'] for r in rows]),
            input_layout='BASKET_M1_ONLY_V1',source_verification=str(verification),source_manifest_sha256=sha(Path(market)/'manifest.json'))
        if any(f['date'].startswith(month) for f in failures):output_manifest['status']='PARTIAL_PREPARATION_FAILURE'
        (root/month).mkdir(exist_ok=True);save_json(root/month/'manifest.json',output_manifest)
        catalogs.append(dict(month=month,status=output_manifest['status'],coverage=output_manifest['coverage'],days=len(rows),missing_weekdays=absent,path=str(root/month)))
    result=dict(status='COMPLETE' if not failures else 'PARTIAL',months=catalogs,failures=failures,seconds=time.perf_counter()-started,
        bytes_added=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),free_bytes=shutil.disk_usage(root).free)
    save_json(root/'catalog.json',result);save_json(root/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1');p.add_argument('--output',default=DEFAULT);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();prepare(a.market,a.output,a.workers)
