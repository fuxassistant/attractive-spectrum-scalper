"""Month preparation with shared canonical quotes: no duplicate L3 or quote tape.

Original payloads are verified once at preparation. Runtime maps small prepared
arrays and reads only the current session's projected quote columns. Source book
validity, regular schedule and spot-reference presence remain separate.
"""
from pathlib import Path
from collections import OrderedDict
import argparse
import hashlib
import json
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.ingest import save_json
from spectrum.finalized import read_json
from spectrum.finalize_market import flags_for_times
from spectrum.core.storage import sha
from spectrum.player.contract import Contract, NS
from spectrum.lab import compute, Context
from spectrum.plugins.volatility_vwap import compute as compute_vwap
from spectrum.player_v3.data import (Day, bounds, minute_values, anchor_values,
    support_segment_starts, BASE_NAMES, FAST_BASE, FAST_NAMES, VWAP_NAMES, ANCHOR_VALUES, EVENT_NAMES)

VERSION='SHARED_CANONICAL_QUOTES_V1'
QUOTE_CACHE=OrderedDict()


def quote_arrays(parts,quality=True):
    arrays={n:[] for n in ('quote_t','source_t','bid','ask','source_ordinal')+ (('valid','epoch') if quality else ())}
    for number,part in enumerate(parts):
        root=Path(part['root']);spec=part['quote_file']
        if (root/'quotes.parquet').stat().st_size!=spec['bytes']:raise ValueError('Certified quote file changed size')
        q=pq.read_table(root/'quotes.parquet',columns=['available_ns','t_ns','bid_nano','ask_nano','valid','source_ordinal'],
            filters=[('available_ns','>=',part['start_ns']),('available_ns','<',part['end_ns'])]).to_pandas()
        t=q.available_ns.to_numpy(np.int64);valid=q.valid.to_numpy(bool)
        # Quality transitions are facts of the published book, not age thresholds.
        values=dict(quote_t=t,source_t=q.t_ns.to_numpy(np.int64),bid=q.bid_nano.to_numpy(np.int64),
            ask=q.ask_nano.to_numpy(np.int64),source_ordinal=q.source_ordinal.to_numpy(np.int64))
        if quality:
            segments=np.cumsum(np.r_[True,(~valid[:-1])&valid[1:]]).astype(np.int32)+number*1_000_000
            closed,_=flags_for_times(t,np.empty(0,np.int64))
            values.update(valid=valid&~closed,epoch=np.where(valid,segments,-1))
        for n in arrays:arrays[n].append(values[n])
    result={n:np.concatenate(v) for n,v in arrays.items()}
    result['ordinal']=np.arange(len(result['quote_t']),dtype=np.int64)
    return result


class MonthDay(Day):
    def __init__(self,path,verify=False):
        self.path=Path(path);self.meta=read_json(self.path/'manifest.json')
        self.cache_key=str(self.path.resolve())
        if self.meta.get('backing')!=VERSION or self.meta.get('status')!='COMPLETE':raise ValueError('Unpublished month input')
        for name,spec in self.meta['files'].items():
            p=self.path/name
            if p.stat().st_size!=spec['bytes'] or (verify and sha(p)!=spec['sha256']):raise ValueError('Prepared array changed')
            setattr(self,name[:-4],np.load(p,mmap_mode='r'))
        self.deadline=int(self.meta['liquidation_deadline_ns'])
        self.valid=self.execution_valid
        self.epoch=self.execution_segment

    def __getattr__(self,name):
        if name not in ('quote_t','source_t','bid','ask','valid','epoch','ordinal','source_ordinal'):raise AttributeError(name)
        key=self.cache_key
        if key not in QUOTE_CACHE:
            QUOTE_CACHE[key]=quote_arrays(self.meta['quote_parts'],quality=False)
            while len(QUOTE_CACHE)>2:QUOTE_CACHE.popitem(last=False)
        QUOTE_CACHE.move_to_end(key)
        return QUOTE_CACHE[key][name]


def load_session(receipts,date,symbol,reference,with_quotes=True):
    c=Contract(initial_capital=1500.,daily_drawdown_stop=float('inf'))
    start,end,london,ny=bounds(date,c)
    dates=(str(pd.Timestamp(start,tz='UTC').date()),date)
    selected=[r for r in receipts if r['entry']['selection']['date'] in dates and r['entry']['selection']['symbol']==symbol]
    sp=[];bp=[];parts=[]
    for number,r in enumerate(selected):
        root=Path(r['market'])
        s=pq.read_table(root/'seconds.parquet',filters=[('t_ns','>=',start),('t_ns','<',end)]).to_pandas()
        b=pq.read_table(root/'bars.parquet',filters=[('t_ns','>=',start),('t_ns','<',end)]).to_pandas()
        if not len(s):continue
        s['scheduled_closed'],s['reference_present']=flags_for_times(s.t_ns.to_numpy(),reference)
        s['valid']=s.book_valid.to_numpy(bool)
        # Counter resets in the existing aggregation retain their own epoch;
        # they are feature warmup boundaries, never execution support-loss rules.
        s['epoch']=s.epoch.to_numpy(np.int64)+number*1_000_000
        s['segment_id']=s.epoch
        sp.append(s);bp.append(b)
        parts.append(dict(root=str(root),key=r['key'],start_ns=start,end_ns=end,
            quote_file=dict(bytes=(root/'quotes.parquet').stat().st_size,sha256=r['market_payloads']['quotes.parquet']),
            source_sha256=r['entry']['sha256'],route=r['entry']['route']))
    if not sp or not bp:return None
    s=pd.concat(sp,ignore_index=True);b=pd.concat(bp,ignore_index=True)
    if not len(b):return None
    context=compute(s)
    cols=dict(t_ns=s.t_ns.to_numpy(),available_ns=s.available_ns.to_numpy(),admissible=context.admitted)
    for n,v in context.values.items():cols[n]=v;cols[n+'__known']=context.masks[n]
    result=dict(seconds=s,bars=b,features=pd.DataFrame(cols),start=start,end=end,london=london,ny=ny,quote_parts=parts)
    if with_quotes:result['quotes']=quote_arrays(parts)
    return result


def verify_source(receipts,config,output):
    """One administrative hash pass, including compressed original DBN members."""
    rows=[];members={r['entry']['member_key'] for r in receipts}
    prior=read_json('reports/data_properties_full_001/protocol.json')
    if sha(Path(config['root'])/'manifest.json')!=prior['input_manifest_sha256']:
        raise ValueError('Canonical manifest differs from prior complete source-identity audit')
    for item in config['members']:
        if item['key'] not in members:continue
        h=hashlib.sha256()
        with zipfile.ZipFile(item['archive']) as z,z.open(item['member']) as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        if h.hexdigest()!=item['sha256']:raise ValueError('Original raw member hash mismatch')
    for r in receipts:
        root=Path(r['market'])
        for name in ('quotes.parquet','seconds.parquet','bars.parquet'):
            if sha(root/name)!=r['market_payloads'][name]:raise ValueError('Canonical payload hash mismatch '+str(root/name))
            pq.ParquetFile(root/name).metadata
        audit_path=Path('reports/data_properties_full_001/market')/(r['key'].replace('/','_')+'.json')
        if not audit_path.exists():raise ValueError('Missing prior complete publication/source identity audit')
        audit=read_json(audit_path)
        if not all(audit['payload_hashes_match'].values()) or any(audit[k] for k in ('source_ordinal_nonincreasing','recv_regressions','publication_regressions','publication_before_source','valid_crossed_or_empty')):
            raise ValueError('Prior source audit was not clean')
        if audit['quote_rows']!=pq.ParquetFile(root/'quotes.parquet').metadata.num_rows:raise ValueError('Audit row-count drift')
        rows.append(dict(key=r['key'],source_member_sha256=next(i['sha256'] for i in config['members'] if i['key']==r['entry']['member_key']),
            payload_hashes=r['market_payloads'],prior_source_identity_audit=str(audit_path),prior_audit_sha256=sha(audit_path)))
    save_json(Path(output)/'source_verification.json',dict(status='PASS',captures=rows,scope='Original compressed member hashes and all canonical payload hashes/footer rechecked; prior all-publication raw identity audit retained'))


def prepare_day(root,receipts,reference,date):
    r=next(r for r in receipts if r['entry']['selection']['date']==date)
    choice=r['entry']['selection'];cur=load_session(receipts,date,choice['symbol'],reference)
    if cur is None:raise ValueError('No session '+date)
    prior=str((pd.Timestamp(date)-pd.offsets.BDay()).date())
    prev=load_session(receipts,prior,choice['symbol'],reference,False)
    s,b,f,q=(cur[n] for n in ('seconds','bars','features','quotes'))
    x,decision,legal=minute_values(b,f,cur['start'],cur['end'])
    legal&=(decision>=cur['london'])&(decision<cur['ny'])&(decision>=int(choice['selection_available_ns']))
    if choice['context_only']:legal[:]=False
    slow=x.copy();slow_t=b.t_ns.to_numpy();slow_av=decision.copy();slow_session=np.ones(len(x),np.int8);previous=None
    if prev:
        px,pav,_=minute_values(prev['bars'],prev['features'],prev['start'],prev['end'])
        slow=np.r_[px,x];slow_t=np.r_[prev['bars'].t_ns,slow_t];slow_av=np.r_[pav,slow_av];slow_session=np.r_[np.zeros(len(px),np.int8),slow_session]
        previous=(prev['seconds'],prev['bars'],prev['start'],prev['london'],prev['end'])
    ctx=Context(s,None);compute_vwap(ctx)
    fv=np.column_stack([f[n] for n in FAST_BASE]+[ctx.values[n] for n in VWAP_NAMES])
    fk=np.column_stack([f[n+'__known'] for n in FAST_BASE]+[ctx.masks[n] for n in VWAP_NAMES])
    ev=np.empty(0,np.int64)
    anchors,mid=anchor_values(s,b,decision,previous,cur['start'],cur['london'],date,prior,prev is not None,ev)
    fast_av=np.maximum.accumulate(s.available_ns.to_numpy(np.int64))
    arrays=dict(x=x,decision_t=decision,bar_t=b.t_ns.to_numpy(np.int64),can_enter=legal,
        execution_valid=q['valid'],execution_segment=q['epoch'].astype(np.int32),
        cuts=np.searchsorted(q['quote_t'],decision,side='right'),fast_x=np.c_[np.where(fk,fv,0.),fk].astype(np.float32),
        fast_t=s.t_ns.to_numpy(np.int64),fast_available=fast_av,fast_valid=s.valid.to_numpy(bool),fast_epoch=s.epoch.to_numpy(np.int64),
        fast_segment_start=support_segment_starts(s.t_ns.to_numpy(),s.valid.to_numpy(),s.epoch.to_numpy()),
        fast_cuts=np.searchsorted(fast_av,decision,side='right'),slow_x=slow,slow_t=slow_t,slow_available=slow_av,slow_session=slow_session,
        slow_cuts=np.searchsorted(slow_av,decision,side='right'),anchors=anchors,decision_mid=mid,
        event_x=np.empty((0,len(EVENT_NAMES)-3),np.float32),event_t=ev,event_available=ev,event_cuts=np.zeros(len(x),np.int64),event_prices=np.empty((0,2)))
    for n,a in arrays.items():
        if np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all():raise ValueError('Nonfinite '+n)
    if shutil.disk_usage(root).free<8*1024**3+sum(a.nbytes for a in arrays.values()):raise RuntimeError('DISK_RESERVE_REACHED')
    dest=Path(root)/date;dest.mkdir()
    for n,a in arrays.items():np.save(dest/(n+'.npy'),a)
    active=(q['quote_t']>=cur['london'])&(q['quote_t']<cur['ny'])
    qt=q['quote_t'][active];valid=q['valid'][active]
    gaps=np.diff(qt);indices=np.flatnonzero(gaps>5*NS)
    diagnostics=dict(quotes=len(qt),invalid_publications=int((~valid).sum()),pauses_over_5s=len(indices),max_pause_seconds=float(gaps.max()/NS) if len(gaps) else None,
        invalid_times=[str(t) for t in qt[~valid]],long_pauses=[dict(start_ns=str(qt[i]),end_ns=str(qt[i+1]),seconds=float(gaps[i]/NS)) for i in indices],
        legal_minute_origins=int(legal.sum()),expected_activity_minutes=int((cur['ny']-cur['london'])/(60*NS)),
        quote_order=bool(np.all(np.diff(q['quote_t'])>=0)),bid_ask_consistent=bool(np.all(q['ask'][q['valid']]>q['bid'][q['valid']])))
    if not diagnostics['quote_order'] or not diagnostics['bid_ask_consistent']:raise ValueError('Execution geometry failure')
    save_json(dest/'diagnostics.json',diagnostics)
    meta=dict(status='COMPLETE',backing=VERSION,support_contract='FINALIZED_SEGMENTS_V2',date=date,symbol=choice['symbol'],initial_capital=1500.,
        quote_parts=cur['quote_parts'],start_ns=str(cur['start']),end_ns=str(cur['end']),london_ns=str(cur['london']),ny_close_ns=str(cur['ny']),
        liquidation_deadline_ns=str(cur['end']-30*NS),feature_names=BASE_NAMES,slow_names=BASE_NAMES,fast_names=FAST_NAMES+[n+'__known' for n in FAST_NAMES],
        anchor_names=ANCHOR_VALUES+[n+'__known' for n in ANCHOR_VALUES],event_names=EVENT_NAMES,previous_available=prev is not None,previous_date=prior,
        event_bank_status='EMPTY_NO_CERTIFIED_AUGUST_EVENTS',legacy_context_note='Original counter epochs retained as feature warmup boundaries; execution support ignores age heuristics and OHLC reference',
        rows=len(x),quotes=len(q['quote_t']),entry_origins=int(legal.sum()),files={n+'.npy':dict(bytes=(dest/(n+'.npy')).stat().st_size,sha256=sha(dest/(n+'.npy'))) for n in arrays})
    save_json(dest/'manifest.json',meta)
    return dict(date=date,rows=len(x),entry_origins=int(legal.sum()),diagnostics=diagnostics,manifest_sha256=sha(dest/'manifest.json'))


def prepare(market,output,month='2026-08'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    config=read_json(Path(market)/'run_config.json');m=read_json(Path(market)/'manifest.json')
    receipts=[r for r in m['captures'] if r['status']=='COMPLETE' and r['entry']['selection']['date'].startswith(month)]
    receipts.sort(key=lambda r:r['entry']['selection']['date'])
    dates=[r['entry']['selection']['date'] for r in receipts if not r['entry']['selection']['context_only'] and pd.Timestamp(r['entry']['selection']['date']).weekday()<5]
    expected=[str(d.date()) for d in pd.bdate_range(month+'-01',pd.Timestamp(month+'-01')+pd.offsets.MonthEnd())]
    if dates!=expected:raise ValueError('Month weekday coverage mismatch '+repr((dates,expected)))
    verify_source(receipts,config,root)
    if sha(config['reference']['path'])!=config['reference']['sha256']:raise ValueError('Reference changed')
    reference=np.load(config['reference']['path'],mmap_mode='r');days=[]
    for date in dates:
        days.append(prepare_day(root,receipts,reference,date))
        save_json(root/'progress.json',dict(status='PREPARING',prepared=len(days),expected=len(dates),last_date=date,seconds=time.perf_counter()-start))
        print(json.dumps(dict(date=date,prepared=len(days),entry_origins=days[-1]['entry_origins'])),flush=True)
    result=dict(status='COMPLETE',version=VERSION,month=month,initial_capital=1500.,days=days,
        split=dict(train=[d for d in dates if d<month+'-17'],validation=[d for d in dates if month+'-17'<=d<month+'-24'],holdout=[d for d in dates if d>=month+'-24']),
        source_manifest_sha256=sha(Path(market)/'manifest.json'),verification_sha256=sha(root/'source_verification.json'),seconds=time.perf_counter()-start,
        bytes_added=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),source_reuse='Canonical MBO-derived quote Parquet files shared, not copied; full separate L3 release remains independent')
    save_json(root/'manifest.json',result)
    save_json(root/'progress.json',dict(status='COMPLETE',prepared=len(days),expected=len(dates),last_date=dates[-1],seconds=result['seconds']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1');p.add_argument('--output',required=True);p.add_argument('--month',default='2026-08')
    a=p.parse_args();print(json.dumps(prepare(a.market,a.output,a.month),default=str))
