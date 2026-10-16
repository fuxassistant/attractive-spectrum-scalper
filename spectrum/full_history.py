"""Bounded, resumable full-history MBO materializer.

One shared book pool, one asynchronous source planner, and a small feature pool.
Only generated staging binaries are retired, after verified durable publication.
Raw archives and prior datasets are never modified.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import defaultdict, Counter
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import struct
import time
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import zstandard
import databento_dbn
from .source import DbnSource
from .ingest import PACKED, save_json
from .build import build_capture
from .core.storage import sha
from .core.state import implementation_id
from .lab import compute

NS=10**9
MINUTE=60*NS
DAY=86400*NS
FAMILIES=['motion','flow','spectral']
BASE=Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def code_identity():
    files=['full_history.py','source.py','ingest.py','build.py','lab.py','feature_stream.py','replay.py']
    return dict(core=implementation_id(),**{f:sha(Path(__file__).with_name(f)) for f in files})


def assert_code(config):
    if code_identity()!=config['code_identity']:
        raise RuntimeError('Implementation changed during immutable run; refusing mixed formulas')


def compatible_identity(config, identity):
    """Only explicitly migrated orchestration versions may reuse unchanged formulas."""
    current=config['code_identity']
    return identity in [current,*config.get('compatible_code_identities',[])] and {
        k:v for k,v in identity.items() if k!='full_history.py'
    }=={k:v for k,v in current.items() if k!='full_history.py'}


def training_assessment(receipt):
    reasons=[]
    if receipt['rows']==0:reasons.append('NO_FINALIZED_SECONDS')
    if receipt['entry']['selection']['source_condition']!='available':reasons.append('SOURCE_DAY_NOT_AVAILABLE')
    if receipt['entry']['selection']['context_only']:reasons.append('CONTEXT_ONLY_CONTRACT_SELECTION')
    if not receipt['entry_allowed_rows']:reasons.append('NO_ADMISSIBLE_ENTRY_ROWS')
    support=receipt['reference_open_rows']
    if not support:reasons.append('NO_OHLC_REFERENCE_SUPPORT')
    elif receipt['admitted_rows']/support<.8:reasons.append('ADMISSIBLE_REFERENCE_COVERAGE_BELOW_80_PERCENT')
    if receipt['rows'] and receipt['independent_ohlc']['status']!='PASS':reasons.append('OHLC_QA_NOT_VERIFIED')
    return dict(eligible=not reasons,reasons=reasons,
                row_filter='entry_allowed AND required feature __known masks; split at capture/epoch/contract and gaps',
                semantics='technical input eligibility only; targets, chronological splits and economic validation remain separate')


def write_training_catalog(root,receipts,complete=False):
    entries=[dict(key=r['key'],date=r['entry']['selection']['date'],symbol=r['entry']['selection']['symbol'],
                  features=r['features'],rows=r['rows'],entry_allowed_rows=r['entry_allowed_rows'],
                  feature_sha256=r['feature_sha256'],**training_assessment(r)) for r in receipts.values()]
    save_json(Path(root)/'training_catalog.json',dict(status='COMPLETE' if complete else 'PARTIAL',
        eligible_captures=sum(e['eligible'] for e in entries),excluded_captures=sum(not e['eligible'] for e in entries),
        captures=entries,utc=now()))


def child_path(root, *parts):
    root=Path(root).resolve()
    path=root.joinpath(*parts).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError('Path escaped owned dataset directory')
    return path


class ObservedCalendar:
    """Missing spot minutes mean reference support unknown/absent, not proven closure."""
    def __init__(self,path,expected_sha=None):
        self.path=Path(path)
        self.sha256=sha(self.path)
        if expected_sha and self.sha256!=expected_sha:
            raise ValueError('Observed calendar drift')
        self.minutes=np.load(self.path,mmap_mode='r')
        if not len(self.minutes) or np.any(np.diff(self.minutes)<=0):
            raise ValueError('Invalid observed minute index')

    def closed(self,timestamps):
        minute=np.asarray(timestamps,dtype=np.int64)//MINUTE
        idx=np.searchsorted(self.minutes,minute)
        safe=np.minimum(idx,len(self.minutes)-1)
        return (idx==len(self.minutes)) | (self.minutes[safe]!=minute)

    def contract(self):
        return dict(version='OBSERVED_OHLC_MINUTES_V1',sha256=self.sha256,
                    semantics='absent reference minute is unavailable; regular breaks, weekends, holidays and unknown feed gaps retained without interpolation')


def prepare_reference(root,start,end):
    source=BASE/'data/raw/ctrader/OHLC/xauusd_m1_full.csv'
    target=child_path(root,'reference');target.mkdir(parents=True,exist_ok=True)
    path=target/'observed_minutes.npy'
    lo,hi=pd.Timestamp(start).value//10**6,pd.Timestamp(end).value//10**6
    parts=[]
    for chunk in pd.read_csv(source,usecols=['timestamp_ms','timestamp_utc'],chunksize=500000):
        selected=chunk[(chunk.timestamp_ms>=lo)&(chunk.timestamp_ms<hi)]
        if len(selected):
            stamps=selected.timestamp_ms.to_numpy(np.int64)
            assert (pd.to_datetime(selected.timestamp_utc,utc=True).astype('int64').to_numpy()//10**6==stamps).all()
            assert (stamps%60000==0).all()
            parts.append(stamps//60000)
    minutes=np.concatenate(parts)
    if np.any(np.diff(minutes)<=0):raise ValueError('OHLC duplicates or clock regression')
    np.save(path,minutes)
    gaps=np.flatnonzero(np.diff(minutes)>1)
    save_json(target/'manifest.json',dict(source=str(source),source_sha256=sha(source),minutes=len(minutes),
        first_utc=str(pd.Timestamp(int(minutes[0])*60,unit='s',tz='UTC')),
        last_utc=str(pd.Timestamp(int(minutes[-1])*60,unit='s',tz='UTC')),sha256=sha(path),
        gaps=[dict(start_utc=str(pd.Timestamp(int(minutes[i]+1)*60,unit='s',tz='UTC')),
                   end_utc=str(pd.Timestamp(int(minutes[i+1])*60,unit='s',tz='UTC')),
                   minutes=int(minutes[i+1]-minutes[i]-1)) for i in gaps]))
    return dict(path=str(path),sha256=sha(path))


def inventory():
    archives=[BASE/'data/raw/GLBX-20260805-FTTJQEK3T3.zip',BASE/'data/raw/GLBX-20260902-AHVTK4GAWB-Aug-1-Sep-2_2026.zip']
    result=[];ranges=[]
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            metadata=json.loads(z.read('metadata.json'))
            manifest=json.loads(z.read('manifest.json'))
            condition={x['date']:x['condition'] for x in json.loads(z.read('condition.json'))}
            hashes={x['filename']:x['hash'].split(':')[-1] for x in manifest['files']}
            ranges.append(dict(start=metadata['query']['start'],end=metadata['query']['end']))
            for member in sorted(i for i in z.namelist() if i.endswith('.mbo.dbn.zst')):
                index=len(result)
                result.append(dict(key=f'member_{index:03d}',index=index,archive=str(archive),member=member,
                    sha256=hashes[member],compressed_bytes=z.getinfo(member).file_size,
                    archive_bytes=archive.stat().st_size,archive_mtime_ns=archive.stat().st_mtime_ns,
                    conditions=condition,metadata_sha256=hashlib.sha256(z.read('metadata.json')).hexdigest()))
    return result,ranges


def mappings(item):
    with zipfile.ZipFile(item['archive']) as z,z.open(item['member']) as raw,zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        head=reader.read(8);body=reader.read(struct.unpack_from('<I',head,4)[0])
        meta=databento_dbn.Metadata.decode(head+body)
    result={}
    for symbol,intervals in meta.mappings.items():
        if not re.fullmatch(r'GC[FGHJKMNQUVXZ]\d{1,2}',symbol):continue
        for interval in intervals:
            result.setdefault(int(interval['symbol']),[]).append(dict(symbol=symbol,start=str(interval['start_date']),end=str(interval['end_date'])))
    if not result:raise ValueError('No GC outright instrument mappings')
    return result


def symbol_at(mapping,instrument,day):
    date=str(pd.Timestamp(day*86400,unit='s',tz='UTC').date())
    symbols={m['symbol'] for m in mapping[instrument] if m['start']<=date<m['end']}
    if len(symbols)!=1:raise ValueError('Ambiguous instrument mapping')
    return next(iter(symbols))


def london_ns(day):
    date=str(pd.Timestamp(int(day)*86400,unit='s',tz='UTC').date())
    return pd.Timestamp(date+' 08:00',tz='Europe/London').value


def prefix_score(records,pending):
    """Only count records through a complete F_LAST boundary in the known prefix."""
    ends=np.flatnonzero(records['flags']&128)
    if not len(ends):
        return (0,0),(pending[0]+int(records['size'][records['action']==b'T'].sum()),pending[1]+len(records))
    end=int(ends[-1])+1
    head,tail=records[:end],records[end:]
    return (pending[0]+int(head['size'][head['action']==b'T'].sum()),pending[1]+len(head)),(int(tail['size'][tail['action']==b'T'].sum()),len(tail))


def stage_member(config,item):
    assert_code(config)
    root=Path(config['root']);stage=child_path(root,'staging',item['key']);stage.mkdir(parents=True,exist_ok=True)
    final=stage/'manifest.json'
    if final.exists():return json.loads(final.read_text())
    source_path=Path(item['archive'])
    if source_path.stat().st_size!=item['archive_bytes'] or source_path.stat().st_mtime_ns!=item['archive_mtime_ns']:
        raise ValueError('Source archive changed')
    mapping=mappings(item);ids=list(mapping)
    scores=defaultdict(lambda:[0,0]);pending=defaultdict(lambda:(0,0));totals=Counter();counts=Counter()
    started=time.perf_counter();src=DbnSource(source_path,item['member'])
    for block in src:
        b=block[np.isin(block['instrument_id'],ids)]
        if not len(b):continue
        days=b['ts_recv']//DAY
        for day in np.unique(days):
            day=int(day);d=b[days==day];cut=london_ns(day)
            routes=np.stack([d['publisher_id'],d['instrument_id'],d['channel_id']],axis=1)
            unique,nums=np.unique(routes,axis=0,return_counts=True)
            for route,n in zip(unique,nums):
                key=(day,*map(int,route));counts[key]+=int(n);totals[key]+=int(n)
                r=d[np.all(routes==route,axis=1)&(d['ts_recv']<cut)]
                if len(r):
                    ready,tail=prefix_score(r,pending[key]);pending[key]=tail
                    scores[key][0]+=ready[0];scores[key][1]+=ready[1]
        save_json(stage/'progress.json',dict(status='CENSUS',records=src.count,seconds=time.perf_counter()-started))
    if src.sha256!=item['sha256']:raise ValueError('Source member SHA mismatch in census')
    choices={}
    for day in sorted({key[0] for key in counts}):
        keys=[k for k in counts if k[0]==day]
        candidates=[k for k in keys if scores[k][1]>0]
        # No morning evidence means storage/context only, never an actionable
        # day selected with future data. Such days get routing_admissible=False.
        chosen=max(candidates,key=lambda k:(*scores[k],-k[2])) if candidates else max(keys,key=lambda k:(totals[k],-k[2]))
        date=str(pd.Timestamp(day*86400,unit='s',tz='UTC').date())
        choices[day]=dict(route=list(chosen[1:]),symbol=symbol_at(mapping,chosen[2],day),date=date,
            selection_available_ns=str(london_ns(day) if candidates else (day+1)*DAY),
            context_only=not bool(candidates),source_condition=item['conditions'].get(date,'unknown'),
            prefix_trade_volume=scores[chosen][0],prefix_complete_records=scores[chosen][1],
            candidates=[dict(route=list(k[1:]),symbol=symbol_at(mapping,k[2],day),trade_volume=scores[k][0],complete_records=scores[k][1]) for k in keys],
            records_expected=counts[chosen])
    estimate=sum(x['records_expected'] for x in choices.values())*PACKED.itemsize
    if psutil.disk_usage(str(root)).free < config['reserve_bytes']+estimate+config['inflight_output_reserve_bytes']:
        raise RuntimeError(f'INSUFFICIENT_SPACE_FOR_STAGE required_extra={estimate}')
    # A retry truncates only its own generated staging files, never source data.
    save_json(stage/'owner.json',dict(run_id=config['run_id'],member=item['key']))
    handles={};current={};entries=[];ordinal=0
    src=DbnSource(source_path,item['member'])
    try:
        for block in src:
            days=block['ts_recv']//DAY
            for day in np.unique(days):
                day=int(day)
                if day not in choices:continue
                choice=choices[day];route=choice['route']
                indices=np.flatnonzero((days==day)&(block['publisher_id']==route[0])&(block['instrument_id']==route[1])&(block['channel_id']==route[2]))
                b=block[indices]
                if not len(b):continue
                a=np.empty(len(b),PACKED);a['source_ordinal']=indices+ordinal;a['record']=b
                cuts=sorted(set([0,len(b),*np.flatnonzero(b['action']==b'R').tolist()]))
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    if lo==hi:continue
                    if b[lo]['action']==b'R':
                        if day in handles:handles[day].close()
                        number=10000+item['index']*1000+len(entries)
                        name=f'capture_{number:06d}.bin'
                        e=dict(name=name,records=0,start_ns=int(b[lo]['ts_recv']),first_ordinal=int(a[lo]['source_ordinal']),
                            reset_flags=int(b[lo]['flags']),route=route,selection=choice,member_key=item['key'])
                        entries.append(e);current[day]=e
                        handles[day]=child_path(stage,name).open('wb')
                    if day not in handles:raise ValueError(f'No source reset for {choice["date"]} {route}; checkpoint adapter required')
                    handles[day].write(a[lo:hi].tobytes());e=current[day]
                    e.update(records=e['records']+hi-lo,end_ns=int(b[hi-1]['ts_recv']),last_ordinal=int(a[hi-1]['source_ordinal']))
            ordinal+=len(block)
            if psutil.disk_usage(str(root)).free<config['reserve_bytes']:
                raise RuntimeError('DISK_RESERVE_REACHED_DURING_STAGE')
            save_json(stage/'progress.json',dict(status='WRITING',records=ordinal,seconds=time.perf_counter()-started))
    finally:
        for h in handles.values():h.close()
    if src.sha256!=item['sha256']:raise ValueError('Source member SHA mismatch in materialization')
    for e in entries:e['sha256']=sha(stage/e['name'])
    result=dict(status='COMPLETE',item=item,raw_records=ordinal,selected_records=sum(e['records'] for e in entries),
        seconds=time.perf_counter()-started,entries=entries,choices={str(k):v for k,v in choices.items()},
        source_member_sha256=src.sha256,raw_bytes=estimate,code_identity=config['code_identity'])
    save_json(final,result)
    return result


def reconstruct(config,entry,output):
    assert_code(config)
    return build_capture(Path(config['root'])/'staging'/entry['member_key'],entry,output,entry['route'],min_free_bytes=config['reserve_bytes'])


def verify_ohlc(capture,seed):
    bars=pq.read_table(Path(capture)/'bars.parquet').to_pandas()
    candidates=bars[bars.valid]
    if not len(candidates):return dict(checks=0,max_error=0.,status='NO_VALID_BARS')
    sample=candidates.sample(min(8,len(candidates)),random_state=seed)
    checked=0;error=0.
    for row in sample.itertuples():
        left=int(row.t_ns)-MINUTE
        q=pq.read_table(Path(capture)/'quotes.parquet',filters=[('t_ns','>=',left-30*NS),('t_ns','<=',int(row.t_ns))],columns=['t_ns','valid','bid_nano','ask_nano']).to_pandas()
        before=q[(q.t_ns<=left)&q.valid].tail(1);after=q[(q.t_ns>left)&q.valid]
        if not len(before) or not len(after):continue
        v=np.r_[(before.bid_nano.to_numpy()+before.ask_nano.to_numpy())/2e9,(after.bid_nano.to_numpy()+after.ask_nano.to_numpy())/2e9]
        err=float(np.max(np.abs(np.array([row.o,row.h,row.l,row.c])-np.array([v[0],v.max(),v.min(),v[-1]]))))
        error=max(error,err);checked+=1
    if error>1e-8:raise ValueError(f'Independent OHLC mismatch {error}')
    return dict(checks=checked,max_error=error,status='PASS' if checked else 'NO_COMPARABLE_SAMPLES')


def generate_features(config,entry,capture,output):
    assert_code(config)
    started=time.perf_counter();capture=Path(capture);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((capture/'manifest.json').read_text())
    if m['source_partition_sha256']!=entry['sha256'] or m['route']!=entry['route'] or m['implementation']!=config['code_identity']['core']:
        raise ValueError('Market source or implementation identity mismatch')
    for filename,expected in m['payloads'].items():
        if sha(capture/filename)!=expected:raise ValueError('Market payload hash drift')
    calendar=ObservedCalendar(config['reference']['path'],config['reference']['sha256'])
    from .feature_stream import FeatureStream
    engine=FeatureStream(FAMILIES,calendar=calendar,source_id=m['source_partition_sha256'])
    writer=None;rows=admitted=routing=entry_allowed=reference_open=0;known=Counter()
    source=pq.ParquetFile(capture/'seconds.parquet')
    if source.metadata.num_rows==0:
        if m['seconds_rows']!=0:raise ValueError('Empty seconds payload contradicts market manifest')
        result=dict(status='EXCLUDED',exclusion_reasons=['NO_FINALIZED_SECONDS'],rows=0,features=0,
            families=FAMILIES,admitted_rows=0,routing_admissible_rows=0,entry_allowed_rows=0,reference_open_rows=0,
            coverage={},selection=entry['selection'],calendar=calendar.contract(),feature_sha256=None,
            market_manifest_sha256=sha(capture/'manifest.json'),elapsed_seconds=time.perf_counter()-started,
            independent_ohlc=dict(checks=0,max_error=0.,status='NOT_APPLICABLE_EMPTY_CAPTURE'),
            evidence=dict(records=m['records'],quotes=m['quotes'],seconds_rows=0,bars=m['bars']),
            code_identity=config['code_identity'],pid=os.getpid())
        save_json(output/'manifest.json',result)
        return result
    try:
        for batch in source.iter_batches(batch_size=32768):
            if psutil.disk_usage(str(output)).free<config['reserve_bytes']:raise RuntimeError('DISK_RESERVE_REACHED_DURING_FEATURES')
            frame=batch.to_pandas();c=engine.feed(frame);t=frame.t_ns.to_numpy()
            opened=~calendar.closed(t)
            usable=frame.valid.to_numpy()&opened&((frame.available_ns-frame.t_ns).to_numpy()<=NS)
            if entry['selection']['source_condition']!='available':usable[:]=False
            route_ok=(t>=int(entry['selection']['selection_available_ns']))&~np.full(len(t),entry['selection']['context_only'],bool)
            times=pd.to_datetime(t,utc=True)
            ld=times.tz_convert('Europe/London');ny=times.tz_convert('America/New_York')
            allowed=usable&route_ok&(ld.hour>=8)&(ny.hour<17)&(ld.dayofweek<5)
            cols={k:frame[k].to_numpy() for k in ['t_ns','available_ns','epoch']}
            cols.update(scheduled_closed=~opened,reference_observed_open=opened,admissible=usable,
                routing_admissible=route_ok,entry_allowed=allowed,
                source_day_degraded=np.full(len(t),entry['selection']['source_condition']!='available',bool))
            for name,value in c['values'].items():
                mask=c['known'][name]&usable
                cols[name]=np.where(mask,value,0.);cols[name+'__known']=mask;known[name]+=int(mask.sum())
            table=pa.table(cols)
            if writer is None:writer=pq.ParquetWriter(output/'features.parquet',table.schema,compression='zstd')
            writer.write_table(table,row_group_size=32768)
            rows+=len(frame);admitted+=int(usable.sum());routing+=int(route_ok.sum());entry_allowed+=int(allowed.sum());reference_open+=int(opened.sum())
    finally:
        if writer is not None:writer.close()
    if not rows:raise ValueError('No finalized seconds in capture')
    saved=pq.ParquetFile(output/'features.parquet')
    if saved.metadata.num_rows!=rows:raise ValueError('Saved feature count mismatch')
    for batch in saved.iter_batches(batch_size=8192):
        columns=batch.to_pydict()
        for name in known:
            v=np.asarray(columns[name]);k=np.asarray(columns[name+'__known'])
            if not np.isfinite(v).all() or np.any(v[~k]!=0):raise ValueError('Nonfinite or unmasked feature')
    qa=verify_ohlc(capture,int(Path(entry['name']).stem.split('_')[-1]))
    result=dict(status='COMPLETE',rows=rows,features=len(known),families=FAMILIES,admitted_rows=admitted,
        routing_admissible_rows=routing,entry_allowed_rows=entry_allowed,reference_open_rows=reference_open,
        coverage={k:v/rows for k,v in known.items()},selection=entry['selection'],calendar=calendar.contract(),
        scheduled_closed_semantics='compatibility alias for missing OHLC reference minute, not an exchange-closure assertion',
        feature_sha256=sha(output/'features.parquet'),market_manifest_sha256=sha(capture/'manifest.json'),
        elapsed_seconds=time.perf_counter()-started,independent_ohlc=qa,code_identity=config['code_identity'],pid=os.getpid())
    save_json(output/'manifest.json',result)
    return result


def make_receipt(config,entry,market,feature):
    if not compatible_identity(config,feature['code_identity']) or market['source_partition_sha256']!=entry['sha256']:
        raise ValueError('Publication identity mismatch')
    root=Path(config['root']);name=Path(entry['name']).stem;key=entry['member_key']+'/'+name
    mpath=child_path(root,'market',key);fpath=child_path(root,'features',key)
    if feature['market_manifest_sha256']!=sha(mpath/'manifest.json'):raise ValueError('Feature market lineage drift')
    receipt=dict(status=feature['status'],key=key,entry=entry,market=str(mpath),features=str(fpath),
        market_manifest_sha256=sha(mpath/'manifest.json'),feature_manifest_sha256=sha(fpath/'manifest.json'),
        market_payloads=market['payloads'],feature_sha256=feature['feature_sha256'],
        bytes=sum((mpath/p).stat().st_size for p in market['payloads'])+((fpath/'features.parquet').stat().st_size if feature['feature_sha256'] else 0),
        raw_bytes=entry['records']*PACKED.itemsize,market_seconds=market['elapsed_seconds'],feature_seconds=feature['elapsed_seconds'],
        rows=feature['rows'],admitted_rows=feature['admitted_rows'],entry_allowed_rows=feature['entry_allowed_rows'],
        reference_open_rows=feature['reference_open_rows'],independent_ohlc=feature['independent_ohlc'],completed_utc=now(),
        code_identity=feature['code_identity'],exclusion_reasons=feature.get('exclusion_reasons',[]))
    receipt['training']=training_assessment(receipt)
    dest=child_path(root,'receipts',key.replace('/','_')+'.json');dest.parent.mkdir(exist_ok=True)
    # Verify durable output before publishing receipt and retiring generated raw.
    for filename,expected in receipt['market_payloads'].items():
        if sha(mpath/filename)!=expected:raise ValueError('Post-publication market verification failed')
    if receipt['feature_sha256'] and sha(fpath/'features.parquet')!=receipt['feature_sha256']:raise ValueError('Post-publication feature verification failed')
    if not receipt['feature_sha256'] and (receipt['status']!='EXCLUDED' or receipt['rows']!=0 or (fpath/'features.parquet').exists()):
        raise ValueError('Invalid empty exclusion receipt')
    save_json(dest,receipt)
    raw=child_path(root,'staging',entry['member_key'],entry['name'])
    if raw.exists():
        if sha(raw)!=entry['sha256']:raise ValueError('Refusing to retire unexpected staging file')
        raw.unlink()
    receipt['staging_retired']=True
    save_json(dest,receipt)
    return receipt


def recover(config):
    root=Path(config['root']);receipts={}
    for path in (root/'receipts').glob('*.json'):
        r=json.loads(path.read_text())
        if not compatible_identity(config,r['code_identity']):raise ValueError('Resume code identity mismatch')
        if sha(Path(r['market'])/'manifest.json')!=r['market_manifest_sha256'] or sha(Path(r['features'])/'manifest.json')!=r['feature_manifest_sha256']:
            raise ValueError('Resume manifest drift')
        for file,h in r['market_payloads'].items():
            if sha(Path(r['market'])/file)!=h:raise ValueError('Resume market payload drift')
        if r['feature_sha256']:
            if sha(Path(r['features'])/'features.parquet')!=r['feature_sha256']:raise ValueError('Resume feature payload drift')
        elif r['status']!='EXCLUDED' or r['rows']!=0 or (Path(r['features'])/'features.parquet').exists():
            raise ValueError('Invalid resumed empty exclusion')
        raw=child_path(root,'staging',r['entry']['member_key'],r['entry']['name'])
        if raw.exists():
            if sha(raw)!=r['entry']['sha256']:raise ValueError('Unexpected resumed staging payload')
            raw.unlink()
        receipts[r['key']]=r
    return receipts


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status')=='COMPLETE':
        recover(config)
        print('Dataset already complete and verified',flush=True)
        return
    lock=root/'owner.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        try:
            p=psutil.Process(previous['pid'])
            if abs(p.create_time()-previous['created'])<.1:raise RuntimeError('Run already owns this output directory')
        except psutil.NoSuchProcess:pass
    save_json(lock,dict(pid=os.getpid(),created=psutil.Process().create_time(),run_id=config['run_id']))
    save_json(root/'status.json',dict(status='VERIFYING_EXISTING_OUTPUTS',pid=os.getpid(),utc=now()))
    receipts=recover(config);queued=[];features=[];member_index=0;staged=set()
    errors=json.loads((root/'errors.json').read_text()) if (root/'errors.json').exists() else []
    write_training_catalog(root,receipts)
    active={};started=time.perf_counter();peak=0;last=0
    with ProcessPoolExecutor(max_workers=config['book_workers']) as books, ProcessPoolExecutor(max_workers=config['feature_workers']) as labs, ProcessPoolExecutor(max_workers=1) as planner:
        while member_index<len(config['members']) or active or queued or features:
            free=psutil.disk_usage(str(root)).free
            if not any(kind=='stage' for kind,_,_ in active.values()) and member_index<len(config['members']):
                item=config['members'][member_index];stage_manifest=root/'staging'/item['key']/'manifest.json'
                # Keep only one staged member ahead of the shared book queue.
                if stage_manifest.exists():
                    stage=json.loads(stage_manifest.read_text())
                    if not compatible_identity(config,stage['code_identity']) or stage['item']!=item:raise ValueError('Staged identity drift')
                    member_index+=1;staged.add(item['key'])
                    for entry in stage['entries']:
                        key=entry['member_key']+'/'+Path(entry['name']).stem
                        if key not in receipts:queued.append(entry)
                elif not queued and free>config['reserve_bytes']+config['inflight_output_reserve_bytes']+2*10**9:
                    future=planner.submit(stage_member,config,item);active[future]=('stage',item,None);member_index+=1
            queued.sort(key=lambda x:x['records'],reverse=True)
            while queued and sum(k=='book' for k,_,_ in active.values())<config['book_workers'] and free>config['reserve_bytes']+config['inflight_output_reserve_bytes']:
                entry=queued.pop(0);key=entry['member_key']+'/'+Path(entry['name']).stem
                output=child_path(root,'market',key);output.parent.mkdir(parents=True,exist_ok=True)
                if (output/'manifest.json').exists():
                    market=json.loads((output/'manifest.json').read_text());features.append((entry,market));continue
                # Preserve incomplete attempts, then restart this owned capture.
                if output.exists():output.rename(output.with_name(output.name+'_interrupted_'+str(time.time_ns())))
                future=books.submit(reconstruct,config,entry,output);active[future]=('book',entry,None)
            while features and sum(k=='feature' for k,_,_ in active.values())<config['feature_workers'] and free>config['reserve_bytes']:
                entry,market=features.pop(0);key=entry['member_key']+'/'+Path(entry['name']).stem
                output=child_path(root,'features',key);output.parent.mkdir(parents=True,exist_ok=True)
                capture=child_path(root,'market',key)
                if (output/'manifest.json').exists():
                    result=json.loads((output/'manifest.json').read_text());receipts[key]=make_receipt(config,entry,market,result);continue
                if output.exists():output.rename(output.with_name(output.name+'_interrupted_'+str(time.time_ns())))
                future=labs.submit(generate_features,config,entry,capture,output);active[future]=('feature',entry,market)
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED) if active else (set(),set())
            for future in done:
                kind,item,market=active.pop(future)
                try:
                    result=future.result()
                    if kind=='stage':
                        staged.add(item['key'])
                        queued.extend(e for e in result['entries'] if e['member_key']+'/'+Path(e['name']).stem not in receipts)
                    elif kind=='book':features.append((item,result))
                    else:
                        key=item['member_key']+'/'+Path(item['name']).stem
                        receipts[key]=make_receipt(config,item,market,result)
                    print(json.dumps(dict(utc=now(),stage=kind,item=item.get('key',item.get('name')),completed=len(receipts))),flush=True)
                except Exception as exc:
                    errors.append(dict(stage=kind,item=item,error=repr(exc),utc=now()))
                    save_json(root/'errors.json',errors)
                    # Stop new admission on an error; let current workers finish.
                    save_json(root/'status.json',dict(status='FAILED_DRAINING',error=errors[-1],pid=os.getpid(),utc=now()))
                    raise
            if time.perf_counter()-last>5:
                process=psutil.Process();rss=process.memory_info().rss
                for child in process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss);last=time.perf_counter()
                state='RUNNING' if active else 'WAITING_FOR_SPACE'
                save_json(root/'status.json',dict(status=state,pid=os.getpid(),utc=now(),elapsed_seconds=time.perf_counter()-started,
                    members_staged=len(staged),members_total=len(config['members']),captures_complete=len(receipts),
                    captures_excluded=sum(r['status']=='EXCLUDED' for r in receipts.values()),
                    books_running=sum(k=='book' for k,_,_ in active.values()),feature_jobs_running=sum(k=='feature' for k,_,_ in active.values()),
                    planner_running=any(k=='stage' for k,_,_ in active.values()),book_queue=len(queued),feature_queue=len(features),
                    durable_bytes=sum(r['bytes'] for r in receipts.values()),free_bytes=psutil.disk_usage(str(root)).free,
                    reserve_bytes=config['reserve_bytes'],rss_bytes=rss,peak_sampled_rss_bytes=peak))
                save_json(root/'catalog.partial.json',dict(status='PARTIAL',captures=list(receipts.values()),source_gaps=config['source_gaps']))
                write_training_catalog(root,receipts)
            if not active:time.sleep(2)
    weeks=defaultdict(list)
    for r in receipts.values():
        date=pd.Timestamp(r['entry']['selection']['date'])
        if date.weekday()==6:date+=pd.Timedelta(days=1)
        monday=(date-pd.Timedelta(days=date.weekday())).strftime('%Y-%m-%d')
        weeks[monday].append(r['key'])
    (root/'weeks').mkdir(exist_ok=True)
    for monday,keys in weeks.items():save_json(root/'weeks'/f'{monday}.json',dict(status='COMPLETE',week=monday,capture_keys=sorted(keys),independence='source reset captures; short feature warmup masked; no implied cross-contract price continuity'))
    result=dict(status='COMPLETE',completed_utc=now(),captures=list(receipts.values()),source_gaps=config['source_gaps'],
        source_members=config['members'],reference=config['reference'],weeks=len(weeks),code_identity=config['code_identity'],
        bytes=sum(r['bytes'] for r in receipts.values()),elapsed_this_run_seconds=time.perf_counter()-started,
        peak_sampled_rss_bytes=peak,book_workers=config['book_workers'],feature_workers=config['feature_workers'])
    save_json(manifest_path,result)
    write_training_catalog(root,receipts,complete=True)
    save_json(root/'status.json',dict(status='COMPLETE',pid=os.getpid(),utc=now(),captures_complete=len(receipts),durable_bytes=result['bytes']))


def prepare(root,book_workers=6,feature_workers=2):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    path=root/'run_config.json'
    if path.exists():
        config=json.loads(path.read_text());assert_code(config);return config
    members,ranges=inventory()
    gaps=[dict(start_utc=str(pd.Timestamp(a['end'],tz='UTC')),end_utc=str(pd.Timestamp(b['start'],tz='UTC'))) for a,b in zip(ranges,ranges[1:]) if a['end']<b['start']]
    reference=prepare_reference(root,pd.Timestamp(min(x['start'] for x in ranges),tz='UTC'),pd.Timestamp(max(x['end'] for x in ranges),tz='UTC'))
    config=dict(run_id='FULL_MBO_V1',root=str(root),created_utc=now(),members=members,source_gaps=gaps,reference=reference,
        book_workers=book_workers,feature_workers=feature_workers,reserve_bytes=12*10**9,inflight_output_reserve_bytes=6*10**9,
        code_identity=code_identity(),families=FAMILIES,
        selection_rule='GC outright per UTC date: max complete pre-London trade volume, then complete records. No morning evidence => context only. Routing usable only after selection. Contracts never price-spliced.',
        cleanup_rule='retire only staging .bin after durable hashes, feature validation and receipt publication; keep original ZIPs and all previous datasets')
    save_json(path,config);return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(BASE/'data/derived/full_mbo_v1'));p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--book-workers',type=int,default=6);p.add_argument('--feature-workers',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.book_workers<=8 or not 1<=args.feature_workers<=2:p.error('Invalid worker count')
    config=prepare(args.root,args.book_workers,args.feature_workers)
    if args.prepare_only:print(json.dumps(dict(root=config['root'],members=len(config['members']),reference=config['reference'])));return
    try:run(config)
    except Exception as exc:
        save_json(Path(config['root'])/'status.json',dict(status='FAILED',pid=os.getpid(),utc=now(),error=repr(exc)))
        raise


if __name__=='__main__':main()
