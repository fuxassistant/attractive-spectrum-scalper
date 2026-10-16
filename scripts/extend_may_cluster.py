"""Add May 18-29 on the same GCM6 route; measure a real pipelined build.

Immutable old datasets remain intact. Final cluster uses hard links to immutable
payloads, not copies. Run from repository root with the project Python.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json, os, time, collections
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd
import psutil
from spectrum.source import DbnSource
from spectrum.ingest import PACKED, save_json
from spectrum.core.storage import sha
from spectrum.build import build_capture
from spectrum.lab import generate_capture

RAW = ROOT/'data/derived/raw_may_extension_v1'
MARKET = ROOT/'data/derived/market_may_extension_v1'
FEATURES = ROOT/'data/derived/features_may_extension_v1'
REPORT = ROOT/'reports/may_extension_v1'
ROUTE = [1, 19181, 22]
SOURCE = ROOT/'data/raw/GLBX-20260805-FTTJQEK3T3.zip'
MEMBER = 'glbx-mdp3-20260501-20260531.mbo.dbn.zst'
FAMILIES = ['motion', 'flow', 'spectral']
START, END = '2026-05-17T00:00Z', '2026-05-30T00:00Z'
FIRST_CAPTURE = 12
BUILD_WORKERS, FEATURE_WORKERS = 6, 2
PUBLISH_CLUSTER = True


def partition():
    RAW.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = DbnSource(SOURCE, MEMBER)
    lo_ns, hi_ns = [pd.Timestamp(t).value for t in (START, END)]
    ordinal = 0
    entries, daily = [], collections.Counter()
    output = None
    selected_wire = __import__('hashlib').sha256()
    try:
        for block in source:
            mask = (block['ts_recv'] >= lo_ns) & (block['ts_recv'] < hi_ns)
            inside = block[mask]
            if len(inside):
                days = inside['ts_recv']//(86400*10**9)
                for day in np.unique(days):
                    day_block = inside[days == day]
                    ids, nums = np.unique(day_block['instrument_id'], return_counts=True)
                    for instrument, count in zip(ids, nums):
                        daily[(int(day), int(instrument))] += int(count)
            indices = np.flatnonzero(mask & (block['publisher_id']==ROUTE[0]) & (block['instrument_id']==ROUTE[1]) & (block['channel_id']==ROUTE[2]))
            records = block[indices]
            packed = np.empty(len(records), dtype=PACKED)
            packed['source_ordinal'] = indices+ordinal
            packed['record'] = records
            selected_wire.update(records.tobytes())
            cuts = sorted(set([0, len(records), *np.flatnonzero(records['action']==b'R').tolist()]))
            for a,b in zip(cuts[:-1],cuts[1:]):
                if a == b:
                    continue
                if records[a]['action'] == b'R':
                    if output:
                        output.close()
                    name = f'capture_{FIRST_CAPTURE+len(entries):03d}.bin'
                    entry = dict(name=name, records=0, start_ns=int(records[a]['ts_recv']), first_ordinal=int(packed[a]['source_ordinal']), reset_flags=int(records[a]['flags']))
                    entries.append(entry)
                    output = (RAW/name).open('wb')
                if output is None:
                    raise ValueError('No initial source reset; refusing empty-book time slice')
                output.write(packed[a:b].tobytes())
                entry.update(records=entry['records']+b-a, end_ns=int(records[b-1]['ts_recv']), last_ordinal=int(packed[b-1]['source_ordinal']))
            ordinal += len(block)
        if output:
            output.close()
            output = None
    finally:
        if output:
            output.close()
    import zipfile
    with zipfile.ZipFile(SOURCE) as z:
        original = json.loads(z.read('manifest.json'))
        expected = next(x['hash'].split(':')[-1] for x in original['files'] if x['filename']==MEMBER)
    assert source.sha256 == expected, 'Original compressed member hash mismatch'
    for e in entries:
        e['sha256'] = sha(RAW/e['name'])
    result = dict(schema='RESET_PARTITIONS_V1', source=str(SOURCE), member=MEMBER, source_sha256=source.sha256,
        raw_records=ordinal, selected_records=sum(e['records'] for e in entries), route=ROUTE, partitions=entries,
        interval_utc=[START,END], selected_wire_sha256=selected_wire.hexdigest(),
        seconds=time.perf_counter()-start, order='original month source ordinal; explicit reset captures; no synthetic book')
    save_json(RAW/'manifest.json', result)
    save_json(REPORT/'daily_instrument_activity.json', [dict(day=str(pd.Timestamp(day*86400,unit='s',tz='UTC').date()), instrument_id=i, records=n) for (day,i),n in sorted(daily.items())])
    print(json.dumps(dict(stage='partition_complete', seconds=result['seconds'], selected_records=result['selected_records'], captures=len(entries))), flush=True)
    return result


def link_cluster(market, features):
    mr, fr = ROOT/'data/derived/market_v2', ROOT/'data/derived/features_v2'
    mr.mkdir(exist_ok=False)
    fr.mkdir(exist_ok=False)
    old_m = json.loads((ROOT/'data/derived/market_v1/manifest.json').read_text())
    old_f = json.loads((ROOT/'data/derived/features_v1/manifest.json').read_text())
    all_m, all_f = [], []
    for source_m,source_f,mm,ff in [(ROOT/'data/derived/market_v1',ROOT/'data/derived/features_v1',old_m,old_f),(MARKET,FEATURES,market,features)]:
        for item in mm['captures']:
            name = Path(item['input']['name']).stem
            (mr/name).mkdir()
            for filename in [*item['payloads'], 'manifest.json']:
                os.link(source_m/name/filename, mr/name/filename)
            all_m.append(item)
        for item in ff['captures']:
            name = item['capture']
            (fr/name).mkdir()
            for filename in ['features.parquet','manifest.json']:
                os.link(source_f/name/filename, fr/name/filename)
            all_f.append(item)
    assert len({e['input']['name'] for e in all_m})==len(all_m)
    save_json(mr/'manifest.json',dict(schema='MARKET_CLUSTER_V2',status='COMPLETE',route=ROUTE,captures=all_m,
        source_manifests=[str(ROOT/'data/derived/market_v1/manifest.json'),str(MARKET/'manifest.json')],
        source_manifest_hashes=[sha(ROOT/'data/derived/market_v1/manifest.json'),sha(MARKET/'manifest.json')],
        storage='immutable hard-linked payloads; zero payload duplication',
        scope='four May weeks; same GCM6 contract; not a continuous front-contract roll policy'))
    save_json(fr/'manifest.json',dict(status='COMPLETE',families=FAMILIES,captures=all_f,market_manifest_sha256=sha(mr/'manifest.json'),storage='immutable hard-linked payloads'))
    return dict(market=str(mr),features=str(fr),captures=len(all_m))


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'measurements.json').exists():
        raise RuntimeError('Completed result already exists')
    started = time.perf_counter()
    proc = psutil.Process()
    save_json(REPORT/'job.json',dict(pid=os.getpid(),status='PARTITIONING',start_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    raw = json.loads((RAW/'manifest.json').read_text()) if (RAW/'manifest.json').exists() else partition()
    MARKET.mkdir(exist_ok=False)
    FEATURES.mkdir(exist_ok=False)
    build_start = time.perf_counter()
    markets, features, events = [], [], []
    peak_rss, min_available, peak_children = 0, psutil.virtual_memory().available, 0
    reconstruction_done = None
    with ProcessPoolExecutor(max_workers=BUILD_WORKERS) as builders, ProcessPoolExecutor(max_workers=FEATURE_WORKERS) as labs:
        pending = {builders.submit(build_capture,RAW,e,MARKET/Path(e['name']).stem,ROUTE):('market',e) for e in raw['partitions']}
        while pending:
            done,_ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                kind, entry = pending.pop(future)
                result = future.result()
                name = Path(entry['name']).stem
                events.append(dict(stage=kind,capture=name,finished_after_start=time.perf_counter()-build_start,worker_seconds=result['elapsed_seconds'],pid=result['pid']))
                if kind=='market':
                    markets.append(result)
                    pending[labs.submit(generate_capture,MARKET/name,FEATURES/name,FAMILIES)] = ('features',entry)
                    if len(markets)==len(raw['partitions']):
                        reconstruction_done=time.perf_counter()-build_start
                else:
                    result['capture']=name
                    features.append(result)
                print(json.dumps(events[-1]),flush=True)
            children=proc.children(recursive=True)
            rss=proc.memory_info().rss
            for child in children:
                try:rss+=child.memory_info().rss
                except psutil.NoSuchProcess:pass
            peak_rss=max(peak_rss,rss)
            peak_children=max(peak_children,len(children))
            min_available=min(min_available,psutil.virtual_memory().available)
            save_json(REPORT/'job.json',dict(pid=os.getpid(),status='RUNNING',elapsed_seconds=time.perf_counter()-started,
                market_completed=len(markets),feature_completed=len(features),captures=len(raw['partitions']),tree_rss_bytes=rss,peak_tree_rss_bytes=peak_rss))
    markets.sort(key=lambda r:r['input']['first_ordinal'])
    features.sort(key=lambda r:r['capture'])
    m=dict(schema='MARKET_DATASET_V1',status='COMPLETE',route=ROUTE,source_sha256=raw['source_sha256'],raw_manifest_sha256=sha(RAW/'manifest.json'),workers=BUILD_WORKERS,elapsed_seconds=reconstruction_done,captures=markets)
    save_json(MARKET/'manifest.json',m)
    f=dict(status='COMPLETE',families=FAMILIES,workers=FEATURE_WORKERS,captures=features,market_manifest_sha256=sha(MARKET/'manifest.json'),elapsed_seconds=time.perf_counter()-build_start,timing_scope='overlaps reconstruction; includes input wait')
    save_json(FEATURES/'manifest.json',f)
    cluster=link_cluster(m,f) if PUBLISH_CLUSTER else dict(market=str(MARKET),features=str(FEATURES),captures=len(markets),storage='separate contract stream; no implicit price splice')
    sizes={}
    for key,root in [('raw',RAW),('market',MARKET),('features',FEATURES)]:
        by_name=collections.Counter()
        for p in root.rglob('*'):
            if p.is_file():by_name[p.suffix if key=='raw' else p.name]+=p.stat().st_size
        sizes[key]=dict(by_name)
    result=dict(status='GENERATED_PENDING_QA',scope=f'{START} through {END}; fixed route {ROUTE}',
        route=ROUTE,cluster=cluster,partition_seconds=raw['seconds'],pipeline_seconds=time.perf_counter()-build_start,
        reconstruction_wall_seconds=reconstruction_done,total_wall_seconds=time.perf_counter()-started,
        feature_worker_seconds=sum(x['elapsed_seconds'] for x in features),reconstruction_worker_seconds=sum(x['elapsed_seconds'] for x in markets),
        reconstruction_workers=BUILD_WORKERS,feature_workers=FEATURE_WORKERS,peak_tree_rss_bytes=peak_rss,peak_children=peak_children,min_system_available_bytes=min_available,
        sizes_bytes=sizes,records=raw['selected_records'],quotes=sum(x['quotes'] for x in markets),seconds_rows=sum(x['seconds_rows'] for x in markets),
        events=events,hardware=dict(cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total),
        timing_note='one end-to-end concurrent run, not a matched serial/parallel speedup claim; output sizes exclude hardlink duplicate paths')
    save_json(REPORT/'measurements.json',result)
    save_json(REPORT/'job.json',dict(pid=os.getpid(),status='GENERATED_PENDING_QA'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
