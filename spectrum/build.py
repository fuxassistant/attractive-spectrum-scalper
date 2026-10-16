"""Independent reset captures run in isolated processes with one book each."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, os, time
import numpy as np
import psutil
from .ingest import PACKED, packed_records, save_json
from .core import StreamConfig
from .core.schema import SNAP_SCHEMA, BAR_SCHEMA
from .core.storage import Sink, sha
from .core.state import implementation_id
from .replay import MarketReplay

QUOTE_SCHEMA=[(n,'int64') for n in ('t_ns','available_ns','source_ordinal','epoch','bid_nano','ask_nano','bid_size','ask_size')]+[('valid','bool')]


def build_capture(raw_root, entry, output, route, block_records=32768, min_free_bytes=0):
    started=time.perf_counter(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    raw=Path(raw_root)/entry['name']
    if sha(raw)!=entry['sha256']:raise ValueError('Partition SHA mismatch')
    a=np.memmap(raw,dtype=PACKED,mode='r')
    if len(a)!=entry['records']:raise ValueError('Partition size mismatch')
    quote=Sink(output/'quotes.parquet',QUOTE_SCHEMA)
    snap=Sink(output/'seconds.parquet',SNAP_SCHEMA)
    bars=Sink(output/'bars.parquet',BAR_SCHEMA)
    def emit_quote(row):
        b=replay.core.accumulator.book
        quote.append((row[0],row[6],row[5],row[4],b.best_bid_nano,b.best_ask_nano,b.bid_size_l1,b.ask_size_l1,row[3]))
    replay=MarketReplay(StreamConfig(*route,snapshot_ns=10**9,fragment=int(Path(entry['name']).stem.split('_')[-1])+1),on_quote=emit_quote,on_snapshot=snap.append,on_bar=bars.append)
    process=psutil.Process();peak=process.memory_info().rss; timings=[]
    try:
        for lo in range(0,len(a),block_records):
            if min_free_bytes and psutil.disk_usage(str(output)).free<min_free_bytes:
                raise RuntimeError('DISK_RESERVE_REACHED_DURING_RECONSTRUCTION')
            t=time.perf_counter();replay.feed_batch(packed_records(a[lo:lo+block_records],route));timings.append(time.perf_counter()-t)
            peak=max(peak,process.memory_info().rss)
            if lo//block_records%8==0:
                save_json(output/'progress.json',dict(pid=os.getpid(),records=min(lo+block_records,len(a)),total=len(a),seconds=time.perf_counter()-started,peak_rss_sample=peak))
        if replay.core.engine.pending:raise ValueError('Partition ends inside F_LAST')
        # No fabricated close/next-minute observation at EOF.
        save_json(output/'checkpoint.json',replay.checkpoint())
    finally:
        for sink in (quote,snap,bars):sink.close()
    elapsed=time.perf_counter()-started
    result=dict(status='COMPLETE',input=entry,route=route,source_partition_sha256=entry['sha256'],implementation=implementation_id(),
                records=len(a),quotes=quote.count,seconds_rows=snap.count,bars=bars.count,
                elapsed_seconds=elapsed,records_per_second=len(a)/elapsed,peak_rss_sample=peak,
                batch_p50_seconds=float(np.median(timings)),batch_p99_seconds=float(np.quantile(timings,.99)),
                reasons=replay.core.accumulator.reason_counts,pid=os.getpid(),
                payloads={n:sha(output/n) for n in ('quotes.parquet','seconds.parquet','bars.parquet','checkpoint.json')})
    save_json(output/'manifest.json',result)
    return result


def build(raw_root,output,workers=4):
    raw_root=Path(raw_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((raw_root/'manifest.json').read_text());start=time.perf_counter(); results=[]
    save_json(output/'job.json',dict(pid=os.getpid(),workers=workers,status='RUNNING',raw_manifest_sha256=sha(raw_root/'manifest.json')))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(build_capture,raw_root,e,output/Path(e['name']).stem,m['route']):e for e in m['partitions']}
        for f in as_completed(futures):
            r=f.result();results.append(r)
            print(json.dumps(dict(capture=futures[f]['name'],records=r['records'],seconds=r['elapsed_seconds'])),flush=True)
    results.sort(key=lambda r:r['input']['first_ordinal'])
    result=dict(schema='MARKET_DATASET_V1',status='COMPLETE',raw_manifest_sha256=sha(raw_root/'manifest.json'),
                source_sha256=m['source_sha256'],route=m['route'],workers=workers,elapsed_seconds=time.perf_counter()-start,captures=results,
                continuity='source resets define capture boundaries; feature warmup restarts there',clock='provider receive research availability; not measured live latency')
    save_json(output/'manifest.json',result);save_json(output/'job.json',dict(pid=os.getpid(),status='COMPLETE'))
    return result
