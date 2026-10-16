"""Real-prefix differential replay, restart, slow book oracle and decoder timing."""
import sys,time,json,statistics
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spectrum.ingest import PACKED,packed_records,save_json
from spectrum.core import StreamConfig
from spectrum.core.vendor.l3 import MboRecord
from spectrum.replay import MarketReplay


def scalar_records(a,route):
    for item in a:
        r=item['record']
        yield MboRecord(int(item['source_ordinal']),*route,int(r['ts_event']),int(r['ts_recv']),int(r['sequence']),
                        int(r['order_id']),int(r['price']),int(r['size']),int(r['flags']),bytes(r['action']).decode(),bytes(r['side']).decode())


def slow_books(records):
    # Independent plain dictionary, recomputed BBO at every completed group.
    # Scoped to a clean first 20k prefix; unsupported anomalies fail explicitly.
    orders={};out=[];ready=False;snapshot=False;seen=False
    for r in records:
        if r.action=='R':orders={};snapshot=bool(r.flags&32);ready=not snapshot;seen=False
        elif r.flags&32:snapshot=True;ready=False
        if r.flags&32:seen=True
        if r.action in ('A','M'):
            old=orders.get(r.order_id)
            if r.action=='A' and old is not None:raise AssertionError('duplicate add')
            if old is not None and r.action=='M' and old[0]!=r.side:raise AssertionError('unsupported side change')
            if r.size:orders[r.order_id]=(r.side,r.price_nano,r.size)
            elif old is not None:del orders[r.order_id]
        elif r.action=='C':
            side,price,size=orders[r.order_id]
            if r.size>size:raise AssertionError('over cancel')
            if r.size==size:del orders[r.order_id]
            else:orders[r.order_id]=(side,price,size-r.size)
        if not r.flags&128:continue
        if snapshot and seen:snapshot=False;ready=True;seen=False
        bids={};asks={}
        for side,price,size in orders.values():
            target=bids if side=='B' else asks;target[price]=target.get(price,0)+size
        bid=max(bids,default=0);ask=min(asks,default=0)
        valid=ready and not snapshot and bid>0 and ask>bid and not r.flags&12
        out.append((r.source_ordinal,bid if valid else 0,ask if valid else 0,bids.get(bid,0) if valid else 0,asks.get(ask,0) if valid else 0,bool(valid)))
    return out


if __name__=='__main__':
    raw=Path('data/derived/raw_v1');m=json.loads((raw/'manifest.json').read_text());route=m['route']
    a=np.memmap(raw/'capture_001.bin',dtype=PACKED,mode='r')[:30000]
    baseline=list(scalar_records(a,route));fast=list(packed_records(a,route))
    assert baseline==fast
    timing={'scalar':[],'columnar':[]}
    for i in range(5):
        for label,fn in ([('scalar',scalar_records),('columnar',packed_records)] if i%2==0 else [('columnar',packed_records),('scalar',scalar_records)]):
            t=time.perf_counter();r=list(fn(a,route));timing[label].append(time.perf_counter()-t);assert r==baseline
    end=20000
    # Make the endpoint an actual F_LAST without changing any original record.
    while not fast[end-1].flags&128:end+=1
    records=fast[:end];outputs=[];snapshots=[];bars=[]
    p=MarketReplay(StreamConfig(*route,snapshot_ns=10**9),on_quote=outputs.append,on_snapshot=snapshots.append,on_bar=bars.append)
    p.feed_batch(records)
    got=[]
    q=MarketReplay(StreamConfig(*route,snapshot_ns=10**9))
    for r in records:
        q.feed(r)
        if r.flags&128:
            b=q.core.accumulator.book
            got.append((r.source_ordinal,b.best_bid_nano,b.best_ask_nano,b.bid_size_l1,b.ask_size_l1,b.valid))
    oracle=slow_books(records)
    assert got==oracle
    restored_outputs=[];restored_snaps=[];restored_bars=[]
    z=MarketReplay(StreamConfig(*route,snapshot_ns=10**9),on_quote=restored_outputs.append,on_snapshot=restored_snaps.append,on_bar=restored_bars.append)
    cut=next(i for i,r in enumerate(records[1000:],1000) if not r.flags&128)+1
    z.feed_batch(records[:cut]);saved=json.loads(json.dumps(z.checkpoint()))
    z=MarketReplay.restore(saved,on_quote=restored_outputs.append,on_snapshot=restored_snaps.append,on_bar=restored_bars.append)
    for lo in range(cut,len(records),137):z.feed_batch(records[lo:lo+137])
    for left,right in ((outputs,restored_outputs),(snapshots,restored_snaps),(bars,restored_bars)):
        np.testing.assert_array_equal(np.array(left),np.array(right))
    result=dict(status='PASS',source_manifest_sha256=__import__('spectrum.core.storage',fromlist=['sha']).sha(raw/'manifest.json'),
                records=len(records),publications=len(oracle),independent_oracle='dictionary orders and full recomputation of levels; clean raw prefix',
                decoder_equal_records=len(baseline),decoder_seconds=timing,
                median_decoder_speedup=statistics.median(timing['scalar'])/statistics.median(timing['columnar']),
                restart_cut=cut,restart_inside_F_LAST=True,batch_size=137,quote_snapshot_bar_parity='PASS',
                timing_scope='decoder only; interleaved repeats; machine may be generating full captures concurrently')
    save_json('reports/foundation_verification.json',result);print(json.dumps(result,indent=2))
