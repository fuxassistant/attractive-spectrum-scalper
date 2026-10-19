"""Trade-side/clock census and bounded independent order-book cross-checks."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pyarrow.parquet as pq
from spectrum.source import DbnSource
from spectrum.ingest import save_json


class ReferenceBook:
    """Small independent dictionary book; no production reconstruction imports."""
    def __init__(self, expected):
        self.orders = {}
        self.levels = {b'B': Counter(), b'A': Counter()}
        self.expected = expected
        self.rows = self.checks = 0
        self.errors = []
        self.unknown_cancels = self.unknown_modifies = 0

    def remove(self, oid, size=None):
        side, price, old_size = self.orders[oid]
        quantity = old_size if size is None else min(size, old_size)
        self.levels[side][price] -= quantity
        if not self.levels[side][price]: del self.levels[side][price]
        if quantity == old_size: del self.orders[oid]
        else: self.orders[oid] = (side, price, old_size-quantity)

    def feed(self, record, ordinal):
        self.rows += 1
        action, side = bytes(record['action']), bytes(record['side'])
        oid, price, size = (int(record[k]) for k in ('order_id','price','size'))
        if action == b'R':
            self.orders.clear(); self.levels = {b'B':Counter(), b'A':Counter()}
        elif action in (b'A',b'M'):
            if action == b'A' and oid in self.orders:
                self.errors.append(dict(ordinal=str(ordinal),reason='DUPLICATE_ADD'))
            if action == b'M' and oid not in self.orders: self.unknown_modifies += 1
            if oid in self.orders: self.remove(oid)
            if size > 0 and side in self.levels:
                self.orders[oid] = (side,price,size);self.levels[side][price] += size
        elif action == b'C':
            if oid in self.orders: self.remove(oid,size)
            else: self.unknown_cancels += 1
        if ordinal in self.expected:
            expected=self.expected[ordinal]
            if expected['valid']:
                bid=max(self.levels[b'B'],default=0);ask=min(self.levels[b'A'],default=0)
                actual=(bid,ask,self.levels[b'B'].get(bid,0),self.levels[b'A'].get(ask,0))
                target=tuple(int(expected[k]) for k in ('bid_nano','ask_nano','bid_size','ask_size'))
                self.checks += 1
                if actual != target:
                    self.errors.append(dict(ordinal=str(ordinal),actual=actual,expected=target))


def scan(job):
    member,captures,probe_keys,output=job
    started=time.perf_counter();states={}
    for r in captures:
        ref=None
        if r['key'] in probe_keys and r['status']=='COMPLETE':
            table=next(pq.ParquetFile(Path(r['market'])/'quotes.parquet').iter_batches(batch_size=20000)).to_pandas()
            ref=ReferenceBook({int(row['source_ordinal']):row for row in table.to_dict('records')})
        states[r['key']]=dict(entry=r['entry'],records=0,trade_count=Counter(),trade_volume=Counter(),
            fill_count=Counter(),fill_volume=Counter(),mutation_undefined_prices=0,zero_receive=0,
            recv_regressions=0,event_regressions=0,sequence_regressions=0,sequence_forward_jumps=0,
            previous=None,reference=ref,negative_recv_event=0)
    offset=0
    for block in DbnSource(member['archive'],member['member'],block_records=524288):
        end=offset+len(block)
        for key,state in states.items():
            e=state['entry']
            if int(e['last_ordinal'])<offset or int(e['first_ordinal'])>=end:continue
            lo,hi=max(0,int(e['first_ordinal'])-offset),min(len(block),int(e['last_ordinal'])-offset+1)
            part=block[lo:hi];route=e['route']
            indices=np.flatnonzero((part['publisher_id']==route[0])&(part['instrument_id']==route[1])&(part['channel_id']==route[2]))
            a=part[indices];ordinals=indices+lo+offset;state['records']+=len(a)
            for action,prefix in ((b'T','trade'),(b'F','fill')):
                selected=a[a['action']==action]
                for side in (b'A',b'B',b'N'):
                    rows=selected[selected['side']==side]
                    state[prefix+'_count'][side.decode()]+=len(rows)
                    state[prefix+'_volume'][side.decode()]+=int(rows['size'].sum())
            state['mutation_undefined_prices']+=int(((a['price']==2**63-1)&np.isin(a['action'],[b'A',b'C',b'M'])).sum())
            live=a[(a['flags']&32)==0]
            if len(live):
                recv=live['ts_recv'].astype(np.int64);event=live['ts_event'].astype(np.int64);seq=live['sequence'].astype(np.int64)
                previous=state['previous']
                for values,index,name in ((recv,0,'recv_regressions'),(event,1,'event_regressions'),(seq,2,'sequence_regressions')):
                    delta=np.diff(values,prepend=previous[index]) if previous else np.diff(values)
                    state[name]+=int((delta<0).sum())
                    if index==2:state['sequence_forward_jumps']+=int((delta>1).sum())
                state['zero_receive']+=int((recv==0).sum())
                state['negative_recv_event']+=int((recv<event).sum())
                state['previous']=(int(recv[-1]),int(event[-1]),int(seq[-1]))
            book=state['reference']
            if book is not None and book.rows<20000:
                for record,ordinal in zip(a[:20000-book.rows],ordinals):book.feed(record,int(ordinal))
        offset=end
    rows=[]
    for key,s in states.items():
        book=s.pop('reference');s.pop('previous');entry=s.pop('entry')
        row=dict(key=key,date=entry['selection']['date'],symbol=entry['selection']['symbol'],**s)
        if book:
            row['reference_book']=dict(scope='First 20000 selected raw records from explicit reset; not a full-history independent rebuild',
                raw_rows=book.rows,valid_publications_checked=book.checks,errors=book.errors,
                unknown_cancels=book.unknown_cancels,unknown_modifies=book.unknown_modifies)
        rows.append(row)
    result=dict(member=member['key'],all_source_records=offset,captures=rows,seconds=time.perf_counter()-started,
                sequence_note='Forward jumps are measured, not labelled packet loss: selected instrument/universe does not expose every channel message')
    save_json(Path(output)/(member['key']+'.json'),result)
    return dict(member=member['key'],seconds=result['seconds'])


def run(output,workers):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    # One complete capture per monthly member, plus three August days; fixed before inspection.
    keys=[]
    chosen=['member_'+str(i).zfill(3) for i in [*range(10),11,21,35]]
    for member in chosen:
        candidates=[r for r in master['captures'] if r['status']=='COMPLETE' and r['entry'].get('member_key')==member]
        if candidates:keys.append(candidates[len(candidates)//2]['key'])
    save_json(out/'protocol.json',dict(probe_keys=keys,workers=workers,scope='All selected raw records: trade sides, volumes, clocks; bounded independent book probes'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(scan,(m,[r for r in master['captures'] if r['entry'].get('member_key')==m['key']],keys,str(out))) for m in config['members']]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    save_json(out/'scan_complete.json',dict(status='COMPLETE',members=len(config['members'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.output,a.workers)
