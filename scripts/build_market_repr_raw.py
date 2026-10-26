"""Re-extract exact T prices and per-price order facts from bound archive members."""
import sys,os,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from spectrum.source import DbnSource
from spectrum.ingest import PACKED,packed_records
from spectrum.core.vendor.l3_v2 import L3EventEngineV2
from spectrum.market_repr.raw import InteractionCollector
from scripts.build_crossing_path_target_v6 import sha,save


class Writer:
    def __init__(self,path):self.path=Path(path);self.rows=[];self.writer=None;self.n=0
    def add(self,row):
        self.rows.append(row)
        if len(self.rows)>=20000:self.flush()
    def add_many(self,rows):
        self.rows.extend(rows)
        if len(self.rows)>=20000:self.flush()
    def flush(self):
        if not self.rows:return
        table=pa.Table.from_pandas(pd.DataFrame(self.rows),preserve_index=False)
        if self.writer is None:self.writer=pq.ParquetWriter(self.path,table.schema,compression='zstd')
        self.writer.write_table(table);self.n+=len(self.rows);self.rows=[]
    def close(self):
        self.flush()
        if self.writer:self.writer.close()


def build(date):
    root=Path('data/derived/full_mbo_v1');out=Path('data/derived/market_representation_v1/raw')/date
    out.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json';config=root/'run_config.json'
    catalog=json.loads(manifest.read_text());cfg=json.loads(config.read_text())
    entries=[r for r in catalog['captures'] if r['status']=='COMPLETE' and r['entry']['selection']['date']==date and not r['entry']['selection']['context_only']]
    if len(entries)!=1:raise ValueError('Expected one declared reset capture for '+date)
    source=entries[0];entry=source['entry'];item=next(x for x in cfg['members'] if x['key']==entry['member_key'])
    receipt=out/'manifest.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get('status')=='COMPLETE':
            if old['implementation']!={p:sha(p) for p in old['implementation']}:raise ValueError('Extractor changed; use fresh output')
            for n,h in old['outputs'].items():assert sha(out/n)==h
            return {'date':date,'status':'REUSED','rows':old['rows']}
    start=time.perf_counter();save(receipt,dict(status='RUNNING',pid=os.getpid(),date=date))
    route=tuple(entry['route']);engine=L3EventEngineV2(publisher_id=route[0],instrument_id=route[1],channel_id=route[2])
    levels=Writer(out/'levels_1s.partial.parquet');trades=Writer(out/'trades.partial.parquet')
    collector=InteractionCollector(levels.add_many,trades.add)
    offset=selected=0;digest=hashlib.sha256();src=DbnSource(item['archive'],item['member'])
    try:
        for block in src:
            ids=np.flatnonzero((block['publisher_id']==route[0])&(block['instrument_id']==route[1])&(block['channel_id']==route[2]))
            ids=ids[(ids+offset>=entry['first_ordinal'])&(ids+offset<=entry['last_ordinal'])]
            packed=np.empty(len(ids),PACKED);packed['source_ordinal']=ids+offset;packed['record']=block[ids]
            digest.update(packed.tobytes())
            for r in packed_records(packed,route):
                event=engine.append(r)
                if event is not None:collector.feed(event,engine)
            selected+=len(ids);offset+=len(block)
            save(out/'progress.json',dict(pid=os.getpid(),date=date,selected=selected,total=entry['records'],seconds=time.perf_counter()-start,events=collector.stats['events']))
        assert selected==entry['records'] and digest.hexdigest()==entry['sha256']
        assert src.sha256==item['sha256'] and not engine.pending
        collector.finish()
    finally:levels.close();trades.close()
    for name in ('levels_1s','trades'):(out/(name+'.partial.parquet')).replace(out/(name+'.parquet'))
    result=dict(status='COMPLETE',date=date,symbol=entry['selection']['symbol'],route=list(route),pid=os.getpid(),
        rows={'levels':levels.n,'trades':trades.n},stats=collector.stats,elapsed_seconds=time.perf_counter()-start,
        source={'archive':item['archive'],'member':item['member'],'member_sha256':src.sha256,'partition_sha256':digest.hexdigest(),'records':selected,'first_ordinal':entry['first_ordinal'],'last_ordinal':entry['last_ordinal']},
        catalog_hashes={str(manifest):sha(manifest),str(config):sha(config)},
        outputs={n:sha(out/n) for n in ('levels_1s.parquet','trades.parquet')},
        implementation={p:sha(p) for p in ('spectrum/market_repr/raw.py','scripts/build_market_repr_raw.py','spectrum/core/event_facts.py','spectrum/core/vendor/l3.py','spectrum/core/vendor/l3_v2.py')},
        semantics={'withdrawal':'unlinked source decrement, not participant intention or proven cancellation',
          'refill':'visible addition matched FIFO to prior linked fill at same price within 2s; not proof of same participant',
          'age':'known observed order birth only; snapshot lifetimes left-censored',
          'stock_check':'mutation ledger against reconstructed engine level stocks; source-partition SHA verified',
          'availability':'second totals published only at the first subsequent complete event; final partial second excluded'})
    save(receipt,result)
    return {'date':date,'status':'COMPLETE','seconds':result['elapsed_seconds'],'rows':result['rows'],'stock_residual':collector.stats['max_stock_residual']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dates',nargs='+',required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(build,d) for d in a.dates]):print(json.dumps(f.result()),flush=True)
