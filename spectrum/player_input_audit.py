"""Verify every cached player quote against its declared reconstruction inputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.core.storage import sha
from spectrum.ingest import save_json


def check(job):
    folder,captures,reference_path=job;folder=Path(folder)
    m=json.loads((folder/'manifest.json').read_text())
    bad_files=[name for name,digest in m['files'].items() if sha(folder/name)!=digest]
    parts=[];lineage_errors=[]
    for number,lineage in enumerate(m['current_lineage']):
        capture=captures[lineage['key']]
        if any(capture['market_payloads'][k]!=v for k,v in lineage['market_payloads'].items()):lineage_errors.append(lineage['key'])
        q=pq.read_table(Path(capture['market'])/'quotes.parquet',filters=[('t_ns','>=',int(m['start_ns'])),('t_ns','<',int(m['end_ns']))]).to_pandas()
        q['epoch']=q.epoch.to_numpy(np.int64)+number*1_000_000;parts.append(q)
    q=pd.concat(parts,ignore_index=True).sort_values(['available_ns','source_ordinal'],kind='stable')
    ref=np.load(reference_path,mmap_mode='r');minute=q.t_ns.to_numpy()//(60*10**9)
    ix=np.searchsorted(ref,minute);known=(ix<len(ref))&(ref[np.minimum(ix,len(ref)-1)]==minute)
    comparisons={}
    for array,column in dict(quote_t='available_ns',source_t='t_ns',bid='bid_nano',ask='ask_nano',source_ordinal='source_ordinal',epoch='epoch').items():
        comparisons[array]=bool(np.array_equal(np.load(folder/(array+'.npy'),mmap_mode='r'),q[column].to_numpy()))
    comparisons['valid_equals_book_and_reference']=bool(np.array_equal(np.load(folder/'valid.npy',mmap_mode='r'),q.valid.to_numpy()&known))
    t=np.load(folder/'decision_t.npy',mmap_mode='r');legal=np.load(folder/'can_enter.npy',mmap_mode='r')
    finite={name:bool(np.isfinite(np.load(folder/(name+'.npy'),mmap_mode='r')).all()) for name in ('x','fast_x','slow_x','anchors','event_x')}
    future={prefix:int((np.load(folder/(prefix+'_available.npy'),mmap_mode='r')<np.load(folder/(prefix+'_t.npy'),mmap_mode='r')).sum()) for prefix in ('fast','slow','event')}
    return dict(date=m['date'],quote_rows=len(q),file_hash_failures=bad_files,lineage_errors=lineage_errors,
        comparisons=comparisons,finite=finite,publication_before_source=future,
        entries_outside_hours=int((legal&((t<int(m['london_ns']))|(t>=int(m['ny_close_ns'])))).sum()),
        raw_valid_quotes_masked_by_reference=int((q.valid.to_numpy()&~known).sum()),
        london_ny_raw_valid_quotes_masked_by_reference=int((q.valid.to_numpy()&~known&(q.available_ns.to_numpy()>=int(m['london_ns']))&(q.available_ns.to_numpy()<int(m['ny_close_ns']))).sum()))


def main(output,workers=2):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    captures={r['key']:r for r in master['captures']}
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    paths=sorted(Path('data/research/player_v3/memory_001').glob('20*'))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(check,(str(p),captures,config['reference']['path'])) for p in paths]
        for f in as_completed(fs):
            r=f.result();rows.append(r);save_json(root/(r['date']+'.json'),r)
    errors=[r['date'] for r in rows if r['file_hash_failures'] or r['lineage_errors'] or
        not all(r['comparisons'].values()) or not all(r['finite'].values()) or any(r['publication_before_source'].values()) or r['entries_outside_hours']]
    result=dict(status='PASS' if not errors else 'FAIL',days=len(rows),rows=sum(r['quote_rows'] for r in rows),failures=errors,
        reference_masked_valid_quotes=sum(r['raw_valid_quotes_masked_by_reference'] for r in rows),
        london_ny_reference_masked_valid_quotes=sum(r['london_ny_raw_valid_quotes_masked_by_reference'] for r in rows),
        interpretation='PASS means adapter matches its declared rules; it does NOT endorse treating spot reference absence as invalid futures book')
    save_json(root/'summary.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();main(a.output,a.workers)
