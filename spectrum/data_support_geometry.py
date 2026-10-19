"""Measure gaps BEFORE applying core validity/epoch/age filters."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.data_properties_audit import Distribution
from spectrum.ingest import save_json


def capture(job):
    r,reference=job
    q=pq.read_table(Path(r['market'])/'quotes.parquet',columns=['t_ns','available_ns','valid','epoch','source_ordinal']).to_pandas()
    t=q.available_ns.to_numpy(np.int64);valid=q.valid.to_numpy(bool);ep=q.epoch.to_numpy(np.int64)
    ref=np.load(reference,mmap_mode='r');ix=np.searchsorted(ref,t//60000000000);safe=np.minimum(ix,len(ref)-1)
    known=(ix<len(ref))&(ref[safe]==t//60000000000)
    prefix=np.r_[0,np.cumsum(np.diff(ref)>1)]
    date=r['entry']['selection']['date'];lo=pd.Timestamp(date+' 08:00',tz='Europe/London').tz_convert('UTC').value;hi=pd.Timestamp(date+' 17:00',tz='America/New_York').tz_convert('UTC').value
    hours=(t>=lo)&(t<hi)
    intervals=known[1:]&known[:-1]&hours[1:]&hours[:-1]&(prefix[safe[1:]]==prefix[safe[:-1]])
    dt=np.diff(t);d=Distribution();d.add(dt[intervals])
    long=np.flatnonzero(intervals&(dt>30*10**9))+1
    examples=[dict(date=date,key=r['key'],left_ns=str(int(t[j-1])),right_ns=str(int(t[j])),seconds=float(dt[j-1]/10**9),
        valid_before=bool(valid[j-1]),valid_after=bool(valid[j]),epoch_changed=bool(ep[j-1]!=ep[j]),
        left_ordinal=str(int(q.source_ordinal.iloc[j-1])),right_ordinal=str(int(q.source_ordinal.iloc[j]))) for j in long]
    return dict(key=r['key'],invalid_inside_known_reference=int((~valid&known).sum()),
        invalid_london_ny_known_reference=int((~valid&known&hours).sum()),
        epoch_changes_london_ny_known_reference=int((intervals&(ep[1:]!=ep[:-1])).sum()),
        interval_stats=d.result(),hist=d.hist,gaps_over_30s=examples)


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text());config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    rows=[];hist=np.zeros(8194,np.int64)
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(capture,(r,config['reference']['path'])) for r in master['captures'] if r['status']=='COMPLETE']
        for f in as_completed(fs):
            r=f.result();hist+=r.pop('hist');rows.append(r)
    np.save(out/'unconditioned_interarrival_hist.npy',hist)
    result=dict(status='COMPLETE',invalid_inside_known_reference=sum(r['invalid_inside_known_reference'] for r in rows),
        invalid_london_ny_known_reference=sum(r['invalid_london_ny_known_reference'] for r in rows),
        epoch_changes_london_ny_known_reference=sum(r['epoch_changes_london_ny_known_reference'] for r in rows),
        gaps_over_30s=[g for r in rows for g in r['gaps_over_30s']],captures=rows,
        semantics='Trading window and known spot reference only; NO valid-book, epoch-continuity or max-age selection. No gap is automatically labelled source loss.')
    save_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('captures','gaps_over_30s')},indent=2));print('Long gaps:',len(result['gaps_over_30s']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
