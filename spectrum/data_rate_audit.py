"""Exact wall-second publication rates, including quiet supported seconds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.ingest import save_json


def capture_rate(job):
    capture,reference=job
    q=pq.read_table(Path(capture['market'])/'quotes.parquet',columns=['available_ns']).column(0).to_numpy()
    date=capture['entry']['selection']['date'];start=pd.Timestamp(date,tz='UTC').value
    london=pd.Timestamp(date+' 08:00',tz='Europe/London').tz_convert('UTC').value
    ny=pd.Timestamp(date+' 17:00',tz='America/New_York').tz_convert('UTC').value
    second=(q-start)//10**9
    counts=np.bincount(second[(second>=0)&(second<86400)],minlength=86400)
    # Capture fragments are combined by date before computing medians/zero rates.
    return dict(date=date,key=capture['key'],symbol=capture['entry']['selection']['symbol'],
                counts=counts,london=london,ny=ny,start=start)


def main(output,workers=2):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    reference=np.load(config['reference']['path']);days={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(capture_rate,(r,config['reference']['path'])) for r in master['captures'] if r['status']=='COMPLETE']
        for future in as_completed(futures):
            r=future.result();date=r['date']
            if date in days:
                days[date]['counts']+=r['counts'];days[date]['keys'].append(r['key'])
            else:days[date]=dict(r,keys=[r['key']])
    global_counts=[];rows=[]
    for date,r in sorted(days.items()):
        t=r['start']+np.arange(86400,dtype=np.int64)*10**9
        ix=np.searchsorted(reference,t//(60*10**9))
        known=(ix<len(reference))&(reference[np.minimum(ix,len(reference)-1)]==t//(60*10**9))
        mask=known&(t>=r['london'])&(t<r['ny'])
        rate=r['counts'][mask];global_counts.append(rate)
        rows.append(dict(date=date,symbol=r['symbol'],capture_keys=r['keys'],seconds=len(rate),
            zero_seconds=int((rate==0).sum()),median=float(np.median(rate)) if len(rate) else None,
            mean=float(rate.mean()) if len(rate) else None,max=int(rate.max()) if len(rate) else None,
            known_reference_minutes=int(known.sum()//60)))
    rates=np.concatenate(global_counts);hist=np.bincount(rates)
    np.save(root/'publication_count_per_second_histogram.npy',hist)
    result=dict(status='COMPLETE',scope='Each UTC date combined across reset fragments; London08 through NewYork17; spot-reference-supported wall seconds, INCLUDING zero-update seconds',
        count=len(rates),zero_seconds=int((rates==0).sum()),mean=float(rates.mean()),
        quantiles={str(q):float(np.quantile(rates,q)) for q in (.01,.1,.5,.9,.95,.99,.999)},
        max=int(rates.max()),days=rows)
    save_json(root/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='days'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();main(a.output,a.workers)
