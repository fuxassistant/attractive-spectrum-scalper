"""Independent candle checks and observed invalid-support stress for the month."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import numpy as np
import pandas as pd
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from spectrum.full_history import verify_ohlc
from spectrum.player_v3.gym import advance_survival
from spectrum.player.contract import NS
from .data import MonthDay


def check_capture(job):
    root,date,seed=job
    return dict(date=date,root=root,**verify_ohlc(root,seed))


def run(data,output):
    root=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(root/'manifest.json');jobs={}
    for row in manifest['days']:
        m=read_json(root/row['date']/'manifest.json')
        for part in m['quote_parts']:jobs[part['key']]=(part['root'],row['date'],1500+len(jobs))
    with ProcessPoolExecutor(max_workers=3) as pool:candles=list(pool.map(check_capture,jobs.values()))
    if not all(x['status']=='PASS' for x in candles):raise ValueError('Candle audit did not pass')
    d=MonthDay(root/'2026-08-28');q=d.quote_t;lo=int(d.meta['london_ns']);hi=int(d.meta['ny_close_ns'])
    indices=np.flatnonzero((q>=lo)&(q<hi)&~d.valid)
    if not len(indices):raise ValueError('Expected raw-confirmed invalid-book episode missing')
    j=int(indices[0]);s=np.zeros(21,np.int64);f=np.zeros(9,float)
    s[0]=1;s[1]=d.ask[j-1];s[2]=q[j-1];s[3]=j-1;s[9]=q[j-1];s[10]=d.epoch[j-1];s[16]=1;s[17]=1;s[18]=q[j-1]
    f[:4]=[1499.85,1499.85,1500.,.15];f[5:]=[1.,1500.,1499.85,.0001]
    before=f.copy();plans=np.asarray([[1000000,1000000,600]]*3,np.int64)
    result=advance_survival(j,j+1,d.quote_t,d.source_t,d.bid,d.ask,d.valid,d.epoch,s,f,d.deadline,int(d.meta['ny_close_ns']),-1,.30,plans,np.empty(0),np.empty(0),np.empty(0,dtype=np.int64))
    assert result[0]<0 and s[11] and s[0]==1 and not s[12] and np.array_equal(f,before)
    aggregate=dict(status='PASS',month=manifest['month'],days=len(manifest['days']),
        activity_publications=sum(r['diagnostics']['quotes'] for r in manifest['days']),
        activity_invalid_publications=sum(r['diagnostics']['invalid_publications'] for r in manifest['days']),
        source_verification_sha256=sha(root/'source_verification.json'),
        candles=candles,total_candle_checks=sum(x['checks'] for x in candles),
        invalid_episode=dict(date='2026-08-28',first_ns=str(q[j]),first_utc=str(pd.Timestamp(q[j],tz='UTC')),count=len(indices),
            kernel_test='Position injected immediately before known invalid publication: no fill, no realized PnL, no death, position retained, equity unchanged.',
            raw_evidence='aug28_independent_book.json',exchange_status='UNDETERMINED: normalized raw events reproduce crossed book'),
        source_limitations='One real in-session invalid-book episode; an exposed position must be censored. Original counter warmup epochs retained for context, not used as execution loss signals.')
    save_json(out/'market_audit.json',aggregate);return aggregate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/derived/playground_august_1500');p.add_argument('--output',default='reports/playground_august_1500')
    a=p.parse_args();print(json.dumps(run(a.data,a.output),indent=2))
