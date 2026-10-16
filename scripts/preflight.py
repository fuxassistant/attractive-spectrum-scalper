import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, time
from collections import Counter
import numpy as np
import pandas as pd
from spectrum.source import DbnSource

root = Path(__file__).resolve().parents[1]
source = DbnSource(root/'data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.zst')
counts, days, hours, actions = Counter(), Counter(), Counter(), Counter()
first = {}; last = {}; resets = Counter(); n = 0
start = time.perf_counter()
for b in source:
    routes = np.stack([b['publisher_id'], b['instrument_id'], b['channel_id']], axis=1)
    vals, ns = np.unique(routes, axis=0, return_counts=True)
    counts.update({str(tuple(map(int,v))):int(c) for v,c in zip(vals,ns)})
    for route in vals:
        key = str(tuple(map(int,route)))
        sub = b[np.all(routes==route,axis=1)]
        first.setdefault(key, dict(time=int(sub['ts_recv'][0]),action=bytes(sub['action'][0]).decode(),flags=int(sub['flags'][0])))
        last[key] = int(sub['ts_recv'][-1])
        resets[key] += int(np.sum(sub['action']==b'R'))
    for field, div, counter in [('ts_recv',86400*10**9,days),('ts_recv',3600*10**9,hours)]:
        vs, cs = np.unique(b[field]//div,return_counts=True)
        counter.update({str(int(v)):int(c) for v,c in zip(vs,cs)})
    vals, cs = np.unique(b['action'],return_counts=True)
    actions.update({bytes(v).decode():int(c) for v,c in zip(vals,cs)})
    n += len(b)
    if n % 1048576 == 0: print(n,flush=True)
report = dict(records=n,seconds=time.perf_counter()-start,sha256=source.sha256,routes=counts,first=first,last=last,resets=resets,days=days,hours=hours,actions=actions)
parts=[]
for chunk in pd.read_csv(root/'data/raw/ctrader/OHLC/xauusd_m1_full.csv',chunksize=500000):
    x=chunk[(chunk.timestamp_ms>=1777766400000)&(chunk.timestamp_ms<1778889600000)]
    if len(x):parts.append(x)
df=pd.concat(parts,ignore_index=True)
df.to_csv(root/'reports/ohlc_may_two_weeks.csv',index=False)
t=pd.to_datetime(df.timestamp_ms,unit='ms',utc=True)
gaps=[]
for i in np.flatnonzero(np.diff(df.timestamp_ms)>60000):
    gaps.append(dict(last=str(t.iloc[i]),next=str(t.iloc[i+1]),missing_minutes=int((df.timestamp_ms.iloc[i+1]-df.timestamp_ms.iloc[i])/60000-1)))
report['ohlc']=dict(rows=len(df),first=str(t.iloc[0]),last=str(t.iloc[-1]),gaps=gaps)
(root/'reports/preflight.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ('days','hours')},indent=2))
