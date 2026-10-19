"""Exact multiset difference of snapshot records in the old May-week extract."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
from spectrum.source import DbnSource
from spectrum.core.vendor.glbx_dbn import MBO_DTYPE
from spectrum.ingest import save_json


def main():
    start,end=1777852800000000000,1778371200000000000
    sources=['data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.zst','data/raw/GC_MBO_2026-05-04_to_2026-05-09.dbn.zst']
    counters=[]
    for path in sources:
        counter=Counter()
        for block in DbnSource(path):
            rows=block[(block['ts_recv']>=start)&(block['ts_recv']<end)&((block['flags']&32)!=0)]
            counter.update(row.tobytes() for row in rows)
            if int(block['ts_recv'][-1])>=end:break
        counters.append(counter)
    missing=counters[0]-counters[1];extra=counters[1]-counters[0]
    rows=np.frombuffer(b''.join(value*count for value,count in missing.items()),dtype=MBO_DTYPE)
    out=Path('reports/data_properties_full_001');np.save(out/'missing_legacy_snapshot_records.npy',rows)
    groups=Counter((str(pd.Timestamp(int(row['ts_recv']),tz='UTC').date()),int(row['instrument_id']),int(row['channel_id'])) for row in rows)
    result=dict(status='CONFIRMED_SNAPSHOT_LOSS' if not extra else 'SNAPSHOT_CONTENT_DIFFERENCES',
        missing_records=len(rows),unexpected_legacy_records=sum(extra.values()),
        selected_GCM6_instrument_19181_missing=int((rows['instrument_id']==19181).sum()),
        by_date_instrument=[dict(date=k[0],instrument_id=k[1],channel_id=k[2],missing=v) for k,v in sorted(groups.items())],
        non_snapshot_record_stream='Verified byte-identical separately in legacy_snapshot_comparison.json',
        impact='Legacy extraction defect; current full_mbo_v1 and memory_001 use canonical monthly archives, independently verified')
    save_json(out/'legacy_snapshot_loss.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='by_date_instrument'}))


if __name__=='__main__':main()
