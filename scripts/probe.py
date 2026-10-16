import sys,time,cProfile,pstats
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spectrum.source import DbnSource
from spectrum.core import MboStream,StreamConfig
from spectrum.core.adapter import records_from_block
path='data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.zst'
c=Counter()
for b in DbnSource(path):
    v,n=np.unique(b['instrument_id'],return_counts=True)
    c.update(dict(zip(map(int,v),map(int,n))))
    if sum(c.values())>500000:break
print(c.most_common(8),flush=True)
route=(1,c.most_common(1)[0][0],22)
core=MboStream(StreamConfig(*route,snapshot_ns=10**9),features=False)
t=time.perf_counter();offset=0;p=cProfile.Profile();p.enable()
for b in DbnSource(path):
    core.feed_many(records_from_block(b,route,offset));offset+=len(b)
    if core.engine.selected_records>=100000:break
p.disable()
print(route,core.engine.selected_records,time.perf_counter()-t,core.accumulator.reason_counts,flush=True)
pstats.Stats(p).sort_stats('tottime').print_stats(16)
p.dump_stats('reports/reference_profile.prof')
