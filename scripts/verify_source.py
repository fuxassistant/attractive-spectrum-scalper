"""Verify the local May subset against its original compressed ZIP member."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectrum.source import DbnSource
from spectrum.ingest import save_json
import pandas as pd

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    declared=json.loads((root/'data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.manifest.json').read_text())
    lo,hi=[pd.Timestamp(x).value for x in declared['interval_utc']]
    parent=DbnSource(root/'data/raw/GLBX-20260805-FTTJQEK3T3.zip',declared['source_member'])
    h=hashlib.sha256();n=0;start=time.perf_counter()
    for b in parent:
        a=b[(b['ts_recv']>=lo)&(b['ts_recv']<hi)]
        h.update(a.tobytes());n+=len(a)
    child=DbnSource(root/'data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.zst')
    k=hashlib.sha256()
    for b in child:k.update(b.tobytes())
    result=dict(status='PASS' if h.hexdigest()==k.hexdigest() and parent.sha256==declared['source_member_sha256'] and n==child.count else 'FAIL',
                selected_parent_records=n,subset_records=child.count,parent_records=parent.count,
                original_member_sha256=parent.sha256,declared_original_sha256=declared['source_member_sha256'],
                subset_compressed_sha256=child.sha256,parent_selected_wire_sha256=h.hexdigest(),subset_wire_sha256=k.hexdigest(),
                seconds=time.perf_counter()-start,boundary='ts_recv in declared [start,end); exact 56-byte rows in original order')
    save_json(root/'reports/source_lineage.json',result);print(json.dumps(result,indent=2))
