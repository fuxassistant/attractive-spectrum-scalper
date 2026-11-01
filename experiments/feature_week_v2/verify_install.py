"""Run with pinned Python -I, outside the source tree, against an installed wheel."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import pkgutil
import sys
from zipfile import ZipFile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.site))
import spectrum
assert Path(spectrum.__file__).is_relative_to(a.site)
assert spectrum.__version__=='0.4.0'
expected={p.relative_to(a.root).as_posix() for p in (a.root/'spectrum').rglob('*') if p.suffix in ('.py','.json')}
with ZipFile(a.wheel) as wheel:
    packaged={n for n in wheel.namelist() if n.startswith('spectrum/') and n.endswith(('.py','.json'))}
assert packaged==expected,(packaged-expected,expected-packaged)
count=0
for item in pkgutil.walk_packages(spectrum.__path__,'spectrum.'):
    if not item.name.endswith('__main__'):
        importlib.import_module(item.name);count+=1
from spectrum.representation import open_session_store
import pandas as pd
store=open_session_store(a.root/'data/derived/feature_week_v2/final_parallel2/stores/2026-08-28')
o=store.observe(int(pd.Timestamp('2026-08-28 14:00:03.5',tz='UTC').value))
assert not o.market['known'] and o.market['reason']=='INVALID_BOOK'
assert len(o.second_features['names'])==480
import numpy as np
saved=np.load(a.root/'data/derived/feature_week_v2/audit/snapshots/20260828T140003.npz')
for group,tensor in [('fast',o.fast),('context',o.context),('second',o.second_features),('minute',o.minute_features)]:
    for name in ('values','known','normalized','t_ns','available_ns'):
        if name in tensor:np.testing.assert_array_equal(saved[group+'_'+name],tensor[name])
from spectrum.representation import open_observation_store
legacy=open_observation_store(a.root/'data/derived/feature_week_v1/serial_v2/stores/2026-08-24','2026-08-24')
assert len(legacy.observe(int(pd.Timestamp('2026-08-24 12:00',tz='UTC').value)).day['names'])==36
r=dict(status='PASS',version=spectrum.__version__,imported_modules=count,packaged_runtime_files=len(packaged),
       import_path=str(spectrum.__file__),cwd=str(Path.cwd()),isolated_python=bool(sys.flags.isolated),source_tree_import=False,
       real_session_loader=True,invalid_book_not_skipped=True,wheel_sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
       saved_v2_observation_bitwise_equal=True,legacy_v1_store_readable=True,
       first_build_disposition='Rejected stale build/lib included deleted legacy modules; clean build exactly matches source file inventory')
(a.root/'data/derived/feature_week_v2/audit/installed_package.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
