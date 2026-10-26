"""Reusable formulas extracted from spectrum/lab.py; see docs/MIGRATION.json."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json, time, os, importlib, inspect
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from spectrum.market.calendar import MarketCalendar
from spectrum.core.storage import sha
from spectrum.io import save_json
from .contract import Context, FeaturePlugin, rolling_valid
from .families.motion import motion
from .families.flow import flow
from .families.spectral import spectral

REGISTRY={p.name:p for p in (FeaturePlugin('motion','1',motion,128),FeaturePlugin('flow','1',flow,30),FeaturePlugin('spectral','1',spectral,128))}


def register(plugin):
    if plugin.name in REGISTRY:raise ValueError('Plugin already registered')
    REGISTRY[plugin.name]=plugin


def load_plugins(specs):
    identities={}
    for spec in specs:
        module,attribute=spec.split(':',1)
        loaded=importlib.import_module(module);plugin=getattr(loaded,attribute)
        if not isinstance(plugin,FeaturePlugin):raise TypeError('Expected FeaturePlugin')
        if plugin.name not in REGISTRY:register(plugin)
        elif REGISTRY[plugin.name]!=plugin:raise ValueError('Plugin name collision')
        identities[spec]=dict(name=plugin.name,version=plugin.version,sha256=sha(inspect.getsourcefile(loaded)))
    return identities


def compute(frame,families=('motion','flow','spectral'),calendar=None):
    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(n not in REGISTRY for n in families):
        raise ValueError('Unknown, empty or duplicate feature family selection')
    c=Context(frame,calendar or MarketCalendar())
    for name in families:REGISTRY[name].compute(c)
    return c


def compute_batch(frame, families=('motion', 'flow', 'spectral'), calendar=None):
    """The same formulas as FeatureStream, returned through the shared contract."""
    from .contract import FeatureBatch
    context = compute(frame, families, calendar)
    return FeatureBatch.from_columns(frame.t_ns.to_numpy(), frame.available_ns.to_numpy(),
                                     context.values, context.masks, context.units)


def generate_capture(capture,output,families,plugins=()):
    plugin_sources=load_plugins(plugins)
    start=time.perf_counter();capture=Path(capture);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((capture/'manifest.json').read_text())
    if m['status']!='COMPLETE' or sha(capture/'seconds.parquet')!=m['payloads']['seconds.parquet']:raise ValueError('Market payload drift')
    frame=pq.read_table(capture/'seconds.parquet').to_pandas();cal=MarketCalendar();c=compute(frame,families,cal)
    columns={k:frame[k].to_numpy() for k in ('t_ns','available_ns','epoch')}
    columns.update(scheduled_closed=c.closed,admissible=c.admitted)
    for name,v in c.values.items():columns[name]=v;columns[name+'__known']=c.masks[name]
    table=pa.table(columns);pq.write_table(table,output/'features.parquet',compression='zstd',row_group_size=32768)
    read=pq.read_table(output/'features.parquet')
    if not read.equals(table):raise ValueError('Parquet roundtrip differs')
    r=dict(status='COMPLETE',rows=len(frame),features=len(c.values),families={n:REGISTRY[n].version for n in families},units=c.units,
           lookback_seconds=max(REGISTRY[n].lookback_seconds for n in families),calendar=cal.contract(),
           market_manifest_sha256=sha(capture/'manifest.json'),source_seconds_sha256=m['payloads']['seconds.parquet'],
           implementation_sha256=implementation_identity(),plugin_sources=plugin_sources,calendar_implementation_sha256=sha(Path(inspect.getfile(MarketCalendar))),
           feature_sha256=sha(output/'features.parquet'),coverage={n:float(mask.mean()) for n,mask in c.masks.items()},
           elapsed_seconds=time.perf_counter()-start,rss_sample=psutil.Process().memory_info().rss,pid=os.getpid(),
           known='false-mask values are technical zero, not observed inactivity',roundtrip='PASS')
    save_json(output/'manifest.json',r);return r


def generate(market,output,families,workers=4,plugins=()):
    load_plugins(plugins)
    if not families or len(set(families))!=len(families) or any(n not in REGISTRY for n in families):raise ValueError('Unknown/duplicate feature family')
    market=Path(market);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((market/'manifest.json').read_text());start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(generate_capture,market/Path(e['input']['name']).stem,output/Path(e['input']['name']).stem,families,plugins):e for e in m['captures']}
        for f in as_completed(jobs):
            r=f.result();r['capture']=Path(jobs[f]['input']['name']).stem;results.append(r)
            print(json.dumps(dict(capture=r['capture'],seconds=r['elapsed_seconds'],rows=r['rows'])),flush=True)
    results.sort(key=lambda r:r['capture'])
    result=dict(status='COMPLETE',captures=results,elapsed_seconds=time.perf_counter()-start,workers=workers,families=families,
                market_manifest_sha256=sha(market/'manifest.json'),independent_jobs='isolated process per complete reset capture, no shared book')
    save_json(output/'manifest.json',result);return result


def implementation_identity():
    """Fingerprint all packaged feature formulas and the publication calendar."""
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob('*.py'))
    identities = {str(p.relative_to(root)): sha(p) for p in files}
    identities['market/calendar.py'] = sha(Path(inspect.getfile(MarketCalendar)))
    identities['calendar_contract'] = MarketCalendar().contract()
    for name in ('leakage.py', 'units.py'):
        identities[name] = sha(root.parent / name)
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


from .families.depth_balance import PLUGIN as _depth
from .families.volatility_vwap import PROXY as _vwap_proxy, EXACT as _vwap_exact
for _plugin in (_depth, _vwap_proxy, _vwap_exact):
    register(_plugin)
