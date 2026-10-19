"""Live inventory and real-source stream parity for the VWAP integration pilot."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from spectrum.core.feature_families import default_families
from spectrum.core.storage import sha
from spectrum.lab import Context,compute,load_plugins,REGISTRY
from spectrum.feature_stream import FeatureStream
from spectrum.full_history import ObservedCalendar
from spectrum.plugins.volatility_vwap import PROXY,EXACT_COLUMNS
from spectrum.ingest import save_json

root=Path('data/derived/full_mbo_v1');out=Path('reports/volatility_vwap_v1/pilot_002')
config=json.loads((root/'run_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
summary=json.loads((out/'summary.json').read_text());key=next(r['key'] for r in summary['reports'] if r['date'].startswith('2026-08'))
r=next(r for r in manifest['captures'] if r['key']==key)
f=pq.read_table(Path(r['market'])/'seconds.parquet').to_pandas().iloc[:16000]
cal=ObservedCalendar(config['reference']['path'],config['reference']['sha256'])
specs=('spectrum.plugins.volatility_vwap:PROXY',);load_plugins(specs)
whole=compute(f,[PROXY.name],cal);stream=FeatureStream([PROXY.name],plugins=specs,calendar=cal,source_id=key)
pieces=[stream.feed(f.iloc[i:i+4096]) for i in range(0,len(f),4096)]
errors={}
for name,values in whole.values.items():
    joined=np.concatenate([p['values'][name] for p in pieces]);masks=np.concatenate([p['known'][name] for p in pieces])
    np.testing.assert_array_equal(masks,whole.masks[name])
    np.testing.assert_allclose(joined,values,rtol=1e-7,atol=1e-7)
    errors[name]=float(np.max(abs(joined-values)))
names=pq.read_schema(Path(r['features'])/'features.parquet').names
base=compute(f.iloc[:400],['motion','flow','spectral'],cal)
counts={}
for family in ('motion','flow','spectral'):
    c=Context(f.iloc[:400],cal);REGISTRY[family].compute(c);counts[family]=dict(count=len(c.values),features=list(c.values))
inventory=dict(status='VERIFIED_CURRENT_CODE_AND_SAVED_SCHEMA',source_capture=key,
    saved_feature_count=sum(not n.endswith('__known') and n in base.values for n in names),active_families=counts,
    core_available_not_in_182={family.name:dict(count=len(family.fields),features=[f.name for f in family.fields]) for family in default_families()},
    response_lab=dict(return_std_seconds=[30,128,512],cross_spectra_window=512,welch_segment=128,period_bands_seconds=[[4,8],[8,16],[16,32]]),
    event_lab=dict(baseline='std(ddof=1) of previous120 five-second price changes; 10min history',peak_energy='largest5s squared-change share over512s'),
    existing_exact_moments=[name for name in EXACT_COLUMNS if name in f.columns],
    source_code_hashes={str(p):sha(p) for p in [Path('spectrum/lab.py'),Path('spectrum/core/price_history.py'),Path('spectrum/core/feature_families.py'),Path('spectrum/plugins/volatility_vwap.py')]})
save_json(out/'inventory.json',inventory)
save_json(out/'verification.json',dict(status='PASS',tests='87 passed',real_stream=dict(capture=key,rows=len(f),chunks=4,
    max_abs_error=max(errors.values()),tolerance='rtol=1e-7,atol=1e-7; masks exact',
    calendar='ObservedCalendar',checkpoint='Not exercised for ObservedCalendar: generic adapter restore only constructs MarketCalendar.'),
    independent_central_moment_checks=summary['independent_checks'],max_sigma_error_ticks=summary['max_sigma_error'],
    pngs_directly_viewed=['png/scene_07.png','png/scene_10.png'],
    event_features_sha256=sha(out/'event_features.parquet'),plugin_sha256=sha('spectrum/plugins/volatility_vwap.py'),
    output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))
print(json.dumps(dict(saved=inventory['saved_feature_count'],counts={k:v['count'] for k,v in counts.items()},real_stream_max_error=max(errors.values()),missing_exact=inventory['existing_exact_moments'])))
