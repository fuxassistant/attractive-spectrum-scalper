"""Read-only full-history integrity, admission and calendar audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from spectrum.full_history import assert_code,compatible_identity,training_assessment,ObservedCalendar,verify_ohlc,now
from spectrum.core.storage import sha
from spectrum.ingest import save_json

root=Path('data/derived/full_mbo_v1');out=Path('reports/full_mbo_v1/audit_20261016');out.mkdir(exist_ok=True)
start=time.perf_counter();issues=[]
def check(ok,kind,detail):
    if not ok:issues.append(dict(kind=kind,detail=str(detail)))
config=json.loads((root/'run_config.json').read_text());assert_code(config)
manifest=json.loads((root/'manifest.json').read_text());catalog=json.loads((root/'training_catalog.json').read_text())
receipts={r['key']:r for r in manifest['captures']};admission={r['key']:r for r in catalog['captures']}
expected={};members=[]
for item in config['members']:
    s=json.loads((root/'staging'/item['key']/'manifest.json').read_text())
    check(s['item']==item and s['source_member_sha256']==item['sha256'],'source_identity',item['key'])
    check(compatible_identity(config,s['code_identity']),'stage_code',item['key'])
    for e in s['entries']:expected[e['member_key']+'/'+Path(e['name']).stem]=e
    members.append(dict(key=item['key'],member=item['member'],raw_records=s['raw_records'],selected_records=s['selected_records'],captures=len(s['entries'])))
check(set(expected)==set(receipts)==set(admission),'catalog_set',set(expected)^set(receipts))
hashes=[];metadata=[]
for key,r in receipts.items():
    check(r['entry']==expected[key],'entry_identity',key)
    check(json.loads((root/'receipts'/(key.replace('/','_')+'.json')).read_text())==r,'receipt_catalog',key)
    check(compatible_identity(config,r['code_identity']),'receipt_code',key)
    check(admission[key]['eligible']==training_assessment(r)['eligible'] and admission[key]['reasons']==training_assessment(r)['reasons'],'admission',key)
    market=Path(r['market']);features=Path(r['features'])
    hashes.extend([(market/'manifest.json',r['market_manifest_sha256']),(features/'manifest.json',r['feature_manifest_sha256'])])
    hashes.extend((market/n,h) for n,h in r['market_payloads'].items())
    fm=json.loads((features/'manifest.json').read_text());mm=json.loads((market/'manifest.json').read_text())
    check(fm['market_manifest_sha256']==r['market_manifest_sha256'],'feature_lineage',key)
    check(mm['source_partition_sha256']==r['entry']['sha256'],'market_lineage',key)
    if r['feature_sha256']:
        hashes.append((features/'features.parquet',r['feature_sha256']))
        pf=pq.ParquetFile(features/'features.parquet')
        check(pf.metadata.num_rows==r['rows']==mm['seconds_rows'],'row_count',key)
        names=pf.schema_arrow.names
        check(len([n for n in names if n.endswith('__known')])==182,'feature_schema',key)
    else:check(r['status']=='EXCLUDED' and r['rows']==0 and not (features/'features.parquet').exists(),'empty_exclusion',key)
    metadata.append(dict(key=key,date=r['entry']['selection']['date'],symbol=r['entry']['selection']['symbol'],rows=r['rows'],entries=r['entry_allowed_rows'],eligible=admission[key]['eligible'],reasons=admission[key]['reasons']))
print('Manifest reconciliation complete; hashing all payloads',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    for path,ok in pool.map(lambda x:(str(x[0]),sha(x[0])==x[1]),hashes):check(ok,'sha256',path)
print('All hashes checked; scanning clocks and admission masks',flush=True)
calendar=ObservedCalendar(config['reference']['path'],config['reference']['sha256'])
rng=np.random.default_rng(20261016);selected=set(rng.choice([k for k,r in receipts.items() if r['rows']],12,replace=False))
scanned=0;sample_values=0;qa=[]
for key,r in receipts.items():
    if not r['rows']:continue
    pf=pq.ParquetFile(Path(r['features'])/'features.parquet');prev=None;counts=collections.Counter()
    columns=['t_ns','available_ns','epoch','reference_observed_open','admissible','routing_admissible','entry_allowed','source_day_degraded']
    sel=r['entry']['selection']
    for b in pf.iter_batches(batch_size=65536,columns=columns):
        d=b.to_pandas();t=d.t_ns.to_numpy();opened=~calendar.closed(t)
        check(np.all(np.diff(t)>0) and (prev is None or t[0]>prev),'clock_order',key);prev=t[-1]
        check(np.all(d.available_ns>=d.t_ns),'causal_clock',key)
        check(np.array_equal(d.reference_observed_open,opened),'calendar_mask',key)
        route=(t>=int(sel['selection_available_ns'])) & (not sel['context_only'])
        check(np.array_equal(d.routing_admissible,route),'route_mask',key)
        tm=pd.to_datetime(t,utc=True);ld=tm.tz_convert('Europe/London');ny=tm.tz_convert('America/New_York')
        expected_entry=d.admissible.to_numpy()&route&(ld.hour>=8)&(ny.hour<17)&(ld.dayofweek<5)
        check(np.array_equal(d.entry_allowed,expected_entry),'entry_hours',key)
        check(not np.any(d.admissible.to_numpy() & (~opened|((d.available_ns-d.t_ns)>10**9).to_numpy())),'invalid_admission',key)
        if sel['source_condition']!='available':check(not d.admissible.any(),'degraded_admitted',key)
        for c in ['admissible','entry_allowed','reference_observed_open']:counts[c]+=int(d[c].sum())
        scanned+=len(d)
    check(counts['admissible']==r['admitted_rows'] and counts['entry_allowed']==r['entry_allowed_rows'] and counts['reference_observed_open']==r['reference_open_rows'],'mask_counts',key)
    if key in selected:
        batch=pf.read_row_group(int(rng.integers(pf.num_row_groups))).to_pandas()
        for n in [n[:-7] for n in batch if n.endswith('__known')]:
            v=batch[n].to_numpy();mask=batch[n+'__known'].to_numpy()
            check(np.isfinite(v).all() and not np.any(v[~mask]!=0),'sample_feature_values',key+':'+n)
            check(not np.any(mask & ~batch.admissible.to_numpy()),'sample_known_admission',key+':'+n)
        sample_values+=len(batch)
        qa.append(dict(key=key,**verify_ohlc(r['market'],int(rng.integers(1000000)))))
    if scanned//1000000 != (scanned-r['rows'])//1000000:print('Scanned rows',scanned,flush=True)
minutes=calendar.minutes;tm=pd.to_datetime(np.asarray(minutes)*60,unit='s',utc=True)
ld=tm.tz_convert('Europe/London');ny=tm.tz_convert('America/New_York')
trade=(ld.hour>=8)&(ny.hour<17)&(ld.dayofweek<5)
expected_daily=collections.Counter(str(d) for d in tm[trade].date)
actual=collections.Counter();eligible=collections.Counter()
for x in metadata:
    actual[x['date']]+=x['entries']
    if x['eligible']:eligible[x['date']]+=x['entries']
coverage=[dict(date=d,expected_reference_seconds=n*60,entry_rows=actual[d],eligible_entry_rows=eligible[d],coverage=actual[d]/(n*60)) for d,n in sorted(expected_daily.items())]
owner=json.loads((root/'owner.json').read_text());alive=psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['created'])<.1
result=dict(status='PASS_WITH_LIMITATIONS' if not issues else 'FAILED',utc=now(),issues=issues,owner_alive=alive,
    complete_utc=manifest['completed_utc'],members=len(members),captures=len(receipts),feature_captures=sum(bool(r['rows']) for r in receipts.values()),
    empty_captures=[x for x in metadata if not x['rows']],eligible=catalog['eligible_captures'],excluded=catalog['excluded_captures'],
    exclusion_reasons=dict(collections.Counter(reason for x in metadata for reason in x['reasons'])),
    bytes=manifest['bytes'],free_bytes=psutil.disk_usage(str(root)).free,rows=scanned,entry_rows=sum(actual.values()),eligible_entry_rows=sum(eligible.values()),
    hashes_checked=len(hashes),feature_sample_rows=sample_values,ohlc_samples=qa,members_detail=members,
    daily_coverage=coverage,august=[x for x in metadata if x['date'].startswith('2026-08')],
    source_gaps=config['source_gaps'],staging_bins_remaining=len(list((root/'staging').rglob('*.bin'))),
    duration_seconds=time.perf_counter()-start,
    limitations=['Full masks and hashes scanned; feature numeric validation sampled across 12 captures.',
                  'OHLC checks reuse reconstruction-derived quotes; not independent exchange truth.',
                  'Reference is spot XAUUSD and may differ from GC sessions.',
                  'Training admission is per capture and does not certify full-day coverage, labels or economic performance.'])
save_json(out/'audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['daily_coverage','august','members_detail','ohlc_samples']},indent=2))
