"""Saved-data QA and causal London-open contract selection for new sessions."""
import sys, json, time, zipfile, struct
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zstandard
import databento_dbn
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.audit import scene

OUT=ROOT/'reports/may_extension_v1'


def frames(stream, kind, lo, hi, columns, clock='t_ns'):
    parts=[]
    for e in stream['manifest']['captures']:
        if e['input']['end_ns'] < lo or e['input']['start_ns'] >= hi:
            continue
        name=Path(e['input']['name']).stem
        path=(stream['features'] if kind=='features' else stream['market'])/name/(kind+'.parquet')
        parts.append(pq.read_table(path,columns=columns,filters=[(clock,'>=',lo),(clock,'<',hi)]).to_pandas())
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=columns)


def main():
    started=time.perf_counter()
    streams={}
    for symbol,market,features in [('GCM6','market_may_extension_v1','features_may_extension_v1'),('GCQ6','market_may_roll_v1','features_may_roll_v1')]:
        root=ROOT/'data/derived'/market
        streams[symbol]=dict(market=root,features=ROOT/'data/derived'/features,manifest=json.loads((root/'manifest.json').read_text()))
    archive=ROOT/'data/raw/GLBX-20260805-FTTJQEK3T3.zip'
    with zipfile.ZipFile(archive) as z,z.open('glbx-mdp3-20260501-20260531.mbo.dbn.zst') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        h=reader.read(8)
        metadata=databento_dbn.Metadata.decode(h+reader.read(struct.unpack_from('<I',h,4)[0]))
    mapping={k:[{key:str(v) for key,v in interval.items()} for interval in metadata.mappings[k]] for k in streams}
    save_json(OUT/'instrument_mapping.json',mapping)
    verified=0
    for stream in streams.values():
        for e in stream['manifest']['captures']:
            name=Path(e['input']['name']).stem
            for file,expected in e['payloads'].items():
                assert sha(stream['market']/name/file)==expected
                verified+=1
            fm=json.loads((stream['features']/name/'manifest.json').read_text())
            assert fm['features']==182 and sha(stream['features']/name/'features.parquet')==fm['feature_sha256']
            assert fm['market_manifest_sha256']==sha(stream['market']/name/'manifest.json')
            verified+=1
    # Source OHLC audit slice retains original timestamp_ms. It supplies a
    # separate observed-market-presence reference, not an exchange calendar.
    spot=pd.read_parquet(ROOT/'data/derived/telegram_labels_v1/xauusd_audit_slice.parquet',columns=['timestamp_ms'])
    spot_minutes=set((spot.timestamp_ms.to_numpy()//60000).tolist())
    rng=np.random.default_rng(20261015)
    sessions=[];scenes=[]
    visual=OUT/'audit';visual.mkdir(exist_ok=True)
    for date in [f'2026-05-{d:02d}' for d in [18,19,20,21,22,25,26,27,28,29]]:
        midnight=pd.Timestamp(date,tz='UTC').value
        london=pd.Timestamp(date+' 08:00',tz='Europe/London').value
        ny=pd.Timestamp(date+' 17:00',tz='America/New_York').value
        scores={}
        for symbol,stream in streams.items():
            q=frames(stream,'quotes',midnight,london,['available_ns','valid'],clock='available_ns')
            scores[symbol]=int(q.valid.sum()) if len(q) else 0
        # Only already published complete groups BEFORE London enter this rule.
        chosen=max(scores,key=scores.get)
        assert scores[chosen]>0
        stream=streams[chosen]
        bars=frames(stream,'bars',london+1,ny+1,['t_ns','valid','coverage'])
        f=frames(stream,'features',london,ny,['t_ns','available_ns','admissible','fft_mid_change_128s_power__known'])
        expected_minutes=[minute for minute in range(london//(60*10**9),ny//(60*10**9)) if minute in spot_minutes]
        minute_ids=(bars.t_ns.to_numpy()-1)//(60*10**9)
        bm=np.isin(minute_ids,expected_minutes)
        sm=np.isin(f.t_ns.to_numpy()//(60*10**9),expected_minutes)
        expected=len(expected_minutes)
        assert expected>0
        cuts=np.flatnonzero(np.diff(expected_minutes)>1)+1
        permitted=[part for part in np.split(np.asarray(expected_minutes,dtype=np.int64),cuts) if len(part)]
        result=dict(date=date,symbol=chosen,route=stream['manifest']['route'],selection_utc=str(pd.Timestamp(london,tz='UTC')),
            score_window_utc=[str(pd.Timestamp(midnight,tz='UTC')),str(pd.Timestamp(london,tz='UTC'))],
            valid_publications_before_london=scores,expected_minutes_from_spot_presence=expected,
            valid_minutes=int(bars.valid[bm].sum()),valid_minute_fraction=float(bars.valid[bm].sum()/expected),
            nominal_clock_minutes=int((ny-london)//(60*10**9)),valid_minutes_before_spot_presence_mask=int(bars.valid.sum()),
            admitted_seconds=int(f.admissible[sm].sum()),admitted_second_fraction=float(f.admissible[sm].sum()/(expected*60)),
            expected_basis='observed spot M1 availability inside entry hours, not official exchange holiday hours',
            market=str(stream['market']),features=str(stream['features']),
            decision_interval_ns=[str(london),str(ny)],
            observed_entry_intervals_ns=[[str(int(part[0])*60*10**9),str((int(part[-1])+1)*60*10**9)] for part in permitted],
            context_start_ns=str(midnight-2*3600*10**9))
        sessions.append(result)
        good=f[(f.admissible)&sm]
        if len(good):
            origin=int(good.iloc[int(rng.integers(len(good)))].t_ns)
            entry=next(e for e in stream['manifest']['captures'] if e['input']['start_ns']<=origin<=e['input']['end_ns'])
            name=Path(entry['input']['name']).stem
            s=scene(stream['market']/name,stream['features']/name,visual/(date+'_'+chosen+'.png'),origin-450*10**9,origin+450*10**9,date+' '+chosen+' | causal London-open selection')
            scenes.append(s)
    checks=sum(s['independent_ohlc_checks'] for s in scenes)
    integrity_passed=checks>=100 and all(s['max_ohlc_error']<1e-8 for s in scenes)
    daily_passed=all(s['valid_minute_fraction']>=.95 and s['admitted_second_fraction']>=.8 for s in sessions)
    warnings=[dict(date=s['date'],reason='DAILY_COVERAGE_BELOW_GATE',valid_minute_fraction=s['valid_minute_fraction'],admitted_second_fraction=s['admitted_second_fraction']) for s in sessions if s['valid_minute_fraction']<.95 or s['admitted_second_fraction']<.8]
    weeks=[]
    for group in (sessions[:5],sessions[5:]):
        denominator=sum(s['expected_minutes_from_spot_presence'] for s in group)
        weeks.append(dict(start=group[0]['date'],end=group[-1]['date'],valid_minute_fraction=sum(s['valid_minutes'] for s in group)/denominator,admitted_second_fraction=sum(s['admitted_seconds'] for s in group)/(denominator*60)))
    status='FAIL' if not integrity_passed else 'PASS' if daily_passed else 'COMPLETE_WITH_QA_WARNING'
    report=dict(status=status,sessions=sessions,payload_hashes_verified=verified,scenes=scenes,
        integrity_gate_passed=integrity_passed,daily_coverage_gate_passed=daily_passed,day_warnings=warnings,weeks=weeks,
        gate=dict(min_valid_minute_fraction=.95,min_admitted_second_fraction=.8,min_independent_ohlc_checks=100),
        independent_ohlc_checks=checks,elapsed_seconds=time.perf_counter()-started,
        contract_selection='max valid F_LAST publications observed 00:00 UTC to London open; effective only from London open; no PnL/future-day volume used',
        limitation='both contract streams remain separate; no price splice, back adjustment or positions transferred across contracts; exact price path is contract-specific')
    save_json(OUT/'quality.json',report)
    cluster=ROOT/'data/derived/cluster_v3';cluster.mkdir(exist_ok=True)
    save_json(cluster/'manifest.json',dict(schema='FOUR_WEEK_SESSION_CATALOG_V1',status=report['status'],
        existing_weeks=dict(market=str(ROOT/'data/derived/market_v1'),features=str(ROOT/'data/derived/features_v1'),intervals=[['2026-05-04','2026-05-08'],['2026-05-11','2026-05-15']]),
        added_sessions=sessions,selection_rule=report['contract_selection'],quality_manifest=str(OUT/'quality.json'),quality_sha256=sha(OUT/'quality.json'),
        consumer_note='session catalog, not a flat concatenated instrument tape; existing fixed-May prepare_week needs a catalog-aware adapter for training on weeks 3-4; respect admissible/known masks and explicit day warnings',day_warnings=warnings))
    html='<!doctype html><meta charset="utf-8"><title>Two additional weeks</title><style>body{font:15px system-ui;max-width:1450px;margin:25px auto}img{width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>18–29 May: market reconstruction and 182 features</h1><p>UTC. Separate GCM6 / GCQ6 streams. Contract chosen at London open using only prior published activity. No synthetic continuous price splice.</p>'
    html+=pd.DataFrame([{k:s[k] for k in ['date','symbol','expected_minutes_from_spot_presence','valid_minute_fraction','admitted_second_fraction']} for s in sessions]).to_html(index=False)
    html+=f'<p>Status: {status}. Daily coverage gate: {daily_passed}. Warnings: {warnings}. Unsupported seconds remain masked; no interpolation or threshold relaxation.</p>'
    html+='<p>May25 expected minutes use observed spot coverage: the broker has an early gap. This is not a claim of official exchange closing time.</p>'
    for s in scenes:html+=f'<h2>{s["image"]}</h2><img loading="lazy" src="audit/{s["image"]}">'
    (OUT/'index.html').write_text(html,encoding='utf8')
    roots=json.loads((OUT/'experiment_roots.json').read_text())
    measurements=[json.loads((ROOT/'reports'/r['report']/'measurements.json').read_text()) for r in roots]
    finish=max((ROOT/'reports'/r['report']/'measurements.json').stat().st_mtime for r in roots)
    sizes={key:sum(sum(m['sizes_bytes'][key].values()) for m in measurements) for key in ['raw','market','features']}
    result=dict(status=report['status'],added_weeks=[['2026-05-18','2026-05-22'],['2026-05-25','2026-05-29']],
        total_generation_wall_seconds=finish-min(r['process_created_unix'] for r in roots),qa_seconds=report['elapsed_seconds'],
        records=sum(m['records'] for m in measurements),quotes=sum(m['quotes'] for m in measurements),seconds_rows=sum(m['seconds_rows'] for m in measurements),
        disk_bytes=sizes,durable_market_and_features_bytes=sizes['market']+sizes['features'],
        sampled_memory_upper_bound_bytes=sum(m['peak_tree_rss_bytes'] for m in measurements),memory_note='sum of independently sampled tree peaks; upper bound, not simultaneous aggregate peak',
        fixed_route_build_seconds=measurements[0]['reconstruction_wall_seconds'],roll_route_build_seconds=measurements[1]['reconstruction_wall_seconds'],
        feature_worker_seconds=sum(m['feature_worker_seconds'] for m in measurements),
        workers='6 primary book workers + 2 GCQ6 book workers; feature pools2+1 overlapped as captures completed; not all workers busy for whole run',
        source_partition_seconds=[m['partition_seconds'] for m in measurements],
        quality=str(OUT/'quality.json'),cluster=str(cluster/'manifest.json'),day_warnings=warnings,daily_coverage_gate_passed=daily_passed,
        caveat='different dates/contracts and machine load from original benchmark; measured duration is not controlled speedup evidence')
    save_json(OUT/'result.json',result)
    print(json.dumps(result,indent=2))
    print(pd.DataFrame([{k:s[k] for k in ['date','symbol','valid_minute_fraction','admitted_second_fraction']} for s in sessions]).to_string(index=False))
    if not integrity_passed:raise SystemExit('Reconstruction integrity checks failed')


if __name__=='__main__':main()
