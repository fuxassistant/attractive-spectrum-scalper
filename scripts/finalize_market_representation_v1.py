"""Seal the bounded research receipt only after tests, source and visual audits."""
import sys, json, re
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_crossing_path_target_v6 import sha, save
from scripts.build_market_representation_v1 import ROOT, REPORT


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    m = read(ROOT / 'manifest.json'); v = read(REPORT / 'verification.json')
    p = read(REPORT / 'probe_scores.json'); s = read(REPORT / 'scene_selection.json')
    visual = read(REPORT / 'sol61_visual_audit.json')
    assert m['status'] == 'COMPLETE' and v['status'] == 'PASS'
    assert visual['status'] in ('PASS', 'WITH_LIMITATIONS', 'PASS_WITH_LIMITATIONS'), visual['status']
    for group in ('sources', 'outputs', 'implementation'):
        for path, digest in m[group].items(): assert sha(path) == digest
    for path, digest in v['reader_implementation'].items(): assert sha(path) == digest
    suite = ET.parse(REPORT / 'tests.xml').getroot().find('testsuite')
    tests = {k: int(suite.attrib[k]) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert tests['failures'] == tests['errors'] == 0 and tests['tests'] >= 39
    overlay_path = Path('reports/full_mbo_v1/audit_20261016/training_admission_overlay.json')
    overlay = read(overlay_path)
    admission = [r for r in overlay['captures'] if r['date'] in m['dates']]
    assert len(admission) == len(m['dates']) and all(r['audit_eligible'] and r['eligible'] for r in admission)
    registry = dict(version='MARKET_REPRESENTATION_V1_DESCRIPTOR_REGISTRY',
        feature_names=m['feature_names'],
        input=dict(fast='51 values x physical 1s prefix up to 900s; known masks, timestamps, age, scheduled-open mask, raw and normalized values',
            day='36 values x completed M1 from previous London08; no token truncation; scheduled close and unavailable minutes retained',
            zones='8 objects: 4 previous-cycle H/L/VWAP/POC and 4 offset controls; recognized_at, contacts and accumulated price-level quantities',
            events='Last 256 prefix-derived tokens; truncation explicit. Same-second tokens have no proven subsecond order.',
            entry_point='spectrum.market_repr.store.open_observation_store(root, date).observe(utc_nanoseconds)'),
        units=dict(price='USD per troy ounce', tick='.10 USD/oz; midpoint uses half-tick grid', quantity='GC contracts',
            age='seconds; removed_age_qseconds is quantity-weighted age, divide by removed_age_known_qty for observed mean',
            time='int64 UTC ns; physical t_ns differs from publication available_ns',
            normalization='Price displacement from current observed midpoint / last known completed BB1 halfwidth with one-tick floor; quantity signed log1p; original values retained'),
        observed=dict(executions='T printed volume once; F is attribution only; passive/aggressor/other roles remain distinct raw channels',
            withdrawal='Unlinked source decrement proxy, not proof of cancellation intention',
            refill='Addition matched to previously linked fill at same absolute price within 2s, capped by unmatched fill budget; not participant identity',
            unknown='NaN plus masks; empty sparse ledger buckets require a complete-publication watermark',
            continuity='Canonical physical validity/epoch drives sequence breaks; fresh_2s is an extra feature, never a history reset',
            vwap='All exact trade prices and sizes; comparison with midpoint proxy uses identical available price/volume support',
            spectrum='Trailing 128 completed M1 Hann return spectrum, bands 2-8/8-32/32-64m, no interpolation across support breaks; unsigned'),
        tags=dict(status='Descriptive conjunctions, not causal mechanisms or prediction labels',
            REFILL_UNDER_PRESSURE='Small absolute 30s displacement, sufficiently active/signed aggression and refill on at least one side; not guaranteed attacked-side absorption',
            EXECUTED_DEPLETION='Move with enough linked executions relative to aggregate withdrawals; not proven depletion of a particular queue; only two matched observations in this week',
            WITHDRAWAL='Move plus large unlinked removal relative to linked fill, summed across both sides',
            AGGRESSION_DECAY='Small displacement and drop in 30s activity versus preceding 30s'),
        control_limits='Offset controls are neither distance/exposure matched nor independent causal controls. Nearest past-geometry pairs may reuse observations.',
        parameter_scope=dict(frozen_schema=['near 12 ticks', '30s descriptor windows', '2s refill budget', 'M1/M3/M6 SMA20 BB', '128m STFT', '5-tick profile bins', '900s fast', '256 event cap'],
            configurable=['explicit admitted dates/previous-date map', 'chronological split lists', 'probe cadence/horizon/deadband', 'probe HGB hyperparameters'],
            rule='Changing a frozen schema constant requires a new version and rebuilding its affected stage; JSON values are a declared contract, not universal live knobs'),
        missing=['Side-aligned causal absorption/depletion validation', 'Standing visible depth per static zone',
                 'Full subsecond event-order branch', 'Reviewed holiday/roll calendar beyond this week',
                 'Sequence encoder consuming the actual multiscale loader', 'Unexposed confirmation and economic player evaluation'])
    save(REPORT / 'descriptor_registry.json', registry)
    m['readiness'] = 'REPRESENTATION_DESCRIPTIVE_READY_BOUNDED_WEEK'
    m['admission_receipt'] = str(REPORT / 'readiness.json')
    m['neural_training_integration'] = 'NOT_IMPLEMENTED'
    m['predictive_admission'] = m['economic_admission'] = False
    save(ROOT / 'manifest.json', m); save(REPORT / 'source_to_tensor_inventory.json', m)
    ready = dict(status=m['readiness'], branch='codex/market-representation-v1', dates=m['dates'],
        scope='Validated development representation of one normal GC week; not whole-history, live, neural-player or economic admission',
        dataset_manifest_sha256=sha(ROOT / 'manifest.json'), tests=tests, visual_status=visual['status'],
        source_records=sum(read(ROOT / 'raw' / d / 'manifest.json')['source']['records'] for d in m['dates']),
        raw_trade_rows=sum(r['rows'] for r in v['raw_trade_parity']), trade_contracts=sum(r['contracts'] for r in v['raw_trade_parity']),
        stock_checks=sum(r['raw_stats']['stock_checks'] for r in m['raw_audit']),
        source_hashes_checked=v['checked_hashes'], actual_prefix_checks=v['prefix_checks'],
        restored_epoch_tail_contracts=sum(r['recovered_epoch_tail_contracts'] for r in v['sampled_volume_reconciliation']),
        regular_queries=s['regular_queries'], mature_queries=s['mature_queries'], scene_count=len(s['scenes']),
        test_metrics={k: value['test'] for k, value in p['results'].items()},
        predictive_conclusion='NOT_ADMITTED: expanded summaries do not improve proper probability loss over price or constant class prior on these exposed test dates.',
        next_gate='Train a bounded causal sequence encoder through the actual loader on multiple chronological weeks; preserve calibration/side/risk/execution distinctions and use untouched confirmation.',
        overlay=dict(path=str(overlay_path), sha256=sha(overlay_path), eligible_dates=[r['date'] for r in admission],
                     interpretation='Capture-level technical eligibility only; diagnostic observation cohort is not a trade-entry admission cohort.'),
        evidence_hashes={str(REPORT / n): sha(REPORT / n) for n in ('verification.json', 'tests.xml', 'sol61_visual_audit.json', 'probe_scores.json', 'descriptor_registry.json')},
        corrections=['Nanoseconds remain int64; sparse zero buckets need a completion watermark',
            'Missing UTC-boundary minutes retained with masks; intrasecond extrema from canonical all-event M1',
            'Quote freshness is separated from physical continuity',
            'Exact trade tape recovers epoch tails missed by sampled counters',
            'VWAP proxy comparison uses matched volume support',
            'Diagnostic FLAT boundaries operate on exact half ticks',
            'Past image labels do not reveal selected errors; distant VWAP does not flatten local price'],
        limitations=registry['missing'])
    save(REPORT / 'readiness.json', ready)
    contract = read('contracts/market_representation_v1.json')
    contract['status'] = m['readiness']; contract['implementation_receipt'] = str(REPORT / 'readiness.json')
    contract['implementation_scope'] = dict(dates=m['dates'], descriptor_registry=str(REPORT / 'descriptor_registry.json'),
        no_claim=['participant intention', 'full zone depth', 'subsecond token ordering', 'neural use of two days', 'predictive or economic edge'],
        intended_design_beyond_this_slice=registry['missing'])
    Path('contracts/market_representation_v1.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    roadmap = read('contracts/market_representation_roadmap_v1.json')
    roadmap['status'] = 'R0_R3_BOUNDED_SLICE_READY_R5_SCALAR_PROBE_ONLY'
    stages = {'R0':'BOUNDED_WEEK_PASS', 'R1':'OBSERVABLE_FACTS_PASS_WITH_LIMITATIONS', 'R2':'MULTISCALE_LOADER_READY_STATIC_ZONES_ONLY',
              'R3':'CAUSAL_AND_VISUAL_PASS_WITH_LIMITATIONS', 'R4':'DIAGNOSTIC_ENDPOINT_DATASET_ONLY_CROSSING_TARGET_UNCHANGED',
              'R5':'SCALAR_PROBES_COMPLETE_SEQUENCE_ENCODER_NOT_RUN'}
    for stage in roadmap['stages']:
        if stage['id'] in stages: stage['status'] = stages[stage['id']]
    roadmap['implementation_receipt'] = str(REPORT / 'readiness.json')
    roadmap['experiment_discipline']['artifact_boundary'] = 'Code/config/contracts are Git-sized; raw data, datasets and heavy visual reports remain local. Implementation status is bound by readiness.json.'
    Path('contracts/market_representation_roadmap_v1.json').write_text(json.dumps(roadmap, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    page = (REPORT / 'index.html').read_text(encoding='utf-8')
    page = re.sub(r'<article><h3>(scene_\d+)', r'<article id="\1"><h3>\1', page)
    marker = '<h2>Память, VWAP и поддержка спектра</h2>'
    notice = '<p class="notice"><b>Итог проверки:</b> репрезентация пригодна для описательных исследований на этой неделе; прогнозное преимущество не подтверждено. <a href="readiness.json">Границы готовности</a> · <a href="descriptor_registry.json">Смысл каждого слоя и измерения</a>.</p>'
    if 'Границы готовности</a>' not in page: page = page.replace(marker, notice + marker)
    (REPORT / 'index.html').write_text(page, encoding='utf-8', newline='\n')
    receipt = read(REPORT / 'report_manifest.json'); receipt['status'] = m['readiness']; receipt['dataset_manifest_sha256'] = sha(ROOT / 'manifest.json')
    paths = [p for p in REPORT.iterdir() if p.is_file() and p.name != 'report_manifest.json' and p.suffix in ('.html','.json','.xml','.parquet')]
    paths += list((REPORT / 'figures').glob('*.png'))
    receipt['output_hashes'] = {str(p):sha(p) for p in paths}
    receipt['implementation']['scripts/finalize_market_representation_v1.py'] = sha('scripts/finalize_market_representation_v1.py')
    save(REPORT / 'report_manifest.json', receipt)
    print(json.dumps(dict(status=ready['status'], tests=tests, scenes=len(s['scenes']), records=ready['source_records'], visual=visual['status'])))


if __name__ == '__main__': main()
