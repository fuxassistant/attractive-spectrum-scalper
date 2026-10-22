"""Bind the completed V2 experiment to frozen code and local saved artifacts.

This verifies recorded tests and execution evidence; it does not rerun training
or turn technical consistency into economic admission.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import pyarrow.parquet as pq

from scripts.flow_context_coverage import save, sha

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/flow_context_v2'
DATA = ROOT / 'data/derived/flow_context_v2'
FROZEN_COMMIT = '55f8b0e'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def main():
    done = read(REPORT / 'completion_receipt.json')
    manifest = read(DATA / 'manifest.json')
    data = read(REPORT / 'data_receipt.json')
    summary = read(REPORT / 'summary.json')
    coverage = read(REPORT / 'coverage_audit.json')
    visual = read(REPORT / 'visual_receipt.json')
    execution = read(REPORT / 'execution_audit.json')
    if any(x['status'] != 'COMPLETE' for x in (done, manifest, data)):
        raise ValueError('Incomplete experiment')
    identities = done['identity']
    if any(x['identity'] != identities for x in (manifest, data, summary)):
        raise ValueError('Identity disagreement')
    frozen = subprocess.check_output(['git', 'rev-parse', FROZEN_COMMIT], cwd=ROOT, text=True).strip()
    if summary['git_head_at_run'] != frozen:
        raise ValueError('Run was not recorded at the preregistered commit')
    for name, expected in identities.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        verify(path, expected)
        relative = path.relative_to(ROOT).as_posix()
        blob = subprocess.check_output(['git', 'show', f'{frozen}:{relative}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f'Git bytes do not match pre-fit identity: {relative}')
    verify(DATA / 'manifest.json', done['manifest_sha256'])
    if data['manifest_sha256'] != done['manifest_sha256']:
        raise ValueError('Data receipt manifest changed')
    verify(REPORT / 'coverage_audit.json', manifest['coverage_sha256'])
    verify(ROOT / 'scripts/flow_context_coverage.py', coverage['script_sha256'])
    verify(ROOT / 'reports/opportunity_episodes_v1/source_code_compatibility.json', manifest['source_execution_proof_sha256'])
    for date, expected in coverage['source_sha256'].items():
        verify(ROOT / 'data/derived/scalper_v1' / f'{date}.parquet', expected)
    rows = 0
    for day in manifest['days']:
        path = DATA / (day['date'] + '.parquet')
        verify(path, day['sha256'])
        n = pq.ParquetFile(path).metadata.num_rows
        if n != day['rows'] or day['max_context_lookahead_ns'] != 0 or day['spread_bbo_max_error'] != 0:
            raise ValueError('Day row/clock/spread receipt failed: ' + day['date'])
        if day['identity'] != identities or day['source_sha256'] != coverage['source_sha256'][day['date']]:
            raise ValueError('Day lineage mismatch')
        rows += n
        for date, expected in ((day['date'], day['current_manifest_sha256']),
                               (day['previous_date'], day['previous_manifest_sha256'])):
            if expected is not None:
                verify(ROOT / 'data/derived/basket_multimonth_v1' / date[:7] / date / 'manifest.json', expected)
    if rows != data['rows'] or len(manifest['days']) != data['days']:
        raise ValueError('Data counts disagree')
    for name, expected in done['prediction_sha256'].items():
        verify(DATA / name, expected)
        pq.ParquetFile(DATA / name).metadata
    if len(done['prediction_sha256']) != 28:
        raise ValueError('Expected seven families, two modes and two stages')
    model_hashes = {p.name: sha(p) for p in sorted(DATA.glob('model_*.pkl'))}
    if len(model_hashes) != 28:
        raise ValueError('Expected 28 saved pairs of heads')
    for expected in (done['summary_sha256'], visual['summary_sha256']):
        verify(REPORT / 'summary.json', expected)
    verify(ROOT / 'scripts/flow_context_report.py', visual['script_sha256'])
    verify(REPORT / 'scene_selection.json', visual['scene_selection_sha256'])
    for name, expected in visual['images'].items():
        verify(REPORT / name, expected)
    if len(visual['images']) != 14:
        raise ValueError('Expected 14 audited images')
    if execution['status'] != 'PASS' or execution['compared'] != 824 or any(d['mismatches'] for d in execution['checks']):
        raise ValueError('Execution audit is not a matching pass')
    verify(ROOT / 'scripts/opportunity_episodes_audit.py', execution['reused_audit_code_sha256'])
    verify(ROOT / 'scripts/flow_context_execution_audit.py', execution['v2_driver_sha256'])
    verify(DATA / 'audit_execution_facts.json', execution['facts_sha256'])
    for name, expected in execution['quote_hashes'].items():
        verify(Path(name), expected)
    log = ROOT / '.tmp/flow_context_v2_full_tests.log'
    match = re.search(r'(\d+) passed in ([0-9.]+)s', log.read_text(encoding='utf-8'))
    if not match or int(match[1]) != 315:
        raise ValueError('Expected the completed 315-test run log')
    docs = ['RESULTS_RU.md', 'REPRODUCE_RU.md', 'FEATURE_CONTEXT_AUDIT_RU.md', 'SOL61_VISUAL_AUDIT_RU.md', 'index.html']
    artifact_hashes = {name: sha(REPORT / name) for name in docs}
    package_names = ['numpy', 'pandas', 'pyarrow', 'numba', 'scikit-learn', 'scipy', 'matplotlib', 'pytest']
    save(REPORT / 'environment.json', dict(python=sys.version, platform=platform.platform(),
         libraries={name: importlib.metadata.version(name) for name in package_names}))
    result = dict(status='PASS', recorded_utc=datetime.now(timezone.utc).isoformat(), frozen_commit=frozen,
        scientific_code_bytes_match_prefit_git=True,
        full_suite=dict(command='.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest_flow_context_v2',
                        passed=int(match[1]), seconds=float(match[2]), log_sha256=sha(log),
                        tests_sha256=sha(ROOT / 'tests/test_flow_context.py'),
                        scope='Recorded completed full-suite run; validator does not rerun pytest'),
        verified_source_label_parquets=len(coverage['source_sha256']),
        verified_daily_parquets=len(manifest['days']), rows=rows,
        verified_prediction_parquets=len(done['prediction_sha256']), saved_model_sha256=model_hashes,
        verified_visuals=len(visual['images']), summary_sha256=done['summary_sha256'],
        manifest_sha256=done['manifest_sha256'], execution_audit_sha256=sha(REPORT / 'execution_audit.json'),
        visual_receipt_sha256=sha(REPORT / 'visual_receipt.json'), artifacts_sha256=artifact_hashes,
        environment_sha256=sha(REPORT / 'environment.json'), validator_sha256=sha(__file__),
        quote_scope=manifest['quote_hash_scope'],
        economic_admission='NONE; all A2 admissions false. Technical consistency only.')
    if any(summary['admissions'].values()):
        raise ValueError('Reported WAIT admission no longer agrees with summary')
    save(REPORT / 'validation_receipt.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'verified_source_label_parquets', 'verified_daily_parquets',
                                           'rows', 'verified_prediction_parquets', 'verified_visuals')}, indent=2))


if __name__ == '__main__':
    main()
