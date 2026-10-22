"""Verify artifact bindings for the completed descriptive crossing-map study."""
import json
import re
import subprocess
from pathlib import Path
from scripts.crossing_map_build import OUT, REPORT
from scripts.flow_context_coverage import save, sha

BASE = 'cb32abf'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    manifest = read(OUT / 'manifest.json')
    build = read(REPORT / 'build_receipt.json')
    summary = read(REPORT / 'summary.json')
    validation = read(REPORT / 'validation.json')
    identity = read(REPORT / 'identity_migration_receipt.json')
    visual = read(REPORT / 'visual_receipt.json')
    audit = read(REPORT / 'sol61_visual_audit_receipt.json')
    assert manifest['status'] == summary['status'] == 'COMPLETE'
    assert build['status'] == validation['status'] == identity['status'] == 'PASS'
    for path, expected in manifest['identity'].items():
        assert sha(path) == expected, path
    for day in manifest['days']:
        for name, expected in day['files'].items():
            assert sha(OUT / name) == expected, name
    assert sha('reports/vwap_bands_week_v1/source_receipt.json') == manifest['source_receipt_sha256']
    assert build['manifest_sha256'] == summary['manifest_sha256'] == identity['current_manifest_sha256'] == sha(OUT / 'manifest.json')
    assert summary['targets_sha256'] == sha(OUT / 'targets.parquet')
    assert summary['code_sha256'] == sha('scripts/crossing_map_analyze.py')
    assert validation['script_sha256'] == sha('scripts/crossing_map_verify.py')
    assert identity['script_sha256'] == sha('scripts/crossing_map_identity_check.py')
    assert validation['raw_first_passage_checks'] == 600
    assert '17 passed' in validation['tests'] and identity['compared_tables'] == 20
    assert visual['page_sha256'] == sha(REPORT / 'index.html')
    assert visual['summary_sha256'] == sha(REPORT / 'summary.json')
    assert visual['script_sha256'] == sha('scripts/crossing_map_report.py')
    assert len(visual['images']) == 7
    for name, expected in visual['images'].items():
        assert sha(REPORT / name) == expected, name
    for key, name in [('audit_sha256', 'SOL61_VISUAL_AUDIT_RU.md'),
                      ('visual_receipt_sha256', 'visual_receipt.json'),
                      ('identity_migration_receipt_sha256', 'identity_migration_receipt.json'),
                      ('summary_sha256', 'summary.json')]:
        assert audit[key] == sha(REPORT / name), key
    # Resolve local links from the saved report without a browser or network.
    page = (REPORT / 'index.html').read_text(encoding='utf-8')
    local_links = sorted(set(re.findall(r'(?:href|src)="([^"#]+)"', page)))
    for link in local_links:
        if '://' not in link:
            assert (REPORT / link).is_file(), link
    baseline = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], text=True).splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE], text=True).splitlines())
    assert not ((changed & baseline) - {'.gitignore'}), 'Pre-existing tracked artifacts changed'
    sources = [Path('spectrum/cognition/crossing_map.py'), Path('tests/test_crossing_map.py')]
    sources += sorted(Path('scripts').glob('crossing_map*.py'))
    sources += sorted(Path('contracts').glob('crossing_map*.json'))
    reports = sorted(p for p in REPORT.iterdir() if p.is_file() and p.name != 'completion_receipt.json')
    checks = build['checks']
    result = dict(status='COMPLETE_RESEARCH_NOT_MODEL_ADMISSION', base_commit=BASE,
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                  origins=summary['origins'], cells=summary['cells'], physical_events=summary['bank_events'],
                  verification=dict(tests=validation['tests'], raw_first_passage_checks=600,
                                    direct_vwap_checks=sum(c['type'] == 'vwap_direct' for c in checks),
                                    direct_path_checks=sum(c['type'] == 'profile_direct' for c in checks),
                                    unchanged_bb_bank_rows=build['bb_parity_rows'],
                                    identity_only_exact_table_parity=20,
                                    resolved_id_links=sum(x['resolved_cells'] for x in identity['bank_map_links']),
                                    gradient_max_abs_error=summary['factorized_gradient_max_error'],
                                    existing_baseline_artifacts_unchanged=True,
                                    local_report_links_checked=len(local_links)),
                  data_manifest_sha256=sha(OUT / 'manifest.json'),
                  source_sha256={str(p).replace('\\', '/'): sha(p) for p in sources},
                  report_sha256={p.name: sha(p) for p in reports},
                  limitations=['One exposed development week, highly dependent origins/cells',
                               'Checkpoint marks are alternative liquidation states, not policy equity',
                               'New predictive learner, survival and quantile heads not trained',
                               'Target stability and correct loss do not establish prediction or profitability'])
    save(REPORT / 'completion_receipt.json', result)
    print(json.dumps(dict(status=result['status'], verification=result['verification'])), flush=True)


if __name__ == '__main__':
    main()
