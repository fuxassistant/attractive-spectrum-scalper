"""Read-only bounded audit of local feature/representation evidence.

Requires private local datasets. Rechecks payload identity and optionally raw
formula samples; never promotes descriptive evidence to model/policy admission.
Run from the checkout with Python 3.12 and an unused --out directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {
    'adapter': 'research_results/scalper_adapter_v1/verified_v5',
    'similarity': 'research_results/story_similarity_v2/verified_v3',
    'age_aligned_similarity': 'research_results/story_similarity_v2/age_aligned_v1',
    'regime_description': 'research_results/regime_language_v1/verified_v1',
    'continuation_phrases': 'research_results/continuation_lab_v1/verified_v1',
    'paired_targets_not_X': 'research_results/paired_action_target_v1/verified_v1',
    'zone_episodes': 'research_results/zone_episode_lab/verified_v3',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--refresh-formulas', action='store_true')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    cache, comparisons = {}, []

    def digest(path):
        path = path.resolve()
        if path not in cache:
            with path.open('rb') as stream:
                cache[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return cache[path]

    def compare(path, expected, scope):
        path = path.resolve()
        exists = path.is_file()
        actual = digest(path) if exists else None
        try:
            label = path.relative_to(ROOT).as_posix()
        except ValueError:
            label = str(path)
        comparisons.append(dict(path=label, scope=scope, expected=expected,
                                actual=actual, matches=actual == expected))

    from spectrum.features.catalog import families
    catalog = []
    for family in families():
        family.resolve()
        catalog.append(family.describe())

    runs = {}
    for name, relative in RUNS.items():
        root = ROOT / relative
        manifest = read(root / 'manifest.json')
        start = len(comparisons)
        for key in ('source_sha256', 'implementation_sha256', 'input_sha256', 'inputs'):
            for path, expected in manifest.get(key, {}).items():
                compare(ROOT / path, expected, name + ':' + key)
        for path, expected in manifest.get('artifacts', {}).items():
            compare(root / path, expected, name + ':artifact')
        evidence = {}
        for file in ('validation.json', 'summary.json', 'independent_review.json', 'execution_review.json'):
            if (root / file).is_file():
                item = read(root / file)
                evidence[file] = dict(sha256=digest(root / file), recorded=item)
                if isinstance(item, dict) and item.get('manifest_sha256'):
                    compare(root / 'manifest.json', item['manifest_sha256'], name + ':receipt_binding')
        checks = comparisons[start:]
        runs[name] = dict(path=relative, manifest_sha256=digest(root / 'manifest.json'),
                          recorded_status=manifest.get('status'),
                          checked=len(checks), mismatches=sum(not c['matches'] for c in checks),
                          evidence=evidence)
        print(name, len(checks), 'checks', runs[name]['mismatches'], 'mismatches', flush=True)

    feature_root = ROOT / 'data/derived/feature_week_v2'
    admission = read(feature_root / 'audit/admission.json')
    for key, file in [('numerical_audit_sha256', 'audit.json'), ('parallel_parity_sha256', 'parallel_parity.json'),
                      ('installed_package_sha256', 'installed_package.json'), ('visual_review_sha256', 'visual_review.json'),
                      ('validation_sha256', 'validation.json')]:
        compare(feature_root / 'audit' / file, admission[key], 'feature_admission_binding')
    for path, expected in admission['experiment_code_sha256'].items():
        compare(ROOT / 'experiments/feature_week_v2' / path, expected, 'feature_audit_source')
    evolution = read(ROOT / 'docs/EVOLUTION_0_4_1.json')
    for path, expected in evolution['current_runtime_sha256'].items():
        compare(ROOT / path, expected, 'runtime_0.4.1')
    compare(ROOT / 'pyproject.toml', evolution['current_project_sha256'], 'runtime_0.4.1')

    # This existing independent auditor reads source/sidecar/payloads and does
    # not overwrite the historical admission or claim a fresh generation.
    from experiments.feature_week_v2 import audit as week
    data_root = feature_root / 'final_parallel2'
    lineage = week.verify_lineage(data_root)
    coverage = week.coverage(data_root)
    week.compare_parallel(data_root, feature_root / 'benchmark_serial', out)
    parallel = read(out / 'parallel_parity.json')
    refreshed = {}
    if args.refresh_formulas:
        for date in ('2026-08-24', '2026-08-28'):
            result, cadence = week.day_audit(data_root, date)
            refreshed[date] = result
            print('raw formula refresh', date, 'PASS', flush=True)
        refreshed['minutes'] = week.minutes_audit(data_root)
        refreshed['spectral_controls'] = week.spectral_controls()

    review = read(feature_root / 'audit/feature_review.json')
    result = dict(schema='REPRESENTATION_READINESS_AUDIT_V1', audit_date='2026-11-01',
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  generator_sha256=digest(Path(__file__)), python=sys.version,
                  packages={name: importlib.metadata.version(name) for name in
                            ('numpy', 'pandas', 'pyarrow', 'numba', 'tzdata')},
                  status='INTEGRITY_PASS_WITH_SCOPE_LIMITATIONS' if all(c['matches'] for c in comparisons)
                         else 'EVIDENCE_DRIFT_REQUIRES_REVIEW',
                  generation_rerun=False, raw_formula_refresh=bool(args.refresh_formulas),
                  comparisons=comparisons, distinct_hashed_files=len(cache), runs=runs,
                  feature_week=dict(admission=admission, lineage=lineage, coverage=coverage, parallel=parallel,
                                    formula_refresh=refreshed, channel_review=review),
                  catalog=catalog, full_history_training_admission=False,
                  predictive_information_proven=False, economic_admission=False, live_admission=False)
    (out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], comparisons=len(comparisons),
                          mismatches=[c for c in comparisons if not c['matches']],
                          catalog_entries=len(catalog), source_payloads=lineage['unique_payloads']), ensure_ascii=False))
    return 0 if all(c['matches'] for c in comparisons) else 1


if __name__ == '__main__':
    raise SystemExit(main())
