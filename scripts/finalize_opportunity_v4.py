"""Verify immutable source/data/run/visual evidence before declaring V4 complete."""
from pathlib import Path
import json
import subprocess

from spectrum.research_v3.learning import sha
from scripts.prepare_opportunity_v3 import identity

ROOT = Path('reports/scientific_pipeline_v4')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    config = read('configs/research/opportunity_pipeline_v4.json')
    suite = read(ROOT/'suite_summary.json')
    if suite['status'] != 'COMPLETE' or len(suite['runs']) != 4:
        raise ValueError('Four completed fits required')
    dataset = Path(suite['dataset_root'])
    manifest = read(dataset/'manifest.json')
    if manifest['status'] != 'COMPLETE' or manifest['identity'] != identity(config):
        raise ValueError('Dataset recipe mismatch')
    checked = {}

    def verify(path, expected):
        key = str(Path(path).resolve())
        if key not in checked:
            checked[key] = sha(path)
        if checked[key] != expected:
            raise ValueError('Changed bound artifact: '+key)

    verify(dataset/'manifest.json', suite['dataset_manifest_sha256'])
    verify('configs/research/opportunity_pipeline_v4.json', suite['config_sha256'])
    previous = Path('reports/scientific_pipeline_v3')
    prior_completion = read(previous/'completion_receipt.json')
    if prior_completion['dataset_manifest_sha256'] != suite['dataset_manifest_sha256']:
        raise ValueError('Inherited causal verification describes another dataset')
    verify(previous/'causal_verification.json', prior_completion['report_files']['causal_verification.json'])
    for name, item in read(previous/'causal_verification.json')['bound_files'].items():
        verify(name, item['sha256'])
    for day in manifest['days']:
        for name, digest in day['files'].items():
            verify(dataset/name, digest)
        for name, digest in day['source_sha256'].items():
            verify(name, digest)
    outputs = 0
    for entry in suite['runs']:
        path = Path(suite['runs_root'])/entry['run_id']
        receipt = read(path/'receipt.json')
        if receipt['status'] != 'COMPLETE' or receipt['dataset_manifest_sha256'] != suite['dataset_manifest_sha256']:
            raise ValueError('Run is not complete and matched')
        for name, digest in receipt['sources'].items():
            verify(name, digest)
        for name, digest in receipt['outputs'].items():
            verify(path/name, digest)
            outputs += 1
    for filename in ('statistical_audit.json', 'economic_audit.json', 'test_receipt.json', 'decision_findings.json'):
        audit = read(ROOT/filename)
        if audit['status'] not in ('PASS', 'COMPLETE'):
            raise ValueError('Audit not complete: '+filename)
        for key in ('source_hashes', 'input_hashes', 'test_sources'):
            for name, digest in audit.get(key, {}).items():
                verify(name, digest)
    visual = read(ROOT/'visual_manifest.json')
    verify('scripts/report_opportunity_v4.py', visual['source_sha256'])
    for name, digest in visual['inputs'].items():
        verify(name, digest)
    for name, digest in visual['figures'].items():
        verify(ROOT/name, digest)
    verify(ROOT/'index.html', visual['index_sha256'])
    sol = read(ROOT/'sol61_visual_audit.json')
    if not sol['status'].startswith('PASS') or sol['model'] != 'gpt-6.1-sol' or sol['actual_pixel_review'] is not True:
        raise ValueError('Actual Sol 6.1 visual review required')
    reviewed = {Path(item['path']).name: item['sha256'] for item in sol['reviewed_images']}
    if reviewed != visual['figures']:
        raise ValueError('Every final image needs exact matching actual review')
    tests = read(ROOT/'test_receipt.json')
    receipt = dict(status='COMPLETE', branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        git_head_before_evidence_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        verifier_sha256=sha(__file__), suite_sha256=sha(ROOT/'suite_summary.json'),
        dataset_manifest_sha256=suite['dataset_manifest_sha256'], dataset_reused_from='scientific_pipeline_v3',
        inherited_causal_verification_sha256=sha(previous/'causal_verification.json'),
        days=len(manifest['days']), roots=manifest['roots'], admitted=manifest['admitted'],
        daily_payloads_verified=sum(len(d['files']) for d in manifest['days']), completed_fits=4,
        run_outputs_verified=outputs, unique_bound_files_verified=len(checked), tests_passed=tests['total_passed'],
        actual_sol_reviewed_images=len(reviewed),
        report_files={p.name: sha(p) for p in ROOT.iterdir() if p.is_file() and p.name != 'completion_receipt.json'},
        interpretation='Controlled follow-up on previously inspected blocks; no untouched holdout or profitability admission claim')
    (ROOT/'completion_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'report_files'}))


if __name__ == '__main__':
    main()
