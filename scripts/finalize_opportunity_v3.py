"""Verify final immutable evidence and publish a v3 completion receipt."""
from pathlib import Path
import json
import subprocess

from spectrum.research_v3.learning import sha
from scripts.prepare_opportunity_v3 import identity

ROOT = Path('reports/scientific_pipeline_v3')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    c = read('configs/research/opportunity_pipeline_v3.json')
    suite = read(ROOT/'suite_summary.json')
    assert suite['status'] == 'COMPLETE' and len(suite['runs']) == 8
    dataset = Path(suite['dataset_root'])
    manifest = read(dataset/'manifest.json')
    assert manifest['status'] == 'COMPLETE' and manifest['identity'] == identity(c)
    assert sha(dataset/'manifest.json') == suite['dataset_manifest_sha256']
    checked = {}

    def verify(path, expected):
        key = str(Path(path).resolve())
        if key not in checked:
            checked[key] = sha(path)
        assert checked[key] == expected, 'Changed bound artifact: '+key

    for day in manifest['days']:
        for name,h in day['files'].items():
            verify(dataset/name,h)
        for name,h in day['source_sha256'].items():
            verify(name,h)
    run_outputs = 0
    for entry in suite['runs']:
        path = Path(suite['runs_root'])/entry['run_id']
        r = read(path/'receipt.json')
        assert r['status'] == 'COMPLETE'
        assert r['dataset_manifest_sha256'] == suite['dataset_manifest_sha256']
        for name,h in r['sources'].items():
            verify(name,h)
        for name,h in r['outputs'].items():
            verify(path/name,h)
            run_outputs += 1
    for filename in ('statistical_audit.json','causal_verification.json','weight_semantics.json',
                     'paired_figure_receipt.json','weight_figure_receipt.json','test_receipt.json','decision_findings.json'):
        assert read(ROOT/filename)['status'] in ('PASS','COMPLETE')
    visual = read(ROOT/'visual_manifest.json')
    verify('scripts/report_opportunity_v3.py',visual['source_sha256'])
    for filename in ('statistical_audit.json','paired_figure_receipt.json'):
        for name,h in read(ROOT/filename)['source_hashes'].items():
            verify(name,h)
    for filename,key in (('causal_verification.json','bound_files'),('weight_semantics.json','verified_files')):
        for name,h in read(ROOT/filename)[key].items():
            verify(name,h['sha256'])
    weight_figure = read(ROOT/'weight_figure_receipt.json')
    for name in ('source_script','input_report','png'):
        verify(weight_figure[name],weight_figure[name+'_sha256'])
    for name,h in visual['inputs'].items():
        verify(name,h)
    for name,h in visual['figures'].items():
        verify(ROOT/name,h)
    verify(ROOT/'index.html',visual['index_sha256'])
    sol = read(ROOT/'sol61_visual_audit.json')
    assert sol['status'].startswith('PASS') and sol['model'] == 'gpt-6.1-sol'
    assert sol['actual_pixel_review'] is True
    reviewed = {Path(v['path']).name:v['sha256'] for v in sol['reviewed_images']}
    assert reviewed == visual['figures'], 'Every final image must have a matching actual Sol review'
    tests = read(ROOT/'test_receipt.json')
    for name,h in tests['test_sources'].items():
        verify(name,h)
    report_files = {p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.name != 'completion_receipt.json'}
    result = dict(status='COMPLETE', branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head_before_evidence_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        verifier_sha256=sha(__file__), suite_sha256=sha(ROOT/'suite_summary.json'),
        dataset_manifest_sha256=suite['dataset_manifest_sha256'], days=len(manifest['days']),
        roots=manifest['roots'], admitted=manifest['admitted'],
        daily_payloads_verified=sum(len(d['files']) for d in manifest['days']),
        completed_fits=len(suite['runs']), run_outputs_verified=run_outputs,
        unique_bound_files_verified=len(checked), tests_passed=tests['total_passed'],
        actual_sol_reviewed_images=len(reviewed), report_files=report_files,
        interpretation='Completed controlled research comparison; does not assert profitability, live readiness or two-day detailed context')
    (ROOT/'completion_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'report_files'}))


if __name__ == '__main__':
    main()
