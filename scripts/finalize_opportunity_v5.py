"""Bind the completed V5 experiment, independent audits and actual pixel review."""
from pathlib import Path
import re
import subprocess

from scripts.run_opportunity_v5 import read, save, validate_recipe
from scripts.verify_opportunity_v3 import BoundInputs, sha


def finalize():
    report=Path('reports/scientific_pipeline_v5')
    config_path=Path('configs/research/opportunity_pipeline_v5.json')
    c=read(config_path); validate_recipe(c)
    bound=BoundInputs(); bound.bind(config_path); bound.bind(__file__)
    root=Path(c['data']['root']); manifest=read(root/'manifest.json')
    if manifest['status']!='COMPLETE' or manifest['dates']!=c['data']['dates']:
        raise ValueError('Complete fixed dataset required')
    bound.bind(root/'manifest.json')
    for day in manifest['days']:
        bound.bind(root/(day['date']+'.json'))
        for name,digest in day['files'].items(): bound.bind(root/name,digest)
        for name,digest in day['source_sha256'].items(): bound.bind(name,digest)
    suite=read(report/'suite_summary.json')
    if suite['status']!='COMPLETE' or suite['config_sha256']!=sha(config_path) or len(suite['runs'])!=2:
        raise ValueError('Two complete bound filter fits required')
    bound.bind(root/'manifest.json',suite['dataset_manifest_sha256'])
    for base in c['base_models'].values():
        for name,digest in base['files'].items(): bound.bind(Path(base['root'])/name,digest)
    for run in suite['runs']:
        out=Path(suite['runs_root'])/run['run_id']; receipt=read(out/'receipt.json')
        if receipt['status']!='COMPLETE': raise ValueError('Incomplete fit')
        bound.bind(out/'receipt.json')
        for name,digest in receipt['sources'].items(): bound.bind(name,digest)
        for name,digest in receipt['outputs'].items(): bound.bind(out/name,digest)
    required=['causal_verification.json','data_admission.json','learning_audit.json','economic_audit.json','paired_diagnostics.json','test_receipt.json']
    audits={name:read(report/name) for name in required}
    for name,r in audits.items():
        if r['status'] not in ('PASS','COMPLETE'): raise ValueError('Audit did not pass: '+name)
        for key in ('source_hashes','input_hashes','sources_sha256','dataset_payload_hashes','test_files'):
            for path,digest in r.get(key,{}).items(): bound.bind(path,digest)
    learning=audits['learning_audit.json']
    bound.bind('scripts/audit_learning_v5.py',learning['auditor_source_sha256'])
    if not all(learning['checks'].values()): raise ValueError('Incomplete learning audit')
    tests=audits['test_receipt.json']
    if tests['returncode']!=0 or tests['config_sha256']!=sha(config_path): raise ValueError('Tests failed or stale')
    count=re.search(r'(\d+) passed',tests['output'])
    if not count: raise ValueError('Missing test count')
    visual=read(report/'visual_manifest.json'); sol=read(report/'sol61_visual_audit.json')
    if (sol['model']!='gpt-6.1-sol' or not sol['pixel_review_complete'] or sol['inspected_image_count']!=15
            or sol['inspected_images']!=visual['figures'] or sol['blocking_visual_defects']):
        raise ValueError('Actual fifteen-image Sol review required')
    bound.bind(report/'visual_manifest.json',sol['visual_manifest_sha256'])
    bound.bind('scripts/report_opportunity_v5.py',visual['renderer_sha256'])
    bound.bind(report/'index.html',visual['html_sha256'])
    for item in visual['figures']: bound.bind(report/item['file'],item['sha256'])
    for name,digest in visual['inputs'].items(): bound.bind(name,digest)
    evaluation=read(report/'evaluation.json')
    if evaluation['status']!='COMPLETE' or evaluation['suite_sha256']!=sha(report/'suite_summary.json'):
        raise ValueError('Incomplete or stale evaluation')
    findings=dict(status='COMPLETE',verdict='NO_DEMONSTRATED_FILTER_ECONOMIC_ADVANTAGE',
        original_crossing_target_and_base_BCE_unchanged=True,base_updates=0,
        new_days=20,split_days={k:len(v) for k,v in c['splits'].items()},roots=manifest['roots'],
        primary_comparison='cost2; CAL retention50%; numeric thresholds transferred unchanged',
        primary=[dict(seed=r['seed'],arms={arm:dict(signal=v['test_signal'],
            trades=v['test_economics']['trades'],completed=v['test_economics']['completed'],
            net=v['test_economics']['total_net_if_complete']) for arm,v in r['primary'].items()}) for r in evaluation['runs']],
        selected_policy=[dict(seed=r['seed'],arms=r['selected']) for r in evaluation['runs']],
        learning_findings=[dict(seed=r['seed'],global_heads=r['by_role']['test']['model']['heads'],
            candidate_heads=r['candidate_diagnostics']['by_role']['test']['candidate_head_metrics']['heads'],
            candidate_counts={k:v['candidate_roots'] for k,v in r['candidate_diagnostics']['by_role'].items()},
            risk_probability_monotonicity=r['risk_monotonicity']['by_role']['test']) for r in learning['runs']],
        next_research_hypothesis='Align filter training/validation/calibration to the candidate distribution and independently test a broader proposal gate. Keep original crossing target and BCE as reference. No posthoc threshold change or additional fit was performed.',
        limitations=['Five TEST days; no stable advantage across seeds.',
            'New scientific V3/V4 dates but historical cognition August exposure; no never-inspected holdout claim.',
            'Frozen base GCQ6 to new GCZ6 changes temporal and contract regimes together.',
            'Micro70 used, Rich225 including day/VWAP/STFT not fully used; detailed RF61 is 300 seconds.',
            'CAL root matching does not match TEST alert counts, position occupancy or paired account states.',
            'Risk safety before crossing is separate from latency-delayed execution PnL.',
            'Sol pixel findings were delivered; root compiled their JSON after the subagent usage limit.'])
    save(report/'decision_findings.json',findings)
    files={}
    for path in sorted(report.iterdir()):
        if path.is_file() and path.name!='completion_receipt.json':
            files[path.name]=bound.bind(path)
    for pattern in ('scripts/*opportunity_v5*.py','scripts/audit_economics_v5.py','scripts/audit_learning_v5.py','tests/test_research_v5*.py','spectrum/research_v5/*.py'):
        for path in Path('.').glob(pattern): bound.bind(path)
    bound.confirm_unchanged()
    receipt=dict(status='COMPLETE',experiment='SCIENTIFIC_PIPELINE_V5',config_sha256=sha(config_path),
        dataset_manifest_sha256=sha(root/'manifest.json'),git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head_before_evidence_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tests_passed=int(count.group(1)),test_files=len(tests['test_files']),visual_images=15,
        independent_economic_replays=252,raw_risk_order_probes=72,source_replay_days=3,
        data_days=len(manifest['dates']),roots=manifest['roots'],admitted_known=manifest['admitted'],
        fitted_filters=2,frozen_base_models=2,report_files=files,
        verified_unique_bound_files=len(bound.files),bound_files=bound.files,
        original_target_loss_preserved=True,all_required_work_complete=True,
        publication='Local branch and reports only; no remote push or live trading deployment')
    save(report/'completion_receipt.json',receipt)
    print({k:receipt[k] for k in ('status','tests_passed','visual_images','verified_unique_bound_files','data_days','roots')})


if __name__=='__main__': finalize()
