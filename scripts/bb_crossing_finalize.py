"""Bind reviewed visuals, saved models, economic replay and test evidence."""
from datetime import datetime,timezone
import json
import pickle
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scripts.bb_crossing_build import DATA,REPORT,SOURCE,identity
from scripts.bb_crossing_learn import account
from scripts.flow_context_coverage import save,sha


def read(path):return json.loads(Path(path).read_text())
def check(path,expected):
    if sha(path)!=expected:raise ValueError('Changed artifact: '+str(path))


def main():
    man=read(DATA/'manifest.json');summary=read(REPORT/'summary.json');learn=read(REPORT/'learning_receipt.json');visual=read(REPORT/'visual_receipt.json')
    if man['identity']!=identity():raise ValueError('Changed scientific build')
    check(SOURCE/'manifest.json',man['source_manifest_sha256'])
    check(REPORT/'summary.json',learn['summary_sha256']);check(REPORT/'summary.json',visual['summary_sha256'])
    check('scripts/bb_crossing_learn.py',learn['script_sha256']);check('scripts/bb_crossing_report.py',visual['script_sha256'])
    for name,expected in visual['images'].items():check(REPORT/name,expected)
    if len(visual['images'])!=17:raise ValueError('Missing scientific visual')
    check(REPORT/'index.html',visual['page_sha256'])
    text=(REPORT/'index.html').read_text(encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"#]+)"',text):
        if not target.startswith(('http:','https:')) and not (REPORT/target).is_file():raise ValueError('Broken report link '+target)
    audit=REPORT/'SOL61_VISUAL_AUDIT_RU.md'
    if not audit.exists() or sha(REPORT/'visual_receipt.json') not in audit.read_text(encoding='utf-8'):raise ValueError('Visual review not bound to final receipt')
    model_checks=[];files={}
    for stage in ['A2','B']:
        for s in [1,3,6,9]:
            modelpath=DATA/f'economic_model_{stage}_m{s}.pkl';check(modelpath,learn['models'][modelpath.name])
            with modelpath.open('rb') as src:models=pickle.load(src)
            path=DATA/f'economic_{stage}_m{s}.parquet';f=pd.read_parquet(path);files[path.name]=sha(path)
            error=0.
            for side in ['L','S']:
                p=models['models'][side].predict(f[models['columns']].to_numpy(np.float32))
                error=max(error,float(np.max(np.abs(p-f['pred_'+side].to_numpy()))))
            if error>1e-12:raise ValueError('Economic prediction parity')
            stats,trades=account(f,s,sorted(f.date.unique()))
            if stats!=summary['stages'][stage]['economics'][str(s)]['candidate_account']:raise ValueError('Account decision replay differs')
            tradepath=DATA/f'trades_{stage}_m{s}.parquet';pd.testing.assert_frame_equal(trades,pd.read_parquet(tradepath));files[tradepath.name]=sha(tradepath)
            eq=DATA/f'equity_{stage}_m{s}.parquet';files[eq.name]=sha(eq)
            model_checks.append(dict(stage=stage,scale=s,heads=2,rows=len(f),max_prediction_error=error,selected_trades=len(trades)))
    tests={}
    for name,expected in [('bb_crossing_v1_tests.log',327),('bb_crossing_v1_focused_tests.log',12)]:
        path=Path('.tmp')/name;match=re.search(r'(\d+) passed in ([\d.]+)s',path.read_text(encoding='utf-8'))
        if not match or int(match[1])!=expected:raise ValueError('Missing test evidence')
        tests[name]=dict(passed=int(match[1]),seconds=float(match[2]),sha256=sha(path))
    unchanged=subprocess.check_output(['git','diff','925bb44','--','contracts/flow_context_v2.json','spectrum/cognition/flow_context.py',
        'spectrum/cognition/flow_context_lab.py','reports/flow_context_v2','reports/target_loss_audit_v1'],text=True)
    if unchanged:raise ValueError('Frozen preceding experiments changed')
    for name in ['numerical_validation.json','raw_quote_checks.json','account_replay.json','full_session_validation.json']:
        if read(REPORT/name)['status']!='PASS':raise ValueError('Required check failed')
    artifact_names=[p.name for p in REPORT.iterdir() if p.suffix in ['.json','.md','.html','.png'] and p.name not in ['completion_receipt.json','summary.running.json']]
    code=[str(p) for p in Path('scripts').glob('bb_crossing*.py')]+['spectrum/cognition/bb_crossing.py']
    result=dict(status='COMPLETE_VERIFIED_RESEARCH',completed_utc=datetime.now(timezone.utc).isoformat(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        build_commit='88afcfe',tests=tests,classification_prediction_pairs=30,economic_prediction_checks=model_checks,
        raw_first_cross_checks=240,raw_execution_checks=640,late_session_checks=120,short_full_session_profile_parity=582805,
        visual_images=17,visual_audit_sha256=sha(audit),preceding_experiments_unchanged=True,
        code_sha256={p:sha(p) for p in code},artifact_sha256={name:sha(REPORT/name) for name in artifact_names},
        economic_data_sha256=files,scope='Technical and visual comparison complete. Every A2 admission failed; operational WAIT. Full-session factual projection is separate from the 30m learned/strategy target.',
        superseded_partial_builds=[dict(path='data/derived/bb_crossing_bank_v1_superseded_id',reason='IDs initially depended on query build; fixed before learning; original partial preserved'),
            dict(path='data/derived/bb_crossing_bank_v1/first_session_superseded_slow',reason='Batched mark calculation; 8 old/new tables exactly equal; partial preserved')],
        test_environment_note='Full 327-test run exited successfully; pytest emitted an atexit Temp symlink permission warning. The later 12-test run used workspace basetemp and completed cleanly.')
    save(REPORT/'completion_receipt.json',result)
    print(json.dumps(dict(status=result['status'],classifiers=30,economic_heads=16,visuals=17,source_unchanged=True)),flush=True)


if __name__=='__main__':main()
