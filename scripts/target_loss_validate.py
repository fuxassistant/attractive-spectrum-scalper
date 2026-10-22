"""Verify the target/loss audit against the immutable source experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scripts.flow_context_coverage import save, sha

REPORT=Path('reports/target_loss_audit_v1');DATA=Path('data/derived/target_loss_audit_v1')
SOURCE=Path('data/derived/flow_context_v2')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def verify(path,expected):
    if sha(path)!=expected:raise ValueError('Changed artifact: '+str(path))


def main():
    m=read(REPORT/'measurements.json');v=read(REPORT/'visual_receipt.json')
    verify(Path('contracts/target_loss_audit_v1.json'),m['contract_sha256'])
    for path,expected in m['code_sha256'].items():verify(path,expected)
    verify(SOURCE/'manifest.json',m['source_manifest_sha256'])
    verify(Path('reports/flow_context_v2/completion_receipt.json'),m['source_completion_sha256'])
    original=read(Path('reports/flow_context_v2/completion_receipt.json'))
    for path,expected in original['identity'].items():verify(path,expected)
    verify(Path('reports/flow_context_v2/summary.json'),original['summary_sha256'])
    for name,expected in m['model_sha256'].items():verify(SOURCE/name,expected)
    comparisons=[]
    for mode in ('P','T'):
        name=f'frame_{mode}.parquet';verify(DATA/name,m['frame_sha256'][name]);all_rows=pd.read_parquet(DATA/name)
        f=all_rows.loc[all_rows.split=='B'].reset_index(drop=True)
        for family in ['baseline','rich_flow','rich_spectrum','shuffled']:
            name=f'predictions_B_{mode}_{family}.parquet';verify(SOURCE/name,original['prediction_sha256'][name]);p=pd.read_parquet(SOURCE/name)
            pd.testing.assert_frame_equal(f[['date','t_ns','side','category']],p[['date','t_ns','side','category']].reset_index(drop=True),check_dtype=False)
            error=max(float(np.max(np.abs(f['pf_'+family].to_numpy()-p.p_fill.to_numpy()))),
                max(float(np.max(np.abs(f[f'pc{k}_{family}'].to_numpy()-p['c_'+label].to_numpy()))) for k,label in enumerate(['TAKE','STOP','TIME'])))
            if error>1e-12:raise ValueError('Saved prediction parity failed')
            comparisons.append(dict(mode=mode,family=family,rows=len(p),max_probability_error=error))
        cat=all_rows.category.to_numpy();filled=(cat>=0)&(cat<3)
        if not np.array_equal(np.isfinite(all_rows.g_path),filled):raise ValueError('Path gradient mask changed')
        if not np.array_equal(np.isfinite(all_rows.g_fill),cat>=0):raise ValueError('Fill gradient mask changed')
        np.testing.assert_allclose(all_rows.loc[filled,'g_path'],all_rows.loc[filled,'pc0_rich_spectrum']-(cat[filled]==0),atol=1e-14)
    for day in m['dense_dates']:
        verify(DATA/(day['date']+'_dense.parquet'),day['dense_sha256'])
        verify(Path('data/derived/scalper_v1')/(day['date']+'.parquet'),day['source_sha256'])
    verify(REPORT/'measurements.json',v['measurements_sha256']);verify('scripts/target_loss_report.py',v['script_sha256'])
    for name,expected in v['images'].items():verify(REPORT/name,expected)
    if len(v['images'])!=14 or len(m['first_tree_leaves'])!=32:raise ValueError('Incomplete audit')
    geometry=read(REPORT/'barrier_geometry.json');verify('spectrum/cognition/scalper.py',geometry['source_units_code_sha256'])
    full_log=Path('.tmp/target_loss_audit_v1_full_tests.log');focused_log=Path('.tmp/target_loss_audit_v1_focused_tests.log')
    logs={}
    for kind,path,n in [('full_before_temporal_hardening',full_log,320),('focused_after_temporal_hardening',focused_log,22)]:
        match=re.search(r'(\d+) passed in ([0-9.]+)s',path.read_text(encoding='utf-8'))
        if not match or int(match[1])!=n:raise ValueError('Incomplete test evidence')
        logs[kind]=dict(passed=int(match[1]),seconds=float(match[2]),log_sha256=sha(path))
    unchanged=subprocess.check_output(['git','diff','349d4b5','--','reports/flow_context_v2','spectrum/cognition/flow_context.py',
                                       'spectrum/cognition/flow_context_lab.py','contracts/flow_context_v2.json'],text=True)
    if unchanged:raise ValueError('Source experiment changed')
    names=['REPORT_RU.md','SOL61_TARGET_LOSS_AUDIT_RU.md','index.html','measurements.json','barrier_geometry.json','visual_receipt.json']
    result=dict(status='PASS',recorded_utc=datetime.now(timezone.utc).isoformat(),tests=logs,
        full_B_saved_prediction_parity=comparisons,source_experiment_unchanged=True,
        derivative_gradient_max_error=max(r['gradient_max_error'] for r in m['derivative_parity']),
        actual_first_tree_leaves=32,leaf_update_max_error=max(r['error'] for r in m['first_tree_leaves']),
        checked_visuals=len(v['images']),artifacts_sha256={name:sha(REPORT/name) for name in names},
        script_sha256=sha(__file__),tests_sha256=sha('tests/test_target_loss_audit.py'),
        scope='Technical diagnosis; no refit, no new economic admission; B gradients are diagnostic, not training updates')
    save(REPORT/'validation_receipt.json',result)
    print(json.dumps(dict(status='PASS',prediction_pairs=len(comparisons),max_prediction_error=max(r['max_probability_error'] for r in comparisons),
                          images=14,leaves=32),indent=2))


if __name__=='__main__':main()
