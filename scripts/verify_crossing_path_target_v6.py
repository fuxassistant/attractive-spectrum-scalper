"""Final saved-data admission: bound inputs, Parquet masks and learning view."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from spectrum.research_v6.targets import learning_view
from scripts.build_crossing_path_target_v6 import sha, save


def main():
    root=Path('data/derived/crossing_path_target_v6')
    report=Path('reports/crossing_path_target_v6')
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='COMPLETE' and len(manifest['days'])==20
    hashes=dict(manifest['identity'])
    total=0
    profile_targets=0
    for day in manifest['days']:
        assert day['status']=='COMPLETE'
        for path,digest in day['inputs'].items():
            if path in hashes: assert hashes[path]==digest
            hashes[path]=digest
        for name,digest in day['outputs'].items(): hashes[str(root/name)]=digest
        frame=pd.read_parquet(root/f"{day['date']}_target.parquet")
        assert len(frame)==day['cells']
        assert (frame.groupby('t_ns').size()==9).all()
        assert not frame.duplicated(['t_ns','scale','role']).any()
        view=learning_view(frame)
        mask=view['profile_mask']
        assert np.isfinite(view['profile'][mask]).all()
        assert not mask[~frame.profile_known.to_numpy()].any()
        assert not np.isfinite(view['profile'][~frame.profile_known.to_numpy()]).any()
        assert set(view['hit'][view['hit_mask']]) <= {0,1}
        assert (view['hit'][~view['hit_mask']]==-1).all()
        assert (frame.loc[~frame.profile_known,'v6_event_id']=='').all()
        assert (frame.loc[frame.profile_known,'target_available_ns']==frame.loc[frame.profile_known,'cross_ns']).all()
        assert frame.loc[frame.profile_known,'packet_rank'].notna().all()
        # Observed invalid directions cannot be falsely made into positive endpoints.
        assert ((frame.net_L>0)&(frame.net_S>0)).sum()==0
        for side in ('L','S'):
            frac=frame[f'underwater_fraction_{side}'].dropna()
            assert frac.between(-1e-10,1+1e-10).all()
            assert (frame[f'max_drawdown_R_{side}'].dropna()>=-1e-10).all()
        clock=pd.read_parquet(root/f"{day['date']}_causal_clock.parquet")
        assert (clock.available_ns==clock.t_ns).all()
        for seconds in (60,180,360,86400):
            assert np.allclose(clock[f'utc_sin_{seconds}']**2+clock[f'utc_cos_{seconds}']**2,1)
        total+=len(frame)
        profile_targets+=int(mask.sum())
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError('bound artifact changed '+path)
    summary=json.loads((report/'summary.json').read_text())
    assert summary['cells']==total
    for name,digest in summary['images'].items(): assert sha(report/name)==digest
    assert summary['artifact_inputs'][str(root/'manifest.json')]==sha(root/'manifest.json')
    sources=['contracts/crossing_path_target_v6.json','configs/research/crossing_path_target_v6.json',
        'spectrum/research_v6/targets.py','scripts/build_crossing_path_target_v6.py',
        'scripts/report_crossing_path_target_v6.py','scripts/verify_crossing_path_target_v6.py',
        'tests/test_research_v6_targets.py']
    outputs=['summary.json','visual_cases.json','data_receipt.json','index.html','sol61_visual_audit.json','test_receipt.json']
    result=dict(status='PASS',cells=total,profile_target_values=profile_targets,
        rehashed_inputs_and_dataset_outputs=len(hashes),
        source_sha256={p:sha(p) for p in sources},
        report_sha256={p:sha(report/p) for p in outputs}, image_sha256=summary['images'],
        admission='TARGET_SEMANTICS_ONLY_NO_TRAINING_OR_EXECUTION_ADVANTAGE',
        checks=['20 COMPLETE days and 9 unique lines per origin','all bound input/output hashes re-read',
            'unknown and no-hit profiles remain NaN and masked','known learning targets finite',
            'same-cross LONG and SHORT cannot both be net-positive','conditional availability and event packets',
            'causal clock time and unit-circle identities','all seven inspected images match receipt'])
    save(report/'completion_receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','report_sha256','image_sha256')},indent=2))


if __name__=='__main__': main()
