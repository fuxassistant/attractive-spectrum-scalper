"""Audit second-stage model predictions and its independent scalar accounts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import checked,digest,dump
from .continuation import orient
from .verify import model_probability,verify_accounts,need

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();out=ROOT/a.run
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    for name,sha in manifest['artifacts'].items():checked(out/name,sha)
    for name,sha in manifest['implementation_sha256'].items():checked(ROOT/name,sha)
    targets=pd.read_parquet(out/'targets.parquet');post=pd.read_parquet(out/'post_cost.parquet');pred=pd.read_parquet(out/'predictions.parquet')
    atlas=ROOT/config['atlas'];am=json.loads((atlas/'manifest.json').read_text(encoding='utf-8'));checked(atlas/'observations.parquet',am['artifacts']['observations.parquet'])
    source=pd.read_parquet(atlas/'observations.parquet');drop=[c for c in targets if c in source and c not in ('candidate_id','t_ns','date')]
    f,cols=orient(source.drop(columns=drop).merge(targets,on=['candidate_id','t_ns','date'],validate='one_to_one'))
    models=json.loads((out/'models.json').read_text(encoding='utf-8'));count=0
    for date,g in pred.groupby('date'):
        test=f[f.date==date];cutoff=int(pd.Timestamp(date,tz='UTC').value)
        need(models['frozen']['max_release_ns']<cutoff and models[date]['max_release_ns']<cutoff,'Future continuation teacher')
        pf=model_probability(test,models['frozen']);pp=model_probability(test,models[date]);w=config['pair_blend_weight']
        for policy,p in [('continuation_frozen',pf),('continuation_blend',(1-w)*pf+w*pp),('anti_impulse_control',None)]:
            actual=g[g.policy==policy].set_index('candidate_id').loc[test.candidate_id]
            accepted=test.impulse_known.to_numpy()
            if p is not None:
                accepted=accepted&(p[:,1]+p[:,2]-p[:,3]-p[:,4]>=config['confidence_gate'])
                physical=p.copy();mask=test.momentum_side.to_numpy()==-1;physical[mask]=p[mask][:,[0,3,4,1,2]]
                need(np.allclose(physical,actual[['p'+str(j) for j in range(5)]],rtol=0,atol=2e-12),'Physical probability inversion')
            need(np.array_equal(actual.accepted,accepted),'Continuation gate')
            need(np.array_equal(actual.side,-test.momentum_side.to_numpy()),'Continuation side')
            need((actual.selection_ns==cutoff).all() and (actual.max_feedback_ns<cutoff).all(),'Continuation clock')
            count+=len(actual)
    accounts=verify_accounts(pd.read_parquet(out/'accounts.parquet'),pd.read_parquet(out/'trades.parquet'),pred,targets,post,config)
    dump(out/'verification.json',dict(status='PASS_SECOND_STAGE_PROBABILITY_ACCOUNT_AUDIT',decisions=count,accounts=accounts,verifier_sha256=digest(__file__),
         limits=['Uses orient() to reconstruct feature coordinates; reflection invariance and impulse gate have separate tests.','Target and post-cost source replay are covered by first-stage verification.','This is a second exploratory hypothesis after observed first-stage results, not an untouched confirmation.']))
    print(dict(decisions=count,accounts=accounts))


if __name__=='__main__':main()
