"""Independent numerical replay of saved model logits and delayed calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .engine import CLASSES,FAMILIES,NS
from .run import ROOT,digest,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();out=ROOT/args.run
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf8'))
    source_checks=0
    for p,expected in manifest['source_hashes'].items():
        if digest(p)!=expected:raise AssertionError(f'Changed source: {p}')
        source_checks+=1
    for name,expected in manifest['artifacts'].items():
        if digest(out/name)!=expected:raise AssertionError(f'Changed result: {name}')
    f=pd.read_parquet(out/'observations.parquet');pred=pd.read_parquet(out/'predictions.parquet')
    models=json.loads((out/'models.json').read_text(encoding='utf8'))
    config=json.loads((out/'config.json').read_text(encoding='utf8'));errors={}
    columns=['p_'+k for k in CLASSES]
    probabilities={k:v.set_index('candidate_id').loc[f.candidate_id,columns].to_numpy() for k,v in pred.groupby('arm')}
    for arm,m in models.items():
        raw=f[m['columns']].to_numpy(float);missing=~np.isfinite(raw)
        a=np.concatenate([np.where(missing,np.array(m['median']),raw),missing],axis=1)
        a=np.clip((a-np.array(m['mean']))/np.array(m['scale']),-8,8)
        logits=a@np.array(m['coefficients']).T+np.array(m['intercept'])
        p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        errors[arm]=float(np.max(np.abs(p-probabilities[arm])))
        if errors[arm]>1e-11:raise AssertionError('Saved classifier mismatch')
    origins=f.t_ns.to_numpy(np.int64)
    releases=np.maximum(origins+config['horizon_seconds']*NS,f.label_available_ns.to_numpy(np.int64))
    online=f.block.ne('train').to_numpy();known=f.label_known.to_numpy()
    labels=f.state.map(dict(zip(CLASSES,range(3)))).fillna(-1).to_numpy(int)
    histp=probabilities['hypotheses'];hflags=f[['h_'+k for k in FAMILIES]].to_numpy(bool)
    prior=config['adaptation']['prior_count']/3;half=config['adaptation']['half_life_seconds']*NS
    maxerr={'hot_global':0.,'hot_families':0.}
    checked_rows=0
    for i in np.flatnonzero(online):
        t=int(origins[i]);visible=online&known&(releases<t)
        indices=np.flatnonzero(visible);weights=2.**(-(t-releases[indices])/half)
        factors=[]
        subsets=[np.ones(len(indices),bool)]+[hflags[indices,k] for k in np.flatnonzero(hflags[i])]
        for subset in subsets:
            ids=indices[subset];w=weights[subset]
            observed=np.bincount(labels[ids],weights=w,minlength=3)
            expected=(histp[ids]*w[:,None]).sum(axis=0)
            factors.append((observed+prior)/(expected+prior))
        for arm,factor in [('hot_global',factors[0]),('hot_families',np.mean(factors,axis=0))]:
            p=histp[i]*factor;p/=p.sum();err=float(np.abs(p-probabilities[arm][i]).max())
            maxerr[arm]=max(err,maxerr[arm])
            if err>1e-10:raise AssertionError('Delayed feedback mismatch')
        checked_rows+=1
    clocks=[c for c in f if c.endswith('available_ns') and not c.startswith('label')]
    past_clocks=[c for c in clocks if c in ['target_available_ns','episode_available_ns','bar_context_available_ns','memory_available_ns','spectrum_available_ns','past_max_available_ns']]
    for c in past_clocks:
        if f[c].dtype.kind not in 'iu' or (f[c]>f.t_ns).any():raise AssertionError('Causal clock violation')
    if np.any(np.diff(origins)<config['horizon_seconds']*NS):raise AssertionError('Overlapping primary windows')
    examples=json.loads((out/'examples.json').read_text(encoding='utf8'))
    for r in examples:
        if not r['h_'+r['family']]:raise AssertionError('Inapplicable illustration')
        if r['selection']=='counterexample' and r['state']==r['expected']:raise AssertionError('Wrong counterexample')
    receipt=dict(status='PASS',source_payload_checks=source_checks,artifact_checks=len(manifest['artifacts']),
                 observations=len(f),saved_model_max_errors=errors,independent_delayed_rows=checked_rows,
                 independent_delayed_max_errors=maxerr,causal_clock_columns=past_clocks,
                 examples=len(examples),scope='Numerical integrity, chronology and contracts; not predictive or economic admission')
    dump(out/'verification.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
