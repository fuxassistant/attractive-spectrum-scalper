"""Build the atlas once in a new directory; all evaluation is chronological."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.zone_conditional_v1.model import FrozenClassifier
from experiments.zone_conditional_v1.targets import build_targets
from .engine import (NS, CLASSES, FAMILIES, PATH_COLUMNS, MEMORY_COLUMNS,
                     decision_cohort, causal_paths, define_hypotheses, DelayedCalibrator)

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARMS=('geometry','context','hypotheses','hot_global','hot_families')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def dump(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def checked(p,sha):
    actual=digest(p)
    if actual!=sha:raise ValueError(f'Changed source: {p}')
    return actual


def scores(labels,p):
    y=np.eye(3)[labels]
    return -np.log(np.clip(p[np.arange(len(labels)),labels],1e-12,1)),((p-y)**2).sum(axis=1)


def summarize(f,pred,config):
    rows=[]; daily=[];family=[]
    for block in ['calibration','evaluation','extension']:
        mask=f.block.eq(block)&f.label_known
        g=f.loc[mask].copy(); labels=g.state.map(dict(zip(CLASSES,range(3)))).to_numpy()
        for arm in ARMS:
            p=pred[arm][mask];ll,br=scores(labels,p)
            table=pd.DataFrame({'date':g.date.to_numpy(),'logloss':ll,'brier':br})
            d=table.groupby('date')[['logloss','brier']].mean()
            rows.append(dict(block=block,arm=arm,n=len(g),days=len(d),**d.mean().to_dict()))
            for date,r in d.iterrows():daily.append(dict(block=block,arm=arm,date=date,**r.to_dict()))
        for key,(title,event) in FAMILIES.items():
            active=mask&f['h_'+key]; part=f.loc[active]
            if not len(part):
                family.append(dict(block=block,family=key,title=title,n=0,days=0));continue
            obs=part.state.eq(CLASSES[event]).to_numpy(float)
            base=pred['context'][active,event]
            labels=part.state.map(dict(zip(CLASSES,range(3)))).to_numpy()
            ll0,_=scores(labels,pred['context'][active]);ll1,_=scores(labels,pred['hypotheses'][active])
            d=pd.DataFrame({'date':part.date.to_numpy(),'observed':obs,'context':base,'residual':obs-base,'delta_logloss':ll1-ll0}).groupby('date').mean()
            family.append(dict(block=block,family=key,title=title,n=len(part),days=len(d),
                coverage=len(part)/int(mask.sum()),row_success=float(obs.mean()),
                day_success=float(d.observed.mean()),context_expected=float(d.context.mean()),
                residual=float(d.residual.mean()),delta_logloss=float(d.delta_logloss.mean()),
                conflict_fraction=float((part.active_hypotheses>1).mean())))
    daily=pd.DataFrame(daily);contrasts=[];rng=np.random.default_rng(271828)
    for block in ['calibration','evaluation','extension']:
        p=daily[daily.block.eq(block)].pivot(index='date',columns='arm',values='logloss')
        for arm,base in config['evaluation']['primary_contrasts']:
            delta=(p[arm]-p[base]).to_numpy()
            boot=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
            contrasts.append(dict(block=block,arm=arm,base=base,delta=float(delta.mean()),
                                  low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),days=len(delta)))
    return rows,daily,family,contrasts


def select_examples(f,pred):
    """Prototypes are X-only; separate failure slots explicitly use outcomes."""
    later=f.block.isin(['evaluation','extension'])
    columns=['distance_r','past_net30_r','past_migration_r','past_range120_r',
             'x_mem_recent_pass_fraction','x_mem_log_median_dwell_seconds']
    train=f.block.eq('train'); med=f.loc[train,columns].median();scale=f.loc[train,columns].std().replace(0,1).fillna(1)
    result=[]
    for key,(_,expected) in FAMILIES.items():
        pool=f.loc[later&f['h_'+key]].copy()
        if pool.empty:continue
        # The prototype centre uses only X in the displayed pool. It is an
        # editorial representative, never a fitted predictor or historical signal.
        x=(pool[columns].fillna(med)-med)/scale
        pool['prototype_distance']=((x-x.median())**2).mean(axis=1)
        pool=pool.sort_values(['prototype_distance','t_ns','candidate_id'])
        first=pool.iloc[0]; chosen=[(first,'prototype')]
        second=pool[pool.date.ne(first.date)]
        if len(second):chosen.append((second.iloc[0],'prototype'))
        failed=pool[pool.label_known&pool.state.ne(CLASSES[expected])&~pool.candidate_id.isin([r.candidate_id for r,_ in chosen])]
        if len(failed):chosen.append((failed.iloc[0],'counterexample'))
        for row,kind in chosen:
            i=int(row.name)
            item={k:row[k] for k in f.columns if k not in ['mbometa_dep_global_counter_deltas','mbometa_now_global_counter_deltas']}
            item.update(family=key,selection=kind,expected=CLASSES[expected],
                        probabilities={arm:pred[arm][i] for arm in ARMS})
            result.append(clean(item))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();config=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    dump(out/'config.json',config)
    source=ROOT/config['source'];manifest=json.loads((source/'manifest.json').read_text())
    hashes={str(source/'manifest.json'):digest(source/'manifest.json')}
    for name in ['features.parquet','targets.parquet']:
        hashes[str(source/name)]=checked(source/name,manifest['artifacts'][name])
    raw=pd.read_parquet(source/'features.parquet');targets=pd.read_parquet(source/'targets.parquet')
    f=decision_cohort(raw,config['horizon_seconds'])
    # Keep the original saved split and check against the frozen date contract.
    for block,(a,b) in config['splits'].items():
        if not f.loc[f.block.eq(block),'date'].between(a,b).all():raise ValueError('Split drift')
    for col in ['target_available_ns','episode_available_ns','memory_available_ns','bar_context_available_ns']:
        if (f[col]>f.t_ns).any():raise ValueError(f'Future X: {col}')
    receipts={r['date']:r for r in manifest['receipts']};paths=[];rechecks=[]
    for date,part in f.groupby('date',sort=True):
        r=receipts[date];market=Path(r['source_market']);qpath=market/'quotes.parquet'
        hashes[str(qpath)]=checked(qpath,r['source_hashes'][str(qpath)])
        mpath=market/'manifest.json';hashes[str(mpath)]=checked(mpath,r['source_hashes'][str(mpath)])
        mm=json.loads(mpath.read_text());q=pd.read_parquet(qpath)
        paths.append(causal_paths(q,part,config['history_seconds']))
        y=build_targets(q,part,coverage_end_ns=int(mm['input']['end_ns'])+1,horizon_seconds=config['horizon_seconds'])
        reference=targets.set_index('candidate_id').loc[y.candidate_id]
        for col in ['state','reason','label_known','first_event_ns','label_available_ns','origin_mid_x2_nano','adverse_x2_nano']:
            if not np.array_equal(y[col].to_numpy(),reference[col].to_numpy()):raise AssertionError(f'Target mismatch {date} {col}')
        rechecks.append(dict(date=date,rows=len(y),target_columns=7,exact=True))
        print(json.dumps(dict(stage='materialize',date=date,rows=len(part))),flush=True)
    f=f.merge(pd.concat(paths,ignore_index=True),on='candidate_id',validate='one_to_one')
    f,thresholds=define_hypotheses(f,config);dump(out/'thresholds.json',thresholds)
    common=[c for c in targets if c in f and c!='candidate_id']
    f=f.merge(targets.drop(columns=common),on='candidate_id',validate='one_to_one').sort_values('t_ns').reset_index(drop=True)
    f.to_parquet(out/'observations.parquet',index=False)
    train=f.block.eq('train')&f.label_known
    cutoff=int(pd.Timestamp('2026-08-17',tz='UTC').value)
    if (f.loc[train,'label_available_ns']>=cutoff).any():raise AssertionError('Unmatured training response')
    base=[c for c in f if c.startswith('x_base_')]
    arms={'geometry':base,'context':base+PATH_COLUMNS,
          'hypotheses':base+PATH_COLUMNS+MEMORY_COLUMNS+['h_'+k for k in FAMILIES]}
    pred={};models={}
    for arm,columns in arms.items():
        model=FrozenClassifier(columns,config['model']).fit(f.loc[train]);pred[arm]=model.predict(f);models[arm]=model.receipt()
    pred['hot_global']=pred['hypotheses'].copy();pred['hot_families']=pred['hypotheses'].copy()
    calibrator=DelayedCalibrator(config['adaptation']['half_life_seconds'],config['adaptation']['prior_count'])
    feedback=[]
    for i,r in f.iterrows():
        keys=[k for k in FAMILIES if r['h_'+k]]
        # The pretrained model owns train data. Online state starts empty on calibration.
        if r.block=='train':feedback.append(dict(candidate_id=r.candidate_id,feedback_count=0,max_feedback_origin_ns=-1,max_feedback_release_ns=-1,global_effective_support=0.,local_effective_support=0.));continue
        t=int(r.t_ns);p=pred['hypotheses'][i]
        pred['hot_global'][i],_=calibrator.predict(t,p,keys,local=False)
        pred['hot_families'][i],audit=calibrator.predict(t,p,keys,local=True)
        audit['candidate_id']=int(r.candidate_id);feedback.append(audit)
        if audit['max_feedback_release_ns']>=t:raise AssertionError('Future feedback')
        label=CLASSES.index(r.state) if r.label_known else -1
        release=max(t+config['horizon_seconds']*NS,int(r.label_available_ns))
        calibrator.enqueue(t,release,p,label,keys)
    feedback=pd.DataFrame(feedback);feedback.to_parquet(out/'feedback.parquet',index=False)
    long=[]
    for arm,p in pred.items():
        z=f[['candidate_id','t_ns','date','block','state','label_known']].copy();z['arm']=arm
        for j,k in enumerate(CLASSES):z['p_'+k]=p[:,j]
        long.append(z)
    pd.concat(long,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    metrics,daily,families,contrasts=summarize(f,pred,config)
    daily.to_parquet(out/'daily_metrics.parquet',index=False)
    dump(out/'models.json',models);dump(out/'examples.json',select_examples(f,pred))
    result=dict(status='COMPLETE_EXPLORATORY_PREDICTION_ONLY',source_rows=len(raw),decision_rows=len(f),
                split_counts=f.groupby('block').size().to_dict(),unknown_targets=int((~f.label_known).sum()),
                unknown_history=int((~f.past_known).sum()),thresholds=thresholds,
                metrics=metrics,families=families,contrasts=contrasts,
                applicability={k:int(f['h_'+k].sum()) for k in FAMILIES},
                conflicts=int((f.active_hypotheses>1).sum()),uncovered=int(f.active_hypotheses.eq(0).sum()),
                target_rechecks=rechecks,elapsed_seconds=time.perf_counter()-start,
                runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_hashes=hashes,source_paths={d:r['source_market'] for d,r in receipts.items()},
                economic_admission=False,predictive_admission=False,untouched_holdout=False)
    dump(out/'summary.json',result)
    code={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*') if p.is_file()}
    dump(out/'manifest.json',dict(status=result['status'],config_sha256=digest(out/'config.json'),implementation_sha256=code,
                                source_hashes=hashes,artifacts={p.name:digest(p) for p in out.glob('*') if p.is_file()}))
    print(json.dumps(dict(stage='complete',rows=len(f),applicability=result['applicability'],metrics=metrics),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
