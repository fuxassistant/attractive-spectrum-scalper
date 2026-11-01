"""Build a causal dataset, fit frozen ablations and replay unchanged mechanics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from experiments.mechanical_lab.data import ROOT,digest,dump
from experiments.mechanical_lab.signals import build_features
from experiments.mechanical_lab.portfolio import account,prepare_outcomes
from experiments.zone_episode_lab.game import run_account
from experiments.zone_episode_lab.run import clean
from .features import make_features,add_past_placebo
from .targets import build_targets
from .model import (FrozenClassifier,feature_arms,day_clock_weights,daily_metrics,
    paired_day_interval,common_policy_clock,deterministic_accept,CLASSES)

HERE=Path(__file__).resolve().parent


def verify(path,expected):
    actual=digest(path)
    if actual!=expected:raise ValueError(f'Source identity mismatch: {path}')
    return actual


def prepare_day(receipt,config,destination):
    from .mbo_features import global_mbo_features,MBO_FEATURE_COLUMNS,BASE_CONTEXT_COLUMNS
    begun=time.perf_counter();date=receipt['date']
    source=ROOT/config['source_run'];old=source/'days'/date
    out=Path(destination)/'days'/date;out.mkdir(parents=True,exist_ok=False)
    market=Path(receipt['source']);verify(market/'manifest.json',receipt['source_manifest_sha256'])
    mm=json.loads((market/'manifest.json').read_text())
    if mm['status']!='COMPLETE':raise ValueError('Incomplete canonical market')
    input_hashes={str(market/'manifest.json'):digest(market/'manifest.json')}
    for name in ['quotes.parquet','bars.parquet','seconds.parquet']:
        input_hashes[str(market/name)]=verify(market/name,mm['payloads'][name])
    frames={}
    for name in ['candidates','events','history_tokens','spectral','outcomes','stress_outcomes']:
        path=old/(name+'.parquet');input_hashes[str(path)]=verify(path,receipt['artifact_sha256'][path.name])
        if name not in ['outcomes','stress_outcomes']:frames[name]=pd.read_parquet(path)
    c=frames['candidates'].loc[~frames['candidates'].inside].copy()
    q=pd.read_parquet(market/'quotes.parquet');seconds=pd.read_parquet(market/'seconds.parquet')
    f=make_features(c,build_features(pd.read_parquet(market/'bars.parquet')),frames['events'],frames['history_tokens'],frames['spectral'])
    coverage=int(mm['input']['end_ns'])+1
    target=build_targets(q,c,coverage_end_ns=coverage,horizon_seconds=config['target']['horizon_seconds'],max_quote_age_seconds=config['target']['max_quote_age_seconds'])
    # Route selection became known at the source's preselected decision clock.
    from experiments.mechanical_lab.data import source_for
    selection=int(source_for(date,config['route'])['entry']['selection']['selection_available_ns'])
    mbo=global_mbo_features(c,seconds,q,selection_available_ns=selection,coverage_end_ns=coverage,window_seconds=config['mbo']['window_seconds'])
    renamed={k:'x_mbo_'+k for k in MBO_FEATURE_COLUMNS}
    renamed.update({k:'x_base_'+k for k in BASE_CONTEXT_COLUMNS})
    f=f.merge(mbo.rename(columns=renamed),on='candidate_id',how='left',validate='one_to_one')
    if len(f)!=len(c):raise AssertionError('Candidate population changed')
    f.to_parquet(out/'features.parquet',index=False);target.to_parquet(out/'targets.parquet',index=False)
    r=dict(status='COMPLETE',date=date,rows=len(c),source_hashes=input_hashes,source_market=str(market),
           source_receipt_sha256=digest(old/'receipt.json'),target_counts=target.state.value_counts().to_dict(),
           elapsed_seconds=time.perf_counter()-begun,
           artifact_sha256={p.name:digest(p) for p in out.glob('*.parquet')})
    dump(out/'receipt.json',clean(r));return r


def assign_blocks(frame,config):
    out=np.full(len(frame),'',object)
    for name,(a,b) in config['splits'].items():out[(frame.date>=a)&(frame.date<=b)]=name
    if (out=='').any():raise ValueError('Unassigned chronological date')
    return out


def fit_and_evaluate(out,config,receipts):
    f=pd.concat([pd.read_parquet(out/'days'/r['date']/'features.parquet') for r in receipts],ignore_index=True)
    y=pd.concat([pd.read_parquet(out/'days'/r['date']/'targets.parquet') for r in receipts],ignore_index=True)
    f['block']=assign_blocks(f,config)
    f,placebo=add_past_placebo(f,f.block.eq('train'))
    common=[c for c in y if c in f and c!='candidate_id']
    data=f.merge(y.drop(columns=common),on='candidate_id',validate='one_to_one')
    if data.candidate_id.duplicated().any():raise ValueError('Duplicate dataset rows')
    train=data.loc[data.block.eq('train') & data.label_known].copy()
    cutoff=int(pd.Timestamp(config['splits']['calibration'][0],tz='UTC').value)
    if (train.label_available_ns>=cutoff).any():raise ValueError('Unmatured training label at prediction boundary')
    f.to_parquet(out/'features.parquet',index=False);y.to_parquet(out/'targets.parquet',index=False)
    data.to_parquet(out/'dataset.parquet',index=False)
    dump(out/'placebo_receipt.json',clean(placebo))
    models={};predictions=[];daily=[];calibration=[];metrics=[]
    arms=feature_arms(f)
    weights=day_clock_weights(train)
    prior=np.array([weights[train.state.eq(k)].sum() for k in CLASSES]);prior/=prior.sum()
    for arm,columns in arms.items():
        model=FrozenClassifier(columns,config['model']).fit(train)
        models[arm]=model.receipt()
        p=model.predict(data)
        prediction=data[['candidate_id','date','t_ns','block','state','label_known','placebo_known']].copy()
        for j,name in enumerate(['p_return','p_adverse','p_timeout']):prediction[name]=p[:,j]
        prediction['arm']=arm;predictions.append(prediction)
        for block in ['calibration','evaluation','extension']:
            for cohort in ['all_known','donor_supported']:
                mask=data.block.eq(block)&data.label_known
                if cohort=='donor_supported':mask &= data.placebo_known
                known=data.loc[mask]
                if known.empty:continue
                d,l=daily_metrics(known,p[mask]);d['arm']=arm;d['block']=block;d['cohort']=cohort;daily.append(d)
                metrics.append(dict(arm=arm,block=block,cohort=cohort,rows=len(known),days=len(d),
                    logloss=float(d.logloss.mean()),brier=float(d.brier.mean()),return_brier=float(d.return_brier.mean())))
                if cohort=='all_known':
                    w=day_clock_weights(known);bins=np.minimum((p[mask,0]*10).astype(int),9)
                    for b in range(10):
                        pick=bins==b
                        if not pick.any():continue
                        calibration.append(dict(arm=arm,block=block,bin=b,rows=int(pick.sum()),weight=float(w[pick].sum()),
                            predicted=float(np.average(p[mask,0][pick],weights=w[pick])),observed=float(np.average(known.state.eq('RETURN_FIRST').to_numpy()[pick],weights=w[pick]))))
        print(json.dumps(dict(stage='model',arm=arm,features=len(columns),iterations=model.estimator.n_iter_.tolist())),flush=True)
    # A constant train prior is evaluated on exactly the same known cohorts.
    for block in ['calibration','evaluation','extension']:
        known=data.loc[data.block.eq(block)&data.label_known]
        d,_=daily_metrics(known,np.tile(prior,(len(known),1)));d['arm']='train_prior';d['block']=block;d['cohort']='all_known';daily.append(d)
        metrics.append(dict(arm='train_prior',block=block,cohort='all_known',rows=len(known),days=len(d),
            logloss=float(d.logloss.mean()),brier=float(d.brier.mean()),return_brier=float(d.return_brier.mean())))
    predictions=pd.concat(predictions,ignore_index=True);daily=pd.concat(daily,ignore_index=True)
    predictions.to_parquet(out/'predictions.parquet',index=False);daily.to_parquet(out/'daily_metrics.parquet',index=False)
    pd.DataFrame(metrics).to_parquet(out/'predictive_metrics.parquet',index=False)
    pd.DataFrame(calibration).to_parquet(out/'calibration.parquet',index=False)
    dump(out/'models.json',clean(dict(fit_scope='train only',models=models,train_prior=prior.tolist(),runtime=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__))))
    contrasts=[]
    for a,b in config['evaluation']['primary_contrasts']:
        for block in ['calibration','evaluation','extension']:
            cohort='donor_supported' if b=='past_placebo' else 'all_known'
            aa=daily.loc[daily.arm.eq(a)&daily.block.eq(block)&daily.cohort.eq(cohort)].set_index('date')
            bb=daily.loc[daily.arm.eq(b)&daily.block.eq(block)&daily.cohort.eq(cohort)].set_index('date')
            if not aa.index.equals(bb.index):raise ValueError('Unmatched daily comparison')
            for metric in ['logloss','brier']:
                contrasts.append(dict(arm=a,reference=b,block=block,cohort=cohort,metric=metric,**paired_day_interval(aa[metric]-bb[metric])))
    pd.DataFrame(contrasts).to_parquet(out/'contrasts.parquet',index=False)
    # Coarse comparisons expose support, not causal effects or independent trials.
    matched=[]
    keys=['side','match_distance_bin','match_spread_bin','match_phase','match_vol_bin']
    for arm,g in predictions.groupby('arm'):
        z=data[['candidate_id',*keys]].merge(g,on='candidate_id',validate='one_to_one')
        for key,h in z.loc[z.label_known].groupby(['block',*keys]):
            w=day_clock_weights(h)
            matched.append(dict(arm=arm,block=key[0],**dict(zip(keys,key[1:])),rows=len(h),days=h.date.nunique(),
                observed_return=float(np.average(h.state.eq('RETURN_FIRST'),weights=w)),predicted_return=float(np.average(h.p_return,weights=w))))
    pd.DataFrame(matched).to_parquet(out/'matched_strata.parquet',index=False)
    return data,predictions,contrasts


def replay(out,config,data,predictions):
    clock=common_policy_clock(data).drop(columns=['deadline_ns'])
    source=ROOT/config['source_run'];outcomes=[];stress=[]
    for date in sorted(data.date.unique()):
        outcomes.append(pd.read_parquet(source/'days'/date/'outcomes.parquet'))
        stress.append(pd.read_parquet(source/'days'/date/'stress_outcomes.parquet'))
    outcomes=pd.concat(outcomes,ignore_index=True);stress=pd.concat(stress,ignore_index=True)
    summaries=[];trades=[];random_rows=[];thresholds={};selection=[];random_strata=[]
    keys=['side','match_distance_bin','match_spread_bin','match_phase','match_vol_bin']
    for arm,g in predictions.groupby('arm',sort=False):
        scores=g.set_index('candidate_id').p_return
        cal=clock.loc[clock.block.eq('calibration')]
        threshold=float(np.quantile(cal.candidate_id.map(scores),config['policy']['calibration_quantile']))
        rate=float(cal.candidate_id.map(scores).ge(threshold).mean())
        cal=cal.assign(accepted=cal.candidate_id.map(scores).ge(threshold))
        stratum_rates={}
        prior_count=config['policy']['random_stratum_prior_count']
        for key,h in cal.groupby(keys):
            probability=(h.accepted.sum()+prior_count*rate)/(len(h)+prior_count)
            stratum_rates[key]=float(probability)
            random_strata.append(dict(arm=arm,**dict(zip(keys,key)),calibration_rows=len(h),accepted=int(h.accepted.sum()),probability=float(probability)))
        thresholds[arm]=dict(threshold=threshold,calibration_rows=len(cal),acceptance_rate=rate,uses_labels=False,
                            available_ns=int(pd.Timestamp(config['splits']['evaluation'][0],tz='UTC').value))
        for block in ['evaluation','extension']:
            candidates=clock.loc[clock.block.eq(block)].copy()
            candidates['score']=candidates.candidate_id.map(scores)
            candidates['accepted']=candidates.score.ge(threshold)
            selected=candidates.loc[candidates.accepted].copy()
            selected_meta=candidates[['candidate_id','t_ns','date','block','score','accepted','label_known']].copy();selected_meta['arm']=arm;selection.append(selected_meta)
            dates=set(candidates.date)
            base=outcomes.loc[outcomes.date.isin(dates)];st=stress.loc[stress.date.isin(dates)]
            for condition,direction,paired in [('base','toward',base),('away','away',base),('stress','toward',st)]:
                s,log=run_account(selected,paired,direction=direction)
                summaries.append(dict(arm=arm,block=block,condition=condition,proposed=len(selected),opportunities=len(candidates),threshold=threshold,**s))
                if len(log):
                    log['arm']=arm;log['block']=block;log['condition']=condition;trades.append(log)
            lookup=prepare_outcomes(base)
            probabilities=np.array([stratum_rates.get(tuple(x),rate) for x in candidates[keys].to_numpy()])
            for seed in range(1,config['policy']['random_seeds']+1):
                offered=candidates.loc[deterministic_accept(candidates.candidate_id,probabilities,seed)]
                s,_=account(offered,lookup,direction='signal')
                random_rows.append(dict(arm=arm,block=block,seed=seed,mean_accept_probability=float(probabilities.mean()),offered=len(offered),
                    trades=s['trades'],net_usd=s['net_usd'],complete=s['capital_curve_complete']))
            rows=random_rows[-config['policy']['random_seeds']:];complete=all(r['complete'] for r in rows)
            for r in summaries[-3:]:
                r.update(random_complete_runs=sum(x['complete'] for x in rows),random_seeds=len(rows),random_reference='base',
                         random_p95_usd=float(np.quantile([x['net_usd'] for x in rows],.95)) if complete else None,
                         random_median_usd=float(np.median([x['net_usd'] for x in rows])) if complete else None)
    pd.DataFrame(summaries).to_parquet(out/'policy_summary.parquet',index=False)
    pd.DataFrame(random_rows).to_parquet(out/'random_controls.parquet',index=False)
    pd.DataFrame(random_strata).to_parquet(out/'random_strata_rates.parquet',index=False)
    pd.concat(selection,ignore_index=True).to_parquet(out/'policy_selection.parquet',index=False)
    (pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()).to_parquet(out/'account_trades.parquet',index=False)
    clock[['candidate_id','t_ns','date','block']].to_parquet(out/'policy_clock.parquet',index=False)
    dump(out/'thresholds.json',clean(thresholds));return summaries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();config=json.loads((HERE/'config.json').read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();source=ROOT/config['source_run'];sm=json.loads((source/'manifest.json').read_text())
    if sm['status']!='COMPLETE_DIAGNOSTIC_ONLY':raise ValueError('Parent experiment is not verified complete')
    for name,h in sm['implementation_sha256'].items():verify(ROOT/name,h)
    identity={str(p.relative_to(ROOT)):digest(p) for p in [HERE/x for x in ['config.json','features.py','model.py','targets.py','mbo_features.py','run.py']]}
    identity.update({str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'experiments/mechanical_lab'/x for x in ['data.py','signals.py','execution.py','portfolio.py']]})
    dump(out/'config.json',config)
    base_manifest=dict(status='RUNNING',source_manifest_sha256=digest(source/'manifest.json'),implementation_sha256=identity,
        scope='Previously inspected dates; conditional known-outcome predictive test and separate replay; no untouched holdout')
    dump(out/'manifest.json',base_manifest);receipts=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(prepare_day,r,config,str(out)) for r in sorted(sm['receipts'],key=lambda r:r['date'])]
            for job in as_completed(jobs):
                r=job.result();receipts.append(r);print(json.dumps({k:r[k] for k in ['date','rows','target_counts','elapsed_seconds']}),flush=True)
        receipts=sorted(receipts,key=lambda r:r['date'])
        data,predictions,contrasts=fit_and_evaluate(out,config,receipts)
        policies=replay(out,config,data,predictions)
        for name,h in identity.items():verify(ROOT/name,h)
    except Exception as exc:
        dump(out/'manifest.json',dict(base_manifest,status='FAILED',error=repr(exc),completed_days=[r['date'] for r in receipts]));raise
    diagnostics=dict(target_counts=data.groupby('block').state.value_counts().to_dict(),
                     placebo_coverage=data.groupby('block').placebo_known.mean().to_dict(),
                     rows=len(data),common_policy_clocks=data.t_ns.nunique(),economic_admission=False,
                     reason='No untouched confirmation; geometric forecast is separate from executable capital')
    diagnostics['target_counts']={str(k):int(v) for k,v in diagnostics['target_counts'].items()}
    dump(out/'diagnostics.json',clean(diagnostics))
    result=dict(base_manifest,status='COMPLETE_DIAGNOSTIC_ONLY',elapsed_seconds=time.perf_counter()-started,days=len(receipts),
                rows=len(data),receipts=receipts,artifacts={p.name:digest(p) for p in out.glob('*.parquet')},
                metadata_hashes={p.name:digest(p) for p in out.glob('*.json') if p.name!='manifest.json'})
    dump(out/'manifest.json',clean(result));print(json.dumps({k:result[k] for k in ['status','elapsed_seconds','days','rows']}),flush=True)


if __name__=='__main__':main()
