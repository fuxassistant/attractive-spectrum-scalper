"""Explicit second stage: intraday correction of daily-model forecast errors.

This stage was specified after seeing the failure of daily/weekly adaptation.
It is exploratory, not a fresh holdout. Every feedback item still matures only
after the full target horizon. Counts describe support, not independent trials.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT,digest
from experiments.hypothesis_atlas_v1.engine import DelayedCalibrator
from .prepare import check,dump
from .engine import joined,contexts
from .evaluate import decisions,daily_quality,summarize,calibration,cost_metrics,accounts

ARMS=[('online_global2h',7200,False),('online_local2h',7200,True),('online_local8h',28800,True)]


def correct(base,features,targets,config):
    frame,_=contexts(joined(features,targets),config); x=frame.set_index('candidate_id'); y=targets.set_index('candidate_id'); results=[]
    for policy,half_life,local in ARMS:
        calibrator=None; segment=None; rows=[]
        for r in base.sort_values('t_ns').itertuples():
            f=x.loc[r.candidate_id]; teacher=y.loc[r.candidate_id]
            if segment!=int(f.segment):
                segment=int(f.segment); calibrator=DelayedCalibrator(half_life_seconds=half_life,prior_count=20)
            p=np.array([getattr(r,'p'+str(j)) for j in range(5)])
            up=p[1]+p[2]; down=p[3]+p[4]; momentum=int(f.momentum_side)
            prior=np.array([p[0],up if momentum==1 else down,down if momentum==1 else up])
            q,receipt=calibrator.predict(r.t_ns,prior,(f.cell,),local=local)
            raw=q[1]-q[2]; side=-momentum*(1 if raw>=0 else -1)
            supported=receipt['global_effective_support']>=12
            accept=bool(f.impulse_known and supported and abs(raw)>=.1)
            reason='ACCEPT' if accept else 'SMALL_OR_UNKNOWN_IMPULSE' if not f.impulse_known else 'INSUFFICIENT_RECENT_FEEDBACK' if not supported else 'LOW_CORRECTED_SCORE'
            row=r._asdict(); row.pop('Index',None)
            row.update(policy=policy,side=side,score=abs(raw),accepted=accept,reason=reason,
                       max_feedback_ns=max(r.max_feedback_ns,receipt['max_feedback_release_ns']),**receipt)
            # Online feedback may be used after midnight but always before t.
            row['selection_ns']=r.t_ns
            corrected_up=q[1] if momentum==1 else q[2]; corrected_down=q[2] if momentum==1 else q[1]
            row['p0']=q[0]; row['p1']=corrected_up*p[1]/up; row['p2']=corrected_up*p[2]/up
            row['p3']=corrected_down*p[3]/down; row['p4']=corrected_down*p[4]/down
            rows.append(row)
            if f.impulse_known:
                label=0 if teacher.event_sign==0 else 1 if teacher.event_sign==momentum else 2
                calibrator.enqueue(r.t_ns,int(teacher.release_ns),prior,label if teacher.known else -1,(f.cell,))
        results.extend(rows)
    return pd.DataFrame(results)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    source=ROOT/args.source; out=ROOT/args.out; out.mkdir(parents=True,exist_ok=False)
    config=json.loads((source/'config.json').read_text()); sm=json.loads((source/'manifest.json').read_text()); inputs=Path(sm['source'])
    check(source/'predictions.parquet',sm['artifacts']['predictions.parquet']); f=pd.read_parquet(inputs/'features.parquet')
    y=pd.read_parquet(inputs/'targets.parquet'); y=y[y.target.eq(config['primary_target'])]
    post=pd.read_parquet(inputs/'post_cost.parquet'); post=post[post.target.eq(config['primary_target'])]
    pred=pd.read_parquet(source/'predictions.parquet'); pred=pred[pred.target.eq(config['primary_target'])]
    base=pred[pred.policy.eq('switch5_g10')]; result=correct(base,f,y,config)
    result=pd.concat([result,pred[pred.policy.eq('anti_impulse')]],ignore_index=True)
    d=decisions(result,y); daily=daily_quality(d); quality,null=summarize(d,daily,config); cal,curves=calibration(d)
    acc,trades,risknull=accounts(result,y,post,config,policies=[a[0] for a in ARMS])
    for name,table in dict(predictions=result,decisions=d,daily_quality=daily,quality=quality,quality_null=null,calibration=cal,
         calibration_curves=curves,cost_metrics=cost_metrics(d,post),accounts=acc,trades=trades,random_accounts=risknull).items():
        table.to_parquet(out/(name+'.parquet'),index=False)
    config['stage']='SECOND_EXPLORATORY_STAGE'; config['online_arms']=ARMS; config['minimum_effective_support']=12; config['online_prior_count']=20
    dump(out/'config.json',config)
    dump(out/'manifest.json',dict(status='COMPLETE_EXPLORATORY',source=str(inputs),parent=str(source),source_manifest_sha256=digest(inputs/'manifest.json'),
        parent_manifest_sha256=digest(source/'manifest.json'),implementation_sha256=digest(__file__),artifacts={p.name:digest(p) for p in out.glob('*.parquet')}))
    print(quality[quality.block.eq('all')][['policy','selected','decisive_accuracy','day_reward','day_reward_lo','day_reward_hi','mean_bid_per_oz']].round(4).to_string(index=False))


if __name__=='__main__': main()
