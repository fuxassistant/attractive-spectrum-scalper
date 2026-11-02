"""Frequency-only proper-score baseline and descriptive market-state contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT,digest
from .prepare import dump,check
from .engine import joined,contexts
from .evaluate import calibration,decisions,bootstrap_mean


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args(); out=ROOT/args.run
    manifest=json.loads((out/'manifest.json').read_text()); source=Path(manifest['source']); config=json.loads((out/'config.json').read_text())
    f=pd.read_parquet(source/'features.parquet'); y=pd.read_parquet(source/'targets.parquet'); y=y[y.target.eq(config['primary_target'])]
    frame,_=contexts(joined(f,y),config); receipt=json.loads((out/'usd2_300_models.json').read_text()); rows=[]
    for date,g in frame.groupby('date'):
        if date+'_full5' not in receipt: continue
        for window in [2,5,10]:
            prior=np.tile(receipt[date+'_full'+str(window)]['prior'],(len(g),1)); p=prior.copy(); neg=g.momentum_side.to_numpy()==-1
            p[neg]=prior[neg][:,[0,3,4,1,2]]
            z=g[['candidate_id','date','target','source_population','t_ns']].copy()
            z['policy']='frequency'+str(window); z['accepted']=False; z['side']=-g.momentum_side.to_numpy()
            for j in range(5): z['p'+str(j)]=p[:,j]
            rows.append(z)
    pred=pd.concat(rows,ignore_index=True); cal,curves=calibration(decisions(pred,y)); cal.to_parquet(out/'frequency_calibration.parquet',index=False)
    # Factor the five-state loss into reachability, direction, and path shape.
    # This prevents an activity/path prediction gain being sold as directional skill.
    primary=pd.read_parquet(out/'predictions.parquet'); primary=primary[primary.target.eq('usd2_300')&primary.policy.isin(['coarse5','fade2_g10','fade5_g10','fade10_g10','no_flow5','no_memory5','multiscale5'])]
    full=pd.concat([primary,pred],ignore_index=True).merge(y[['candidate_id','known','class_index']],on='candidate_id',validate='many_to_one')
    components=[]
    for (policy,date),g in full[full.known].groupby(['policy','date']):
        p=g[['p'+str(j) for j in range(5)]].to_numpy(); cls=g.class_index.to_numpy(int); hit=cls!=0
        hitprob=1-p[:,0]; up=p[:,1]+p[:,2]; down=p[:,3]+p[:,4]; upwards=np.isin(cls,[1,2])
        direction=np.where(upwards,up,down); true=p[np.arange(len(p)),cls]
        reach=-np.log(np.clip(np.where(hit,hitprob,p[:,0]),1e-12,1))
        direct=np.where(hit,-np.log(np.clip(direction/np.maximum(hitprob,1e-12),1e-12,1)),0.)
        shape=np.where(hit,-np.log(np.clip(true/np.maximum(direction,1e-12),1e-12,1)),0.)
        assert np.allclose(reach+direct+shape,-np.log(np.maximum(true,1e-12)))
        components.append(dict(policy=policy,date=date,reach_ll=reach.mean(),direction_ll_contribution=direct.mean(),shape_ll_contribution=shape.mean(),
            total_ll=(reach+direct+shape).mean(),conditional_direction_ll=direct[hit].mean(),
            conditional_direction_accuracy=((up[hit]>=down[hit])==upwards[hit]).mean(),hit_fraction=hit.mean()))
    components=pd.DataFrame(components); components.to_parquet(out/'score_components.parquet',index=False)
    paired=components.pivot(index='date',columns='policy',values='total_ll'); contrasts=[]
    for policy in ['coarse5','fade2_g10','fade5_g10','fade10_g10','no_flow5','no_memory5','multiscale5']:
        delta=paired[policy]-paired.frequency5; lo,hi=bootstrap_mean(delta)
        contrasts.append(dict(policy=policy,baseline='frequency5',mean_logloss_delta=delta.mean(),descriptive_day_lo=lo,descriptive_day_hi=hi))
    dump(out/'score_contrasts.json',contrasts)
    selected=pd.read_parquet(out/'decisions.parquet')
    selected=selected[selected.target.eq('usd2_300')&selected.accepted&selected.known].merge(frame[['candidate_id','momentum_side']],on='candidate_id',validate='many_to_one')
    selected['always_long']=-selected.event_sign; selected['always_short']=selected.event_sign
    selected['anti_impulse']=selected.momentum_side*selected.event_sign
    matched=[]
    for policy,g in selected.groupby('policy'):
        for block in ['all','additional','original']:
            z=g if block=='all' else g[g.source_population.eq(block)]
            means=z.groupby('date')[['reward','always_long','always_short','anti_impulse']].mean().mean()
            matched.append(dict(policy=policy,block=block,n=len(z),**means.to_dict()))
    pd.DataFrame(matched).to_parquet(out/'matched_direction_controls.parquet',index=False)
    # This is a description of realised regimes, not a regime label supplied to a model.
    valid=frame[frame.date.ge(pred.date.min())&frame.impulse_known&frame.known].copy()
    valid['week']=pd.to_datetime(valid.date).dt.to_period('W-SUN').astype(str)
    valid['continuation_reward']=valid.momentum_side*valid.event_sign
    markets=[]
    for (week,cell),g in valid.groupby(['week','cell']):
        markets.append(dict(week=week,cell=cell,n=len(g),continuations=int(g.continuation_reward.eq(1).sum()),reversals=int(g.continuation_reward.eq(-1).sum()),
            timeouts=int(g.event_sign.eq(0).sum()),mean_continuation_reward=g.continuation_reward.mean(),mean_risk=g.risk_per_oz.mean(),
            mean_flow=g.f_flow.mean(),mean_activity_log=g.f_trade_activity.mean()))
    pd.DataFrame(markets).to_parquet(out/'market_cells.parquet',index=False)
    dump(out/'diagnostics_receipt.json',dict(parent_manifest_sha256=digest(out/'manifest.json'),implementation_sha256=digest(__file__),
        description='Post-result descriptive benchmark. Frequency-only baseline uses the SAME past-only day-balanced class prior as each saved model; no features or future tuning.',
        artifacts={name:digest(out/name) for name in ['frequency_calibration.parquet','market_cells.parquet','score_components.parquet','score_contrasts.json','matched_direction_controls.parquet']}))
    print(cal[cal.block.eq('all')][['policy','day_logloss','day_brier']].round(5).to_string(index=False))
    print(components.groupby('policy').mean(numeric_only=True).round(5).to_string())


if __name__=='__main__': main()
