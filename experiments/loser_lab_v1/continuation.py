"""Second exploratory hypothesis: learn WHEN an impulse persists, fade it.

Added after observing that a same-origin anti-momentum control matches the
generic directional model. No claim of untouched confirmation for this stage.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import dump,digest,checked
from .models import EventModel
from .run import quality,deposits

ROOT=Path(__file__).resolve().parents[2]


def orient(frame):
    f=frame.copy();physical=f.side.to_numpy()*f.past_net30_r.to_numpy()
    momentum=np.where(physical>=0,1,-1);relative=momentum*f.side.to_numpy()
    f['momentum_side']=momentum;f['impulse_known']=f.past_known&f.past_net30_r.abs().ge(.25)
    fields={'distance_r':f.distance_r,'risk':f.risk_per_oz,'spread_r':f.spread_per_oz/f.risk_per_oz,
            'net30':f.past_net30_r*relative,'net120':f.past_net120_r*relative,'migration':f.past_migration_r*relative,
            'efficiency':f.past_efficiency30,'range120':f.past_range120_r,'zone_alignment':relative,
            'zone_old':f.past_zone_share_old,'zone_new':f.past_zone_share_new,
            'memory_dwell':f.x_mem_log_median_dwell_seconds,'memory_pass':f.x_mem_recent_pass_fraction,
            'trade_activity':f.x_base_mbo_now_trade_total_log,'unknown_trade':f.x_mbo_mbo_now_trade_unknown_fraction,
            'mbo_known':f.x_base_mbo_now_known,'flow':f.x_mbo_mbo_now_trade_imbalance*relative,
            'depth':f.x_mbo_mbo_now_depth_1_toward_imbalance*relative,'pressure':f.x_mbo_mbo_now_book_net_pressure*relative}
    for key,val in fields.items():f['f_'+key]=val
    # UP now means continuation of the CURRENT known impulse, not physical up.
    mapping=np.array([0,3,4,1,2]);classes=f.class_index.to_numpy(int).copy();valid=f.known.to_numpy(bool)&(momentum==-1)
    classes[valid]=mapping[classes[valid]];f['class_index']=classes
    return f,list('f_'+k for k in fields)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    source=ROOT/a.source;out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
    config=json.loads((source/'config.json').read_text(encoding='utf-8'));config['primary_policy']='continuation_blend'
    config['stage']='SECOND_EXPLORATORY_STAGE_AFTER_FIRST_RESULTS';config['hypothesis']='Predict continuation in impulse-oriented coordinates; short the impulse only when predicted continuation-minus-reversal>=0.1, |30s net|>=0.25R. Controls use same amplitude gate.'
    dump(out/'config.json',config)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name in ('targets.parquet','post_cost.parquet'):checked(source/name,manifest['artifacts'][name])
    y=pd.read_parquet(source/'targets.parquet');y=y[y.target==config['primary_target']]
    post=pd.read_parquet(source/'post_cost.parquet');post=post[post.target==config['primary_target']]
    f=pd.read_parquet(ROOT/config['atlas']/'observations.parquet')
    overlap=[c for c in y if c in f and c not in ('candidate_id','t_ns','date')]
    joined=f.drop(columns=overlap).merge(y,on=['candidate_id','t_ns','date'],validate='one_to_one')
    oriented,columns=orient(joined);train=oriented[oriented.date.le(config['train_end'])&oriented.known]
    frozen=EventModel(columns,config['model']).fit(train);models={'frozen':frozen.receipt()};pred=[];dates=sorted(oriented.date.unique())
    for date in dates:
        if not config['deployment_start']<=date<=config['extension_end']:continue
        cutoff=int(pd.Timestamp(date,tz='UTC').value);before=dates[dates.index(date)-2:dates.index(date)]
        pair=oriented[oriented.date.isin(before)&oriented.known&oriented.release_ns.lt(cutoff)]
        model=EventModel(columns,config['model']).fit(pair);models[date]=model.receipt();test=oriented[oriented.date.eq(date)]
        pf=frozen.predict(test);pp=model.predict(test);w=config['pair_blend_weight']
        for policy,prob in [('continuation_frozen',pf),('continuation_blend',(1-w)*pf+w*pp),('anti_impulse_control',None)]:
            r=test[['candidate_id','t_ns','date','target']].copy();r['policy']=policy;r['side']=-test.momentum_side.to_numpy()
            r['score']=(prob[:,1]+prob[:,2]-prob[:,3]-prob[:,4]) if prob is not None else np.nan
            r['accepted']=test.impulse_known.to_numpy() & ((r.score>=config['confidence_gate']) if prob is not None else True)
            r['selection_ns']=cutoff;r['max_feedback_ns']=model.max_release if policy=='continuation_blend' else frozen.max_release if policy=='continuation_frozen' else -1
            if prob is not None:
                # Persist PHYSICAL direction probabilities for common scoring.
                physical=prob.copy();negative=test.momentum_side.to_numpy()==-1
                physical[negative]=prob[negative][:,[0,3,4,1,2]]
                for j in range(5):r['p'+str(j)]=physical[:,j]
            pred.append(r)
    pred=pd.concat(pred,ignore_index=True);metrics,daily,nulls,decisions=quality(pred,y,config)
    # New policy names: reuse primary's identical risk/null machinery via alias,
    # then restore policy identity. Selection and sizing are still separate.
    alias=pred.copy();alias.loc[alias.policy.eq('continuation_blend'),'policy']='blend_context'
    accounts,trades,random=deposits(alias,y,post,config)
    for table in (accounts,trades,random):
        if 'policy' in table:table.loc[table.policy.eq('blend_context'),'policy']='continuation_blend'
    for name,table in dict(targets=y,post_cost=post,predictions=pred,quality=metrics,daily_quality=daily,random_quality=nulls,
                           decisions=decisions,accounts=accounts,trades=trades,random_accounts=random).items():table.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'models.json',models)
    dump(out/'summary.json',dict(status=config['stage'],hypothesis=config['hypothesis'],primary=metrics[metrics.policy=='continuation_blend'].to_dict('records'),equity_used_for_selection=False))
    dump(out/'manifest.json',dict(source_manifest_sha256=digest(source/'manifest.json'),implementation_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__)},
          artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(metrics[['block','policy','selected','decisive_accuracy','day_reward','random_p95','mean_bid_per_oz']].round(4).to_string(index=False))


if __name__=='__main__':main()
