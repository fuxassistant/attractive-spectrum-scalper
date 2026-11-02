"""Run all declared comparisons without choosing a winner from final equity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.mechanical_lab.data import ROOT,digest
from .prepare import check,dump,HERE
from .engine import joined,generate,guarded,weekly_selector
from .evaluate import decisions,daily_quality,summarize,calibration,cost_metrics,accounts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    source=ROOT/args.input; out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'config.json').read_text()); dump(out/'config.json',config)
    manifest=json.loads((source/'manifest.json').read_text()); assert manifest['status']=='COMPLETE_RESEARCH_INPUT'
    for name,sha in manifest['artifacts'].items(): check(source/name,sha)
    identity=dict(config=digest(source/'config.json'),inputs=manifest['artifacts'],
                  code={name:digest(ROOT/name) for name in ('experiments/loser_walkforward_v2/engine.py','experiments/loser_lab_v1/models.py','experiments/loser_lab_v1/continuation.py')})
    frozen=out/'prediction_identity.json'
    if frozen.exists() and json.loads(frozen.read_text())!=identity: raise ValueError('Prediction cache identity changed: use a new output directory')
    dump(frozen,identity)
    f=pd.read_parquet(source/'features.parquet'); y=pd.read_parquet(source/'targets.parquet'); post=pd.read_parquet(source/'post_cost.parquet')
    predictions=[]; models={}; guards=[]; selections=[]
    for spec in config['targets']:
        target=spec['id']; target_y=y[y.target.eq(target)]; data=joined(f,target_y)
        cache=out/(target+'_predictions.parquet'); receipt=out/(target+'_models.json')
        cache_hash=out/(target+'_cache.json')
        if cache.exists() and receipt.exists():
            hashes=json.loads(cache_hash.read_text()); check(cache,hashes['predictions']); check(receipt,hashes['models'])
            p=pd.read_parquet(cache); model=json.loads(receipt.read_text())
        else:
            p,model=generate(data,config); p.to_parquet(cache,index=False); dump(receipt,model)
            dump(cache_hash,dict(predictions=digest(cache),models=digest(receipt)))
        guard,gr=guarded(p,target_y,config); selected,sr=weekly_selector(p,target_y,config)
        predictions.extend([p,guard,selected]); models[target]=model; guards.extend(gr); selections.extend(sr)
        print(json.dumps(dict(stage='predicted',target=target,rows=len(p),days=p.date.nunique())),flush=True)
    pred=pd.concat(predictions,ignore_index=True); d=decisions(pred,y); daily=daily_quality(d)
    quality,null=summarize(d,daily,config); cal,curves=calibration(d); costs=cost_metrics(d,post)
    acc,trades,risknull=accounts(pred,y,post,config)
    tables=dict(predictions=pred,decisions=d,daily_quality=daily,quality=quality,quality_null=null,calibration=cal,calibration_curves=curves,
                cost_metrics=costs,accounts=acc,trades=trades,random_accounts=risknull,guard_receipts=pd.DataFrame(guards),selection_receipts=pd.DataFrame(selections))
    for name,table in tables.items(): table.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'summary.json',dict(status='COMPLETE_EXPLORATORY',input_days=f.date.nunique(),added_days=f.loc[f.source_population.eq('additional'),'date'].nunique(),
        origins=len(f),deployment_days=pred.date.nunique(),deployment_start=pred.date.min(),deployment_end=pred.date.max(),
        targets=len(config['targets']),equity_used_for_selection=False,scope=config['protocol'],
        primary=quality[quality.target.eq(config['primary_target'])&quality.policy.isin(['anti_impulse','fade5_g10','switch5_g10','guard_switch5_g10','weekly_selector'])].to_dict('records')))
    dump(out/'manifest.json',dict(status='COMPLETE_EXPLORATORY',source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        input_artifacts=manifest['artifacts'],implementation={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*.py')},
        artifacts={p.name:digest(p) for p in out.glob('*.parquet')}))
    print(quality[quality.target.eq(config['primary_target'])&quality.block.eq('all')][['policy','selected','decisive_accuracy','day_reward','day_reward_lo','day_reward_hi','opportunity_reward','mean_bid_per_oz']].round(4).to_string(index=False),flush=True)


if __name__=='__main__': main()
