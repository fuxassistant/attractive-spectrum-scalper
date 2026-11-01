"""Frozen experiment catalog: event quality first, paper deposit second."""
from __future__ import annotations
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import checked,digest,dump
from experiments.two_day_strategy_v1.engine import stable_uniform
from .targets import make_targets,post_cost_pairs
from .models import rolling_predictions
from .account import replay

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def quality(pred,targets,config):
    joined=pred.merge(targets,on=['candidate_id','target','date','t_ns'],validate='many_to_one')
    joined['block']=np.where(joined.date.le(config['primary_end']),'primary','extension')
    joined['reward']=-joined.side*joined.event_sign
    joined['loss_hit']=joined.reward.eq(1);joined['gain_hit']=joined.reward.eq(-1)
    joined['clean_loss']=joined.loss_hit&joined.class_index.isin([1,3])
    joined['clean_reward']=joined.clean_loss.astype(int)-joined.gain_hit.astype(int)
    joined['bid_unit_pnl']=joined.side*joined.bid_delta_per_oz
    joined['favorable_before_exit']=np.where(joined.side.eq(1),joined.bid_high_per_oz,-joined.bid_low_per_oz)
    rows=[];daily=[];nulls=[]
    for (target,block,policy),g in joined.groupby(['target','block','policy'],sort=True):
        selected=g[g.accepted];known=selected[selected.known];decisive=known[known.event_sign.ne(0)]
        d=known.groupby('date')[['reward','clean_reward','loss_hit','gain_hit','clean_loss','bid_unit_pnl']].mean()
        row=dict(target=target,block=block,policy=policy,opportunities=len(g),selected=len(selected),known=len(known),unknown=int((~selected.known).sum()),
                 coverage=len(selected)/len(g),active_days=len(d),losses=int(known.loss_hit.sum()),gains=int(known.gain_hit.sum()),timeouts=int(known.event_sign.eq(0).sum()),
                 decisive_accuracy=float(decisive.loss_hit.mean()) if len(decisive) else None,
                 mean_reward=float(known.reward.mean()) if len(known) else None,
                 day_reward=float(d.reward.mean()) if len(d) else None,
                 clean_rate=float(known.clean_loss.mean()) if len(known) else None,
                 day_clean_reward=float(d.clean_reward.mean()) if len(d) else None,
                 mean_bid_per_oz=float(known.bid_unit_pnl.mean()) if len(known) else None,
                 median_seconds=float(known.seconds.median()) if len(known) else None,
                 mean_favorable_x=float((known.favorable_before_exit/known.threshold_per_oz).mean()) if len(known) else None,
                 worst_day_reward=float(d.reward.min()) if len(d) else None)
        if 'p0' in g and g.p0.notna().all():
            k=g[g.known];proba=k[['p'+str(i) for i in range(5)]].to_numpy(float)
            loss=-np.log(np.clip(proba[np.arange(len(k)),k.class_index.to_numpy(int)],1e-12,1))
            row['all_day_logloss']=float(pd.Series(loss,index=k.index).groupby(k.date).mean().mean())
        for date,v in d.iterrows():daily.append(dict(target=target,block=block,policy=policy,date=date,n=int((known.date==date).sum()),**v.to_dict()))
        scores=[]
        for seed in range(1,config['random_seeds']+1):
            signs=np.where(stable_uniform(known.candidate_id,seed)>.5,1,-1)
            reward=-signs*known.event_sign.to_numpy()
            score=float(pd.Series(reward,index=known.index).groupby(known.date).mean().mean()) if len(known) else 0.
            scores.append(score);nulls.append(dict(target=target,block=block,policy=policy,seed=seed,day_reward=score))
        row.update(random_p05=float(np.quantile(scores,.05)),random_median=float(np.median(scores)),random_p95=float(np.quantile(scores,.95)))
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(daily),pd.DataFrame(nulls),joined


def deposits(pred,targets,post,config):
    rows=[];logs=[];controls=[]
    for target in targets.target.unique():
        ts=targets[targets.target.eq(target)];pc=post[post.target.eq(target)]
        for policy,dec in pred[pred.target.eq(target)].groupby('policy',sort=True):
            for style in config['risk']['styles']:
                for mode in ('bid','base','stress'):
                    stats,log=replay(dec,ts,pc,config,mode=mode,style=style)
                    meta=dict(target=target,policy=policy,style=style,mode=mode,reverse=False)
                    rows.append({**meta,**stats})
                    if len(log):
                        for k,v in meta.items():log[k]=v
                        logs.append(log)
            # Opposite actions, fixed size and fixed event clock: cost assessment only.
            for mode in ('bid','base','stress'):
                stats,log=replay(dec,ts,pc,config,mode=mode,style='fixed',reverse=True)
                meta=dict(target=target,policy=policy,style='fixed',mode=mode,reverse=True);rows.append({**meta,**stats})
                if len(log):
                    for k,v in meta.items():log[k]=v
                    logs.append(log)
            if target==config['primary_target'] and policy in ('blend_context','clean_blend','pair_context','expert_pair','memory_return'):
                for seed in range(1,config['random_seeds']+1):
                    side=dict(zip(dec.candidate_id,np.where(stable_uniform(dec.candidate_id,seed)>.5,1,-1)))
                    for style in config['risk']['styles']:
                        stats,_=replay(dec,ts,pc,config,mode='bid',style=style,side_override=side)
                        controls.append(dict(target=target,policy=policy,style=style,seed=seed,**stats))
        print(json.dumps(dict(stage='deposit_replay',target=target)),flush=True)
    return pd.DataFrame(rows),pd.concat(logs,ignore_index=True),pd.DataFrame(controls)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();config=json.loads((HERE/'config.json').read_text(encoding='utf-8'));dump(out/'config.json',config)
    code={str(p.relative_to(ROOT)):digest(p) for p in (HERE/'config.json',HERE/'targets.py',HERE/'models.py',HERE/'account.py',HERE/'run.py')}
    atlas=ROOT/config['atlas'];am=json.loads((atlas/'manifest.json').read_text(encoding='utf-8'))
    source={str(atlas/'manifest.json'):digest(atlas/'manifest.json')};p=atlas/'observations.parquet'
    source[str(p)]=checked(p,am['artifacts'][p.name]);f=pd.read_parquet(p)
    manifest_path=ROOT/config['market_manifest'];source[str(manifest_path)]=checked(manifest_path,am['source_hashes'][str(manifest_path)])
    receipts=json.loads(manifest_path.read_text(encoding='utf-8'))['receipts'];receipts={r['date']:r for r in receipts}
    expert_path=ROOT/'experiments/two_day_strategy_v1/config.json';expert_config=json.loads(expert_path.read_text(encoding='utf-8'));source[str(expert_path)]=digest(expert_path)
    for c in ('target_available_ns','episode_available_ns','memory_available_ns','bar_context_available_ns','past_max_available_ns','mbometa_now_max_source_available_ns'):
        if (f[c]>f.t_ns).any():raise AssertionError('Future X '+c)
    targets=[];post=[]
    for date,day in f.groupby('date',sort=True):
        market=Path(receipts[date]['source_market']);mp=market/'manifest.json';qp=market/'quotes.parquet'
        source[str(mp)]=checked(mp,am['source_hashes'][str(mp)]);source[str(qp)]=checked(qp,am['source_hashes'][str(qp)])
        mm=json.loads(mp.read_text(encoding='utf-8'));q=pd.read_parquet(qp);coverage=int(mm['input']['end_ns'])+1
        for spec in config['targets']:
            y=make_targets(q,day,spec,coverage=coverage,config=config);targets.append(y)
            post.append(post_cost_pairs(q,y,config,coverage=coverage))
        print(json.dumps(dict(stage='native_targets',date=date,origins=len(day))),flush=True)
    targets=pd.concat(targets,ignore_index=True);post=pd.concat(post,ignore_index=True)
    targets.to_parquet(out/'targets.parquet',index=False);post.to_parquet(out/'post_cost.parquet',index=False)
    predictions=[];models={};experts=[]
    for target,y in targets.groupby('target',sort=False):
        overlap=[c for c in y if c in f and c not in ('candidate_id','t_ns','date')]
        x=f.drop(columns=overlap).merge(y,on=['candidate_id','t_ns','date'],validate='one_to_one')
        pred,receipts,rank=rolling_predictions(x,config,expert_config);predictions.append(pred);models[target]=receipts;rank['target']=target;experts.append(rank)
        print(json.dumps(dict(stage='models',target=target,predictions=len(pred))),flush=True)
    pred=pd.concat(predictions,ignore_index=True);metrics,daily,nulls,joined=quality(pred,targets,config)
    accounts,trades,random_accounts=deposits(pred,targets,post,config)
    for name,table in dict(predictions=pred,quality=metrics,daily_quality=daily,random_quality=nulls,decisions=joined,
                           accounts=accounts,trades=trades,random_accounts=random_accounts,expert_ranks=pd.concat(experts,ignore_index=True)).items():
        table.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'models.json',models)
    summary=dict(status='COMPLETE_EXPLORATORY_LOSER_TARGETS',origins=len(f),target_rows=len(targets),
                 known_by_target=targets.groupby('target').known.agg(['sum','count']).reset_index().to_dict('records'),
                 primary_policy=config['primary_policy'],primary_target=config['primary_target'],
                 primary=metrics[(metrics.target==config['primary_target'])&(metrics.policy==config['primary_policy'])].to_dict('records'),
                 scope=config['scope'],equity_used_for_selection=False,posthoc_risk_only=True,
                 elapsed_seconds=time.perf_counter()-started,python=platform.python_version())
    dump(out/'summary.json',summary)
    dump(out/'manifest.json',dict(implementation_sha256=code,source_hashes=source,artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
