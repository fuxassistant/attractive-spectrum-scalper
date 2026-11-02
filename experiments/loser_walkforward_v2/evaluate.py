"""Evaluate bounded event quality before ever touching a paper account."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from experiments.loser_lab_v1.account import replay


def decisions(predictions,targets):
    columns=['candidate_id','target','known','event_sign','class_index','release_ns','threshold_per_oz','bid_delta_per_oz','seconds']
    d=predictions.merge(targets[columns],on=['candidate_id','target'],validate='many_to_one')
    d['reward']=np.where(d.known,-d.side*d.event_sign,np.nan)
    d['unit_bid_pnl']=np.where(d.known,d.side*d.bid_delta_per_oz,np.nan)
    d['opportunity_reward']=np.where(d.accepted,d.reward,0.)
    d['loss_first']=d.reward.eq(1); d['gain_first']=d.reward.eq(-1)
    d['clean_loss']=d.loss_first & d.class_index.isin([1,3])
    return d


def daily_quality(d):
    rows=[]
    for (target,policy,date),g in d.groupby(['target','policy','date'],sort=True):
        a=g[g.accepted&g.known]; known=g[g.known]
        n=len(a); decisive=a.event_sign.ne(0).sum()
        rows.append(dict(target=target,policy=policy,date=date,symbol=g.symbol.iloc[0],source_population=g.source_population.iloc[0],
            origins=len(g),selected=int(g.accepted.sum()),unknown_selected=int((g.accepted&~g.known).sum()),known_selected=n,
            losses=int(a.loss_first.sum()),gains=int(a.gain_first.sum()),timeouts=int(a.event_sign.eq(0).sum()),clean_losses=int(a.clean_loss.sum()),
            coverage=n/len(g),day_reward=float(a.reward.mean()) if n else np.nan,
            opportunity_reward=float(known.opportunity_reward.mean()) if len(known) else np.nan,
            mean_bid_per_oz=float(a.unit_bid_pnl.mean()) if n else np.nan,
            decisive_accuracy=float(a.loss_first.sum()/decisive) if decisive else np.nan,
            long_fraction=float(a.side.eq(1).mean()) if n else np.nan))
    return pd.DataFrame(rows)


def block_mask(frame,block):
    if block=='all': return np.ones(len(frame),bool)
    return frame.source_population.eq('additional' if block=='added_forward' else 'original')


def bootstrap_mean(values,seeds=999,seed=271828):
    x=np.asarray(values,float); x=x[np.isfinite(x)]
    if len(x)<2: return np.nan,np.nan
    rng=np.random.default_rng(seed); means=x[rng.integers(len(x),size=(seeds,len(x)))].mean(axis=1)
    return tuple(np.quantile(means,[.025,.975]))


def summarize(d,daily,config):
    rows=[]; null_rows=[]
    for target in sorted(d.target.unique()):
        source=d[d.target.eq(target)]; control=source[source.policy.eq('anti_impulse')&source.accepted&source.known]
        for policy in sorted(source.policy.unique()):
            for block in ('added_forward','original','all'):
                g=source[source.policy.eq(policy)&block_mask(source,block)]; a=g[g.accepted&g.known]
                days=daily[daily.target.eq(target)&daily.policy.eq(policy)&block_mask(daily,block)]
                decisive=int(a.event_sign.ne(0).sum()); lo,hi=bootstrap_mean(days.day_reward,config['bootstrap_seeds'])
                opportunity_lo,opportunity_hi=bootstrap_mean(days.opportunity_reward,config['bootstrap_seeds'])
                seed=int.from_bytes(hashlib.sha256((target+policy+block).encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed); random_means=[]; matched_means=[]
                for date,active in a.groupby('date'):
                    signs=active.event_sign.to_numpy(); k=len(active); pool=control[control.date.eq(date)].reward.to_numpy()
                    if k>len(pool): raise AssertionError('Selected outside common impulse cohort')
                    random_means.append((rng.choice([-1,1],size=(config['random_seeds'],k))*signs).mean(axis=1))
                    # Same day and number of entries as the policy, all from the
                    # identical causal impulse opportunity set; random timing.
                    idx=np.argsort(rng.random((config['random_seeds'],len(pool))),axis=1)[:,:k]
                    matched_means.append(pool[idx].mean(axis=1))
                random=np.mean(random_means,axis=0) if random_means else np.zeros(config['random_seeds'])
                matched=np.mean(matched_means,axis=0) if matched_means else np.zeros(config['random_seeds'])
                for j in range(config['random_seeds']): null_rows.append(dict(target=target,policy=policy,block=block,seed=j,random_side_day_reward=random[j],matched_anti_day_reward=matched[j]))
                rows.append(dict(target=target,policy=policy,block=block,days=len(days),active_days=int(days.selected.gt(0).sum()),
                    origins=len(g),selected=len(a),unknown_selected=int((g.accepted&~g.known).sum()),coverage=len(a)/len(g) if len(g) else 0,
                    losses=int(a.loss_first.sum()),gains=int(a.gain_first.sum()),timeouts=int(a.event_sign.eq(0).sum()),clean_losses=int(a.clean_loss.sum()),
                    decisive_accuracy=float(a.loss_first.sum()/decisive) if decisive else None,
                    day_reward=float(days.day_reward.mean()),day_reward_lo=lo,day_reward_hi=hi,
                    opportunity_reward=float(days.opportunity_reward.mean()),opportunity_lo=opportunity_lo,opportunity_hi=opportunity_hi,
                    positive_days=int(days.day_reward.gt(0).sum()),negative_days=int(days.day_reward.lt(0).sum()),
                    mean_bid_per_oz=float(a.unit_bid_pnl.mean()),mean_threshold=float(a.threshold_per_oz.mean()),
                    long_fraction=float(a.side.eq(1).mean()),random_side_p95=float(np.quantile(random,.95)),
                    matched_anti_median=float(np.median(matched)),matched_anti_p95=float(np.quantile(matched,.95))))
    return pd.DataFrame(rows),pd.DataFrame(null_rows)


def calibration(d):
    records=[]; curves=[]
    for (target,policy),g in d.groupby(['target','policy']):
        if 'p0' not in g or g.p0.isna().all(): continue
        for block in ('added_forward','original','all'):
            h=g[block_mask(g,block)&g.known].copy()
            if h.empty: continue
            p=h[['p'+str(i) for i in range(5)]].to_numpy(float); y=h.class_index.to_numpy(int)
            h['logloss']=-np.log(np.maximum(1e-12,p[np.arange(len(y)),y])); h['brier']=((p-np.eye(5)[y])**2).sum(axis=1)
            day=h.groupby('date')[['logloss','brier']].mean()
            records.append(dict(target=target,policy=policy,block=block,rows=len(h),day_logloss=day.logloss.mean(),day_brier=day.brier.mean()))
            # Calibration of the selected side, even on WAIT observations.
            predicted=np.where(h.side.to_numpy()==1,p[:,3]+p[:,4],p[:,1]+p[:,2])
            h['predicted']=predicted; h['bin']=np.minimum(9,(predicted*10).astype(int))
            for b,q in h.groupby('bin'):
                curves.append(dict(target=target,policy=policy,block=block,bin=int(b),n=len(q),predicted=q.predicted.mean(),observed=q.loss_first.mean()))
    return pd.DataFrame(records),pd.DataFrame(curves)


def cost_metrics(d,post):
    rows=[]
    for (target,policy),g in d.groupby(['target','policy']):
        a=g[g.accepted&g.known]
        for reverse in (False,True):
            chosen=a[['candidate_id','target','side','source_population']].copy()
            if reverse: chosen['side']=-chosen.side
            z=chosen.merge(post,on=['candidate_id','target','side'],validate='one_to_many')
            for (mode,population),q in z.groupby(['mode','source_population']):
                rows.append(dict(target=target,policy=policy,reverse=reverse,mode=mode,source_population=population,
                    n=len(q),known=int(q.known.sum()),mean_net_per_oz=q.loc[q.known,'net_per_oz'].mean()))
    return pd.DataFrame(rows)


def fast_terminal(unit,threshold,style,config):
    """Random-side sensitivity on the same nonoverlapping bid opportunities."""
    cfg=config['risk']; cash=cfg['initial_deposit_usd']; multiplier=1; depleted=False; trades=0
    for u,x in zip(unit,threshold):
        if cash<x*cfg['contract_oz']*multiplier: continue
        pnl=u*cfg['contract_oz']*multiplier; cash+=pnl; trades+=1
        depleted |= cash <= cfg['initial_deposit_usd']*(1-cfg['depletion_fraction'])
        if style=='after_loss': multiplier=min(cfg['max_multiplier'],multiplier*2) if pnl<0 else 1
        elif style=='after_gain': multiplier=min(cfg['max_multiplier'],multiplier*2) if pnl>0 else 1
        if cash<=0: break
    return dict(terminal_usd=cash,trades=trades,depleted80=depleted,bankrupt=cash<=0)


def accounts(pred,targets,post,config,policies=None):
    if policies is None: policies=['anti_impulse','blend2_fade','fade5_g10','switch5_g10','guard_switch5_g10','weekly_selector']
    target=config['primary_target']; y=targets[targets.target.eq(target)]; z=post[post.target.eq(target)]; summaries=[]; logs=[]; null=[]
    for policy in policies:
        p=pred[pred.policy.eq(policy)&pred.target.eq(target)]
        for style in config['risk']['styles']:
            summary,log=replay(p,y,z,config,style=style)
            summaries.append(dict(target=target,policy=policy,style=style,mode='bid',reverse=False,**summary))
            if len(log):
                log['policy']=policy; log['style']=style; log['target']=target; log['mode']='bid'; log['reverse']=False; logs.append(log)
            # Future thresholds originate in causal features; exits are fixed and
            # symmetric, hence random side cannot alter the event holding clock.
            eligible=p[p.accepted].merge(y[['candidate_id','known','bid_delta_per_oz','threshold_per_oz','exit_ns']],on='candidate_id',validate='one_to_one').sort_values('t_ns')
            if not eligible.known.all(): continue
            last=-1; take=[]
            for r in eligible.itertuples():
                if r.t_ns>last: take.append(r.Index); last=r.exit_ns
            h=eligible.loc[take]; rng=np.random.default_rng(314159)
            for seed in range(config['random_seeds']):
                unit=rng.choice([-1,1],size=len(h))*h.bid_delta_per_oz.to_numpy()
                result=fast_terminal(unit,h.threshold_per_oz.to_numpy(),style,config)
                null.append(dict(policy=policy,style=style,seed=seed,**result))
        for mode in ('bid','base','stress'):
            summary,log=replay(p,y,z,config,style='fixed',mode=mode,reverse=True)
            summaries.append(dict(target=target,policy=policy,style='fixed',mode=mode,reverse=True,**summary))
            if len(log):
                log['policy']=policy; log['style']='fixed'; log['target']=target; log['mode']=mode; log['reverse']=True; logs.append(log)
    return pd.DataFrame(summaries),pd.concat(logs,ignore_index=True),pd.DataFrame(null)
