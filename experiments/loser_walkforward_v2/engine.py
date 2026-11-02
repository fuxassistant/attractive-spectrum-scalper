"""Past-only event models, interpretable loss mechanisms, and revocable licences.

Every model is fitted before a day. Weekly selection sees earlier OUT-OF-SAMPLE
predictions, not refitted historical predictions. Neither dollars nor equity
appear in any learning, gate, or ranking expression.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from experiments.loser_lab_v1.models import EventModel
from experiments.loser_lab_v1.continuation import orient


def joined(features, targets):
    overlap=[c for c in targets if c in features and c not in ('candidate_id','t_ns','date')]
    return features.drop(columns=overlap).merge(targets,on=['candidate_id','t_ns','date'],validate='one_to_one')


def contexts(frame, config=None):
    f, cols=orient(frame)
    if config is not None:
        f['impulse_known']=f.past_known & f.past_net30_r.abs().ge(config['minimum_impulse_r'])
    relative=f.momentum_side.to_numpy()*f.side.to_numpy()
    for name in ('trend_atr','bb_z'):
        f['f_slow_'+name]=f['x_base_'+name]*relative
    for name in ('efficiency20','log_atr_ratio'):
        f['f_slow_'+name]=f['x_base_'+name]
    slow=[c for c in f if c.startswith('f_slow_')]
    dates=f[['date','symbol']].drop_duplicates().sort_values('date')
    segment=0; previous=None; segments={}
    for r in dates.itertuples():
        if previous and (r.symbol!=previous.symbol or (pd.Timestamp(r.date)-pd.Timestamp(previous.date)).days>(config['guard']['reset_after_calendar_days'] if config else 4)): segment+=1
        segments[r.date]=segment; previous=r
    f['segment']=f.date.map(segments)
    flow=np.where(f.f_mbo_known.eq(1)&np.isfinite(f.f_flow),
                  np.where(f.f_flow>.2,'aligned',np.where(f.f_flow<-.2,'opposed','neutral')),'unknown')
    f['cell']=np.where(f.f_efficiency>=.35,'efficient','choppy')+'_'+pd.Series(flow,index=f.index)
    return f,dict(full=cols,
        no_flow=[c for c in cols if c not in ('f_flow','f_depth','f_pressure','f_trade_activity','f_unknown_trade','f_mbo_known')],
        no_memory=[c for c in cols if c not in ('f_memory_dwell','f_memory_pass','f_zone_old','f_zone_new','f_zone_alignment','f_distance_r')],
        coarse=['f_risk','f_net30','f_efficiency','f_range120','f_net120','f_migration'],
        multiscale=cols+slow)


def score_probability(prob, mode):
    up=prob[:,1]+prob[:,2]; down=prob[:,3]+prob[:,4]
    score=up-down
    if mode=='clean': score=prob[:,1]-down
    sign=np.where(score>=0,1,-1) if mode=='switch' else np.ones(len(score),int)
    return (np.abs(score) if mode=='switch' else score), sign


def generate(frame, config):
    f, sets=contexts(frame,config); dates=sorted(f.date.unique()); warm=dates[:config['initial_days']]
    frozen=EventModel(sets['full'],config['model']).fit(f[f.date.isin(warm)&f.known])
    receipts={'frozen':frozen.receipt()}; predictions=[]
    for date in dates[config['initial_days']:]:
        cutoff=int(pd.Timestamp(date,tz='UTC').value); test=f[f.date.eq(date)]; index=dates.index(date)
        fits={}; probs={}
        for window in config['windows']:
            train=f[f.date.isin(dates[index-window:index])&f.known&f.release_ns.lt(cutoff)]
            model=EventModel(sets['full'],config['model']).fit(train)
            key=f'full{window}'; fits[key]=model; probs[key]=model.predict(test)
            receipts[date+'_'+key]={**model.receipt(),'train_dates':sorted(train.date.unique())}
        if frame.target.iloc[0]==config['primary_target']:
            for arm in ('no_flow','no_memory','coarse','multiscale'):
                train=f[f.date.isin(dates[index-5:index])&f.known&f.release_ns.lt(cutoff)]
                model=EventModel(sets[arm],config['model']).fit(train)
                fits[arm]=model; probs[arm]=model.predict(test)
                receipts[date+'_'+arm]={**model.receipt(),'train_dates':sorted(train.date.unique())}
        probs['frozen']=frozen.predict(test); fits['frozen']=frozen
        probs['blend2']=.5*(probs['frozen']+probs['full2']); fits['blend2']=fits['full2']
        policies=[]
        for w in config['windows']:
            for gate in config['gates']:
                for mode in ('fade','switch'):
                    policies.append((f'{mode}{w}_g{round(gate*100):02d}',f'full{w}',mode,gate))
        policies += [('frozen_fade','frozen','fade',.1),('blend2_fade','blend2','fade',.1),('clean5','full5','clean',.1)]
        for arm in ('no_flow','no_memory','coarse','multiscale'):
            if arm in probs: policies.append((arm+'5',''+arm,'fade',.1))
        for name,key,mode,gate in policies:
            prob=probs[key]; score, sign=score_probability(prob,mode)
            r=test[['candidate_id','t_ns','date','target','symbol','source_population','segment','cell']].copy()
            r['policy']=name; r['side']=-test.momentum_side.to_numpy()*sign; r['score']=score
            r['accepted']=test.impulse_known.to_numpy()&(score>=gate)
            r['selection_ns']=cutoff; r['max_feedback_ns']=max(fits[key].max_release,frozen.max_release if key=='blend2' else -1)
            r['reason']=np.where(~test.impulse_known,'SMALL_OR_UNKNOWN_IMPULSE',np.where(r.accepted,'ACCEPT','LOW_PREDICTED_SCORE'))
            physical=prob.copy(); neg=test.momentum_side.to_numpy()==-1; physical[neg]=prob[neg][:,[0,3,4,1,2]]
            for j in range(5): r['p'+str(j)]=physical[:,j]
            if r.max_feedback_ns.max()>=cutoff: raise AssertionError('Future training feedback')
            predictions.append(r)
        controls={'anti_impulse':-test.momentum_side.to_numpy(),'follow_impulse':test.momentum_side.to_numpy(),
                  'always_long':np.ones(len(test),int),'always_short':-np.ones(len(test),int),
                  'toward_memory':test.side.to_numpy()}
        for name,side in controls.items():
            r=test[['candidate_id','t_ns','date','target','symbol','source_population','segment','cell']].copy()
            r['policy']=name; r['side']=side; r['score']=np.nan; r['accepted']=test.impulse_known.to_numpy()
            r['selection_ns']=cutoff; r['max_feedback_ns']=-1
            r['reason']=np.where(r.accepted,'CONTROL','SMALL_OR_UNKNOWN_IMPULSE'); predictions.append(r)
    return pd.concat(predictions,ignore_index=True),receipts


def day_bound(history, prior=0, penalty=1):
    """Descriptive stability penalty, NOT an IID confidence guarantee."""
    daily=history.groupby('date').reward.mean()
    mean=float(daily.mean()) if len(daily) else 0.
    se=float(daily.std(ddof=1)/np.sqrt(len(daily))) if len(daily)>1 else 1.
    shrink=len(history)/(len(history)+prior) if len(history)+prior else 0.
    return mean*shrink-penalty*se, mean, se


def guarded(predictions, targets, config):
    labels=targets[['candidate_id','known','event_sign','release_ns']]
    result=[]; receipts=[]; dates=sorted(predictions.date.unique()); gc=config['guard']
    for base in ('fade5_g10','switch5_g10'):
        p=predictions[predictions.policy.eq(base)].merge(labels,on='candidate_id',validate='one_to_one')
        p['reward']=-p.side*p.event_sign
        for date in dates:
            cutoff=int(pd.Timestamp(date,tz='UTC').value); ix=dates.index(date); prior=dates[max(0,ix-gc['history_days']):ix]
            test=p[p.date.eq(date)].copy(); segment=int(test.segment.iloc[0])
            hist=p[p.date.isin(prior)&p.accepted&p.known&p.release_ns.lt(cutoff)&p.segment.eq(segment)]
            for cell, group in test.groupby('cell'):
                h=hist[hist.cell.eq(cell)]; lower,mean,se=day_bound(h,gc['prior_rows'],gc['day_se_penalty'])
                support=len(h)>=gc['minimum_rows'] and h.date.nunique()>=gc['minimum_days']
                licence=bool(support and lower>0)
                row=dict(date=date,target=test.target.iloc[0],policy='guard_'+base,cell=cell,n=len(h),days=h.date.nunique(),
                         lower=lower,mean=mean,se=se,licence=licence,cutoff_ns=cutoff,
                         max_release_ns=int(h.release_ns.max()) if len(h) else -1,segment=segment)
                receipts.append(row); ids=group.index
                test.loc[ids,'accepted']=group.accepted & licence
                test.loc[ids,'reason']=np.where(~group.accepted,group.reason,'LICENCE' if licence else 'NO_CELL_SUPPORT' if not support else 'UNSTABLE_RECENT_CELL')
                test.loc[ids,'licence_lower']=lower
                test.loc[ids,'max_feedback_ns']=np.maximum(group.max_feedback_ns,row['max_release_ns'])
            test['policy']='guard_'+base
            result.append(test.drop(columns=['known','event_sign','release_ns','reward']))
    return pd.concat(result,ignore_index=True),receipts


def weekly_selector(predictions, targets, config):
    pool=[f'{mode}{w}_g{round(gate*100):02d}' for w in config['windows'] for gate in config['gates'] for mode in ('fade','switch')]
    p=predictions[predictions.policy.isin(pool)].merge(targets[['candidate_id','known','event_sign','release_ns']],on='candidate_id',validate='many_to_one')
    p['reward']=np.where(p.accepted,-p.side*p.event_sign,0.)
    p['week']=pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)
    dates=sorted(p.date.unique()); records=[]; receipts=[]; sc=config['selector']
    for week,g in p.groupby('week',sort=True):
        date=g.date.min(); cutoff=int(pd.Timestamp(date,tz='UTC').value); ix=dates.index(date); previous=dates[max(0,ix-sc['history_days']):ix]
        h=p[p.date.isin(previous)&p.known&p.release_ns.lt(cutoff)]; ranks=[]
        for policy in pool:
            q=h[h.policy.eq(policy)]; active=q[q.accepted]
            lower,mean,se=day_bound(q,0,sc['day_se_penalty'])
            supported=len(previous)==sc['history_days'] and len(active)>=sc['minimum_rows'] and active.date.nunique()>=sc['minimum_active_days']
            row=dict(week=week,target=p.target.iloc[0],policy=policy,n=len(active),active_days=active.date.nunique(),history_days=len(previous),
                score=lower,mean_opportunity_reward=mean,day_se=se,supported=supported,cutoff_ns=cutoff,
                max_feedback_ns=int(q.release_ns.max()) if len(q) else -1)
            ranks.append(row)
        allowed=sorted([r for r in ranks if r['supported'] and r['score']>0],key=lambda r:(-r['score'],r['policy']))
        choice=allowed[0]['policy'] if allowed else 'WAIT'
        for r in ranks: r['chosen']=r['policy']==choice; r['selection']=choice
        receipts.extend(ranks)
        z=g[g.policy.eq(choice if choice!='WAIT' else pool[0])].copy()
        z['selected_recipe']=choice; z['policy']='weekly_selector'
        if choice=='WAIT': z['accepted']=False; z['reason']='NO_STABLE_RECIPE'
        z['recipe_selection_ns']=cutoff
        z['recipe_max_feedback_ns']=max(r['max_feedback_ns'] for r in ranks)
        records.append(z.drop(columns=['known','event_sign','release_ns','reward','week']))
    return pd.concat(records,ignore_index=True),receipts
