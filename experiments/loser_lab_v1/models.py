"""Predict price events; neither equity nor a risk multiplier enters learning."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from experiments.hypothesis_atlas_v1.engine import PATH_COLUMNS,MEMORY_COLUMNS,FAMILIES
from experiments.two_day_strategy_v1.engine import CATALOG,build_masks,stable_uniform


class EventModel:
    def __init__(self,columns,config):self.columns=list(columns);self.config=config

    def fit(self,f):
        if not len(f) or not f.known.all():raise ValueError('Known mature targets required')
        x=f[self.columns].to_numpy(float);missing=~np.isfinite(x)
        self.median=np.array([np.median(x[~missing[:,j],j]) if (~missing[:,j]).any() else 0 for j in range(x.shape[1])])
        z=np.c_[np.where(missing,self.median,x),missing]
        w=1/f.groupby('date').candidate_id.transform('count').to_numpy(float);w=w*len(w)/w.sum()
        self.mean=np.average(z,axis=0,weights=w);self.scale=np.sqrt(np.average((z-self.mean)**2,axis=0,weights=w));self.scale[self.scale<1e-12]=1
        y=f.class_index.to_numpy(int);self.n=len(f);self.prior=(np.bincount(y,weights=w,minlength=5)+1)/(w.sum()+5)
        self.est=None
        if len(np.unique(y))>1:
            self.est=LogisticRegression(**{k:v for k,v in self.config.items() if k!='seed'},random_state=self.config['seed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                self.est.fit(np.clip((z-self.mean)/self.scale,-8,8),y,sample_weight=w)
        self.max_release=int(f.release_ns.max());self.max_origin=int(f.t_ns.max())
        return self

    def predict(self,f):
        if self.est is None:return np.tile(self.prior,(len(f),1))
        x=f[self.columns].to_numpy(float);missing=~np.isfinite(x);z=np.c_[np.where(missing,self.median,x),missing]
        p=np.zeros((len(f),5));p[:,self.est.classes_]=self.est.predict_proba(np.clip((z-self.mean)/self.scale,-8,8))
        return self.n/(self.n+5)*p+5/(self.n+5)*self.prior

    def receipt(self):
        return dict(columns=self.columns,n=self.n,prior=self.prior,median=self.median,mean=self.mean,scale=self.scale,
                    max_release_ns=self.max_release,max_origin_ns=self.max_origin,
                    classes=self.est.classes_ if self.est is not None else [],
                    coef=self.est.coef_ if self.est is not None else [],intercept=self.est.intercept_ if self.est is not None else [])


def feature_sets(f):
    base=[c for c in f if c.startswith('x_base_')]
    mbo=['x_mbo_mbo_now_'+s for s in ('trade_imbalance','trade_unknown_fraction','depth_1_toward_imbalance','depth_12_toward_imbalance','book_net_pressure','replenishment_fraction','matched_remove_fraction')]
    return dict(geometry=base,context=base+PATH_COLUMNS+MEMORY_COLUMNS+['h_'+k for k in FAMILIES]+mbo)


def actions(p,gate,clean=False):
    up=p[:,1]+p[:,2];down=p[:,3]+p[:,4]
    long_score=(p[:,3] if clean else down)-up
    short_score=(p[:,1] if clean else up)-down
    side=np.where(long_score>short_score,1,-1)
    score=np.maximum(long_score,short_score)
    return side,score,score>=gate


def rolling_predictions(f,config,expert_config):
    """Full historical prefix fits fixed heads; every refit has a strict cutoff."""
    cols=feature_sets(f);records=[];receipts={};expert_rows=[]
    train=f[f.date.le(config['train_end'])&f.known]
    if not len(train):raise ValueError('Missing initialization data')
    frozen={k:EventModel(v,config['model']).fit(train) for k,v in cols.items()}
    receipts['frozen_geometry']=frozen['geometry'].receipt();receipts['frozen_context']=frozen['context'].receipt()
    dates=sorted(f.date.unique());initial=f[f.date.le(config['train_end'])]
    activity=float(initial.x_base_mbo_now_trade_total_log.median());masks=build_masks(f,expert_config,activity)
    for date in dates:
        if date<config['deployment_start'] or date>config['extension_end']:continue
        cutoff=int(pd.Timestamp(date,tz='UTC').value);index=dates.index(date);prior_dates=dates[index-2:index]
        pair=f[f.date.isin(prior_dates)&f.known&(f.release_ns<cutoff)]
        older=f[f.date.lt(date)&f.known&(f.release_ns<cutoff)]
        test=f[f.date.eq(date)]
        pair_model=EventModel(cols['context'],config['model']).fit(pair)
        expand_model=EventModel(cols['context'],config['model']).fit(older)
        if max(pair_model.max_release,expand_model.max_release,frozen['context'].max_release)>=cutoff:raise AssertionError('Future teacher')
        receipts[date+'_pair']=pair_model.receipt();receipts[date+'_expanding']=expand_model.receipt()
        pf=frozen['context'].predict(test);pp=pair_model.predict(test);w=config['pair_blend_weight']
        probabilities={'frozen_geometry':frozen['geometry'].predict(test),'frozen_context':pf,
                       'pair_context':pp,'blend_context':(1-w)*pf+w*pp,
                       'expanding_context':expand_model.predict(test),'clean_blend':(1-w)*pf+w*pp}
        for name,p in probabilities.items():
            side,score,accept=actions(p,config['confidence_gate'],clean=name=='clean_blend')
            a=test[['candidate_id','t_ns','date','target']].copy();a['policy']=name;a['side']=side;a['score']=score;a['accepted']=accept
            a['selection_ns']=cutoff;a['max_feedback_ns']=pair_model.max_release if name in ('pair_context','blend_context','clean_blend') else expand_model.max_release if name=='expanding_context' else frozen['context'].max_release
            for j in range(5):a['p'+str(j)]=p[:,j]
            records.append(a)
        # Expert ranking uses bounded event quality, not net PnL or equity.
        candidates=[]
        for key,spec in CATALOG.items():
            if key=='WAIT':continue
            h=pair.loc[masks.loc[pair.index,key]];side=h.side.to_numpy()*spec['direction']
            reward=-side*h.event_sign.to_numpy();daily=h.groupby('date').size()
            score=float(reward.sum()/(len(h)+config['expert_selection']['prior_count']))
            supported=all(int(daily.get(d,0))>=config['expert_selection']['min_each_day'] for d in prior_dates)
            candidates.append(dict(date=date,strategy=key,n=len(h),score=score,supported=supported,day1=prior_dates[0],day2=prior_dates[1]))
        expert_rows.extend(candidates)
        allowed=sorted([r for r in candidates if r['supported'] and r['score']>=config['confidence_gate']],key=lambda r:(-r['score'],r['strategy']))
        key=allowed[0]['strategy'] if allowed else 'WAIT'
        controls={'expert_pair':(test.side.to_numpy()*CATALOG[key]['direction'],masks.loc[test.index,key].to_numpy()),
                  'memory_return':(test.side.to_numpy(),test.h_memory_return.to_numpy(bool)),
                  'momentum':(np.where(test.past_net30_r.to_numpy()*test.side.to_numpy()>=0,1,-1),test.past_known.to_numpy(bool)),
                  'anti_momentum':(np.where(test.past_net30_r.to_numpy()*test.side.to_numpy()>=0,-1,1),test.past_known.to_numpy(bool)),
                  'always_long':(np.ones(len(test),int),np.ones(len(test),bool)),
                  'always_short':(-np.ones(len(test),int),np.ones(len(test),bool)),
                  'random_all':(np.where(stable_uniform(test.candidate_id,314159)>.5,1,-1),np.ones(len(test),bool))}
        for name,(side,accept) in controls.items():
            a=test[['candidate_id','t_ns','date','target']].copy();a['policy']=name;a['side']=side;a['accepted']=accept
            a['score']=allowed[0]['score'] if name=='expert_pair' and allowed else np.nan
            a['selection_ns']=cutoff;a['max_feedback_ns']=pair_model.max_release if name=='expert_pair' else -1
            a['expert']=key if name=='expert_pair' else '';records.append(a)
    return pd.concat(records,ignore_index=True),receipts,pd.DataFrame(expert_rows)
