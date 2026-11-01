"""Frozen regularized classifier, train-only preprocessing, honest day metrics."""
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

CLASSES=['RETURN_FIRST','ADVERSE_FIRST','TIMEOUT']


def day_clock_weights(frame, normalize=True):
    n=frame.groupby(['date','t_ns']).candidate_id.transform('size').to_numpy(float)
    w=1/n
    totals=pd.Series(w,index=frame.index).groupby(frame.date).transform('sum').to_numpy()
    w=w/totals
    return w*len(w)/w.sum() if normalize else w


class FrozenClassifier:
    def __init__(self, columns, config):
        self.columns=list(columns);self.config=config

    def fit(self, frame):
        if not len(frame) or not frame.state.isin(CLASSES).all():
            raise ValueError('Fit requires known geometric outcomes')
        w=day_clock_weights(frame)
        raw=frame[self.columns].to_numpy(float)
        raw[~np.isfinite(raw)]=np.nan
        # An all-missing train column receives zero plus its missing indicator.
        self.median=np.array([np.nanmedian(raw[:,j]) if np.isfinite(raw[:,j]).any() else 0. for j in range(raw.shape[1])])
        a=np.concatenate([np.where(np.isfinite(raw),raw,self.median),~np.isfinite(raw)],axis=1).astype(float)
        self.mean=np.average(a,axis=0,weights=w)
        self.scale=np.sqrt(np.average((a-self.mean)**2,axis=0,weights=w))
        self.scale=np.where(self.scale>1e-12,self.scale,1.)
        y=frame.state.map({k:i for i,k in enumerate(CLASSES)}).to_numpy(int)
        if len(np.unique(y))!=3:
            raise ValueError('Three target classes required in training')
        c=self.config
        self.estimator=LogisticRegression(C=c['C'],solver=c['solver'],max_iter=c['max_iter'],tol=c['tol'],random_state=c['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.estimator.fit(np.clip((a-self.mean)/self.scale,-8,8),y,sample_weight=w)
        self.training_rows=len(frame);self.max_training_label_ns=int(frame.label_available_ns.max())
        self.max_training_origin_ns=int(frame.t_ns.max())
        return self

    def transform(self, frame):
        raw=frame[self.columns].to_numpy(float)
        missing=~np.isfinite(raw)
        a=np.concatenate([np.where(~missing,raw,self.median),missing],axis=1).astype(float)
        return np.clip((a-self.mean)/self.scale,-8,8)

    def predict(self, frame):
        p=self.estimator.predict_proba(self.transform(frame))
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1):
            raise ValueError('Invalid predictive probabilities')
        return p

    def receipt(self):
        return dict(columns=self.columns,median=self.median.tolist(),mean=self.mean.tolist(),scale=self.scale.tolist(),
                    coefficients=self.estimator.coef_.tolist(),intercept=self.estimator.intercept_.tolist(),
                    classes=CLASSES,iterations=self.estimator.n_iter_.tolist(),training_rows=self.training_rows,
                    max_training_label_ns=self.max_training_label_ns,max_training_origin_ns=self.max_training_origin_ns)


def feature_arms(frame):
    groups={g:[c for c in frame if c.startswith('x_'+g+'_')] for g in ['base','mem','spec','mbo','placebo']}
    if any(not groups[g] for g in groups):raise ValueError('Missing feature family')
    b,m,s,o,p=(groups[g] for g in ['base','mem','spec','mbo','placebo'])
    return dict(base=b,memory=b+m,memory_spectrum=b+m+s,memory_mbo=b+m+o,all=b+m+s+o,past_placebo=b+p)


def losses(frame, probabilities):
    y=frame.state.map({k:i for i,k in enumerate(CLASSES)}).to_numpy(int)
    oh=np.eye(3)[y]
    return pd.DataFrame(dict(candidate_id=frame.candidate_id.to_numpy(),date=frame.date.to_numpy(),
        t_ns=frame.t_ns.to_numpy(),logloss=-np.log(np.clip(probabilities[np.arange(len(y)),y],1e-12,1)),
        brier=((probabilities-oh)**2).sum(axis=1),return_brier=(probabilities[:,0]-(y==0))**2,
        p_return=probabilities[:,0],observed_return=(y==0).astype(float)),index=frame.index)


def daily_metrics(frame, probabilities):
    l=losses(frame,probabilities)
    rows=[]
    for date,g in l.groupby('date'):
        w=1/g.groupby('t_ns').candidate_id.transform('size').to_numpy(float)
        rows.append(dict(date=date,rows=len(g),clocks=g.t_ns.nunique(),**{k:float(np.average(g[k],weights=w)) for k in ['logloss','brier','return_brier']}))
    return pd.DataFrame(rows),l


def paired_day_interval(deltas,seed=271828,n=2000):
    a=np.asarray(deltas,float)
    rng=np.random.default_rng(seed)
    boot=a[rng.integers(0,len(a),(n,len(a)))].mean(axis=1)
    return dict(days=len(a),delta=float(a.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                interpretation='descriptive paired day bootstrap; few days and previously inspected tape')


def common_policy_clock(frame):
    return frame.sort_values(['t_ns','distance_per_oz','zone_low_nano','candidate_id']).drop_duplicates('t_ns').copy()


def deterministic_accept(candidate_ids, probability, seed):
    # A stable integer hash needs neither future opportunities nor daily counts.
    x=np.asarray(candidate_ids,dtype=np.uint64)^np.uint64(seed)
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    x=x^(x>>np.uint64(31))
    return (x>>np.uint64(11)).astype(float)/(2**53)<probability
