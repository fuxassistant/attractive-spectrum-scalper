"""Supervised policy improvement against fixed continuation, with explicit controls.

This is not PPO or reward-only RL: the target is a paired dollar difference.
All samples from an origin stay on one date/split. Training weights balance
roots within dates. No target clipping; selection uses validation MSE only.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from spectrum.core.storage import sha
from spectrum.ingest import save_json

SEEDS=(61021,61022,61023)


def load_dataset(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='COMPLETE':raise ValueError('Training requires complete declared dataset')
    xs=[];frames=[];names=None;market_dim=None
    for d in manifest['days']:
        directory=root/d['date']
        for name,digest in d['files'].items():
            if sha(directory/name)!=digest:raise ValueError('Dataset integrity failure')
        if names is None:names=d['feature_names'];market_dim=d['market_dim']
        if names!=d['feature_names']:raise ValueError('Schema changed between days')
        x=np.load(directory/'x.npy');f=pd.read_parquet(directory/'pairs.parquet')
        if len(x)!=len(f):raise ValueError('Feature/target alignment')
        f['day_row']=np.arange(len(f));xs.append(x);frames.append(f)
    frame=pd.concat(frames,ignore_index=True);x=np.concatenate(xs)
    if np.any(frame.market_available_ns>frame.decision_ns):raise ValueError('Future feature availability')
    if frame.groupby('root_id').split.nunique().max()!=1:raise ValueError('Root crossed splits')
    return x,frame,names,market_dim


def sample_weights(frame):
    """Equal dates, equal observed roots within date, equal ages within root."""
    root_count=frame.groupby('date').root_id.transform('nunique').to_numpy(float)
    ages=frame.groupby('root_id').root_id.transform('size').to_numpy(float)
    w=1/(root_count*ages);return (w/w.mean()).astype(np.float32)


def mask_index(name,names):
    for suffix in ('__known','_known'):
        if name+suffix in names:return names.index(name+suffix)
    for window in (5,15,60):
        if name in (f'return_{window}m_ticks',f'range_{window}m_ticks'):
            return names.index(f'history_{window}m_known')
    return None


class FeatureScaler:
    def fit(self,x,names,weights):
        self.names=list(names);self.mean=np.zeros(x.shape[1]);self.scale=np.ones(x.shape[1])
        for j,name in enumerate(names):
            if name.endswith('_known') or name=='candle_known':continue
            mi=mask_index(name,names);ok=np.ones(len(x),bool) if mi is None else x[:,mi]>.5
            if not ok.any():continue
            v=x[ok,j];w=weights[ok];mean=np.average(v,weights=w)
            self.mean[j]=mean;self.scale[j]=max(float(np.sqrt(np.average((v-mean)**2,weights=w))),1e-4)
        return self

    def transform(self,x):
        y=(np.asarray(x)-self.mean)/self.scale
        # Feature winsorization is a declared input transform; outcomes never clipped.
        y=np.clip(y,-12,12)
        for j,name in enumerate(self.names):
            mi=mask_index(name,self.names)
            if mi is not None:y[:,j]=np.where(x[:,mi]>.5,y[:,j],0.)
        return y.astype(np.float32)

    def dict(self):return dict(names=self.names,mean=self.mean.tolist(),scale=self.scale.tolist())
    @classmethod
    def from_dict(cls,d):
        s=cls();s.names=d['names'];s.mean=np.asarray(d['mean']);s.scale=np.asarray(d['scale']);return s


class AdvantageModel(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def metrics(y,p,w,frame=None):
    err=y-p;close=p<0;gain=np.where(close,-y,0.)
    return dict(weighted_mse=float(np.average(err**2,weights=w)),weighted_mae=float(np.average(abs(err),weights=w)),
        predicted_mean=float(np.average(p,weights=w)),actual_mean=float(np.average(y,weights=w)),
        close_fraction=float(np.average(close,weights=w)),
        paired_gain_vs_hold=float(np.average(gain,weights=w)),
        paired_gain_vs_close=float(np.average(np.where(~close,y,0.),weights=w)),
        hindsight_regret=float(np.average(np.maximum(y,0)-np.where(~close,y,0),weights=w)))


def predict(model,z,target_scale,device):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(z),4096):out.append(model(torch.as_tensor(z[start:start+4096],device=device)).cpu().numpy()*target_scale)
    return np.concatenate(out) if out else np.empty(0)


def train(data='data/derived/hold_close_v1',output='reports/hold_close_v1',epochs=40):
    started=time.perf_counter();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    x,frame,names,md=load_dataset(data)
    known=frame.target_known.to_numpy(bool)
    masks={s:known&(frame.split.to_numpy()==s) for s in ('train','validation','test')}
    if any(not m.any() for m in masks.values()):raise ValueError('Missing split')
    y=frame.target_advantage.to_numpy(float)
    weights={s:sample_weights(frame.loc[m]) for s,m in masks.items()}
    scaler=FeatureScaler().fit(x[masks['train']],names,weights['train']);z=scaler.transform(x)
    save_json(out/'normalizer.json',scaler.dict())
    target_scale=max(.5,float(np.sqrt(np.average(y[masks['train']]**2,weights=weights['train']))))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    protocol=dict(status='TRAINING',data_manifest_sha256=sha(Path(data)/'manifest.json'),source_sha256=sha(Path(__file__)),
        seeds=SEEDS,epochs=epochs,network='MLP input -> 64 tanh -> 32 tanh -> dollar advantage',
        objective='Root/day-weighted MSE of HOLD_fixed_plan_cash minus CLOSE_cash, normalized by train-only RMS; no outcome clipping.',
        selection='Minimum validation weighted MSE per seed. Main full seed fixed to61021 in advance; never pick by test.',
        optimizer='AdamW lr0.001 weight_decay0.0001, batch1024, gradient norm cap5',
        action='CLOSE when predicted advantage <0; otherwise HOLD. No fitted test threshold.',
        controls=['account_only','permuted_train_targets','fixed_HOLD','CLOSE_at_first_decision','age60'],
        scope='Technical chronological development on previously researched market history; independent roots, not live profit admission.',
        target_scale=target_scale,device=device,feature_names=names,market_dim=md,
        rows={s:int(m.sum()) for s,m in masks.items()},censored_pairs=int((~known).sum()))
    save_json(out/'training_protocol.json',protocol)
    variants=[(v,seed) for v in ('full','account_only') for seed in SEEDS]+[('permuted_targets',SEEDS[0])]
    all_predictions={};summaries=[]
    for variant,seed in variants:
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
        cols=np.arange(md,len(names)) if variant=='account_only' else np.arange(len(names))
        model=AdvantageModel(len(cols)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        tx=torch.as_tensor(z[masks['train']][:,cols],device=device)
        ty=y[masks['train']].copy()/target_scale
        if variant=='permuted_targets':ty=ty[rng.permutation(len(ty))]
        ty=torch.as_tensor(ty,dtype=torch.float32,device=device);tw=torch.as_tensor(weights['train'],device=device)
        history=[];best=float('inf');best_state=None;best_epoch=0
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(len(tx));gradient_max=0.;loss_sum=0.
            for start in range(0,len(order),1024):
                ix=torch.as_tensor(order[start:start+1024],device=device)
                opt.zero_grad();pred=model(tx[ix]);loss=((pred-ty[ix])**2*tw[ix]).sum()/tw[ix].sum()
                loss.backward();norm=float(nn.utils.clip_grad_norm_(model.parameters(),5.));gradient_max=max(gradient_max,norm)
                opt.step();loss_sum+=float(loss.detach())*len(ix)
            vp=predict(model,z[masks['validation']][:,cols],target_scale,device)
            vm=metrics(y[masks['validation']],vp,weights['validation'])
            history.append(dict(epoch=epoch,train_loss=loss_sum/len(tx),gradient_norm_max=gradient_max,**vm))
            if vm['weighted_mse']<best:
                best=vm['weighted_mse'];best_epoch=epoch;best_state=copy.deepcopy(model.cpu().state_dict());model.to(device)
        model.load_state_dict(best_state)
        key=f'{variant}_{seed}';preds={}
        for split,m in masks.items():
            p=predict(model,z[m][:,cols],target_scale,device);preds[split]=metrics(y[m],p,weights[split])
        pred_all=predict(model,z[:,cols],target_scale,device);all_predictions[key]=pred_all
        torch.save(dict(state_dict=best_state,columns=cols.tolist(),target_scale=target_scale,
            normalizer=scaler.dict(),variant=variant,seed=seed,best_epoch=best_epoch),out/(key+'.pt'))
        save_json(out/(key+'_history.json'),history)
        summaries.append(dict(model=key,best_epoch=best_epoch,splits=preds))
        print(json.dumps(summaries[-1]),flush=True)
    prediction_frame=frame.drop(columns=['snapshot_json']).copy()
    for name,value in all_predictions.items():prediction_frame[name]=value
    prediction_frame.to_parquet(out/'paired_predictions.parquet',index=False)
    protocol['status']='COMPLETE';protocol['seconds']=time.perf_counter()-started
    save_json(out/'training_protocol.json',protocol);save_json(out/'training_summary.json',summaries)


def load_predictor(path):
    c=torch.load(path,map_location='cpu',weights_only=False)
    model=AdvantageModel(len(c['columns']));model.load_state_dict(c['state_dict']);model.eval()
    scaler=FeatureScaler.from_dict(c['normalizer']);cols=np.asarray(c['columns'])
    def fn(x):
        z=scaler.transform(np.atleast_2d(x))[:,cols]
        with torch.no_grad():return model(torch.from_numpy(z)).numpy()*c['target_scale']
    return fn,c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/derived/hold_close_v1');p.add_argument('--output',default='reports/hold_close_v1');p.add_argument('--epochs',type=int,default=40)
    a=p.parse_args();train(a.data,a.output,a.epochs)
