from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import json,time,shutil
import numpy as np
import pandas as pd
import torch
from spectrum.player.contract import Contract
from spectrum.player.data import Day
from spectrum.player.model import Scaler,advantages
from spectrum.player.gym import TradingGym
from spectrum.player.experiment import save_replay
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from .policy import HierarchicalPlayer,distribution,distribution_numpy,greedy


def identities(root):
    return {p.relative_to(root).as_posix():sha(p) for name in ('player','player_v2') for p in sorted((root/'spectrum'/name).glob('*.py'))}


def rollout(model,scaler,day,c,rng=None,record=False,flat=False):
    start=time.perf_counter();device=next(model.parameters()).device
    x=scaler.transform(day.x);model.eval()
    with torch.no_grad():
        h=model.encode(torch.from_numpy(x)[None].to(device))
        base=model.head(h)[0].cpu().numpy();vb=model.critic(h)[0,:,0].cpu().numpy()
        aw=model.account_head.weight.detach().cpu().numpy();vw=model.account_critic.weight.detach().cpu().numpy()[0]
    env=TradingGym(day,c,record);data={k:[] for k in ('account','mask','action','logp','value','reward','probability','intent','gate','direction','plans','blocked')}
    for i in range(len(x)):
        account=env.account();mask=env.mask();joint,intent,gate,direction,plans=distribution_numpy(base[i]+aw@account,mask)
        action=0 if flat else int(rng.choice(7,p=joint.astype(float)/joint.sum(dtype=float))) if rng is not None else greedy(gate,direction,plans)
        blocked='eligible' if mask[1:].any() else 'position_or_pending' if env.s[0] or env.s[5] else 'risk_halt' if env.s[12] else 'pre_london_context' if day.decision_t[i]<int(day.meta['london_ns']) else 'data_or_session_or_plan_mask'
        _,reward,_,_,_=env.step(action)
        values=dict(account=account,mask=mask,action=action,logp=np.log(max(float(joint[action]),1e-30)),value=vb[i]+vw@account,
                    reward=reward/c.reward_scale,probability=joint,intent=intent,gate=gate,direction=direction,plans=plans,blocked=blocked)
        for k,v in values.items():data[k].append(v)
    b={k:np.asarray(v) for k,v in data.items()};b['x']=x
    for k in ('account','logp','value','reward','probability','intent','gate','direction','plans'):b[k]=b[k].astype(np.float32)
    b['advantage'],b['returns']=advantages(b['reward'],b['value'],c.gamma,c.gae_lambda)
    m=env.metrics();eligible=b['mask'][:,1:].any(axis=1)
    m.update(date=day.meta['date'],seconds=time.perf_counter()-start,reward_sum=float(b['reward'].sum(dtype=np.float64)*c.reward_scale),
             mean_intent=float(b['intent'][eligible].mean()) if eligible.any() else None,eligible=int(eligible.sum()))
    return b,m,env


def optimize(model,optimizer,rollouts,c):
    # Entire unsupported episodes are excluded, including predecessor advantages.
    batches=[b for b,m in rollouts if m['complete']]
    eligible=[b['advantage'][b['mask'][:,1:].any(axis=1)] for b in batches]
    n=sum(len(a) for a in eligible)
    if not n:return dict(updated=False,eligible=0,clean_days=len(batches))
    a=np.concatenate(eligible);center=float(a.mean());scale=max(float(a.std()),1e-6)
    total_steps=sum(len(b['action']) for b in batches);device=next(model.parameters()).device;stats=[];model.train()
    for _ in range(c.ppo_passes):
        optimizer.zero_grad();s=np.zeros(4)
        for b in batches:
            t=lambda name:torch.from_numpy(b[name]).to(device)
            logits,value=model(t('x')[None],t('account')[None]);joint,*_=distribution(logits[0],t('mask'))
            dist=torch.distributions.Categorical(probs=joint);lp=dist.log_prob(t('action'));ratio=(lp-t('logp')).exp()
            legal=t('mask')[:,1:].any(-1);adv=(t('advantage')-center)/scale
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-c.clip_ratio,1+c.clip_ratio)*adv)[legal].sum()/n
            critic=.5*(value[0]-t('returns')).square().sum()/total_steps
            entropy=dist.entropy()[legal].sum()/n
            loss=actor+.5*critic-c.entropy_weight*entropy;loss.backward()
            s+=np.asarray([float(z.detach()) for z in (actor,critic,entropy,((ratio-1)-(lp-t('logp')))[legal].sum()/n)])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.gradient_clip);optimizer.step();stats.append(np.r_[s,float(norm)])
    return dict(updated=True,eligible=n,clean_days=len(batches),**dict(zip(('actor_loss','critic_loss','entropy','kl','gradient_norm'),np.mean(stats,axis=0).tolist())))


def save(root,label,day,b,m,e):
    save_replay(root,label,day,b,m,e)
    dest=Path(root)/label/day.meta['date']
    np.savez_compressed(dest/'hierarchy.npz',intent=b['intent'],gate=b['gate'],direction=b['direction'],plans=b['plans'],blocked=b['blocked'])
    if e.trades:
        tr=pd.DataFrame(e.trades);idx=tr.decision_index.to_numpy(int)
        tr['participation_intent']=b['intent'][idx];tr['direction_probability']=b['direction'][idx,(tr.side.to_numpy()<0).astype(int)]
        tr['plan_probability']=b['plans'][idx,(tr.side.to_numpy()<0).astype(int),tr.plan.to_numpy(int)]
        tr.to_csv(dest/'trades.csv',index=False)


def run(root='.',name='run_001',epochs=60,seeds=(61015,61016,61017)):
    root=Path(root).resolve();out=root/'data/research/player_v2'/name;out.mkdir(parents=True,exist_ok=False)
    report=root/'reports/player_v2'/name;report.mkdir(parents=True,exist_ok=False)
    c=replace(Contract(),version='HIERARCHICAL_LSTM_V2',epochs=epochs)
    train_dates=[f'2026-05-{i:02d}' for i in range(4,8)];test_dates=[f'2026-05-{i:02d}' for i in range(11,16)]
    protocol=dict(contract=c.dict(),seeds=list(seeds),train_dates=train_dates,validation_date='2026-05-08',evaluation_dates=test_dates,
                  hierarchy='Binary participation >0.5, then argmax direction, then argmax conditional plan; stochastic policy samples exact product.',
                  selection='Per seed: maximum complete Friday greedy net, earliest tie; no selection across seeds from evaluation.',
                  censor='Skip entire censored training episodes; censored validation excluded; evaluation censor retained and blocks economic admission.',
                  exposure='Week2 is development-exposed by v1. This is a controlled development comparison, NOT fresh OOS confirmation.',
                  diagnostics='Frozen stochastic replay seeds 101,202,303 on each week2 day; fixed intention bins [0,.2,.4,.6,.8,1]. No threshold fitting.',
                  attribution='Signed gradient times standardized market input for gate log-odds; local sensitivity, not causal market explanation or calibrated profit probability.',
                  created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=identities(root))
    save_json(out/'protocol.json',protocol);save_json(report/'protocol.json',protocol)
    for file,h in protocol['code_sha256'].items():
        dest=out/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/file,dest)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    days=[Day(root/'data/derived/player_v1/week1'/d) for d in train_dates+['2026-05-08']]
    scaler=Scaler().fit(days[:4]);scaler.save(out/'scaler.npz');frozen=[]
    for seed in seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=HierarchicalPlayer().to(device);opt=torch.optim.AdamW(model.parameters(),lr=c.learning_rate)
        sd=out/f'seed_{seed}';sd.mkdir();torch.save(model.state_dict(),sd/'untrained.pt')
        initial=model.lstm.weight_ih_l0.detach().clone();best=-np.inf;selected=None;history=[];start=time.perf_counter()
        # Epoch zero is a declared baseline, not a selection candidate.
        b,m,e=rollout(model,scaler,days[4],c);save(sd,'validation_epoch_000',days[4],b,m,e)
        for epoch in range(1,epochs+1):
            batches=[];train=[]
            for d in days[:4]:
                b,m,e=rollout(model,scaler,d,c,rng);batches.append((b,m));train.append(m)
                save(sd,f'train_epoch_{epoch:03d}',d,b,m,e)
            stats=optimize(model,opt,batches,c)
            b,m,e=rollout(model,scaler,days[4],c);save(sd,f'validation_epoch_{epoch:03d}',days[4],b,m,e)
            torch.save(model.state_dict(),sd/f'epoch_{epoch:03d}.pt')
            row=dict(seed=seed,epoch=epoch,train_net=sum(x['net'] for x in train),train_trades=sum(x['trades'] for x in train),
                     train_censored=sum(x['censored'] for x in train),validation_net=m['net'],validation_trades=m['trades'],
                     validation_censored=m['censored'],validation_intent=m['mean_intent'],**stats)
            history.append(row);pd.DataFrame(history).to_csv(sd/'epochs.csv',index=False)
            if m['complete'] and m['net']>best:best=m['net'];selected=epoch;torch.save(model.state_dict(),sd/'selected.pt')
            if epoch%10==0:print(json.dumps(row),flush=True)
        if selected is None:raise RuntimeError(f'No complete validation candidate for seed {seed}; evaluation not opened')
        freeze=dict(seed=seed,selected_epoch=selected,validation_net=best,selected_sha256=sha(sd/'selected.pt'),scaler_sha256=sha(out/'scaler.npz'),
                    protocol_sha256=sha(out/'protocol.json'),seconds=time.perf_counter()-start,lstm_delta=float((model.lstm.weight_ih_l0.detach()-initial).norm()),
                    frozen_utc=datetime.now(timezone.utc).isoformat())
        save_json(sd/'freeze.json',freeze);frozen.append(freeze)
    # All seeds frozen before any evaluation pass. No cross-seed winner picked.
    save_json(out/'all_frozen_before_evaluation.json',frozen);save_json(report/'all_frozen_before_evaluation.json',frozen)
    test=[Day(root/'data/derived/player_v1/week2'/d) for d in test_dates];results=[]
    for seed in seeds:
        sd=out/f'seed_{seed}';model=HierarchicalPlayer().to(device)
        configurations=[('selected_train','selected',False,None,days),('selected_test','selected',False,None,test),
                        ('untrained_test','untrained',False,None,test),('wait_test','selected',True,None,test)]
        configurations += [(f'sampled_test_{rs}','selected',False,rs,test) for rs in (101,202,303)]
        for label,checkpoint,flat,rs,ds in configurations:
            model.load_state_dict(torch.load(sd/(checkpoint+'.pt'),weights_only=True,map_location=device));rng=np.random.default_rng(rs) if rs else None;ms=[]
            for d in ds:
                b,m,e=rollout(model,scaler,d,c,rng,record=label in ('selected_train','selected_test'),flat=flat)
                save(sd,label,d,b,m,e);ms.append(m)
            row=dict(seed=seed,policy=label,net=sum(m['net'] for m in ms),trades=sum(m['trades'] for m in ms),censored=sum(m['censored'] for m in ms),days=ms)
            results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='days'}),flush=True)
    save_json(out/'summary.json',results);save_json(report/'summary.json',results)
    return out
