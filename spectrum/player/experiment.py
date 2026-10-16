"""Predeclared week-one PPO experiment; week two is opened only after freeze."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,random
import numpy as np
import pandas as pd
import torch
from .contract import Contract,code_identity
from .data import Day,prepare_week
from .gym import TradingGym
from .model import Player,Scaler,advantages
from spectrum.ingest import save_json
from spectrum.core.storage import sha


def rollout(model,scaler,day,c,rng=None,record=False,flat=False):
    start=time.perf_counter();device=next(model.parameters()).device
    x=scaler.transform(day.x);xt=torch.from_numpy(x).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        h=model.encode(xt);base=model.actor(h)[0].cpu().numpy();vbase=model.critic(h)[0,:,0].cpu().numpy()
        aw=model.actor_account.weight.detach().cpu().numpy();vw=model.critic_account.weight.detach().cpu().numpy()[0]
    env=TradingGym(day,c,record);accounts=[];masks=[];actions=[];logps=[];values=[];rewards=[];probs=[]
    for i in range(len(x)):
        account=env.account();mask=env.mask();logits=base[i]+aw@account;logits[~mask]=-1e9
        p=np.exp(logits-logits.max());p/=p.sum()
        a=0 if flat else int(rng.choice(7,p=p.astype(float)/p.sum(dtype=float))) if rng is not None else int(np.argmax(logits))
        _,reward,_,_,_=env.step(a)
        accounts.append(account);masks.append(mask);actions.append(a);logps.append(np.log(max(float(p[a]),1e-30)))
        values.append(float(vbase[i]+vw@account));rewards.append(reward/c.reward_scale);probs.append(p.copy())
    adv,ret=advantages(np.asarray(rewards),np.asarray(values,dtype=np.float32),c.gamma,c.gae_lambda)
    result=dict(x=x,account=np.asarray(accounts),mask=np.asarray(masks),action=np.asarray(actions),
                logp=np.asarray(logps,dtype=np.float32),value=np.asarray(values,dtype=np.float32),
                reward=np.asarray(rewards),advantage=adv,returns=ret,probability=np.asarray(probs))
    metrics=env.metrics();metrics.update(date=day.meta['date'],seconds=time.perf_counter()-start,
                                      reward_sum=float(np.sum(rewards)*c.reward_scale))
    return result,metrics,env


def optimize(model,optimizer,batches,c):
    model.train();device=next(model.parameters()).device
    eligible=[b for b in batches if b['mask'][:,1:].any()]
    actor_adv=np.concatenate([b['advantage'][b['mask'][:,1:].any(axis=1)] for b in eligible])
    center=actor_adv.mean();scale=max(float(actor_adv.std()),1e-6);stats=[]
    for _ in range(c.ppo_passes):
        optimizer.zero_grad();pass_stats=[]
        for b in batches:
            tensor=lambda name:torch.from_numpy(b[name]).to(device)
            logits,value=model(tensor('x')[None],tensor('account')[None]);logits=logits[0];value=value[0]
            mask=tensor('mask');dist=torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1e9))
            lp=dist.log_prob(tensor('action'));ratio=(lp-tensor('logp')).exp()
            adv=(tensor('advantage')-float(center))/float(scale)
            legal=mask[:,1:].any(dim=1)
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-c.clip_ratio,1+c.clip_ratio)*adv)[legal].mean()
            critic=.5*(value-tensor('returns')).square().mean()
            entropy=dist.entropy()[legal].mean()
            loss=actor+.5*critic-c.entropy_weight*entropy
            (loss/len(batches)).backward()
            pass_stats.append([float(z.detach()) for z in (loss,actor,critic,entropy,((ratio-1)-(lp-tensor('logp')))[legal].mean())])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.gradient_clip)
        optimizer.step();stats.append(np.r_[np.mean(pass_stats,axis=0),float(norm)])
    return dict(zip(('loss','actor_loss','critic_loss','entropy','approx_kl','gradient_norm'),np.mean(stats,axis=0).tolist()))


def save_replay(root,name,day,result,metrics,env):
    out=Path(root)/name/day.meta['date'];out.mkdir(parents=True,exist_ok=True)
    save_json(out/'metrics.json',metrics)
    trades=pd.DataFrame(env.trades)
    trades.to_csv(out/'trades.csv',index=False)
    # Exact clocks are int64 parquet; JSON clock fields remain decimal strings.
    pd.DataFrame(dict(t_ns=day.decision_t,action=result['action'],equity_before=np.asarray(env.equity[:-1]),
                      equity_after=np.asarray(env.equity[1:]),reward=result['reward']*env.c.reward_scale,
                      entry_allowed=day.can_enter)).to_parquet(out/'decisions.parquet',index=False)
    np.savez_compressed(out/'policy_trace.npz',mask=result['mask'],account=result['account'],logp=result['logp'],value=result['value'],probability=result['probability'])
    if env.record:
        # Keep all quote-level equity for independent drawdown and barrier audit.
        np.save(out/'quote_equity.npy',env.trace)


def run(root='.',run_name='run_001',epochs=30):
    root=Path(root).resolve();c=Contract(epochs=epochs)
    out=root/'data/research/player_v1'/run_name;out.mkdir(parents=True,exist_ok=False)
    reports=root/'reports/player_v1'/run_name;reports.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(c.seed);np.random.seed(c.seed);random.seed(c.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(c.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    protocol=dict(contract=c.dict(),code_sha256=code_identity(),train_days=['2026-05-04','2026-05-05','2026-05-06','2026-05-07'],
                  selection_day='2026-05-08',test_days=['2026-05-11','2026-05-12','2026-05-13','2026-05-14','2026-05-15'],
                  selection='Highest deterministic Friday net among uncensored epochs; earliest tie. All-flat separately reported.',
                  reward='Unclipped change of liquidation equity / 5; gamma=1, GAE=.95; no inactivity penalty.',
                  censor_policy='Support-loss trades close at next valid quote. Observed proxy return may train; any censor excludes selection and invalidates economic admission.',
                  prior_exposure='Week two was visible in foundation QA, but not used by this model or checkpoint selection. This is not pristine blind research.',
                  created_utc=datetime.now(timezone.utc).isoformat(),device=device,torch=torch.__version__)
    save_json(out/'protocol.json',protocol);save_json(reports/'protocol.json',protocol)
    cache=root/'data/derived/player_v1';week1=cache/'week1'
    if not week1.exists():prepare_week(root/'data/derived/market_v1',root/'data/derived/features_v1',week1,1,c)
    days=[Day(week1/d) for d in protocol['train_days']+[protocol['selection_day']]]
    scaler=Scaler().fit(days[:4]);scaler.save(out/'scaler.npz')
    model=Player(days[0].x.shape[1],c.hidden_size).to(device)
    torch.save(model.state_dict(),out/'untrained.pt');initial_lstm=model.lstm.weight_ih_l0.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate)
    rng=np.random.default_rng(c.seed);history=[];best=-np.inf;best_epoch=None;started=time.perf_counter()
    for epoch in range(1,c.epochs+1):
        batches=[];train=[]
        for day in days[:4]:
            b,m,e=rollout(model,scaler,day,c,rng);batches.append(b);train.append(m)
            save_replay(out,f'train_sampled_epoch_{epoch:03d}',day,b,m,e)
        stats=optimize(model,optimizer,batches,c)
        b,val,e=rollout(model,scaler,days[4],c)
        row=dict(epoch=epoch,train_sampled_net=sum(m['net'] for m in train),train_trades=sum(m['trades'] for m in train),
                 train_censored=sum(m['censored'] for m in train),validation_net=val['net'],validation_trades=val['trades'],
                 validation_drawdown=val['max_drawdown'],validation_censored=val['censored'],**stats)
        history.append(row);save_json(out/'epochs.json',history);pd.DataFrame(history).to_csv(reports/'epochs.csv',index=False)
        save_json(out/f'epoch_{epoch:03d}.json',dict(train=train,validation=val,optimization=stats))
        torch.save(model.state_dict(),out/f'epoch_{epoch:03d}.pt')
        save_replay(out,f'validation_epoch_{epoch:03d}',days[4],b,val,e)
        if val['complete'] and val['net']>best:
            best=val['net'];best_epoch=epoch;torch.save(model.state_dict(),out/'selected.pt')
        print(json.dumps(row),flush=True)
    if best_epoch is None:
        save_json(reports/'blocked.json',dict(reason='All validation epochs censored. No checkpoint admitted.'))
        raise RuntimeError('All validation epochs censored; test stays unopened')
    delta=float((model.lstm.weight_ih_l0.detach()-initial_lstm).norm())
    freeze=dict(selected_epoch=best_epoch,selection_net=best,flat_validation_net=0.,selected_sha256=sha(out/'selected.pt'),
                scaler_sha256=sha(out/'scaler.npz'),protocol_sha256=sha(out/'protocol.json'),code_sha256=code_identity(),
                frozen_utc=datetime.now(timezone.utc).isoformat(),training_seconds=time.perf_counter()-started,lstm_weight_change_l2=delta,
                deployment_selection='LSTM' if best>0 else 'WAIT baseline; LSTM evaluated as experiment')
    save_json(out/'frozen_before_evaluation.json',freeze);save_json(reports/'frozen_before_evaluation.json',freeze)
    # This is the sole opening of evaluation arrays, strictly after checkpoint freeze.
    week2=cache/'week2'
    if not week2.exists():prepare_week(root/'data/derived/market_v1',root/'data/derived/features_v1',week2,2,c)
    test=[Day(week2/d) for d in protocol['test_days']]
    summary=dict(freeze=freeze,results={})
    for label,checkpoint,flat,which_days in [('selected_train','selected.pt',False,days),('selected_test','selected.pt',False,test),
                                          ('untrained_test','untrained.pt',False,test),('wait_test','untrained.pt',True,test)]:
        model.load_state_dict(torch.load(out/checkpoint,weights_only=True,map_location=device));ms=[]
        for day in which_days:
            b,m,e=rollout(model,scaler,day,c,record=True,flat=flat);save_replay(out,label,day,b,m,e);ms.append(m)
        summary['results'][label]=dict(days=ms,net=sum(m['net'] for m in ms),trades=sum(m['trades'] for m in ms),censored=sum(m['censored'] for m in ms))
        print(json.dumps(dict(evaluation=label,**{k:v for k,v in summary['results'][label].items() if k!='days'})),flush=True)
    save_json(out/'summary.json',summary);save_json(reports/'summary.json',summary)
    return out,reports
