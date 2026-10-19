"""Frozen chronological comparison of baseline and mixed-memory populations."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,random
import numpy as np
import pandas as pd
import torch
from spectrum.player.contract import Contract
from spectrum.player.model import advantages
from spectrum.player_v2.policy import distribution,distribution_numpy,greedy
from spectrum.player_v3.data import Day
from spectrum.player_v3.policy import Standardizer,PreparedDay,make_model,encode
from spectrum.player_v3.gym import CapitalState,SurvivalGym,population_metrics,eligible_intervals,risk_contract
from spectrum.core.storage import sha
from spectrum.ingest import save_json


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def prefix_advantages(rewards,values,bootstrap=0.,gamma=1.,lam=.95):
    """A support-truncated clean prefix bootstraps; a real terminal has value zero."""
    result=np.zeros(len(rewards),np.float32);carry=0.
    for i in range(len(rewards)-1,-1,-1):
        nxt=values[i+1] if i+1<len(values) else bootstrap
        carry=rewards[i]+gamma*nxt-values[i]+gamma*lam*carry;result[i]=carry
    return result,result+np.asarray(values)


def rollout(model,prepared,c,rng=None,ablation=None,record_dir=None):
    """One account survives across the whole ordered population horizon."""
    device=next(model.parameters()).device;model.eval()
    state=CapitalState(initial_capital=c.initial_capital,cash=c.initial_capital,peak=c.initial_capital)
    batches=[];daily=[];bootstrap=0.;observed_reward=0.
    scheduled=sum((e-s).sum()/1e9 for p in prepared for s,e in [eligible_intervals(p.day)])
    for p in prepared:
        if state.ruined or state.censored:break
        d=p.day; n=len(d.x); ix=np.flatnonzero(d.can_enter)
        with torch.no_grad(),torch.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
            encoded=encode(model,p,ix,ablation,diagnostics=record_dir is not None)
            h,diag=encoded if record_dir is not None else (encoded,{})
            base=np.zeros((n,10),np.float32);vb=np.zeros(n,np.float32)
            base[ix]=model.head(h).float().cpu().numpy();vb[ix]=model.critic(h).squeeze(-1).float().cpu().numpy()
            aw=model.account_head.weight.float().cpu().numpy();vw=model.account_critic.weight.float().cpu().numpy()[0]
        env=SurvivalGym(d,c,record=record_dir is not None,capital_state=state)
        acc=[];masks=[];actions=[];logs=[];values=[];rewards=[];intents=[];gates=[];longs=[];plan=[]
        risk_before_rows=[];risk_after_rows=[]
        for i in range(n):
            a0=env.account();mask=env.mask()
            joint,intent,gate,direction,plans=distribution_numpy(base[i]+aw@a0,mask)
            action=int(rng.choice(7,p=joint/joint.sum())) if rng is not None else greedy(gate,direction,plans)
            before=max(env.f[1],c.initial_capital*1e-8)
            risk_before=env.risk_snapshot();was_ruined=bool(env.s[12])
            env.step(action)
            risk_after=env.risk_snapshot()
            reward=np.log(max(env.f[1],c.initial_capital*1e-8)/before)-.25*(
                risk_after['max_series_loss_fraction']-risk_before['max_series_loss_fraction'])
            risk_before_rows.append(risk_before);risk_after_rows.append(risk_after)
            if not was_ruined and env.s[12]:
                ruin=int(env.s[15]);remaining=0.
                for other in prepared:
                    starts,ends=eligible_intervals(other.day)
                    remaining+=np.maximum(0,ends-np.maximum(starts,ruin)).sum()/1e9
                reward-=.25*remaining/max(scheduled,1.)
            acc.append(a0);masks.append(mask);actions.append(action);logs.append(np.log(max(joint[action],1e-30)))
            values.append(float(vb[i]+vw@a0) if d.can_enter[i] else 0.)
            rewards.append(float(reward)*100.)
            intents.append(intent);gates.append(gate);longs.append(direction[0]);plan.append(int(np.argmax(plans[int(direction[1]>direction[0])])))
            if env.done or env.s[12] or (env.trades and env.trades[-1]['censored']):break
        batch=dict(prepared=p,account=np.asarray(acc),mask=np.asarray(masks),action=np.asarray(actions),
                   logp=np.asarray(logs,np.float32),value=np.asarray(values,np.float32),reward=np.asarray(rewards,np.float32))
        observed_reward+=sum(rewards)/100.
        uncertain=[t['decision_index'] for t in env.trades if t['censored']]
        if env.s[0]:uncertain.append(int(env.s[14]))
        if uncertain:
            cut=min(uncertain);bootstrap=float(batch['value'][cut])
            for key in ('account','mask','action','logp','value','reward'):batch[key]=batch[key][:cut]
        batches.append(batch)
        metrics=dict(env.metrics(),date=d.meta['date']);daily.append(metrics);state=env.capital_state()
        if record_dir is not None:
            out=Path(record_dir)/d.meta['date'];out.mkdir(parents=True,exist_ok=True)
            steps=len(actions);ix=ix[ix<steps]
            if diag:diag={key:value[:len(ix)] for key,value in diag.items()}
            frame=pd.DataFrame(dict(t_ns=d.decision_t[:steps],mid=d.decision_mid[:steps],action=actions,
                equity_before=env.equity[:-1],equity_after=env.equity[1:],intent=intents,gate=gates,p_long=longs,
                plan=plan,can_enter=d.can_enter[:steps],legal_entry=np.asarray(masks)[:,1:].any(1),reward=rewards))
            # Risk state is aligned BEFORE each origin, just like equity_before.
            # After-step state is exported under explicit *_after names.
            for key in risk_before_rows[0]:
                frame[key]=[row[key] for row in risk_before_rows]
                frame[key+'_after']=[row[key] for row in risk_after_rows]
            for j,branch in enumerate(('fast','slow','anchors','events')):
                frame['branch_'+branch]=np.nan
                if diag:frame.loc[ix,'branch_'+branch]=diag['branch_weights'][:,j]
            for j,key in enumerate(('fast_count','slow_count','event_count')):
                frame[key]=0
                if diag:frame.loc[ix,key]=diag['counts'][:,j]
            frame.to_parquet(out/'decisions.parquet',index=False)
            pd.DataFrame(env.trades).to_csv(out/'trades.csv',index=False)
            pd.DataFrame(state.series_history).to_csv(out/'series.csv',index=False)
            save_json(out/'metrics.json',metrics)
            # Exact quote trace for first-hit/barrier/fee audit, not rounded M1 equity.
            np.save(out/'quote_equity.npy',env.trace)
            np.save(out/'quote_series_floor.npy',env.floor_trace)
            np.save(out/'quote_series_id.npy',env.trace_series_id)
            save_json(out/'risk_contract.json',risk_contract())
            if diag:
                frame['previous_M1_attention_mass']=np.nan
                if diag['counts'].shape[1]>3:frame.loc[ix,'previous_M1_attention_mass']=diag['counts'][:,3]
                frame.to_parquet(out/'decisions.parquet',index=False)
                np.savez_compressed(out/'memory_attention.npz',indices=ix,**diag)
            save_json(out/'lineage.json',dict(day_manifest_sha256=sha(d.path/'manifest.json'),day_path=str(d.path.resolve()),
                ablation=ablation,attention='router/attention diagnostics are not causal explanations'))
    all_rewards=np.concatenate([b['reward'] for b in batches]) if batches else np.empty(0)
    all_values=np.concatenate([b['value'] for b in batches]) if batches else np.empty(0)
    adv,ret=prefix_advantages(all_rewards,all_values,bootstrap,c.gamma,c.gae_lambda)
    cursor=0
    for b in batches:
        size=len(b['action']);b['advantage']=adv[cursor:cursor+size];b['returns']=ret[cursor:cursor+size];cursor+=size
    metric=population_metrics([p.day for p in prepared],state,daily);metric['days']=daily
    metric['reward_score']=float(observed_reward);metric['learning_prefix_reward']=float(sum(all_rewards)/100.)
    metric['learning_prefix_steps']=len(all_rewards);metric['learning_bootstrap']=bootstrap
    return batches,metric


def optimize(model,optimizer,batches,c):
    """PPO updates only at executable choices; outcomes still include held intervals."""
    batches=[b for b in batches if b['mask'][:,1:].any()]
    if not batches:return dict(loss=0.,actor_loss=0.,critic_loss=0.,entropy=0.,updates=0)
    a=np.concatenate([b['advantage'][b['mask'][:,1:].any(1)] for b in batches]);center=float(a.mean());std=max(float(a.std()),1e-6)
    device=next(model.parameters()).device;stats=[];scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    model.train()
    for repeat in range(c.ppo_passes):
        optimizer.zero_grad(set_to_none=True)
        for b in batches:
            ix=np.flatnonzero(b['mask'][:,1:].any(1))
            cv=lambda name:torch.from_numpy(b[name][ix]).to(device)
            with torch.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
                h=encode(model,b['prepared'],ix)
                logits=model.head(h)+model.account_head(cv('account'))
                value=(model.critic(h)+model.account_critic(cv('account'))).squeeze(-1)
            joint,*_=distribution(logits.float(),cv('mask'));dist=torch.distributions.Categorical(probs=joint)
            lp=dist.log_prob(cv('action'));ratio=(lp-cv('logp')).exp();adv=(cv('advantage')-center)/std
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-c.clip_ratio,1+c.clip_ratio)*adv).mean()
            critic=.5*(value.float()-cv('returns')).square().mean();entropy=dist.entropy().mean()
            loss=actor+.5*critic-c.entropy_weight*entropy
            scaler.scale(loss/len(batches)).backward()
            stats.append([float(x.detach()) for x in (loss,actor,critic,entropy)])
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),c.gradient_clip)
        scaler.step(optimizer);scaler.update()
    return dict(zip(('loss','actor_loss','critic_loss','entropy'),np.mean(stats,axis=0).tolist()),updates=c.ppo_passes)


def grouped_weeks(prepared):
    weeks={}
    for p in prepared:weeks.setdefault(str(pd.Timestamp(p.day.meta['date']).to_period('W-SUN')),[]).append(p)
    return list(weeks.values())


def run(data,report,epochs=8,seeds=(61018,61019),variants=('baseline','mixed_local','mixed_full')):
    start=time.perf_counter();root=Path(report);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.76)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c=Contract(initial_capital=200.,daily_drawdown_stop=0.,ppo_passes=2,epochs=epochs,learning_rate=.0003)
    days=[Day(p) for p in sorted(Path(data).glob('20*')) if p.is_dir()]
    train=[d for d in days if d.meta['date']<='2026-05-08'];val=[d for d in days if '2026-05-11'<=d.meta['date']<='2026-05-15']
    test=[d for d in days if d.meta['date']>='2026-05-18']
    scale=Standardizer().fit(train);scale.save(root/'normalization.json')
    tr=[PreparedDay(d,scale) for d in train];va=[PreparedDay(d,scale) for d in val]
    # Test values are normalized/evaluated only after all selection is frozen.
    contract=dict(version='MIXED_PLAYER_SERIES_V1',risk_contract=risk_contract(),account_dim=SurvivalGym.account_dim,
        data_manifest_sha256=sha(Path(data)/'manifest.json'),
        split={k:[d.meta['date'] for d in group] for k,group in [('train',train),('validation',val),('evaluation',test)]},
        epochs=epochs,seeds=list(seeds),variants=list(variants),contract=c.dict(),
        population='one 200USD account per training calendar week; cash AND unfinished series carry across days; val/test one week each',
        objective='log(E/initial)+0.25*(survived eligible time / scheduled time - 1)-0.25*max_series_loss_fraction; WAIT=0',
        training_reward='100 times increments of log equity minus increments of historical maximum series loss fraction; remaining-life penalty once at series death; telescopes to100*score for complete populations',
        selection='highest complete validation score, earliest epoch on ties; no seed selection on evaluation; fixed final epoch for diagnostic audit if no complete validation; WAIT deployment if score<=0 or incomplete',
        censor_training='stop population at first support censor; remove uncertain trade from its entry origin onward; bootstrap clean prefix at pre-entry critic; never treat unknown outcome as zero loss',
        evaluation_status='chronological development evaluation; dates exposed to earlier descriptive research',
        model_memory='900s fast LSTM32; up to2760 causal M1 LSTM48 states with explicit prefix attention retrieval;38 anchorfields;attention up to256causal events28',
        revision='User corrected risk semantics: loss MORE THAN half of capital at START OF MARTINGALE SERIES; fixed initial100USD floor was wrong. This rerun corrects risk, account observations, reward, series sizing, and reports. Dates were previously exposed.',
        ablations='frozen intervention diagnostics, not separately tuned models or causal proof',
        execution='conditional GC quote proxy1oz, bid/ask, .30USD/oz roundtrip, no broker margin model',
        daily_stop='disabled; series base is immutable until net recovery and persists across session/day boundaries',
        source_hashes={str(Path(p)):sha(p) for p in (
            'spectrum/player_v3/data.py','spectrum/player_v3/experiment.py','spectrum/player_v3/gym.py','spectrum/player_v3/policy.py',
            'spectrum/player/contract.py','spectrum/player_v2/policy.py','spectrum/core/storage.py','spectrum/ingest.py')})
    import shutil
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name,digest in contract['source_hashes'].items():
        destination=snapshot/Path(name);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,destination)
        if sha(destination)!=digest:raise ValueError('Source changed before training: '+name)
    save_json(root/'contract.json',contract);hist=[];selected=[]
    for variant in variants:
        for seed in seeds:
            seed_all(seed);model=make_model(variant).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate)
            rng=np.random.default_rng(seed);best=-np.inf;folder=root/'models'/f'{variant}_s{seed}';folder.mkdir(parents=True)
            torch.save(model.state_dict(),folder/'untrained.pt')
            for epoch in range(1,epochs+1):
                t0=time.perf_counter();batches=[];metrics=[]
                for week in grouped_weeks(tr):
                    b,m=rollout(model,week,c,rng=rng);metrics.append(m)
                    batches.extend(b)
                stats=optimize(model,optimizer,batches,c)
                _,v=rollout(model,va,c)
                score=v['score'] if v['score'] is not None else -np.inf
                row=dict(variant=variant,seed=seed,epoch=epoch,
                    train_score=float(np.mean([m['score'] for m in metrics])) if all(m['complete'] for m in metrics) else None,
                    train_clean_prefix_reward=float(np.mean([m['learning_prefix_reward'] for m in metrics])),
                    train_equity=float(np.mean([m['final_equity'] for m in metrics])),train_ruins=sum(m['ruined'] for m in metrics),
                    train_complete=sum(m['complete'] for m in metrics),train_trades=sum(m['trades'] for m in metrics),
                    val_score=v['score'],val_equity=v['final_equity'],val_trades=v['trades'],val_ruined=v['ruined'],val_complete=v['complete'],
                    seconds=time.perf_counter()-t0,**stats)
                hist.append(row);pd.DataFrame(hist).to_csv(root/'history.csv',index=False)
                save_json(folder/f'epoch_{epoch:02d}.json',dict(train=metrics,validation=v,optimization=stats))
                torch.save(model.state_dict(),folder/f'epoch_{epoch:02d}.pt')
                if score>best:
                    best=score;selected_epoch=epoch;torch.save(model.state_dict(),folder/'selected.pt');selected_val=v
                print(json.dumps(row,allow_nan=False),flush=True)
            if not np.isfinite(best):
                selected_epoch=epochs;selected_val=v;torch.save(model.state_dict(),folder/'selected.pt')
            selected.append(dict(variant=variant,seed=seed,selected_epoch=selected_epoch,validation=selected_val,
                checkpoint=str(folder/'selected.pt'),checkpoint_sha256=sha(folder/'selected.pt'),parameters=sum(p.numel() for p in model.parameters()),
                status='COMPLETE_VALIDATION' if np.isfinite(best) else 'FINAL_EPOCH_DIAGNOSTIC_ONLY',
                deployment='candidate' if best>0 else 'WAIT'))
            del model,optimizer;torch.cuda.empty_cache()
    save_json(root/'freeze.json',dict(models=selected,history_sha256=sha(root/'history.csv'),contract_sha256=sha(root/'contract.json')))
    te=[PreparedDay(d,scale) for d in test]
    for item in selected:
        if 'checkpoint' not in item:continue
        model=make_model(item['variant']).to(device);model.load_state_dict(torch.load(item['checkpoint'],weights_only=True,map_location=device))
        _,item['evaluation']=rollout(model,te,c,record_dir=root/'replays'/f"{item['variant']}_s{item['seed']}")
        item['ablations']={}
        if item['variant']=='mixed_full':
            for ablation in ('fast','previous','anchors','events'):
                label='previous_M1' if ablation=='previous' else ablation
                _,m=rollout(model,te,c,ablation=ablation,record_dir=root/'replays'/f"{item['variant']}_s{item['seed']}_without_{label}")
                item['ablations'][label]=m
        item['stochastic_evaluation']=[]
        for simseed in (101,202,303):
            _,m=rollout(model,te,c,rng=np.random.default_rng(simseed));item['stochastic_evaluation'].append(dict(simulation_seed=simseed,**m))
        model.load_state_dict(torch.load(Path(item['checkpoint']).parent/'untrained.pt',weights_only=True,map_location=device))
        _,item['untrained_evaluation']=rollout(model,te,c)
        del model;torch.cuda.empty_cache()
    result=dict(status='COMPLETE',risk_contract=risk_contract(),models=selected,elapsed_seconds=time.perf_counter()-start,
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
        contract_sha256=sha(root/'contract.json'),freeze_sha256=sha(root/'freeze.json'),
        wait_baseline=dict(initial_capital=200.,final_equity=200.,net=0.,score=0.,trades=0,survival_fraction=1.),
        admission='Research comparison only; profits, if any, are not independent deployment evidence')
    save_json(root/'summary.json',result)
    print(json.dumps(dict(status='COMPLETE',report=str(root),seconds=result['elapsed_seconds'])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/research/player_v3/memory_001');p.add_argument('--report',required=True)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--seeds',type=int,nargs='+',default=[61018,61019]);p.add_argument('--variants',nargs='+',default=['baseline','mixed_local','mixed_full'])
    a=p.parse_args();run(a.data,a.report,a.epochs,tuple(a.seeds),tuple(a.variants))

if __name__=='__main__':main()
