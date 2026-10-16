"""Controlled replay of immutable saved checkpoints or a manual action file.

Examples:
 python -m scripts.player_replay --run run_001 --week 1 --policy greedy
 python -m scripts.player_replay --run run_001 --week 1 --policy sampled --seed 42
 python -m scripts.player_replay --run run_001 --week 1 --actions actions.csv

Manual CSV columns: date,decision_index,action. Missing rows mean WAIT.
Actions: 0 WAIT; 1..3 LONG fast/medium/wide; 4..6 SHORT fast/medium/wide.
Outputs are diagnostics and never feed checkpoint selection.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from spectrum.player.contract import Contract
from spectrum.player.data import Day
from spectrum.player.model import Player,Scaler
from spectrum.player.gym import TradingGym
from spectrum.player.experiment import rollout,save_replay
from spectrum.core.storage import sha
from spectrum.ingest import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',default='.');p.add_argument('--run',default='run_001');p.add_argument('--week',type=int,choices=[1,2],default=1)
    p.add_argument('--policy',choices=['greedy','sampled','wait'],default='greedy');p.add_argument('--seed',type=int,default=61015)
    p.add_argument('--checkpoint',choices=['selected','untrained'],default='selected');p.add_argument('--actions',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args();root=Path(a.root).resolve()
    source=root/'data/research/player_v1'/a.run;freeze=json.loads((source/'frozen_before_evaluation.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text());c=Contract(**protocol['contract'])
    if sha(source/'selected.pt')!=freeze['selected_sha256'] or sha(source/'scaler.npz')!=freeze['scaler_sha256']:raise ValueError('Frozen model/scaler drift')
    for name,digest in freeze['code_sha256'].items():
        if sha(root/'spectrum/player'/name)!=digest:raise ValueError(f'Executor/code drift: {name}; use frozen source snapshot or a new declared experiment')
    out=a.output or source/'diagnostics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted((root/f'data/derived/player_v1/week{a.week}').glob('2026*'))
    if len(paths)!=5:raise ValueError('Prepared week missing')
    torch.set_num_threads(4);model=Player(71,c.hidden_size);model.load_state_dict(torch.load(source/(a.checkpoint+'.pt'),weights_only=True,map_location='cpu'))
    scaler=Scaler.load(source/'scaler.npz');rng=np.random.default_rng(a.seed) if a.policy=='sampled' else None
    manual=pd.read_csv(a.actions,dtype={'date':str}) if a.actions else None
    if manual is not None:
        if not {'date','decision_index','action'}.issubset(manual):raise ValueError('Manual CSV requires date,decision_index,action')
        if manual.duplicated(['date','decision_index']).any():raise ValueError('Duplicate manual decisions')
        if not set(manual.date).issubset({x.name for x in paths}):raise ValueError('Manual date outside selected week')
    metrics=[];start=time.perf_counter()
    for path in paths:
        day=Day(path)
        if manual is None:b,m,e=rollout(model,scaler,day,c,rng,True,a.policy=='wait')
        else:
            commands=manual[manual.date==path.name];actions=np.zeros(len(day.x),np.int64)
            for row in commands.itertuples():
                if int(row.decision_index)!=row.decision_index or not 0<=row.decision_index<len(day.x):raise ValueError('Manual index out of range')
                if int(row.action)!=row.action:raise ValueError('Action must be integer')
                actions[int(row.decision_index)]=int(row.action)
            e=TradingGym(day,c,True);accounts=[];masks=[];rewards=[]
            for i,action in enumerate(actions):
                accounts.append(e.account());masks.append(e.mask());rewards.append(e.step(action)[1]/c.reward_scale)
            b=dict(action=actions,account=np.asarray(accounts),mask=np.asarray(masks),reward=np.asarray(rewards),
                   logp=np.full(len(actions),np.nan),value=np.full(len(actions),np.nan),probability=np.full((len(actions),7),np.nan))
            m=e.metrics();m.update(date=path.name,manual=True)
        save_replay(out,'replay',day,b,m,e);metrics.append(m);print(json.dumps(m),flush=True)
    save_json(out/'diagnostic_manifest.json',dict(source_run=a.run,checkpoint=a.checkpoint,checkpoint_sha256=sha(source/(a.checkpoint+'.pt')),
              week=a.week,policy='manual' if manual is not None else a.policy,seed=a.seed,seconds=time.perf_counter()-start,
              manual_sha256=sha(a.actions) if a.actions else None,days=metrics,
              status='POST_FREEZE_DIAGNOSTIC_NOT_SELECTION',test_exposure='Any week-two inspection is research exposure; never re-label it blind.'))
    print('Saved: '+str(out))

if __name__=='__main__':main()
