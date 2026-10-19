"""Reproducible strategy experiments with a frozen chronological holdout."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import time
import numpy as np
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from spectrum.player.contract import Contract
from spectrum.player_v3.gym import evaluate_policy, risk_contract
from spectrum.player_v3.martingale import MartingaleConfig, MartingalePolicy, cycle_rows
from .data import MonthDay

DEFAULT_DATA='data/derived/playground_august_1500'
DEFAULT_OUTPUT='reports/playground_august_1500'


def contract():
    return Contract(initial_capital=1500.,size_oz=1.,fee_roundtrip=.30,daily_drawdown_stop=float('inf'))


def source_identity():
    return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('data.py'),
        Path(__file__).parents[1]/'player_v3/gym.py',Path(__file__).parents[1]/'player_v3/martingale.py',
        Path(__file__).parents[1]/'player_v3/data.py',Path(__file__).parents[1]/'lab.py',
        Path(__file__).parents[1]/'plugins/volatility_vwap.py',
        Path(__file__).parents[1]/'player/contract.py']}


def days_for(data,split):
    manifest=read_json(Path(data)/'manifest.json')
    if manifest['status']!='COMPLETE':raise ValueError('Month is not ready')
    if split not in (*manifest['split'],'all'):raise ValueError('Unknown split')
    dates=[r['date'] for r in manifest['days']] if split=='all' else manifest['split'][split]
    return [MonthDay(Path(data)/d) for d in dates]


def run(data=DEFAULT_DATA,output=DEFAULT_OUTPUT,config=None,split='train',control='martingale',frozen=None):
    if control not in ('martingale','fixed','wait'):raise ValueError('Unknown control')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    config=config or MartingaleConfig()
    if split in ('holdout','all'):
        if frozen is None:raise ValueError('Holdout requires a frozen recipe')
        f=read_json(frozen)
        if f['config']!=config.dict() or f['data_sha256']!=sha(Path(data)/'manifest.json') or f['source_identity']!=source_identity():
            raise ValueError('Frozen recipe drift')
    started=time.perf_counter();days=days_for(data,split)
    policy=(lambda e:0) if control=='wait' else MartingalePolicy(replace(config,multiplier=1.,cap_oz=config.base_oz) if control=='fixed' else config)
    result=evaluate_policy(days,policy,contract(),record=False,record_decisions=True)
    # Audit each realized fill directly against its immutable quote row and fees.
    lookup={d.meta['date']:d for d in days};checks=0
    for trade in result['trade_rows']:
        d=lookup[trade['date']];i,j=trade['entry_index'],trade['exit_index'];side=trade['side']
        entry=int(d.ask[i] if side>0 else d.bid[i]);exit=int(d.bid[j] if side>0 else d.ask[j])
        expected=side*(exit-entry)/1e9*trade['size_oz']-.30*trade['size_oz']
        if abs(expected-trade['net'])>1e-7 or entry!=trade['entry_nano'] or exit!=trade['exit_nano']:
            raise ValueError('Fill/cost replay mismatch')
        if not int(d.meta['london_ns'])<=trade['entry_ns']<int(d.meta['ny_close_ns']):raise ValueError('Entry outside window')
        if abs(trade['series_floor']-.5*trade['series_start_equity'])>1e-8:raise ValueError('Series base/floor mismatch')
        checks+=1
    decisions=[dict(date=r['date'],t_ns=str(r['t_ns']),action=r['action'],size_oz=r['size_oz'],equity=r['equity'],
        series_base=r['risk_series_base_after'],series_floor=r['risk_series_floor_after'],series_loss=r['risk_series_loss_fraction_after']) for r in result.pop('decisions')]
    trades=result.pop('trade_rows');series=cycle_rows(trades,result['series_rows'])
    label=time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)+'_'+split+'_'+control
    dest=root/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
    record=dict(id=label,split=split,control=control,config=config.dict(),contract={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in contract().dict().items()},
        risk_contract=risk_contract(),source_identity=source_identity(),data_sha256=sha(Path(data)/'manifest.json'),
        result=result,seconds=time.perf_counter()-started,checks=dict(fill_rows=checks,status='PASS'),decisions=decisions,trades=trades,series=series)
    save_json(dest/'result.json',record)
    save_json(root/'latest.json',dict(id=label,path=str(dest/'result.json')))
    return record


def freeze(data,output,config):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    recipe=dict(status='FROZEN_BEFORE_HOLDOUT',config=config.dict(),initial_capital=1500.,data_sha256=sha(Path(data)/'manifest.json'),source_identity=source_identity())
    ident=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:16]
    path=root/('frozen_'+ident+'.json')
    if not path.exists():save_json(path,recipe)
    return str(path)


def baseline(data=DEFAULT_DATA,output=DEFAULT_OUTPUT):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    config=MartingaleConfig()
    # Prespecified baseline, not a parameter search against held-out outcomes.
    frozen=freeze(data,output,config)
    results=[]
    for split in ('train','validation','holdout'):
        for control in ('wait','fixed','martingale'):
            r=run(data,output,config,split,control,frozen if split=='holdout' else None)
            results.append(dict(id=r['id'],split=split,control=control,metrics=r['result'],checks=r['checks']))
            print(json.dumps(dict(split=split,control=control,complete=r['result']['complete'],trades=r['result']['trades'],equity=r['result']['final_equity'])),flush=True)
    receipt=dict(status='COMPLETE',initial_capital=1500.,frozen=frozen,config=config.dict(),runs=results,
        interpretation='Fixed baseline and matched controls, not trained or optimized policy. Censored populations have no valid complete score.')
    save_json(root/'baseline.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--baseline',action='store_true')
    p.add_argument('--split',choices=['train','validation','holdout','all'],default='train');p.add_argument('--mode',choices=['trend','fade'],default='trend');p.add_argument('--plan',type=int,default=0)
    p.add_argument('--multiplier',type=float,default=1.5);p.add_argument('--cap',type=float,default=4.);p.add_argument('--threshold',type=float,default=2.);p.add_argument('--freeze',action='store_true')
    a=p.parse_args();cfg=MartingaleConfig(mode=a.mode,plan=a.plan,multiplier=a.multiplier,cap_oz=a.cap,threshold_ticks=a.threshold)
    if a.baseline:baseline(a.data,a.output)
    else:
        frozen=freeze(a.data,a.output,cfg) if a.freeze else None
        r=run(a.data,a.output,cfg,a.split,frozen=frozen);print(json.dumps(dict(id=r['id'],result=r['result']),indent=2))
