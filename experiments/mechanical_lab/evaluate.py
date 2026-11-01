"""Frozen discovery/validation selection and later diagnostic comparisons."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT,dump,digest,dates_for
from .portfolio import account,AccountConfig,SIZING_CATALOG,prepare_outcomes
from .signals import SIGNAL_CATALOG,VETO_CATALOG,accept_mask
from .execution import EXIT_PLANS

def load_run(root,stress=False):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    config=json.loads((root/'frozen_config.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETE':raise ValueError('Only COMPLETE immutable runs can be evaluated')
    if digest(root/'frozen_config.json')!=manifest['config_sha256']:raise ValueError('Frozen config hash mismatch')
    for path,sha in manifest['code_identity'].items():
        if digest(root/'source_snapshot'/Path(path).name)!=sha:raise ValueError('Archived implementation hash mismatch')
        if digest(ROOT/path)!=sha:raise ValueError('Live implementation differs from frozen run; create a new run')
    expected={date for date,split in dates_for(config) if not stress or split in ('diagnostic_test','extension')}
    paths=sorted((root/'days').glob('*'))
    if {p.name for p in paths}!=expected or {d['date'] for d in manifest['days']}!=expected:
        raise ValueError('Missing or unexpected dates in frozen run')
    for p in paths:
        receipt=json.loads((p/'receipt.json').read_text(encoding='utf-8'))
        declared=next(d for d in manifest['days'] if d['date']==p.name)
        if receipt!=declared or receipt['status']!='COMPLETE':raise ValueError('Day receipt mismatch')
        files=receipt.get('artifact_sha256',{})
        if set(files)!={'features.parquet','candidates.parquet','all_signals.parquet','outcomes.parquet'}:
            raise ValueError('Incomplete output identity')
        for file,sha in files.items():
            if digest(p/file)!=sha:raise ValueError('Derived artifact hash mismatch')
    c=pd.concat([pd.read_parquet(p/'candidates.parquet') for p in paths],ignore_index=True)
    o=pd.concat([pd.read_parquet(p/'outcomes.parquet') for p in paths],ignore_index=True)
    return c,o

def evaluate(root,config,stress_root=None):
    root=Path(root)
    frozen=json.loads((root/'frozen_config.json').read_text(encoding='utf-8'))
    if config!=frozen:raise ValueError('Evaluation config must equal frozen base config')
    if stress_root and json.loads((Path(stress_root)/'frozen_config.json').read_text(encoding='utf-8'))!=frozen:
        raise ValueError('Stress run must use the same frozen experiment config')
    c,o=load_run(root)
    ac=AccountConfig(**config['account'])
    custom=config.get('recipe')
    signals=[custom['entry']['id']] if custom else [s for s in SIGNAL_CATALOG if s!='control_clock']
    vetos=[custom['veto']['id']] if custom else list(VETO_CATALOG)
    plans=[custom['exit']['id']] if custom else list(EXIT_PLANS)
    subsets={(s,v): c.loc[c.signal_id.eq(s)&accept_mask(c,v)].copy() for s,v in itertools.product(signals,vetos)}
    pairs={(p,k):prepare_outcomes(o.loc[o.plan.eq(p)&o.split.eq(k)]) for p,k in itertools.product(plans,config['splits'])}
    grid=[]
    for signal,veto,plan in itertools.product(signals,vetos,plans):
        for split in ('discovery','validation'):
            cc=subsets[(signal,veto)]
            stats,_=account(cc.loc[cc.split.eq(split)],pairs[(plan,split)],config=ac)
            grid.append(dict(signal=signal,veto=veto,plan=plan,split=split,**stats))
    grid=pd.DataFrame(grid);grid.to_parquet(root/'search.parquet',index=False)
    print(f'Search evaluated: {len(signals)*len(vetos)*len(plans)} fixed-size recipes',flush=True)
    selected=[]
    for signal in signals:
        g=grid.loc[grid.signal.eq(signal)]
        d=g.loc[g.split.eq('discovery')].set_index(['veto','plan'])
        v=g.loc[g.split.eq('validation')].set_index(['veto','plan'])
        eligible=(d.trades.ge(40)&d.active_days.ge(6)&d.capital_curve_complete
                  &v.trades.ge(20)&v.active_days.ge(3)&v.capital_curve_complete)
        qualified=v.loc[eligible]
        if len(qualified):
            best=qualified.sort_values(['net_usd','max_drawdown_usd','veto','plan'],ascending=[False,True,True,True]).iloc[0]
            veto,plan=best.name;eligible_selection=True
        else:
            # Keep sparse/failed ideas visible without calling them winners.
            veto,plan=vetos[0],('atr_rr' if 'atr_rr' in plans else plans[0]);eligible_selection=False
        selected.append(dict(signal=signal,veto=veto,plan=plan,selection_eligible=eligible_selection))
    dump(root/'selected.json',selected)
    rows=[];trade_frames=[];coins=[];regimes=[];baseline=[];day_stats=[]
    stress_c=stress_o=None
    if stress_root: stress_c,stress_o=load_run(stress_root,stress=True)
    for recipe in selected:
        signal,veto,plan=(recipe[k] for k in ('signal','veto','plan'))
        cc=subsets[(signal,veto)]
        for split in config['splits']:
            sub=cc.loc[cc.split.eq(split)];po=pairs[(plan,split)]
            sizing_options=list(SIZING_CATALOG) if split in ('diagnostic_test','extension') and not custom else list(dict.fromkeys(['fixed',custom['sizing']])) if custom else ['fixed']
            for sizing in sizing_options:
                for direction in (('signal','reverse') if sizing=='fixed' else ('signal',)):
                    stats,trades=account(sub,po,config=ac,sizing=sizing,direction=direction)
                    rows.append(dict(**recipe,split=split,sizing=sizing,direction=direction,**stats))
                    if len(trades):
                        for k,val in dict(signal=signal,veto=veto,plan=plan,split=split,sizing=sizing,direction=direction).items():trades[k]=val
                        trade_frames.append(trades)
                        if sizing=='fixed' and direction=='signal':
                            for regime,g in trades.groupby('regime'):
                                regimes.append(dict(signal=signal,split=split,regime=regime,trades=len(g),net_usd=float(g.pnl_usd.sum()),mean_usd=float(g.pnl_usd.mean()),win_rate=float(g.pnl_usd.gt(0).mean())))
                            for date,g in trades.groupby('date'):
                                day_stats.append(dict(signal=signal,split=split,date=date,trades=len(g),net_usd=float(g.pnl_usd.sum())))
            if split in ('diagnostic_test','extension'):
                for seed in range(config['coin_seeds']):
                    stats,_=account(sub,po,config=ac,direction='coin',seed=config['random_seed']+seed)
                    coins.append(dict(signal=signal,split=split,seed=seed,**stats))
                # Regular clock is a distinct timing control; it is explicitly
                # not claimed to match event counts or regimes.
                clock=c.loc[c.signal_id.eq('control_clock')&c.split.eq(split)]
                for seed in range(32):
                    stats,_=account(clock,po,config=ac,direction='coin',seed=config['random_seed']+seed)
                    baseline.append(dict(signal=signal,plan=plan,split=split,seed=seed,**stats))
            if stress_root and split in ('diagnostic_test','extension'):
                sc=stress_c.loc[stress_c.signal_id.eq(signal)&stress_c.split.eq(split)&accept_mask(stress_c,veto)]
                so=prepare_outcomes(stress_o.loc[stress_o.plan.eq(plan)&stress_o.split.eq(split)])
                stats,_=account(sc,so,config=ac)
                rows.append(dict(**recipe,split=split,sizing='fixed',direction='stress',**stats))
        print(f'Controls evaluated: {signal}',flush=True)
    results=pd.DataFrame(rows);coin=pd.DataFrame(coins)
    results.to_parquet(root/'results.parquet',index=False)
    pd.concat(trade_frames,ignore_index=True).to_parquet(root/'trades.parquet',index=False) if trade_frames else None
    coin.to_parquet(root/'coins.parquet',index=False)
    pd.DataFrame(baseline).to_parquet(root/'clock_controls.parquet',index=False)
    pd.DataFrame(regimes).to_parquet(root/'regimes.parquet',index=False)
    pd.DataFrame(day_stats).to_parquet(root/'daily.parquet',index=False)
    admission=[]
    for recipe in selected:
        s=recipe['signal'];r=results.loc[results.signal.eq(s)&results.sizing.eq('fixed')]
        vals={split:r.loc[r.split.eq(split)&r.direction.eq('signal')].iloc[0] for split in config['splits']}
        coin_test=coin.loc[coin.signal.eq(s)&coin.split.eq('diagnostic_test')]
        complete_coin=coin_test.loc[coin_test.capital_curve_complete]
        threshold=float(complete_coin.net_usd.quantile(.95)) if len(complete_coin)==config['coin_seeds'] else None
        tests={
            'selection_eligible':recipe['selection_eligible'],
            'n_and_days':all(vals[k].trades>=30 and vals[k].active_days>=4 for k in ('validation','diagnostic_test')),
            'positive_validation_and_test':all(vals[k].net_usd>0 for k in ('validation','diagnostic_test')),
            'complete_accounts':all(bool(vals[k].capital_curve_complete) for k in ('discovery','validation','diagnostic_test')),
            'beats_coin_p95':threshold is not None and vals['diagnostic_test'].net_usd>threshold,
            'drawdown':vals['diagnostic_test'].max_drawdown_usd<=ac.initial_capital_usd*.1,
        }
        sr=r.loc[r.split.eq('diagnostic_test')&r.direction.eq('stress')]
        tests['stress_positive_complete']=bool(len(sr) and sr.iloc[0].net_usd>0 and sr.iloc[0].capital_curve_complete)
        observed=vals['diagnostic_test'].net_usd
        # Descriptive randomization p-value; not adjusted for family search.
        p=float((1+(complete_coin.net_usd>=observed).sum())/(1+len(complete_coin))) if len(complete_coin)==config['coin_seeds'] else None
        admission.append(dict(**recipe,gates={k:bool(v) for k,v in tests.items()},
            research_candidate=all(tests.values()),economic_admission='NOT_ESTABLISHED',
            coin_p95_usd=threshold,coin_complete_seeds=len(complete_coin),coin_randomization_p=p,
            coin_median_usd=float(complete_coin.net_usd.median()) if len(complete_coin) else None))
    dump(root/'admission.json',admission)
    return dict(recipes=len(signals)*len(vetos)*len(plans),selected=len(selected),research_candidates=sum(x['research_candidate'] for x in admission))
