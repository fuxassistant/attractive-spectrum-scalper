"""Reproducible two-day expert selection on ten previously inspected days."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
from pathlib import Path
import time
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import digest, dump, checked
from experiments.mechanical_lab.portfolio import prepare_outcomes
from .engine import (CATALOG, NS, build_masks, offered, replay, pair_record,
                     select_pair, profiles, acceptance_rates, stable_uniform)

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def load_sources(config):
    atlas=ROOT/config['atlas'];old=ROOT/config['execution_source']
    am=json.loads((atlas/'manifest.json').read_text());em=json.loads((old/'manifest.json').read_text())
    hashes={str(atlas/'manifest.json'):digest(atlas/'manifest.json'),str(old/'manifest.json'):digest(old/'manifest.json')}
    for name in ('observations.parquet','thresholds.json'):
        hashes[str(atlas/name)]=checked(atlas/name,am['artifacts'][name])
    f=pd.read_parquet(atlas/'observations.parquet')
    init=f[f.date.lt(config['dates'][0])].copy()
    # The historical ten-day prefix is explicit, even if extra older data later appears.
    init=init[init.date.between('2026-08-03','2026-08-14')]
    threshold=float(init.x_base_mbo_now_trade_total_log.median())
    f=f[f.date.isin(config['dates'])].copy().reset_index(drop=True)
    if sorted(f.date.unique())!=config['dates']:raise ValueError('Ten-day source coverage changed')
    if f.candidate_id.duplicated().any():raise ValueError('Candidate IDs must be unique')
    for col in ('target_available_ns','episode_available_ns','memory_available_ns','bar_context_available_ns','past_max_available_ns','mbometa_now_max_source_available_ns'):
        if (f[col]>f.t_ns).any():raise AssertionError(f'Future feature clock: {col}')
    # Config equality is a prerequisite for reusing execution consequences.
    source_config=json.loads((old/'config.json').read_text()) if (old/'config.json').exists() else json.loads((ROOT/'experiments/zone_episode_lab/config.json').read_text())
    if source_config['execution']!=config['execution'] or source_config['stress']!=config['stress'] or source_config['holding_seconds']!=300:
        raise ValueError('Execution source/config mismatch')
    hashes[str(ROOT/'experiments/zone_episode_lab/config.json')]=checked(ROOT/'experiments/zone_episode_lab/config.json',em['config_sha256'])
    receipts={r['date']:r for r in em['receipts']};tables={'base':[],'stress':[]};source_days=[]
    for date in config['dates']:
        d=old/'days'/date;receipt=json.loads((d/'receipt.json').read_text())
        if receipt!=receipts[date]:raise ValueError('Day receipt differs from assembled manifest')
        hashes[str(d/'receipt.json')]=digest(d/'receipt.json')
        ids=f.loc[f.date.eq(date),'candidate_id']
        for name in ('candidates','outcomes','stress_outcomes'):
            p=d/(name+'.parquet');hashes[str(p)]=checked(p,receipt['artifact_sha256'][p.name])
            table=pd.read_parquet(p);table=table[table.candidate_id.isin(ids)].copy()
            if name=='candidates':
                a=f[f.date.eq(date)].set_index('candidate_id').sort_index();b=table.set_index('candidate_id').sort_index()
                for field in ('origin_index','t_ns','side','target_nano','risk_per_oz','zone_low_nano','zone_high_nano'):
                    if not a[field].equals(b[field]):raise AssertionError(f'Candidate identity mismatch: {date} {field}')
            else:
                if len(table)!=2*len(ids) or table.duplicated(['candidate_id','side']).any():raise ValueError('Missing/duplicate paired outcome')
                if not table.plan.eq('zone_target_300s').all():raise ValueError('Wrong exit plan')
                tables['base' if name=='outcomes' else 'stress'].append(table)
        source_days.append(dict(date=date,source=receipt['source'],source_manifest_sha256=receipt['source_manifest_sha256'],source_payloads=receipt['source_payloads']))
    return f,threshold,len(init),{k:pd.concat(v,ignore_index=True) for k,v in tables.items()},hashes,source_days


def verify_native(f, tables, source_days, config, hashes):
    """Recompute every selected LONG/SHORT outcome from verified native quotes."""
    from experiments.mechanical_lab.data import source_for
    from experiments.mechanical_lab.execution import QuoteTape
    from experiments.zone_episode_lab.game import simulate_zone_pairs
    results=[]
    for r in source_days:
        date=r['date'];market=Path(r['source']);mp=market/'manifest.json'
        hashes[str(mp)]=checked(mp,r['source_manifest_sha256']);mm=json.loads(mp.read_text())
        qp=market/'quotes.parquet';hashes[str(qp)]=checked(qp,r['source_payloads']['quotes.parquet'])
        q=pd.read_parquet(qp);selection=int(source_for(date,[1,41271,22])['entry']['selection']['selection_available_ns'])
        day=int(pd.Timestamp(date,tz='UTC').value)
        tape=QuoteTape.from_frame(q,coverage_end_ns=int(mm['input']['end_ns'])+1,
              entry_start_ns=max(selection,day+7*3600*NS),deadline_ns=day+(20*3600+30*60)*NS)
        cf=f[f.date.eq(date)][['candidate_id','origin_index','t_ns','side','target_nano','risk_per_oz']]
        for condition in ('base','stress'):
            ec={**config['execution'],**(config['stress'] if condition=='stress' else {})}
            actual=simulate_zone_pairs(tape,cf,holding_seconds=300,config=ec).set_index(['candidate_id','side']).sort_index()
            expected=tables[condition][tables[condition].date.eq(date)].set_index(['candidate_id','side']).sort_index()
            cols=['status','reason','entry_ns','trigger_ns','exit_ns','net_per_oz','mae_per_oz','mfe_per_oz','max_drawdown_per_oz','target_nano']
            for col in cols:
                if not actual[col].equals(expected[col]):
                    if actual[col].dtype.kind=='f' and np.allclose(actual[col],expected[col],rtol=0,atol=1e-12,equal_nan=True):continue
                    raise AssertionError(f'Native replay mismatch {date} {condition} {col}')
            results.append(dict(date=date,condition=condition,rows=len(actual),fields=len(cols),exact=True))
        print(json.dumps(dict(stage='native_verified',date=date,candidates=len(cf))),flush=True)
    return results


def evaluate(f,tables,config,threshold):
    masks=build_masks(f,config,threshold);lookups={k:prepare_outcomes(v) for k,v in tables.items()}
    dates=config['dates'];daily=[];pair_scores=[];selections=[];profile_rows=[];decisions=[]
    baskets={k:[] for k in ('adaptive','frozen_first_pair','frozen_first_raw_pair','raw_leader','always_toward','always_away','WAIT')}
    random_baskets={(ref,seed):[] for ref in ('adaptive','raw_leader') for seed in range(1,config['controls']['random_seeds']+1)}
    first_selected=None;first_raw=None
    for date in dates:
        day=f[f.date.eq(date)]
        for key in CATALOG:
            c=offered(day,masks.loc[day.index],key)
            for condition,lookup in lookups.items():
                s,_=replay(c,lookup,config)
                daily.append(dict(date=date,strategy=key,condition=condition,**{**s,'offered':len(c)}))
    for i in range(1,len(dates)):
        pair=dates[i-1:i+1];history=f[f.date.isin(pair)];hm=masks.loc[history.index]
        # Midnight after second day: same information cutoff for latest diagnostic pair.
        cutoff=int((pd.Timestamp(pair[-1],tz='UTC')+pd.Timedelta(days=1)).value)
        records=[pair_record(k,pair,offered(history,hm,k),lookups,config) for k in CATALOG]
        choice,scores=select_pair(records,config)
        for condition in ('base','stress'):
            if (scores[condition+'_max_feedback_ns']>=cutoff).any():raise AssertionError('Unmatured pair feedback')
        pair_scores.append(scores);key=choice['selected_strategy'];raw=choice['raw_leader']
        if first_selected is None:
            first_selected=key
            first_raw=raw
        chosen=offered(history,hm,key);rr=scores.set_index('strategy').loc[key]
        rawchosen=offered(history,hm,raw);rawrow=scores.set_index('strategy').loc[raw]
        reverse={condition:replay(chosen,lookup,config,direction='reverse')[0] for condition,lookup in lookups.items()}
        next_date=dates[i+1] if i+1<len(dates) else ''
        row=dict(pair_start=pair[0],pair_end=pair[1],next_date=next_date,selection_ns=cutoff,
                 **choice,selected_pair_base_net_usd=rr.base_net_usd,selected_pair_stress_net_usd=rr.stress_net_usd,
                 selected_pair_offered=len(chosen),selected_pair_long_fraction=float(chosen.side.eq(1).mean()) if len(chosen) else None,
                 reverse_pair_base_net_usd=reverse['base']['net_usd'],reverse_pair_stress_net_usd=reverse['stress']['net_usd'],
                 reverse_pair_complete=reverse['base']['capital_curve_complete'] and reverse['stress']['capital_curve_complete'],
                 reverse_pair_base_complete=reverse['base']['capital_curve_complete'],reverse_pair_stress_complete=reverse['stress']['capital_curve_complete'],
                 raw_pair_base_net_usd=rawrow.base_net_usd,raw_pair_stress_net_usd=rawrow.stress_net_usd,raw_pair_offered=len(rawchosen),
                 raw_pair_base_complete=rawrow.base_complete,raw_pair_stress_complete=rawrow.stress_complete,
                 raw_interpretation=('Лидер сравнения получил положительный суммарный базовый результат. Проверка второго дня, stress и поддержки остаётся отдельным условием.' if rawrow.base_net_usd>0 else 'Лидер сравнения не получил положительного суммарного базового результата. Его выбор означает меньший убыток среди поддержанных вариантов, а не вознаграждение этой стратегии рынком.'),
                 raw_next_offered=0,raw_next_base_net_usd=None,raw_next_stress_net_usd=None,
                 raw_next_base_trades=0,raw_next_stress_trades=0,raw_next_base_complete=None,raw_next_stress_complete=None,
                 interpretation=CATALOG[key]['meaning'],next_offered=0,next_base_net_usd=None,next_stress_net_usd=None,
                 next_base_trades=0,next_stress_trades=0,next_base_complete=None,next_stress_complete=None)
        for ref,c in [('adaptive',chosen),('raw_leader',rawchosen)]:
            profile_rows.extend([{**p,'reference_policy':ref} for p in profiles(history,c,pair_start=pair[0],pair_end=pair[1],period='pair')])
        if next_date:
            day=f[f.date.eq(next_date)];dm=masks.loc[day.index]
            if not (day.t_ns>cutoff).all():raise AssertionError('Selection was not before deployment')
            choices={'adaptive':key,'frozen_first_pair':first_selected,'frozen_first_raw_pair':first_raw,'raw_leader':raw,'always_toward':'toward_all','always_away':'away_all','WAIT':'WAIT'}
            for policy,k in choices.items():baskets[policy].append(offered(day,dm,k))
            selected=baskets['adaptive'][-1];row['next_offered']=len(selected)
            for condition,lookup in lookups.items():
                s,_=replay(selected,lookup,config)
                row['next_'+condition+'_net_usd']=s['net_usd'];row['next_'+condition+'_trades']=s['trades'];row['next_'+condition+'_complete']=s['capital_curve_complete']
            rawselected=baskets['raw_leader'][-1];row['raw_next_offered']=len(rawselected)
            for condition,lookup in lookups.items():
                s,_=replay(rawselected,lookup,config)
                row['raw_next_'+condition+'_net_usd']=s['net_usd'];row['raw_next_'+condition+'_trades']=s['trades'];row['raw_next_'+condition+'_complete']=s['capital_curve_complete']
            for ref,c in [('adaptive',selected),('raw_leader',rawselected)]:
                profile_rows.extend([{**p,'reference_policy':ref} for p in profiles(day,c,pair_start=pair[0],pair_end=pair[1],period='next')])
            receipt=day[['candidate_id','t_ns','date','side']].copy();receipt['selection_ns']=cutoff
            receipt['selected_strategy']=key;receipt['accepted']=dm[key].to_numpy();receipt['chosen_side']=day.side.to_numpy()*CATALOG[key]['direction']
            receipt['pair_start']=pair[0];receipt['pair_end']=pair[1];decisions.append(receipt)
            for ref,k in [('adaptive',key),('raw_leader',raw)]:
                rate,mapping=acceptance_rates(history,hm[k],config['controls']['strata'],config['controls']['acceptance_prior_count'])
                probs=np.array([mapping.get(tuple(v),rate) for v in day[config['controls']['strata']].to_numpy()])
                for seed in range(1,config['controls']['random_seeds']+1):
                    accepted=stable_uniform(day.candidate_id,seed)<probs
                    random=day.loc[accepted].drop(columns=['deadline_ns'],errors='ignore').copy()
                    random['original_signal_side']=random.side;random['side']*=CATALOG[k]['direction'];random['selected_strategy']=k
                    random_baskets[(ref,seed)].append(random)
        selections.append(row)
    results=[];logs=[];assembled={}
    for policy,parts in baskets.items():
        c=pd.concat(parts,ignore_index=True).sort_values('t_ns');assembled[policy]=c
        for condition,lookup in lookups.items():
            s,log=replay(c,lookup,config);results.append(dict(policy=policy,condition=condition,**s))
            if len(log):log['policy']=policy;log['condition']=condition;logs.append(log)
    for ref in ('adaptive','raw_leader'):
        for condition,lookup in lookups.items():
            s,log=replay(assembled[ref],lookup,config,direction='reverse')
            results.append(dict(policy='reverse_'+ref,condition=condition,**s))
            if len(log):log['policy']='reverse_'+ref;log['condition']=condition;logs.append(log)
    controls=[]
    for (ref,seed),parts in random_baskets.items():
        random=pd.concat(parts,ignore_index=True).sort_values('t_ns')
        for condition,lookup in lookups.items():
            for kind,c,direction in [('matched_activity',random,'signal'),('coin_direction',assembled[ref],'coin')]:
                s,_=replay(c,lookup,config,direction=direction,seed=seed)
                controls.append(dict(reference_policy=ref,kind=kind,condition=condition,seed=seed,**s))
    return dict(masks=masks,daily=pd.DataFrame(daily),pairs=pd.concat(pair_scores,ignore_index=True),
                selections=pd.DataFrame(selections),profiles=pd.DataFrame(profile_rows),
                decisions=pd.concat(decisions,ignore_index=True),results=results,
                trades=pd.concat(logs,ignore_index=True) if logs else pd.DataFrame(),controls=pd.DataFrame(controls))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    config=json.loads((HERE/'config.json').read_text(encoding='utf-8'));dump(out/'config.json',config)
    # Freeze implementation identity before opening outcome data.
    code={str(p.relative_to(ROOT)):digest(p) for p in [HERE/'config.json',HERE/'engine.py',HERE/'run.py',ROOT/'experiments/mechanical_lab/portfolio.py',ROOT/'experiments/mechanical_lab/execution.py',ROOT/'experiments/zone_episode_lab/game.py']}
    f,threshold,ninit,tables,hashes,source_days=load_sources(config)
    native=verify_native(f,tables,source_days,config,hashes)
    result=evaluate(f,tables,config,threshold)
    f.to_parquet(out/'observations.parquet',index=False)
    masks=result['masks'].copy();masks.insert(0,'candidate_id',f.candidate_id);masks.to_parquet(out/'masks.parquet',index=False)
    for key,name in [('daily','daily_policy'),('pairs','pair_scores'),('selections','selections'),('profiles','market_profiles'),('decisions','decisions'),('trades','trades'),('controls','random_controls')]:
        result[key].to_parquet(out/(name+'.parquet'),index=False)
    for condition,table in tables.items():table.to_parquet(out/(condition+'_outcomes.parquet'),index=False)
    controls=[]
    for (ref,kind,condition),g in result['controls'].groupby(['reference_policy','kind','condition']):
        complete=g[g.capital_curve_complete]
        controls.append(dict(reference_policy=ref,kind=kind,condition=condition,n=len(g),complete_runs=len(complete),
                             median_usd=float(complete.net_usd.median()) if len(complete) else None,
                             p05_usd=float(complete.net_usd.quantile(.05)) if len(complete) else None,
                             p95_usd=float(complete.net_usd.quantile(.95)) if len(complete) else None))
    selection=result['selections'];forward=selection[selection.next_date.ne('')]
    summary=dict(status='COMPLETE_EXPLORATORY_TWO_DAY_STRATEGY_SELECTION',dates=config['dates'],opportunities=len(f),
        initialization_rows=ninit,activity_log_threshold=threshold,strategy_catalog=CATALOG,
        execution=config['execution'],stress=config['stress'],account=config['account'],exit_contract=config['exit_contract'],
        scope=config['scope'],untouched_holdout=False,results=result['results'],random_controls=controls,
        selection_counts=forward.selected_strategy.value_counts().to_dict(),pairs=len(selection),forward_days=len(forward),
        source_outcome_status={k:v.status.value_counts().to_dict() for k,v in tables.items()},native_rechecks=native,
        notes=['Pair/day diagnostic accounts reset; aggregate deployment uses one continuous account for all eight forward days.',
               'Pair winners are retrospective. Forward choices are frozen before the next day.',
               'Global MBO is not price-local defense. Feature contrasts include the rule definition and are not causal effects.',
               'WAIT can mean no supported candidate in this finite catalog, not absence of all possible market edge.',
               '99 random controls are descriptive and dependent; their quantiles are not multiple-testing-adjusted significance.',
               'Raw-leader profiles, random controls and frozen-first-raw-pair comparison were added as diagnostics after the first all-WAIT result; the catalog, thresholds and primary selection were unchanged.',
               'Incomplete capital curves must not be ranked as completed PnL.'],
        elapsed_seconds=time.perf_counter()-started,python=platform.python_version())
    dump(out/'summary.json',summary)
    dump(out/'manifest.json',dict(status='COMPLETE',implementation_sha256=code,source_hashes=hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        artifacts={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(output=str(out),opportunities=len(f),selection_counts=summary['selection_counts'],results=result['results']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
