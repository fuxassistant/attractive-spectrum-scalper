"""Reconcile colleague's arithmetic and materialize an explicit target contract."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.mechanical_lab.data import dump
from experiments.continuation_lab_v1.run import load_day
from experiments.regime_language_v1.measures import TimeAtPrice
from .targets import paired_targets,decision_from_values
from .future_description import describe_future

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
NS=1_000_000_000


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def cost_decomposition(quotes,outcomes):
    """Same observed executions/exit rule, exact nano-price spread attribution.

    Removing costs here is an arithmetic diagnostic, NOT a zero-cost replay
    with reoptimized barriers or hypothetical earlier execution.
    """
    f=outcomes[outcomes.status=='COMPLETE'].copy()
    t=quotes.available_ns.to_numpy(np.int64);bid=quotes.bid_nano.to_numpy(np.int64);ask=quotes.ask_nano.to_numpy(np.int64)
    e=f.entry_ordinal.to_numpy(np.int64);x=f.exit_ordinal.to_numpy(np.int64);s=f.side.to_numpy(np.int64)
    assert np.array_equal(t[e],f.entry_ns.to_numpy(np.int64)) and np.array_equal(t[x],f.exit_ns.to_numpy(np.int64))
    entry=np.where(s==1,ask[e],bid[e]);exit_=np.where(s==1,bid[x],ask[x])
    exact_net=s*(exit_-entry)/NS-.3
    assert np.max(np.abs(exact_net-f.net_per_oz))<1e-10
    mid=s*((bid[x]+ask[x])-(bid[e]+ask[e]))/(2*NS)
    spread=((ask[e]-bid[e])+(ask[x]-bid[x]))/(2*NS)
    assert np.max(np.abs(mid-spread-.3-exact_net))<1e-10
    f['future_mid_move_same_executions']=mid
    f['future_actual_spread_cost']=spread
    f['future_actual_total_cost']=spread+.3
    return f


def pair_summary(p,*,weighted=False):
    full=p[p.future_both_filled_complete]
    weights=full.exact_decision_weight.to_numpy() if weighted else np.ones(len(full))
    avg=lambda a:float(np.average(a,weights=weights))
    L=full.future_u_long_r.to_numpy();S=full.future_u_short_r.to_numpy()
    c=(L+S)/2;d=(L-S)/2;value=decision_from_values(avg(c),avg(d))
    return dict(rows=len(full),weight_sum=float(weights.sum()),
                joint_counts={str(k):int(v) for k,v in full.future_joint_profit_state.value_counts().items()},
                mean_long_r=avg(L),mean_short_r=avg(S),mean_common_r=avg(c),mean_advantage_r=avg(d),
                hindsight_best_side_r=avg(np.maximum(L,S)),hindsight_with_wait_r=avg(np.maximum(0,np.maximum(L,S))),
                best_constant_action_with_wait_r=float(max(0,value['entry_value'])),
                mean_abs_advantage_r=avg(np.abs(d)),abs_mean_advantage_r=abs(avg(d)),
                near_equal_pairs=int(full.future_near_equal_01r.sum()),
                near_equal_states={str(k):int(v) for k,v in full[full.future_near_equal_01r].future_joint_profit_state.value_counts().items()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);begun=time.perf_counter()
    cfg=read(HERE/'config.json');parent=ROOT/cfg['parent'];manifest=read(parent/'manifest.json')
    source_files=list(HERE.glob('*.py'))+[HERE/'config.json',ROOT/'tests/test_paired_action_target.py']
    dependency_files=[ROOT/'experiments/continuation_lab_v1/run.py',ROOT/'experiments/continuation_lab_v1/context.py',
                      ROOT/'experiments/regime_language_v1/measures.py']
    identity={str(p.relative_to(ROOT)):sha(p) for p in source_files+dependency_files}
    for p in source_files+dependency_files:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    dump(out/'config.json',cfg);dump(out/'manifest.json',dict(status='RUNNING',source_sha256=identity))
    try:
        assert manifest['status']=='COMPLETE'
        inputs={str((parent/'manifest.json').relative_to(ROOT)):sha(parent/'manifest.json')}
        for name in ('features.parquet','outcomes.parquet','config.json'):
            assert sha(parent/name)==manifest['artifacts'][name]
            inputs[str((parent/name).relative_to(ROOT))]=sha(parent/name)
        f=pd.read_parquet(parent/'features.parquet')
        o=pd.read_parquet(parent/'outcomes.parquet');o=o[o.scenario==cfg['scenario']].copy()
        paired=paired_targets(f,o);paired.to_parquet(out/'paired_targets.parquet',index=False)
        frames=[];descriptions=[];input_cfg=read(parent/'config.json')
        native_rows=0
        for date in input_cfg['dates']:
            q,_,sources,_,_,_=load_day(date,input_cfg);inputs.update(sources);native_rows+=len(q)
            money=cost_decomposition(q,o[o.date==date]);frames.append(money)
            mapper=TimeAtPrice(q)
            # Cache exactly identical clock/risk/price descriptions without
            # treating template aliases as new independent market evidence.
            cache={}
            day_f=f[(f.date==date)&f.origin_admitted]
            for r in day_f.itertuples():
                key=(int(r.t_ns),float(r.decision_price),float(r.risk_hint))
                if key not in cache:
                    cache[key]=describe_future(q,key[0],key[1],key[2],mapper=mapper)
                descriptions.append(dict(origin_index=int(r.origin_index),date=date,**cache[key]))
            print(json.dumps(dict(date=date,quote_rows=len(q),descriptions=len(day_f),unique_descriptions=len(cache)),ensure_ascii=False),flush=True)
        costs=pd.concat(frames,ignore_index=True).merge(f[['origin_index','family','level_kind','side','cost_over_r']],
                                                       on='origin_index',suffixes=('_action','_signal'),validate='many_to_one')
        costs['relation']=np.where(costs.side_action==costs.side_signal,'signal','reverse')
        table=[]
        for (family,relation),g in costs.groupby(['family','relation']):
            R=g.risk_per_oz
            table.append(dict(family=family,relation=relation,complete_actions=len(g),
                              win_fraction=float((g.net_per_oz>0).mean()),net_mean_r=float((g.net_per_oz/R).mean()),
                              actual_cost_mean_r=float((g.future_actual_total_cost/R).mean()),
                              approximate_origin_cost_mean_r=float(g.cost_over_r.mean()),
                              same_execution_mid_move_mean_r=float((g.future_mid_move_same_executions/R).mean()),
                              stopped_mfe_median_r=float((g.mfe_per_oz/R).median()),stopped_mae_median_r=float((g.mae_per_oz/R).median())))
        costs.to_parquet(out/'exact_costs.parquet',index=False);dump(out/'colleague_reconciliation.json',table)
        descriptive=pd.DataFrame(descriptions);descriptive.to_parquet(out/'future_descriptions.parquet',index=False)
        merged=paired.merge(descriptive,on=['origin_index','date'],validate='one_to_one')
        complete=merged[merged.future_both_filled_complete & merged.future_description_known]
        geo=dict(rows=len(descriptive),known=int(descriptive.future_description_known.sum()),
                 range_ge_1r=int((descriptive.future_range_r>=1).sum()),
                 range_median_r=float(descriptive.future_range_r.median()),
                 efficiency_1s_median=float(descriptive.future_efficiency_1s.median()),
                 dwell_median=float(descriptive.future_origin_band_dwell_fraction.median()),
                 known_complete_pairs=len(complete),
                 neither_despite_range_ge_1r=int(((complete.future_range_r>=1)&complete.future_joint_profit_state.eq('NEITHER')).sum()),
                 one_positive_with_range_ge_1r=int(((complete.future_range_r>=1)&complete.future_joint_profit_state.isin(['LONG_ONLY','SHORT_ONLY'])).sum()),
                 contract='120s physical [decision,decision+120s), frozen current R, native midpoint range and held-time profiles; not stopped-trade MFE/MAE')
        dump(out/'future_description_summary.json',geo)
        summary=dict(status='AUDITED_NOT_TRAINED',admitted_origins=len(paired),
                     exact_clock_risk_contracts=int(round(paired.exact_decision_weight.sum())),
                     exact_clock_count=int(paired.t_ns.nunique()),native_quote_rows=native_rows,
                     value_known_pairs=int(paired.future_pair_known.sum()),
                     incomplete_pairs=int((~paired.future_pair_known).sum()),
                     complete_filled_pairs=int(paired.future_both_filled_complete.sum()),
                     nofill_pairs=int((paired.future_status_long.eq('EMPTY')&paired.future_status_short.eq('EMPTY')).sum()),
                     sides_status_counts={str(k):int(v) for k,v in o.status.value_counts().items()},
                     complete_unweighted=pair_summary(paired),exact_decision_weighted=pair_summary(paired,weighted=True),
                     outcome_reasons_both_sides={str(k):int(v) for k,v in o[o.status=='COMPLETE'].reason.value_counts().items()},
                     exact_cost_rows=len(costs),future_description=geo,
                     no_causal_prediction_tested=True,training_admission='BLOCKED_PENDING_ALL_PHASE_DATA_AND_CHRONOLOGICAL_PROTOCOL',
                     economic_admission='NOT_ESTABLISHED')
        dump(out/'summary.json',summary)
        daily=[]
        for date,p in paired.groupby('date'):daily.append(dict(date=date,**pair_summary(p)))
        dump(out/'daily.json',daily)
        if any(sha(ROOT/p)!=v for p,v in identity.items()):raise ValueError('Source changed during run')
        dump(out/'manifest.json',dict(status='COMPLETE',source_sha256=identity,inputs=inputs,
             elapsed_seconds=time.perf_counter()-begun,artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json' and 'source_snapshot' not in p.parts}))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(out/'manifest.json',dict(status='FAILED',source_sha256=identity,error=repr(exc)));raise


if __name__=='__main__':main()
