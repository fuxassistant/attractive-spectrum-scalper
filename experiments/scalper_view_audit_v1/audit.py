"""Measure representation sensitivity without learning or replaying a policy."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.zone_episode_lab.episodes import build_episodes, NS, TICK
from experiments.price_level_memory_v1.memory import build_memory

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def dump(path,data):
    Path(path).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def describe(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),p10=float(np.quantile(a,.1)),p50=float(np.median(a)),p90=float(np.quantile(a,.9)))


def shifted_episodes(q,end,variant,gap=30):
    """Translate prices, run original state machine, restore absolute addresses.

    This changes the grid origin only; it does not snap, interpolate or shift
    timestamps. UNKNOWN previous-price sentinels remain zero.
    """
    offset=variant.get('grid_offset_ticks',0)
    if isinstance(offset,bool) or int(offset)!=offset:raise ValueError('Integer grid offset ticks required')
    shift=int(offset)*TICK
    x=q.copy() if shift else q
    if shift:
        for c in ('bid_nano','ask_nano'):x[c]=x[c]-shift
    ev,ep=build_episodes(x,end_ns=int(end),bin_ticks=variant['bin_ticks'],departure_ticks=variant['departure_ticks'],
                         confirmation_seconds=variant['confirmation_seconds'],max_quote_age_seconds=gap)
    if shift:
        for c in ('zone_low_nano','mid_nano'):ev[c]+=shift
        for c in ('zone_low_nano','zone_high_nano','arrival_mid_nano','end_mid_nano'):ep[c]+=shift
        ep.loc[ep.has_previous_departure,'previous_departure_mid_nano']+=shift
    return ev,ep


def recognition_table(ev,ep,start,end):
    completed=ep.loc[ep.completed & ep.arrival_ns.ge(start)&ep.arrival_ns.lt(end)].copy()
    c=ev[ev.event_type.eq('DEPARTURE_CONFIRMED')][['episode_id','available_ns','mid_nano','side']]
    c=c.rename(columns={'available_ns':'confirmation_ns','mid_nano':'confirmation_mid_nano'})
    out=completed[['episode_id','zone_low_nano','arrival_ns','end_ns','outcome']].merge(c,on='episode_id',validate='one_to_one')
    for kind,label in [('EDGE_EXIT','last_edge'),('GATE_START','last_gate')]:
        part=ev[ev.event_type.eq(kind)].drop_duplicates('episode_id',keep='last')
        part=part[['episode_id','available_ns','mid_nano']].rename(columns={'available_ns':label+'_ns','mid_nano':label+'_mid_nano'})
        out=out.merge(part,on='episode_id',validate='one_to_one')
    out['last_edge_to_confirmation_s']=(out.confirmation_ns-out.last_edge_ns)/NS
    out['last_gate_to_confirmation_s']=(out.confirmation_ns-out.last_gate_ns)/NS
    out['signed_move_after_last_edge_per_oz']=out.side*(out.confirmation_mid_nano-out.last_edge_mid_nano)/NS
    out['signed_move_after_last_gate_per_oz']=out.side*(out.confirmation_mid_nano-out.last_gate_mid_nano)/NS
    assert (out.last_edge_to_confirmation_s>=0).all() and (out.last_gate_to_confirmation_s>=0).all()
    return out


def same_arrival_comparison(base,variant):
    """Only same-core experiments can share exactly the same arrival identity.

    Report unmatched objects too. This is definition sensitivity, not an error
    rate, and cannot be used for shifted grids with different physical cores.
    """
    keys=['zone_low_nano','arrival_ns','arrival_ordinal']
    p=base.merge(variant,on=keys,suffixes=('_base','_variant'),how='inner',validate='one_to_one')
    both=p.completed_base&p.completed_variant
    return dict(base_episode_count=len(base),variant_episode_count=len(variant),shared_arrivals=len(p),unmatched_base=len(base)-len(p),
                unmatched_variant=len(variant)-len(p),both_completed=int(both.sum()),
                outcome_disagreement_fraction=float((p.loc[both,'outcome_base']!=p.loc[both,'outcome_variant']).mean()),
                confirmation_delta_s=describe((p.loc[both,'end_ns_variant']-p.loc[both,'end_ns_base'])/NS),
                interpretation='Same observed arrival, changed definition. Unmatched episodes retained in denominator report; no truth/error label.')


def gate_selection(ev,date,start,end,count=6):
    g=ev[ev.event_type.eq('GATE_START')].drop_duplicates('episode_id',keep='first')
    g=g[g.available_ns.ge(start)&g.available_ns.lt(end)].copy()
    g['selection_key']=[hashlib.sha256(f'{date}|{r.zone_low_nano}|{r.episode_id}|{r.available_ns}'.encode()).hexdigest() for r in g.itertuples()]
    return g.sort_values('selection_key').head(count)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');args=ap.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());out=ROOT/(args.out or cfg['output'])
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=ROOT/cfg['source_run'];sm=json.loads((source/'manifest.json').read_text())
    assert sm['status']=='COMPLETE_DIAGNOSTIC_ONLY'
    inputs={str((source/'manifest.json').relative_to(ROOT)):digest(source/'manifest.json')}
    dump(out/'manifest.json',dict(status='RUNNING',config=cfg))
    rows=[];comparisons=[];micro=[];latencies=[];clocks=[];atlas=[];flow=[]
    for date in cfg['dates']:
        receipt=next(r for r in sm['receipts'] if r['date']==date)
        market=Path(receipt['source']);mp=market/'manifest.json'
        assert digest(mp)==receipt['source_manifest_sha256']
        mm=json.loads(mp.read_text());qp=market/'quotes.parquet'
        assert digest(qp)==receipt['source_payloads']['quotes.parquet']==mm['payloads']['quotes.parquet']
        for p in (mp,qp):inputs[str(p.relative_to(ROOT))]=digest(p)
        q=pd.read_parquet(qp,columns=['available_ns','bid_nano','ask_nano','valid','epoch','source_ordinal'])
        end=int(mm['input']['end_ns'])+1;day=int(pd.Timestamp(date,tz='UTC').value)
        a,b=[day+x*3600*NS for x in cfg['arrival_window_utc']]
        baseline=None;base_events=None
        for variant in cfg['variants']:
            t=time.perf_counter();ev,ep=shifted_episodes(q,end,variant,cfg['max_quote_age_seconds'])
            visible=ev[ev.available_ns.ge(a)&ev.available_ns.lt(b)]
            selected=ep[ep.arrival_ns.ge(a)&ep.arrival_ns.lt(b)]
            lat=recognition_table(ev,ep,a,b);lat['date']=date;lat['variant']=variant['name'];latencies.append(lat)
            row=dict(date=date,variant=variant['name'],episodes_started=len(selected),completed_started=int(selected.completed.sum()),
                     events_by_type=visible.event_type.value_counts().to_dict(),outcomes=selected.outcome.value_counts().to_dict(),
                     last_edge_to_confirmation_s=describe(lat.last_edge_to_confirmation_s),
                     last_gate_to_confirmation_s=describe(lat.last_gate_to_confirmation_s),
                     signed_move_after_last_edge_per_oz=describe(lat.signed_move_after_last_edge_per_oz),
                     elapsed_seconds=time.perf_counter()-t)
            rows.append(row)
            if variant['name']=='base':
                for name,new in [('events',ev),('episodes',ep)]:
                    oldpath=source/'days'/date/(name+'.parquet')
                    assert digest(oldpath)==receipt['artifact_sha256'][name+'.parquet']
                    inputs[str(oldpath.relative_to(ROOT))]=digest(oldpath)
                    old=pd.read_parquet(oldpath)
                    assert_frame_equal(new.reset_index(drop=True),old[new.columns].reset_index(drop=True))
                baseline=selected;base_events=ev
                qt=q.available_ns.to_numpy(np.int64);mid=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
                for r in gate_selection(ev,date,a,b).itertuples():
                    at=int(r.available_ns);lo=np.searchsorted(qt,at-30*NS,'left');hi=np.searchsorted(qt,at+30*NS,'right')
                    part=q.iloc[lo:hi].copy();part['mid_nano']=mid[lo:hi]
                    case=f'{date}_{int(r.episode_id)}_{at}'
                    part.to_parquet(out/(case+'_quotes.parquet'),index=False)
                    events=ev[(ev.zone_low_nano==r.zone_low_nano)&ev.available_ns.between(at-30*NS,at+30*NS)].copy()
                    events['known_at_decision']=events.available_ns<=at
                    events.to_parquet(out/(case+'_events.parquet'),index=False)
                    atlas.append(dict(case_id=case,date=date,episode_id=r.episode_id,zone_low_nano=r.zone_low_nano,
                                      decision_ns=at,selection_key=r.selection_key,
                                      state_at_decision='GATE_START_FORMING_NOT_A_COMPLETED_BREAKOUT',
                                      future_panel='Gray region is audit-only, never supplied as decision X'))
            elif not variant.get('grid_offset_ticks'):
                comparisons.append(dict(date=date,variant=variant['name'],**same_arrival_comparison(baseline,selected)))
            print(json.dumps(dict(date=date,variant=variant['name'],episodes=len(selected),seconds=row['elapsed_seconds'])),flush=True)
        # Candidate clock is measured separately from recognition and not used
        # to choose a more favorable map definition.
        cp=source/'days'/date/'candidates.parquet';assert digest(cp)==receipt['artifact_sha256']['candidates.parquet']
        inputs[str(cp.relative_to(ROOT))]=digest(cp);c=pd.read_parquet(cp);c=c[~c.inside]
        clocks.append(dict(date=date,candidates=len(c),confirmation_to_candidate_s=describe((c.t_ns-c.episode_available_ns)/NS),
                           distance_r=describe(c.distance_r),risk_per_oz=describe(c.risk_per_oz),cadence='completed M1; per-zone cooldown300s; distance1-3R; same frozen admission'))
        # Micro-map visits include hysteresis and are a different definition.
        for hysteresis in cfg['micro_map']['hysteresis_ticks']:
            # A huge viewport requests all observed price cells at this asof;
            # aggregation uses only observed cells, not future-selected prices.
            m=build_memory(q,np.array([b],np.int64),bin_ticks=5,hysteresis_ticks=hysteresis,radius_ticks=5000)
            micro.append(dict(date=date,hysteresis_ticks=hysteresis,asof_ns=b,
                              arrivals=int(m.arrivals.sum()),passages=int((m.pass_up+m.pass_down).sum()),
                              rejections=int((m.reject_up+m.reject_down).sum()),jump_over=int(m.jump_over.sum()),
                              active=int(m.active.sum()),mid_dwell_s=float(m.mid_dwell_s.sum()),
                              expected_supported_seconds=float(m.history_observed_seconds.iloc[0]),
                              scope='Capture start through20UTC, not just07-20; cells cover current viewport +/-500USD'))
            assert abs(m.mid_dwell_s.sum()-m.history_observed_seconds.iloc[0])<1e-7
        # Unknown-side frequency is distinct from unknown volume fraction.
        folder=ROOT/cfg['feature_week']/'days'/date
        fm=json.loads((folder/'manifest.json').read_text())
        fp=folder/'features.parquet'
        # Frozen week audit records this actual payload hash in its lineage.
        week_audit=ROOT/'data/derived/feature_week_v2/audit/audit.json'
        wa=json.loads(week_audit.read_text())
        expected=wa['lineage']['payload_sha256'][str(fp)]
        assert digest(fp)==expected
        inputs[str(fp.relative_to(ROOT))]=expected;inputs[str(week_audit.relative_to(ROOT))]=digest(week_audit)
        columns=['activity_128s_unknown_fraction','activity_128s_unknown_fraction__known',
                 'activity_128s_signed_lower_bound','activity_128s_signed_upper_bound']
        f=pd.read_parquet(fp,columns=columns);known=f.iloc[:,1].to_numpy(bool);u=f.iloc[:,0].to_numpy(float)[known]
        flow.append(dict(date=date,rows=len(f),known_total_windows=int(known.sum()),
                         any_unknown_side_window_fraction=float((u>0).mean()),
                         unknown_volume_fraction=describe(u),
                         bound_width=describe((f.iloc[:,3]-f.iloc[:,2])[known]),
                         interpretation='128s overlapping windows; any unknown trade != most volume unknown. No independence claim.'))
    pd.concat(latencies,ignore_index=True).to_parquet(out/'recognition_latency.parquet',index=False)
    dump(out/'atlas_index.json',atlas)
    result=dict(status='COMPLETE_DESCRIPTIVE_ONLY',scope=cfg['scope'],variants=rows,same_arrival_comparisons=comparisons,
                candidate_clocks=clocks,micro_map=micro,unknown_side=flow,
                grid_comparison='Shifted grids define different cores: counts are sensitivity, not matched label error or proof of false levels.',
                timing='Reference is LAST OBSERVED EDGE_EXIT before confirmation, not an unknowable true movement onset. Timer, geometry and candidate admission are separate.',
                base_full_ledger_parity=True,input_sha256=inputs,elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',result)
    sources=[HERE/'audit.py',HERE/'config.json',ROOT/'experiments/zone_episode_lab/episodes.py',ROOT/'experiments/price_level_memory_v1/memory.py']
    dump(out/'manifest.json',dict(status=result['status'],source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256=inputs,artifacts={p.name:digest(p) for p in out.iterdir() if p.name!='manifest.json'},
        future_targets=False,model_fit=False,policy_replay=False,elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'],output=str(out))))


if __name__=='__main__':main()
