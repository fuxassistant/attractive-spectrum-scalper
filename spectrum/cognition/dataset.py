"""Paired CLOSE vs fixed-plan HOLD targets, never oracle exit targets.

Roots deliberately cover both sides and three plans on a price-independent
clock. All siblings and ages stay in one chronological split. Root resets are
experiments, not a tradable equity path. Quote buffers are shared and bounded.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import PLANS, NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import ExecutionReplay, ExecutionConfig, FEATURE_NAMES

VERSION = 'HOLD_CLOSE_PAIRED_V1'
DEFAULT_DATA = 'data/derived/basket_multimonth_v1'
DEFAULT_OUTPUT = 'data/derived/hold_close_v1'
OFFSETS = (5, 15, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540)
HISTORY_NAMES = ('return_5m_ticks','return_15m_ticks','return_60m_ticks',
                 'range_5m_ticks','range_15m_ticks','range_60m_ticks',
                 'history_5m_known','history_15m_known','history_60m_known',
                 'quote_move_since_m1_ticks','m1_age_seconds','quote_move_since_m1_ticks_known')


def split_for(date):
    if '2026-05-01' <= date <= '2026-06-30': return 'train'
    if '2026-07-01' <= date <= '2026-07-31': return 'validation'
    if '2026-08-01' <= date <= '2026-08-31': return 'test'
    raise ValueError('Date outside preregistered May-August experiment')


def market_features(day, now):
    """Only completed M1 available at now; never read later rows."""
    i = int(np.searchsorted(day.decision_t, now, side='right')) - 1
    if i < 0: raise ValueError('No available M1 context')
    x = np.asarray(day.x[i], dtype=float).copy()
    history = []; known = []
    for window in (5,15,60):
        lo=i-window+1
        ok = lo>=1 and bool(np.all(day.x[lo-1:i+1,8]>.5)) and bool(np.all(np.diff(day.bar_t[lo-1:i+1])==60*NS))
        history.append((float(day.x[lo:i+1,3].sum()),float(day.x[lo:i+1,1].mean())) if ok else (0.,0.))
        known.append(float(ok))
    q = int(np.searchsorted(day.quote_t,now,side='right'))-1
    mid=(float(day.bid[q])+float(day.ask[q]))/2e9 if q>=0 else float(day.decision_mid[i])
    refq=int(day.cuts[i])-1
    contrast_known=bool(q>=0 and refq>=0 and day.valid[q] and day.valid[refq] and day.epoch[q]==day.epoch[refq] and x[8]>.5)
    contrast=(mid-float(day.decision_mid[i]))/.1 if contrast_known else 0.
    extra=[h[0] for h in history]+[h[1] for h in history]+known+[contrast,(now-int(day.decision_t[i]))/NS,float(contrast_known)]
    return np.r_[x,extra].astype(np.float32), i


def features(day, replay):
    market,i=market_features(day,replay.state.clock_ns)
    return np.r_[market,replay.position_features()].astype(np.float32),i


def outcome(replay):
    s=replay.state
    return dict(cash=float(s.cash),equity=float(s.equity),net=float(s.cash-1500),
                min_equity=float(s.min_equity),max_equity=float(s.peak_equity),
                exit_ns=int(s.exit_ns),exit_reason=s.exit_reason,
                censored=bool(s.censored),dead=bool(s.dead),open=bool(s.side),
                snapshot=replay.snapshot())


def prepare_day(path, output, protocol):
    started=time.perf_counter();path=Path(path);output=Path(output)
    date=path.name;dest=output/date;manifest=dest/'manifest.json'
    identity=dict(day_manifest=sha(path/'manifest.json'),protocol_sha256=sha(output/'protocol.json'))
    if manifest.exists():
        m=json.loads(manifest.read_text(encoding='utf8'))
        if m['identity']!=identity:raise ValueError('Completed dataset identity changed')
        for name,digest in m['files'].items():
            if sha(dest/name)!=digest:raise ValueError('Completed dataset payload changed')
        return m
    day=MonthDay(path,verify=True);config=ExecutionConfig()
    feature_names=list(day.meta['feature_names'])+list(HISTORY_NAMES)+list(FEATURE_NAMES)
    xs=[];rows=[];roots=[];exclusions=[]
    london=int(day.meta['london_ns']);deadline=day.deadline
    eligible=np.flatnonzero(day.can_enter)
    # First causal available minute in each fixed half-hour slot, not first future profitable event.
    slots={}
    for i in eligible:
        t=int(day.decision_t[i]);slot=(t-london)//(1800*NS)
        if t>=london and slot not in slots:slots[slot]=int(i)
    for slot,i in slots.items():
        intent=int(day.decision_t[i])
        for side in (-1,1):
            for plan_id,plan in enumerate(PLANS):
                root_id=f'{date}:{slot}:{side}:{plan_id}'
                if intent+(plan[2]+4)*NS>=deadline:
                    exclusions.append(dict(root_id=root_id,reason='insufficient_declared_session_time'));continue
                base=ExecutionReplay(day,config=config)
                base.open_at(intent,side,plan_id)
                base.advance_until(intent+3*NS)
                if base.state.entry_ns<=0:
                    exclusions.append(dict(root_id=root_id,reason='entry_not_filled',snapshot=base.snapshot()));continue
                horizon=min(int(base.state.entry_ns)+(plan[2]+3)*NS,deadline+3*NS)
                ref=base.clone();ref.reference_exit(horizon)
                root=dict(root_id=root_id,origin_id=f'{date}:{slot}',date=date,split=split_for(date),slot=int(slot),side=side,plan_id=plan_id,
                          intent_ns=intent,entry_ns=int(base.state.entry_ns),entry_price_nano=int(base.state.entry_price_nano),
                          horizon_ns=horizon,reference=outcome(ref),source_path=str(path))
                roots.append(root)
                for age in OFFSETS:
                    now=int(base.state.entry_ns)+age*NS
                    if now>=horizon:break
                    base.advance_until(now)
                    if base.state.censored or base.state.dead or not base.state.side:break
                    x,mi=features(day,base)
                    close=base.clone();close.request_close(now);close.reference_exit(horizon)
                    co=outcome(close);ho=root['reference']
                    known=not (co['censored'] or ho['censored'] or co['open'] or ho['open'])
                    advantage=ho['cash']-co['cash'] if known else None
                    rows.append(dict(root_id=root_id,origin_id=f'{date}:{slot}',date=date,split=split_for(date),side=side,plan_id=plan_id,
                        age_seconds=age,decision_ns=now,market_available_ns=int(day.decision_t[mi]),horizon_ns=horizon,
                        target_known=known,target_advantage=advantage,hold_cash=ho['cash'],close_cash=co['cash'],
                        hold_censored=ho['censored'],close_censored=co['censored'],
                        hold_exit_ns=ho['exit_ns'],close_exit_ns=co['exit_ns'],
                        snapshot_json=json.dumps(base.snapshot(),separators=(',',':'))))
                    xs.append(x)
    dest.mkdir(parents=True,exist_ok=True)
    matrix=np.stack(xs) if xs else np.empty((0,len(feature_names)),np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite causal features')
    np.save(dest/'x.npy',matrix)
    pd.DataFrame(rows).to_parquet(dest/'pairs.parquet',index=False)
    save_json(dest/'roots.json',roots);save_json(dest/'exclusions.json',exclusions)
    m=dict(status='COMPLETE',version=VERSION,date=date,split=split_for(date),identity=identity,
        roots=len(roots),pairs=len(rows),known_pairs=sum(r['target_known'] for r in rows),
        censored_roots=sum(r['reference']['censored'] for r in roots),excluded_origins=len(exclusions),
        zero_decision_roots=len(roots)-len({r['root_id'] for r in rows}),
        feature_names=feature_names,market_dim=len(day.meta['feature_names'])+len(HISTORY_NAMES),
        seconds=time.perf_counter()-started,
        files={n:sha(dest/n) for n in ('x.npy','pairs.parquet','roots.json','exclusions.json')})
    save_json(manifest,m);QUOTE_CACHE.clear()
    return m


def prepare(data=DEFAULT_DATA,output=DEFAULT_OUTPUT,limit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    code={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('execution.py')]}
    protocol=dict(version=VERSION,config=asdict(ExecutionConfig()),plans=PLANS,offset_seconds=OFFSETS,
        origin_spacing_seconds=1800,sides=[-1,1],months=['2026-05','2026-06','2026-07','2026-08'],
        splits=dict(train='2026-05..2026-06',validation='2026-07',test='2026-08'),
        semantics='HOLD follows fixed quote-level protection plan; CLOSE intent uses latency, then cash. Common root horizon. No reentry in this option.',
        sequential='Repeated learner decisions are evaluated separately; fitted Q is relative to fixed continuation, not Q of the new policy.',
        economics='Independent-reset paired roots; not one continuous account. All historical dates previously researched: development evidence.',
        sampling='Both sides and all plans, first eligible available M1 in each session half-hour; no future outcome selection. Session-fit exclusion explicit.',
        coverage='Empty event memory not fabricated. Current compact M1 plus causal 5/15/60 minute summaries and full position state.',
        code=code,catalog_sha256=sha(Path(data)/'catalog.json'))
    if (output/'protocol.json').exists():
        old=json.loads((output/'protocol.json').read_text(encoding='utf8'))
        if json.dumps(old,sort_keys=True)!=json.dumps(protocol,sort_keys=True):raise ValueError('Protocol drift; new dataset directory required')
    else:save_json(output/'protocol.json',protocol)
    paths=[];coverage=[]
    for month in protocol['months']:
        m=json.loads((Path(data)/month/'manifest.json').read_text(encoding='utf8'))
        if m['status']!='COMPLETE':raise ValueError('Unfinished source month')
        coverage.append(dict(month=month,days=len(m['days']),missing_weekdays=m['missing_weekdays']))
        for row in m['days']:
            p=Path(data)/month/row['date']
            if not (p/'manifest.json').exists() or sha(p/'manifest.json')!=row['manifest_sha256']:raise ValueError('Missing or changed expected day '+str(p))
            paths.append(p)
    paths.sort();save_json(output/'source_coverage.json',coverage)
    if limit is not None:paths=paths[:limit]
    started=time.perf_counter();results=[]
    for path in paths:
        result=prepare_day(path,output,protocol);results.append(result)
        print(json.dumps({k:result[k] for k in ('date','roots','pairs','known_pairs','seconds')}),flush=True)
        save_json(output/'progress.json',dict(status='RUNNING',days=results))
    save_json(output/'manifest.json',dict(status='COMPLETE' if limit is None else 'PILOT',days=results,seconds=time.perf_counter()-started,
        roots=sum(d['roots'] for d in results),pairs=sum(d['pairs'] for d in results),code=code))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--limit',type=int)
    a=p.parse_args();prepare(a.data,a.output,a.limit)
