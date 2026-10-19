"""Reproducible month search, chronological selection, and exact event receipts."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,time,hashlib
import numpy as np
import numba
from spectrum.core.storage import sha
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from .data import MonthDay
from .runner import DEFAULT_DATA
from .basket import BasketConfig,signals,initial_state,batch_day,replay_day,summary,EVENT_FIELDS

DEFAULT_OUTPUT='reports/basket_search_august_1500'


def grid(n=320,seed=61018):
    rng=np.random.default_rng(seed);configs=[BasketConfig(),BasketConfig(mode='long'),BasketConfig(mode='short'),BasketConfig(mode='trend')]
    seen={tuple(asdict(c).values()) for c in configs}
    while len(configs)<n:
        c=BasketConfig(mode=str(rng.choice(['long','short','trend','fade'])),
            lookback=int(rng.choice([15,30,60,120])),deviation=float(rng.choice([0,.5,1,2])),
            er_max=float(rng.choice([.1,.25,.5,1.])),vol_max=float(rng.choice([.8,1.2,2.,10.])),
            step_scale=float(rng.choice([1,2,4,8])),step_growth=float(rng.choice([1.,1.5,2.])),
            target_scale=float(rng.choice([1,2,4,8])),multiplier=float(rng.choice([1.25,1.5,2.])),
            max_orders=int(rng.choice([3,4,5])),cap_oz=float(rng.choice([4,8,16])),
            max_hold_minutes=int(rng.choice([30,120,360])),basket_stop_fraction=float(rng.choice([.05,.1,.2,.4])),
            cooldown_minutes=int(rng.choice([5,15,30])))
        key=tuple(asdict(c).values())
        if key not in seen:configs.append(c);seen.add(key)
    return configs


def evaluate_many(data,dates,configs,out=None,stage='train'):
    states=np.stack([initial_state() for _ in configs]);params=np.stack([c.array() for c in configs]);daily=[]
    for date in dates:
        start=time.perf_counter();d=MonthDay(Path(data)/date)
        sig=[signals(d,c) for c in configs]
        sides=np.stack([v[0] for v in sig]);scales=np.stack([v[1] for v in sig])
        states=batch_day(np.asarray(d.quote_t),np.asarray(d.bid),np.asarray(d.ask),np.asarray(d.valid),np.asarray(d.epoch),
            np.asarray(d.decision_t),sides,scales,params,states,int(d.meta['london_ns']),int(d.meta['ny_close_ns']),d.deadline)
        daily.append(dict(date=date,seconds=time.perf_counter()-start,results=[summary(s) for s in states]))
        print(json.dumps(dict(stage=stage,date=date,seconds=daily[-1]['seconds'],best_net=max(s[1]-1500 for s in states),
            alive=int(sum(not(s[8] or s[9]) for s in states)))),flush=True)
        if out:save_json(Path(out)/(stage+'_progress.json'),dict(status='RUNNING',days=daily))
    if out:save_json(Path(out)/(stage+'_progress.json'),dict(status='COMPLETE',days=daily))
    return states,daily


def evaluate_record(data,dates,config,destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);s=initial_state();events=[];daily=[];traces=[];audited=0
    for date in dates:
        if s[8] or s[9]:break
        d=MonthDay(Path(data)/date);side,scale=signals(d,config)
        s,t,e,trace=replay_day(np.asarray(d.quote_t),np.asarray(d.bid),np.asarray(d.ask),np.asarray(d.valid),np.asarray(d.epoch),
            np.asarray(d.decision_t),side,scale,config.array(),s,int(d.meta['london_ns']),int(d.meta['ny_close_ns']),d.deadline,True)
        for ts,row in zip(t,e):
            item={k:float(v) if np.isfinite(v) else None for k,v in zip(EVENT_FIELDS,row)};item.update(t_ns=str(int(ts)),date=date)
            j=int(item['quote_index']);kind=int(item['kind']);sg=int(item['side'])
            if kind!=7:
                expected=int((d.ask if (kind in (1,2))==(sg>0) else d.bid)[j])/1e9
                if abs(expected-item['price'])>1e-8 or int(d.quote_t[j])!=int(ts) or not d.valid[j]:raise AssertionError('Fill audit failure')
                if kind in (1,2) and not int(d.meta['london_ns'])<=ts<int(d.meta['ny_close_ns']):raise AssertionError('Entry outside window')
                audited+=1
            events.append(item)
        for ts,v in zip(d.decision_t,trace):
            traces.append(dict(date=date,t_ns=str(int(ts)),equity=float(v[0]) if np.isfinite(v[0]) else None,
                size_oz=float(v[1]) if np.isfinite(v[1]) else None,floor=float(v[2]) if np.isfinite(v[2]) else None))
        daily.append(dict(date=date,**summary(s)))
    record=dict(config=asdict(config),result=summary(s),days=daily,events=events,trace=traces,fill_audit=dict(status='PASS',rows=audited))
    save_json(dest/'result.json',record);return record


def score(s):
    if s[8] or s[9] or s[11]<3:return -1e20
    return np.log(max(s[1]/1500,1e-8))-.5*s[6]


def search(data=DEFAULT_DATA,output=DEFAULT_OUTPUT,n=320):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter();numba.set_num_threads(min(8,numba.config.NUMBA_NUM_THREADS))
    manifest=read_json(Path(data)/'manifest.json');configs=grid(n)
    protocol=dict(initial_capital=1500,fee_per_oz=.30,latency_ms=100,entry_timeout_seconds=2,
        risk='Strictly more than 50% loss from active series starting capital; no daily series reset',
        session='08:00 London to 17:00 New York; adds included; flatten before daily support end',
        execution='Actual bid/ask after intent latency; invalid exposed book censors; no fabricated recovery',
        data_sha256=sha(Path(data)/'manifest.json'),source={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('basket.py')]},
        splits=manifest['split'],configs=[asdict(c) for c in configs],seed=61018,
        selection='Top 24 train by log growth minus .5 maximum series loss; minimum 3 closed baskets and complete alive. Select train/validation positive candidate by minimum score, fallback same score.',
        caveat='August was previously inspected. Later dates are a selection check, not a pristine blind holdout. No neural training. No exchange margin or impact model.')
    save_json(root/'protocol.json',protocol)
    train,td=evaluate_many(data,manifest['split']['train'],configs,root,'train')
    order=sorted(range(n),key=lambda i:score(train[i]),reverse=True)[:24]
    val,vd=evaluate_many(data,manifest['split']['validation'],[configs[i] for i in order],root,'validation')
    positive=[j for j,i in enumerate(order) if train[i,1]>1500 and val[j,1]>1500 and score(train[i])>-1e19 and score(val[j])>-1e19]
    eligible=positive or [j for j,i in enumerate(order) if score(train[i])>-1e19 and score(val[j])>-1e19]
    if not eligible:save_json(root/'blocked.json',dict(reason='No complete surviving train/validation candidate'));return
    chosen=max(eligible,key=lambda j:min(score(train[order[j]]),score(val[j])))
    idx=order[chosen];config=configs[idx]
    freeze=dict(index=idx,config=asdict(config),train=summary(train[idx]),validation=summary(val[chosen]),
        frozen_before_holdout=True,source=protocol['source'],data_sha256=protocol['data_sha256'])
    save_json(root/'frozen.json',freeze)
    hold=evaluate_record(data,manifest['split']['holdout'],config,root/'holdout')
    dates=[d['date'] for d in manifest['days']]
    month=evaluate_record(data,dates,config,root/'month')
    fixed=evaluate_record(data,dates,replace(config,multiplier=1.),root/'same_grid_fixed_adds')
    single=evaluate_record(data,dates,replace(config,max_orders=1,cap_oz=1.),root/'same_signal_no_adds')
    leaderboard=[dict(index=i,config=asdict(configs[i]),train=summary(train[i]),validation=summary(val[j])) for j,i in enumerate(order)]
    save_json(root/'leaderboard.json',leaderboard)
    save_json(root/'summary.json',dict(status='COMPLETE',seconds=time.perf_counter()-start,selected=freeze,
        holdout=hold['result'],month=month['result'],fixed_adds=fixed['result'],no_adds=single['result']))
    print(json.dumps(dict(status='COMPLETE',month=month['result'],holdout=hold['result'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--n',type=int,default=320)
    a=p.parse_args();search(a.data,a.output,a.n)
