"""Explicit preparation from certified tapes; training loads only frozen arrays.

Market diagnostics and daily hindsight summaries are never passed into X.
The event bank stays empty until a separately versioned causal event job exists.
"""
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
from spectrum.finalized import FinalizedReplay, read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from spectrum.player.contract import Contract, NS
from spectrum.lab import compute, Context
from spectrum.plugins.volatility_vwap import compute as compute_vwap
from .data import (Day, bounds, minute_values, anchor_values, support_segment_starts,
    BASE_NAMES, FAST_NAMES, FAST_BASE, VWAP_NAMES, ANCHOR_VALUES, EVENT_NAMES)


class FinalizedDay(Day):
    def __init__(self, path, verify=False):
        self.path=Path(path);self.meta=read_json(self.path/'manifest.json')
        if self.meta.get('support_contract')!='FINALIZED_SEGMENTS_V2' or self.meta.get('status')!='COMPLETE':
            raise ValueError('Expected finalized gym input')
        for name,spec in self.meta['files'].items():
            p=self.path/name
            if p.stat().st_size!=spec['bytes'] or (verify and sha(p)!=spec['sha256']):
                raise ValueError('Prepared input changed')
            setattr(self,name[:-4],np.load(p,mmap_mode='r'))
        self.deadline=int(self.meta['liquidation_deadline_ns'])


def load_session(root,date,symbol,contract,include_quotes=True):
    start,end,london,ny=bounds(date,contract)
    entries=read_json(Path(root)/'catalog.json')['days']
    pieces={n:[] for n in ('seconds','bars','quotes')};lineage=[]
    for number,e in enumerate(entries):
        if e['status']!='COMPLETE' or e.get('symbol')!=symbol:continue
        if e['date'] not in (str(pd.Timestamp(start,tz='UTC').date()),date):continue
        folder=Path(root)/e['key']
        for layer in pieces:
            if layer=='quotes' and not include_quotes:continue
            frame=FinalizedReplay(folder,layer).read_window(start,end).to_pandas()
            # Disjoint source namespaces. Never join two independently reset books.
            if len(frame):
                frame['epoch']=np.where(frame.segment_id>=0,frame.segment_id+number*1_000_000,-1)
                frame['segment_id']=frame.epoch
                pieces[layer].append(frame)
        lineage.append(dict(key=e['key'],identity=e['identity']))
    if not pieces['seconds'] or not pieces['bars']:return None
    result={k:pd.concat(v,ignore_index=True) if v else None for k,v in pieces.items()}
    for k,f in result.items():
        if f is not None and np.any(np.diff(f.available_ns.to_numpy())<0):raise ValueError('Catalog not chronologically ordered')
    s=result['seconds'];c=compute(s)
    columns=dict(t_ns=s.t_ns.to_numpy(),available_ns=s.available_ns.to_numpy(),admissible=c.admitted)
    for n,v in c.values.items():columns[n]=v;columns[n+'__known']=c.masks[n]
    f=pd.DataFrame(columns)
    result.update(features=f,start=start,end=end,london=london,ny=ny,lineage=lineage)
    return result


def prepare(root,output,date,contract=None):
    contract=contract or Contract();root=Path(root);output=Path(output)
    entries=read_json(root/'catalog.json')['days']
    selected=[e for e in entries if e.get('date')==date and e['status']=='COMPLETE']
    if len(selected)!=1:raise ValueError('Date is not uniquely published')
    selected=selected[0];symbol=selected['symbol']
    cur=load_session(root,date,symbol,contract)
    if cur is None:raise ValueError('No session data')
    previous_date=str((pd.Timestamp(date)-pd.offsets.BDay(1)).date())
    prev=load_session(root,previous_date,symbol,contract,False)
    s,b,f,q=(cur[n] for n in ('seconds','bars','features','quotes'))
    if q is None or len(b)==0:raise ValueError('No bars or execution observations')
    x,decision,legal=minute_values(b,f,cur['start'],cur['end'])
    legal&=(decision>=cur['london'])&(decision<cur['ny'])
    selection=read_json(root/selected['key']/'manifest.json')['selection']
    legal&=decision>=int(selection['selection_available_ns'])
    if selection['context_only']:legal[:]=False
    slow_x=x.copy();slow_t=b.t_ns.to_numpy();slow_available=decision.copy();slow_session=np.ones(len(x),np.int8)
    previous_bundle=None
    if prev is not None:
        px,pav,_=minute_values(prev['bars'],prev['features'],prev['start'],prev['end'])
        slow_x=np.r_[px,x];slow_t=np.r_[prev['bars'].t_ns,slow_t];slow_available=np.r_[pav,slow_available]
        slow_session=np.r_[np.zeros(len(px),np.int8),slow_session]
        previous_bundle=(prev['seconds'],prev['bars'],prev['start'],prev['london'],prev['end'])
    context=Context(s,None);compute_vwap(context)
    fv=np.column_stack([f[n].to_numpy() for n in FAST_BASE]+[context.values[n] for n in VWAP_NAMES])
    fk=np.column_stack([f[n+'__known'].to_numpy() for n in FAST_BASE]+[context.masks[n] for n in VWAP_NAMES])
    fast=np.c_[np.where(fk,fv,0.),fk].astype(np.float32)
    event_available=np.empty(0,np.int64)
    anchors,mid=anchor_values(s,b,decision,previous_bundle,cur['start'],cur['london'],date,previous_date,prev is not None,event_available)
    quote_t=q.available_ns.to_numpy(np.int64);fast_available=np.maximum.accumulate(s.available_ns.to_numpy(np.int64))
    arrays=dict(x=x,decision_t=decision,bar_t=b.t_ns.to_numpy(np.int64),can_enter=legal,
        quote_t=quote_t,source_t=q.t_ns.to_numpy(np.int64),bid=q.bid_nano.to_numpy(np.int64),ask=q.ask_nano.to_numpy(np.int64),
        valid=q.valid.to_numpy(bool)&~q.scheduled_closed.to_numpy(bool),epoch=q.segment_id.to_numpy(np.int64),
        ordinal=np.arange(len(q),dtype=np.int64),source_ordinal=q.source_ordinal.to_numpy(np.int64),
        fast_x=fast,fast_t=s.t_ns.to_numpy(np.int64),fast_available=fast_available,fast_valid=s.valid.to_numpy(bool),
        slow_x=slow_x,slow_t=slow_t,slow_available=slow_available,slow_session=slow_session,anchors=anchors,decision_mid=mid,
        event_x=np.empty((0,len(EVENT_NAMES)-3),np.float32),event_available=event_available,event_t=event_available.copy(),event_prices=np.empty((0,2)),
        cuts=np.searchsorted(quote_t,decision,side='right'),fast_cuts=np.searchsorted(fast_available,decision,side='right'),
        slow_cuts=np.searchsorted(slow_available,decision,side='right'),event_cuts=np.zeros(len(decision),np.int64),
        fast_epoch=s.segment_id.to_numpy(np.int64),fast_segment_start=support_segment_starts(s.t_ns.to_numpy(),s.valid.to_numpy(),s.segment_id.to_numpy()))
    required=sum(a.nbytes for a in arrays.values())+8*1024**3
    if shutil.disk_usage(output.parent if output.parent.exists() else root).free<required:
        raise RuntimeError('DISK_RESERVE_BELOW_PREPARED_ARRAY_BUDGET')
    for n,a in arrays.items():
        if np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all():raise ValueError('Nonfinite model input '+n)
    output.mkdir(parents=True,exist_ok=False)
    for n,a in arrays.items():
        np.save(output/(n+'.npy'),a)
    meta=dict(status='COMPLETE',support_contract='FINALIZED_SEGMENTS_V2',date=date,symbol=symbol,
        start_ns=str(cur['start']),end_ns=str(cur['end']),ny_close_ns=str(cur['ny']),london_ns=str(cur['london']),
        liquidation_deadline_ns=str(cur['end']-contract.terminal_reserve_seconds*NS),
        feature_names=BASE_NAMES,fast_names=FAST_NAMES+[n+'__known' for n in FAST_NAMES],slow_names=BASE_NAMES,
        anchor_names=ANCHOR_VALUES+[n+'__known' for n in ANCHOR_VALUES],event_names=EVENT_NAMES,
        previous_date=previous_date,previous_available=prev is not None,current_lineage=cur['lineage'],previous_lineage=prev['lineage'] if prev else [],
        event_bank_status='EMPTY_NO_CERTIFIED_EVENT_JOB',entry_origins=int(legal.sum()),quotes=len(q),rows=len(x),
        implementation={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).parents[1]/'lab.py',Path(__file__).parents[1]/'plugins/volatility_vwap.py']},
        files={n+'.npy':dict(bytes=(output/(n+'.npy')).stat().st_size,sha256=sha(output/(n+'.npy'))) for n in arrays})
    save_json(output/'manifest.json',meta)
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',required=True);p.add_argument('--date',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=prepare(a.market,a.output,a.date)
    print({k:m[k] for k in ('status','date','quotes','rows','entry_origins','event_bank_status')})
