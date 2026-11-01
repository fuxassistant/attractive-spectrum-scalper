"""Exact incoming/outgoing contribution to a rolling volume-weighted mean.

This post-experiment descriptive overlay changes neither entries nor outcomes.
Contributions explain arithmetic, not hidden intention or causal market impact.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.mechanical_lab.data import dump

MINUTE=60_000_000_000


def decompose(frame, clock_ns, window=60, lag=15):
    if not 0<lag<=window:
        raise ValueError('Require 0 < lag <= window')
    av=frame.available_ns.to_numpy(np.int64)
    j=int(av.searchsorted(clock_ns,'left'))-1
    out=dict(known=False,window_minutes=window,lag_minutes=lag,
             incoming=None,outgoing=None,total=None,old_vwap=None,new_vwap=None,
             incoming_volume=None,outgoing_volume=None,new_volume=None)
    if j<0:return out
    end=int(frame.t_ns.iloc[j]);f=frame.iloc[:j+1]
    f=f[(f.t_ns>end-(window+lag)*MINUTE)&(f.t_ns<=end)]
    good=(len(f)==window+lag and f.t_ns.diff().dropna().eq(MINUTE).all()
          and f.known.all() and f.interval_open.all() and f.epoch.nunique()==1
          and 0<=clock_ns-end<=120_000_000_000)
    if not good:return out
    old=f.iloc[:window];new=f.iloc[lag:];added=f.iloc[-lag:];removed=f.iloc[:lag]
    w0=float(old.volume.sum());w1=float(new.volume.sum())
    if w0<=0 or w1<=0:return out
    old_mean=float(old.pv.sum()/w0);new_mean=float(new.pv.sum()/w1)
    incoming=float((added.pv.sum()-old_mean*added.volume.sum())/w1)
    outgoing=float(-(removed.pv.sum()-old_mean*removed.volume.sum())/w1)
    total=new_mean-old_mean
    assert abs(incoming+outgoing-total)<1e-8
    out.update(known=True,incoming=incoming,outgoing=outgoing,total=total,
               old_vwap=4000+old_mean,new_vwap=4000+new_mean,
               incoming_volume=float(added.volume.sum()),outgoing_volume=float(removed.volume.sum()),new_volume=w1,
               source_release_ns=int(frame.available_ns.iloc[j]),endpoint_ns=end)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();root=Path(args.run)
    target=root/'anchor_diagnostics';target.mkdir(exist_ok=False)
    features=pd.read_parquet(root/'features.parquet');rows=[];prefix=0
    for date,f in features.groupby('date'):
        context=pd.read_parquet(root/'days'/date/'context.parquet')
        for r in f.itertuples():
            for window in (60,240):
                value=decompose(context,int(r.t_ns),window)
                if value['known']:
                    assert abs(value['total']-getattr(r,'slope'+str(window)+'_r')*r.risk_hint)<1e-8
                rows.append(dict(candidate_id=int(r.candidate_id),date=date,**value))
        for r in f.iloc[::max(1,len(f)//8)].itertuples():
            cut=context[context.available_ns<int(r.t_ns)]
            for window in (60,240):
                assert decompose(context,int(r.t_ns),window)==decompose(cut,int(r.t_ns),window);prefix+=1
    out=pd.DataFrame(rows);out.to_parquet(target/'decomposition.parquet',index=False)
    known=out[out.known]
    receipt=dict(status='PASS',source_sha256=sha(Path(__file__)),parent_manifest_sha256=sha(root/'manifest.json'),
                 features_sha256=sha(root/'features.parquet'),output_sha256=sha(target/'decomposition.parquet'),
                 rows=len(out),known=len(known),identity_checks=len(known),prefix_checks=prefix,
                 opposing_contribution_rows=int((known.incoming*known.outgoing<0).sum()),
                 outgoing_dominates_abs_rows=int((known.outgoing.abs()>known.incoming.abs()).sum()),
                 interpretation='Arithmetic attribution of rolling VWAP changes; no order-flow intent; post-experiment diagnostic, no changed signals or monetary comparisons')
    dump(target/'receipt.json',receipt)
    print(receipt)


if __name__=='__main__':main()
