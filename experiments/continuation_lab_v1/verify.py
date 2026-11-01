"""Independent receipts, integer clocks, raw gate holds, and scalar execution."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.mechanical_lab.data import dump
from experiments.regime_language_v1.measures import VwapContext, TimeAtPrice, scalar_time_mean

ROOT=Path(__file__).resolve().parents[2]
NS=1_000_000_000


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def scalar_execution(q, row):
    """Separate Decimal quote scan for the single fixed common120 exit contract."""
    t=q.available_ns.to_numpy(np.int64)
    bid=q.bid_nano.to_numpy(np.int64);ask=q.ask_nano.to_numpy(np.int64)
    e=int(t.searchsorted(int(row.t_ns)+100_000_000,'right'))
    assert t[e]-int(row.t_ns)<=2*NS
    side=int(row.side);entry=int(ask[e] if side==1 else bid[e]);due=int(t[e])+120*NS
    fee=300_000_000;R=Decimal(str(row.risk_hint))*NS
    request=None;reason=None
    for j in range(e,len(t)):
        if int(t[j])>=due:
            request=due;reason='TIME';break
        assert bool(q.valid.iloc[j]) and q.epoch.iloc[j]==q.epoch.iloc[e]
        if j>e:assert t[j]-t[j-1]<=30*NS
        mark=side*(int(bid[j] if side==1 else ask[j])-entry)-fee
        if Decimal(mark)<=-R:
            request=int(t[j]);reason='STOP';break
        if Decimal(mark)>=Decimal('1.5')*R:
            request=int(t[j]);reason='TAKE';break
    assert request is not None
    x=int(t.searchsorted(request+100_000_000,'right'))
    assert x<len(t) and t[x]-request<=2*NS
    for j in range(e,x+1):
        assert q.valid.iloc[j] and q.epoch.iloc[j]==q.epoch.iloc[e]
        if j>e:assert t[j]-t[j-1]<=30*NS
    net=(side*(int(bid[x] if side==1 else ask[x])-entry)-fee)/NS
    return dict(entry_ns=int(t[e]),trigger_ns=request,exit_ns=int(t[x]),reason=reason,net_per_oz=net)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();root=ROOT/args.run
    m=read(root/'manifest.json');assert m['status']=='COMPLETE'
    checks=0
    for path,digest in m['source_sha256'].items():
        assert sha(ROOT/path)==digest,(path,'source drift');checks+=1
        assert sha(root/'source_snapshot'/path)==digest;checks+=1
    for path,digest in m['artifacts'].items():assert sha(root/path)==digest,path;checks+=1
    inputs={}
    for d in m['days']:
        assert d['prefix_equal'];inputs.update(d['inputs'])
    for path,digest in inputs.items():assert sha(ROOT/path)==digest,path;checks+=1
    f=pd.read_parquet(root/'features.parquet');o=pd.read_parquet(root/'outcomes.parquet')
    assert len(f)==2645 and f.candidate_id.is_unique
    assert (f.context_release_ns<f.t_ns).all() and (f.frozen_release_ns<f.t_ns).all()
    assert (f.t_ns-f.gate_ns>=2*NS).all()
    assert (f.trough_ns<=f.gate_ns).all() and (f.peak_ns<=f.trough_ns).all()
    assert (f.rebound_r>=.3-1e-12).all()
    p=f[f.family=='pullback'];assert p.pullback_r.between(.5-1e-12,1.25+1e-12).all()
    assert (p.impulse_r>=1-1e-12).all()
    assert set(o.status)<=set(['COMPLETE','CENSORED','EMPTY','INELIGIBLE'])
    assert o[o.status!='COMPLETE'].net_per_oz.isna().all()
    assert not o.duplicated(['origin_index','side','scenario']).any()
    assert len(o)==4*int(f.origin_admitted.sum())
    complete=o[o.status=='COMPLETE']
    assert (complete.entry_ns>complete.signal_ns+np.where(complete.scenario=='base',100_000_000,250_000_000)).all()
    assert (complete.exit_ns>complete.trigger_ns+np.where(complete.scenario=='base',100_000_000,250_000_000)).all()
    # All model-facing rows are observations. Outcomes and actual account paths
    # remain physically separate; there is no implicit target in features.
    assert not any(x in f for x in ('net_per_oz','entry_price','exit_price','mfe_per_oz','mae_per_oz'))
    meta=sum([read(p) for p in sorted((root/'days').glob('*/scenes.json'))],[])
    atlas=read(root/'atlas.json');assert len(meta)==len(atlas)==15
    gates=money=profiles=vwaps=unknown_vwaps=0
    for item in meta:
        c=f[f.candidate_id==item['candidate_id']].iloc[0];q=pd.read_parquet(root/item['path'])
        gate=q[(q.available_ns>=int(c.gate_ns))&(q.available_ns<=int(c.t_ns))]
        mid=(gate.bid_nano.to_numpy(np.int64)+gate.ask_nano.to_numpy(np.int64))/(2*NS)
        assert (c.side*(mid-c.trough_price)>=.3*c.risk_hint-1e-9).all()
        assert gate.valid.all() and gate.epoch.nunique()==1 and gate.available_ns.diff().dropna().le(30*NS).all();gates+=1
        observed=o[(o.origin_index==c.origin_index)&(o.side==c.side)&(o.scenario=='base')].iloc[0]
        independent=scalar_execution(q,c)
        for key in ('entry_ns','trigger_ns','exit_ns','reason'):assert independent[key]==observed[key],(c.candidate_id,key)
        assert abs(independent['net_per_oz']-observed.net_per_oz)<1e-10;money+=1
        mean,support=scalar_time_mean(q,int(c.start_ns),int(c.t_ns))
        prof=TimeAtPrice(q).profile(int(c.start_ns),int(c.t_ns))
        assert support==int(c.t_ns)-int(c.start_ns)
        assert abs(mean-prof['time_mean'])<1e-8;profiles+=1
        ctx=pd.read_parquet(root/'days'/c.date/'context.parquet')
        v=VwapContext(ctx).at(int(c.t_ns))
        for minutes in (60,240):
            a=v['anchors']['rolling_'+str(minutes)+'m']
            if a['known']:
                assert abs(a['value']-c['vwap'+str(minutes)])<1e-8
                assert abs(a['sigma']-c['sigma'+str(minutes)])<1e-7;vwaps+=1
            else:
                assert np.isnan(c['vwap'+str(minutes)]) and np.isnan(c['sigma'+str(minutes)])
                unknown_vwaps+=1
    for item in atlas:
        assert sha(root/item['file'])==item['sha256'];checks+=1
        assert abs(item['time_mass_seconds']-item['episode_seconds'])<1e-8
    accounts=pd.read_parquet(root/'accounts.parquet');trades=pd.read_parquet(root/'trades.parquet')
    for a in accounts.itertuples():
        ts=trades[(trades.recipe==a.recipe)&(trades.scenario==a.scenario)&(trades.direction==a.direction)]
        assert len(ts)==a.trades and abs(ts.pnl_usd.sum()-a.net_usd)<1e-6
        assert (ts.signal_ns.iloc[1:].to_numpy()>ts.exit_ns.iloc[:-1].to_numpy()).all()
        assert a.capital_curve_complete == (a.unresolved==0)
    body=(root/'report.html').read_text(encoding='utf-8');links=0
    for href in re.findall(r'(?:src|href)="([^"]+)"',body):
        if href=='validation.json':continue
        assert (root/href).exists(),href;links+=1
    report=read(root/'report_receipt.json')
    assert report['html_sha256']==sha(root/'report.html')
    assert report['reporter_sha256']==sha(Path(__file__).with_name('report.py'))
    result=dict(status='PASS',hash_checks=checks,input_files=len(inputs),proposals=len(f),
                native_gate_checks=gates,decimal_execution_checks=money,scalar_profile_checks=profiles,
                independent_vwap_checks=vwaps,unknown_vwap_mask_checks=unknown_vwaps,exact_prefix_days=len(m['days']),
                all_phase_prefix_rows=sum(d['source_prefix_rows'] for d in m['days']),
                feature_prefix_rows=sum(d['feature_prefix_rows'] for d in m['days']),
                account_checks=len(accounts),atlas_mass_checks=len(atlas),html_links=links,
                verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(root/'manifest.json'),
                economic_admission='NOT_ESTABLISHED',browser_rendered=False)
    dump(root/'validation.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
