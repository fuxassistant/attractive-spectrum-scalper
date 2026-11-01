"""Independent label accounting and scalar physical-time checks on real windows."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.mechanical_lab.data import dump
from experiments.continuation_lab_v1.run import load_day
from .future_description import describe_future

ROOT=Path(__file__).resolve().parents[2]
NS=1_000_000_000


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def scalar_time(quotes,left,right,origin,R):
    qt=quotes.available_ns.to_numpy(np.int64)
    lo=max(0,int(qt.searchsorted(left,'right'))-1);hi=int(qt.searchsorted(right,'left'))
    q=quotes.iloc[lo:hi];mid=left+(right-left)//2
    weights={};support=total=inside=0
    hsupport=[0,0];hsum=[0,0]
    last_epoch=None
    for i,r in enumerate(q.itertuples()):
        assert r.valid and r.bid_nano>0 and r.ask_nano>r.bid_nano and r.epoch>=0
        if last_epoch is not None:assert last_epoch==r.epoch
        last_epoch=r.epoch
        start=max(left,int(r.available_ns))
        following=int(qt[lo+i+1]) if lo+i+1<len(qt) else right
        end=min(right,following,int(r.available_ns)+30*NS)
        dt=max(0,end-start);price=(int(r.bid_nano)+int(r.ask_nano))//2
        support+=dt;total+=price*dt
        key=price//500_000_000;weights[key]=weights.get(key,0)+dt
        if abs(price/NS-origin)<=.5*R:inside+=dt
        for h,(a,b) in enumerate([(left,mid),(mid,right)]):
            part=max(0,min(end,b)-max(start,a));hsupport[h]+=part;hsum[h]+=price*part
    assert support==right-left
    return dict(concentration=max(weights.values())/support,dwell=inside/support,
                migration=(hsum[1]/hsupport[1]-hsum[0]/hsupport[0])/NS/R)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();root=ROOT/args.run
    manifest=read(root/'manifest.json');assert manifest['status']=='COMPLETE';checks=0
    for path,digest in manifest['source_sha256'].items():
        assert sha(ROOT/path)==digest;assert sha(root/'source_snapshot'/path)==digest;checks+=2
    for path,digest in manifest['inputs'].items():assert sha(ROOT/path)==digest;checks+=1
    for path,digest in manifest['artifacts'].items():assert sha(root/path)==digest;checks+=1
    cfg=read(root/'config.json');parent=ROOT/cfg['parent'];f=pd.read_parquet(parent/'features.parquet')
    o=pd.read_parquet(parent/'outcomes.parquet');o=o[o.scenario=='base']
    p=pd.read_parquet(root/'paired_targets.parquet');desc=pd.read_parquet(root/'future_descriptions.parquet')
    assert p.origin_index.is_unique and len(p)==len(desc)==int(f.origin_admitted.sum())
    assert np.isclose(p.exact_decision_weight.sum(),p.drop_duplicates(['date','t_ns','risk_nano']).shape[0])
    for name,side in [('long',1),('short',-1)]:
        reference=o[o.side==side].set_index('origin_index').reindex(p.origin_index)
        filled=reference.status.eq('COMPLETE').to_numpy();empty=reference.status.eq('EMPTY').to_numpy()
        values=p['future_u_'+name].to_numpy(float)
        np.testing.assert_allclose(values[filled],reference.net_per_oz.to_numpy()[filled],atol=0,rtol=0)
        assert (values[empty]==0).all() and np.isnan(values[~(filled|empty)]).all()
        matured=p['future_available_ns_'+name]
        assert matured.dtype==pd.Int64Dtype()
        assert np.array_equal(matured[filled].to_numpy(np.int64),reference.exit_ns.to_numpy(np.int64)[filled])
        assert np.array_equal(matured[empty].to_numpy(np.int64),p.t_ns.to_numpy(np.int64)[empty]+2*NS)
        assert matured[~(filled|empty)].isna().all()
    k=p.future_pair_known
    np.testing.assert_allclose((p.future_common_r+p.future_direction_advantage_r)[k],p.future_u_long_r[k],atol=1e-12)
    np.testing.assert_allclose((p.future_common_r-p.future_direction_advantage_r)[k],p.future_u_short_r[k],atol=1e-12)
    assert p.loc[~k,'future_joint_profit_state'].eq('UNKNOWN').all()
    assert p.loc[~k,'future_common_r'].isna().all()
    for name in p:
        if name not in ['origin_index','date','t_ns','family','level_kind','side','risk_hint','risk_nano','exact_decision_weight']:
            assert name.startswith('future_')
    for name in desc:
        if name not in ['origin_index','date']:assert name.startswith('future_')
    known=desc.future_description_known
    for name in ('future_range_r','future_efficiency_1s','future_time_concentration','future_origin_band_dwell_fraction'):
        assert desc.loc[~known,name].isna().all()
    assert desc.loc[known,'future_efficiency_1s'].between(-1e-12,1+1e-12).all()
    assert desc.loc[known,'future_origin_band_dwell_fraction'].between(0,1).all()
    cost=pd.read_parquet(root/'exact_costs.parquet')
    residual=(cost.future_mid_move_same_executions-cost.future_actual_total_cost-cost.net_per_oz).abs()
    assert residual.max()<1e-10
    scalar=prefix=0;parent_cfg=read(parent/'config.json')
    for date in parent_cfg['dates']:
        q,_,_,_,_,deadline=load_day(date,parent_cfg)
        candidates=f[(f.date==date)&f.origin_admitted].merge(desc[['origin_index','future_description_known']],on='origin_index')
        candidates=candidates[candidates.future_description_known]
        # Evenly spaced in causal proposal order, without outcome selection.
        for i in np.linspace(0,len(candidates)-1,3,dtype=int):
            r=candidates.iloc[i];clock=int(r.t_ns);end=clock+120*NS
            reference=desc[desc.origin_index==r.origin_index].iloc[0]
            result=scalar_time(q,clock,end,float(r.decision_price),float(r.risk_hint))
            assert abs(result['concentration']-reference.future_time_concentration)<1e-9
            assert abs(result['dwell']-reference.future_origin_band_dwell_fraction)<1e-9
            assert abs(result['migration']-reference.future_time_center_migration_r)<1e-8;scalar+=1
            a=describe_future(q,clock,float(r.decision_price),float(r.risk_hint),observed_through_ns=deadline)
            clipped=q[q.available_ns<end]
            b=describe_future(clipped,clock,float(r.decision_price),float(r.risk_hint),observed_through_ns=end)
            assert a==b;prefix+=1
    receipt=read(root/'report_receipt.json')
    assert sha(root/'report.html')==receipt['html_sha256'] and sha(root/'opportunity_vs_value.png')==receipt['image_sha256']
    assert sha(Path(__file__).with_name('report.py'))==receipt['reporter_sha256']
    links=0
    for href in re.findall(r'(?:href|src)="([^"]+)"',(root/'report.html').read_text(encoding='utf-8')):
        if href.startswith('https:') or href=='validation.json':continue
        assert (root/href).exists(),href;links+=1
    result=dict(status='PASS',hash_checks=checks,paired_rows=len(p),exact_cost_identities=len(cost),
                scalar_native_time_checks=scalar,future_endpoint_truncation_checks=prefix,
                known_future_descriptions=int(known.sum()),html_local_links=links,
                manifest_sha256=sha(root/'manifest.json'),verifier_sha256=sha(Path(__file__)),
                training=False,causal_predictive_skill_tested=False,economic_admission='NOT_ESTABLISHED')
    dump(root/'validation.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
