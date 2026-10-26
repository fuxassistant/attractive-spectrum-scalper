"""Saved labels, feature boundary, raw paths and predictive metric verification."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.build_crossing_path_target_v6 import sha,save


def main():
    c=json.loads(Path('configs/research/favorable_series_v1.json').read_text())
    base=json.loads(Path(c['base_config']).read_text());out=Path(c['output']);report=Path(c['report'])
    stats=json.loads((report/'summary.json').read_text())
    assert stats['status']=='COMPLETE'
    bound=dict(stats['input_hashes']);bound.update(stats['outputs'])
    source_manifest=json.loads((Path(c['source'])/'manifest.json').read_text())
    for day in source_manifest['days']:
        bound.update(day['inputs'])
    raw_probes=0;max_error=0.;boundaries={'L':{'mfe':0,'endpoint':0},'S':{'mfe':0,'endpoint':0}}
    totals={s:dict(mfe=0,endpoint=0,mfe_nonpositive_endpoint=0,endpoint_risk_before=0) for s in ('L','S')}
    for date in base['data']['dates']:
        a=pd.read_parquet(out/f'{date}_cells.parquet')
        known=a.hit_known.to_numpy();observed=a.profile_known.to_numpy()
        assert len(a)==90666 and not a.duplicated(['t_ns','scale','role']).any()
        for s in ('L','S'):
            adverse=np.maximum(-a[f'mae_R_{s}'].to_numpy(),0)
            for kind,value in (('mfe',a[f'mfe_R_{s}'].to_numpy()),('endpoint',a[f'net_R_{s}'].to_numpy())):
                margin=value-c['ratio']*adverse
                expected=np.where(known,(observed&(value>0)&(margin>=-1e-10)).astype(int),-1)
                assert np.array_equal(expected,a[f'{kind}_qualified_{s}'])
                assert np.allclose(margin[observed],a.loc[observed,f'{kind}_margin_{s}'],atol=1e-12)
                assert a.loc[~observed,f'{kind}_margin_{s}'].isna().all()
                boundaries[s][kind]+=int((observed&(margin<0)&(margin>=-1e-10)&(value>0)).sum())
                totals[s][kind]+=int((expected==1).sum())
            totals[s]['mfe_nonpositive_endpoint']+=int(((a[f'mfe_qualified_{s}']==1)&(a[f'net_R_{s}']<=0)).sum())
            totals[s]['endpoint_risk_before']+=int(((a[f'endpoint_qualified_{s}']==1)&(a[f'risk_before_cross_{s}']==1)).sum())
            assert not ((a[f'endpoint_qualified_{s}']==1)&(a[f'mfe_qualified_{s}']!=1)).any()
        assert not ((a.endpoint_qualified_L==1)&(a.endpoint_qualified_S==1)).any()
        # Direct quote audit: fixed evenly spaced observed cells, both sides.
        day=MonthDay(Path(base['data']['market_root'])/date[:7]/date,verify=True);q=QuoteTape.from_day(day)
        ix=np.flatnonzero(observed);positions=ix[np.linspace(0,len(ix)-1,3,dtype=int)]
        for i in positions:
            row=a.iloc[i];e=int(row.entry_ordinal);x=int(row.cross_ordinal)
            assert q.quote_t[e]==row.entry_ns and q.quote_t[x]==row.cross_ns
            assert q.valid[e:x+1].all() and (q.epoch[e:x+1]==q.epoch[e]).all()
            for s in ('L','S'):
                net=((q.bid[e:x+1]-q.ask[e]) if s=='L' else (q.bid[e]-q.ask[e:x+1]))/1e9-.3
                vals=dict(net_R=net[-1]/row.R_event,mae_R=net.min()/row.R_event,mfe_R=net.max()/row.R_event)
                for key,val in vals.items():
                    err=abs(val-row[f'{key}_{s}']);max_error=max(max_error,err);assert err<1e-8
                assert int((net[:-1]<=-row.R_event).any())==row[f'risk_before_cross_{s}']
                raw_probes+=1
        QUOTE_CACHE.clear()
    origins=pd.read_parquet(out/'origins.parquet');pred=pd.read_parquet(out/'probe_predictions.parquet')
    assert np.array_equal(origins.t_ns,pred.t_ns)
    assert (origins.loc[pred.admitted,'source_available_ns']<=origins.loc[pred.admitted,'t_ns']).all()
    names=json.loads((out/'feature_names.json').read_text());x=np.load(out/'x.npy',mmap_mode='r')
    allowed=stats['probe']['base_features']+stats['probe']['flow_features']
    assert len(allowed)==69 and len(set(allowed))==69
    assert not any('context__' in n or 'fft_' in n or 'cross_' in n or 'target' in n for n in allowed)
    for s in ('L','S'):
        mask=pred.admitted & (pred.split=='test');y=pred.loc[mask,f'endpoint_{s}']
        for arm in ('base','flow'):
            p=pred.loc[mask,f'p_{s}_{arm}'];m=stats['probe']['per_side'][s][arm]['test']
            assert abs(roc_auc_score(y,p)-m['auc'])<1e-12
            assert abs(brier_score_loss(y,p)-m['brier'])<1e-12
    pairs=pd.read_parquet(out/'matched_pairs.parquet');l=pairs.left.to_numpy();r=pairs.right.to_numpy()
    assert (origins.date.to_numpy()[l]!=origins.date.to_numpy()[r]).all()
    assert (origins.split.to_numpy()[l]=='test').all() and (origins.split.to_numpy()[r]=='test').all()
    for name,cap in [('minutes_since_london',30),('spread_ticks',1)]+[(f'bb{s}_position',.5) for s in (1,3,6)]+[(f'bb{s}_log_half_ticks',.35) for s in (1,3,6)]:
        j=names.index(name);assert (abs(x[l,j]-x[r,j])<=cap+1e-6).all()
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError('bound file drift '+path)
    visual=json.loads((report/'visual_findings.json').read_text())
    assert visual['source_summary_sha256']==sha(report/'summary.json')
    for name,digest in visual['images'].items():assert sha(report/name)==digest
    result=dict(status='PASS',cells=stats['cells'],origins=len(origins),raw_paths=raw_probes,
        maximum_raw_profile_error=max_error,verified_bound_files=len(bound),
        numeric_boundary_inclusions=boundaries,totals=totals,matched_pairs=len(pairs),
        images=visual['images'],
        source_hashes={p:sha(p) for p in ['configs/research/favorable_series_v1.json','spectrum/research_v6/favorable.py',
            'scripts/run_favorable_series_v1.py','scripts/report_favorable_series_v1.py','scripts/verify_favorable_series_v1.py','tests/test_favorable_series_v1.py']},
        checks=['independent saved-label formula reconstruction','unknown/no-hit masks and same-cross polarity',
            'full-quote net/MFE/MAE and pre-cross risk order','frozen feature publication <= decision clock',
            '69-feature allow-list without future/long-context columns','saved TEST AUC and Brier independently recalculated',
            'matched pairs obey clock/spread/position/width limits and cross dates','all input/output/image hashes'],
        admission='DESCRIPTIVE_TARGET_AUDIT_AND_EXPLORATORY_PREDICTIVE_PROBE_NO_POLICY_EDGE')
    save(report/'completion_receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('images','source_hashes')},indent=2))


if __name__=='__main__':main()
