"""Check persisted exact-age results against saved native paths and source receipts."""
import argparse
import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,HERE,read
from .run_alignment import DEFAULT
from .verify import Links


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');cfg=read(folder/'config.json');r=read(folder/'report_receipt.json')
    assert m['status']=='COMPLETE_DESCRIPTIVE_AGE_ALIGNMENT'
    hashes=0
    for bank,base in ((m['source_sha256'],ROOT),(m['input_sha256'],ROOT),(m['artifacts'],folder),(r['artifacts'],folder)):
        for name,digest in bank.items():assert sha(base/name)==digest,name;hashes+=1
    assert r['source_sha256']==sha(HERE/'alignment_report.py')
    assert r['input_manifest_sha256']==sha(folder/'manifest.json')
    rows=read(folder/'matches.json');statuses=pd.read_parquet(folder/'statuses.parquet')
    assert len(statuses)==36*2*2
    max_error=0.;intervals={}
    for row in rows:
        assert row['available_ns']<row['query_context_start_ns']<=row['query_ns']
        assert row['age_ns']>0 and row['net_direction_conflict']==row['slot_direction_conflict']==0
        method=next(x for x in cfg['alignment_methods'] if x['name']==row['method'])
        expected=sum(w*row[k] for k,w in method['weights'].items() if w)
        max_error=max(max_error,abs(expected-row['score']))
        assert np.isclose(expected,row['score'],rtol=2*np.finfo(np.float32).eps,atol=2e-15)
        assert row['total_duration_factor']<=method['total_duration_factor']+1e-12
        assert row['slot_duration_factor']<=method['slot_duration_factor']+1e-12
        for key,limit in [('shape_rmse','max_shape_rmse'),('amplitude_rmse','max_amplitude_rmse'),
                          ('trajectory_rmse','max_trajectory_rmse'),('score','max_score')]:assert row[key]<=cfg['acceptance'][limit]
        if method.get('mechanism_required'):
            assert row['book_common_fraction']>=cfg['mechanism_min_common_fraction']
            assert row['book_min_slot_fraction']>=cfg['mechanism_min_slot_fraction']
            assert row['mechanism_rmse']<=cfg['acceptance']['max_mechanism_rmse']
        key=(row['query_token_id'],row['method'],row['scope']);a=row['context_start_ns'];b=row['available_ns']
        assert all(not (a<=y and x<=b) for x,y in intervals.get(key,[]))
        intervals.setdefault(key,[]).append((a,b))
    samples=0;contexts=0;age_pairs=0
    for case in read(folder/'atlas_index.json'):
        qrows=pd.read_parquet(folder/(case['case_id']+'_query_tokens.parquet'))
        qage=int(qrows.available_ns.iloc[-1])-int(qrows.start_ns.iloc[-1])
        for name in ['query']+[k for k,v in case['methods'].items() if v is not None]:
            stem=case['case_id']+'_'+name
            f=pd.read_parquet(folder/(stem+'_tokens.parquet'));q=pd.read_parquet(folder/(stem+'_quotes.parquet'))
            p=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2;qt=q.available_ns.to_numpy(np.int64)
            features=np.load(folder/(stem+'_features.npz'),allow_pickle=False);base=int(f.start_index.min())
            if name.startswith('age_'):
                assert int(f.available_ns.iloc[-1])-int(f.start_ns.iloc[-1])==qage
                assert f.available_ns.max()<int(qrows.start_ns.min());age_pairs+=1
            for i,row in enumerate(f.itertuples()):
                assert features['book_release_ns'][i]<row.available_ns
                assert (features['book'][i,~features['book_known'][i]]==0).all()
                assert qt[int(row.source_index)-base]<=row.available_ns
                for k in range(1,17):
                    clock=int(row.start_ns)+(int(row.available_ns)-int(row.start_ns))*k//16
                    ix=min(int(row.source_index)-base,max(int(row.start_index)-base,int(qt.searchsorted(clock,side='right'))-1))
                    expected=(int(p[ix])-int(row.start_mid_nano))/(int(row.amplitude_ticks)*100_000_000)
                    assert np.isclose(expected,features['raw_shape'][i,k-1],atol=1e-6);samples+=1
            direct=(int(f.mid_nano.iloc[-1])-int(f.start_mid_nano.iloc[0]))/(int(f.amplitude_ticks.iloc[0])*100_000_000)
            assert np.isclose(features['amplitude'][:,0].sum(),direct,atol=1e-6)
            contexts+=1
    parser=Links();parser.feed((folder/'report.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (folder/link).is_file(),link
    result=dict(status='PASS_BOUNDED_AGE_ALIGNMENT_CHECKS',manifest_sha256=sha(folder/'manifest.json'),
        verifier_sha256=sha(__file__),hash_checks=hashes,ranked_neighbor_checks=len(rows),
        exact_age_atlas_pairs=age_pairs,scalar_atlas_shape_samples=samples,joined_context_checks=contexts,
        local_links=len(parser.links),max_saved_score_reconstruction_error=max_error,
        scope='Same-task source, arithmetic, masks and causal-clock checks; not human semantic or economic validation')
    dump(folder/'validation.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=DEFAULT);main(p.parse_args().run)
