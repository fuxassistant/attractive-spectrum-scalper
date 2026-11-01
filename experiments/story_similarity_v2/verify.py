"""Independent saved-record arithmetic and source/clock checks for the comparison."""
import argparse
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,HERE,read


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');r=read(folder/'report_receipt.json');cfg=read(folder/'config.json')
    assert m['status']=='COMPLETE_DESCRIPTIVE_SIMILARITY'
    hashes=0
    for bank,base in ((m['source_sha256'],ROOT),(m['input_sha256'],ROOT),(m['artifacts'],folder),(r['artifacts'],folder)):
        for name,value in bank.items():assert sha(base/name)==value,name;hashes+=1
    assert sha(HERE/'report.py')==r['source_sha256']
    assert sha(folder/'manifest.json')==r['input_manifest_sha256']
    matches=read(folder/'matches.json');statuses=pd.read_parquet(folder/'statuses.parquet')
    assert len(statuses)==36*2*5
    checked=0;coherent=0;max_score_error=0.
    for row in matches:
        assert row['available_ns']<row['query_context_start_ns']<=row['query_ns']
        assert len(row['context_token_ids'])==3 and len(row['query_context_token_ids'])==3
        method=next(x for x in cfg['methods'] if x['name']==row['method'])
        if method['name']=='legacy':assert row['score']==row['legacy']
        else:
            expected=sum(weight*row[name] for name,weight in method['weights'].items() if weight)
            # Stored feature blocks are float32. Multiplication in the vectorized
            # search rounds each weighted float32 block before float64 addition.
            # Reconstruct independently in float64, allowing two relative ulps
            # of float32 rather than demanding fictitious double precision.
            max_score_error=max(max_score_error,abs(expected-row['score']))
            assert np.isclose(expected,row['score'],rtol=2*np.finfo(np.float32).eps,atol=2e-15)
            assert row['shape_rmse']<=cfg['acceptance']['max_shape_rmse']
            assert row['amplitude_rmse']<=cfg['acceptance']['max_amplitude_rmse']
            assert row['score']<=cfg['acceptance']['max_score']
        if method.get('trajectory_required'):
            assert row['total_duration_factor']<=method['total_duration_factor']+1e-12
            assert row['slot_duration_factor']<=method['slot_duration_factor']+1e-12
            assert row['trajectory_rmse']<=cfg['acceptance']['max_trajectory_rmse']
            assert row['net_direction_conflict']==row['slot_direction_conflict']==0
            q=row['query_total_net'];c=row['candidate_total_net'];floor=cfg['direction_floor_amplitudes']
            assert not (abs(q)>=floor and abs(c)>=floor and q*c<0)
            coherent+=1
        if method.get('mechanism_required'):
            assert row['book_common_fraction']>=cfg['mechanism_min_common_fraction']
            assert row['book_min_slot_fraction']>=cfg['mechanism_min_slot_fraction']
        checked+=1
    samples=0;contexts=0
    for case in read(folder/'atlas_index.json'):
        for name in ['query']+[n for n,v in case['methods'].items() if v is not None]:
            stem=case['case_id']+'_'+name
            rows=pd.read_parquet(folder/(stem+'_tokens.parquet'))
            quotes=pd.read_parquet(folder/(stem+'_quotes.parquet'))
            f=np.load(folder/(stem+'_features.npz'),allow_pickle=False)
            base=int(rows.start_index.min())
            times=quotes.available_ns.to_numpy(np.int64)
            price=(quotes.bid_nano.to_numpy(np.int64)+quotes.ask_nano.to_numpy(np.int64))//2
            assert rows.available_ns.max()<=case['query_ns']
            for i,row in enumerate(rows.itertuples()):
                assert f['book_release_ns'][i]<row.available_ns
                assert (f['book'][i,~f['book_known'][i]]==0).all()
                for k in range(1,17):
                    at=int(row.start_ns)+(int(row.available_ns)-int(row.start_ns))*k//16
                    where=min(int(row.source_index)-base,max(int(row.start_index)-base,int(times.searchsorted(at,side='right'))-1))
                    expected=(int(price[where])-int(row.start_mid_nano))/(int(row.amplitude_ticks)*100_000_000)
                    assert np.isclose(expected,f['raw_shape'][i,k-1],atol=1e-6)
                    samples+=1
            # The local signed endpoints must join into the actual whole-prefix displacement.
            reconstructed=float(f['amplitude'][:,0].sum())
            direct=(int(rows.mid_nano.iloc[-1])-int(rows.start_mid_nano.iloc[0]))/(int(rows.amplitude_ticks.iloc[0])*100_000_000)
            assert np.isclose(reconstructed,direct,atol=1e-6)
            contexts+=1
    old=read(ROOT/'research_results/story_similarity_v2/verified_v2/atlas_index.json')
    bad=next(c for c in old if c['case_id']=='120000_slow')['methods']['rhythm_gate']['token_id']
    new=[x for x in matches if x['scope']=='previous' and x['method'] in ('rhythm_gate','book_gate')
         and x['scale']=='slow' and x['query_ns']==int(pd.Timestamp('2026-08-25T12:00:00Z').value)]
    assert all(x['token_id']!=bad for x in new)
    parser=Links();parser.feed((folder/'report.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (folder/link).is_file(),link
    result=dict(status='PASS_BOUNDED_SIMILARITY_CHECKS',manifest_sha256=sha(folder/'manifest.json'),
        verifier_sha256=sha(__file__),hash_checks=hashes,ranked_neighbor_checks=checked,
        whole_trajectory_gate_checks=coherent,scalar_atlas_shape_samples=samples,
        exact_joined_context_checks=contexts,local_links=len(parser.links),
        max_saved_score_reconstruction_error=max_score_error,
        score_relative_tolerance=2*float(np.finfo(np.float32).eps),
        exposed_noon_wrong_direction_regression_passed=True,
        scope='Same-task separate arithmetic and invariants; not independent human semantic or economic validation')
    dump(folder/'validation.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=read(HERE/'config.json')['output']);main(p.parse_args().run)
