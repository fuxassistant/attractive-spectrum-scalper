"""Saved-record, mass conservation and source-clock checks for the regime probe."""
import argparse
import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from experiments.story_similarity_v2.verify import Links
from .run import ROOT,HERE,DEFAULT,read
from .measures import scalar_time_mean,NS


def profile_check(p):
    mass=np.asarray(p['bin_seconds']);addresses=np.asarray(p['bin_low_nano'])
    assert (mass>=0).all() and (np.diff(addresses)>0).all()
    assert np.isclose(mass.sum(),p['supported_seconds'],rtol=0,atol=1e-8)
    assert np.isclose(p['coverage'],p['supported_seconds']/p['duration_seconds'])
    assert p['known']==(p['coverage']==1.)
    if 'first_half' in p:
        a=p['first_half'];b=p['second_half'];profile_check(a);profile_check(b)
        assert a['left_ns']==p['left_ns'] and a['right_ns']==b['left_ns'] and b['right_ns']==p['right_ns']
        assert np.isclose(a['supported_seconds']+b['supported_seconds'],p['supported_seconds'])
        first=dict(zip(a['bin_low_nano'],np.asarray(a['bin_seconds'])/a['supported_seconds']))
        second=dict(zip(b['bin_low_nano'],np.asarray(b['bin_seconds'])/b['supported_seconds']))
        expected=sum(min(first.get(k,0),second.get(k,0)) for k in first.keys()|second.keys())
        assert np.isclose(expected,p['half_profile_overlap'])


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');r=read(folder/'report_receipt.json')
    assert m['status']=='COMPLETE_DESCRIPTIVE_REGIME_PROBE';hashes=0
    for bank,base in ((m['source_sha256'],ROOT),(m['input_sha256'],ROOT),(m['artifacts'],folder),(r['artifacts'],folder)):
        for name,value in bank.items():assert sha(base/name)==value,name;hashes+=1
    assert r['source_sha256']==sha(HERE/'report.py') and r['manifest_sha256']==sha(folder/'manifest.json')
    parent=ROOT/read(folder/'config.json')['parent'];pm=read(parent/'manifest.json')
    dependency='experiments/story_similarity_v2/run.py'
    recorded={k.replace('\\','/'):v for k,v in pm['source_sha256'].items()}
    assert sha(ROOT/dependency)==recorded[dependency]
    tokens=read(folder/'tokens.json');sentences=read(folder/'sentences.json')
    lookup={t['token_id']:t for t in tokens};assert len(lookup)==len(tokens)==108 and len(sentences)==36
    checks=0
    for row in tokens+sentences:
        profile_check(row['profile'])
        assert row['profile']['left_ns']==row['start_ns'] and row['profile']['right_ns']==row['available_ns']
        for anchor,clock in [(row['anchor_start'],row['start_ns']),(row['anchor_end'],row['available_ns'])]:
            assert anchor['release_ns']<clock and anchor['endpoint_ns']<=anchor['release_ns']
        for a in row['anchor_relations'].values():
            if a['relative_identity_residual'] is not None:assert abs(a['relative_change_A']-(a['price_change_A']-a['line_change_A']))<1e-10
        if 'token_ids' in row:
            parts=[lookup[k] for k in row['token_ids']]
            assert parts[0]['start_ns']==row['start_ns'] and parts[-1]['available_ns']==row['available_ns']
            assert parts[0]['available_ns']==parts[1]['start_ns'] and parts[1]['available_ns']==parts[2]['start_ns']
        checks+=1
    heat_checks=0
    for row in sentences:
        if 'case_id' not in row:continue
        name=row['case_id'];a=np.load(folder/(name+'_heatmap.npz'),allow_pickle=False)
        edges=a['time_edges_ns'];heat=a['time_fraction'];assert np.isfinite(heat).all()
        assert edges[0]==row['start_ns'] and edges[-1]==row['available_ns']
        assert np.allclose(heat.sum(axis=0),1,rtol=0,atol=1e-10)
        collapsed=heat@(np.diff(edges)/NS)
        expected=dict(zip(row['profile']['bin_low_nano'],row['profile']['bin_seconds']))
        for low,value in zip(a['price_low_nano'],collapsed):assert np.isclose(value,expected.get(int(low),0),rtol=0,atol=1e-8)
        q=pd.read_parquet(folder/(name+'_quotes.parquet'))
        mean,support=scalar_time_mean(q,row['start_ns'],row['available_ns'])
        assert np.isclose(mean,row['profile']['time_mean'],rtol=0,atol=1e-8)
        assert support==row['available_ns']-row['start_ns'];heat_checks+=1
    parser=Links();parser.feed((folder/'report.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (folder/link).is_file(),link
    result=dict(status='PASS_BOUNDED_REGIME_PROBE_CHECKS',manifest_sha256=sha(folder/'manifest.json'),
        verifier_sha256=sha(__file__),hash_checks=hashes,profile_and_anchor_records=checks,
        heatmap_mass_conservation_scenes=heat_checks,local_links=len(parser.links),
        limitation='Arithmetic and causal-clock checks; no semantic class, independent human, model or economic admission')
    dump(folder/'validation.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=DEFAULT);main(p.parse_args().run)
