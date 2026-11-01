"""A separate, immutable observation-age experiment after the snapshot-grid finding."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_adapter_v1.adapter import live_states
from experiments.scalper_adapter_v1.tokens import quote_arrays
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,HERE,read,load_corpus,query_clocks,summarize
from .search import SimilaritySearch
from .features import extract
from .age_aligned import prepare_aligned,rank_aligned,context_frame

DEFAULT='research_results/story_similarity_v2/age_aligned_v1'


def save_frame(folder,stem,frame,search):
    s=search.sessions[frame.session_id.iloc[0]]
    frame.to_parquet(folder/(stem+'_tokens.parquet'),index=False)
    q=s.quotes.iloc[int(frame.start_index.min()):int(frame.source_index.max())+1]
    q[['available_ns','bid_nano','ask_nano','source_ordinal']].to_parquet(folder/(stem+'_quotes.parquet'),index=False)
    pack=extract(frame,s.quotes,search.books[s.session_id],samples=search.config['shape_samples'])
    np.savez_compressed(folder/(stem+'_features.npz'),**pack)


def scalar_check(prepared,search):
    f=prepared['virtual'];checks=0
    if not len(f):return 0
    # Fixed evenly spaced candidates, not selected by their similarity.
    for ix in np.unique(np.linspace(0,len(f)-1,6,dtype=int)):
        r=f.iloc[ix];s=search.sessions[r.session_id]
        qt,price,_,_=quote_arrays(s.quotes)
        i=int(np.searchsorted(qt,int(r.available_ns),side='right'))-1
        assert i==r.source_index
        p=price[int(r.start_index):i+1];delta=np.diff(p);sign=np.sign(delta[delta!=0])
        assert r.low_nano==p.min() and r.high_nano==p.max()
        assert r.path_nano==abs(delta).sum() and r.up_nano==np.maximum(0,delta).sum()
        assert r.down_nano==np.maximum(0,-delta).sum() and r.turns==np.count_nonzero(np.diff(sign))
        known=s.motion[(s.motion.story_id==r.story_id)&(s.motion.available_ns<=r.available_ns)]
        assert known.phase.iloc[-1] not in ('CONFIRMED','TIME_LIMIT','CENSORED')
        assert r.direction==known.direction.iloc[-1]
        checks+=1
    return checks


def main(out):
    folder=ROOT/out;folder.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'research_results/story_similarity_v2/verified_v3'
    cfg=read(prior/'config.json');methods=[]
    for old,new in [('rhythm_gate','age_rhythm'),('book_gate','age_book')]:
        methods.append(dict(next(m for m in cfg['methods'] if m['name']==old),name=new))
    cfg['alignment_methods']=methods;cfg['alignment_parent']=str(prior.relative_to(ROOT))
    cfg['alignment_design']='Replay each past active prefix at exactly query age; use only updates at or before that historical clock. Keep all previously specified weights and tolerances.'
    cfg['alignment_output']=out
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[ROOT/'tests/test_story_similarity_v2.py',ROOT/'tests/test_story_prefix_alignment.py']
    source={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target=folder/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    dump(folder/'config.json',cfg);dump(folder/'manifest.json',dict(status='RUNNING',source_sha256=source))
    begun=time.perf_counter();sessions,inputs=load_corpus(cfg);search=SimilaritySearch(sessions,cfg)
    inputs[str(prior/'manifest.json')]=sha(prior/'manifest.json')
    pm=read(prior/'manifest.json')
    for name in ('atlas_index.json','summary.json'):
        assert sha(prior/name)==pm['artifacts'][name]
        inputs[str(prior/name)]=sha(prior/name)
    current=search.sessions[cfg['query_session']];clocks,anchors=query_clocks(cfg)
    display=anchors|{next(t for t in clocks if t not in anchors)}
    prior_cases={c['case_id']:c for c in read(prior/'atlas_index.json')}
    statuses=[];matches=[];scenes=[];checks=dict(scalar_prefixes=0,aligned_candidates=0,prefix_equal=False)
    for clock in clocks:
        for _,token in live_states(current,clock).query("family == 'motion'").iterrows():
            for scope in ('previous','current'):
                prepared=prepare_aligned(search,token,scope)
                f=prepared['virtual']
                assert ((f.available_ns-f.start_ns)==prepared['age_ns']).all()
                assert f.available_ns.lt(prepared['query_context_start_ns']).all()
                checks['aligned_candidates']+=len(f)
                if scope=='previous':checks['scalar_prefixes']+=scalar_check(prepared,search)
                by_method={};by_status={}
                for method in methods:
                    status,neighbors=rank_aligned(search,prepared,method)
                    statuses.append(status);matches.extend(neighbors)
                    by_method[method['name']]=neighbors;by_status[method['name']]=status
                if scope=='previous' and clock in display:
                    case_id=pd.Timestamp(clock,tz='UTC').strftime('%H%M%S')+'_'+str(token.scale)
                    scene=dict(case_id=case_id,query_ns=clock,scale=token.scale,methods={},statuses=by_status)
                    save_frame(folder,case_id+'_query',prepared['query_frame'],search)
                    old=prior_cases[case_id]
                    scene['methods']['snapshot']=old['methods']['rhythm_gate']
                    scene['statuses']['snapshot']=old['statuses']['rhythm_gate']
                    for suffix in ('tokens.parquet','quotes.parquet','features.npz'):
                        name=case_id+'_rhythm_gate_'+suffix
                        assert sha(prior/name)==pm['artifacts'][name]
                        (folder/(case_id+'_snapshot_'+suffix)).write_bytes((prior/name).read_bytes())
                    for name,neighbors in by_method.items():
                        scene['methods'][name]=neighbors[0] if neighbors else None
                        if neighbors:save_frame(folder,case_id+'_'+name,context_frame(search,prepared,neighbors[0]),search)
                    scenes.append(scene)
        print(dict(stage='age_query',time=pd.Timestamp(clock,tz='UTC').isoformat()),flush=True)
    noon=int(pd.Timestamp(cfg['query_session']+'T12:00:00Z').value)
    prefix=copy.copy(current);prefix.quotes=current.quotes[current.quotes.available_ns<=noon]
    prefix.motion=current.motion[current.motion.available_ns<=noon];prefix.fast=current.fast[current.fast.available_ns<noon]
    prefix.observed_through_ns=noon;short=SimilaritySearch([sessions[0],prefix],cfg)
    for _,row in live_states(current,noon).query("family == 'motion'").iterrows():
        for scope in ('previous','current'):
            a=prepare_aligned(search,row,scope);b=prepare_aligned(short,row,scope)
            assert a['virtual'].token_id.tolist()==b['virtual'].token_id.tolist()
            for key in a['parts']:assert np.allclose(a['parts'][key],b['parts'][key],equal_nan=True)
            for method in methods:assert rank_aligned(search,a,method)==rank_aligned(short,b,method)
    checks['prefix_equal']=True
    sf=pd.DataFrame(statuses);mf=pd.DataFrame(matches)
    sf.to_parquet(folder/'statuses.parquet',index=False);dump(folder/'matches.json',matches);dump(folder/'atlas_index.json',scenes)
    summary=dict(status='COMPLETE_DESCRIPTIVE_AGE_ALIGNMENT',queries=36,methods=summarize(sf,mf),checks=checks,
        elapsed_seconds=time.perf_counter()-begun,
        snapshot_baselines=read(prior/'summary.json')['methods'],
        limitation='Exposed dates and visual cases; observation-grid repair, not independent semantic or economic admission')
    dump(folder/'summary.json',summary)
    artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'}
    dump(folder/'manifest.json',dict(status=summary['status'],source_sha256=source,input_sha256=inputs,artifacts=artifacts))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=DEFAULT);main(p.parse_args().out)
