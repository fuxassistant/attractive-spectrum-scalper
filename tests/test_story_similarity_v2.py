import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.scalper_adapter_v1.tokens import NS
from experiments.scalper_adapter_v1.adapter import build_session,live_states
from experiments.scalper_adapter_v1.measures import FLOW_FIELDS,FlowView
from experiments.story_similarity_v2.features import PublishedBook,extract,mirrored,BOOK_NAMES
from experiments.story_similarity_v2.metric import compare,score_components
from experiments.story_similarity_v2.search import SimilaritySearch,sensitivity_methods

ROOT=Path(__file__).parents[1]
CFG=json.loads((ROOT/'experiments/story_similarity_v2/config.json').read_text())
OLD=json.loads((ROOT/'experiments/scalper_adapter_v1/config.json').read_text())


def pack():
    shape=np.array([np.linspace(0,1,16),np.linspace(0,-.6,16),np.sin(np.linspace(0,2,16))])
    return dict(shape=shape,amplitude=np.array([[1,1,1,0,0,1],[-.6,.6,0,.6,.6,0],[.8,1,1,0,.2,.8]]),
        texture=np.array([[1,.3,.1,.1,.2,.3,.4],[-1,.4,.2,.4,.3,.2,.1],[.8,.3,.2,.25,.25,.25,.25]]),
        duration=np.array([15.,14.,6.]),structure=np.array([[1,0,1],[-1,1,0],[1,0,0]]),
        book=np.arange(39).reshape(3,13)/40,book_known=np.ones((3,13),bool))


def batch(*packs):return {k:np.stack([p[k] for p in packs]) for k in packs[0]}


def method(name):return next(m for m in CFG['methods'] if m['name']==name)


def test_identity_is_zero_in_each_block():
    p=pack();d=compare(p,batch(p))
    for k in ('shape','amplitude','texture','structure','rhythm','mechanism'):
        assert d[k][0]==0
    assert d['total_duration_factor'][0]==1


def test_threefold_time_stretch_is_rejected_even_with_identical_shape():
    q=pack();candidate=copy.deepcopy(q);candidate['duration']*=3
    d=compare(q,batch(candidate))
    assert d['shape'][0]==0 and d['total_duration_factor'][0]==3
    _,eligible,accepted=score_components(d,method('rhythm_gate'),CFG)
    assert not eligible[0] and not accepted[0]
    _,eligible,_=score_components(d,method('rhythm_soft'),CFG)
    assert eligible[0]


def test_internal_rhythm_differs_despite_equal_total_duration():
    q=pack();candidate=copy.deepcopy(q);candidate['duration']=np.array([27.,2.,6.])
    d=compare(q,batch(candidate))
    assert d['total_duration_ratio'][0]==1 and d['slot_duration_factor'][0]==7
    assert not score_components(d,method('rhythm_gate'),CFG)[1][0]


def test_equal_aggregate_reordered_stories_are_not_identical():
    q=pack();other={k:v[[1,0,2]].copy() for k,v in q.items()}
    d=compare(q,batch(q,other))
    assert np.allclose(q['amplitude'].sum(0),other['amplitude'].sum(0))
    assert d['shape'][0]==0<d['shape'][1] and d['structure'][1]>0


def test_missing_mechanism_cannot_score_as_a_perfect_match():
    q=pack();other=copy.deepcopy(q);other['book_known'][:]=False
    d=compare(q,batch(other))
    assert np.isnan(d['mechanism'][0])
    assert not score_components(d,method('book_gate'),CFG)[2][0]
    assert score_components(d,method('rhythm_gate'),CFG)[2][0]


def test_same_shape_opposite_observed_flow_remains_distinguishable():
    q=pack();other=copy.deepcopy(q);other['book'][:,:2]*=-1
    d=compare(q,batch(q,other))
    assert d['shape'][1]==0 and d['mechanism'][1]>d['mechanism'][0]


def test_price_mirror_is_involutive_and_swaps_flow_and_book_sides():
    p=pack();m=mirrored(p);back=mirrored(m)
    for name in p: assert np.array_equal(p[name],back[name]),name
    assert np.array_equal(m['book'][:,0],-p['book'][:,1])
    assert np.array_equal(m['book'][:,4],p['book'][:,5])
    assert np.array_equal(m['book'][:,12],-p['book'][:,12])


def fast_frame():
    f=pd.DataFrame(dict(t_ns=np.arange(1,11,dtype='int64')*NS,
        available_ns=np.arange(1,11,dtype='int64')*NS+123,epoch=1,
        trade_known=True,ledger_known=True,ret1_ticks=np.arange(10)/10))
    for i,name in enumerate(FLOW_FIELDS):f[name]=np.arange(10)+i
    return f


def token(clock,seconds):return pd.DataFrame([dict(available_ns=clock,flow_seconds=seconds,amplitude_ticks=4)])


def test_book_window_agrees_with_scalar_existing_flow_on_common_release_support():
    fast=fast_frame();t=token(8*NS+124,5)
    reference=FlowView(fast).measure(t.iloc[0]);b=PublishedBook(fast).measure(t)
    assert b['book_release_ns'][0]<int(t.available_ns.iloc[0])
    assert np.isclose(b['book'][0,0],reference['flow_imbalance_low'])
    assert np.isclose(b['book'][0,1],reference['flow_imbalance_high'])
    assert np.isclose(b['book'][0,4],np.log1p(reference['bid_near_fill']/5)/np.log(11))


def test_same_nanosecond_release_is_conservatively_not_borrowed():
    fast=fast_frame();b=PublishedBook(fast).measure(token(8*NS+123,1))
    assert b['book_release_ns'][0]==7*NS+123


def test_future_buckets_cannot_leak_through_unknown_values():
    f=fast_frame();t=token(NS,1)
    a=PublishedBook(f).measure(t)
    f.loc[0,FLOW_FIELDS]=1e9
    b=PublishedBook(f).measure(t)
    assert not a['book_known'].any() and np.array_equal(a['book'],b['book'])
    assert not a['book'].any() and a['book_release_ns'][0]==-1


@pytest.mark.parametrize('break_type',['clock','epoch','book'])
def test_broken_book_windows_do_not_claim_full_mechanism(break_type):
    f=fast_frame()
    if break_type=='clock':f=f.drop(index=5).reset_index(drop=True)
    elif break_type=='epoch':f.loc[5:,'epoch']=2
    else:f.loc[5,'ledger_known']=False
    b=PublishedBook(f).measure(token(8*NS+124,5))
    assert not b['book_known'][0,4:12].any()


def native_session():
    base=1787659200000000123
    p=np.rint((100+2*np.sin(np.arange(301)/5))*NS).astype('int64')
    q=pd.DataFrame(dict(available_ns=base+np.arange(301,dtype='int64')*NS,
        bid_nano=p-50_000_000,ask_nano=p+50_000_000,valid=True,epoch=1,
        source_ordinal=np.arange(301,dtype='int64')))
    fast=fast_frame()
    fast['t_ns']+=base;fast['available_ns']+=base
    s=build_session(q,session_id='s',previous_session_id='p',symbol='TEST',
        start_ns=base,end_ns=base+400*NS,observed_through_ns=base+300*NS,
        scales=[OLD['scales'][0]],fast=fast)
    return s


def test_native_dense_shape_is_translation_invariant_and_uses_only_prefix():
    s=native_session();clock=int(s.quotes.available_ns.iloc[150])
    t=live_states(s,clock);t=t[t.family.eq('motion')]
    a=extract(t,s.quotes,samples=16)
    b=extract(t,s.quotes[s.quotes.available_ns<=clock],samples=16)
    for key in a:assert np.array_equal(a[key],b[key]),key
    shifted=s.quotes.copy(); shifted[['bid_nano','ask_nano']]+=1000*NS
    t=t.copy()
    for name in ('start_mid_nano','mid_nano','low_nano','high_nano'):t[name]+=1000*NS
    c=extract(t,shifted,samples=16)
    for key in a:assert np.array_equal(a[key],c[key]),key


def test_real_search_returns_no_temporal_overlap_and_handles_missing_prior():
    s=native_session();clock=int(s.quotes.available_ns.iloc[250])
    row=live_states(s,clock);row=row[row.family.eq('motion')].iloc[0]
    search=SimilaritySearch([s],CFG)
    previous=search.prepare(row,'previous')
    summary,matches=search.rank(previous,method('rhythm_gate'))
    assert summary['status']=='NO_CAUSAL_HISTORY' and not matches
    current=search.prepare(row,'current')
    summary,matches=search.rank(current,method('rhythm_gate'))
    for m in matches:assert m['available_ns']<m['query_context_start_ns']<=clock


def test_query_and_rank_prefix_invariance_including_book_masks():
    s=native_session();clock=int(s.quotes.available_ns.iloc[150])
    row=live_states(s,clock);row=row[row.family.eq('motion')].iloc[0]
    a=SimilaritySearch([s],CFG)
    prefix=copy.copy(s)
    prefix.quotes=s.quotes[s.quotes.available_ns<=clock]
    prefix.motion=s.motion[s.motion.available_ns<=clock]
    prefix.fast=s.fast[s.fast.available_ns<clock]
    b=SimilaritySearch([prefix],CFG)
    pa=a.prepare(row,'current');pb=b.prepare(row,'current')
    for key in pa['parts']:assert np.allclose(pa['parts'][key],pb['parts'][key],equal_nan=True),key
    for m in CFG['methods']:assert a.rank(pa,m)==b.rank(pb,m)


def test_sensitivity_weights_remain_valid_and_explicit():
    methods=sensitivity_methods(CFG)
    assert len(methods)==20
    for m in methods:
        assert np.isclose(sum(m['weights'].values()),1)
        assert all(v>=0 for v in m['weights'].values())


def test_unacceptable_geometry_is_allowed_to_return_no_analogue():
    p=pack();q=copy.deepcopy(p);q['shape']+=10
    d=compare(p,batch(q))
    _,eligible,accepted=score_components(d,method('rhythm_gate'),CFG)
    assert eligible[0] and not accepted[0]


def test_whole_story_rejects_accumulated_opposite_drift_hidden_by_small_slots():
    q=pack();c=copy.deepcopy(q)
    q['amplitude'][:,0]=[-.3,-.3,-.2]
    c['amplitude'][:,0]=[.3,.3,.2]
    d=compare(q,batch(c))
    assert d['net_direction_conflict'][0]==1 and d['slot_direction_conflict'][0]==0
    assert not score_components(d,method('rhythm_gate'),CFG)[1][0]


def test_joined_path_preserves_level_across_story_boundaries():
    from experiments.story_similarity_v2.metric import joined_curve
    q=pack();q['amplitude'][:,0]=[1,1,1]
    q['amplitude'][:,1]=1;q['amplitude'][:,2]=1;q['amplitude'][:,3]=0
    q['shape'][:]=np.linspace(0,1,16)
    joined=joined_curve(q)
    assert np.isclose(joined[15],1/3) and np.isclose(joined[31],2/3) and joined[47]==1
