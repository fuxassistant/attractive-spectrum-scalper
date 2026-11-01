import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.scalper_adapter_v1.adapter import build_session,live_states
from experiments.scalper_adapter_v1.tokens import NS
from experiments.scalper_adapter_v1.measures import FLOW_FIELDS
from experiments.story_similarity_v2.search import SimilaritySearch
from experiments.story_similarity_v2.age_aligned import prepare_aligned,rank_aligned,context_frame

ROOT=Path(__file__).parents[1]
CFG=json.loads((ROOT/'experiments/story_similarity_v2/config.json').read_text())
SCALE=json.loads((ROOT/'experiments/scalper_adapter_v1/config.json').read_text())['scales'][0]


def session(heartbeat=1,moving=False):
    base=1787659200000000123
    p=np.rint((100+(2*np.sin(np.arange(301)/5) if moving else np.zeros(301)))*NS).astype(np.int64)
    q=pd.DataFrame(dict(available_ns=base+np.arange(301,dtype=np.int64)*NS,bid_nano=p-50_000_000,
        ask_nano=p+50_000_000,valid=True,epoch=1,source_ordinal=np.arange(301,dtype=np.int64)))
    fast=pd.DataFrame(dict(t_ns=base+np.arange(1,301,dtype=np.int64)*NS,
        available_ns=base+np.arange(1,301,dtype=np.int64)*NS+11,epoch=1,trade_known=True,
        ledger_known=True,ret1_ticks=0.))
    for name in FLOW_FIELDS:fast[name]=0.
    s=build_session(q,session_id='s',previous_session_id='p',symbol='TEST',start_ns=base,
        end_ns=base+400*NS,observed_through_ns=base+300*NS,
        scales=[dict(SCALE,update_seconds=heartbeat)],fast=fast)
    return s,base


def query(s,clock):return live_states(s,clock).query("family == 'motion'").iloc[0]


def test_early_forming_prefix_exists_before_first_periodic_snapshot():
    s,base=session();row=query(s,base+150*NS+200_000_000);search=SimilaritySearch([s],CFG)
    p=prepare_aligned(search,row,'current')
    assert len(p['virtual'])>0 and (p['virtual'].elapsed_seconds==.2).all()
    assert p['virtual'].available_ns.max()<p['query_context_start_ns']
    status,matches=rank_aligned(search,p,CFG['methods'][3])
    assert status['returned']>0
    assert all(m['available_ns']<m['query_context_start_ns'] for m in matches)
    frames=[context_frame(search,p,m) for m in matches]
    assert all(f.available_ns.iloc[-1]-f.start_ns.iloc[-1]==200_000_000 for f in frames)


def test_alignment_matches_scalar_live_state_on_native_tape():
    s,base=session(moving=True);search=SimilaritySearch([s],CFG)
    p=prepare_aligned(search,query(s,base+250*NS+123),'current')
    for r in p['virtual'].head(12).itertuples():
        scalar=query(s,int(r.available_ns))
        assert scalar.story_id==r.story_id
        for field in ('mid_nano','low_nano','high_nano','path_nano','turns','up_nano','down_nano','direction','source_index'):
            assert getattr(scalar,field)==getattr(r,field),field


def test_heartbeat_schedule_does_not_change_aligned_candidates_or_components():
    a,base=session(1,moving=True);b,_=session(3,moving=True)
    row=query(a,base+250*NS+123)
    pa=prepare_aligned(SimilaritySearch([a],CFG),row,'current')
    pb=prepare_aligned(SimilaritySearch([b],CFG),query(b,int(row.available_ns)),'current')
    assert pa['virtual'].token_id.tolist()==pb['virtual'].token_id.tolist()
    for key in pa['parts']:assert np.allclose(pa['parts'][key],pb['parts'][key],equal_nan=True),key


def test_future_after_observation_boundary_cannot_change_current_search():
    s,base=session(moving=True);clock=base+150*NS+123
    short=copy.copy(s);short.quotes=s.quotes[s.quotes.available_ns<=clock]
    short.motion=s.motion[s.motion.available_ns<=clock];short.fast=s.fast[s.fast.available_ns<clock]
    short.observed_through_ns=clock
    a=SimilaritySearch([s],CFG);b=SimilaritySearch([short],CFG)
    pa=prepare_aligned(a,query(s,clock),'current');pb=prepare_aligned(b,query(short,clock),'current')
    assert pa['virtual'].token_id.tolist()==pb['virtual'].token_id.tolist()
    for key in pa['parts']:assert np.allclose(pa['parts'][key],pb['parts'][key],equal_nan=True),key
    for method in CFG['methods'][3:]:assert rank_aligned(a,pa,method)==rank_aligned(b,pb,method)


def test_no_story_is_kept_if_its_terminal_is_already_known_at_requested_age():
    s,base=session(moving=True);search=SimilaritySearch([s],CFG)
    p=prepare_aligned(search,query(s,base+250*NS),'current')
    for r in p['virtual'].itertuples():
        known=s.motion[(s.motion.story_id==r.story_id)&(s.motion.available_ns<=r.available_ns)]
        assert known.phase.iloc[-1] not in ('CONFIRMED','TIME_LIMIT','CENSORED')
    empty=prepare_aligned(search,query(s,base+250*NS),'previous')
    status,matches=rank_aligned(search,empty,CFG['methods'][3])
    assert status['status']=='NO_CAUSAL_HISTORY' and not matches
