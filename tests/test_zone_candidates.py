import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from experiments.zone_episode_lab.episodes import build_episodes,occupancy_grid,NS
from experiments.zone_episode_lab.candidates import make_candidates,select_policy


def fixture():
    cfg=json.loads(Path('experiments/zone_episode_lab/config.json').read_text())
    cfg['zone_quality'].update(minimum_completed_visits=2,minimum_history_span_seconds=60,minimum_dwell_seconds=0)
    cfg['candidate']['cooldown_seconds']=120
    start=int(pd.Timestamp('2026-08-24T07:00:00Z').value)
    seconds=np.arange(2401);p=np.rint((100.5+3*np.sin(2*np.pi*seconds/120))*20)/20
    q=pd.DataFrame(dict(available_ns=start+seconds*NS,bid_nano=np.rint((p-.05)*NS).astype(np.int64),
        ask_nano=np.rint((p+.05)*NS).astype(np.int64),valid=True,epoch=1,source_ordinal=seconds.astype(np.int64)))
    t=start+np.arange(60,2401,60)*NS
    f=pd.DataFrame(dict(t_ns=t,ready=True,atr14=1.2,regime='range_normal',time_regime='07_12_utc',minute_utc=7*60+(t-start)//(60*NS),epoch=1))
    return cfg,start,q,f


def build(cfg,start,q,f,end):
    ev,ep=build_episodes(q,end_ns=int(end),**cfg['geometry'])
    grid=occupancy_grid(q,start_ns=start,end_ns=int(end))
    return make_candidates(q.loc[q.available_ns<=end],f.loc[f.t_ns<=end],ev,ep,grid,cfg,1,'2026-08-24')


def test_last_eight_tokens_mature_before_decision_and_clock_types_exact():
    cfg,start,q,f=fixture();c,parents,tokens,s=build(cfg,start,q,f,int(q.available_ns.iloc[-1]))
    assert len(c)>0 and len(parents)>0 and len(tokens)>0
    assert c.episode_available_ns.dtype.kind in 'iu'
    assert (c.zone_available_ns<=c.t_ns).all()
    times=c.set_index('candidate_id').t_ns
    assert (tokens.available_ns<=tokens.candidate_id.map(times)).all()
    assert tokens.groupby('candidate_id').size().max()<=8
    assert not c.candidate_id.duplicated().any()


def test_candidate_ids_and_features_do_not_depend_on_later_parent_count():
    cfg,start,q,f=fixture();end=start+1800*NS
    full=build(cfg,start,q,f,int(q.available_ns.iloc[-1]))[0]
    prefix=build(cfg,start,q,f,end)[0]
    a=full[full.t_ns<=end].sort_values('candidate_id').reset_index(drop=True)
    b=prefix.sort_values('candidate_id').reset_index(drop=True)
    assert_frame_equal(a,b)


def test_parent_cohort_keeps_wait_and_inside_outside_link_is_same_episode():
    cfg,start,q,f=fixture();c,parents,_,_=build(cfg,start,q,f,int(q.available_ns.iloc[-1]))
    assert (parents.outside_candidate_id<0).any()
    indexed=c.set_index('candidate_id')
    for p in parents.itertuples():
        for cid in (p.inside_candidate_id,p.outside_candidate_id):
            if cid>=0:assert indexed.loc[cid,'parent_episode_id']==p.episode_id
    chosen=select_policy(c,'return_sticky')
    assert (~chosen.inside).all() and chosen.t_ns.is_unique

def test_outside_wager_has_no_intervening_contact_since_departure():
    cfg,start,q,f=fixture();end=int(q.available_ns.iloc[-1])
    ev,ep=build_episodes(q,end_ns=end,**cfg['geometry'])
    c=build(cfg,start,q,f,end)[0]
    for r in c.loc[~c.inside].itertuples():
        new=ev[(ev.zone_low_nano==r.zone_low_nano)&ev.event_type.isin(['ARRIVAL','PRESENT_ON_RESTART'])
            &(ev.available_ns>r.episode_available_ns)&(ev.available_ns<=r.t_ns)]
        assert new.empty

def test_inside_only_policy_keeps_valid_sort_schema():
    cfg,start,q,f=fixture();c=build(cfg,start,q,f,int(q.available_ns.iloc[-1]))[0]
    only=c.loc[c.inside]
    assert len(select_policy(only,'inside_now'))>0
