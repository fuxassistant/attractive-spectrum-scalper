import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.engine import (NS, causal_paths, decision_cohort,
                                                    DelayedCalibrator, define_hypotheses)


def quotes():
    t=np.arange(700,dtype=np.int64)*NS
    return pd.DataFrame(dict(available_ns=t,bid_nano=4000*NS+np.arange(700)*100_000,
                             ask_nano=4000*NS+np.arange(700)*100_000+100_000_000,
                             epoch=np.zeros(700,dtype=np.int64),valid=True))


def candidate():
    return pd.DataFrame([dict(candidate_id=1,t_ns=650*NS,side=1,risk_per_oz=1.,
                              zone_low_nano=4001*NS,zone_high_nano=4002*NS)])


def test_native_invalid_between_samples_is_unknown():
    q=quotes();extra=q.iloc[[100]].copy();extra['available_ns']+=NS//2;extra['valid']=False
    q=pd.concat([q,extra]).sort_values('available_ns').reset_index(drop=True)
    assert not causal_paths(q,candidate()).past_known.iloc[0]


def test_prefix_append_invariance():
    q=quotes();c=candidate();a=causal_paths(q[q.available_ns<=650*NS],c)
    q.loc[q.available_ns>650*NS,'bid_nano']=1
    pd.testing.assert_frame_equal(a,causal_paths(q,c))


def test_epoch_reset_inside_history_is_unknown():
    q=quotes();q.loc[q.index>=100,'epoch']=1
    assert not causal_paths(q,candidate()).past_known.iloc[0]


def test_stale_endpoint_is_unknown():
    q=quotes().iloc[:610]
    assert not causal_paths(q,candidate()).past_known.iloc[0]


def test_gap_between_samples_is_unknown():
    q=quotes().drop(index=range(100,140))
    assert not causal_paths(q,candidate()).past_known.iloc[0]


def test_side_orientation_and_price_translation():
    q=quotes();c=candidate();a=causal_paths(q,c)
    c['side']=-1;b=causal_paths(q,c)
    assert np.isclose(a.past_net30_r.iloc[0],-b.past_net30_r.iloc[0])
    q[['bid_nano','ask_nano']]+=20*NS;c[['zone_low_nano','zone_high_nano']]+=20*NS
    z=causal_paths(q,c)
    np.testing.assert_allclose(b.select_dtypes('number'),z.select_dtypes('number'),atol=1e-10)


def test_cohort_is_target_independent_and_nonoverlapping():
    f=pd.DataFrame(dict(t_ns=[0,0,100*NS,300*NS,301*NS],candidate_id=[1,2,3,4,5],
                        distance_r=[2,1,1,1,1],zone_low_nano=[5,5,5,5,5],state=['a']*5))
    a=decision_cohort(f);assert a.candidate_id.tolist()==[2,4]
    f['state']='other';assert decision_cohort(f).candidate_id.tolist()==[2,4]


def test_feedback_has_common_maturity_and_strict_order():
    c=DelayedCalibrator();p=np.array([.3,.4,.3])
    c.enqueue(0,300*NS,p,0,['a'])
    for t in [1,299*NS,300*NS]:
        q,a=c.predict(t,p,['a'],True);np.testing.assert_allclose(q,p);assert a['feedback_count']==0
    q,a=c.predict(300*NS+1,p,['a'],True)
    assert q[0]>p[0] and a['feedback_count']==1 and a['max_feedback_release_ns']==300*NS


def test_unknown_outcome_never_becomes_failure():
    c=DelayedCalibrator();p=np.array([.3,.4,.3]);c.enqueue(0,300*NS,p,-1,['a'])
    q,a=c.predict(301*NS,p,['a'],True)
    np.testing.assert_allclose(q,p);assert a['feedback_count']==0


def test_repeated_read_does_not_learn_and_inactive_family_has_no_evidence():
    c=DelayedCalibrator();p=np.array([.3,.4,.3]);c.enqueue(0,300*NS,p,0,['a'])
    q,a=c.predict(301*NS,p,['b'],True);q2,a2=c.predict(301*NS,p,['b'],True)
    np.testing.assert_array_equal(q,q2);assert a==a2 and a['local_effective_support']==0


def test_adaptation_forgets_after_long_gap():
    c=DelayedCalibrator(half_life_seconds=10);p=np.array([.3,.4,.3]);c.enqueue(0,NS,p,0,['a'])
    q,_=c.predict(2*NS,p,['a'],True);late,_=c.predict(1000*NS,p,['a'],True)
    assert abs(late[0]-p[0])<abs(q[0]-p[0])
    np.testing.assert_allclose(late,p)


def test_probability_revision_responds_to_future_completed_outcomes_only():
    p=np.array([.3,.4,.3]);a=DelayedCalibrator();b=DelayedCalibrator()
    a.enqueue(0,300*NS,p,0,['x']);b.enqueue(0,300*NS,p,1,['x'])
    np.testing.assert_array_equal(a.predict(100*NS,p,['x'],True)[0],b.predict(100*NS,p,['x'],True)[0])
    assert not np.array_equal(a.predict(301*NS,p,['x'],True)[0],b.predict(301*NS,p,['x'],True)[0])


def hypothesis_rows():
    return pd.DataFrame(dict(block=['train','train','evaluation'],completed_visits=[8]*3,
        median_dwell_seconds=[10.,20.,15.],x_mem_log_recent_gate_aborts=np.log1p([8.,16.,12.]),
        x_mem_recent_pass_fraction=[.2,.3,.2],x_mem_last_pass=[0,0,0],past_known=[True]*3,
        past_net30_r=[.3]*3,past_migration_r=[-1.]*3,past_zone_share_old=[.2]*3,
        past_zone_share_new=[0.]*3,past_range120_r=[.5]*3,past_net120_r=[.2]*3))


def test_template_thresholds_do_not_learn_from_later_x_or_y():
    config=json.loads((Path(__file__).parents[1]/'experiments/hypothesis_atlas_v1/config.json').read_text())
    f=hypothesis_rows();a,limits=define_hypotheses(f,config)
    f.loc[2,'median_dwell_seconds']=1e9;f['state']='ADVERSE_FIRST'
    _,limits2=define_hypotheses(f,config)
    assert limits==limits2 and limits['median_dwell_seconds']==15.
    assert a.loc[2,'h_memory_return'] and a.loc[2,'h_place_migration']


def test_missing_recent_path_blocks_path_dependent_hypotheses():
    config=json.loads((Path(__file__).parents[1]/'experiments/hypothesis_atlas_v1/config.json').read_text())
    f=hypothesis_rows();f['past_known']=False
    a,_=define_hypotheses(f,config)
    assert not a[['h_failed_escape','h_place_migration','h_tight_range','h_permeable_departure']].any().any()
    assert a.h_memory_return.any()


def test_before_figure_is_invariant_to_future_prices_and_labels(tmp_path):
    from experiments.hypothesis_atlas_v1.report import scene
    q=quotes();r=candidate().iloc[0].to_dict()
    r.update(family='memory_return',origin_mid_x2_nano=8000*NS+230_000_000,
             adverse_x2_nano=7998*NS+230_000_000,target_nano=4001*NS,
             probabilities={'context':[.3,.4,.3],'hot_families':[.32,.38,.3]},
             first_event_ns=680*NS,state='RETURN_FIRST')
    a=tmp_path/'before_a.png';b=tmp_path/'before_b.png'
    scene(q,r,a,reveal=False)
    q.loc[q.available_ns>650*NS,['bid_nano','ask_nano']]+=500*NS
    r.update(first_event_ns=690*NS,state='ADVERSE_FIRST')
    scene(q,r,b,reveal=False)
    assert a.read_bytes()==b.read_bytes()
