import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.zone_conditional_v1.features import make_features, add_past_placebo
from experiments.zone_conditional_v1.model import (
    CLASSES, FrozenClassifier, common_policy_clock, day_clock_weights,
    deterministic_accept,
)

NS = 1_000_000_000


def inputs(count=3):
    c = pd.DataFrame([dict(candidate_id=1, origin_index=1, t_ns=1000*NS,
        date='2026-08-03', inside=False, side=1, zone_low_nano=100*NS,
        distance_r=1.5, distance_per_oz=3., risk_per_oz=2., spread_per_oz=.1,
        completed_visits=count, cumulative_dwell_seconds=100., median_dwell_seconds=20.,
        age_since_departure_seconds=100., permeability=.5, zone_available_ns=900*NS,
        history_start_ns=100*NS)])
    bars = pd.DataFrame([dict(t_ns=1000*NS, ready=True, atr14=2., atr_ratio=1.,
        body_atr=.2, trend_atr=.1, bb_z=-.2, efficiency20=.4, range_atr=1., volume=1000.)])
    tokens=[];events=[]
    for k in range(count):
        arrival=(100+k*80)*NS;end=(150+k*80)*NS
        tokens.append(dict(candidate_id=1, position=k, episode_id=10+k, arrival_ns=arrival,
            available_ns=end, entry_side=1, exit_side=-1 if k%2==0 else 1,
            outcome='REJECT_DOWN' if k%2==0 else 'PASS_UP', core_dwell_seconds=10.+k,
            duration_seconds=50., previous_absence_seconds=np.nan if k==0 else 30.,
            approach_efficiency=np.nan if k==0 else .4, previous_peak_distance_ticks=np.nan if k==0 else 20.,
            date='2026-08-03'))
        events.extend([dict(episode_id=10+k, available_ns=arrival, event_type='ARRIVAL',duration_ns=0,excursion_nano=0,date='2026-08-03'),
            dict(episode_id=10+k,available_ns=arrival+13*NS,event_type='EDGE_RETURN',duration_ns=3*NS,excursion_nano=NS,date='2026-08-03'),
            dict(episode_id=10+k,available_ns=end,event_type='DEPARTURE_CONFIRMED',duration_ns=50*NS,excursion_nano=NS,date='2026-08-03')])
    spectral=pd.DataFrame([dict(candidate_id=1,available_ns=1000*NS,occupancy_mean=.2,
        occupancy_variance=.05,main_lobe_power_fraction=.4,spectral_entropy=.6,
        autocorr_at_peak=.2,return_interval_cv=.7,period_candidate_supported=True,
        spectrum_known=True,arrival_count=6,dominant_period_seconds=300.,return_interval_mean_seconds=300.,date='2026-08-03')])
    return c,bars,pd.DataFrame(events),pd.DataFrame(tokens),spectral


def test_future_history_token_is_rejected():
    args=list(inputs());args[3].loc[0,'available_ns']=1000*NS+1
    with pytest.raises(ValueError):make_features(*args)


def test_referenced_episode_cannot_add_probe_after_its_recorded_completion():
    args=list(inputs())
    future=dict(args[2].iloc[1]);future['available_ns']=1001*NS
    args[2]=pd.concat([args[2],pd.DataFrame([future])],ignore_index=True)
    with pytest.raises(ValueError):make_features(*args)


def test_daily_episode_identity_cannot_be_merged_across_dates():
    args=list(inputs())
    other=args[0].copy();other['date']='2026-08-04';other['candidate_id']=2;other['origin_index']=2
    args[0]=pd.concat([args[0],other],ignore_index=True)
    with pytest.raises(ValueError):make_features(*args)


def test_all_eight_completed_tokens_are_used_in_position_order():
    args=list(inputs(8));args[3]=args[3].sample(frac=1,random_state=19)
    f=make_features(*args).iloc[0]
    assert f.x_mem_recent_pass_fraction==.5
    assert f.x_mem_log_recent_edge_returns==pytest.approx(np.log1p(8))
    assert f.x_mem_log_last_duration==pytest.approx(np.log1p(50))
    assert f.x_mem_last_exit_toward==1
    assert f.memory_available_ns==(150+7*80)*NS
    args=list(inputs(9))
    with pytest.raises(ValueError):make_features(*args)


def test_unrelated_later_episode_does_not_change_past_features():
    args=list(inputs());expected=make_features(*args)
    future=args[2].copy();future['episode_id']=99;future['available_ns']+=2000*NS
    args[2]=pd.concat([args[2],future],ignore_index=True)
    assert_frame_equal(make_features(*args),expected,check_exact=True)


def test_placebo_is_strictly_prior_different_zone_and_prefix_stable():
    f=pd.DataFrame(dict(candidate_id=[1,2,3,4,5],t_ns=[100,200,200,300,400],
        zone_low_nano=[100,100,101,102,100],side=1,match_atr=2.,match_distance_bin=1,
        match_spread_bin=0,match_phase=0,x_mem_number=[10.,20.,30.,40.,50.]))
    train=f.candidate_id.le(3)
    a,receipt=add_past_placebo(f,train)
    b,_=add_past_placebo(f.iloc[:4],train.iloc[:4])
    assert_frame_equal(a.iloc[:4].reset_index(drop=True),b,check_exact=True)
    indexed=a.set_index('candidate_id')
    assert not indexed.loc[2,'placebo_known']  # Earlier donor is the same zone.
    assert indexed.loc[3,'placebo_donor_id']==1  # Row2 shares this clock, so is unavailable.
    assert indexed.loc[4,'placebo_donor_id']==3
    assert indexed.loc[4,'x_placebo_number']==30.
    for row in a.loc[a.placebo_known].itertuples():
        donor=indexed.loc[row.placebo_donor_id]
        assert donor.t_ns<row.t_ns and donor.zone_low_nano!=row.zone_low_nano
    assert receipt['atr_tertiles']==[2.,2.]


def test_weights_equalize_clocks_then_days_without_changing_regularization_scale():
    f=pd.DataFrame(dict(candidate_id=np.arange(6),date=['a','a','a','b','b','b'],t_ns=[1,1,2,3,4,5]))
    w=day_clock_weights(f);g=f.assign(weight=w)
    assert w.mean()==pytest.approx(1.)
    assert g.groupby('date').weight.sum().tolist()==pytest.approx([3.,3.])
    assert g[g.date=='a'].groupby('t_ns').weight.sum().tolist()==pytest.approx([1.5,1.5])


def test_prediction_cannot_refit_train_preprocessing_on_future_extremes_or_missingness():
    n=90
    f=pd.DataFrame(dict(candidate_id=np.arange(n),date=np.repeat(['2026-08-03','2026-08-04','2026-08-05'],30),
        t_ns=np.arange(n)*NS,state=np.resize(CLASSES,n),label_available_ns=(np.arange(n)+1)*NS,
        x_base_value=np.tile([-2.,0.,2.],30),x_base_missing=np.nan))
    cfg=dict(C=.1,solver='lbfgs',max_iter=2000,tol=1e-8,seed=271828)
    m=FrozenClassifier(['x_base_value','x_base_missing'],cfg).fit(f)
    before=copy.deepcopy(m.receipt());p=m.predict(f)
    future=f.iloc[:3].copy();future['x_base_value']=[1e30,-1e30,np.nan];future['x_base_missing']=[1.,np.nan,1e20]
    fp=m.predict(future)
    assert np.isfinite(fp).all() and np.allclose(fp.sum(axis=1),1)
    assert m.receipt()==before
    assert np.array_equal(m.predict(f),p)
    assert m.estimator.classes_.tolist()==[0,1,2]
    assert m.median.tolist()==[0.,0.]


def test_common_policy_clock_ignores_scores_labels_and_future_rows():
    f=pd.DataFrame(dict(candidate_id=[1,2,3],t_ns=[100,100,200],distance_per_oz=[3.,1.5,2.],
        zone_low_nano=[100,101,100],state=['RETURN_FIRST','CENSORED','ADVERSE_FIRST'],score=[.99,.01,.5]))
    chosen=common_policy_clock(f)
    assert chosen.candidate_id.tolist()==[2,3]  # Censor mask cannot erase the closest offer.
    mutated=f.copy();mutated['score']=1-mutated.score;mutated['state']='TIMEOUT'
    assert common_policy_clock(mutated).candidate_id.tolist()==[2,3]
    assert common_policy_clock(f.iloc[:2]).candidate_id.tolist()==[2]
    assert np.array_equal(deterministic_accept([1,2],.4,31),deterministic_accept([1,2,999],.4,31)[:2])
