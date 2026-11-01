import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from experiments.zone_conditional_v1.features import add_past_placebo
from experiments.zone_conditional_v1.model import (FrozenClassifier, day_clock_weights,
    deterministic_accept, common_policy_clock, daily_metrics)


CFG=dict(C=.1,solver='lbfgs',max_iter=2000,tol=1e-8,seed=271828)


def train_frame():
    rng=np.random.default_rng(7)
    x=rng.normal(size=180)
    return pd.DataFrame(dict(candidate_id=np.arange(180),date=np.repeat(['a','b','c'],60),
        t_ns=np.arange(180,dtype=np.int64),label_available_ns=np.arange(180,dtype=np.int64)+1,
        x=x,state=np.array(['ADVERSE_FIRST','TIMEOUT','RETURN_FIRST'])[np.digitize(x,[-.4,.4])]))


def test_preprocessing_is_train_only_and_prediction_cannot_mutate_it():
    f=train_frame();f.loc[0,'x']=np.nan
    m=FrozenClassifier(['x'],CFG).fit(f)
    original=m.receipt()
    p=m.predict(f)
    future=f.iloc[:2].copy();future['x']=[1e12,np.nan]
    m.predict(future)
    assert m.receipt()==original
    assert np.allclose(m.predict(f),p)
    assert (p[f.x.fillna(0)>1,0]>.4).all()
    assert np.isfinite(m.transform(future)).all()


def test_clock_day_weights_do_not_overcount_duplicate_opportunities():
    f=pd.DataFrame(dict(candidate_id=[1,2,3,4,5],date=['a','a','a','b','b'],t_ns=[1,1,2,3,4]))
    w=day_clock_weights(f)
    assert np.isclose(w.sum(),len(f))
    assert np.isclose(w[:3].sum(),w[3:].sum())
    assert np.isclose(w[0]+w[1],w[2])


def test_past_placebo_preserves_prefix_and_never_uses_same_zone_or_same_clock():
    f=pd.DataFrame(dict(candidate_id=np.arange(8),t_ns=np.array([1,1,2,3,4,5,6,7],np.int64),
        zone_low_nano=[10,20,10,30,10,20,40,50],match_atr=1.,side=1,
        match_distance_bin=0,match_spread_bin=0,match_phase=0,
        x_mem_a=np.arange(8,dtype=float),x_mem_b=np.arange(8,dtype=float)*2))
    old,_=add_past_placebo(f.iloc[:5],f.iloc[:5].candidate_id.lt(3))
    full,_=add_past_placebo(f,f.candidate_id.lt(3))
    assert_frame_equal(old,full.iloc[:5].reset_index(drop=True))
    assert not full.placebo_known.iloc[:2].any()
    for r in full.loc[full.placebo_known].itertuples():
        donor=f.loc[f.candidate_id.eq(r.placebo_donor_id)].iloc[0]
        assert donor.t_ns<r.t_ns and donor.zone_low_nano!=r.zone_low_nano
        assert donor.x_mem_a==r.x_placebo_a and donor.x_mem_b==r.x_placebo_b


def test_policy_clock_is_chosen_without_scores_and_hash_control_is_prefix_stable():
    f=pd.DataFrame(dict(candidate_id=[1,2,3],t_ns=[5,5,6],distance_per_oz=[2.,1.,1.],zone_low_nano=[10,20,30],score=[.99,.01,.5]))
    assert common_policy_clock(f).candidate_id.tolist()==[2,3]
    ids=np.arange(1000,dtype=np.int64)
    a=deterministic_accept(ids,.2,17)
    assert np.array_equal(a[:100],deterministic_accept(ids[:100],.2,17))
    assert .15<a.mean()<.25


def test_known_cohort_metrics_count_unique_clocks_and_hand_computed_loss():
    f=pd.DataFrame(dict(candidate_id=[1,2,3],date=['a']*3,t_ns=[1,1,2],state=['RETURN_FIRST','ADVERSE_FIRST','TIMEOUT']))
    p=np.array([[.5,.25,.25],[.2,.6,.2],[.1,.1,.8]])
    d,_=daily_metrics(f,p)
    expected=(-np.log(.5)/2-np.log(.6)/2-np.log(.8))/2
    assert np.isclose(d.logloss.iloc[0],expected)
    assert d.clocks.iloc[0]==2
