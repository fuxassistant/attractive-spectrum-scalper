import numpy as np
import pandas as pd
import pytest

from experiments.paired_action_target_v1.targets import paired_targets,mature_pair_mask,decision_from_values
from experiments.paired_action_target_v1.future_description import describe_future


def sample(utilities, statuses=None):
    n=len(utilities);start=1_787_500_000_000_000_007
    f=pd.DataFrame(dict(origin_index=np.arange(n),date=['2026-08-25']*n,
                        t_ns=np.arange(n,dtype=np.int64)*1000+start,family='test',level_kind='none',side=1,
                        risk_hint=1.,origin_admitted=True))
    rows=[]
    for i,(long,short) in enumerate(utilities):
        for j,(side,u) in enumerate([(1,long),(-1,short)]):
            status='COMPLETE' if statuses is None else statuses[i][j]
            t=int(f.t_ns.iloc[i]);full=status=='COMPLETE'
            rows.append(dict(origin_index=i,side=side,signal_ns=t,risk_per_oz=1.,status=status,reason='TIME',
                             net_per_oz=u if full else np.nan,entry_ns=t+101 if full else 0,
                             trigger_ns=t+300+j*100 if full else 0,exit_ns=t+401+j*100 if full else 0,
                             mfe_per_oz=u if full else np.nan,mae_per_oz=min(u,0) if full else np.nan))
    return f,pd.DataFrame(rows)


def test_pair_states_do_not_force_equal_or_negative_values_to_a_side():
    f,o=sample([(-1,-1),(.4,.4),(1,-1),(-1,1)])
    p=paired_targets(f,o)
    assert p.future_joint_profit_state.tolist()==['NEITHER','BOTH','LONG_ONLY','SHORT_ONLY']
    assert p.future_near_equal_01r.tolist()==[True,True,False,False]
    decision=decision_from_values(p.future_common_r,p.future_direction_advantage_r)
    assert decision['action'].tolist()==[0,1,1,-1]


def test_asymmetric_loss_can_have_clear_direction_but_no_entry_value():
    d=decision_from_values(-2.55,2.45)
    assert d['q_long']==pytest.approx(-.1) and d['q_short']==pytest.approx(-5.)
    assert d['action']==0


def test_one_sided_future_does_not_imply_predictable_profitable_action():
    u=np.array([[1.,-1.4],[-1.4,1.]])
    C=u.mean(axis=1);D=(u[:,0]-u[:,1])/2
    assert np.max(u,axis=1).mean()==1.
    d=decision_from_values(C.mean(),D.mean())
    assert d['entry_value']==pytest.approx(-.2) and d['action']==0
    # Informative histories with a 70% conditional direction change the answer.
    d=decision_from_values(-.2,.7*1.2+.3*(-1.2))
    assert d['q_long']==pytest.approx(.28) and d['action']==1


def test_censoring_is_not_a_zero_or_neither_and_empty_is_explicit():
    f,o=sample([(1,0),(0,0),(1,-1)],statuses=[('COMPLETE','CENSORED'),('EMPTY','EMPTY'),('COMPLETE','COMPLETE')])
    p=paired_targets(f,o)
    assert p.future_joint_profit_state.tolist()==['UNKNOWN','NEITHER','LONG_ONLY']
    assert pd.isna(p.future_u_short.iloc[0]) and pd.isna(p.future_common_r.iloc[0])
    assert p.future_u_long.iloc[1]==p.future_u_short.iloc[1]==0
    assert not p.future_both_filled_complete.iloc[1]
    assert p.future_duration_seconds_long.iloc[1]==2


def test_maturity_is_exact_and_never_comes_from_a_float_clock():
    f,o=sample([(1,-1)]);p=paired_targets(f,o)
    expected=int(f.t_ns.iloc[0])+501
    assert p.future_pair_available_ns.dtype==pd.Int64Dtype()
    assert int(p.future_pair_available_ns.iloc[0])==expected
    assert not mature_pair_mask(p,expected-1).iloc[0]
    assert mature_pair_mask(p,expected).iloc[0]
    with pytest.raises(ValueError):mature_pair_mask(p,float(expected))


def test_duplicates_are_weighted_as_one_exact_action_contract():
    f,o=sample([(1,-1),(1,-1)])
    f.loc[1,'t_ns']=f.t_ns.iloc[0];o.loc[o.origin_index==1,'signal_ns']=f.t_ns.iloc[0]
    p=paired_targets(f,o)
    assert p.exact_decision_weight.tolist()==[.5,.5]


def test_side_exchange_preserves_common_and_flips_advantage():
    f,o=sample([(1,-2),(.3,.7)])
    p=paired_targets(f,o);o['side']=-o.side;r=paired_targets(f,o)
    np.testing.assert_allclose(p.future_common_r,r.future_common_r)
    np.testing.assert_allclose(p.future_direction_advantage_r,-r.future_direction_advantage_r)


def test_missing_or_duplicate_counterfactuals_are_rejected():
    f,o=sample([(1,-1)])
    with pytest.raises(ValueError):paired_targets(f,o.iloc[:1])
    with pytest.raises(ValueError):paired_targets(f,pd.concat([o,o.iloc[:1]]))


def test_value_decomposition_reconstructs_observed_utilities():
    f,o=sample([(1,-2),(-3,4),(.1,-.4)])
    p=paired_targets(f,o)
    np.testing.assert_allclose(p.future_common_r+p.future_direction_advantage_r,p.future_u_long_r)
    np.testing.assert_allclose(p.future_common_r-p.future_direction_advantage_r,p.future_u_short_r)


def quotes(prices):
    prices=np.asarray(prices,float);n=len(prices)
    return pd.DataFrame(dict(available_ns=np.arange(n,dtype=np.int64)*1_000_000_000,
                             bid_nano=np.rint((prices-.1)*1e9).astype(np.int64),
                             ask_nano=np.rint((prices+.1)*1e9).astype(np.int64),valid=True,epoch=1))


def test_fixed_window_description_keeps_full_path_not_only_a_stopped_trade():
    q=quotes([100,102,98,101,100,999])
    r=describe_future(q,0,100,1,horizon_seconds=5)
    assert r['future_description_known'] and r['future_range_r']==4.
    assert r['future_net_r']==0 and r['future_efficiency_1s']==0.
    assert r['future_origin_band_dwell_fraction']==pytest.approx(.4)
    # A jump at the right-open boundary cannot enter the fixed-window label.
    q.loc[5,['bid_nano','ask_nano']]=[1,2]
    assert describe_future(q,0,100,1,horizon_seconds=5)==r


def test_future_description_support_and_observation_horizon_are_separate():
    q=quotes([100]*6)
    assert not describe_future(q,0,100,1,horizon_seconds=10)['future_description_known']
    assert describe_future(q,0,100,1,horizon_seconds=5)['future_description_known']
    q.loc[2,'epoch']=2
    assert not describe_future(q,0,100,1,horizon_seconds=5)['future_description_known']


def test_same_range_and_end_price_can_hide_different_time_order():
    a=quotes([100,100,101,102,102,102,100])
    b=quotes([100,102,102,102,101,100,100])
    x=describe_future(a,0,100,1,horizon_seconds=6)
    y=describe_future(b,0,100,1,horizon_seconds=6)
    assert x['future_range_r']==y['future_range_r']==2
    assert x['future_time_center_migration_r']>0
    assert y['future_time_center_migration_r']<0


def test_future_source_truncation_at_declared_end_does_not_change_labels():
    q=quotes([100,100,101,102,102,102,999,999])
    x=describe_future(q,0,100,1,horizon_seconds=6)
    y=describe_future(q.iloc[:6],0,100,1,horizon_seconds=6,observed_through_ns=6_000_000_000)
    assert x==y
