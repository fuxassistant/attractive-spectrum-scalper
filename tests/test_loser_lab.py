"""First-passage, chronology, anti-triviality and independent risk contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.loser_lab_v1.targets import make_targets,post_cost_pairs,NS
from experiments.loser_lab_v1.models import EventModel,actions,rolling_predictions
from experiments.loser_lab_v1.account import replay
from experiments.loser_lab_v1.continuation import orient
from experiments.hypothesis_atlas_v1.engine import PATH_COLUMNS,MEMORY_COLUMNS,FAMILIES

CONFIG=json.loads((Path(__file__).parents[1]/'experiments/loser_lab_v1/config.json').read_text())


def tape(prices,times=None,invalid=None):
    n=len(prices);times=np.arange(n)*NS if times is None else np.array(times,dtype=np.int64)
    q=pd.DataFrame(dict(available_ns=times,bid_nano=np.rint(np.array(prices)*NS).astype(np.int64),epoch=np.zeros(n,np.int64),valid=True))
    q['ask_nano']=q.bid_nano+100_000_000
    if invalid is not None:q.loc[invalid,'valid']=False
    return q


def origin(t=0):return pd.DataFrame([dict(candidate_id=1,t_ns=t,date='2026-08-19',side=1,risk_per_oz=2.04)])
def target(q,seconds=3,**kw):return make_targets(q,origin(),dict(id='test',seconds=seconds,threshold=kw.get('threshold','fixed'),usd_per_oz=2),coverage=10*NS,config=CONFIG)


def test_first_passage_is_not_mfe_mae():
    y=target(tape([100,102,98,100]))
    assert y.iloc[0].state=='UP_CLEAN' and y.iloc[0].exit_ns==NS
    assert y.iloc[0].bid_delta_per_oz==2


def test_loser_clean_and_dirty_paths():
    assert target(tape([100,100.5,98,100])).iloc[0].state=='DOWN_CLEAN'
    assert target(tape([100,100.6,98,100])).iloc[0].state=='DOWN_DIRTY'


def test_threshold_exact_and_scaled_rounded_up():
    q=tape([100,101.999999999,102,102.1])
    assert target(q).iloc[0].exit_ns==2*NS
    scaled=target(q,threshold='max_fixed_risk').iloc[0]
    assert scaled.threshold_per_oz==2.1 and scaled.exit_ns==3*NS


def test_timeout_uses_deadline_and_common_feedback():
    y=target(tape([100,100.1,100.2,100.3])).iloc[0]
    assert y.state=='NONE' and y.exit_ns==3*NS and y.release_ns==3*NS
    early=target(tape([100,98,99,100])).iloc[0]
    assert early.exit_ns==NS and early.release_ns==3*NS


@pytest.mark.parametrize('kind',['invalid','epoch','gap'])
def test_missing_path_is_unknown(kind):
    q=tape([100,100,98,100])
    if kind=='invalid':q.loc[1,'valid']=False
    if kind=='epoch':q.loc[1:,'epoch']=1
    if kind=='gap':q.available_ns=[0,31*NS,32*NS,33*NS]
    y=target(q,seconds=33).iloc[0]
    assert not y.known and np.isnan(y.bid_delta_per_oz)


def test_gap_after_first_hit_does_not_erase_known_answer():
    q=tape([100,98,102,103],invalid=2)
    assert target(q).iloc[0].known


def test_bid_label_ignores_spread_fee_and_future_after_exit():
    q=tape([100,98,99,100]);a=target(q)
    q.ask_nano+=10*NS;q.loc[2:,'bid_nano']=10*NS
    b=target(q)
    pd.testing.assert_frame_equal(a,b)


def test_symmetric_actions_have_opposite_unit_outcomes():
    for prices in ([100,98,99,100],[100,102,100,99],[100,100.2,100.1,100]):
        r=target(tape(prices)).iloc[0]
        assert r.bid_delta_per_oz+-r.bid_delta_per_oz==0
        assert -r.event_sign+r.event_sign==0


def test_gate_does_not_reward_uninformative_activity():
    p=np.array([[0,.25,.25,.25,.25],[.9,.025,.025,.025,.025],[0,.1,0,.8,.1]])
    s,v,a=actions(p,.1)
    assert a.tolist()==[False,False,True] and s[2]==1
    _,clean,_=actions(np.array([[0,.1,0,0,.9]]),.1,clean=True)
    assert clean[0]<0


def test_post_cost_does_not_relabel_bid_target():
    q=tape([100,100,98,98,98],times=[0,200_000_000,NS,1_200_000_000,2*NS])
    y=target(q);before=y.copy(deep=True)
    p=post_cost_pairs(q,y,CONFIG,coverage=10*NS)
    pd.testing.assert_frame_equal(y,before)
    base=p[p['mode']=='base'].set_index('side')
    assert base.loc[1,'net_per_oz']==pytest.approx(-2.4)
    assert base.loc[-1,'net_per_oz']==pytest.approx(1.6)


def account_data(deltas):
    n=len(deltas);ids=np.arange(n)+1
    d=pd.DataFrame(dict(candidate_id=ids,t_ns=ids*10*NS,date='2026-08-19',side=1,accepted=True))
    y=pd.DataFrame(dict(candidate_id=ids,threshold_per_oz=2.,known=True,bid_delta_per_oz=deltas,exit_ns=ids*10*NS+NS,event_sign=np.sign(deltas)))
    return d,y


def test_martingale_only_changes_size_not_event_quality():
    d,y=account_data([-2,-2,2,-2]);a,la=replay(d,y,None,CONFIG)
    b,lb=replay(d,y,None,CONFIG,style='after_loss')
    assert lb.multiplier.tolist()==[1,2,4,1]
    assert la.event_reward.tolist()==lb.event_reward.tolist()
    assert a['net_usd']==-400 and b['net_usd']==0


def test_bounded_staking_and_no_borrowing():
    d,y=account_data([-2]*40);s,log=replay(d,y,None,CONFIG,style='after_loss')
    assert log.multiplier.max()==4 and (log.planned_risk_usd<=log.equity_before).all()
    assert s['unaffordable']>0 and s['depleted80'] and not s['bankrupt']


def test_unresolved_account_not_zero_and_nonoverlap():
    d,y=account_data([-2,-2,2]);y.loc[0,'exit_ns']=25*NS
    s,l=replay(d,y,None,CONFIG)
    assert s['overlap']==1 and l.candidate_id.tolist()==[1,3]
    y.loc[0,'known']=False;s,l=replay(d,y,None,CONFIG)
    assert not s['complete'] and s['unknown']==1 and l.empty


def test_feature_whitelist_excludes_teacher_and_equity():
    f=pd.DataFrame(dict(x=np.linspace(-1,1,20),date=['2026-08-03']*20,candidate_id=np.arange(20),known=True,class_index=[1]*10+[3]*10,release_ns=100,t_ns=0))
    model=EventModel(['x'],CONFIG['model']).fit(f);a=model.predict(f)
    f['equity_usd']=-1e50;f['class_index']=0;f['future_loss']=1e50
    assert np.array_equal(a,model.predict(f))


def test_continuation_representation_is_invariant_to_price_direction_reflection():
    fields=['distance_r','risk_per_oz','spread_per_oz','past_net30_r','past_net120_r','past_migration_r',
            'past_efficiency30','past_range120_r','past_zone_share_old','past_zone_share_new',
            'x_mem_log_median_dwell_seconds','x_mem_recent_pass_fraction','x_base_mbo_now_trade_total_log',
            'x_mbo_mbo_now_trade_unknown_fraction','x_base_mbo_now_known','x_mbo_mbo_now_trade_imbalance',
            'x_mbo_mbo_now_depth_1_toward_imbalance','x_mbo_mbo_now_book_net_pressure']
    f=pd.DataFrame([{**{k:.5 for k in fields},'side':1,'known':True,'past_known':True,'class_index':1}])
    a,cols=orient(f);reflected=f.copy();reflected['side']=-1;reflected['class_index']=3;b,_=orient(reflected)
    np.testing.assert_array_equal(a[cols],b[cols])
    assert a.class_index.iloc[0]==b.class_index.iloc[0] and a.momentum_side.iloc[0]==-b.momentum_side.iloc[0]


def test_small_impulse_cannot_activate_continuation_strategy():
    fields=['distance_r','risk_per_oz','spread_per_oz','past_net30_r','past_net120_r','past_migration_r',
            'past_efficiency30','past_range120_r','past_zone_share_old','past_zone_share_new',
            'x_mem_log_median_dwell_seconds','x_mem_recent_pass_fraction','x_base_mbo_now_trade_total_log',
            'x_mbo_mbo_now_trade_unknown_fraction','x_base_mbo_now_known','x_mbo_mbo_now_trade_imbalance',
            'x_mbo_mbo_now_depth_1_toward_imbalance','x_mbo_mbo_now_book_net_pressure']
    f=pd.DataFrame([{**{k:.01 for k in fields},'side':1,'known':True,'past_known':True,'class_index':1}])
    a,_=orient(f);assert not a.impulse_known.iloc[0]


def test_future_day_mutation_cannot_change_already_frozen_decisions():
    rows=[];rng=np.random.default_rng(123)
    for day in pd.bdate_range('2026-08-03','2026-08-10'):
        for j in range(12):
            r=dict(candidate_id=len(rows)+1,date=str(day.date()),t_ns=int(day.value)+j*600*NS+7*3600*NS,
                   side=1,target='test',known=True,class_index=1 if j%2 else 3,event_sign=1 if j%2 else -1,
                   past_known=True,x_base_mbo_now_known=1,x_base_mbo_now_trade_total_log=3.5,
                   median_dwell_seconds=20,completed_visits=4,x_mem_last_pass=1,past_aborts_per_visit=5)
            for c in PATH_COLUMNS+MEMORY_COLUMNS:r[c]=float(rng.normal())
            r['past_coverage']=1
            for k in FAMILIES:r['h_'+k]=j%2==0
            for s in ('trade_imbalance','trade_unknown_fraction','depth_1_toward_imbalance','depth_12_toward_imbalance','book_net_pressure','replenishment_fraction','matched_remove_fraction'):r['x_mbo_mbo_now_'+s]=.1
            r['release_ns']=r['t_ns']+300*NS;rows.append(r)
    f=pd.DataFrame(rows);c={**CONFIG,'train_end':'2026-08-04','deployment_start':'2026-08-07','extension_end':'2026-08-10'}
    e=json.loads((Path(__file__).parents[1]/'experiments/two_day_strategy_v1/config.json').read_text())
    a,_,_=rolling_predictions(f,c,e)
    poison=f.copy();future=poison.date.ge('2026-08-07');poison.loc[future,'class_index']=0;poison.loc[future,'event_sign']=0
    b,_,_=rolling_predictions(poison,c,e)
    pd.testing.assert_frame_equal(a[a.date=='2026-08-07'].reset_index(drop=True),b[b.date=='2026-08-07'].reset_index(drop=True))
