import numpy as np
import pandas as pd
import pytest

from experiments.mechanical_lab.execution import QuoteTape, ExecutionConfig
from experiments.zone_episode_lab.game import simulate_zone_pairs, run_account


def tape(times, bids, *, valid=None, coverage=1000., base=0):
    ts = np.rint(np.asarray(times)*1e9).astype(np.int64)+base
    q = pd.DataFrame(dict(available_ns=ts,
        bid_nano=np.rint(np.asarray(bids)*1e9).astype(np.int64),
        ask_nano=np.rint((np.asarray(bids)+.2)*1e9).astype(np.int64),
        valid=np.ones(len(ts), bool) if valid is None else valid,
        epoch=np.zeros(len(ts), np.int64)))
    return QuoteTape.from_frame(q, coverage_end_ns=base+int(coverage*1e9), entry_start_ns=base)


def candidates(times=(0,), *, target=101.5, side=1, risk=10., base=0):
    n=len(times)
    return pd.DataFrame(dict(candidate_id=np.arange(n,dtype=np.int64),origin_index=np.arange(n,dtype=np.int64),
        t_ns=np.rint(np.asarray(times)*1e9).astype(np.int64)+base,
        side=np.full(n,side,np.int64),target_nano=np.full(n,round(target*1e9),np.int64),
        risk_per_oz=np.full(n,risk,float),date=['2026-08-24']*n,regime=['test']*n,time_regime=['test']*n,
        episode_id=[f'episode{i}' for i in range(n)],zone_available_ns=np.full(n,base,np.int64)))


def test_frozen_absolute_zone_target_and_matched_mirrored_control():
    q=tape([0,.1,.11,.3,.4,.41,2.11,2.21,2.22],[100,100,100,101.5,101.5,101.3,101.3,101.3,101.3])
    out=simulate_zone_pairs(q,candidates(),holding_seconds=2.)
    long,short=out.iloc[0],out.iloc[1]
    assert long.status=='COMPLETE' and long.reason=='TAKE'
    assert (long.entry_ns,long.trigger_ns,long.exit_ns)==(110_000_000,300_000_000,410_000_000)
    assert long.target_nano==101_500_000_000 and short.target_nano==98_700_000_000
    assert long.target_expected_net_at_signal==pytest.approx(1.)
    assert short.target_expected_net_at_signal==pytest.approx(1.)
    assert long.net_per_oz==pytest.approx(.8)
    assert long.episode_id==short.episode_id=='episode0'


def test_unconditional_target_overrun_keeps_fill_latency_and_loss():
    q=tape([0,.1,.11,.21,.22],[100,100,101.6,101.7,101.4])
    row=simulate_zone_pairs(q,candidates(),holding_seconds=2.).iloc[0]
    assert row.status=='COMPLETE' and row.reason=='TARGET_OVERRUN'
    assert row.target_overrun_at_entry
    assert row.entry_ns==row.trigger_ns==110_000_000
    assert row.exit_ns==220_000_000 and row.net_per_oz==pytest.approx(-.7)
    assert row.entry_price==pytest.approx(101.8)


def test_overrun_with_unsupported_exit_is_censored_not_relabelled_success():
    q=tape([0,.11,.16,.22],[100,101.6,101.5,101.4],valid=[True,True,False,True])
    row=simulate_zone_pairs(q,candidates(),holding_seconds=2.).iloc[0]
    assert row.status=='CENSORED' and row.reason=='SUPPORT'
    assert row.target_overrun_at_entry and row.entry_ns==110_000_000
    assert np.isnan(row.net_per_oz)


def test_adverse_entry_slippage_can_overrun_target_and_both_slips_are_charged():
    q=tape([0,.11,.21,.22],[100]*4)
    row=simulate_zone_pairs(q,candidates(target=100.3),holding_seconds=2.,
        config=ExecutionConfig(slippage_ticks_per_fill=2.)).iloc[0]
    assert row.reason=='TARGET_OVERRUN' and row.status=='COMPLETE'
    assert row.net_per_oz==pytest.approx(-.9)


def test_fee_does_not_move_frozen_quote_target_or_reject_negative_expected_net():
    q=tape([0,.11,.3,.4,.41],[100,100,100.4,100.4,100.4])
    c=candidates(target=100.4)
    a=simulate_zone_pairs(q,c,holding_seconds=2.,config=ExecutionConfig(fee_roundtrip_per_oz=.1)).iloc[0]
    b=simulate_zone_pairs(q,c,holding_seconds=2.,config=ExecutionConfig(fee_roundtrip_per_oz=.6)).iloc[0]
    assert not b.target_overrun_at_entry and b.target_net_at_entry<0
    assert a.status==b.status=='COMPLETE' and a.trigger_ns==b.trigger_ns==300_000_000
    assert b.net_per_oz==pytest.approx(a.net_per_oz-.5)


def test_invalid_entry_prefix_cannot_be_skipped_even_when_target_is_ahead():
    q=tape([0,.05,.11,.3,.41],[100,100,100,102,102],valid=[True,False,True,True,True])
    out=simulate_zone_pairs(q,candidates(),holding_seconds=2.)
    assert (out.status=='CENSORED').all() and (out.reason=='SUPPORT').all()
    assert (out.entry_ns==0).all() and out.target_net_at_entry.isna().all()


def test_timer_at_deadline_wins_against_same_time_target_quote():
    q=tape([0,.11,1.,1.11,1.21,1.22],[100,100,100,102,102,102])
    row=simulate_zone_pairs(q,candidates(),holding_seconds=1.).iloc[0]
    assert row.reason=='TIME' and row.trigger_ns==1_110_000_000
    assert row.exit_ns==1_220_000_000


def test_outcome_of_inside_zone_origin_is_not_filtered_by_adapter():
    c=candidates(target=100.4);c['location']='inside';c['zone_low_nano']=99_500_000_000;c['zone_high_nano']=100_500_000_000
    q=tape([0,.11,.3,.41],[100,100,100.4,100.4])
    out=simulate_zone_pairs(q,c,holding_seconds=2.)
    assert out.location.eq('inside').all() and out.iloc[0].status=='COMPLETE'


def test_account_one_position_and_pending_or_open_trade_overlap():
    times=np.arange(0,6,.1);q=tape(times,np.full(len(times),100.))
    c=candidates([0,.5,2.]);out=simulate_zone_pairs(q,c,holding_seconds=1.)
    stats,trades=run_account(c,out)
    assert stats['trades']==2 and stats['overlap']==1
    assert stats['net_usd']==pytest.approx(-100.)
    assert (trades.signal_ns.to_numpy()[1:]>trades.exit_ns.to_numpy()[:-1]).all()
    assert trades.episode_id.tolist()==['episode0','episode2']


def test_filled_censoring_halts_account_without_invented_zero_return():
    times=np.arange(0,6,.1);valid=np.ones(len(times),bool);valid[5]=False
    q=tape(times,np.full(len(times),100.),valid=valid)
    c=candidates([0,2.]);out=simulate_zone_pairs(q,c,holding_seconds=1.)
    stats,trades=run_account(c,out)
    assert stats['halted'] and stats['unresolved']==1 and not stats['capital_curve_complete']
    assert stats['trades']==0 and trades.empty


def test_random_direction_stays_matched_to_candidate_identity_after_overlap():
    times=np.arange(0,6,.1);q=tape(times,np.full(len(times),100.))
    c=candidates([0,2.]);c['candidate_id']=[100,200];c['origin_index']=[100,200]
    extra=c.iloc[[0]].copy();extra['candidate_id']=999;extra['origin_index']=999;extra['t_ns']=500_000_000
    expanded=pd.concat([c,extra],ignore_index=True)
    _,a=run_account(c,simulate_zone_pairs(q,c,holding_seconds=1.),direction='coin',seed=34)
    _,b=run_account(expanded,simulate_zone_pairs(q,expanded,holding_seconds=1.),direction='coin',seed=34)
    assert a[['candidate_id','side']].equals(b[['candidate_id','side']])


def test_stress_keeps_origin_geometry_and_changes_executable_costs():
    times=np.arange(0,6,.1);q=tape(times,np.full(len(times),100.));c=candidates()
    base=simulate_zone_pairs(q,c,holding_seconds=1.)
    stress=simulate_zone_pairs(q,c,holding_seconds=1.,config=ExecutionConfig(fee_roundtrip_per_oz=.6,latency_ms=250.,slippage_ticks_per_fill=1.))
    np.testing.assert_array_equal(base.target_nano,stress.target_nano)
    np.testing.assert_allclose(base.net_per_oz,-.5)
    np.testing.assert_allclose(stress.net_per_oz,-1.)
    assert (stress.entry_ns>base.entry_ns).all()


def test_nanosecond_identity_causal_target_and_duplicate_validation():
    base=1_787_558_880_000_000_000
    q=tape([0,.11,.3,.41],[100,100,102,102],base=base);c=candidates(base=base)
    row=simulate_zone_pairs(q,c,holding_seconds=2.).iloc[0]
    assert int(row.entry_ns)==base+110_000_000
    future=c.copy();future['zone_available_ns']=base+1
    with pytest.raises(ValueError,match='later than'):
        simulate_zone_pairs(q,future,holding_seconds=2.)
    with pytest.raises(ValueError,match='unique'):
        simulate_zone_pairs(q,pd.concat([c,c],ignore_index=True),holding_seconds=2.)


def test_only_decision_time_wrong_side_geometry_is_ineligible():
    q=tape([0,.11,.3,.41],[100,100,102,102])
    out=simulate_zone_pairs(q,candidates(target=99.),holding_seconds=2.)
    assert (out.status=='INELIGIBLE').all() and (out.reason=='TARGET_GEOMETRY').all()
    assert (out.entry_ns==0).all()


def test_empty_candidates_and_missing_paired_failure_are_explicit():
    q=tape([0,.11],[100,100]);c=candidates().iloc[:0]
    assert simulate_zone_pairs(q,c).empty
    stats,trades=run_account(c,simulate_zone_pairs(q,c))
    assert stats['trades']==0 and trades.empty
    c=candidates();out=simulate_zone_pairs(q,c)
    with pytest.raises(ValueError,match='Missing counterfactual'):
        run_account(c,out.iloc[:1])


def test_absolute_parent_deadline_closes_both_directions_on_same_known_clock():
    q=tape([0,.11,.8,1.,1.1,1.11],[100]*6)
    c=candidates();c['deadline_ns']=np.int64(1_000_000_000);c['deadline_available_ns']=np.int64(0)
    out=simulate_zone_pairs(q,c,holding_seconds=2.)
    assert out.status.eq('COMPLETE').all() and out.reason.eq('TIME').all()
    assert out.trigger_ns.eq(1_000_000_000).all() and out.exit_ns.eq(1_110_000_000).all()
    np.testing.assert_allclose(out.effective_holding_limit_seconds,.89)
    assert not out.deadline_expired_at_entry.any()


def test_parent_deadline_expiring_during_entry_latency_keeps_costed_fill():
    q=tape([0,.11,.21,.22],[100]*4)
    c=candidates();c['deadline_ns']=np.int64(50_000_000)
    out=simulate_zone_pairs(q,c,holding_seconds=2.)
    assert out.status.eq('COMPLETE').all() and out.reason.eq('DEADLINE_EXPIRED_AT_ENTRY').all()
    assert out.deadline_expired_at_entry.all()
    assert out.entry_ns.eq(110_000_000).all() and out.trigger_ns.eq(110_000_000).all()
    assert out.exit_ns.eq(220_000_000).all()
    np.testing.assert_allclose(out.net_per_oz,-.5)
    np.testing.assert_allclose(out.effective_holding_limit_seconds,0.)


def test_deadline_availability_and_future_prefix_do_not_change_completed_path():
    q=tape([0,.11,.8,1.,1.11],[100]*5)
    c=candidates();c['deadline_ns']=np.int64(1_000_000_000);c['deadline_available_ns']=np.int64(1)
    with pytest.raises(ValueError,match='later than'):
        simulate_zone_pairs(q,c,holding_seconds=2.)
    c['deadline_available_ns']=np.int64(0)
    full=tape([0,.11,.8,1.,1.11,1.5],[100,100,100,100,100,1],valid=[True,True,True,True,True,False])
    pd.testing.assert_frame_equal(simulate_zone_pairs(q,c,holding_seconds=2.),simulate_zone_pairs(full,c,holding_seconds=2.))
