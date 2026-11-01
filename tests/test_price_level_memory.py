import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from experiments.price_level_memory_v1.memory import build_memory,NS

def tape(times,prices,valid=None,epoch=None):
    p=np.round(np.asarray(prices)*1e9).astype(np.int64)
    return pd.DataFrame(dict(available_ns=np.asarray(times,dtype=np.int64)*NS,bid_nano=p-50_000_000,
        ask_nano=p+50_000_000,valid=valid if valid is not None else True,epoch=epoch if epoch is not None else 1))

def band(f,t,price):
    return f[(f.query_ns==t*NS)&(f.price_low==price)].iloc[0]

def test_known_duration_conserved_and_native_quotes_not_counted_as_time():
    q=tape([0,1,3,8],[100.1,100.1,100.6,100.6])
    f=build_memory(q,np.array([7])*NS)
    assert band(f,7,100).mid_dwell_s==3
    assert band(f,7,100.5).mid_dwell_s==4
    assert f.mid_dwell_s.sum()==7
    assert f.bid_dwell_s.sum()==7
    assert f.ask_dwell_s.sum()==7

def test_jump_over_does_not_fabricate_contacts():
    f=build_memory(tape([0,1],[100.1,102.1]),np.array([1])*NS)
    for p in [100.5,101,101.5]:
        row=band(f,1,p)
        assert row.jump_over==1
        assert row.arrivals==0 and row.present_at_restart==0
        assert row.mid_dwell_s==0 and np.isnan(row.observed_age_s)

def test_first_observation_not_known_arrival():
    f=build_memory(tape([0,1],[100.1,100.1]),np.array([1])*NS)
    r=band(f,1,100)
    assert r.arrivals==0 and r.present_at_restart==1 and r.active==1

def test_hysteresis_and_later_maturity():
    q=tape([0,1,2,3,4],[99.8,100.1,99.95,100.05,99.7])
    f=build_memory(q,np.array([1,3,4])*NS)
    assert band(f,1,100).arrivals==1
    assert band(f,3,100).arrivals==1 and band(f,3,100).reject_down==0
    assert band(f,4,100).reject_down==1

def test_invalid_gap_censors_and_does_not_infer_return():
    q=tape([0,1,2,3],[99.8,100.1,100.1,100.1],[True,True,False,True])
    f=build_memory(q,np.array([3])*NS)
    r=band(f,3,100)
    assert r.censored==1 and r.arrivals==1 and r.present_at_restart==1
    assert r.pass_up+r.pass_down+r.reject_up+r.reject_down==0
    assert f.mid_dwell_s.sum()==2

def test_staleness_not_carried_through_missing_time():
    q=tape([0,100],[100.1,101.1])
    f=build_memory(q,np.array([50,100])*NS,max_gap_seconds=30)
    assert f[f.query_ns==50*NS].mid_dwell_s.sum()==30
    r=band(f,50,100)
    assert r.censored==1 and not r.age_interval_known
    assert band(f,100,101).present_at_restart==1

def test_prefix_causality_and_view_independent_absolute_identity():
    q=tape([0,1,2,3,4,100],[100.1,100.6,99.8,100.1,100.8,120.1])
    queries=np.array([2,3,4])*NS
    full=build_memory(q,queries)
    prefix=build_memory(q.iloc[:5],queries)
    assert_frame_equal(full,prefix)
    for t in queries:
        direct=build_memory(q,np.array([t]))
        existing=full[full.query_ns==t]
        assert_frame_equal(existing.reset_index(drop=True),direct.reset_index(drop=True))

def test_age_tracks_supported_persistence_until_departure():
    q=tape([0,20],[100.1,101.1])
    f=build_memory(q,np.array([25])*NS)
    assert band(f,25,100).observed_age_s==5

def test_future_range_does_not_change_earlier_viewport():
    q=tape([0,1,100],[100.1,100.6,200.1])
    full=build_memory(q,np.array([1,100])*NS)
    prefix=build_memory(q.iloc[:2],np.array([1])*NS)
    assert_frame_equal(full[full.query_ns==NS].reset_index(drop=True),prefix)

def test_memory_survives_leaving_viewport_and_returning():
    q=tape([0,1,2],[100.1,200.1,100.1])
    f=build_memory(q,np.array([0,1,2])*NS,radius_ticks=10)
    assert not len(f[(f.query_ns==NS)&(f.price_low==100)])
    assert band(f,2,100).mid_dwell_s==1
    assert band(f,2,100).present_at_restart==1
    assert band(f,2,100).arrivals==1

def test_float_clock_rejected_instead_of_silent_ns_rounding():
    q=tape([0,1],[100.1,100.6]);q.available_ns=q.available_ns.astype(float)
    with pytest.raises(ValueError,match='Integer'):
        build_memory(q,np.array([1])*NS)

def test_age_interval_support_is_not_unvisited_or_market_interest():
    f=build_memory(tape([0,1],[100.1,101.1]),np.array([1])*NS)
    r=band(f,1,100.5)
    assert r.no_contact_in_observed_history and not r.age_interval_known
    assert r.history_missing_seconds==0
    assert band(f,1,101).age_interval_known
