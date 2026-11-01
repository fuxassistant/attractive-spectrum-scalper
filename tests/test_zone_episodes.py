import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from experiments.zone_episode_lab.episodes import build_episodes,occupancy_grid,NS


def tape(times,prices,valid=None,epoch=None):
    p=np.rint(np.array(prices)*NS).astype(np.int64)
    return pd.DataFrame(dict(available_ns=np.array(times,dtype=np.int64)*NS,bid_nano=p-50_000_000,
        ask_nano=p+50_000_000,valid=True if valid is None else valid,epoch=1 if epoch is None else epoch,
        source_ordinal=np.arange(len(p),dtype=np.int64)))


def zone(f,low=100):return f[f.zone_low_nano==int(low*NS)]


def test_short_edge_excursion_is_not_a_confirmed_break():
    q=tape([0,1,2,3,4,8],[99.5,100.5,101.2,100.5,102.2,102.2])
    ev,ep=build_episodes(q)
    z=zone(ev)
    assert len(z[z.event_type=='ARRIVAL'])==1
    r=z[z.event_type=='EDGE_RETURN'].iloc[0]
    assert r.available_ns==3*NS and r.duration_ns==NS
    d=z[z.event_type=='DEPARTURE_CONFIRMED'].iloc[0]
    assert d.available_ns==7*NS
    p=zone(ep).iloc[0]
    assert p.outcome=='PASS_UP' and p.core_dwell_seconds==2


def test_gate_abort_and_return_history_are_separate_events():
    q=tape([0,1,2,4,5,9,12,13,17],[99.5,100.5,102.2,101.5,102.2,103.2,100.5,98.5,98.5])
    ev,ep=build_episodes(q)
    z=zone(ev);p=zone(ep)
    assert len(z[z.event_type=='GATE_ABORT'])==1
    assert list(p.outcome)==['PASS_UP','PASS_DOWN']
    assert p.iloc[1].previous_departure_ns==8*NS
    assert p.iloc[1].return_available_ns==12*NS
    assert p.iloc[1].absence_seconds==4
    assert p.iloc[1].absence_peak_distance_ticks==22
    assert 0<p.iloc[1].approach_efficiency<=1


def test_jump_does_not_create_an_arrival_and_restart_is_not_return():
    q=tape([0,1,2,3],[99.5,102.5,100.5,100.5],[True,True,False,True])
    ev,ep=build_episodes(q)
    z=zone(ev)
    assert len(z[z.event_type=='JUMP_OVER'])==1
    assert not len(z[z.event_type=='ARRIVAL'])
    assert len(z[z.event_type=='PRESENT_ON_RESTART'])==1
    assert not zone(ep).has_previous_departure.any()


def test_gap_censors_and_never_invents_full_dwell():
    ev,ep=build_episodes(tape([0,100],[100.5,100.5]))
    p=zone(ep)
    assert p.iloc[0].end_ns==30*NS
    assert p.iloc[0].core_dwell_seconds==30
    assert list(p.outcome)==['CENSORED','CENSORED']


def test_confirmed_history_prefix_invariance_with_future_price_range():
    q=tape([0,1,2,6,7,11,100],[99.5,100.5,102.5,100.5,98.5,98.5,150.5])
    ev,ep=build_episodes(q)
    a,b=build_episodes(q,end_ns=11*NS)
    # Endpoint right censoring is deliberately not a matured market outcome.
    assert_frame_equal(ep[(ep.end_ns<=11*NS)&ep.completed].reset_index(drop=True),b[b.completed].reset_index(drop=True))
    assert_frame_equal(ev[(ev.available_ns<=11*NS)&ev.event_type.ne('CENSORED')].reset_index(drop=True),a[a.event_type.ne('CENSORED')].reset_index(drop=True))


def test_exact_occupancy_split_and_missingness():
    q=tape([0,3,5,7,10],[100.5,101.5,101.5,100.5,100.5],[True,True,False,True,True])
    g=occupancy_grid(q,start_ns=0,end_ns=10*NS)
    i=int(np.flatnonzero(g['price_bin_nano']==100*NS)[0])
    assert np.allclose(g['occupancy'][:,i],[.6,.6])
    assert list(g['known'])==[True,False]
    assert np.allclose(g['support_seconds'],[5,3])


def test_exact_confirmation_time_not_first_gate_contact():
    q=tape([0,1,2,3,4,5],[99.5,100.5,102,102,102,102])
    ev,_=build_episodes(q)
    d=zone(ev);d=d[d.event_type=='DEPARTURE_CONFIRMED']
    assert list(d.available_ns)==[5*NS]

def test_zero_duration_invalid_event_invalidates_occupancy_support():
    q=tape([0,5,5,10],[100.5]*4,[True,False,True,True])
    g=occupancy_grid(q,start_ns=0,end_ns=10*NS)
    assert list(g['known'])==[True,False]

def test_occupancy_extent_does_not_look_past_declared_end():
    q=tape([0,5,10,20],[100.5,100.5,100.5,200.5])
    a=occupancy_grid(q,start_ns=0,end_ns=10*NS)
    b=occupancy_grid(q.iloc[:3],start_ns=0,end_ns=10*NS)
    for k in a:assert np.array_equal(a[k],b[k])

def test_endpoint_float_clock_is_rejected():
    with pytest.raises(ValueError,match='Integer end_ns'):
        build_episodes(tape([0,1],[100.5,101.5]),end_ns=float(NS))
