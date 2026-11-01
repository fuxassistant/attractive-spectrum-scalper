import numpy as np
import pandas as pd

from experiments.regime_language_v1.measures import TimeAtPrice,VwapContext,anchor_relations,scalar_time_mean,NS,MINUTE


def quotes(times,prices):
    p=np.asarray(prices,dtype=np.int64)*NS
    return pd.DataFrame(dict(available_ns=np.asarray(times,dtype=np.int64)*NS,
        bid_nano=p-50_000_000,ask_nano=p+50_000_000,valid=True,epoch=1))


def test_time_mass_is_seconds_not_number_of_messages():
    a=quotes([0,1,9,10],[100,101,102,999]);b=quotes([0,1,2,3,4,5,6,7,8,9,10],[100,101,101,101,101,101,101,101,101,102,999])
    pa=TimeAtPrice(a).profile(0,10*NS,10);pb=TimeAtPrice(b).profile(0,10*NS,10)
    assert pa['known'] and pa['supported_seconds']==10
    assert np.array_equal(pa['bin_seconds'],np.array([1.,8.,1.]))
    assert np.array_equal(pa['bin_seconds'],pb['bin_seconds'])
    assert pa['time_mean']==101 and pa['time_mean']==pb['time_mean']


def test_equal_whole_profile_can_have_opposite_temporal_migration():
    a=TimeAtPrice(quotes([0,5],[100,102])).facets(0,10*NS,10,10)
    b=TimeAtPrice(quotes([0,5],[102,100])).facets(0,10*NS,10,10)
    assert np.array_equal(a['bin_seconds'],b['bin_seconds'])
    assert a['center_migration_A']==2 and b['center_migration_A']==-2
    assert a['half_profile_overlap']==b['half_profile_overlap']==0


def test_grid_shift_does_not_move_continuous_time_center():
    q=quotes([0,2,4],[100,101,102]);m=TimeAtPrice(q)
    a=m.facets(0,6*NS,4,5,0);b=m.facets(0,6*NS,4,5,2.5)
    assert a['time_mean']==b['time_mean'] and a['time_sigma']==b['time_sigma']
    assert a['center_migration_A']==b['center_migration_A']
    assert a['bin_seconds'].sum()==b['bin_seconds'].sum()==6


def test_unknown_and_stale_time_are_not_filled_with_last_known_price():
    q=quotes([0,1,5],[100,101,102]);q.loc[1,'valid']=False
    p=TimeAtPrice(q,gap_seconds=2).profile(0,10*NS)
    assert p['supported_seconds']==3 and p['coverage']==.3 and not p['known']


def test_profile_is_prefix_invariant_and_matches_scalar_integral():
    q=quotes([0,1,3,7,9],[100,101,99,102,999]);m=TimeAtPrice(q)
    p=m.profile(NS//2,8*NS)
    short=TimeAtPrice(q[q.available_ns<8*NS]).profile(NS//2,8*NS)
    assert p['time_mean']==short['time_mean']
    assert np.array_equal(p['bin_seconds'],short['bin_seconds'])
    mean,support=scalar_time_mean(q,NS//2,8*NS)
    assert support==int(7.5*NS) and np.isclose(p['time_mean'],mean)


def context():
    f=pd.DataFrame(dict(t_ns=np.arange(1,66,dtype=np.int64)*MINUTE,
        available_ns=np.arange(1,66,dtype=np.int64)*MINUTE+100,known=True,interval_open=True,epoch=1,
        volume=2.,pv=20.,p2v=202.,day_coverage=1.,combined_coverage=1.,previous_session_coverage=1.))
    for n in ('day','combined','previous'):
        f[n+'_vwap']=4010.;f[n+'_vwap__known']=True
        if n!='previous':f[n+'_sigma']=1.;f[n+'_sigma__known']=True
    return f


def test_anchor_cannot_borrow_same_clock_publication_or_future_row():
    f=context();clock=int(f.available_ns.iloc[60]);a=VwapContext(f).at(clock)
    assert a['release_ns']<clock and a['endpoint_ns']==60*MINUTE
    assert a['anchors']['rolling_60m']['value']==4010.
    assert a['anchors']['rolling_60m']['sigma']==1.
    assert not a['anchors']['rolling_240m']['known']
    f.loc[60:,'day_vwap']=9999.
    assert VwapContext(f).at(clock)==a


def test_partial_rolling_anchor_is_unknown():
    f=context();f.loc[30,'known']=False
    a=VwapContext(f).at(int(f.available_ns.iloc[60])+1)
    assert not a['anchors']['rolling_60m']['known']


def test_relative_change_separates_price_and_line_motion():
    a=dict(anchors={'day':dict(value=100.,sigma=2.,known=True)})
    b=dict(anchors={'day':dict(value=102.,sigma=2.,known=True)})
    r=anchor_relations(a,b,101.,101.,100.5,10)['day']
    assert r['price_change_A']==0 and r['line_change_A']==2
    assert r['relative_change_A']==-2 and r['relative_identity_residual']==0
    assert r['state']=='BELOW'
