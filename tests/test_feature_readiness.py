"""Failure-oriented contracts found in the real week audit."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from spectrum.core import StreamConfig
from spectrum.core.vendor.l3 import MboRecord,F_LAST,F_SNAPSHOT
from spectrum.market.replay import MarketReplay
from spectrum.market.continuity import verified_handoff
from spectrum.features.pipeline import compute
from spectrum.features.families.spectral_activity import white_reference
from spectrum.representation.representation import grouped_rolling,make_fast,make_minutes,MarketObservationStore
from spectrum.representation.session import SessionObservationStore,session_context,session_bounds
from spectrum.representation.session import latest_features
from test_foundation import frame
from test_observations import synthetic_fast

NS=10**9


def record(i,t,action,side='N',price=0,size=0,oid=0,flags=F_LAST):
    t=round(t*NS)
    return MboRecord(i,1,7,22,t,t,i,oid,round(price*NS),size,flags,action,side)


def handoff_fixture(drift=False):
    old=MarketReplay(StreamConfig(1,7,22,snapshot_ns=NS))
    old.feed_batch([record(0,0,'R',flags=F_SNAPSHOT),record(1,0,'A','B',100,20,1,F_SNAPSHOT),
                    record(2,0,'A','A',100.2,20,2,F_SNAPSHOT|F_LAST),record(3,9.9,'N')])
    new=[record(0,10,'R',flags=F_SNAPSHOT),record(1,10,'A','B',100,21 if drift else 20,1,F_SNAPSHOT),
         record(2,10,'A','A',100.2,20,2,F_SNAPSHOT|F_LAST)]
    new += [record(i+3,10.2+i,'N') for i in range(5)]
    return old.checkpoint(),new


def test_handoff_requires_full_book_and_real_frontier():
    checkpoint,records=handoff_fixture()
    h=verified_handoff(checkpoint,records,next_config=StreamConfig(1,7,22,snapshot_ns=NS,fragment=1),
                       boundary_ns=10*NS,prefix_end_ns=14*NS)
    assert h.proof['endpoint_recovered']
    assert h.seconds.t_ns.tolist()==list(range(10*NS,15*NS,NS))
    assert (h.seconds.available_ns>h.seconds.t_ns).all()
    assert h.seconds.epoch.nunique()==1
    checkpoint,bad=handoff_fixture(drift=True)
    with pytest.raises(ValueError,match='Full order book differs'):
        verified_handoff(checkpoint,bad,next_config=StreamConfig(1,7,22,snapshot_ns=NS),boundary_ns=10*NS,prefix_end_ns=14*NS)
    with pytest.raises(ValueError,match='Incomplete handoff'):
        verified_handoff(checkpoint,records[:3],next_config=StreamConfig(1,7,22,snapshot_ns=NS),boundary_ns=10*NS,prefix_end_ns=14*NS)


@pytest.mark.parametrize('op',['sum','mean','std','min','max'])
def test_vectorized_rolling_equals_independent_run_windows(op):
    rng=np.random.default_rng(4)
    x=rng.normal(size=300);x[152]=np.nan
    g=np.r_[np.arange(80),np.repeat(80,100),np.repeat(81,120)]
    expected=pd.Series(x).groupby(g,sort=False).transform(lambda a:getattr(a.rolling(9,min_periods=9),op)()).to_numpy()
    np.testing.assert_allclose(grouped_rolling(x,g,9,op),expected,rtol=1e-12,atol=1e-12,equal_nan=True)
    assert len(grouped_rolling([],[],9,op))==0


def test_activity_keeps_unknown_trade_and_signed_bounds():
    f=frame(400)
    f.c_trade_buy=np.arange(400)*3.
    f.c_trade_sell=np.arange(400)*1.
    f.c_trade_unknown=np.arange(400)*2.
    c=compute(f,['spectral','spectral_activity'])
    assert not c.masks['fft_signed_trade_128s_power'][200]
    assert c.masks['fft_total_trade_128s_power'][200]
    assert c.values['activity_30s_unknown_fraction'][200]==pytest.approx(1/3)
    assert c.values['activity_30s_signed_lower_bound'][200]==pytest.approx(0)
    assert c.values['activity_30s_signed_upper_bound'][200]==pytest.approx(2/3)
    # Unknown is observable uncertainty; no guessed aggressor direction.
    assert c.values['fft_unknown_trade_128s_mean'][200]==pytest.approx(2)


def test_white_reference_matches_exact_operator_covariance():
    n=32;k=np.arange(n);w=.5-.5*np.cos(2*np.pi*k/n)
    operator=np.diag(w)@(np.eye(n)-np.ones((n,n))/n)
    fourier=np.exp(-2j*np.pi*np.arange(n//2+1)[:,None]*k[None,:]/n)
    p=np.sum(abs(fourier@operator)**2,axis=1)/(w@w)
    p[1:-1]*=2;p[0]=0
    mask=(np.fft.rfftfreq(n)>=1/8)&(np.fft.rfftfreq(n)<=1/4)
    assert white_reference(n,mask)==pytest.approx(p[mask].sum()/p.sum(),abs=1e-14)


def test_closing_interval_is_known_but_decision_is_closed():
    s,l,t=synthetic_fast(180,start='2026-08-25 20:58')
    f=make_fast(s,l,t)
    row=f[f.t_ns==pd.Timestamp('2026-08-25 21:00',tz='UTC').value].iloc[0]
    assert row.known and row.interval_open and not row.decision_open
    m=make_minutes(f,t)
    assert m.loc[m.t_ns==row.t_ns,'known'].item()


def test_late_trade_delays_fast_and_minute_publication_without_rounding():
    s,l,t=synthetic_fast(120)
    delayed=int(s.t_ns.iloc[65])+113
    t.loc[0,'available_ns']=delayed
    f=make_fast(s,l,t)
    assert f.available_ns.iloc[0]==delayed
    m=make_minutes(f,t)
    assert m.available_ns.iloc[0]>=delayed


def test_missing_ledger_does_not_erase_exact_trade_channel():
    s,l,t=synthetic_fast(70)
    s['ledger_complete']=True;s.loc[30,'ledger_complete']=False
    s['trade_complete']=True
    f=make_fast(s,l,t)
    tensor=MarketObservationStore._tensor(f,['trade_quantity','bid_add'],int(f.available_ns.max()))
    assert tensor['known'][30,0] and not tensor['known'][30,1]


def test_price_is_centered_before_float32_conversion():
    f=pd.DataFrame(dict(mid=[4600.05],known=[True],scheduled_open=[True],t_ns=[1],available_ns=[1]))
    tensor=MarketObservationStore._normalize(MarketObservationStore._tensor(f,['mid'],1),4600.,.1)
    assert tensor['values'][0,0]==4600.05
    assert tensor['normalized'][0,0]==pytest.approx(.5,abs=1e-7)


def test_historical_context_without_optional_spectral_channels_stays_readable():
    from spectrum.representation.representation import OPTIONAL_DAY_VALUES
    s,l,t=synthetic_fast(1800,start='2026-08-24 08:00')
    f=make_fast(s,l,t);m=make_minutes(f,t)
    old=session_context(m,t,'2026-08-24','2026-08-21').drop(columns=OPTIONAL_DAY_VALUES)
    zones=pd.DataFrame(dict(zone_id=pd.Series(dtype=str),available_ns=pd.Series(dtype='int64')))
    events=pd.DataFrame(dict(available_ns=pd.Series(dtype='int64')))
    store=MarketObservationStore(f,old,zones,events,symbol='fixture',date='2026-08-24',previous_date='2026-08-21')
    result=store.observe(int(f.available_ns.iloc[-1]))
    assert len(result.day['names'])==36
    assert not set(OPTIONAL_DAY_VALUES)&set(result.day['names'])


def test_float_feature_row_does_not_round_nanosecond_clock():
    t=int(pd.Timestamp('2026-08-24',tz='UTC').value)+113
    f=pd.DataFrame(dict(t_ns=[t],available_ns=[t+7],a=[1.],a_known=[1.]))
    r=latest_features(f,t+7,mask_suffix='_known')
    assert r['t_ns']==t and r['available_ns']==t+7


def test_session_poc_is_nonempty_and_future_suffix_does_not_change_prefix():
    s,l,t=synthetic_fast(1800,start='2026-08-23 22:00')
    m=make_minutes(make_fast(s,l,t),t)
    full=session_context(m,t,'2026-08-24','2026-08-21')
    cutoff=int(m.t_ns.iloc[14]);partial=session_context(m[m.t_ns<=cutoff],t[t.t_ns<cutoff],'2026-08-24','2026-08-21')
    a=full[full.t_ns<=cutoff].reset_index(drop=True);b=partial[partial.t_ns<=cutoff].reset_index(drop=True)
    assert a.day_poc.notna().sum()==15
    for n in ('day_poc','day_vwap','day_sigma','day_coverage'):
        np.testing.assert_allclose(a[n],b[n],equal_nan=True,rtol=0,atol=1e-10)
    assert not a.previous_poc__known.any()


def test_completed_geometry_has_explicit_sample_and_legacy_population_modes():
    s,l,t=synthetic_fast(1800)
    f=make_fast(s,l,t)
    sample=make_minutes(f,t);population=make_minutes(f,t,bb_ddof=0)
    valid=sample.bb1_half.notna()
    np.testing.assert_allclose(sample.loc[valid,'bb1_half']/population.loc[valid,'bb1_half'],np.sqrt(20/19))
    assert sample.attrs['bb_geometry']['clock']=='completed_native'


def test_session_bounds_dst_and_night_access_do_not_skip_invalid_native_quote(tmp_path):
    assert session_bounds('2026-08-24')[0]==pd.Timestamp('2026-08-23 22:00',tz='UTC').value
    start,_=session_bounds('2026-11-02')
    assert start==pd.Timestamp('2026-11-01 23:00',tz='UTC').value
    s,l,t=synthetic_fast(1800,start='2026-08-23 22:00')
    f=make_fast(s,l,t);m=make_minutes(f,t)
    context=session_context(m,t,'2026-08-24','2026-08-21')
    assert not context.attrs['context_contract']['previous_admitted']
    q=pd.DataFrame(dict(t_ns=[s.t_ns.iloc[1],s.t_ns.iloc[2]],available_ns=[s.t_ns.iloc[1]+111,s.t_ns.iloc[2]+113],
                       source_ordinal=[5,6],epoch=[1,2],bid_nano=[100*NS,0],ask_nano=[101*NS,0],
                       bid_size=[1,0],ask_size=[1,0],valid=[True,False]))
    x=pd.DataFrame(dict(t_ns=s.t_ns,available_ns=s.available_ns,a=1.,a__known=True))
    core=x.rename(columns={'a__known':'a_known'})
    zones=pd.DataFrame(dict(zone_id=pd.Series(dtype=str),available_ns=pd.Series(dtype='int64')))
    events=pd.DataFrame(dict(available_ns=pd.Series(dtype='int64')))
    store=SessionObservationStore(f,context,zones,events,quotes=q,features=x,core_features=core,
                                  symbol='fixture',date='2026-08-24',previous_date='2026-08-21')
    first=store.observe(int(q.available_ns.iloc[0]))
    assert first.market['known']
    second=store.observe(int(q.available_ns.iloc[1]))
    assert not second.market['known'] and second.market['reason']=='INVALID_BOOK'
    assert second.market['source_ordinal']==6
    assert second.second_features['known'].tolist()==[True]
    from spectrum.representation.session_store import publish_session_store,open_session_store
    from spectrum.io import sha
    quote_path=tmp_path/'quotes.parquet';q.to_parquet(quote_path,index=False)
    path=tmp_path/'store'
    publish_session_store(path,date='2026-08-24',previous_date='2026-08-21',symbol='fixture',
        tables=dict(fast=f,context=context,zones=zones,events=events,features=x,core_features=core),
        quote_sources=[dict(path=str(quote_path),sha256=sha(quote_path))],source_identity=dict(fixture=True))
    restored=open_session_store(path).observe(int(q.available_ns.iloc[1]))
    assert restored.market==second.market
    with (path/'features.parquet').open('ab') as file:file.write(b'drift')
    with pytest.raises(ValueError,match='Session payload drift'):
        open_session_store(path)
