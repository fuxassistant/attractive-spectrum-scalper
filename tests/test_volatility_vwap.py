import json
import numpy as np
import pytest
from spectrum.lab import Context
from spectrum.calendar import MarketCalendar
from spectrum.plugins.volatility_vwap import compute,trade_moments,PROXY,EXACT
from spectrum.feature_stream import FeatureStream
from test_foundation import frame


def calc(f,basis='midpoint_proxy',scales=(60,300)):
    c=Context(f,MarketCalendar());compute(c,basis=basis,scales=scales);return c


def test_weighted_geometry_against_direct_oracle_and_zero_volume():
    f=frame(400);f.c_trade_buy=np.cumsum(np.arange(len(f))%7+1);f.c_trade_sell=0.
    c=calc(f);end=350;n=60;px=f.mid.to_numpy()[end-n+1:end+1]
    w=np.diff(f.c_trade_buy.to_numpy()[end-n:end+1]);mean=np.average(px,weights=w)
    sigma=np.sqrt(np.average((px-mean)**2,weights=w));s='vwap_proxy_60s_'
    assert c.values[s+'center'][end]==pytest.approx(mean,abs=1e-11)
    assert c.values[s+'sigma_ticks'][end]==pytest.approx(sigma/.1,rel=1e-10)
    assert c.values[s+'effective_seconds'][end]==pytest.approx(w.sum()**2/(w@w))
    assert c.values[s+'weighted_age_seconds'][end]==pytest.approx(np.average(np.arange(59,-1,-1),weights=w))
    parts=sum(c.values[s+k][end] for k in ('z_change_price_part_5s','z_change_center_part_5s','z_change_width_part_5s'))
    assert parts==pytest.approx(c.values[s+'z'][end]-c.values[s+'z'][end-5],abs=1e-10)
    f.c_trade_buy=0.;d=calc(f)
    assert not d.masks[s+'center'].any()
    assert d.masks[s+'realized_variation_1s'][60:].all()


def test_exact_within_bucket_dispersion_cannot_be_recovered_from_midpoint():
    f=frame(400);f.mid=100.;f.c_trade_buy=np.arange(400)*2.;f.c_trade_sell=0.
    f.c_trade_unknown=np.arange(400)*2.
    m=trade_moments([99.,101.],[2.,2.])
    for k,v in m.items():f[k]=v
    f['trade_moments_valid']=True;f['trade_moments_available_ns']=f.available_ns
    a=calc(f,'trade_exact');b=calc(f)
    assert a.values['vwap_exact_60s_sigma_ticks'][100]==pytest.approx(10)
    assert b.values['vwap_proxy_60s_sigma_ticks'][100]==0
    assert not b.masks['vwap_proxy_60s_z'][100]
    assert a.values['vwap_exact_60s_unknown_fraction'][100]==.5
    with pytest.raises(ValueError,match='aligned T-print'):calc(frame(),'trade_exact')
    f.loc[100,'trade_volume']=3
    with pytest.raises(ValueError,match='disagrees'):calc(f,'trade_exact')


def test_prefix_gap_epoch_calendar_and_late_dependency():
    f=frame(500);a=calc(f.iloc[:350]);f.loc[350:,'mid']+=1000;b=calc(f)
    for k in a.values:
        np.testing.assert_array_equal(a.values[k],b.values[k][:350])
        np.testing.assert_array_equal(a.masks[k],b.masks[k][:350])
    f=frame(500);f.loc[150,'valid']=False;f.loc[300:,'epoch']=2
    c=calc(f);mask=c.masks['vwap_proxy_60s_center']
    assert not mask[150:211].any() and mask[211]
    assert not mask[300:360].any() and mask[360]
    f=frame(500);f.loc[100,'available_ns']=f.t_ns.iloc[150];c=calc(f)
    assert not c.masks['vwap_proxy_60s_center'][101:150].any()
    assert c.masks['vwap_proxy_60s_center'][150]
    f=frame(500);f.t_ns+=11*3600*10**9;f.available_ns+=11*3600*10**9
    assert not calc(f).masks['vwap_proxy_60s_center'].any()


def test_sampling_rv_constant_trend_and_unweighted_vs_volume_sigma():
    f=frame(400);f.mid=100+np.arange(400)*.1;f.c_trade_unknown=np.arange(400)
    c=calc(f);s='vwap_proxy_60s_';i=300
    assert c.values[s+'realized_variation_1s'][i]==pytest.approx(60)
    assert c.values[s+'realized_variation_5s'][i]==pytest.approx(300)
    assert c.values[s+'return_std_1s'][i]<1e-6
    assert c.values[s+'sigma_ticks'][i]>10
    assert c.values[s+'sigma_ticks'][i]==pytest.approx(c.values[s+'time_price_sigma_ticks'][i],rel=1e-9)


def test_stream_halo_checkpoint_and_batch_parity():
    from spectrum.lab import compute as compute_lab,load_plugins
    specs=('spectrum.plugins.volatility_vwap:PROXY',);load_plugins(specs)
    f=frame(8000);f.loc[1800,'valid']=False
    whole=compute_lab(f,(PROXY.name,));engine=FeatureStream((PROXY.name,),plugins=specs,source_id='vwap-test')
    pieces=[]
    for lo in range(0,len(f),499):
        pieces.append(engine.feed(f.iloc[lo:lo+499]))
        if lo==998:engine=FeatureStream.restore(json.loads(json.dumps(engine.checkpoint())),expected_source_id='vwap-test')
    for k in whole.values:
        np.testing.assert_array_equal(np.concatenate([p['known'][k] for p in pieces]),whole.masks[k])
        np.testing.assert_allclose(np.concatenate([p['values'][k] for p in pieces]),whole.values[k],rtol=1e-8,atol=1e-8)
    assert len(engine.tail)<=3606


def test_exact_stable_moments_and_dependency_mask():
    m=trade_moments([1e9,1e9+.1],[1,3])
    assert m['trade_price_m2']==pytest.approx(.0075,rel=1e-6)
    f=frame(400);f['trade_volume']=2.;f['trade_price_mean']=f.mid
    f['trade_price_m2']=0.;f['trade_moments_valid']=True;f['trade_moments_available_ns']=f.available_ns
    f.loc[100,'trade_moments_available_ns']=f.t_ns.iloc[120]
    c=calc(f,'trade_exact')
    assert not c.masks['vwap_exact_60s_center'][100:120].any()
    assert c.masks['vwap_exact_60s_center'][120]
