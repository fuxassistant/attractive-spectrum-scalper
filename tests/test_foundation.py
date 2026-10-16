from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from spectrum.core import StreamConfig
from spectrum.core.schema import SNAP_SCHEMA
from spectrum.core.vendor.l3 import F_LAST
from spectrum.replay import MarketReplay
from spectrum.calendar import MarketCalendar
from spectrum.lab import compute,rolling_valid,FeaturePlugin,register
from test_mbo_adapter import bootstrap_rows,rec


def test_query_never_exposes_partial_group_and_checkpoint_retains_original_ordinal():
    p=MarketReplay(StreamConfig(1,7,22))
    p.feed_batch(bootstrap_rows())
    before=p.query()
    p.feed(rec(100,1,'M','B',101,20,1,flags=0))
    assert p.query().bid_nano==before.bid_nano
    assert p.query().source_ordinal==2
    q=MarketReplay.restore(json.loads(json.dumps(p.checkpoint())))
    assert q.query()==p.query()
    # An arbitrary source ordinal gap must not become an invented committed ID.
    p.feed(rec(500,1,'M','A',101.2,20,2))
    q.feed(rec(500,1,'M','A',101.2,20,2))
    assert p.query()==q.query() and p.query().source_ordinal==500
    assert before.bid_nano==100*10**9
    with pytest.raises(Exception):before.bid_nano=0


def test_no_future_on_bounded_delivery():
    p=MarketReplay(StreamConfig(1,7,22))
    rows=bootstrap_rows()+[rec(3,3,'M','B',99,20,1)]
    assert p.feed_batch(rows,stop_available_ns=1*10**9)==3
    assert p.query().bid_nano==100*10**9
    with pytest.raises(TypeError):p.feed_batch(iter(rows),stop_available_ns=10**9)


def frame(n=500):
    t=pd.Timestamp('2026-05-04T10:00Z').value+np.arange(n)*10**9
    d={name:np.zeros(n,dtype=dtype) for name,dtype in SNAP_SCHEMA}
    d.update(t_ns=t,available_ns=t+100,epoch=np.ones(n,dtype='int64'),valid=np.ones(n,bool),book_valid=np.ones(n,bool),
             mid=100+np.sin(np.arange(n)*2*np.pi/8),bid=np.full(n,99.9),ask=np.full(n,100.1),
             bid_size=np.full(n,10.),ask_size=np.full(n,12.))
    for k in d:
        if k.startswith('c_'):d[k]=np.arange(n,dtype=float)
    d['c_trade_unknown']=np.zeros(n)
    d['c_path_ticks']=np.r_[0,np.cumsum(abs(np.diff(d['mid']))/.1)]
    return pd.DataFrame(d)


def test_future_suffix_does_not_change_any_feature():
    f=frame();a=compute(f.iloc[:300]);b=compute(f)
    for name in a.values:
        np.testing.assert_array_equal(a.values[name],b.values[name][:300])
        np.testing.assert_array_equal(a.masks[name],b.masks[name][:300])


def test_gap_reset_and_unknown_trade_cannot_be_zero_filled_into_fft():
    f=frame();f.loc[200,'valid']=False;f.loc[350:,'epoch']=2
    f.loc[150:,'c_trade_unknown']=1
    c=compute(f)
    assert not c.masks['fft_mid_change_32s_power'][200:233].any()
    assert not c.masks['fft_signed_trade_32s_power'][150:182].any()
    assert not c.masks['fft_mid_change_32s_power'][350:382].any()
    assert np.all(c.values['fft_mid_change_32s_power'][~c.masks['fft_mid_change_32s_power']]==0)


def test_fft_independent_dft_and_direction_invariance():
    f=frame(200);c=compute(f,['spectral']);n=32;t=100
    x=np.diff(f.mid.to_numpy()[t-n:t+1])/.1
    w=.5-.5*np.cos(2*np.pi*np.arange(n)/n);a=(x-x.mean())*w
    z=np.array([sum(a[j]*np.exp(-2j*np.pi*k*j/n) for j in range(n)) for k in range(n//2+1)])
    p=abs(z)**2/sum(w*w);p[1:-1]*=2
    assert c.values['fft_mid_change_32s_power'][t]==pytest.approx(p[1:].sum()/n,rel=1e-12)
    f.mid=-f.mid;d=compute(f,['spectral'])
    np.testing.assert_allclose(c.values['fft_mid_change_32s_power'],d.values['fft_mid_change_32s_power'])
    np.testing.assert_allclose(c.values['fft_mid_change_32s_mean'],-d.values['fft_mid_change_32s_mean'])


def test_calendar_boundaries_and_weekend():
    stamps=['2026-05-04T20:59:59Z','2026-05-04T21:00Z','2026-05-04T22:00Z',
            '2026-05-08T21:00Z','2026-05-09T15:00Z','2026-05-10T21:59:59Z','2026-05-10T22:00Z']
    assert MarketCalendar().closed(pd.to_datetime(stamps,format='mixed').asi8).tolist()==[False,True,False,True,True,True,False]


def test_plugin_selection_and_duplicate_guard():
    c=compute(frame(),['motion'])
    assert not any(n.startswith('fft_') for n in c.values)
    with pytest.raises(ValueError):register(FeaturePlugin('motion','1',lambda c:None,1))


def test_stale_publication_excluded_without_erasing_history():
    f=frame();f.loc[170,'available_ns']=f.loc[170,'t_ns']+2*10**9
    c=compute(f)
    assert not c.admitted[170]
    assert not c.masks['displacement_5s'][170]
    assert c.masks['displacement_5s'][176]
    assert c.admitted[176]


def test_streamed_feature_halo_restart_and_batch_parity():
    from spectrum.feature_stream import FeatureStream
    f=frame(500);f.loc[250,'valid']=False
    whole=compute(f);stream=FeatureStream(source_id='fixture')
    pieces=[]
    for lo in range(0,len(f),37):
        pieces.append(stream.feed(f.iloc[lo:lo+37]))
        if lo==148:stream=FeatureStream.restore(json.loads(json.dumps(stream.checkpoint())),expected_source_id='fixture')
    for name in whole.values:
        np.testing.assert_allclose(np.concatenate([p['values'][name] for p in pieces]),whole.values[name],rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(np.concatenate([p['known'][name] for p in pieces]),whole.masks[name])
    assert len(stream.tail)<=129
    with pytest.raises(ValueError):FeatureStream.restore(stream.checkpoint(),expected_source_id='wrong-source')


def test_feature_checkpoint_detects_calendar_implementation_drift():
    from spectrum.feature_stream import FeatureStream
    from spectrum.core.state import seal,unseal
    stream=FeatureStream(source_id='fixture');stream.feed(frame(150))
    saved=unseal(stream.checkpoint());saved['calendar_sha256']='obsolete-implementation'
    with pytest.raises(ValueError,match='calendar drift'):
        FeatureStream.restore(seal(saved),expected_source_id='fixture')
