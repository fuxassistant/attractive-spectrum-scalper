from types import SimpleNamespace
import numpy as np
import pandas as pd
from spectrum.player.contract import Contract, NS
from spectrum.player_v3.data import (asof_indices, bounds, encode_events, EVENT_VALUES,
    EVENT_NAMES, Day, observed_previous_date, anchor_values, support_segment_starts)


def test_asof_late_publication_cannot_join_back_or_skip_sequence_order():
    t=np.arange(1,6,dtype=np.int64)*NS
    av=np.array([1,2,9,4,5],np.int64)*NS
    i,k=asof_indices(t,av,np.array([0,3,8,9],np.int64)*NS)
    assert i.tolist()==[0,1,1,4] and k.tolist()==[False,True,True,True]
    # Future mutation cannot alter an earlier as-of lookup.
    av[4]=20*NS
    j,_=asof_indices(t,av,np.array([3,8],np.int64)*NS)
    assert j.tolist()==[1,1]


def test_session_bounds_dst_and_previous_observed_friday():
    c=Contract()
    a,b,l,n=bounds('2026-05-04',c)
    assert pd.Timestamp(a,tz='UTC').hour==22
    assert pd.Timestamp(b,tz='UTC').hour==21
    assert pd.Timestamp(l,tz='UTC').hour==7
    assert n==b and b-a==23*3600*NS
    class Calendar:
        def closed(self,t):
            d=pd.to_datetime(t,utc=True)
            return np.asarray(d.dayofweek>=5)
    assert observed_previous_date('2026-05-04',Calendar(),c)=='2026-05-01'


def test_event_allowlist_ignores_future_labels_and_has_no_raw_prices():
    fields={n:[1.] for n in EVENT_VALUES}
    fields.update(event_class=['fresh_impulse'],activity=['price_led'],future_pnl=[123.],
                  retrospectively_no_repeat=[True],event_start_price=[4500.])
    d=pd.DataFrame(fields); x=encode_events(d)
    d['future_pnl']=-1e9;d['retrospectively_no_repeat']=False;d['event_start_price']=9999
    np.testing.assert_array_equal(x,encode_events(d))
    assert x.shape==(1,len(EVENT_NAMES)-3)
    assert not any('future' in n or n.endswith('_price') for n in EVENT_NAMES)


def test_memory_slices_expire_seconds_and_keep_only_available_events():
    d=object.__new__(Day)
    d.decision_t=np.array([1000,2000],np.int64)*NS
    d.fast_t=np.arange(1,2001,dtype=np.int64)*NS
    d.fast_available=d.fast_t+NS//2
    d.fast_x=np.zeros((2000,64),np.float32);d.fast_valid=np.ones(2000,bool)
    d.slow_t=np.array([60,120,960,1020],np.int64)*NS
    d.slow_x=np.zeros((4,71),np.float32);d.slow_session=np.array([0,0,1,1])
    d.slow_available=d.slow_t+NS//2
    d.event_t=np.array([900,999],np.int64)*NS
    d.event_available=np.array([900,1001],np.int64)*NS
    d.event_x=np.zeros((2,25));d.event_prices=np.array([[100,101],[101,102]])
    d.anchors=np.zeros((2,38));d.decision_mid=np.array([102,103])
    d.fast_cuts=np.searchsorted(d.fast_available,d.decision_t,side='right')
    d.slow_cuts=np.searchsorted(d.slow_available,d.decision_t,side='right')
    d.event_cuts=np.searchsorted(d.event_available,d.decision_t,side='right')
    m=d.memory(0)
    assert m['fast_t'].min()>100*NS and len(m['fast'])<=900
    assert m['slow_t'].max()==960*NS and len(m['events'])==1
    assert m['events'][0,-2:].tolist()==[20.,10.]
    d.event_x[1]=1e8
    np.testing.assert_array_equal(m['events'],d.memory(0)['events'])


def test_fast_support_reset_after_gap_invalid_or_epoch():
    t=np.array([1,2,3,4,6,7,8,9])*NS
    valid=np.array([1,1,0,1,1,1,1,1],bool)
    epoch=np.array([0,0,0,0,0,0,1,1])
    assert support_segment_starts(t,valid,epoch).tolist()==[0,0,3,3,4,4,6,6]


def test_anchors_openings_not_visible_until_publication_and_running_not_future():
    s=pd.DataFrame(dict(t_ns=np.array([1,60,61,120])*NS,
        available_ns=np.array([2,62,63,121])*NS,valid=[True]*4,mid=[100.,101.,102.,999.]))
    b=pd.DataFrame(dict(h=[101.,102.,999.],l=[100.,101.,102.],valid=[True]*3))
    decision=np.array([2,62,121])*NS
    x,_=anchor_values(s,b,decision,None,0,60*NS,'2026-05-04',None,False,np.array([],np.int64))
    assert x[0,19+2]==0 and x[1,19+2]==1 # London opening mask
    assert x[0,9]==10 and x[1,9]==20 # causal high/low prefix
    b.loc[2,'h']=1e9
    y,_=anchor_values(s,b,decision,None,0,60*NS,'2026-05-04',None,False,np.array([],np.int64))
    np.testing.assert_array_equal(x[:2],y[:2])
