"""Causal phrase boundary regressions and physical/volume measure checks."""
import numpy as np
import pandas as pd
import pytest

from experiments.continuation_lab_v1.episodes import _pullbacks, _places, COLS
from experiments.continuation_lab_v1.context import context_features, quote_context, coarse_texture, frozen_levels, NS, MINUTE


def inputs(times, prices, bad=None, epochs=None):
    n=len(times)
    t=np.rint(np.asarray(times)*NS).astype(np.int64)
    p=np.rint(np.asarray(prices)*NS).astype(np.int64)
    good=np.ones(n,bool)
    if bad is not None:good[bad]=False
    e=np.ones(n,np.int64) if epochs is None else np.asarray(epochs,np.int64)
    return t,p,good,e


def pull(times,prices,**kwargs):
    t,p,g,e=inputs(times,prices,**kwargs)
    f=_pullbacks(t,p,g,e,np.zeros(len(t),np.int64),np.full(len(t),int(1.2*NS),np.int64),0,int((max(times)+1)*NS),1)
    return pd.DataFrame(f,columns=COLS)


def place(times,prices,family=3,**kwargs):
    t,p,g,e=inputs(times,prices,**kwargs)
    f=_places(t,p,g,e,0,int((max(times)+1)*NS),100*NS,int(1.2*NS),0,1,0,1,family)
    return pd.DataFrame(f,columns=COLS)


def test_pullback_confirms_after_hold_and_keeps_actual_extreme():
    f=pull([0,1,2,3,4,5,6.1],[100,101.3,100.6,100.5,100.9,101,101.1])
    r=f[f.phase_code==5].iloc[0]
    assert r.available_ns==6_100_000_000 and r.trough_index==3 and r.gate_index==4
    assert r.risk_nano==1_200_000_000


def test_first_pullback_overshoot_cannot_recover_into_valid_story():
    f=pull([0,1,2,3,4,5,6],[100,101.5,99.8,100.8,100.9,100.9,100.9])
    first=f[f.episode_id==1]
    assert first.phase_code.tolist()==[1,6]


def test_failed_hold_restarts_from_new_crossing():
    f=pull([0,1,2,3,4,5,6,7,8.1],[100,101.3,100.6,100.5,100.9,100.7,100.9,100.9,100.9])
    assert 4 in f.phase_code.to_list()
    r=f[f.phase_code==5].iloc[0]
    assert r.gate_index==6 and r.available_ns==8_100_000_000


@pytest.mark.parametrize('kwargs',[{'bad':[5]},{'epochs':[1,1,1,1,1,2,2]}])
def test_intervening_invalid_or_epoch_break_cancels_pullback(kwargs):
    f=pull([0,1,2,3,4,4.1,6.1],[100,101.3,100.6,100.5,100.9,101,101.1],**kwargs)
    assert not (f.phase_code==5).any() and (f.phase_code==8).any()


def test_stale_gap_cannot_count_as_hold():
    f=pull([0,1,2,3,4,40],[100,101.3,100.6,100.5,100.9,101.1])
    assert not (f.phase_code==5).any()


def test_side_mirror_has_identical_confirmation_clock():
    t,p,g,e=inputs([0,1,2,3,4,6.1],[100,101.3,100.6,100.5,100.9,101.1])
    ci=np.zeros(len(t),np.int64);r=np.full(len(t),1_200_000_000,np.int64)
    a=_pullbacks(t,p,g,e,ci,r,0,8*NS,1)
    b=_pullbacks(t,200*NS-p,g,e,ci,r,0,8*NS,-1)
    assert [(x[1],x[5]) for x in a]==[(x[1],x[5]) for x in b]


def test_repeat_requires_significant_departure_and_same_quote_can_prove_it():
    times=[0,1,2,3,4.1,6,7,8,9.1]
    prices=[101,100.2,100.6,100.7,101.3,100.2,100.6,100.6,100.7]
    f=place(times,prices)
    departure=f[f.phase_code==11].iloc[0]
    assert departure.available_ns==4_100_000_000
    assert f[f.phase_code==5].available_ns.tolist()==[9_100_000_000]
    prices[4]=100.8
    f=place(times,prices)
    assert not (f.phase_code==5).any()


def test_old_frozen_place_not_reactivated_after_epoch_change():
    f=place([0,1,2,3,4.1,6,7,8,9.1],[101,100.2,100.6,100.7,101.3,100.2,100.6,100.6,100.7],
            epochs=[1,1,1,1,1,2,2,2,2])
    assert not (f.phase_code==5).any() and (f.phase_code==8).any()


def test_break_retest_has_total_deadline_from_contact():
    times=list(range(0,202,10))+[205,206,209]
    prices=[100.1]*17+[100.7,100.7,100.7,100.2,100.2,100.7,100.7]
    f=place(times,prices,family=2)
    first=f[f.episode_id==1]
    assert not (first.phase_code==5).any()
    assert (first.phase_code==7).any()


def minute_frame(n=300):
    t=np.arange(1,n+1,dtype=np.int64)*MINUTE
    price=4000+np.arange(n)*.02
    vol=1+np.arange(n)%5
    return pd.DataFrame(dict(t_ns=t,available_ns=t+1000,known=True,interval_open=True,epoch=1,
                             high=price+.5,low=price-.5,close=price,volume=vol,
                             pv=(price-4000)*vol,p2v=(price-4000)**2*vol,bb1=price,bb1_half=1.))


def test_exact_trade_vwap_and_slope_use_only_complete_past():
    raw=minute_frame(); f=context_features(raw)
    j=280;window=raw.iloc[j-59:j+1]
    expected=4000+window.pv.sum()/window.volume.sum()
    assert f.vwap60.iloc[j]==pytest.approx(expected,abs=1e-10)
    assert f.slope60.iloc[j]==pytest.approx(f.vwap60.iloc[j]-f.vwap60.iloc[j-15],abs=1e-10)
    clipped=context_features(raw.iloc[:j+1])
    pd.testing.assert_frame_equal(f.iloc[:j+1],clipped)
    raw.loc[270,'known']=False
    assert np.isnan(context_features(raw).vwap60.iloc[j])


def test_quote_at_release_does_not_read_same_time_context():
    f=context_features(minute_frame())
    now=int(f.available_ns.iloc[-1])
    q=pd.DataFrame({'available_ns':[now,now+1],'epoch':[1,1]})
    ci,r=quote_context(q,f)
    assert ci.tolist()==[len(f)-2,len(f)-1]
    assert (r>0).all()


def quote_frame(times,prices):
    t,p,g,e=inputs(times,prices)
    return pd.DataFrame(dict(available_ns=t,bid_nano=p-100_000_000,ask_nano=p+100_000_000,epoch=e,valid=g))


def test_texture_does_not_read_future_before_first_quote_or_skip_hidden_invalid():
    q=quote_frame([1,1.1,1.2,2,3],[100,101,102,103,104])
    assert coarse_texture(q,0,3*NS)=={'texture_known':False}
    q.loc[2,'valid']=False
    assert not coarse_texture(q,NS,3*NS)['texture_known']


def test_frozen_mode_never_reads_current_block_and_grid_is_explicit():
    context=context_features(minute_frame(60))
    start=60*MINUTE+NS;end=start+30*MINUTE
    t=np.arange(0,5402,dtype=np.int64)
    p=np.where(t<=3601,4001.1,4500.)
    q=quote_frame(t,p)
    a=frozen_levels(q,context,start,end)
    b=frozen_levels(q[q.available_ns<start],context,start,end)
    assert a==b
    assert a[0]['known'] and a[0]['level']==4001.25
    assert a[1]['kind']=='grid' and a[1]['level']==4001.


def test_administrative_end_is_not_backdated_to_last_quote():
    f=pull([0,1,2],[100,101.3,100.6])
    r=f[f.phase_code==14].iloc[0]
    assert r.available_ns==3*NS and r.source_index==2
