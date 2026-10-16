from types import SimpleNamespace
import numpy as np
import pytest
import torch
from spectrum.player.contract import Contract,NS
from spectrum.player.gym import TradingGym
from spectrum.player.model import Player,Scaler
from spectrum.player.data import session_bounds


def day(times,bids,asks=None,decision=(0,60),valid=None,epoch=None,allowed=None,deadline=1000):
    t=np.asarray(times,dtype=np.int64)*NS
    b=np.rint(np.asarray(bids)*1e9).astype(np.int64)
    a=b+100_000_000 if asks is None else np.rint(np.asarray(asks)*1e9).astype(np.int64)
    dt=np.asarray(decision,dtype=np.int64)*NS
    return SimpleNamespace(quote_t=t,source_t=t.copy(),bid=b,ask=a,
                           valid=np.ones(len(t),bool) if valid is None else np.asarray(valid),
                           epoch=np.zeros(len(t),np.int64) if epoch is None else np.asarray(epoch,np.int64),
                           decision_t=dt,cuts=np.searchsorted(t,dt,side='right'),x=np.zeros((len(dt),71),np.float32),
                           can_enter=np.ones(len(dt),bool) if allowed is None else np.asarray(allowed),
                           deadline=deadline*NS,meta={'ny_close_ns':str(deadline*NS)})

def c(**kw):return Contract(support_gap_ns=2000*NS,**kw)

@pytest.mark.parametrize('side,bids,expected',[(1,[100,100,102,102],1.6),(4,[100,100,98,98],1.6)])
def test_spread_fee_and_next_quote(side,bids,expected):
    d=day([0,1,2,60],bids);env=TradingGym(d,c(),True)
    _,r,_,_,_=env.step(side)
    assert r==pytest.approx(expected)
    assert env.trades[0]['entry_index']==1
    assert env.trades[0]['exit_index']==2
    assert env.trades[0]['reason']=='take'
    assert env.trades[0]['net']==pytest.approx(expected)
    assert np.nanmax(env.trace[:env.q])-np.nanmin(env.trace[:env.q])>0

def test_intraminute_stop_overshoot_and_telescoping():
    d=day([0,1,2,3,60,61,120],[100,100,98,103,103,103,103],decision=(0,60,120))
    e=TradingGym(d,c());rewards=[]
    for a in [1,0,0]:rewards.append(e.step(a)[1])
    assert e.trades[0]['reason']=='stop'
    assert e.trades[0]['net']==pytest.approx(-2.4)
    assert sum(rewards)==pytest.approx(e.metrics()['net'])
    assert e.f[3]==pytest.approx(2.4)

def test_wait_keeps_future_participation_and_context_gate():
    d=day([0,1,60,61,62,120],[100,100,100,100,102,102],decision=(0,60,120),allowed=[False,True,True])
    e=TradingGym(d,c())
    with pytest.raises(ValueError):e.step(1)
    e.step(0);assert e.mask()[1];e.step(1);e.step(0)
    assert len(e.trades)==1 and e.trades[0]['decision_index']==1

def test_latency_equal_timestamp_and_pending_expiry():
    d=day([0,1,1,2,3,60],[100]*6)
    e=TradingGym(d,c(latency_ns=NS));e.step(1)
    assert e.s[3]==3 # all publications exactly at the due timestamp excluded
    d2=day([0,10,60],[100]*3);e=TradingGym(d2,c());e.step(1)
    assert e.metrics()['trades']==0 and e.cancelled==1

def test_support_loss_never_silent():
    d=day([0,1,2,3,60],[100]*5,valid=[True,True,False,True,True])
    e=TradingGym(d,c());e.step(1);e.step(0)
    assert e.trades[0]['reason']=='support_loss' and e.metrics()['censored']==1
    assert not e.metrics()['complete']

def test_epoch_boundary_and_terminal_censor():
    d=day([0,1,2,60],[100]*4,epoch=[0,0,1,1]);e=TradingGym(d,c());e.step(1)
    assert e.trades[0]['censored']==1
    d=day([0,1,60],[100]*3);e=TradingGym(d,c());e.step(1);e.step(0)
    assert e.metrics()['open_terminal'] and not e.metrics()['complete']

def test_daily_risk_and_last_plan_deadline():
    d=day([0,1,2,60],[100,100,90,90]);e=TradingGym(d,c(daily_drawdown_stop=2));e.step(1)
    assert e.trades[0]['reason']=='risk_stop';assert not e.mask()[1:].any()
    d=day([0,1,60],[100]*3,deadline=121);e=TradingGym(d,c())
    assert not e.mask()[1:].any()

def test_dst_and_agreed_session_hours():
    import pandas as pd
    _,_,l,ny=session_bounds('2026-05-04',Contract())
    assert pd.Timestamp(l,tz='UTC').hour==7
    assert pd.Timestamp(ny,tz='UTC').hour==21
    _,_,l,ny=session_bounds('2026-03-16',Contract())
    assert pd.Timestamp(l,tz='UTC').hour==8 # US and UK DST do not change together
    assert pd.Timestamp(ny,tz='UTC').hour==21

def test_recurrent_prefix_invariance_and_gradient():
    torch.manual_seed(2);m=Player(71);x=torch.randn(1,30,71);a=torch.zeros(1,30,7)
    with torch.no_grad():
        full,_=m(x,a);prefix,_=m(x[:,:12],a[:,:12])
    torch.testing.assert_close(full[:,:12],prefix,rtol=1e-5,atol=1e-6)
    logits,v=m(x,a);(logits.square().mean()+v.square().mean()).backward()
    assert m.lstm.weight_ih_l0.grad.norm()>0

def test_scaler_only_fit_inputs():
    d1=SimpleNamespace(x=np.zeros((3,71),np.float32));d2=SimpleNamespace(x=np.ones((3,71),np.float32)*100)
    s=Scaler().fit([d1]);before=s.center.copy();s.transform(d2.x)
    np.testing.assert_array_equal(s.center,before)
