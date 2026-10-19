import numpy as np
from spectrum.playground.basket import BasketConfig,initial_state,replay_day

def go(b,a,signal=1,state=None,valid=None,**kwargs):
    n=len(b);t=np.arange(n,dtype=np.int64)*200_000_000
    p=BasketConfig(step_scale=1,target_scale=1,cooldown_minutes=1).array()
    for k,v in kwargs.pop('params',{}).items():p[k]=v
    return replay_day(t,np.array(b,dtype=np.int64)*100_000_000,
        np.array(a,dtype=np.int64)*100_000_000,np.ones(n,dtype=bool) if valid is None else np.array(valid),
        np.ones(n,np.int32),np.array([0],np.int64),np.array([signal],np.int8),np.array([1.]),p,
        initial_state() if state is None else state,0,10**12,int(t[-1]),True,**kwargs)

def test_latency_target_and_commission():
    s,t,e,_=go([999,999,1014,1015,1015],[1000,1000,1015,1016,1016])
    assert list(e[:,1])==[1,3]
    assert list(t)==[200_000_000,600_000_000]
    assert abs(s[0]-1501.2)<1e-8
    assert s[12]==1 and s[9]==0

def test_add_is_real_bidask_fill_not_grid_level():
    s,t,e,_=go([999,999,989,979,999,1019,1020],[1000,1000,990,980,1000,1020,1021])
    assert list(e[:2,1])==[1,2]
    assert e[1,3]==98 and e[1,4]==1.5 and e[1,5]==2.5
    assert abs(s[14]-.75)<1e-8
    assert s[0]>1500

def test_support_loss_does_not_invent_exit_or_death():
    s,t,e,_=go([999,999,900,999],[1000,1000,901,1000],valid=[True,True,False,True])
    assert list(e[:,1])==[1,7]
    assert s[9]==1 and s[8]==0 and s[11]==0
    assert abs(s[1]-1499.6)<1e-8

def test_strict_series_half_and_carry():
    s,t,e,_=go([9999,9999,2500,2499,2499],[10000,10000,2501,2500,2500],fee=0.,params={7:.5})
    assert s[8]==1 and s[1]<750
    assert t[-1]==600_000_000  # exactly $750 at preceding publication survives
    z=initial_state();z[0:3]=1400;z[4]=1500;z[5]=1400;z[7]=1
    s,t,e,_=go([999,999,1014,1015,1015],[1000,1000,1015,1016,1016],state=z)
    assert s[4]==1500 and s[7]==1 and s[12]==0

def test_short_symmetry():
    s,t,e,_=go([1000,1000,985,984,984],[1001,1001,986,985,985],signal=-1)
    assert list(e[:,1])==[1,3] and s[0]>1500

def test_configuration_rejects_nonfinite():
    import pytest
    with pytest.raises(ValueError):BasketConfig(multiplier=float('nan'))

def test_features_do_not_depend_on_future_candles():
    from types import SimpleNamespace
    from spectrum.playground.basket import signals
    rng=np.random.default_rng(123);x=np.zeros((500,71),np.float32)
    x[:,8]=1;x[:,1]=rng.uniform(3,20,500);x[:,3]=rng.normal(0,5,500)
    day=SimpleNamespace(x=x,bar_t=np.arange(500)*60*10**9,can_enter=np.ones(500,bool))
    before=signals(day,BasketConfig())
    day.x[350:,1:4]*=100
    after=signals(day,BasketConfig())
    assert np.array_equal(before[0][:350],after[0][:350])
    assert np.allclose(before[1][:350],after[1][:350])

def test_expired_entry_does_not_fill_later():
    p=BasketConfig().array();t=np.array([0,3*10**9,4*10**9],np.int64)
    price=np.full(3,100*10**9,np.int64)
    s,_,e,_=replay_day(t,price,price+100_000_000,np.ones(3,bool),np.ones(3,np.int32),
        np.array([0],np.int64),np.array([1],np.int8),np.array([1.]),p,initial_state(),0,10*10**9,9*10**9,True)
    assert len(e)==0 and s[0]==1500 and s[10]==0

def test_censored_run_is_saved_without_fabricated_pnl(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import json
    from spectrum.playground import basket_search as search
    t=np.array([0,200_000_000,400_000_000,600_000_000],np.int64)
    d=SimpleNamespace(quote_t=t,bid=np.full(4,100*10**9,np.int64),ask=np.full(4,100*10**9+100_000_000,np.int64),
        valid=np.array([True,True,False,True]),epoch=np.ones(4,np.int32),decision_t=np.array([0],np.int64),
        meta={'london_ns':0,'ny_close_ns':10**12},deadline=10**12)
    monkeypatch.setattr(search,'MonthDay',lambda path:d)
    monkeypatch.setattr(search,'signals',lambda day,c:(np.array([1],np.int8),np.array([1.])))
    r=search.evaluate_record(tmp_path,['synthetic'],BasketConfig(),tmp_path/'result')
    saved=json.loads((tmp_path/'result/result.json').read_text())
    assert not saved['result']['complete'] and saved['result']['censored']
    assert saved['events'][-1]['kind']==7 and saved['events'][-1]['basket_net'] is None
    assert saved['result']['closed_baskets']==0 and saved['result']['ruined']==0
