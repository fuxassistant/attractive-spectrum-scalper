from types import SimpleNamespace
import numpy as np
import pytest
from spectrum.playground.equity_envelope import envelope

def fixture():
    t=np.array([0,1,5,10,20,40,60,90],np.int64)*10**9
    bid=np.array([100,100,103,110,90,101,101,101])*10**9
    d=SimpleNamespace(meta={'start_ns':'0','end_ns':str(120*10**9)},quote_t=t,bid=bid.astype(np.int64),
        ask=(bid+100_000_000).astype(np.int64),valid=np.ones(8,bool))
    enter=dict(quote_index=1,t_ns=str(t[1]),kind=1,side=1,quantity=1,price=100.1,cash=1499.7,equity=1499.6)
    close=dict(quote_index=5,t_ns=str(t[5]),kind=5,side=1,quantity=1,price=101.,cash=1500.6,equity=1500.6)
    return d,enter,close

def test_intraminute_extrema_include_spike_and_drawdown():
    d,a,b=fixture();r=envelope(d,[a,b],1500.)
    first=r['rows'][0]
    assert first['equity_max']==pytest.approx(1509.6)
    assert first['equity_min']==pytest.approx(1489.6)
    assert first['equity_close']==pytest.approx(1500.6)
    assert first['balance_min']==pytest.approx(1499.7)
    assert first['balance_max']==pytest.approx(1500.6)
    assert r['rows'][1]['equity_min']==pytest.approx(1500.6)

def test_censored_tail_is_not_filled():
    d,a,b=fixture();d.valid[5]=False;b['kind']=7;b['price']=0
    r=envelope(d,[a,b],1500.)
    assert len(r['rows'])==1 and not r['rows'][0]['complete']
    assert r['summary']['censored'] and r['summary']['balance_close'] is None

def test_wait_balance_and_equity_identical():
    d,_,_=fixture();r=envelope(d,[],1500.)
    assert all(x['equity_min']==x['equity_max']==x['balance_close']==1500 for x in r['rows'])

def test_invalid_held_book_is_not_silently_valued():
    d,a,b=fixture();d.valid[3]=False
    with pytest.raises(ValueError):envelope(d,[a,b],1500.)


def test_add_keeps_weighted_cost_and_pre_fill_equity():
    d,a,b=fixture()
    add=dict(quote_index=2,t_ns=str(d.quote_t[2]),kind=2,side=1,quantity=1.5,
             price=103.1,cash=1499.25,equity=1502.)
    b.update(quantity=2.5,cash=1497.,equity=1497.)
    r=envelope(d,[a,add,b],1500.)
    assert r['summary']['equity_max']==pytest.approx(1519.5)
    assert r['summary']['equity_min']==pytest.approx(1469.5)
    assert r['summary']['balance_close']==pytest.approx(1497.)


def test_short_liquidation_uses_ask():
    d,a,b=fixture();a.update(side=-1,price=100.)
    b.update(side=-1,price=101.1,cash=1498.6,equity=1498.6)
    r=envelope(d,[a,b],1500.)
    assert r['summary']['equity_max']==pytest.approx(1509.6)
    assert r['summary']['equity_min']==pytest.approx(1489.6)
    assert r['summary']['balance_close']==pytest.approx(1498.6)


def test_end_of_tape_censor_retains_last_observed_quote_not_unknown_tail():
    d,a,_=fixture();r=envelope(d,[a],1500.,terminal_censored=True)
    assert r['summary']['censored']
    assert r['summary']['balance_close'] is None
    assert int(r['summary']['observed_until_ns'])==int(d.quote_t[-1])+1
    assert r['summary']['equity_min']==pytest.approx(1489.6)
    assert r['rows'][-1]['equity_close']==pytest.approx(1500.6)
    assert not r['rows'][-1]['complete']
