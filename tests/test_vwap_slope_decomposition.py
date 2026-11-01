import numpy as np
import pandas as pd
import pytest
from experiments.continuation_lab_v1.anchor_diagnostics import decompose,MINUTE


def minutes(prices,volume):
    prices=np.asarray(prices,float);volume=np.asarray(volume,float)
    t=np.arange(1,len(prices)+1,dtype=np.int64)*MINUTE
    return pd.DataFrame(dict(t_ns=t,available_ns=t+1,known=True,interval_open=True,epoch=1,
                             pv=(prices-4000)*volume,volume=volume))


def test_removing_old_low_prices_can_raise_average_without_new_price_impulse():
    f=minutes([4000,4002,4002,4002],[1,1,1,1])
    d=decompose(f,int(f.available_ns.iloc[-1])+1,3,1)
    assert d['known'] and d['total']==pytest.approx(2/3)
    assert d['outgoing']==pytest.approx(4/9)
    assert d['incoming']==pytest.approx(2/9)
    assert d['outgoing']>d['incoming']


def test_weighted_identity_and_strict_publication():
    f=minutes([4001,4002,4004,4000,4005],[2,7,1,5,3]);clock=int(f.available_ns.iloc[-1])
    d=decompose(f,clock,3,1)
    assert d==decompose(f.iloc[:-1],clock,3,1)
    d=decompose(f,clock+1,3,1)
    assert d['total']==pytest.approx(d['incoming']+d['outgoing'])
    f.loc[2,'known']=False
    assert not decompose(f,clock+1,3,1)['known']


def test_constant_price_gives_zero_components_even_with_unequal_volume():
    f=minutes([4002]*5,[1,3,5,7,11])
    d=decompose(f,int(f.available_ns.iloc[-1])+1,3,2)
    assert d['total']==d['incoming']==d['outgoing']==0
