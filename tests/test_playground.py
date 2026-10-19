from types import SimpleNamespace
import numpy as np
import pytest
from spectrum.player.contract import Contract, NS
from spectrum.player_v3.gym import SurvivalGym
from spectrum.playground.runner import contract, days_for
from spectrum.playground.server import wire


def synthetic(last_bid):
    t=np.array([0,1,2,60],np.int64)*NS
    d=SimpleNamespace(quote_t=t,source_t=t.copy(),bid=np.rint(np.array([1999.9,1999.9,last_bid,last_bid])*1e9).astype(np.int64),
        ask=np.rint(np.array([2000.,2000.,last_bid+.1,last_bid+.1])*1e9).astype(np.int64),valid=np.ones(4,bool),epoch=np.zeros(4,np.int64),
        decision_t=np.array([0,60],np.int64)*NS,cuts=np.array([1,4]),x=np.zeros((2,71),np.float32),can_enter=np.ones(2,bool),
        deadline=1000*NS,bar_t=np.array([0,60],np.int64)*NS,meta=dict(date='test',ny_close_ns=str(1000*NS),support_contract='FINALIZED_SEGMENTS_V2'))
    return d


def test_playground_starts_at_1500():
    e=SurvivalGym(synthetic(1250),contract())
    assert e.f[0]==1500 and e.initial_state.initial_capital==1500
    assert contract().size_oz==1 and contract().fee_roundtrip==.3


@pytest.mark.parametrize('last_bid,ruined',[(1250.,False),(1249.9,True)])
def test_series_strict_half_of_1500(last_bid,ruined):
    e=SurvivalGym(synthetic(last_bid),Contract(initial_capital=1500.,fee_roundtrip=0,daily_drawdown_stop=float('inf')))
    e.step(3)
    assert bool(e.s[12]) is ruined
    assert e.trades[0]['series_start_equity']==1500
    assert e.trades[0]['series_floor']==750


def test_ui_preserves_exact_nanosecond_integers():
    assert wire(dict(t=1787000000000000001,n=1500))==dict(t='1787000000000000001',n=1500)
