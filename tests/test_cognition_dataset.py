from types import SimpleNamespace
import numpy as np
import pandas as pd
from spectrum.cognition.dataset import market_features, HISTORY_NAMES
from spectrum.cognition.learn import FeatureScaler, sample_weights
from spectrum.player.contract import NS


def day():
    n=80;t=np.arange(1,n+1,dtype=np.int64)*60*NS
    x=np.zeros((n,71));x[:,1]=2;x[:,3]=1;x[:,8]=1
    return SimpleNamespace(x=x,bar_t=t-60*NS,decision_t=t,quote_t=t,
        cuts=np.arange(1,n+1),bid=np.full(n,100_000_000_000,np.int64),
        ask=np.full(n,100_100_000_000,np.int64),valid=np.ones(n,bool),epoch=np.ones(n,int),
        decision_mid=np.full(n,100.05))


def test_completed_available_prefix_not_later_row():
    d=day();a,i=market_features(d,int(d.decision_t[65])+10*NS)
    d.x[66:]=99999;d.bid[66:]=1;d.ask[66:]=999_000_000_000
    b,j=market_features(d,int(d.decision_t[65])+10*NS)
    assert i==j==65
    np.testing.assert_array_equal(a,b)


def test_return_history_requires_predecessor_support():
    d=day();d.x[60,8]=0
    x,_=market_features(d,int(d.decision_t[65]))
    assert x[71+HISTORY_NAMES.index('history_5m_known')]==0
    assert x[71+HISTORY_NAMES.index('return_5m_ticks')]==0


def test_quote_contrast_unknown_reference_is_zero_and_masked():
    d=day();d.valid[65]=False
    x,_=market_features(d,int(d.decision_t[65]))
    assert x[71+HISTORY_NAMES.index('quote_move_since_m1_ticks_known')]==0
    assert x[71+HISTORY_NAMES.index('quote_move_since_m1_ticks')]==0


def test_root_and_day_balancing_not_row_count():
    f=pd.DataFrame(dict(date=['a']*4+['b']*2,root_id=['r1']*3+['r2','r3','r3']))
    w=sample_weights(f)
    assert np.isclose(w[:4].sum(),w[4:].sum())
    assert np.isclose(w[:3].sum(),w[3])


def test_feature_scaler_train_only_masked_values_do_not_pollute():
    names=['flow','flow__known','position_side']
    x=np.array([[2,1,-1],[4,1,1],[1000,0,-1]],float)
    s=FeatureScaler().fit(x,names,np.ones(3))
    assert s.mean[0]==3 and s.scale[0]==1
    z=s.transform(x)
    np.testing.assert_array_equal(z[:,1],[1,1,0])
    assert z[2,0]==0
    saved=s.dict();s.transform(np.ones((3,3))*1e9)
    assert s.dict()==saved


def test_saved_case_preserves_json_safe_decisions_and_delayed_close():
    import json
    from spectrum.cognition.evaluate import replay_option
    from spectrum.cognition.execution import ExecutionReplay
    from pathlib import Path
    d=day();qt=np.arange(0,3600,2,dtype=np.int64)*100_000_000
    d.quote_t=qt;d.bid=np.full(len(qt),100_000_000_000,np.int64)
    d.ask=d.bid+100_000_000;d.valid=np.ones(len(qt),bool);d.epoch=np.ones(len(qt),np.int32)
    d.cuts=np.searchsorted(qt,d.decision_t,side='right')
    d.meta=dict(end_ns=str(400*NS),london_ns='0',ny_close_ns=str(390*NS))
    d.deadline=380*NS;d.path=Path('synthetic_case')
    initial=ExecutionReplay(d);initial.open_at(60*NS,1,0);initial.advance_until(63*NS)
    root=dict(intent_ns=60*NS,entry_ns=initial.state.entry_ns,entry_price_nano=initial.state.entry_price_nano,
        side=1,plan_id=0,horizon_ns=184*NS,root_id='synthetic',date='synthetic',reference={'net':-.4})
    result,_,_,picture=replay_option(d,root,'close_first',trace=True)
    json.dumps(picture,allow_nan=False)
    assert picture['decisions'][0]['action']=='CLOSE'
    assert result['exit_ns']>root['entry_ns']+5*NS+100_000_000
    assert abs(result['net']+.4)<1e-8
