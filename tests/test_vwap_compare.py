import numpy as np
from spectrum.cognition.vwap_compare import TradePrefix,weighted_native_closes,support_start,NS


def test_exact_weighted_population_and_empty():
    t=np.array([1,2,3,8])*NS;p=np.array([100.,102,105,999]);q=np.array([1,3,2,8])
    pref=TradePrefix(t,p,q)
    m,h,age,v=pref.query(np.array([1,4])*NS,np.array([3,7])*NS)
    mu=np.average(p[:3],weights=q[:3]);sd=np.sqrt(np.average((p[:3]-mu)**2,weights=q[:3]))
    np.testing.assert_allclose([m[0],h[0],age[0],v[0]],[mu,2*sd,3-np.average([1,2,3],weights=q[:3]),6])
    assert np.isnan(m[1]) and v[1]==0


def test_native_equal_volume_reduces_to_unweighted_population():
    qt=np.arange(25*60)*NS;p=100+np.arange(len(qt))/100
    tt=(np.arange(25)*60+20)*NS;pref=TradePrefix(tt,np.ones(25)*100,np.ones(25))
    query=np.array([20*60+30])*NS
    m,h,age,bbage=weighted_native_closes(qt,p,query,1,pref)
    samples=p[np.r_[np.arange(2,21)*60-1,1230]]
    np.testing.assert_allclose([m[0],h[0]],[samples.mean(),2*samples.std(ddof=0)])
    np.testing.assert_allclose(age,bbage)


def test_causal_prefix_and_boundary():
    t=np.arange(1,51)*NS;p=np.arange(50.)+5000;q=np.arange(1,51)
    a=TradePrefix(t,p,q).query(np.array([10])*NS,np.array([30])*NS)
    b=TradePrefix(t[:30],p[:30],q[:30]).query(np.array([10])*NS,np.array([30])*NS)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert a[3][0]==sum(q[9:30])


def test_support_breaks_are_not_bridged():
    t=np.array([0,1,2,3,100,101,102])*NS
    v=np.array([1,1,0,1,1,1,1],bool);e=np.array([0,0,0,0,0,1,1])
    np.testing.assert_array_equal(support_start(t,v,e)//NS,[0,0,2,3,100,101,101])


def test_sampled_first_crossing_tie_none_and_censor():
    import pandas as pd
    from scripts.vwap_week_report import targets
    n=2005
    f=pd.DataFrame(dict(t_ns=np.arange(n)*NS,mid=np.r_[np.zeros(10),np.full(n-10,4.)],
        bb=np.full(n,2.),bb_half=np.ones(n),segment=np.zeros(n,dtype=int)))
    y,delay,_=targets(f,'bb')
    assert y[0]==7 and delay[0]==10
    assert y[1]==8 and y[-1]==-1
    f.loc[5:,'segment']=1
    y,delay,_=targets(f,'bb')
    assert y[0]==-1 and np.isnan(delay[0])
