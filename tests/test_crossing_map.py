import numpy as np
from spectrum.cognition.crossing_map import vwap_geometry,lookup_line,first_passage,factorized_loss,NS


def test_geometry_identity_does_not_depend_on_loss(monkeypatch):
    from spectrum.cognition import crossing_map as module
    parts=[dict(quote_file=dict(sha256='raw'),start_ns=1,end_ns=2)]
    before=module.geometry_id('VWAP',5,parts,'trades')
    monkeypatch.setattr(module,'factorized_loss',lambda *args:999.)
    assert module.geometry_id('VWAP',5,parts,'trades')==before
    assert module.geometry_id('VWAP',6,parts,'trades')!=before


def test_same_timestamp_future_publication_does_not_enter_vwap():
    t=np.array([60,60,60])*NS;k=np.arange(3);p=np.array([100.,110.,120.]);q=np.ones(3)
    m,h,s=vwap_geometry(t,k,k,t,p,q,1,np.zeros(3,dtype=int))
    assert m[1]==105 and h[1]==10
    mt,ht,_=vwap_geometry(t[:2],k[:2],k[:2],t[:2],p[:2],q[:2],1,np.zeros(2,dtype=int))
    np.testing.assert_allclose(m[:2],mt,equal_nan=True);np.testing.assert_allclose(h[:2],ht,equal_nan=True)


def test_line_lookup_is_strict_by_ordinal_and_censored_by_support():
    t=np.array([0,1,1,2,3])*NS
    a=lookup_line(np.array([2,4]),np.array([1,2,4]),np.ones(3,bool),np.array([4,3,5])*NS,t)
    np.testing.assert_array_equal(a,[2,-1,-1])


def test_first_hit_order_blocks_hindsight_reverse_success():
    t=np.arange(5)*NS;bid=(np.array([100,102,105,98,95])*1e9).astype(np.int64);ask=bid+700_000_000
    hits,stops=first_passage(t,bid,ask,np.arange(5),np.array([0]),np.array([5])*NS,np.array([2.]))
    assert hits[0,0,2]==2 and stops[0,1]==1
    assert (hits[0,1]==-1).all()


def test_factorization_gradient_masks_and_joint_equivalence():
    z=np.array([.2,-.5,.7]);d=np.array([[.1,.2,.3],[.3,.1,-.5],[.1,.3,.2]])
    y=np.array([0.,1.,1.]);direction=np.array([-1,1,2]);known=np.ones(3,bool);dk=np.array([False,True,False])
    loss,ge,gd=factorized_loss(z,d,y,direction,known,dk)
    for a,g in [(z,ge),(d,gd)]:
        for ix in np.ndindex(a.shape):
            old=a[ix];a[ix]=old+1e-6;plus=factorized_loss(z,d,y,direction,known,dk)[0]
            a[ix]=old-1e-6;minus=factorized_loss(z,d,y,direction,known,dk)[0];a[ix]=old
            np.testing.assert_allclose((plus-minus)/2e-6,g[ix],atol=1e-8)
    assert (gd[[0,2]]==0).all()
    dk[2]=True
    p=1/(1+np.exp(-z));pd=np.exp(d-d.max(1,keepdims=True));pd/=pd.sum(1,keepdims=True)
    joint=np.c_[1-p,p[:,None]*pd]
    expected=-np.log(joint[np.arange(3),[0,2,3]]).mean()
    np.testing.assert_allclose(factorized_loss(z,d,y,direction,known,dk)[0],expected)
