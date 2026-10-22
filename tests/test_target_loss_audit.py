import numpy as np
import pytest
from sklearn._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
from spectrum.cognition.target_loss_audit import loss_gradient_hessian, head_targets, consecutive_pairs, lag_record, run_lengths


@pytest.mark.parametrize('classes',[2,3])
def test_native_loss_derivatives_and_finite_difference(classes):
    rng=np.random.default_rng(71); n=17
    z=rng.normal(size=(n,1 if classes==2 else classes))*3; y=rng.integers(classes,size=n)
    loss=HalfBinomialLoss() if classes==2 else HalfMultinomialLoss(n_classes=classes)
    ll,g,h=loss_gradient_hessian(y,z)
    ng,nh=loss.gradient_hessian(y_true=y.astype(float),raw_prediction=z)
    np.testing.assert_allclose(g,np.asarray(ng).reshape(g.shape),atol=1e-14)
    np.testing.assert_allclose(h,np.asarray(nh).reshape(h.shape),atol=1e-14)
    np.testing.assert_allclose(ll,loss.loss(y.astype(float),z),atol=1e-14)
    for j in range(z.shape[1]):
        dz=np.zeros_like(z);dz[:,j]=1e-5
        hi=loss_gradient_hessian(y,z+dz)[0];lo=loss_gradient_hessian(y,z-dz)[0]
        np.testing.assert_allclose((hi-lo)/2e-5,g if classes==2 else g[:,j],atol=1e-9)
    if classes==3: np.testing.assert_allclose(g.sum(axis=1),0,atol=1e-15)


def test_unknown_and_empty_do_not_become_conditional_negatives():
    cats=np.array([-1,0,1,2,3]); m,y=head_targets(cats,'fill')
    assert m.tolist()==[False,True,True,True,True] and y.tolist()==[1,1,1,0]
    m,y=head_targets(cats,'path')
    assert m.tolist()==[False,True,True,True,False] and y.tolist()==[0,1,2]


def test_lag_never_compresses_gaps_resets_or_invalid_interiors():
    t=np.array([0,30,60,120,150,180,210])*10**9
    seg=np.array([1,1,1,1,1,2,2]); valid=np.array([True,False,True,True,True,True,True])
    a,b=consecutive_pairs(t,valid,seg,30,30)
    assert list(zip(a,b))==[(3,4),(5,6)]
    a,b=consecutive_pairs(t,valid,seg,30,60)
    assert len(a)==0


def test_missing_path_label_is_not_a_false_transition():
    t=np.arange(4)*30*10**9
    r=lag_record(t,[0,np.nan,1,1],np.ones(4),np.ones(4,bool),30,30)
    assert r['n']==1 and r['changed']==0
    assert run_lengths(t,[0,np.nan,1,1],np.ones(4),np.ones(4,bool),30)==[(0,1),(1,2)]
    filled,_=head_targets(np.array([0,3,1,1]),'path')
    a,b=consecutive_pairs(t,filled,np.ones(4),30,60)
    assert len(a)==0  # the EMPTY between two fills breaks conditional continuity
