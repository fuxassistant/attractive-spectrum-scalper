import numpy as np
import torch
from spectrum.player_v2.policy import HierarchicalPlayer,distribution,distribution_numpy,greedy
from spectrum.player_v2.experiment import optimize
from spectrum.player.contract import Contract


def test_hierarchy_gate_not_diluted_by_six_children():
    logits=np.zeros(10,np.float32);logits[1]=np.log(1.5) # participation .6
    joint,raw,g,d,p=distribution_numpy(logits,np.ones(7,bool))
    assert np.argmax(joint)==0 # old flat argmax would incorrectly wait
    assert greedy(g,d,p)==1
    np.testing.assert_allclose(joint[1:].sum(),.6,atol=1e-6)
    assert greedy(.5,d,p)==0


def test_numpy_torch_product_masks_agree():
    rng=np.random.default_rng(6)
    for _ in range(30):
        l=rng.normal(size=10).astype(np.float32);m=rng.random(7)>.5;m[0]=True
        a=distribution_numpy(l,m);b=distribution(torch.tensor(l),torch.tensor(m))
        for x,y in zip(a,b):np.testing.assert_allclose(x,y.numpy(),atol=2e-7)
        assert abs(a[0].sum()-1)<1e-6
        assert not a[0][~m].any()
        assert m[greedy(a[2],a[3],a[4])]


def test_forced_wait_finite_and_causal_prefix():
    torch.manual_seed(3);model=HierarchicalPlayer();x=torch.randn(1,30,71);account=torch.randn(1,30,7)
    logits,v=model(x,account);short,_=model(x[:,:10],account[:,:10])
    torch.testing.assert_close(logits[:,:10],short,rtol=1e-5,atol=1e-6)
    mask=torch.zeros(30,7,dtype=torch.bool);mask[:,0]=True
    joint,raw,g,d,p=distribution(logits[0],mask)
    assert torch.isfinite(joint).all() and (joint[:,0]==1).all() and (g==0).all()
    (logits.square().mean()+v.square().mean()).backward();assert model.lstm.weight_ih_l0.grad.norm()>0


def test_joint_logprob_factorization():
    logits=torch.randn(12,10,requires_grad=True);m=torch.ones(12,7,dtype=torch.bool)
    joint,raw,g,d,p=distribution(logits,m)
    torch.testing.assert_close(joint[:,5].log(),g.log()+d[:,1].log()+p[:,1,1].log())
    (-joint[:,5].log().mean()).backward()
    assert all(logits.grad[:,a:b].abs().sum()>0 for a,b in [(0,2),(2,4),(4,10)])


def test_empty_or_censored_training_is_safe_no_update():
    m=HierarchicalPlayer();opt=torch.optim.AdamW(m.parameters());before={n:v.clone() for n,v in m.state_dict().items()}
    b={'advantage':np.zeros(3,np.float32),'mask':np.array([[True]+[False]*6]*3),'action':np.zeros(3,np.int64)}
    for rs in [[],[(b,{'complete':False})],[(b,{'complete':True})]]:
        assert not optimize(m,opt,rs,Contract())['updated']
    for n,v in m.state_dict().items():torch.testing.assert_close(v,before[n])
