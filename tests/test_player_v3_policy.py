import copy
from types import SimpleNamespace
import numpy as np
import torch
from spectrum.player_v3.policy import MixedPlayer,Standardizer


class FakePrepared:
    def __init__(self):
        rng=np.random.default_rng(42)
        self.slow=rng.normal(size=(6,71)).astype('f')
        self.anchors=rng.normal(size=(2,38)).astype('f')
        self.fast=rng.normal(size=(2,12,64)).astype('f')
        self.events=rng.normal(size=(2,4,28)).astype('f')
        self.day=SimpleNamespace(x=np.zeros((2,71)),slow_session=np.array([0,0,0,1,1,1]),slow_cuts=np.array([4,5]))
    def batch(self,ix,device):
        cv=lambda x:torch.as_tensor(x,device=device)
        return cv(self.fast[ix]),cv(np.full(len(ix),12)),cv(self.events[ix]),cv(np.ones((len(ix),4),bool)),np.ones((len(ix),3))


def test_future_m1_never_changes_earlier_encoding():
    torch.set_num_threads(1);torch.manual_seed(1)
    model=MixedPlayer().eval();p=FakePrepared();q=copy.deepcopy(p)
    q.slow[4:]=999.
    with torch.no_grad():
        a=model.encode_day(p,[0]);b=model.encode_day(q,[0])
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_disabled_events_cannot_change_router_or_output():
    torch.manual_seed(2);model=MixedPlayer().eval();p=FakePrepared();q=copy.deepcopy(p);q.events[:]=999
    with torch.no_grad():
        a,da=model.encode_day(p,[0],ablation='events',diagnostics=True)
        b,db=model.encode_day(q,[0],ablation='events',diagnostics=True)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    np.testing.assert_array_equal(da['branch_weights'],db['branch_weights'])
    assert da['branch_weights'][0,3]==0


def test_local_capacity_control_cannot_see_previous_or_anchors():
    torch.manual_seed(3);model=MixedPlayer(variant='mixed_local').eval();p=FakePrepared();q=copy.deepcopy(p)
    q.slow[:3]=999;q.events[:]=999;q.anchors[:]=999
    with torch.no_grad():
        a=model.encode_day(p,[0]);b=model.encode_day(q,[0])
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_missing_value_stays_zero_after_normalization():
    s=Standardizer();s.stats={'fast':dict(center=np.array([20,0]),scale=np.array([5,1]),names=['volume','volume__known'])}
    np.testing.assert_array_equal(s.transform('fast',np.array([[0,0],[25,1]])),[[0,0],[1,1]])


def test_explicit_yesterday_read_survives_long_current_prefix():
    torch.set_num_threads(1);torch.manual_seed(4);model=MixedPlayer().eval();p=FakePrepared()
    p.slow=np.zeros((600,71),np.float32);p.day.slow_session=np.r_[np.zeros(60),np.ones(540)]
    p.day.slow_cuts=np.array([600,600]);q=copy.deepcopy(p);q.slow[:60,:40]=5.
    with torch.no_grad():
        a,diag=model.encode_day(p,[0],diagnostics=True);b=model.encode_day(q,[0])
    assert float((a-b).abs().max())>1e-5
    assert 0<float(diag['counts'][0,3])<1
