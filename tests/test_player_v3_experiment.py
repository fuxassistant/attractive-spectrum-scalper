"""Learning may use verified prefixes; support-censored outcomes are never targets."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from spectrum.player.contract import Contract, NS
from spectrum.player_v3 import experiment


class ConstantLong(torch.nn.Module):
    """Strong long/fast policy with a known, nonzero continuation estimate."""
    def __init__(self, value=2.):
        super().__init__()
        self.head=torch.nn.Linear(1,10)
        self.critic=torch.nn.Linear(1,1)
        self.account_head=torch.nn.Linear(experiment.SurvivalGym.account_dim,10,bias=False)
        self.account_critic=torch.nn.Linear(experiment.SurvivalGym.account_dim,1,bias=False)
        with torch.no_grad():
            for p in self.parameters():p.zero_()
            self.head.bias.copy_(torch.tensor([0.,20.,20.,0.,20.,0.,0.,20.,0.,0.]))
            self.critic.bias.fill_(value)


@pytest.fixture
def model(monkeypatch):
    def encode(model,prepared,indices=None,ablation=None,diagnostics=False):
        n=len(prepared.day.x) if indices is None else len(indices)
        h=torch.zeros((n,1),device=next(model.parameters()).device)
        return (h,{}) if diagnostics else h
    monkeypatch.setattr(experiment,'encode',encode)
    return ConstantLong()


def prepared(times,bids,*,decision=(0,60,120),valid=None,allowed=None,offset=0,date='2026-05-04'):
    t=(np.asarray(times,np.int64)+offset)*NS
    dt=(np.asarray(decision,np.int64)+offset)*NS
    bid=np.rint(np.asarray(bids)*1e9).astype(np.int64)
    d=SimpleNamespace(quote_t=t,source_t=t.copy(),bid=bid,ask=bid+100_000_000,
        valid=np.ones(len(t),bool) if valid is None else np.asarray(valid,bool),
        epoch=np.zeros(len(t),np.int64),decision_t=dt,cuts=np.searchsorted(t,dt,side='right'),
        x=np.zeros((len(dt),71),np.float32),can_enter=np.asarray(allowed,bool) if allowed is not None else np.ones(len(dt),bool),
        deadline=(offset+1000)*NS,meta={'date':date,'ny_close_ns':str((offset+1000)*NS)})
    return SimpleNamespace(day=d)


def successful_day(**kw):
    return prepared([0,1,2,60,61,62,120],[100,100,102,102,102,104,104],allowed=[1,1,0],**kw)


def censored_day(recovery=102.,**kw):
    # First completed winner; second trade loses support; a third apparent
    # opportunity is future information and must never enter this population.
    return prepared([0,1,2,60,61,62,63,120,121,122,180],
        [100,100,102,102,102,102,recovery,recovery,recovery,recovery+20,recovery+20],
        decision=(0,60,120,180),valid=[1,1,1,1,1,0,1,1,1,1,1],**kw)


def contract(**kw):
    return Contract(initial_capital=200.,support_gap_ns=2000*NS,gamma=1.,gae_lambda=1.,**kw)


def test_prefix_advantages_bootstraps_without_fake_terminal_zero():
    a,r=experiment.prefix_advantages(np.array([1.,2.]),np.array([10.,20.]),
                                     bootstrap=30.,gamma=.9,lam=.8)
    np.testing.assert_allclose(a,[15.48,9.],rtol=1e-6)
    np.testing.assert_allclose(r,[25.48,29.],rtol=1e-6)


def test_complete_population_reward_telescopes_and_gae_crosses_days(model):
    p=[successful_day(),successful_day(offset=86400,date='2026-05-05')]
    batches,metric=experiment.rollout(model,p,contract())
    assert metric['complete'] and metric['trades']==4
    assert metric['final_equity']==pytest.approx(206.4)
    rewards=np.concatenate([b['reward'] for b in batches])
    returns=np.concatenate([b['returns'] for b in batches])
    expected=np.cumsum(rewards[::-1])[::-1]
    np.testing.assert_allclose(returns,expected,atol=1e-5,rtol=1e-5)
    # The last forced-WAIT origin of day one still precedes day-two earnings.
    assert batches[0]['returns'][-1]==pytest.approx(sum(batches[1]['reward']),abs=1e-5)
    assert rewards.sum()/100==pytest.approx(metric['score'],abs=1e-6)
    assert metric['reward_score']==pytest.approx(metric['score'],abs=1e-6)


def test_censor_cuts_at_entry_and_future_recovery_cannot_change_training(model):
    first,ma=experiment.rollout(model,[censored_day(102.)],contract())
    second,mb=experiment.rollout(model,[censored_day(9000.)],contract())
    assert not ma['complete'] and ma['score'] is None
    assert not mb['complete'] and mb['score'] is None
    assert sum(len(b['action']) for b in first)==1
    assert sum(len(b['action']) for b in second)==1
    # Both the censored entry action and its downstream price path are absent.
    for name in ('account','mask','action','logp','value','reward','advantage','returns'):
        np.testing.assert_allclose(first[0][name],second[0][name],rtol=0,atol=0)
    b=first[0]
    assert b['returns'][0]==pytest.approx(b['reward'][0]+2.,abs=1e-6)
    assert b['advantage'][0]==pytest.approx(b['reward'][0],abs=1e-6)
    assert ma['days'][0]['trades']==2 # one certain and one censored, no third trade


def test_censor_next_day_bootstraps_last_clean_day_instead_of_discarding_it(model):
    bad=prepared([0,1,2,3,60],[100]*5,decision=(0,60),valid=[1,1,0,1,1],
                 offset=86400,date='2026-05-05')
    batches,metric=experiment.rollout(model,[successful_day(),bad],contract())
    assert metric['score'] is None and not metric['complete']
    kept=[b for b in batches if len(b['action'])]
    assert len(kept)==1 and len(kept[0]['action'])==3
    assert kept[0]['returns'][-1]==pytest.approx(2.,abs=1e-6)
    expected=np.cumsum(kept[0]['reward'][::-1])[::-1]+2.
    np.testing.assert_allclose(kept[0]['returns'],expected,atol=1e-5)


def test_open_terminal_trade_is_also_excluded_at_its_entry(model):
    p=prepared([0,1,2,60,61,120],[100,100,102,102,102,102],decision=(0,60,120),allowed=[1,1,0])
    batches,metric=experiment.rollout(model,[p],contract())
    assert not metric['complete'] and metric['score'] is None
    assert metric['days'][0]['open_terminal']
    assert sum(len(b['action']) for b in batches)==1
    b=next(b for b in batches if len(b['action']))
    assert b['returns'][0]==pytest.approx(b['reward'][0]+2.,abs=1e-6)


def test_first_trade_censor_produces_no_trainable_action(model):
    p=prepared([0,1,2,3,60],[100]*5,decision=(0,60),valid=[1,1,0,1,1])
    batches,metric=experiment.rollout(model,[p],contract())
    assert not metric['complete'] and metric['score'] is None
    assert sum(len(b['action']) for b in batches)==0
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    stats=experiment.optimize(model,optimizer,batches,contract())
    assert stats['updates']==0


def test_verified_prefix_of_incomplete_population_still_updates_critic(model):
    batches,metric=experiment.rollout(model,[censored_day()],contract())
    assert not metric['complete'] and metric['learning_prefix_steps']==1
    before=model.critic.bias.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    stats=experiment.optimize(model,optimizer,batches,contract(ppo_passes=1))
    assert stats['updates']==1 and np.isfinite(stats['loss'])
    assert not torch.equal(before,model.critic.bias.detach())
