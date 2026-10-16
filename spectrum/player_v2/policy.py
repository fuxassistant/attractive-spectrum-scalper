import numpy as np
import torch
from torch import nn


class HierarchicalPlayer(nn.Module):
    """P(WAIT)=1-g; P(direction,plan)=g*d(direction)*p(plan|direction)."""
    def __init__(self,input_dim=71,hidden=48):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(input_dim,64),nn.Tanh())
        self.lstm=nn.LSTM(64,hidden,batch_first=True)
        # 2 gate logits, 2 direction logits, 2 x 3 conditional plan logits.
        self.head=nn.Linear(hidden,10);self.account_head=nn.Linear(7,10,bias=False)
        self.critic=nn.Linear(hidden,1);self.account_critic=nn.Linear(7,1,bias=False)
        nn.init.orthogonal_(self.head.weight,gain=.01);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.account_head.weight)
    def encode(self,x):return self.lstm(self.project(x))[0]
    def forward(self,x,account):
        h=self.encode(x)
        return self.head(h)+self.account_head(account),(self.critic(h)+self.account_critic(account)).squeeze(-1)


def distribution(logits,mask):
    """Torch distribution with executable masks, finite even at forced WAIT."""
    raw_gate=logits[...,:2].softmax(-1)
    pm=mask[...,1:].reshape(*mask.shape[:-1],2,3)
    dm=pm.any(-1);allowed=dm.any(-1)
    safe_pm=pm|~dm.unsqueeze(-1)
    plans=logits[...,4:].reshape(*logits.shape[:-1],2,3).masked_fill(~safe_pm,-1e9).softmax(-1)
    direction=logits[...,2:4].masked_fill(~(dm|~allowed.unsqueeze(-1)),-1e9).softmax(-1)
    g=raw_gate[...,1]*allowed
    joint=torch.cat([(1-g).unsqueeze(-1),(g[...,None,None]*direction.unsqueeze(-1)*plans).flatten(-2)],-1)
    return joint,raw_gate[...,1],g,direction,plans


def distribution_numpy(logits,mask):
    def soft(x):
        y=np.exp(x-x.max(axis=-1,keepdims=True));return y/y.sum(axis=-1,keepdims=True)
    raw=float(soft(logits[:2])[1]);pm=mask[1:].reshape(2,3);dm=pm.any(-1);allowed=bool(dm.any())
    plans=soft(np.where(pm|~dm[:,None],logits[4:].reshape(2,3),-1e9))
    direction=soft(np.where(dm|~np.bool_(allowed),logits[2:4],-1e9))
    gate=raw if allowed else 0.
    joint=np.r_[1-gate,(gate*direction[:,None]*plans).ravel()]
    return joint,raw,gate,direction,plans


def greedy(gate,direction,plans):
    # Binary participation first. Number of downstream alternatives cannot veto entry.
    if gate<=.5:return 0
    d=int(np.argmax(direction));return 1+3*d+int(np.argmax(plans[d]))
