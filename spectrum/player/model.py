"""Causal recurrent encoder and account-conditioned joint action/plan policy."""
import numpy as np
import torch
from torch import nn


class Scaler:
    def fit(self,days):
        x=np.concatenate([np.asarray(d.x) for d in days])
        self.center=np.mean(x,axis=0,dtype=np.float64).astype(np.float32)
        self.scale=np.std(x,axis=0,dtype=np.float64).astype(np.float32)
        self.scale=np.maximum(self.scale,1e-4)
        # Flags preserve their literal semantics.
        for j in [8]+list(range(x.shape[1]-30,x.shape[1])):self.center[j]=0;self.scale[j]=1
        return self
    def transform(self,x):return np.clip((x-self.center)/self.scale,-10,10).astype(np.float32)
    def save(self,path):np.savez(path,center=self.center,scale=self.scale)
    @classmethod
    def load(cls,path):
        v=np.load(path);s=cls();s.center=v['center'];s.scale=v['scale'];return s


class Player(nn.Module):
    def __init__(self,input_dim,hidden=48):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(input_dim,64),nn.Tanh())
        self.lstm=nn.LSTM(64,hidden,batch_first=True)
        self.actor=nn.Linear(hidden,7);self.actor_account=nn.Linear(7,7,bias=False)
        self.critic=nn.Linear(hidden,1);self.critic_account=nn.Linear(7,1,bias=False)
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.actor_account.weight)

    def encode(self,x):return self.lstm(self.project(x))[0]
    def forward(self,x,account):
        h=self.encode(x)
        return self.actor(h)+self.actor_account(account),(self.critic(h)+self.critic_account(account)).squeeze(-1)


def advantages(rewards,values,gamma=1.,lam=.95):
    result=np.zeros(len(rewards),dtype=np.float32);carry=0.
    for i in range(len(rewards)-1,-1,-1):
        next_value=values[i+1] if i+1<len(rewards) else 0.
        carry=rewards[i]+gamma*next_value-values[i]+gamma*lam*carry;result[i]=carry
    return result,result+values
