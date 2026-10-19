"""Mixed-resolution causal policy. Encoders are recomputed after every update."""
import numpy as np
import torch
from torch import nn
from spectrum.player_v2.policy import HierarchicalPlayer
from spectrum.player_v3.gym import SurvivalGym


class Standardizer:
    """Train-only moments; literal masks and masked missing values are preserved."""
    def fit(self, days):
        self.stats = {}
        names = {'base': days[0].meta['feature_names'], 'fast': days[0].meta['fast_names'],
                 'slow': days[0].meta['slow_names'], 'anchors': days[0].meta['anchor_names'],
                 'events': days[0].meta['event_names']}
        for key in names:
            if key == 'events':
                arrays = [d.memory(i)['events'] for d in days for i in range(0,len(d.x),60)]
            else:
                attr = {'base':'x','fast':'fast_x','slow':'slow_x','anchors':'anchors'}[key]
                arrays = [np.asarray(getattr(d,attr))[::10 if key=='fast' else 1] for d in days]
            x = np.concatenate([a for a in arrays if len(a)],axis=0).astype(np.float64)
            center = np.nanmean(x,axis=0); scale = np.maximum(np.nanstd(x,axis=0),1e-4)
            literal = [j for j,n in enumerate(names[key]) if n.endswith('__known') or n.endswith('_known')
                       or n in ('fresh_impulse','second_in_cluster','recurrent_irregular','recurrent_regular',
                                'price_led','volume_led','joint','candle_known','previous_same_contract')]
            center[literal]=0.;scale[literal]=1.
            self.stats[key] = dict(center=center.astype(np.float32),scale=scale.astype(np.float32),names=names[key])
        return self

    def transform(self,key,x):
        s=self.stats[key]; x=np.asarray(x,dtype=np.float32)
        y=np.clip((x-s['center'])/s['scale'],-10,10)
        for j,n in enumerate(s['names']):
            if n+'__known' in s['names']:
                y[...,j]=np.where(x[...,s['names'].index(n+'__known')]>0,y[...,j],0.)
        return np.nan_to_num(y).astype(np.float32)

    def save(self,path):
        from spectrum.ingest import save_json
        save_json(path,{k:{'center':v['center'].tolist(),'scale':v['scale'].tolist(),'names':v['names']} for k,v in self.stats.items()})

    @classmethod
    def load(cls,path):
        import json
        s=cls();s.stats=json.loads(path.read_text())
        for v in s.stats.values():
            v['center']=np.asarray(v['center'],np.float32);v['scale']=np.asarray(v['scale'],np.float32)
        return s


class PreparedDay:
    def __init__(self,day,scaler):
        self.day=day;self.scaler=scaler
        self.base=scaler.transform('base',day.x)
        self.fast=scaler.transform('fast',day.fast_x)
        self.slow=scaler.transform('slow',day.slow_x)
        self.anchors=scaler.transform('anchors',day.anchors)

    def batch(self,indices,device):
        d=self.day; n=len(indices)
        fast=np.zeros((n,900,self.fast.shape[1]),np.float32);length=np.zeros(n,np.int64)
        events=np.zeros((n,256,len(d.meta['event_names'])),np.float32);known=np.zeros((n,256),bool)
        counts=[]
        for j,i in enumerate(indices):
            fl,fr,sl,sr,el,er=d.memory_bounds(int(i));length[j]=fr-fl
            fast[j,:fr-fl]=self.fast[fl:fr]
            e=self.scaler.transform('events',d.memory(int(i))['events'])
            events[j,:len(e)]=e;known[j,:len(e)]=True;counts.append((fr-fl,sr-sl,er-el))
        cv=lambda a:torch.from_numpy(a).to(device)
        return cv(fast),cv(length),cv(events),cv(known),np.asarray(counts)


class MixedPlayer(nn.Module):
    """LSTM fast900 seconds + causal M1 + explicit anchors + bounded event attention."""
    branches=('fast','slow','anchors','events')
    def __init__(self,fast_dim=64,slow_dim=71,anchor_dim=38,event_dim=28,variant='mixed_full'):
        super().__init__();self.variant=variant
        self.fast_project=nn.Sequential(nn.Linear(fast_dim,32),nn.Tanh())
        self.fast_lstm=nn.LSTM(32,32,batch_first=True)
        self.slow_project=nn.Sequential(nn.Linear(slow_dim+1,48),nn.Tanh())
        self.slow_lstm=nn.LSTM(48,48,batch_first=True)
        # Explicit retrieval prevents yesterday from being accessible only via
        # hundreds of subsequent recurrent transitions before London opens.
        self.minute_query=nn.Linear(80,32)
        self.minute_key=nn.Linear(48,32)
        self.minute_value=nn.Linear(48,48)
        self.minute_combine=nn.Linear(96,48)
        self.anchor_project=nn.Sequential(nn.Linear(anchor_dim,32),nn.Tanh())
        self.event_key=nn.Linear(event_dim,32);self.event_value=nn.Linear(event_dim,32)
        self.query=nn.Linear(80,32)
        self.branch_project=nn.ModuleList([nn.Linear(k,48) for k in (32,48,32,32)])
        self.branch_gate=nn.Linear(144,4)
        self.head=nn.Linear(48,10);self.account_head=nn.Linear(SurvivalGym.account_dim,10,bias=False)
        self.critic=nn.Linear(48,1);self.account_critic=nn.Linear(SurvivalGym.account_dim,1,bias=False)
        nn.init.orthogonal_(self.head.weight,gain=.01);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.account_head.weight)

    def encode_day(self,p,indices=None,ablation=None,diagnostics=False,batch_size=64):
        device=next(self.parameters()).device;d=p.day
        indices=np.arange(len(d.x)) if indices is None else np.asarray(indices)
        slow=p.slow; sessions=np.asarray(d.slow_session,dtype=np.float32)
        offset=0
        if self.variant=='mixed_local' or ablation=='previous':
            offset=int(np.searchsorted(sessions,1));slow=slow[offset:];sessions=sessions[offset:]
        sx=torch.from_numpy(np.c_[slow,sessions]).to(device)[None]
        sh=self.slow_lstm(self.slow_project(sx))[0][0]
        counts_slow=np.asarray(d.slow_cuts[indices])-offset
        if np.any(counts_slow<=0):raise ValueError('Current M1 must be available before policy evaluation')
        if len(slow)>2760:raise ValueError('Prepared two-session M1 memory exceeds declared bound')
        positions=counts_slow-1
        hs=sh[torch.from_numpy(positions).to(device)]
        minute_keys=self.minute_key(sh);minute_values=self.minute_value(sh)
        out=[];gates=[];attentions=[];counts=[]
        for lo in range(0,len(indices),batch_size):
            ix=indices[lo:lo+batch_size];s=hs[lo:lo+batch_size]
            fast,length,events,known,count=p.batch(ix,device)
            fh=self.fast_lstm(self.fast_project(fast))[0]
            f=fh[torch.arange(len(ix),device=device),(length-1).clamp(min=0)]
            f=f*(length>0)[:,None]
            if ablation=='fast':f=f*0
            # Store every completed M1 representation and read only its prefix.
            # Current and previous sessions are separately tagged in the LSTM input.
            ms=(self.minute_query(torch.cat([f,s],-1))@minute_keys.T)/(32**.5)
            allowed=torch.arange(len(sh),device=device)[None,:]<torch.as_tensor(counts_slow[lo:lo+len(ix)],device=device)[:,None]
            mw=ms.masked_fill(~allowed,-1e4).softmax(-1)
            s=torch.tanh(self.minute_combine(torch.cat([s,mw@minute_values],-1)))
            if ablation=='slow':s=s*0
            a=self.anchor_project(torch.from_numpy(p.anchors[ix]).to(device))
            score=(self.event_key(events)*self.query(torch.cat([f,s],-1))[:,None,:]).sum(-1)/(32**.5)
            weight=score.masked_fill(~known,-1e4).softmax(-1)*known
            weight=weight/weight.sum(-1,keepdim=True).clamp(min=1e-8)
            e=(weight[:,:,None]*self.event_value(events)).sum(1)
            parts=[f,s,a,e];enabled=torch.ones((len(ix),4),device=device,dtype=torch.bool)
            if self.variant=='mixed_local':enabled[:,2:]=False
            if ablation in self.branches:enabled[:,self.branches.index(ablation)]=False
            # A disabled branch cannot influence the router through its input.
            parts=[v*enabled[:,j,None] for j,v in enumerate(parts)]
            w=self.branch_gate(torch.cat(parts,-1)).masked_fill(~enabled,-1e4).softmax(-1)
            h=sum(w[:,j,None]*torch.tanh(layer(parts[j])) for j,layer in enumerate(self.branch_project))
            out.append(h)
            if diagnostics:
                previous_mass=(mw*torch.as_tensor(sessions==0,device=device)[None,:]).sum(-1)
                gates.append(w.detach().float().cpu().numpy());attentions.append(weight.detach().float().cpu().numpy());counts.append(np.c_[count,previous_mass.detach().float().cpu().numpy()])
        h=torch.cat(out)
        return (h,dict(branch_weights=np.concatenate(gates),event_attention=np.concatenate(attentions),counts=np.concatenate(counts))) if diagnostics else h


def make_model(variant):
    if variant!='baseline':return MixedPlayer(variant=variant)
    model=HierarchicalPlayer()
    # Retain the previous market encoder, with the SAME new risk observation
    # available to every comparison arm. Old fixed-floor weights are incompatible.
    model.account_head=nn.Linear(SurvivalGym.account_dim,10,bias=False)
    model.account_critic=nn.Linear(SurvivalGym.account_dim,1,bias=False)
    nn.init.zeros_(model.account_head.weight)
    return model


def encode(model,p,indices=None,ablation=None,diagnostics=False):
    if isinstance(model,MixedPlayer):return model.encode_day(p,indices,ablation,diagnostics)
    device=next(model.parameters()).device
    h=model.encode(torch.from_numpy(p.base).to(device)[None])[0]
    if indices is not None:h=h[torch.as_tensor(indices,device=device)]
    return (h,{}) if diagnostics else h
