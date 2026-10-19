"""Mixed-resolution causal policy. Encoders are recomputed after every update."""
import numpy as np
import torch
from torch import nn
from spectrum.player_v2.policy import HierarchicalPlayer
from spectrum.player_v3.gym import SurvivalGym


class Standardizer:
    """Train-only moments; literal masks and masked missing values are preserved.

    An absent event bank has schema-shaped identity moments, not fabricated
    observations. Counts describe sampled rows (event windows can overlap),
    rather than a number of distinct market events.
    """
    @staticmethod
    def _array(key, value, width):
        x = np.asarray(value)
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(f'{key}: expected shape (rows, {width}), got {x.shape}')
        if key == 'events' and not np.isfinite(x).all():
            raise ValueError('events: nonfinite causal event field')
        return x

    def fit(self, days):
        days = list(days)
        if not days:
            raise ValueError('Standardizer.fit requires at least one training day')
        self.stats = {}
        fields = {'base': 'feature_names', 'fast': 'fast_names', 'slow': 'slow_names',
                  'anchors': 'anchor_names', 'events': 'event_names'}
        names = {key: list(days[0].meta[field]) for key, field in fields.items()}
        for key in names:
            if not names[key] or len(set(names[key])) != len(names[key]):
                raise ValueError(f'{key}: feature names must be nonempty and unique')
            if any(list(d.meta[fields[key]]) != names[key] for d in days):
                raise ValueError(f'{key}: training-day feature schemas differ')
            if key == 'events':
                arrays = [d.memory(i)['events'] for d in days for i in range(0,len(d.x),60)]
            else:
                attr = {'base':'x','fast':'fast_x','slow':'slow_x','anchors':'anchors'}[key]
                arrays = [np.asarray(getattr(d,attr))[::10 if key=='fast' else 1] for d in days]
            arrays = [self._array(key, a, len(names[key])) for a in arrays]
            nonempty = [a for a in arrays if len(a)]
            sample_count = sum(len(a) for a in nonempty)
            if not nonempty:
                if key != 'events':
                    raise ValueError(f'{key}: no training observations')
                center = np.zeros(len(names[key]), dtype=np.float64)
                scale = np.ones(len(names[key]), dtype=np.float64)
                finite_counts = np.zeros(len(names[key]), dtype=np.int64)
            else:
                x = np.concatenate(nonempty, axis=0).astype(np.float64)
                finite_counts = np.isfinite(x).sum(axis=0)
                if not finite_counts.all() or np.isinf(x).any():
                    raise ValueError(f'{key}: no finite moments for one or more features')
                center = np.nanmean(x,axis=0); scale = np.maximum(np.nanstd(x,axis=0),1e-4)
            literal = [j for j,n in enumerate(names[key]) if n.endswith('__known') or n.endswith('_known')
                       or n in ('fresh_impulse','second_in_cluster','recurrent_irregular','recurrent_regular',
                                'price_led','volume_led','joint','candle_known','previous_same_contract')]
            center[literal]=0.;scale[literal]=1.
            self.stats[key] = dict(center=center.astype(np.float32),scale=scale.astype(np.float32),names=names[key],
                                   sample_count=sample_count, finite_counts=finite_counts.tolist(),
                                   status='observed' if sample_count else 'empty')
        return self

    def transform(self,key,x):
        s=self.stats[key]; x=np.asarray(x,dtype=np.float32)
        if x.ndim == 0 or x.shape[-1] != len(s['names']):
            raise ValueError(f"{key}: expected final dimension {len(s['names'])}, got {x.shape}")
        if key == 'events' and not np.isfinite(x).all():
            raise ValueError('events: nonfinite causal event field')
        y=np.clip((x-s['center'])/s['scale'],-10,10)
        for j,n in enumerate(s['names']):
            mask = next((m for m in (n+'__known', n+'_known') if m in s['names']), None)
            if mask is not None:
                y[...,j]=np.where(x[...,s['names'].index(mask)]>0,y[...,j],0.)
        return np.nan_to_num(y).astype(np.float32)

    def save(self,path):
        from spectrum.ingest import save_json
        save_json(path,{k:{**v,'center':v['center'].tolist(),'scale':v['scale'].tolist()} for k,v in self.stats.items()})

    @classmethod
    def load(cls,path):
        import json
        s=cls();s.stats=json.loads(path.read_text())
        for key,v in s.stats.items():
            v['center']=np.asarray(v['center'],np.float32);v['scale']=np.asarray(v['scale'],np.float32)
            width = len(v['names'])
            if (not width or len(set(v['names'])) != width or v['center'].shape != (width,)
                    or v['scale'].shape != (width,) or not np.isfinite(v['center']).all()
                    or not np.isfinite(v['scale']).all() or np.any(v['scale'] <= 0)):
                raise ValueError(f'{key}: invalid saved normalization contract')
            # Old files remain readable; their training support is unknown.
            v.setdefault('sample_count', None)
            v.setdefault('finite_counts', None)
            v.setdefault('status', 'legacy_unknown_support')
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
            # No event is an absent branch: learned projection biases must not
            # masquerade as evidence or receive router mass on an empty bank.
            enabled[:,3]=known.any(dim=1)
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
