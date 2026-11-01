"""Named distance blocks, physical gates and explicit absence of an acceptable match."""
import numpy as np

BLOCKS=['shape','amplitude','texture','structure','rhythm','trajectory','mechanism']


def huber(x):
    a=np.abs(x)
    return np.where(a<=1,.5*a*a,a-.5)


def joined_curve(pack):
    """Restore cumulative location across boundaries; local anchors are not three resets of the market."""
    a=pack['amplitude'];net=a[...,0]
    offsets=np.cumsum(net,axis=-1)-net
    raw=pack['shape']*np.maximum(.5,a[...,1])[...,None]+offsets[...,None]
    low=np.min(offsets-a[...,3],axis=-1)
    high=np.max(offsets+a[...,2],axis=-1)
    span=np.maximum(.5,high-low)
    return raw.reshape(raw.shape[:-2]+(raw.shape[-2]*raw.shape[-1],))/span[...,None]


def compare(query,candidates,*,duration_floor=.25,direction_floor=.5):
    """Query arrays [3, features], candidates [N, 3, features]. No fitted scales."""
    out={}
    for name in ('shape','amplitude','texture'):
        delta=candidates[name]-query[name]
        out[name]=np.mean(huber(delta),axis=(1,2))
        out[name+'_rmse']=np.sqrt(np.mean(delta**2,axis=(1,2)))
    # All three direction states and only the two already completed end reasons.
    direction=np.mean(candidates['structure'][...,0]!=query['structure'][...,0],axis=1)
    ending=np.mean(candidates['structure'][:,:2,1:]!=query['structure'][:2,1:],axis=(1,2))
    out['structure']=.5*(direction+ending)
    delta=joined_curve(candidates)-joined_curve(query)
    out['trajectory']=np.mean(huber(delta),axis=1)
    out['trajectory_rmse']=np.sqrt(np.mean(delta**2,axis=1))
    qnet=query['amplitude'][...,0];cnet=candidates['amplitude'][...,0]
    out['query_total_net']=np.full(len(cnet),float(qnet.sum()))
    out['candidate_total_net']=cnet.sum(axis=1)
    out['net_direction_conflict']=((abs(qnet.sum())>=direction_floor)&(abs(cnet.sum(axis=1))>=direction_floor)&(qnet.sum()*cnet.sum(axis=1)<0)).astype(float)
    out['slot_direction_conflict']=np.any((abs(qnet)>=direction_floor)&(abs(cnet)>=direction_floor)&(qnet*cnet<0),axis=1).astype(float)
    qd=np.maximum(duration_floor,query['duration']);cd=np.maximum(duration_floor,candidates['duration'])
    ratio=cd/qd
    out['rhythm']=np.mean(np.log(ratio)**2,axis=1)
    total=candidates['duration'].sum(axis=1)/max(duration_floor,float(query['duration'].sum()))
    out['total_duration_ratio']=total
    out['total_duration_factor']=np.maximum(total,1/np.maximum(total,1e-12))
    out['slot_duration_factor']=np.max(np.maximum(ratio,1/ratio),axis=1)
    common=candidates['book_known']&query['book_known']
    delta=candidates['book']-query['book']
    count=common.sum(axis=(1,2))
    out['book_common_fraction']=common.mean(axis=(1,2))
    out['book_min_slot_fraction']=common.mean(axis=2).min(axis=1)
    out['mechanism']=np.divide((huber(delta)*common).sum(axis=(1,2)),count,
        out=np.full(len(count),np.nan),where=count>0)
    out['mechanism_rmse']=np.sqrt(np.divide((delta**2*common).sum(axis=(1,2)),count,
        out=np.full(len(count),np.nan),where=count>0))
    # Missing evidence is never a zero-distance advantage.
    out['mechanism'] += .1*(1-out['book_common_fraction'])
    return out


def score_components(parts, method, config):
    count=len(parts['shape']); eligible=np.ones(count,bool)
    if method['name']=='legacy':
        return parts['legacy'].copy(),eligible,eligible.copy()
    weights=method['weights']
    if any(v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Nonnegative block weights must sum to one')
    score=np.zeros(count)
    for name,weight in weights.items():
        if weight: score+=weight*parts[name]
    for key,col in [('total_duration_factor','total_duration_factor'),('slot_duration_factor','slot_duration_factor')]:
        if method.get(key) is not None: eligible &= parts[col]<=method[key]+1e-12
    if method.get('mechanism_required'):
        eligible &= (parts['book_common_fraction']>=config['mechanism_min_common_fraction'])
        eligible &= (parts['book_min_slot_fraction']>=config['mechanism_min_slot_fraction'])
    if method.get('trajectory_required'):
        eligible &= (parts['net_direction_conflict']==0)&(parts['slot_direction_conflict']==0)
    limits=config['acceptance']
    accepted=eligible&np.isfinite(score)&(score<=limits['max_score'])
    accepted &= (parts['shape_rmse']<=limits['max_shape_rmse'])&(parts['amplitude_rmse']<=limits['max_amplitude_rmse'])
    if method.get('mechanism_required'): accepted &= parts['mechanism_rmse']<=limits['max_mechanism_rmse']
    if method.get('trajectory_required'): accepted &= parts['trajectory_rmse']<=limits['max_trajectory_rmse']
    return score,eligible,accepted
