"""Read a historical forming story at the query's age, without using its later end.

All starts are considered. A story is excluded only if its terminal update is
already known at the requested age. It need not survive a future heartbeat.
"""
import numpy as np
import pandas as pd
from numba import njit

from experiments.scalper_adapter_v1.tokens import NS,TERMINAL,quote_arrays
from experiments.scalper_adapter_v1.similarity import descriptors,ordered_distance
from .features import extract
from .search import FEATURE_KEYS
from .metric import compare,score_components


@njit(cache=True)
def prefix_stats(price,times,starts,ends,directions,start_times):
    result=np.zeros((len(starts),8),np.int64)
    for r in range(len(starts)):
        lo=hi=price[starts[r]];path=up=down=turns=sign=0
        extreme=price[starts[r]];extreme_ns=start_times[r]
        for j in range(starts[r]+1,ends[r]+1):
            p=price[j];delta=p-price[j-1]
            lo=min(lo,p);hi=max(hi,p);path+=abs(delta)
            up+=max(0,delta);down+=max(0,-delta)
            if delta:
                new_sign=1 if delta>0 else -1
                if sign and sign!=new_sign:turns+=1
                sign=new_sign
            if directions[r]*(p-extreme)>0:extreme=p;extreme_ns=times[j]
        result[r]=np.array([lo,hi,path,turns,up,down,extreme,extreme_ns],np.int64)
    return result


def prepare_aligned(search,token,scope='previous'):
    if scope not in ('previous','current'):raise ValueError('Explicit scope required')
    frame,query,old=search.query_context(token)
    age=int(token.available_ns)-int(token.start_ns)
    if age<=0:raise ValueError('A positive observed forming age is required')
    boundary=int(frame.start_ns.min());current=search.sessions[token.session_id]
    date=current.previous_session_id if scope=='previous' else current.session_id
    f=search.base.frame
    session=search.sessions.get(date)
    chosen=f[f.session_id.eq(date)&f.scale.eq(token.scale)].copy()
    clocks=chosen.start_ns.to_numpy(np.int64)+age
    valid=(chosen.available_ns.to_numpy(np.int64)<=clocks)&(clocks<boundary)
    if session is not None:valid&=(clocks<=session.observed_through_ns)&(clocks<session.end_ns)
    chosen=chosen[valid].drop_duplicates('story_id',keep='last')
    # Only information published by the historical observation clock can exclude
    # a candidate. No filtering on its full-history duration or terminal label.
    chosen=chosen[~chosen.phase.isin(TERMINAL)&search.base.mask[chosen.index].all(axis=1)].copy()
    ids=chosen.index.to_numpy(np.int64)
    parents=search.base.parents[ids]
    if len(chosen):
        qt,price,good,_=quote_arrays(session.quotes)
        clocks=chosen.start_ns.to_numpy(np.int64)+age
        ends=np.searchsorted(qt,clocks,side='right')-1
        assert (ends>=chosen.start_index.to_numpy(np.int64)).all()
        assert good[ends].all() and (clocks-qt[ends]<=30*NS).all()
        stats=prefix_stats(price,qt,chosen.start_index.to_numpy(np.int64),ends,
                           chosen.direction.to_numpy(np.int64),chosen.start_ns.to_numpy(np.int64))
        for i,key in enumerate(['low_nano','high_nano','path_nano','turns','up_nano','down_nano','extreme_nano','extreme_ns']):chosen[key]=stats[:,i]
        chosen['available_ns']=clocks;chosen['source_index']=ends
        chosen['source_ordinal']=session.quotes.source_ordinal.to_numpy(np.int64)[ends]
        chosen['mid_nano']=price[ends];chosen['elapsed_seconds']=age/NS
        chosen['kind']='LIVE_STATE';chosen['phase']='FORMING'
        chosen['token_id']=[str(story)+':asof:'+str(clock) for story,clock in zip(chosen.story_id,clocks)]
        live=extract(chosen,session.quotes,search.books[date],samples=search.config['shape_samples'])
        candidate={k:np.concatenate([v[parents],live[k][:,None]],axis=1) for k,v in search.features.items()}
        old_candidates=np.concatenate([search.base.vectors[parents],descriptors(chosen,session.quotes)[:,None]],axis=1)
    else:
        candidate={k:np.empty((0,)+v.shape,dtype=v.dtype) for k,v in query.items()}
        old_candidates=np.empty((0,3,old.shape[-1]))
    parts=compare(query,candidate,duration_floor=search.config['duration_floor_seconds'],
                  direction_floor=search.config['direction_floor_amplitudes'])
    parts['legacy']=ordered_distance(old,old_candidates,[True,True,True])
    return dict(token=token,query_frame=frame,query_pack=query,scope=scope,query_context_start_ns=boundary,
                virtual=chosen.reset_index(drop=True),parents=parents,parts=parts,age_ns=age)


def rank_aligned(search,prepared,method,neighbors=3):
    score,eligible,accepted=score_components(prepared['parts'],method,search.config)
    valid=np.flatnonzero(accepted&np.isfinite(score));valid=valid[np.argsort(score[valid],kind='stable')]
    result=[];intervals=[]
    for ix in valid:
        row=prepared['virtual'].iloc[ix];parents=search.base.frame.iloc[prepared['parents'][ix]]
        start=int(parents.start_ns.min());end=int(row.available_ns)
        if any(start<=b and a<=end for a,b in intervals):continue
        intervals.append((start,end))
        parts={k:float(v[ix]) if np.isfinite(v[ix]) else None for k,v in prepared['parts'].items()}
        result.append(dict(method=method['name'],scope=prepared['scope'],scale=row.scale,rank=len(result)+1,
            query_token_id=prepared['token'].token_id,query_ns=int(prepared['token'].available_ns),
            query_context_start_ns=prepared['query_context_start_ns'],token_id=row.token_id,
            story_id=row.story_id,session_id=row.session_id,context_start_ns=start,available_ns=end,
            context_token_ids=parents.token_id.tolist()+[row.token_id],score=float(score[ix]),
            age_ns=prepared['age_ns'],virtual_index=int(ix),weights=method['weights'],**parts))
        if len(result)>=neighbors:break
    known=prepared['query_pack']['book_known']
    insufficient=known.mean()<search.config['mechanism_min_common_fraction'] or known.mean(axis=1).min()<search.config['mechanism_min_slot_fraction']
    status=('QUERY_BOOK_CONTEXT_INCOMPLETE' if method.get('mechanism_required') and insufficient else
            'NO_CAUSAL_HISTORY' if not len(score) else 'WITHIN_DECLARED_TOLERANCE' if result else
            'NO_CANDIDATE_MEETS_CONTEXT_GATES' if not eligible.any() else 'NO_ANALOGUE_WITHIN_DECLARED_TOLERANCE')
    summary=dict(method=method['name'],scope=prepared['scope'],scale=prepared['token'].scale,
        query_token_id=prepared['token'].token_id,query_ns=int(prepared['token'].available_ns),
        causal_candidates=len(score),context_eligible=int(eligible.sum()),within_tolerance=int(accepted.sum()),
        returned=len(result),status=status)
    return summary,result


def context_frame(search,prepared,match):
    ix=match['virtual_index']
    return pd.concat([search.base.frame.iloc[prepared['parents'][ix]],prepared['virtual'].iloc[[ix]]],ignore_index=True)
