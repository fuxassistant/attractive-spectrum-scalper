"""Causal retrieval on a fixed candidate population with auditable component scores."""
import numpy as np
import pandas as pd

from experiments.scalper_adapter_v1.similarity import StoryIndex,descriptors,ordered_distance
from .features import PublishedBook,extract
from .metric import compare,score_components,BLOCKS

FEATURE_KEYS=['shape','amplitude','texture','duration','structure','book','book_known']


class SimilaritySearch:
    def __init__(self,sessions,config):
        self.config=config;self.sessions={s.session_id:s for s in sessions}
        if len({s.symbol for s in sessions})!=1: raise ValueError('One contract required')
        self.base=StoryIndex(sessions)
        self.books={s.session_id:PublishedBook(s.fast) for s in sessions}
        packs=[extract(s.motion,s.quotes,self.books[s.session_id],samples=config['shape_samples']) for s in sessions]
        self.features={k:np.concatenate([p[k] for p in packs]) for k in FEATURE_KEYS}
        self.book_endpoints=np.concatenate([p['book_endpoint_ns'] for p in packs])
        self.book_releases=np.concatenate([p['book_release_ns'] for p in packs])
        self.parent_indices=np.column_stack([self.base.parents,np.arange(len(self.base.frame))])

    def query_context(self,token):
        f=self.base.frame
        ids=self.base.completed.get((token.session_id,token.scale,token.segment),np.array([],int))
        ids=ids[f.loc[ids,'available_ns'].to_numpy()<=int(token.start_ns)][-2:]
        if len(ids)!=2: raise ValueError('Two observed completed predecessors required')
        frame=pd.concat([f.loc[ids],pd.DataFrame([token])],ignore_index=True)
        session=self.sessions[token.session_id]
        pack=extract(frame,session.quotes,self.books[token.session_id],samples=self.config['shape_samples'])
        old=np.vstack([self.base.vectors[ids],descriptors(pd.DataFrame([token]),session.quotes)])
        return frame,{k:pack[k] for k in FEATURE_KEYS},old

    def prepare(self,token,scope='previous'):
        frame,query,legacy=self.query_context(token)
        current=self.sessions[token.session_id]
        if scope not in ('previous','current'): raise ValueError('Explicit session scope required')
        date=current.previous_session_id if scope=='previous' else token.session_id
        start=int(frame.start_ns.min());f=self.base.frame
        eligible=(f.scale.eq(token.scale)&f.phase.eq(token.phase)&f.elapsed_seconds.gt(0)
                  &f.available_ns.lt(start)&f.session_id.eq(date)&self.base.mask.all(axis=1))
        ids=f.index[eligible].to_numpy()
        positions=self.parent_indices[ids]
        candidate={k:v[positions] for k,v in self.features.items()}
        parts=compare(query,candidate,duration_floor=self.config['duration_floor_seconds'],
                      direction_floor=self.config['direction_floor_amplitudes'])
        parts['legacy']=ordered_distance(legacy,self.base.context[ids],[True,True,True])
        return dict(token=token,query_frame=frame,query_pack=query,ids=ids,parts=parts,
                    scope=scope,query_context_start_ns=start)

    def rank(self,prepared,method,*,neighbors=None):
        ids=prepared['ids'];parts=prepared['parts']
        score,eligible,accepted=score_components(parts,method,self.config)
        n=self.config['neighbors'] if neighbors is None else neighbors
        valid=np.flatnonzero(eligible&np.isfinite(score))
        valid=valid[np.argsort(score[valid],kind='stable')]
        chosen=[];intervals=[];stories=set()
        for ix in valid:
            if method['name']!='legacy' and not accepted[ix]: continue
            i=int(ids[ix]);row=self.base.frame.iloc[i]
            a=int(self.base.context_start[i]);b=int(row.available_ns)
            if row.story_id in stories or any(a<=y and x<=b for x,y in intervals):continue
            stories.add(row.story_id);intervals.append((a,b))
            chosen.append(self._record(prepared,method,ix,float(score[ix]),len(chosen)+1))
            if len(chosen)>=n: break
        best=int(valid[0]) if len(valid) else None
        query_known=float(prepared['query_pack']['book_known'].mean())
        query_slot=float(prepared['query_pack']['book_known'].mean(axis=1).min())
        query_insufficient=(query_known<self.config['mechanism_min_common_fraction'] or
                            query_slot<self.config['mechanism_min_slot_fraction'])
        if method.get('mechanism_required') and query_insufficient: status='QUERY_BOOK_CONTEXT_INCOMPLETE'
        elif not len(ids): status='NO_CAUSAL_HISTORY'
        elif not len(valid): status='NO_CANDIDATE_MEETS_CONTEXT_GATES'
        elif not len(chosen): status='NO_ANALOGUE_WITHIN_DECLARED_TOLERANCE'
        else: status='UNCALIBRATED_NEAREST' if method['name']=='legacy' else 'WITHIN_DECLARED_TOLERANCE'
        summary=dict(method=method['name'],scope=prepared['scope'],scale=prepared['token'].scale,
            query_token_id=prepared['token'].token_id,query_ns=int(prepared['token'].available_ns),
            query_context_start_ns=prepared['query_context_start_ns'],causal_candidates=len(ids),
            context_eligible=int(eligible.sum()),within_tolerance=int(accepted.sum()),
            returned=len(chosen),status=status,
            query_book_known_fraction=query_known,query_book_min_slot_fraction=query_slot,
            best_eligible_score=float(score[best]) if best is not None else None,
            best_eligible_shape_rmse=float(parts['shape_rmse'][best]) if best is not None else None)
        return summary,chosen

    def _record(self,prepared,method,ix,score,rank):
        i=int(prepared['ids'][ix]);row=self.base.frame.iloc[i];p=prepared['parts']
        return dict(method=method['name'],scope=prepared['scope'],scale=row.scale,rank=rank,
            query_token_id=prepared['token'].token_id,query_ns=int(prepared['token'].available_ns),
            query_context_start_ns=prepared['query_context_start_ns'],token_id=row.token_id,session_id=row.session_id,
            context_start_ns=int(self.base.context_start[i]),available_ns=int(row.available_ns),score=score,
            context_token_ids=self.base.frame.iloc[self.parent_indices[i]].token_id.tolist(),
            query_context_token_ids=prepared['query_frame'].token_id.tolist(),
            **{key:float(value[ix]) if np.isfinite(value[ix]) else None for key,value in p.items()},
            weights=method['weights'],meaning='Declared resemblance tolerances; not calibrated probability, target or trading admission')


def sensitivity_methods(config):
    result=[]
    template=next(m for m in config['methods'] if m['name']=='rhythm_gate')
    non_rhythm={'shape':4/15,'amplitude':2/15,'texture':1/15,'structure':2/15,'trajectory':6/15}
    for weight in config['sensitivity']['rhythm_weights']:
        for factor in config['sensitivity']['total_duration_factors']:
            result.append(dict(template,name=f'time_w{weight}_f{factor}',
                weights={**{k:(1-weight)*v for k,v in non_rhythm.items()},'rhythm':weight},
                total_duration_factor=factor,slot_duration_factor=2. if factor else None))
    for weight in config['sensitivity']['book_weights']:
        result.append(dict(template,name=f'book_w{weight}',mechanism_required=weight>0,
            weights={**{k:(1-weight)*v for k,v in template['weights'].items()},'mechanism':weight}))
    return result
