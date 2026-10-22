"""Full line-addressed crossing maps and explicitly separated economic queries."""
import hashlib
import inspect
import numpy as np
from numba import njit
from spectrum.cognition import bb_crossing as bb

SCALES=(1,3,5,6,9)
NS=10**9


def geometry_id(family,scale,quote_parts,trade_source_sha=None):
    """A physical crossing ID cannot depend on entries, risk queries or loss code."""
    formula={'family':family,'scale':scale,'geometry':bb.geometry_definition(),
        'clock':'NATIVE_FULL_PUBLICATION_ORDINAL_V1',
        'vwap_code':hashlib.sha256(inspect.getsource(vwap_geometry).encode()).hexdigest() if family=='VWAP' else None}
    return bb.identity(formula,trade_source_sha if family=='VWAP' else None,
        [(p['quote_file']['sha256'],p['start_ns'],p['end_ns']) for p in quote_parts])


def vwap_geometry(qt,k,trade_ord,trade_t,price,quantity,scale,support):
    """Native 19 completed buckets plus current; right edge is publication ORDINAL.

    Two publications can have identical nanoseconds. Never include later trades
    simply because their timestamp equals that of the currently processed quote.
    """
    left=(qt[k]//(scale*60*NS)-19)*(scale*60*NS)
    a=np.searchsorted(trade_t,left,side='left');b=np.searchsorted(trade_ord,k,side='right')
    q=np.asarray(quantity,float);p=np.asarray(price,float);ref=p[0];z=p-ref
    sums=[np.r_[0,np.cumsum(v)] for v in (q,q*z,q*z*z)]
    w,sp,sq=[v[b]-v[a] for v in sums]
    mu=np.divide(sp,w,out=np.full(len(k),np.nan),where=w>0)
    variance=np.divide(sq,w,out=np.full(len(k),np.nan),where=w>0)-mu*mu
    ready=(support>=0)&(w>0)
    if np.any(variance[ready]<-1e-7):raise ValueError('Negative supported VWAP variance')
    half=2*np.sqrt(np.maximum(variance,0));ready &= np.isfinite(half)&(half>1e-10)
    return np.where(ready,ref+mu,np.nan),np.where(ready,half,np.nan),np.where(ready,support,-1)


def lookup_line(ordinals,entries,ready,end_ns,qt):
    ix=np.searchsorted(ordinals,entries,side='right')
    if not len(ordinals):return np.full(len(entries),-1,np.int64)
    c=ordinals[np.minimum(ix,len(ordinals)-1)]
    known=ready&(ix<len(ordinals))&(qt[c]<end_ns)
    return np.where(known,c,-1)


@njit(cache=True)
def first_passage(qt,bid,ask,k,entries,limits,risk):
    """First net gain before first net loss; three gains .5R/R/2R, two sides.

    Return gain/stop publication indices. Censoring and horizon masks are
    interpreted by the caller. The fixed R comes only from decision-time X.
    """
    hits=np.full((len(entries),2,3),-1,np.int64);stops=np.full((len(entries),2),-1,np.int64)
    gains=np.array([.5,1.,2.])
    for n in range(len(entries)):
        e=entries[n]
        if e<0 or not np.isfinite(risk[n]):continue
        end=min(limits[n],qt[e]+1800*NS+1)
        i=np.searchsorted(k,e,side='left')
        # Entry publication must be evaluated even when BBO compression omits it.
        r=e
        while r<len(qt) and qt[r]<end:
            net=np.array([(bid[r]-ask[e])/1e9-.3,(bid[e]-ask[r])/1e9-.3])
            for side in range(2):
                if stops[n,side]>=0:continue
                if net[side]<=-risk[n]:stops[n,side]=r
                else:
                    for g in range(3):
                        if hits[n,side,g]<0 and net[side]>=gains[g]*risk[n]:hits[n,side,g]=r
            done=True
            for side in range(2):
                if stops[n,side]<0 and hits[n,side,2]<0:done=False
            if done:break
            while i<len(k) and k[i]<=r:i+=1
            if i>=len(k):break
            r=k[i]
    return hits,stops


def factorized_loss(event_logits,direction_logits,event,direction,event_known,direction_known):
    """E BCE + conditional {L-only,S-only,BOTH} CE, averaged over known E roots.

    Returns exact logits gradients. No direction loss on NONE/unknown roots.
    A positive observed witness may train E while the other side is censored.
    """
    z=np.asarray(event_logits,float);d=np.asarray(direction_logits,float)
    y=np.asarray(event,float);known=np.asarray(event_known,bool)
    conditional=known&(y==1)&np.asarray(direction_known,bool)
    n=max(1,int(known.sum()));p=1/(1+np.exp(-z))
    ez=d-d.max(axis=1,keepdims=True);prob=np.exp(ez);prob/=prob.sum(axis=1,keepdims=True)
    loss=(np.logaddexp(0,z)-y*z)[known].sum()
    selected=np.flatnonzero(conditional);labels=np.asarray(direction,int)[selected]
    loss-=np.log(prob[selected,labels]).sum()
    ge=np.where(known,p-y,0)/n;gd=np.zeros_like(d)
    gd[selected]=prob[selected];gd[selected,labels]-=1;gd/=n
    return float(loss/n),ge,gd
