"""Identities of crossing geometries and the native VWAP band geometry."""
import hashlib
import inspect
import numpy as np
import spectrum.crossings.bank as bb

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
