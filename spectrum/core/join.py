"""Target-free temporal enrichment; instrument identity never implies price parity."""
import numpy as np


def asof_enrichment(origin_ns,available_ns,values,valid,*,max_age_ns,source_instrument,destination_instrument):
    """Return aligned values, age and mask; cross-instrument output is context.

    Caller provides already causal source features. This does not map a GC price
    level/order to a CFD level or invent data before a source publication.
    """
    origin=np.asarray(origin_ns,np.int64);available=np.asarray(available_ns,np.int64)
    x=np.asarray(values);support=np.asarray(valid,bool)
    if x.ndim!=2 or len(x)!=len(available) or len(support)!=len(available):raise ValueError('Unaligned source features')
    if len(available)==0 or np.any(np.diff(available)<=0):raise ValueError('Source publications must increase strictly')
    if max_age_ns<0:raise ValueError('Negative age limit')
    if not source_instrument or not destination_instrument:raise ValueError('Instrument identity required')
    ix=np.searchsorted(available,origin,side='right')-1;safe=np.maximum(ix,0)
    age=np.where(ix>=0,origin-available[safe],np.iinfo(np.int64).max)
    mask=(ix>=0)&support[safe]&(age<=max_age_ns)&np.isfinite(x[safe]).all(1)
    values=np.where(mask[:,None],x[safe],0.)
    return dict(values=values,source_rows=np.where(mask,ix,-1),age_ns=age,valid=mask,
        source_instrument=source_instrument,destination_instrument=destination_instrument,
        relation='same_instrument' if source_instrument==destination_instrument else 'external_context',
        price_level_mapping=False,y_access=False)
