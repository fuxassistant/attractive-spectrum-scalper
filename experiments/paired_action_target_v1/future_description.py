"""Future geometry on a common clock window, separate from stopped-trade marks.

These are auxiliary labels, never features. No synonym such as 'good market'
or 'viscous regime' is substituted for the actual measured coordinates.
"""
import numpy as np

from experiments.regime_language_v1.measures import TimeAtPrice

NS=1_000_000_000


def describe_future(quotes, clock_ns, origin_price, risk_per_oz, *, horizon_seconds=120, mapper=None, observed_through_ns=None):
    if isinstance(clock_ns,bool) or not isinstance(clock_ns,(int,np.integer)):
        raise ValueError('Exact integer decision clock required')
    if not np.isfinite(risk_per_oz) or risk_per_oz<=0:
        raise ValueError('Positive frozen risk scale required')
    end=int(clock_ns)+int(horizon_seconds*NS)
    if end<=clock_ns:raise ValueError('Positive horizon required')
    mapper=mapper or TimeAtPrice(quotes)
    observed_through=int(mapper.t[-1]) if observed_through_ns is None else int(observed_through_ns)
    qt=mapper.t;lo=max(0,int(qt.searchsorted(clock_ns,'right'))-1);hi=int(qt.searchsorted(end,'left'))
    frame=quotes.iloc[lo:hi]
    profile=mapper.facets(int(clock_ns),end,risk_per_oz/.1)
    known=bool(end<=observed_through and len(frame) and int(qt[lo])<=clock_ns and profile['known']
               and frame.epoch.nunique()==1 and frame.epoch.iloc[0]>=0 and frame.valid.all())
    out=dict(future_description_known=known,future_description_available_ns=end,
             future_range_r=None,future_net_r=None,future_efficiency_1s=None,
             future_time_concentration=None,future_time_sigma_r=None,
             future_half_time_overlap=None,future_time_center_migration_r=None,
             future_origin_band_dwell_fraction=None)
    if not known:return out
    p,duration,_,_=mapper.pieces(int(clock_ns),end)
    # Time occupancy excludes zero-duration publications. Range also retains
    # every observed native intermediate state, including equal-clock extrema.
    native=(frame.bid_nano.to_numpy(np.int64)+frame.ask_nano.to_numpy(np.int64))/(2*NS)
    out['future_range_r']=float((max(float(native.max()),origin_price)-min(float(native.min()),origin_price))/risk_per_oz)
    out['future_net_r']=float((native[-1]-origin_price)/risk_per_oz)
    grid=np.unique(np.r_[int(clock_ns),np.arange(int(clock_ns)+NS,end,NS,dtype=np.int64),end-1])
    ids=qt.searchsorted(grid,'right')-1
    sampled=mapper.p[ids]/NS
    sampled[0]=origin_price
    path=float(np.abs(np.diff(sampled)).sum())
    out['future_efficiency_1s']=float(abs(sampled[-1]-sampled[0])/path) if path else 0.
    out['future_time_concentration']=profile['largest_bin_fraction']
    out['future_time_sigma_r']=profile['time_sigma']/risk_per_oz
    out['future_half_time_overlap']=profile['half_profile_overlap']
    out['future_time_center_migration_r']=profile['center_migration_A']
    out['future_origin_band_dwell_fraction']=float(duration[np.abs(p/NS-origin_price)<=.5*risk_per_oz].sum()/(end-clock_ns))
    return out
