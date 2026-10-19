"""Causal multi-scale volatility and volume-weighted price geometry.

PROXY uses endpoint midpoint weighted by all T volume of that second.
EXACT requires a separate, aligned trade-moments sidecar; no silent fallback.
Finite rolling windows deliberately fit the existing bounded-halo adapter.
"""
import numpy as np
import pandas as pd
from spectrum.lab import FeaturePlugin, rolling_valid

SCALES = (60, 300, 900, 3600)
TICK = .1
VERSION = '1'
EXACT_COLUMNS = ('trade_volume', 'trade_price_mean', 'trade_price_m2',
                 'trade_moments_valid', 'trade_moments_available_ns')


def trade_moments(prices, quantities):
    """Central moments of accepted T prints, including unknown aggressor side.

    Caller supplies ONE complete publication/bucket, never passive F duplicates.
    This helper does not itself reconstruct, schedule or prove source support.
    """
    p=np.asarray(prices,float);q=np.asarray(quantities,float)
    if p.shape!=q.shape or p.ndim!=1 or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(q<=0):
        raise ValueError('Invalid trade prices or positive quantities')
    if not len(p): return dict(trade_volume=0.,trade_price_mean=0.,trade_price_m2=0.)
    total=float(q.sum());origin=p[0];offset=float(np.dot(q,p-origin)/total)
    return dict(trade_volume=total,trade_price_mean=float(origin+offset),
                trade_price_m2=float(np.dot(q,(p-origin-offset)**2)))


def roll(x,n,op='sum'):
    return getattr(pd.Series(x).rolling(n,min_periods=n),op)().to_numpy()


def divide(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.zeros(a.shape,float),where=np.isfinite(b)&(b>1e-12))


def lag(x,n):
    return np.r_[np.full(min(n,len(x)),np.nan),x[:-n]] if len(x)>n else np.full(len(x),np.nan)


def compute(c, *, scales=SCALES, basis='midpoint_proxy'):
    f=c.df; count=len(f)
    if basis not in ('midpoint_proxy','trade_exact'): raise ValueError('Unknown VWAP price basis')
    if not scales or any(type(n)!=int or n<10 or n%5 for n in scales) or tuple(sorted(set(scales)))!=tuple(scales):
        raise ValueError('Scales must be ascending unique seconds, multiples of 5, >=10')
    if count==0: raise ValueError('Empty input')
    if np.any(np.diff(c.t)<=0): raise ValueError('Non-increasing feature clock')
    p=f.mid.to_numpy(float);av=f.available_ns.to_numpy(np.int64)
    buy,sell,u=(c.delta('c_trade_'+s) for s in ('buy','sell','unknown'))
    v=buy+sell+u
    edge=c.edge & np.isfinite(p) & np.isfinite(lag(p,1))
    edge &= np.isfinite(v)&(buy>=0)&(sell>=0)&(u>=0)
    within=np.zeros(count)
    if basis=='trade_exact':
        missing=set(EXACT_COLUMNS)-set(f.columns)
        if missing: raise ValueError('Exact VWAP requires aligned T-print moments: '+', '.join(sorted(missing)))
        q=f.trade_volume.to_numpy(float);price=f.trade_price_mean.to_numpy(float)
        within=f.trade_price_m2.to_numpy(float);ma=f.trade_moments_available_ns.to_numpy(np.int64)
        moments_ok=f.trade_moments_valid.to_numpy(bool)&np.isfinite(q)&(q>=0)&np.isfinite(within)&(within>=0)
        moments_ok &= (q==0)|np.isfinite(price)
        if np.any(c.edge & moments_ok & ~np.isclose(q,v,rtol=0,atol=1e-8)):
            raise ValueError('Exact T volume disagrees with reconstructed buy+sell+unknown')
        if np.any(moments_ok & (q==0)&(within!=0)): raise ValueError('Zero-volume bucket has nonzero M2')
        edge &= moments_ok
        dependency=np.maximum(av,ma)
    else:
        q=v;price=p;dependency=av
    prefix='vwap_proxy' if basis=='midpoint_proxy' else 'vwap_exact'
    q=np.where(edge,q,0.)
    # First supported nonempty bucket is already in the past of any known VWAP.
    # A fixed reference reduces cancellation, including captures starting with NaN.
    anchors=np.flatnonzero(edge&(q>0)&np.isfinite(price))
    origin=float(price[anchors[0]]) if len(anchors) else 0.
    px=np.where(edge&(q>0),price-origin,0.)
    m2=np.where(edge,within,0.)
    r=np.where(edge,(p-lag(p,1))/TICK,0.)
    u=np.where(edge,u,0.);idx=np.arange(count,dtype=float)
    scale_values={}
    for n in scales:
        supported=rolling_valid(edge,n)
        # Counter difference also depends on the left endpoint (n+1 snapshots).
        published=roll(av,n+1,'max')<=av
        published &= roll(dependency,n,'max')<=av
        good=supported & published
        weight=roll(q,n);mu_offset=divide(roll(q*px,n),weight)
        variance=divide(roll(m2+q*px*px,n),weight)-mu_offset**2
        # Roundoff only: materially negative variance invalidates, not clips.
        tolerance=1e-10*np.maximum(1.,divide(roll(m2+q*px*px,n),weight))
        var_ok=np.isfinite(variance)&(variance>=-tolerance)
        sigma=np.sqrt(np.maximum(variance,0.));mu=origin+mu_offset
        known=good&(weight>0)&var_ok
        nonzero=known&(sigma>1e-10)
        stem=f'{prefix}_{n}s_'
        def add(name,value,mask=good,unit='ratio'):
            c.add(stem+name,value,mask,unit)
        add('center',mu,known,'USD');add('sigma_ticks',sigma/TICK,known,'tick')
        for k in (1,2):
            add(f'upper{k}',mu+k*sigma,known,'USD');add(f'lower{k}',mu-k*sigma,known,'USD')
        add('z',divide(p-mu,sigma),nonzero)
        add('distance_ticks',(p-mu)/TICK,known,'tick')
        add('volume_rate',weight/n,good,'contract/second')
        add('unknown_fraction',divide(roll(u,n),weight),known)
        add('effective_seconds',divide(weight**2,roll(q*q,n)),known,'effective buckets')
        add('weighted_age_seconds',idx-divide(roll(q*idx,n),weight),known,'second')
        # Separate dispersion of price levels from variation of price increments.
        rv=roll(r*r,n);rmean=roll(r,n)/n
        rms=np.sqrt(np.maximum(rv/n,0.))
        add('realized_variation_1s',rv,good,'tick squared')
        add('return_rms_1s',rms,good,'tick')
        add('return_std_1s',np.sqrt(np.maximum(rv/n-rmean*rmean,0.)),good,'tick')
        add('time_price_sigma_ticks',pd.Series(p).rolling(n,min_periods=n).std(ddof=0).to_numpy()/TICK,good,'tick')
        # n/5 NONOVERLAPPING 5s increments ending at the current endpoint.
        r5=(p-lag(p,5))/TICK;rv5=np.full(count,np.nan)
        for phase in range(5):rv5[phase::5]=roll(r5[phase::5]**2,n//5)
        add('realized_variation_5s',rv5,good,'tick squared')
        add('sampling_variation_ratio',divide(rv,rv5),good&(rv5>1e-12))
        add('spread_over_sigma',divide(f.ask.to_numpy()-f.bid.to_numpy(),sigma),nonzero)
        prev_mu=lag(mu,5);prev_sigma=lag(sigma,5)
        paired=known & np.r_[np.zeros(min(5,count),bool),known[:-5]] if count>5 else np.zeros(count,bool)
        paired &= rolling_valid(edge,5)
        add('center_change_5s_ticks',(mu-prev_mu)/TICK,paired,'tick')
        add('sigma_change_5s_ticks',(sigma-prev_sigma)/TICK,paired,'tick')
        # Exact algebraic split of delta z into price, center and width motion.
        split_ok=paired&(sigma>1e-10)&(prev_sigma>1e-10)
        price_part=divide(p-lag(p,5),prev_sigma)
        center_part=divide(prev_mu-mu,prev_sigma)
        width_part=divide(p-mu,sigma)-divide(p-mu,prev_sigma)
        add('z_change_price_part_5s',price_part,split_ok)
        add('z_change_center_part_5s',center_part,split_ok)
        add('z_change_width_part_5s',width_part,split_ok)
        prior_good=np.r_[False,good[:-1]]&good
        add('return_over_prior_rms',divide(r,lag(rms,1)),prior_good&(lag(rms,1)>1e-12))
        add('volume_over_prior_rate',divide(q,lag(weight/n,1)),prior_good&(lag(weight,1)>0))
        scale_values[n]=(mu,sigma,weight,known)
    for short,long in zip(scales,scales[1:]):
        a,sa,wa,ka=scale_values[short];b,sb,wb,kb=scale_values[long]
        stem=f'{prefix}_{short}_{long}s_'
        c.add(stem+'width_ratio',divide(sa,sb),ka&kb&(sb>1e-10),'ratio')
        c.add(stem+'center_gap_long_sigma',divide(a-b,sb),ka&kb&(sb>1e-10),'ratio')
        c.add(stem+'volume_rate_ratio',divide(wa/short,wb/long),ka&kb,'ratio')


def proxy(c): compute(c,basis='midpoint_proxy')
def exact(c): compute(c,basis='trade_exact')

# 3600s window + 5s derivative + one extra endpoint retained by FeatureStream.
PROXY = FeaturePlugin('volatility_vwap_proxy',VERSION,proxy,max(SCALES)+5)
EXACT = FeaturePlugin('volatility_vwap_exact',VERSION,exact,max(SCALES)+5)
PLUGIN = PROXY
