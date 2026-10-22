"""Causal trade-weighted bands and native-bar close controls."""
import numpy as np

NS=10**9


class TradePrefix:
    def __init__(self,t,price,quantity):
        self.t=np.asarray(t,np.int64);p=np.asarray(price,float);q=np.asarray(quantity,float)
        if not(len(p)==len(q)==len(self.t)) or not len(p):raise ValueError('Empty/misaligned trades')
        if np.any(np.diff(self.t)<0) or np.any(q<=0) or not np.isfinite(p).all():raise ValueError('Invalid trades')
        self.origin=p[0];self.clock=self.t[0];z=p-self.origin
        self.sums=[np.r_[0,np.cumsum(v)] for v in (q,q*z,q*z*z,q*((self.t-self.clock)/NS))]

    def volume(self,left,right):
        a=np.searchsorted(self.t,left,side='left');b=np.searchsorted(self.t,right,side='right')
        return self.sums[0][b]-self.sums[0][a]

    def query(self,left,right):
        """All published prints in [left,right]; left is a native bar opening."""
        a=np.searchsorted(self.t,left,side='left');b=np.searchsorted(self.t,right,side='right')
        w,p,p2,wt=[s[b]-s[a] for s in self.sums]
        good=w>0
        mu=np.divide(p,w,out=np.full_like(w,np.nan),where=good)
        var=np.divide(p2,w,out=np.full_like(w,np.nan),where=good)-mu*mu
        if np.any(var[good]<-1e-7):raise ValueError('Material negative variance')
        age=(np.asarray(right)-self.clock)/NS-np.divide(wt,w,out=np.full_like(w,np.nan),where=good)
        return self.origin+mu,2*np.sqrt(np.maximum(var,0)),age,w


def support_start(t,valid,epoch):
    breaks=np.r_[True,(~valid[1:])|(~valid[:-1])|(epoch[1:]!=epoch[:-1])|(np.diff(t)>60*NS)]
    return np.maximum.accumulate(np.where(breaks,t,np.iinfo(np.int64).min))


def weighted_native_closes(qt,price,queries,scale,prefix):
    """19 completed close samples + current price, weighted by each bar T volume.

    Population weighted variance. This is VWMA(close), NOT exact trade VWAP.
    Caller applies identical native BB support/warmup masks.
    """
    width=scale*60*NS;bucket=queries//width*width
    values=[];weights=[];ages=[]
    for i in range(19,0,-1):
        left=bucket-i*width;right=left+width-1
        j=np.searchsorted(qt,right,side='right')-1
        values.append(price[np.maximum(j,0)])
        weights.append(prefix.volume(left,right))
        ages.append((queries-qt[np.maximum(j,0)])/NS)
    j=np.searchsorted(qt,queries,side='right')-1
    values.append(price[np.maximum(j,0)]);weights.append(prefix.volume(bucket,queries));ages.append(np.zeros(len(queries)))
    p=np.asarray(values);w=np.asarray(weights);age=np.asarray(ages);total=w.sum(axis=0)
    center=p[-1];z=p-center
    mean=np.divide((w*z).sum(axis=0),total,out=np.full(len(total),np.nan),where=total>0)
    var=np.divide((w*(z-mean)**2).sum(axis=0),total,out=np.full(len(total),np.nan),where=total>0)
    weighted_age=np.divide((w*age).sum(axis=0),total,out=np.full(len(total),np.nan),where=total>0)
    return center+mean,2*np.sqrt(var),weighted_age,age.mean(axis=0)


def statistics(f,name):
    """Common paired support; differences of central lines do not imply profit."""
    base=f.bb.to_numpy();other=f[name].to_numpy();half=f.bb_half.to_numpy()
    ok=np.isfinite(base)&np.isfinite(other)&np.isfinite(half)&(half>0)
    d=other[ok]-base[ok];ratio=f[name+'_half'].to_numpy()[ok]/half[ok]
    paired=ok&np.r_[np.zeros(60,bool),ok[:-60]]
    paired &= f.segment.to_numpy()==np.r_[np.full(60,-99),f.segment.to_numpy()[:-60]]
    slope_b=base-np.r_[np.full(60,np.nan),base[:-60]];slope_v=other-np.r_[np.full(60,np.nan),other[:-60]]
    active=paired&(abs(slope_b)>.1)&(abs(slope_v)>.1)
    contiguous=ok[1:]&ok[:-1]&(np.diff(f.segment.to_numpy())==0)
    def quant(x,p):return float(np.quantile(x,p)) if len(x) else None
    return dict(rows=int(ok.sum()),center_bias_usd=quant(d,.5),center_abs_median_usd=quant(abs(d),.5),center_abs_p90_usd=quant(abs(d),.9),
        center_rmse_usd=float(np.sqrt(np.mean(d*d))),center_abs_over_bb_half_median=quant(abs(d)/half[ok],.5),
        half_width_ratio_median=quant(ratio,.5),half_width_ratio_p10=quant(ratio,.1),half_width_ratio_p90=quant(ratio,.9),
        price_side_disagreement=float(np.mean((f.mid.to_numpy()[ok]>base[ok])!=(f.mid.to_numpy()[ok]>other[ok]))),
        slope_active_pairs=int(active.sum()),slope_sign_disagreement=float(np.mean(np.sign(slope_b[active])!=np.sign(slope_v[active]))) if active.any() else None,
        slope_correlation=float(np.corrcoef(slope_b[paired],slope_v[paired])[0,1]) if paired.any() else None,
        line_total_variation_ratio=float(abs(np.diff(other))[contiguous].sum()/abs(np.diff(base))[contiguous].sum()),
        age_seconds_median=quant(f[name+'_age'].to_numpy()[ok],.5),bb_age_seconds_median=quant(f.bb_age.to_numpy()[ok],.5))
