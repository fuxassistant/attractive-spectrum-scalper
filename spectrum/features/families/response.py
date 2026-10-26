"""Reusable formulas extracted from spectrum/response_lab.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd

NS=10**9


WINDOW=512


HOP=128


SEGMENT=128


BANDS=((4,8),(8,16),(16,32))


HORIZONS=(30,120)


COUNTERS=('trade_buy','trade_sell','trade_unknown','path_ticks','add_bid','add_ask',
          'remove_bid','remove_ask','matched_fill_remove_bid','matched_fill_remove_ask',
          'post_trade_add_bid','post_trade_add_ask')


def ratio(a,b):
    return float(a/b) if b>1e-12 else 0.


def spectral_features(channels):
    """Seven overlapping past segments; smooth spectra BEFORE normalization."""
    names=list(channels);v=np.stack([channels[n] for n in names])
    z=np.lib.stride_tricks.sliding_window_view(v,SEGMENT,axis=-1)[:,::64,:]
    w=.5-.5*np.cos(2*np.pi*np.arange(SEGMENT)/SEGMENT)
    ft=np.fft.rfft((z-z.mean(axis=-1,keepdims=True))*w,axis=-1)
    psd=(abs(ft)**2).mean(axis=1)/(w@w)
    one_sided=np.ones(SEGMENT//2+1);one_sided[1:-1]=2
    psd*=one_sided
    f=np.fft.rfftfreq(SEGMENT)
    auto={};cross={}
    for k,n in enumerate(names):
        total=psd[k,1:].sum()
        auto[n+'_power']=float(np.log1p(total/SEGMENT))
        auto[n+'_power_known']=float(total>1e-12)
        prob=psd[k,1:]/total if total>1e-12 else np.zeros(SEGMENT//2)
        auto[n+'_entropy']=float(-(prob*np.log(np.maximum(prob,1e-300))).sum()/np.log(len(prob)))
        for a,b in BANDS:
            mask=(f>=1/b)&(f<1/a)
            auto[f'{n}_fraction_{a}_{b}']=ratio(psd[k,mask].sum(),total)
    for x,y in [('classified_flow','return'),('book_flow','return'),('volume','abs_return')]:
        i,j=names.index(x),names.index(y)
        xy=(ft[i].conj()*ft[j]).mean(axis=0)/(w@w)*one_sided
        den=psd[i]*psd[j]
        coherence=np.divide(abs(xy)**2,den,out=np.zeros_like(den),where=den>1e-20)
        for a,b in BANDS:
            mask=(f>=1/b)&(f<1/a);good=mask&(den>1e-20)
            stem=f'{x}_to_{y}_{a}_{b}'
            known=bool(np.any(good))
            cross[stem+'_known']=float(known)
            cross[stem+'_coherence']=float(np.clip(coherence[good].mean(),0,1)) if known else 0.
            # Sum cross spectra and input power within the band: weighted complex gain.
            gain=xy[good].sum()/psd[i,good].sum() if known else 0j
            cross[stem+'_gain_log']=float(np.log1p(abs(gain)))
            phase_known=known and abs(gain)>1e-12
            cross[stem+'_phase_known']=float(phase_known)
            cross[stem+'_phase_cos']=float(np.cos(np.angle(gain))) if phase_known else 0.
            cross[stem+'_phase_sin']=float(np.sin(np.angle(gain))) if phase_known else 0.
    return auto,cross


def window_features(d,end):
    """Only [end-512,end] is accessed. Unknown side is an observed separate channel."""
    z=d.iloc[end-WINDOW:end+1];deltas={n:np.diff(z['c_'+n].to_numpy()) for n in COUNTERS}
    ret=np.diff(z.mid.to_numpy())/.1
    buy,sell,u=(deltas[n] for n in ('trade_buy','trade_sell','trade_unknown'))
    q=buy-sell;v=buy+sell+u
    book=(deltas['add_bid']-deltas['remove_bid'])-(deltas['add_ask']-deltas['remove_ask'])
    base={}
    for n in (30,128,512):
        vol=float(v[-n:].sum());known=float((buy+sell)[-n:].sum());net=float(q[-n:].sum());unknown=float(u[-n:].sum())
        move=float(ret[-n:].sum());path=float(deltas['path_ticks'][-n:].sum())
        vals=dict(volume_log=np.log1p(vol),classified_flow=np.sign(net)*np.log1p(abs(net)),
            imbalance=ratio(net,known),unknown_fraction=ratio(unknown,vol),volume_known=float(vol>0),
            pressure_lower=ratio(net-unknown,vol),pressure_upper=ratio(net+unknown,vol),
            progress=move,path_log=np.log1p(max(path,0)),efficiency=ratio(abs(move),path),
            ticks_per_100=ratio(100*move,vol),return_std=float(ret[-n:].std()),
            flow_std=float(q[-n:].std()),corr=float(np.corrcoef(q[-n:],ret[-n:])[0,1]) if q[-n:].std()>0 and ret[-n:].std()>0 else 0.)
        for k in COUNTERS[4:]:vals[k+'_log']=np.log1p(max(0,float(deltas[k][-n:].sum())))
        for k,value in vals.items():base[f'{k}_{n}']=float(value)
    last=z.iloc[-1]
    base['spread_ticks']=float((last.ask-last.bid)/.1)
    for n in (1,4,12):
        bid=float(last[f'depth_bid_{n}']);ask=float(last[f'depth_ask_{n}'])
        base[f'depth_{n}_log']=float(np.log1p(bid+ask));base[f'depth_{n}_imbalance']=ratio(bid-ask,bid+ask)
    tm=pd.Timestamp(int(last.t_ns),tz='UTC');hour=tm.hour+tm.minute/60
    base['utc_hour_sin']=float(np.sin(hour*2*np.pi/24));base['utc_hour_cos']=float(np.cos(hour*2*np.pi/24))
    auto,cross=spectral_features({
        'return':ret,'classified_flow':q,'volume':v,'book_flow':book,'abs_return':abs(ret)})
    result={**{'A__'+k:v for k,v in base.items()},**{'B__'+k:v for k,v in auto.items()},**{'C__'+k:v for k,v in cross.items()}}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite laboratory feature')
    return result
