"""Observable activity spectra and explicit aggressor-side uncertainty.

An unknown side is still an observed trade. ``observed_signed`` means B-A of
classified prints, never an assertion that unclassified prints have zero sign.
The original strict signed spectrum remains a separate optional family output.
"""
import numpy as np
import pandas as pd
from spectrum.features.contract import FeaturePlugin,rolling_valid


def white_reference(n, selected):
    """Ratio of expected powers for iid noise under this exact linear window.

    This is not E[P_band/P_total], a p-value, or a claim of market stationarity.
    Demeaning, periodic Hann, one-sided doubling and Nyquist are included.
    """
    w=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
    expected=np.maximum(0.,(w@w)-abs(np.fft.rfft(w))**2/n)/(w@w)
    expected[1:-1]*=2
    expected[0]=0
    return float(expected[np.asarray(selected,bool)].sum()/expected.sum())


def activity(c):
    buy,sell,unknown=[c.delta('c_trade_'+s) for s in ('buy','sell','unknown')]
    total=buy+sell+unknown
    valid=c.edge & (buy>=0)&(sell>=0)&(unknown>=0)
    for n in (30,128):
        roll=lambda x:pd.Series(x).rolling(n,min_periods=n).sum().to_numpy()
        q,u,s=roll(total),roll(unknown),roll(buy-sell)
        good=rolling_valid(valid,n)&(q>0)
        divide=lambda a:np.divide(a,q,out=np.zeros(len(q)),where=q>0)
        for name,value in (('observed_imbalance',divide(s)),('unknown_fraction',divide(u)),
                           ('signed_lower_bound',divide(s-u)),('signed_upper_bound',divide(s+u))):
            c.add(f'activity_{n}s_'+name,value,good,'ratio')
    for channel,x in (('total_trade',total),('observed_signed_trade',buy-sell),('unknown_trade',unknown)):
        for n in (32,64,128):
            good=rolling_valid(valid,n)
            freq=np.fft.rfftfreq(n)
            bands=[(a,b) for a,b in ((4,8),(8,16),(16,32)) if n>=4*b]
            fields={name:np.zeros(len(x)) for name in ('mean','power','entropy')}
            powers={band:np.zeros(len(x)) for band in bands}
            if len(x)>=n:
                windows=np.lib.stride_tricks.sliding_window_view(np.nan_to_num(x),n)
                w=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
                for lo in range(0,len(windows),4096):
                    z=windows[lo:lo+4096];mean=z.mean(axis=1)
                    p=abs(np.fft.rfft((z-mean[:,None])*w,axis=1))**2/(w@w)
                    p[:,1:-1]*=2
                    denom=p[:,1:].sum(axis=1)
                    q=np.divide(p[:,1:],denom[:,None],out=np.zeros_like(p[:,1:]),where=denom[:,None]>0)
                    dest=slice(lo+n-1,lo+n-1+len(z))
                    fields['mean'][dest]=mean;fields['power'][dest]=denom/n
                    fields['entropy'][dest]=-np.sum(q*np.log(np.maximum(q,1e-300)),axis=1)/np.log(n//2)
                    for a,b in bands:
                        selected=(freq>=1/b)&((freq<=1/a) if a==4 else (freq<1/a))
                        powers[a,b][dest]=p[:,selected].sum(axis=1)/n
            stem=f'fft_{channel}_{n}s'
            for name,v in fields.items():
                unit='ratio' if name=='entropy' else ('contract' if name=='mean' else 'contract squared')
                c.add(stem+'_'+name,v,good & ((fields['power']>0) if name=='entropy' else True),unit)
            for (a,b),p in powers.items():
                selected=(freq>=1/b)&((freq<=1/a) if a==4 else (freq<1/a))
                reference=white_reference(n,selected)
                fraction=np.divide(p,fields['power'],out=np.zeros(len(x)),where=fields['power']>0)
                known=good&(fields['power']>0)
                c.add(stem+f'_period{a}_{b}_fraction',fraction,known,'ratio')
                c.add(stem+f'_period{a}_{b}_relative_white_power',fraction/reference,known,'ratio')


PLUGIN=FeaturePlugin('spectral_activity','1',activity,128)
