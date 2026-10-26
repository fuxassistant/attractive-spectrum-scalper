"""Reusable formulas extracted from spectrum/lab.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.features.contract import rolling_valid

def spectral(c):
    channels={'mid_change':c.delta('mid')/.1,
              'signed_trade':c.delta('c_trade_buy')-c.delta('c_trade_sell'),
              'bid_add':c.delta('c_add_bid'),'ask_add':c.delta('c_add_ask'),
              'bid_remove':c.delta('c_remove_bid'),'ask_remove':c.delta('c_remove_ask')}
    for channel,x in channels.items():
        supported=c.edge.copy()
        if channel=='signed_trade':supported&=c.delta('c_trade_unknown')==0
        unit='tick' if channel=='mid_change' else 'contract'
        for n in (32,64,128):
            good=rolling_valid(supported,n);power=np.zeros(len(x));entropy=np.zeros(len(x));mean=np.zeros(len(x))
            bands=[(lo,hi) for lo,hi in ((4,8),(8,16),(16,32)) if n>=4*hi]
            powers={band:np.zeros(len(x)) for band in bands}
            if len(x)>=n:
                windows=np.lib.stride_tricks.sliding_window_view(np.nan_to_num(x),n)
                w=.5-.5*np.cos(2*np.pi*np.arange(n)/n);freq=np.fft.rfftfreq(n)
                for lo in range(0,len(windows),4096):
                    z=windows[lo:lo+4096];m=z.mean(axis=1)
                    ft=np.fft.rfft((z-m[:,None])*w,axis=1)
                    psd=abs(ft)**2/(w@w);psd[:,1:-1]*=2
                    dest=slice(lo+n-1,lo+n-1+len(z));total=psd[:,1:].sum(axis=1)/n
                    power[dest]=total;mean[dest]=m
                    q=np.divide(psd[:,1:],psd[:,1:].sum(axis=1)[:,None],out=np.zeros_like(psd[:,1:]),where=total[:,None]>0)
                    entropy[dest]=-np.sum(q*np.log(np.maximum(q,1e-300)),axis=1)/np.log(n//2)
                    for a,b in bands:
                        mask=(freq>=1/b)&((freq<=1/a) if a==4 else (freq<1/a))
                        powers[a,b][dest]=psd[:,mask].sum(axis=1)/n
            stem=f'fft_{channel}_{n}s'
            c.add(stem+'_mean',mean,good,unit);c.add(stem+'_power',power,good,unit+' squared')
            c.add(stem+'_entropy',entropy,good&(power>0),'ratio')
            for (a,b),p in powers.items():
                c.add(stem+f'_period{a}_{b}_power',p,good,unit+' squared')
                c.add(stem+f'_period{a}_{b}_fraction',np.divide(p,power,out=np.zeros_like(p),where=power>0),good&(power>0),'ratio')
