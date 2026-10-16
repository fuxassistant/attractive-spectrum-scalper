"""Versioned feature plugins run on one verified second tape per capture.

No new order book, no future labels. Unknown values have false masks.
"""
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time,os,importlib,inspect
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from .calendar import MarketCalendar
from .core.storage import sha
from .ingest import save_json


def rolling_valid(valid,n):
    prefix=np.r_[0,np.cumsum(np.asarray(valid,dtype=np.int64))]
    out=np.zeros(len(valid),bool)
    if len(out)>=n:out[n-1:]=prefix[n:]-prefix[:-n]==n
    return out


class Context:
    def __init__(self,frame,calendar):
        self.df=frame;self.t=frame.t_ns.to_numpy();self.closed=calendar.closed(self.t)
        self.supported=frame.valid.to_numpy()&~self.closed
        self.admitted=(self.supported&
                       ((frame.available_ns-frame.t_ns).to_numpy()<=10**9))
        self.same=np.r_[False,(np.diff(self.t)==10**9)&(np.diff(frame.epoch.to_numpy())==0)]
        # A late publication can prove supported PAST without being a fresh
        # decision origin. Once known, that past remains usable by later windows.
        self.edge=self.supported&np.r_[False,self.supported[:-1]]&self.same
        self.values={};self.masks={};self.units={}

    def add(self,name,values,known,unit):
        if name in self.values:raise ValueError('Duplicate feature '+name)
        v=np.asarray(values,float);mask=np.asarray(known,bool)&np.isfinite(v)&self.admitted
        if v.shape!=self.t.shape or mask.shape!=self.t.shape:raise ValueError('Feature shape')
        self.values[name]=np.where(mask,v,0.);self.masks[name]=mask;self.units[name]=unit

    def delta(self,column):
        v=self.df[column].to_numpy(float)
        return np.r_[0.,np.diff(v)]


@dataclass(frozen=True)
class FeaturePlugin:
    name: str
    version: str
    compute: object
    lookback_seconds: int


def motion(c):
    f=c.df;mid=f.mid.to_numpy();bid=f.bid.to_numpy();ask=f.ask.to_numpy()
    c.add('spread_ticks',(ask-bid)/.1,c.admitted,'tick')
    for n in (1,5,15,30,60,128):
        good=rolling_valid(c.edge,n)
        previous=np.r_[np.full(n,np.nan),mid[:-n]] if len(mid)>n else np.full(len(mid),np.nan)
        change=(mid-previous)/.1
        c.add(f'displacement_{n}s',change,good,'tick')
        delta=c.delta('c_path_ticks')
        path=pd.Series(delta).rolling(n,min_periods=n).sum().to_numpy()
        c.add(f'path_{n}s',path,good,'tick')
        c.add(f'efficiency_{n}s',np.divide(abs(change),path,out=np.zeros(len(mid)),where=path>0),good&(path>0),'ratio')
    denom=(f.bid_size+f.ask_size).to_numpy()
    c.add('imbalance_l1',np.divide((f.bid_size-f.ask_size).to_numpy(),denom,out=np.zeros(len(f)),where=denom>0),c.admitted&(denom>0),'ratio')
    for side in ('bid','ask'):
        for n in (1,4,12):c.add(f'depth_{side}_{n}',f[f'depth_{side}_{n}'],c.admitted,'contract')


def flow(c):
    for col in ('trade_buy','trade_sell','trade_unknown','add_bid','add_ask','remove_bid','remove_ask','matched_fill_remove_bid','matched_fill_remove_ask','event_count'):
        d=c.delta('c_'+col)
        for n in (1,5,30):
            v=pd.Series(d).rolling(n,min_periods=n).sum().to_numpy()
            c.add(f'{col}_{n}s',v,rolling_valid(c.edge,n)&(v>=0),'count' if col=='event_count' else 'contract')


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


REGISTRY={p.name:p for p in (FeaturePlugin('motion','1',motion,128),FeaturePlugin('flow','1',flow,30),FeaturePlugin('spectral','1',spectral,128))}

def register(plugin):
    if plugin.name in REGISTRY:raise ValueError('Plugin already registered')
    REGISTRY[plugin.name]=plugin

def load_plugins(specs):
    identities={}
    for spec in specs:
        module,attribute=spec.split(':',1)
        loaded=importlib.import_module(module);plugin=getattr(loaded,attribute)
        if not isinstance(plugin,FeaturePlugin):raise TypeError('Expected FeaturePlugin')
        if plugin.name not in REGISTRY:register(plugin)
        elif REGISTRY[plugin.name]!=plugin:raise ValueError('Plugin name collision')
        identities[spec]=dict(name=plugin.name,version=plugin.version,sha256=sha(inspect.getsourcefile(loaded)))
    return identities

def compute(frame,families=('motion','flow','spectral'),calendar=None):
    c=Context(frame,calendar or MarketCalendar())
    for name in families:REGISTRY[name].compute(c)
    return c

def generate_capture(capture,output,families,plugins=()):
    plugin_sources=load_plugins(plugins)
    start=time.perf_counter();capture=Path(capture);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((capture/'manifest.json').read_text())
    if m['status']!='COMPLETE' or sha(capture/'seconds.parquet')!=m['payloads']['seconds.parquet']:raise ValueError('Market payload drift')
    frame=pq.read_table(capture/'seconds.parquet').to_pandas();cal=MarketCalendar();c=compute(frame,families,cal)
    columns={k:frame[k].to_numpy() for k in ('t_ns','available_ns','epoch')}
    columns.update(scheduled_closed=c.closed,admissible=c.admitted)
    for name,v in c.values.items():columns[name]=v;columns[name+'__known']=c.masks[name]
    table=pa.table(columns);pq.write_table(table,output/'features.parquet',compression='zstd',row_group_size=32768)
    read=pq.read_table(output/'features.parquet')
    if not read.equals(table):raise ValueError('Parquet roundtrip differs')
    r=dict(status='COMPLETE',rows=len(frame),features=len(c.values),families={n:REGISTRY[n].version for n in families},units=c.units,
           lookback_seconds=max(REGISTRY[n].lookback_seconds for n in families),calendar=cal.contract(),
           market_manifest_sha256=sha(capture/'manifest.json'),source_seconds_sha256=m['payloads']['seconds.parquet'],
           implementation_sha256=sha(__file__),plugin_sources=plugin_sources,calendar_implementation_sha256=sha(Path(__file__).with_name('calendar.py')),
           feature_sha256=sha(output/'features.parquet'),coverage={n:float(mask.mean()) for n,mask in c.masks.items()},
           elapsed_seconds=time.perf_counter()-start,rss_sample=psutil.Process().memory_info().rss,pid=os.getpid(),
           known='false-mask values are technical zero, not observed inactivity',roundtrip='PASS')
    save_json(output/'manifest.json',r);return r

def generate(market,output,families,workers=4,plugins=()):
    load_plugins(plugins)
    if not families or len(set(families))!=len(families) or any(n not in REGISTRY for n in families):raise ValueError('Unknown/duplicate feature family')
    market=Path(market);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=json.loads((market/'manifest.json').read_text());start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(generate_capture,market/Path(e['input']['name']).stem,output/Path(e['input']['name']).stem,families,plugins):e for e in m['captures']}
        for f in as_completed(jobs):
            r=f.result();r['capture']=Path(jobs[f]['input']['name']).stem;results.append(r)
            print(json.dumps(dict(capture=r['capture'],seconds=r['elapsed_seconds'],rows=r['rows'])),flush=True)
    results.sort(key=lambda r:r['capture'])
    result=dict(status='COMPLETE',captures=results,elapsed_seconds=time.perf_counter()-start,workers=workers,families=families,
                market_manifest_sha256=sha(market/'manifest.json'),independent_jobs='isolated process per complete reset capture, no shared book')
    save_json(output/'manifest.json',result);return result
