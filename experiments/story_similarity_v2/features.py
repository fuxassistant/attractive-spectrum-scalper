"""Separate shape, amplitude, texture, elapsed time and observable book context."""
import numpy as np

from experiments.scalper_adapter_v1.tokens import NS, TICK, quote_arrays
from experiments.scalper_adapter_v1.measures import FLOW_FIELDS

BOOK_NAMES = ['imbalance_low','imbalance_high','unknown_trade_fraction','log_trade_rate',
              'log_bid_fill_rate','log_ask_fill_rate','log_bid_withdraw_rate','log_ask_withdraw_rate',
              'log_bid_refill_rate','log_ask_refill_rate','log_bid_add_rate','log_ask_add_rate',
              'window_response_in_amplitudes']


def sum_windows(prefix, left, right):
    return prefix[right+1]-prefix[left]


class PublishedBook:
    """Vectorized complete 1s buckets; release strictly BEFORE token clock.

    Strict comparison conservatively avoids borrowing a bucket released by a
    later native publication with the same nanosecond timestamp as a token.
    These fixed contextual windows are not claimed to equal the story interval.
    """
    def __init__(self, fast):
        self.fast = fast
        self.t = fast.t_ns.to_numpy(np.int64)
        self.av = fast.available_ns.to_numpy(np.int64)
        assert fast.t_ns.is_monotonic_increasing and fast.available_ns.is_monotonic_increasing
        self.prefix = {n:np.r_[0.,np.cumsum(fast[n].to_numpy(float))] for n in FLOW_FIELDS}
        self.trade_bad = np.r_[0,np.cumsum(~fast.trade_known.to_numpy(bool))]
        self.book_bad = np.r_[0,np.cumsum(~fast.ledger_known.to_numpy(bool))]
        self.price_bad = np.r_[0,np.cumsum(~np.isfinite(fast.ret1_ticks.to_numpy(float)))]
        self.ret = np.r_[0.,np.cumsum(np.nan_to_num(fast.ret1_ticks.to_numpy(float)))]
        epoch=fast.epoch.to_numpy(np.int64)
        self.breaks = np.r_[0,np.cumsum(np.r_[0,(np.diff(self.t)!=NS)|(np.diff(epoch)!=0)])]

    def measure(self, tokens):
        n=len(tokens); clock=tokens.available_ns.to_numpy(np.int64)
        seconds=tokens.flow_seconds.to_numpy(np.int64)
        j=np.searchsorted(self.av,clock,side='left')-1
        safe=np.maximum(0,j); endpoint=self.t[safe]
        i=np.searchsorted(self.t,endpoint-seconds*NS,side='right')
        complete=(j>=0)&(safe-i+1==seconds)&(self.breaks[safe+1]-self.breaks[i+1]==0)
        age=(clock-endpoint)/NS
        fresh=(age>=0)&(age<=seconds)
        trade=complete&fresh&(sum_windows(self.trade_bad,i,safe)==0)
        book=complete&fresh&(sum_windows(self.book_bad,i,safe)==0)
        price=complete&fresh&(sum_windows(self.price_bad,i,safe)==0)
        sums=np.column_stack([sum_windows(self.prefix[name],i,safe) for name in FLOW_FIELDS])
        b,s,u=sums[:,:3].T; total=b+s+u
        values=np.zeros((n,len(BOOK_NAMES)))
        known=np.zeros_like(values,dtype=bool)
        positive=trade&(total>0)
        for col,v in enumerate((b-s-u,b-s+u,u)):
            values[:,col]=np.divide(v,total,out=np.zeros(n),where=total>0)
            known[:,col]=positive
        values[:,3]=np.log1p(total/seconds)/np.log(21.)
        known[:,3]=trade
        values[:,4:12]=np.log1p(sums[:,3:]/seconds[:,None])/np.log(11.)
        known[:,4:12]=book[:,None]
        values[:,12]=sum_windows(self.ret,i,safe)/tokens.amplitude_ticks.to_numpy(float)
        known[:,12]=price
        return dict(book=np.where(known,values,0.).astype(np.float32),book_known=known,
                    book_endpoint_ns=np.where(j>=0,endpoint,-1),
                    book_release_ns=np.where(j>=0,self.av[safe],-1),book_age_seconds=age)


def extract(tokens, quotes, book=None, samples=16):
    """All sample clocks are integer fractions of an already observed prefix."""
    if not isinstance(samples,int) or samples<4 or samples%4:
        raise ValueError('Sample count must be a positive multiple of four')
    qt,p,_,_=quote_arrays(quotes)
    si=tokens.start_index.to_numpy(np.int64); ei=tokens.source_index.to_numpy(np.int64)
    start=tokens.start_ns.to_numpy(np.int64); end=tokens.available_ns.to_numpy(np.int64)
    dt=end-start
    anchor=tokens.start_mid_nano.to_numpy(np.int64)
    amp=tokens.amplitude_ticks.to_numpy(float)*TICK
    low=tokens.low_nano.to_numpy(np.int64); high=tokens.high_nano.to_numpy(np.int64)
    span=(high-low)/amp; net=(p[ei]-anchor)/amp
    clocks=start[:,None]+dt[:,None]*np.arange(1,samples+1,dtype=np.int64)[None,:]//samples
    ix=np.searchsorted(qt,clocks,side='right')-1
    ix=np.minimum(ei[:,None],np.maximum(si[:,None],ix))
    raw_shape=(p[ix]-anchor[:,None])/amp[:,None]
    shape=raw_shape/np.maximum(.5,span)[:,None]
    path=tokens.path_nano.to_numpy(float)/amp
    efficiency=np.divide(net,path,out=np.zeros(len(tokens)),where=path>0)
    prefix=np.r_[0,np.cumsum(np.abs(np.diff(p)),dtype=np.int64)]
    quarters=ix[:,np.array([samples//4-1,samples//2-1,3*samples//4-1,samples-1])]
    cumulative=prefix[quarters]-prefix[si,None]
    chunks=np.diff(np.column_stack([np.zeros(len(tokens)),cumulative]),axis=1)
    shares=np.divide(chunks,tokens.path_nano.to_numpy(float)[:,None],out=np.zeros_like(chunks),where=tokens.path_nano.to_numpy(float)[:,None]>0)
    amplitude=np.column_stack([net,span,(high-anchor)/amp,(anchor-low)/amp,
                               (high-p[ei])/amp,(p[ei]-low)/amp])
    texture=np.column_stack([efficiency,np.log1p(path/np.maximum(.5,span))/3,
                            np.log1p(tokens.turns.to_numpy(float))/6,shares])
    structure=np.column_stack([tokens.direction.to_numpy(np.int64),tokens.kind.eq('TURN_CONFIRMED').to_numpy(),
                               tokens.kind.eq('TIME_LIMIT').to_numpy()]).astype(np.int8)
    result=dict(shape=shape.astype(np.float32),amplitude=amplitude.astype(np.float32),
                texture=texture.astype(np.float32),duration=dt/NS,structure=structure,
                raw_shape=raw_shape.astype(np.float32))
    if book is None:
        result.update(book=np.zeros((len(tokens),len(BOOK_NAMES)),np.float32),
                      book_known=np.zeros((len(tokens),len(BOOK_NAMES)),bool))
    else: result.update(book.measure(tokens))
    return result


def mirrored(pack):
    """Price reflection also swaps bid/ask and buy/sell evidence; no real-market symmetry claim."""
    result={k:v.copy() for k,v in pack.items()}
    result['shape'] *= -1
    if 'raw_shape' in result: result['raw_shape'] *= -1
    a=pack['amplitude']
    result['amplitude'][...,0]=-a[...,0]
    result['amplitude'][...,2:6]=a[...,[3,2,5,4]]
    result['texture'][...,0]*=-1
    result['structure'][...,0]*=-1
    b=pack['book']; m=pack['book_known']
    result['book'][...,0]=-b[...,1];result['book'][...,1]=-b[...,0]
    result['book_known'][...,0]=m[...,1];result['book_known'][...,1]=m[...,0]
    order=[5,4,7,6,9,8,11,10]
    result['book'][...,4:12]=b[...,order]
    result['book_known'][...,4:12]=m[...,order]
    result['book'][...,12]*=-1
    return result
