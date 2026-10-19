"""Independent vector accounting of all exposed quotes, plus frozen stresses."""
from pathlib import Path
import argparse
import numpy as np
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from .data import MonthDay
from .basket import BasketConfig,signals,replay_day,initial_state,summary
from .runner import DEFAULT_DATA
from .basket_search import DEFAULT_OUTPUT


def audit_record(data,path):
    r=read_json(path);cash=1500.;qty=0.;cost=0.;side=0;base=None;peak=1500.;dd=0.;loss=0.;fees=0.;quotes=0;checks=0
    def observe(values):
        nonlocal peak,dd,loss,quotes
        if not len(values):return
        peaks=np.maximum.accumulate(np.r_[peak,values])[1:]
        dd=max(dd,float(np.max(peaks-values)));peak=max(peak,float(np.max(values)))
        if base is not None:loss=max(loss,max(0.,(base-float(np.min(values)))/base))
        quotes+=len(values)
    for daily in r['days']:
        date=daily['date'];d=MonthDay(Path(data)/date);previous=0;book_epoch=None
        for e in [x for x in r['events'] if x['date']==date]:
            j=int(e['quote_index']);kind=int(e['kind']);sg=int(e['side'])
            if qty:
                for lo in range(previous,j+1,100000):
                    hi=min(j+1,lo+100000)
                    if not np.all(d.valid[lo:hi]) or not np.all(d.epoch[lo:hi]==book_epoch):raise AssertionError('Unacknowledged support loss')
                    marks=np.asarray(d.bid[lo:hi] if side>0 else d.ask[lo:hi],float)/1e9
                    observe(cash+side*(qty*marks-cost))
            if kind==7:raise AssertionError('Censored record is not an accepted economic result')
            px=int((d.ask if (kind in (1,2))==(sg>0) else d.bid)[j])/1e9
            assert abs(px-e['price'])<1e-8 and int(d.quote_t[j])==int(e['t_ns'])
            if kind in (1,2):
                if not qty:
                    side=sg;book_epoch=int(d.epoch[j]);basket_base=cash
                    if base is None:base=cash
                assert side==sg
                amount=e['quantity'];cash-=.30*amount;fees+=.30*amount;cost+=amount*px;qty+=amount
                assert qty<=r['config']['cap_oz']+1e-8
            else:
                assert abs(qty-e['quantity'])<1e-8
                cash+=side*(qty*px-cost);qty=0.;cost=0.
                assert abs((cash-basket_base)-e['basket_net'])<1e-7
            mark=int((d.bid if side>0 else d.ask)[j])/1e9
            eq=cash+side*(qty*mark-cost)
            observe(np.array([eq]))
            assert abs(cash-e['cash'])<1e-7 and abs(eq-e['equity'])<1e-7 and abs(base-e['series_base'])<1e-7
            if not qty and cash>=base-1e-8:base=None
            previous=j+1;checks+=1
        assert qty==0 and abs(cash-daily['cash'])<1e-7
    assert abs(cash-r['result']['equity'])<1e-7
    assert abs(dd-r['result']['max_drawdown'])<1e-7
    assert abs(loss-r['result']['max_series_loss'])<1e-7
    assert abs(fees-r['result']['fees'])<1e-7
    return dict(status='PASS',events=checks,exposed_quote_valuations=quotes,final_equity=cash,fees=fees,
                max_drawdown=dd,max_series_loss=loss,unrecovered_series_base=base)


def stress(data,dates,config,latency,fee):
    s=initial_state()
    for date in dates:
        if s[8] or s[9]:break
        d=MonthDay(Path(data)/date);side,scale=signals(d,config)
        s=replay_day(np.asarray(d.quote_t),np.asarray(d.bid),np.asarray(d.ask),np.asarray(d.valid),np.asarray(d.epoch),
            np.asarray(d.decision_t),side,scale,config.array(),s,int(d.meta['london_ns']),int(d.meta['ny_close_ns']),d.deadline,
            False,latency,fee)[0]
    return dict(latency_ms=latency/1e6,fee_per_oz=fee,**summary(s))


def run(data=DEFAULT_DATA,output=DEFAULT_OUTPUT):
    root=Path(output);freeze=read_json(root/'frozen.json');config=BasketConfig(**freeze['config'])
    audits={name:audit_record(data,root/name/'result.json') for name in ('month','holdout','same_grid_fixed_adds','same_signal_no_adds')}
    manifest=read_json(Path(data)/'manifest.json');dates=[d['date'] for d in manifest['days']]
    stresses=[stress(data,dates,config,lat,fee) for lat,fee in [(500_000_000,.3),(100_000_000,.6),(500_000_000,.6)]]
    result=dict(status='PASS',accounting=audits,stresses=stresses,selection='Frozen configuration; stress results are not used for reselection')
    save_json(root/'audit.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT)
    a=p.parse_args();run(a.data,a.output)
