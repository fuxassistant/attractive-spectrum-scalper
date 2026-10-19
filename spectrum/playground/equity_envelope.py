"""Exact minute ranges reconstructed from fills and every exposed quote.

Reporting only, never a model input. Balance reserves roundtrip commission at
entry. Equity values an open position at its liquidation bid/ask. No midpoint
substitute, no interpolation through censored support, no end-of-minute extrema.
"""
from pathlib import Path
import numpy as np
from numba import njit
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from .data import MonthDay

MINUTE=60*10**9


@njit(cache=True)
def mark_range(t,px,lo,hi,cash,side,qty,cost,start,mn,mx,close,count):
    for j in range(lo,hi):
        k=(t[j]-start)//MINUTE
        if 0<=k<len(mn):
            eq=cash+side*(qty*(px[j]/1e9)-cost)
            mn[k]=min(mn[k],eq);mx[k]=max(mx[k],eq);close[k]=eq;count[k]+=1


def envelope(day,events,start_balance,terminal_censored=False):
    start=int(day.meta['start_ns']);end=int(day.meta['end_ns']);n=(end-start+MINUTE-1)//MINUTE
    bmin=np.full(n,np.inf);bmax=np.full(n,-np.inf);bclose=np.full(n,np.nan)
    emin=np.full(n,np.inf);emax=np.full(n,-np.inf);eclose=np.full(n,np.nan);count=np.zeros(n,np.int64)
    cash=float(start_balance);qty=0.;cost=0.;side=0;last_time=start;previous=0;censored=False;cut=end
    def span(a,b,balance,flat):
        lo=max(0,(a-start)//MINUTE);hi=min(n-1,(b-start)//MINUTE)
        if hi<lo:return
        z=slice(lo,hi+1);bmin[z]=np.minimum(bmin[z],balance);bmax[z]=np.maximum(bmax[z],balance);bclose[z]=balance
        if flat:emin[z]=np.minimum(emin[z],balance);emax[z]=np.maximum(emax[z],balance);eclose[z]=balance
    def point(now,equity):
        k=(now-start)//MINUTE
        if 0<=k<n:
            emin[k]=min(emin[k],equity);emax[k]=max(emax[k],equity);eclose[k]=equity
            bmin[k]=min(bmin[k],cash);bmax[k]=max(bmax[k],cash);bclose[k]=cash
    qt=np.asarray(day.quote_t)
    for event in events:
        j=int(event['quote_index']);now=int(event['t_ns']);kind=int(event['kind'])
        span(last_time,now,cash,qty==0)
        if qty:
            stop=j if kind==7 else j+1
            if not np.all(day.valid[previous:stop]):raise ValueError('Unsupported exposed quote in envelope')
            mark_range(qt,np.asarray(day.bid if side>0 else day.ask),previous,stop,cash,side,qty,cost,start,emin,emax,eclose,count)
        if kind==7:censored=True;cut=now;break
        if kind in (1,2):
            side=int(event['side']);qty+=event['quantity'];cost+=event['quantity']*event['price'];cash=float(event['cash'])
        else:
            cash=float(event['cash']);qty=0.;cost=0.
        mark=int((day.bid if side>0 else day.ask)[j])/1e9
        eq=cash+side*(qty*mark-cost)
        if abs(eq-event['equity'])>1e-7:raise ValueError('Envelope event accounting mismatch')
        point(now,eq);last_time=now;previous=j+1
    if not censored:
        if qty:
            if not terminal_censored:raise ValueError('Open terminal basket requires a censored terminal state')
            # End-of-tape censoring has no executable event. Include remaining
            # observed marks, then stop immediately after the last publication.
            if not np.all(day.valid[previous:]):raise ValueError('Unsupported exposed terminal quote')
            cut=min(end,int(qt[-1])+1);censored=True
            span(last_time,cut-1,cash,False)
            mark_range(qt,np.asarray(day.bid if side>0 else day.ask),previous,len(qt),cash,side,qty,cost,start,emin,emax,eclose,count)
        else:span(last_time,end-1,cash,True)
    def nullable(v):return float(v) if np.isfinite(v) else None
    rows=[]
    for k in range(n):
        if start+k*MINUTE>=cut:break
        rows.append(dict(t_ns=str(start+k*MINUTE),end_ns=str(min(start+(k+1)*MINUTE,end)),
            balance_min=nullable(bmin[k]),balance_max=nullable(bmax[k]),balance_close=nullable(bclose[k]),
            equity_min=nullable(emin[k]),equity_max=nullable(emax[k]),equity_close=nullable(eclose[k]),
            exposed_quote_count=int(count[k]),complete=start+(k+1)*MINUTE<=cut))
    finite=lambda key:[row[key] for row in rows if row[key] is not None]
    return dict(rows=rows,summary=dict(balance_min=min(finite('balance_min')),balance_max=max(finite('balance_max')),
        equity_min=min(finite('equity_min')),equity_max=max(finite('equity_max')),balance_close=cash if not censored else None,
        exposed_quote_count=int(count.sum()),censored=censored,observed_until_ns=str(cut),
        unrealized_min=min((row['equity_min']-row['balance_max'] for row in rows if row['equity_min'] is not None and row['balance_max'] is not None),default=0.),
        unrealized_note='Conservative within-bin equity_min minus balance_max; not an exact simultaneous unrealized PnL'))


def attach(data,result_path):
    path=Path(result_path);record=read_json(path);cash=1500.;days=[]
    for row in record['days']:
        date=row['date'];d=MonthDay(Path(data)/date)
        daily=envelope(d,[e for e in record['events'] if e['date']==date],cash,bool(row['censored']))
        days.append(dict(date=date,**daily));cash=row['cash']
    output=dict(version='BALANCE_EQUITY_MINMAX_V1',interval_seconds=60,days=days,
        definition='Balance includes realized PnL and reserved roundtrip fees. Equity uses liquidation bid/ask at every exposed publication, including pre/post fill states. Unknown tails are not forward-filled.',
        reporting_only=True)
    save_json(path.parent/'equity_envelope.json',output);return output
