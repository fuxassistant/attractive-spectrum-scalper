"""Full venue-session observations with native BBO and per-family masks."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from spectrum.features.contract import reject_future
from spectrum.representation.representation import (NS,MINUTE,DAY_VALUES,MarketObservationStore,
                                                     anchored_context,scheduled_open)


def session_bounds(date):
    day=pd.Timestamp(date)
    previous=(day-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return (int(pd.Timestamp(previous+' 18:00',tz='America/New_York').value),
            int(pd.Timestamp(date+' 17:00',tz='America/New_York').value))


def session_context(minutes,trades,date,previous_date,*,minimum_coverage=.99):
    """Explicit current/previous CME dates; missing rows never inflate coverage.

    ``minimum_coverage`` is a declared admission threshold, not imputation.
    Prior-session anchors are published at the new session start even before
    the first new minute. Calendar-only placeholder rows remain price-unknown.
    """
    current,end=session_bounds(date);previous,_=session_bounds(previous_date)
    index=np.arange(previous+MINUTE,end+MINUTE,MINUTE,dtype=np.int64)
    source=minutes.set_index('t_ns')
    publication=source.available_ns.reindex(index,fill_value=0).to_numpy(np.int64)
    m=source.reindex(index).rename_axis('t_ns').reset_index()
    missing=m.available_ns.isna()
    m['known']=m.known.eq(True)
    m['scheduled_open']=scheduled_open(m.t_ns)
    m['interval_open']=scheduled_open(m.t_ns-1)
    # Clock placeholders state unknown, not quiet-market observations.
    m['available_ns']=np.where(missing|~m.interval_open,m.t_ns.to_numpy(np.int64),publication)
    m['available_ns']=np.maximum.accumulate(m.available_ns)
    for name in ('volume','pv','p2v','buy','sell','unknown','trade_delta'):
        m[name]=m[name].fillna(0)
    result=anchored_context(m,trades,date,previous_date,current_anchor_ns=current,
                            previous_anchor_ns=previous,end_ns=end)
    prior=result[result.t_ns<=current]
    expected=scheduled_open(prior.t_ns-1)
    coverage=float(prior.known[expected].sum()/max(1,expected.sum()))
    prior_known=prior[prior.known & expected]
    release=max(current,int(prior_known.available_ns.max())) if len(prior_known) else current
    admitted=bool(len(prior_known) and coverage>=minimum_coverage)
    previous_values=dict(high=prior_known.high.max(),low=prior_known.low.min(),
                         vwap=prior.combined_vwap.iloc[-1],poc=prior.combined_poc.iloc[-1])
    for name,value in previous_values.items():
        key='previous_'+name
        known=(result.available_ns>=release)&admitted&np.isfinite(value)
        result[key]=np.where(known,value,np.nan)
        result[key+'__known']=known
    result['previous_recognized_ns']=np.where((result.available_ns>=release)&admitted,release,0)
    result['previous_session_coverage']=np.where(result.available_ns>=release,coverage,np.nan)
    # Context geometry is only as complete as its observed source prefix.
    for label in ('day','combined'):
        for field in ('vwap','sigma','high','low','poc'):
            name=label+'_'+field
            result[name+'__known']=result.known & (result[label+'_coverage']>=minimum_coverage) & result[name].notna()
    result.attrs['context_contract']=dict(version='CME_SESSION_CONTEXT_V1',date=date,previous_date=previous_date,
        start_ns=current,end_ns=end,previous_start_ns=previous,minimum_coverage=minimum_coverage,
        previous_coverage=coverage,previous_admitted=admitted)
    return result


@dataclass
class SessionObservation:
    t_ns:int
    market:dict
    fast:dict
    context:dict
    second_features:dict
    minute_features:dict
    zones:pd.DataFrame
    events:pd.DataFrame
    receipt:dict


def latest_features(frame,t_ns,*,mask_suffix):
    """Select the most recent published row, retaining each feature's mask."""
    names=[n for n in frame if n+mask_suffix in frame]
    i=int(np.searchsorted(frame.available_ns.to_numpy(np.int64),t_ns,side='right'))-1
    if i<0:
        return dict(names=names,values=np.zeros(len(names)),known=np.zeros(len(names),bool),t_ns=None,available_ns=None)
    row=frame.iloc[i]
    endpoint=int(frame.t_ns.iloc[i]);publication=int(frame.available_ns.iloc[i])
    if endpoint>t_ns:
        raise ValueError('Feature endpoint is in the future')
    known=np.array([bool(row[n+mask_suffix]) for n in names])
    values=np.array([float(row[n]) for n in names])
    known &= np.isfinite(values)
    return dict(names=names,values=np.where(known,values,0),known=known,
                t_ns=endpoint,available_ns=publication,age_seconds=(t_ns-endpoint)/NS)


class SessionObservationStore(MarketObservationStore):
    """As-of access throughout a declared session; exact BBO gates market state.

    Quotes are canonical complete publications, including invalid books. Never
    search backward past an invalid publication for an older tradable quote.
    """
    def __init__(self,fast,context,zones,events,*,quotes,features,core_features,symbol,date,previous_date):
        super().__init__(fast,context,zones,events,symbol=symbol,date=date,previous_date=previous_date)
        for frame in (quotes,features,core_features):
            reject_future(frame.columns,frame)
            if not frame.available_ns.is_monotonic_increasing or (frame.available_ns<frame.t_ns).any():
                raise ValueError('Publication clocks must be ordered and causal')
        self.quotes,self.features,self.core_features=quotes,features,core_features
        self.start,self.end=session_bounds(date)
        self.previous_start,_=session_bounds(previous_date)

    def observe(self,t_ns):
        t_ns=int(t_ns)
        if not self.start<=t_ns<=self.end:
            raise ValueError('Outside declared venue session')
        base=self._observe(t_ns,self.previous_start)
        i=int(np.searchsorted(self.quotes.available_ns.to_numpy(np.int64),t_ns,side='right'))-1
        opened=bool(scheduled_open(np.array([t_ns],np.int64))[0])
        if i<0:
            market=dict(known=False,decision_open=opened,reason='NO_PUBLISHED_QUOTE')
        else:
            q=self.quotes.iloc[i]
            age=t_ns-int(q.t_ns)
            good=bool(q.valid and opened and 0<=age<=30*NS)
            market=dict(t_ns=int(q.t_ns),available_ns=int(q.available_ns),source_ordinal=int(q.source_ordinal),
                epoch=int(q.epoch),bid=float(q.bid_nano)/NS,ask=float(q.ask_nano)/NS,
                bid_size=int(q.bid_size),ask_size=int(q.ask_size),fact_age_ns=age,known=good,decision_open=opened,
                reason='OK' if good else ('CLOSED' if not opened else ('INVALID_BOOK' if not q.valid else 'STALE_QUOTE')))
        second=latest_features(self.features,t_ns,mask_suffix='__known')
        minute=latest_features(self.core_features,t_ns,mask_suffix='_known')
        if market['known']:
            mid=(market['bid']+market['ask'])/2
            scale=base.fast['normalization']['reference_scale_dollars']
            self._normalize(base.fast,mid,scale);self._normalize(base.day,mid,scale)
        eligible=self.context[self.context.available_ns<=t_ns]
        quality_names=['day_coverage','combined_coverage','previous_session_coverage']
        quality=np.array([float(eligible[n].iloc[-1]) if len(eligible) else np.nan for n in quality_names])
        base.day['quality']=dict(names=quality_names,values=np.nan_to_num(quality),known=np.isfinite(quality),
            previous_session_complete=bool(np.isfinite(quality[-1]) and quality[-1]>=1.),
            level_scope='Observed supported minutes; 99% admission is not a claim of complete prior-session extrema')
        receipt=dict(base.receipt,contract='CME_SESSION_OBSERVATION_V1',start_ns=self.start,end_ns=self.end,
                     native_quote_known=market['known'],complete_case_filter=False,
                     second_known=int(second['known'].sum()),second_channels=len(second['names']))
        return SessionObservation(t_ns,market,base.fast,base.day,second,minute,base.zones,base.events,receipt)
