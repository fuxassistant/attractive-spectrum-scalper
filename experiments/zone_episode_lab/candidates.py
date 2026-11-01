"""Past-only zone selection, explicit last-eight-episode memory, no future PnL."""
import json
import numpy as np
import pandas as pd
from .episodes import NS,TICK
from .spectral import zone_spectrum


def history_stats(h,at,cfg):
    quality=cfg['zone_quality']
    if len(h)<quality['minimum_completed_visits']:return None
    first=int(h.arrival_ns.min());last=int(h.end_ns.max())
    if last-first<quality['minimum_history_span_seconds']*NS or at-last>quality['maximum_last_visit_age_seconds']*NS:return None
    passes=int(h.outcome.str.startswith('PASS').sum());rejects=int(h.outcome.str.startswith('REJECT').sum())
    dwell=float(h.core_dwell_seconds.sum());median=float(h.core_dwell_seconds.median())
    return dict(completed_visits=len(h),history_start_ns=first,zone_available_ns=last,
        past_passes=passes,past_rejections=rejects,permeability=passes/(passes+rejects),
        cumulative_dwell_seconds=dwell,median_dwell_seconds=median,age_since_departure_seconds=(at-last)/NS,
        sticky=dwell>=quality['minimum_dwell_seconds'],permeable=passes/(passes+rejects)>=.65 and median<5,
        history_episode_ids=json.dumps(h.episode_id.tail(8).astype(int).tolist()))


def make_candidates(quotes,features,events,episodes,grid,config,day_index,date):
    ready=features.loc[features.ready].sort_values('t_ns')
    qt=quotes.available_ns.to_numpy(np.int64);bid=quotes.bid_nano.to_numpy(np.int64);ask=quotes.ask_nano.to_numpy(np.int64)
    completed=episodes.loc[episodes.completed & episodes.entry_side.ne(0)].sort_values('end_ns')
    grouped={int(p):h.reset_index(drop=True) for p,h in completed.groupby('zone_low_nano')}
    ends={p:h.end_ns.to_numpy(np.int64) for p,h in grouped.items()}
    arrivals={int(p):h.available_ns.to_numpy(np.int64) for p,h in events.loc[events.event_type=='ARRIVAL'].groupby('zone_low_nano')}
    contacts={int(p):h.available_ns.to_numpy(np.int64) for p,h in events.loc[events.event_type.isin(['ARRIVAL','PRESENT_ON_RESTART'])].groupby('zone_low_nano')}
    bad=~quotes.valid.to_numpy(bool)|(quotes.epoch.to_numpy()<0)|(bid<=0)|(ask<=bid)
    qe=quotes.epoch.to_numpy(np.int64)
    breaks=np.sort(np.r_[qt[bad],qt[1:][qe[1:]!=qe[:-1]],qt[:-1][np.diff(qt)>int(config['geometry']['max_quote_age_seconds']*NS)]+int(config['geometry']['max_quote_age_seconds']*NS)])
    width=int(config['geometry']['bin_ticks']*TICK);rules=config['candidate'];sc=config['spectral']
    fclock=ready.t_ns.to_numpy(np.int64);rows=[];parents=[];tokens=[];used=set();last_by_zone={}
    parent_by_episode={}
    def origin(at,fr,side,target,low,episode_id,stats,inside,parent_id=-1,deadline=-1):
        j=int(np.searchsorted(qt,at,'right')-1)
        if j<0 or not bool(quotes.valid.iloc[j]) or quotes.epoch.iloc[j]<0:return None
        if (ask[j]-bid[j])/NS>rules['maximum_spread_per_oz'] or ask[j]<=bid[j] or bid[j]<=0:return None
        if at-qt[j]>int(config['geometry']['max_quote_age_seconds']*NS):return None
        # Identity depends on the past episode and arm, not later cohort size.
        cid=day_index*1_000_000_000+int(episode_id)*2+(0 if inside else 1)
        risk=float(np.clip(fr.atr14,rules['minimum_risk_per_oz'],rules['maximum_risk_per_oz']))
        r=dict(candidate_id=cid,origin_index=cid,t_ns=int(at),side=int(side),target_nano=int(target),risk_per_oz=risk,
            date=date,regime=fr.regime,time_regime=fr.time_regime,zone_id=f'{date}:{low}:{width}',zone_low_nano=int(low),
            zone_high_nano=int(low+width),episode_id=int(episode_id),parent_episode_id=int(parent_id),inside=bool(inside),
            target_available_ns=int(at),episode_available_ns=int(at),decision_mid_per_oz=(bid[j]+ask[j])/2e9,spread_per_oz=(ask[j]-bid[j])/1e9,
            deadline_ns=int(deadline),distance_per_oz=np.nan,distance_r=np.nan,**stats)
        rows.append(r)
        return r
    # Common parent-contact cohort, qualified strictly from completed past visits.
    for e in episodes.sort_values('arrival_ns').itertuples():
        if e.entry_side==0 or e.zone_low_nano not in grouped:continue
        at=int(e.arrival_ns);fi=int(np.searchsorted(fclock,at,'right')-1)
        if fi<0 or at-fclock[fi]>60*NS:continue
        fr=ready.iloc[fi]
        if fr.minute_utc>=20*60 or fr.epoch!=e.epoch:continue
        low=int(e.zone_low_nano);h=grouped[low].iloc[:np.searchsorted(ends[low],at,'left')]
        stats=history_stats(h,at,config)
        if stats is None or not stats['sticky']:continue
        center=low+width//2;side=int(np.sign(center-e.arrival_mid_nano))
        deadline=at+int(config['parent_comparison']['deadline_seconds_from_arrival']*NS)
        parent=dict(episode_id=int(e.episode_id),zone_low_nano=low,arrival_ns=at,deadline_ns=deadline,inside_candidate_id=-1,
            outside_candidate_id=-1,inside_wait_reason='EXACT_CENTER' if side==0 else 'QUALITY_REJECTED')
        if side:
            row=origin(at,fr,side,center,low,e.episode_id,stats,True,e.episode_id,deadline)
            if row is not None:parent.update(inside_candidate_id=row['candidate_id'],inside_wait_reason='')
        parent_by_episode[int(e.episode_id)]=parent;parents.append(parent)
    # Ready M1 is a decision opportunity; emit at most once per departure and
    # zone, with a separate cooldown. No raw quote becomes a model query.
    for fr in ready.itertuples():
        at=int(fr.t_ns)
        if fr.minute_utc>=20*60:continue
        j=int(np.searchsorted(qt,at,'right')-1);mid=(int(bid[j])+int(ask[j]))//2
        risk=float(np.clip(fr.atr14,rules['minimum_risk_per_oz'],rules['maximum_risk_per_oz']))
        for low,hall in grouped.items():
            if low<=mid<low+width:continue
            side=1 if mid<low else -1;target=low if side==1 else low+width
            dist=abs(target-mid)/NS
            if not max(rules['minimum_distance_per_oz'],rules['minimum_distance_r']*risk)<=dist<=rules['maximum_distance_r']*risk:continue
            stop=int(np.searchsorted(ends[low],at,'right'));h=hall.iloc[:stop]
            stats=history_stats(h,at,config)
            if stats is None:continue
            last=h.iloc[-1];key=(low,int(last.episode_id))
            # A later observed arrival has already ended this absence, even
            # while its new episode has not yet completed. Never wager against
            # the stale last-COMPLETED episode during a later active contact.
            contact_clock=contacts[low];ci=int(np.searchsorted(contact_clock,at,'right')-1)
            if ci>=0 and contact_clock[ci]>int(last.end_ns):continue
            if np.searchsorted(breaks,at,'right')>np.searchsorted(breaks,int(last.end_ns),'right'):continue
            if key in used or at-last_by_zone.get(low,-10**30)<rules['cooldown_seconds']*NS:continue
            parent=parent_by_episode.get(int(last.episode_id));pid=-1
            if parent is not None and at<parent['deadline_ns'] and parent['outside_candidate_id']<0:pid=int(last.episode_id)
            row=origin(at,fr,side,target,low,last.episode_id,stats,False,pid,parent['deadline_ns'] if pid>=0 else -1)
            if row is None:continue
            used.add(key);last_by_zone[low]=at
            row['distance_per_oz']=dist;row['distance_r']=dist/risk
            row['episode_available_ns']=int(last.end_ns)
            if pid>=0:parent['outside_candidate_id']=row['candidate_id']
    candidates=pd.DataFrame(rows)
    if candidates.empty:return candidates,pd.DataFrame(parents),pd.DataFrame(),pd.DataFrame()
    # Spectral diagnostics and token histories are only needed on outside origins.
    spec=[];pricecols={int(p):i for i,p in enumerate(grid['price_bin_nano'])}
    for r in candidates.itertuples():
        low=int(r.zone_low_nano);at=int(r.t_ns)
        h=grouped[low].iloc[:np.searchsorted(ends[low],at,'right')].tail(8)
        for position,e in enumerate(h.itertuples()):
            tokens.append(dict(candidate_id=r.candidate_id,position=position,episode_id=e.episode_id,
                arrival_ns=e.arrival_ns,available_ns=e.end_ns,entry_side=e.entry_side,exit_side=e.exit_side,
                outcome=e.outcome,core_dwell_seconds=e.core_dwell_seconds,duration_seconds=e.duration_seconds,
                previous_absence_seconds=e.absence_seconds,approach_efficiency=e.approach_efficiency,
                previous_peak_distance_ticks=e.absence_peak_distance_ticks))
        if r.inside:continue
        a=arrivals.get(low,np.array([],dtype=np.int64))
        s=zone_spectrum(grid['times_ns'],grid['occupancy'][:,pricecols[low]],at,
            window_seconds=sc['window_seconds'],sample_seconds=sc['sample_seconds'],known=grid['known'],epoch=grid['epoch'],arrival_ns=a)
        s['candidate_id']=r.candidate_id
        s['spectral_gate']=bool(s['recurrence_evidence_known'] and s['period_candidate_supported'] and s['autocorr_at_peak'] is not None and s['arrival_count']>=sc['minimum_arrivals_for_veto']
            and s['main_lobe_power_fraction']>=sc['minimum_peak_concentration'] and s['autocorr_at_peak']>=sc['minimum_peak_autocorrelation'])
        s['interval_gate']=bool(s['return_interval_cv_known'] and s['arrival_count']>=sc['minimum_arrivals_for_veto'] and s['return_interval_cv']<=1)
        spec.append(s)
    spectral=pd.DataFrame(spec)
    candidates=candidates.merge(spectral[['candidate_id','spectral_gate','interval_gate']],on='candidate_id',how='left') if len(spectral) else candidates.assign(spectral_gate=False,interval_gate=False)
    candidates[['spectral_gate','interval_gate']]=candidates[['spectral_gate','interval_gate']].astype('boolean').fillna(False).astype(bool)
    return candidates.sort_values(['t_ns','candidate_id']).reset_index(drop=True),pd.DataFrame(parents),pd.DataFrame(tokens),spectral


def select_policy(candidates,name):
    outside=~candidates.inside
    if name=='return_frequent':mask=outside
    elif name=='return_sticky':mask=outside & candidates.sticky
    elif name=='return_permeable':mask=outside & candidates.permeable
    elif name=='return_sticky_spectral':mask=outside & candidates.sticky & candidates.spectral_gate
    elif name=='return_sticky_interval':mask=outside & candidates.sticky & candidates.interval_gate
    elif name=='inside_now':mask=candidates.inside & candidates.parent_episode_id.ge(0)
    elif name=='wait_outside':mask=outside & candidates.parent_episode_id.ge(0)
    else:raise ValueError('Unknown frozen policy')
    selected=candidates.loc[mask].copy()
    if selected.empty:return selected
    # Equal-time alternatives are ranked by distance only, never later outcomes.
    return selected.sort_values(['t_ns','distance_per_oz','zone_low_nano'],na_position='last').drop_duplicates('t_ns')
