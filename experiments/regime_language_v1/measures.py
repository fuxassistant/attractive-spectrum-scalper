"""Observed regime facets: physical-time occupancy and published VWAP contexts.

These are evidence descriptors, not hidden regime truth or trading labels.
Profiles live at absolute price addresses; anchors never move those addresses.
"""
import numpy as np
import pandas as pd

NS=1_000_000_000
TICK=100_000_000
MINUTE=60*NS


class TimeAtPrice:
    def __init__(self,quotes,gap_seconds=30):
        self.t=quotes.available_ns.to_numpy(np.int64)
        self.p=(quotes.bid_nano.to_numpy(np.int64)+quotes.ask_nano.to_numpy(np.int64))//2
        self.good=(quotes.valid&quotes.epoch.ge(0)&quotes.bid_nano.gt(0)&quotes.ask_nano.gt(quotes.bid_nano)).to_numpy(bool)
        if not len(self.t) or np.any(np.diff(self.t)<0):raise ValueError('Nonempty ordered publication tape required')
        self.gap=int(gap_seconds*NS)

    def pieces(self,left,right):
        if not isinstance(left,(int,np.integer)) or not isinstance(right,(int,np.integer)) or right<=left:
            raise ValueError('Increasing integer-nanosecond interval required')
        first=max(0,int(self.t.searchsorted(left,side='right'))-1)
        last=int(self.t.searchsorted(right,side='left'))
        ids=np.arange(first,last,dtype=np.int64)
        starts=np.maximum(left,self.t[ids])
        next_time=np.r_[self.t[first+1:last],right] if len(ids) else np.empty(0,np.int64)
        ends=np.minimum(np.minimum(right,next_time),self.t[ids]+self.gap)
        dt=np.maximum(0,ends-starts)
        valid=self.good[ids]&(dt>0)
        return self.p[ids[valid]],dt[valid],starts[valid],ends[valid]

    def profile(self,left,right,width_ticks=5,shift_ticks=0.):
        width=int(round(width_ticks*TICK));shift=int(round(shift_ticks*TICK))
        if width<=0:raise ValueError('Positive absolute bin width required')
        p,dt,starts,ends=self.pieces(left,right)
        bins=(p-shift)//width
        addresses,inverse=np.unique(bins,return_inverse=True)
        seconds=np.bincount(inverse,weights=dt/NS,minlength=len(addresses))
        support=int(dt.sum());duration=int(right-left)
        result=dict(left_ns=int(left),right_ns=int(right),duration_seconds=duration/NS,
            supported_seconds=support/NS,coverage=support/duration,known=support==duration,
            bin_low_nano=addresses*width+shift,bin_seconds=seconds,width_nano=width,
            time_mean=None,time_sigma=None,mode_price=None,largest_bin_fraction=None,
            entropy=None,effective_bins=None)
        if support:
            origin=int(p[0]);z=(p-origin)/NS;w=dt/support
            mean=float(w@z);variance=float(w@((z-mean)**2))
            mass=seconds/(support/NS)
            result.update(time_mean=origin/NS+mean,time_sigma=float(np.sqrt(max(0,variance))),
                mode_price=float((addresses[int(np.argmax(seconds))]*width+shift+width/2)/NS),
                largest_bin_fraction=float(mass.max()),entropy=float(-(mass*np.log(mass)).sum()),
                effective_bins=float(1/(mass@mass)))
        return result

    def facets(self,left,right,amplitude_ticks,width_ticks=5,shift_ticks=0.):
        whole=self.profile(left,right,width_ticks,shift_ticks)
        middle=left+(right-left)//2
        a=self.profile(left,middle,width_ticks,shift_ticks)
        b=self.profile(middle,right,width_ticks,shift_ticks)
        common=np.union1d(a['bin_low_nano'],b['bin_low_nano'])
        pa=np.zeros(len(common));pb=np.zeros(len(common))
        if a['supported_seconds']:pa[np.searchsorted(common,a['bin_low_nano'])]=a['bin_seconds']/a['supported_seconds']
        if b['supported_seconds']:pb[np.searchsorted(common,b['bin_low_nano'])]=b['bin_seconds']/b['supported_seconds']
        known=a['known'] and b['known']
        whole.update(first_half_mean=a['time_mean'],second_half_mean=b['time_mean'],
            center_migration_A=(b['time_mean']-a['time_mean'])/(amplitude_ticks*.1) if known else None,
            half_profile_overlap=float(np.minimum(pa,pb).sum()) if known else None,
            first_half=a,second_half=b)
        return whole


class VwapContext:
    def __init__(self,context,max_age_seconds=120):
        self.frame=context.reset_index(drop=True)
        self.av=context.available_ns.to_numpy(np.int64)
        self.max_age=max_age_seconds
        if np.any(np.diff(self.av)<0):raise ValueError('Ordered context required')

    def at(self,clock):
        j=int(self.av.searchsorted(clock,side='left'))-1
        if j<0:return dict(clock_ns=int(clock),release_ns=None,endpoint_ns=None,anchors={})
        r=self.frame.iloc[j];end=int(r.t_ns);age=(clock-end)/NS
        anchors={}
        for name in ('day','combined','previous'):
            key=name+'_vwap';known=bool(r.get(key+'__known',False)) and np.isfinite(r[key]) and 0<=age<=self.max_age
            sigma=float(r[name+'_sigma']) if name!='previous' and bool(r.get(name+'_sigma__known',False)) else None
            anchors[name]=dict(value=float(r[key]) if known else None,sigma=sigma if known else None,known=known,
                population='accepted T, volume weights',kind='frozen_previous_session' if name=='previous' else 'developing_session_anchor')
        # Only completed, published minutes. These are not tick-updated VWAPs.
        for minutes in (60,240):
            f=self.frame.iloc[:j+1];f=f[(f.t_ns>end-minutes*MINUTE)&(f.t_ns<=end)]
            complete=len(f)==minutes and f.t_ns.diff().dropna().eq(MINUTE).all()
            known=bool(complete and f.known.all() and f.interval_open.all() and f.epoch.nunique()==1 and 0<=age<=self.max_age)
            volume=float(f.volume.sum());known=known and volume>0
            mean=float(f.pv.sum()/volume) if known else 0.
            variance=float(f.p2v.sum()/volume-mean**2) if known else 0.
            anchors['rolling_'+str(minutes)+'m']=dict(value=4000.+mean if known else None,
                sigma=float(np.sqrt(max(0,variance))) if known else None,known=known,
                population='accepted T, volume weights',kind='completed_minute_rolling')
        return dict(clock_ns=int(clock),release_ns=int(r.available_ns),endpoint_ns=end,age_seconds=age,
            day_coverage=float(r.day_coverage),combined_coverage=float(r.combined_coverage),
            previous_coverage=float(r.previous_session_coverage),anchors=anchors)


def anchor_relations(start,end,start_price,end_price,time_mean,amplitude_ticks):
    amplitude=amplitude_ticks*.1;rows={}
    for name,anchor in end['anchors'].items():
        old=start['anchors'].get(name,{})
        known=anchor['known'];both=known and old.get('known',False)
        distance=(end_price-anchor['value'])/amplitude if known else None
        delta_price=(end_price-start_price)/amplitude
        delta_line=(anchor['value']-old['value'])/amplitude if both else None
        relative_change=distance-(start_price-old['value'])/amplitude if both else None
        sigma=anchor['sigma']
        rows[name]=dict(known=known,value=anchor['value'],sigma=sigma,
            endpoint_distance_A=distance,endpoint_distance_sigma=(end_price-anchor['value'])/sigma if known and sigma is not None and sigma>0 else None,
            occupancy_center_distance_A=(time_mean-anchor['value'])/amplitude if known and time_mean is not None else None,
            price_change_A=delta_price,line_change_A=delta_line,relative_change_A=relative_change,
            relative_identity_residual=relative_change-(delta_price-delta_line) if both else None,
            state=('ABOVE' if distance>.5 else 'BELOW' if distance<-.5 else 'NEAR') if known else 'UNKNOWN')
    return rows


def terms(profile,net_A,path_A,elapsed_fraction,relations):
    """Overlapping visible definitions, not exhaustive or learned regime classes."""
    result=[]
    if profile['known']:
        efficiency=abs(net_A)/path_A if path_A else 0.
        if abs(net_A)>=1 and efficiency>=.65:result.append('DIRECTIONAL_PROGRESS')
        if path_A>=2 and efficiency<=.25:result.append('HIGH_PATH_LOW_PROGRESS')
        migration=profile['center_migration_A'];overlap=profile['half_profile_overlap']
        if migration is not None:
            result.append('OCCUPANCY_MIGRATES_UP' if migration>.5 else 'OCCUPANCY_MIGRATES_DOWN' if migration<-.5 else 'OCCUPANCY_CENTER_STABLE')
        if overlap is not None:
            if overlap>=.65:result.append('HALF_PROFILES_OVERLAP')
            elif overlap<=.25:result.append('HALF_PROFILES_SEPARATE')
    else:result.append('PRICE_TIME_SUPPORT_INCOMPLETE')
    for name,row in relations.items():result.append(name.upper()+':'+row['state'])
    return result


def scalar_time_mean(quotes,left,right,gap_seconds=30):
    """Separate loop over raw held quotes for a bounded audit interval."""
    f=quotes.iloc[max(0,int(quotes.available_ns.searchsorted(left,side='right'))-1):int(quotes.available_ns.searchsorted(right,side='left'))]
    times=f.available_ns.to_numpy(np.int64);support=0;total=0.
    for i,r in enumerate(f.itertuples()):
        if not (r.valid and r.epoch>=0 and r.bid_nano>0 and r.ask_nano>r.bid_nano):continue
        dt=max(0,min(right,int(times[i+1]) if i+1<len(times) else right,int(r.available_ns)+gap_seconds*NS)-max(left,int(r.available_ns)))
        support+=dt;total+=(((int(r.bid_nano)+int(r.ask_nano))//2)/NS)*dt
    return (total/support if support else None),support
