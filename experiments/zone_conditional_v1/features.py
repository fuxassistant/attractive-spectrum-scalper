"""Known-at-decision geometry and event memory, without target/outcome columns."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

NS = 1_000_000_000


def log_positive(value):
    return np.log1p(np.maximum(0, value))


def make_features(candidates, bars, events, tokens, spectral):
    c = candidates.loc[~candidates.inside].copy().sort_values(['t_ns', 'candidate_id']).reset_index(drop=True)
    if c.empty:
        return c
    if c.date.nunique()!=1:
        raise ValueError('make_features requires one date; episode IDs are day-local')
    if c.candidate_id.duplicated().any():
        raise ValueError('Duplicate candidate identity')
    b = bars.sort_values('t_ns')
    ix = np.searchsorted(b.t_ns.to_numpy(np.int64), c.t_ns.to_numpy(np.int64), 'right')-1
    if (ix < 0).any():
        raise ValueError('Missing completed M1 context')
    f = b.iloc[ix].reset_index(drop=True)
    if not f.ready.all() or ((c.t_ns-f.t_ns) > 60*NS).any():
        raise ValueError('Stale or unsupported M1 context')
    c['bar_context_available_ns'] = f.t_ns.to_numpy(np.int64)
    c['x_base_distance_r'] = c.distance_r
    c['x_base_distance_r2'] = c.distance_r**2
    c['x_base_log_atr'] = log_positive(f.atr14)
    c['x_base_log_atr_ratio'] = log_positive(f.atr_ratio)
    c['x_base_spread_r'] = c.spread_per_oz/c.risk_per_oz
    c['x_base_log_spread'] = log_positive(c.spread_per_oz)
    c['x_base_side'] = c.side
    c['x_base_log_departure_age'] = log_positive(c.age_since_departure_seconds)
    minutes = (c.t_ns//(60*NS)) % 1440
    c['x_base_time_sin'] = np.sin(2*np.pi*minutes/1440)
    c['x_base_time_cos'] = np.cos(2*np.pi*minutes/1440)
    for name in ['body_atr', 'trend_atr', 'bb_z']:
        c['x_base_'+name] = f[name]*c.side
    for name in ['efficiency20', 'range_atr', 'volume']:
        c['x_base_'+name] = log_positive(f[name]) if name=='volume' else f[name]
    for name in ['completed_visits', 'cumulative_dwell_seconds', 'median_dwell_seconds', 'age_since_departure_seconds']:
        c['x_mem_log_'+name] = log_positive(c[name])
    c['x_mem_permeability'] = c.permeability
    c['x_mem_log_span_seconds'] = log_positive((c.zone_available_ns-c.history_start_ns)/NS)
    c['x_mem_log_dwell_per_visit'] = log_positive(c.cumulative_dwell_seconds/c.completed_visits)
    ev = events.loc[events.episode_id.ge(0)].groupby('episode_id')
    short = {}
    for eid, g in ev:
        r = g.loc[g.event_type.eq('EDGE_RETURN')]
        short[int(eid)] = (len(r), int(g.event_type.eq('GATE_ABORT').sum()),
                           float(r.duration_ns.max()/NS) if len(r) else 0.,
                           float(r.excursion_nano.max()/NS) if len(r) else 0.,
                           int(g.available_ns.max()))
    by_id = {int(k):v.sort_values('position') for k,v in tokens.groupby('candidate_id')}
    memory=[]
    for r in c.itertuples():
        h=by_id.get(int(r.candidate_id))
        if h is None or h.empty or len(h)>8 or (h.available_ns>r.t_ns).any():
            raise ValueError('Missing, overlong or future event history')
        if any(int(e.episode_id) not in short or short[int(e.episode_id)][4]>int(e.available_ns) for e in h.itertuples()):
            raise ValueError('Episode events exceed completed history availability')
        z=np.array([short[int(e)][:4] for e in h.episode_id],float)
        last=h.iloc[-1]
        memory.append(dict(x_mem_recent_pass_fraction=float(h.outcome.str.startswith('PASS').mean()),
            x_mem_recent_exit_toward=float((h.exit_side*r.side).mean()),
            x_mem_last_exit_toward=float(last.exit_side*r.side),
            x_mem_last_pass=float(str(last.outcome).startswith('PASS')),
            x_mem_recent_log_dwell=float(log_positive(h.core_dwell_seconds).mean()),
            x_mem_recent_log_duration=float(log_positive(h.duration_seconds).mean()),
            x_mem_log_last_duration=float(log_positive(last.duration_seconds)),
            x_mem_log_last_absence=float(log_positive(last.previous_absence_seconds)),
            x_mem_last_approach_efficiency=float(last.approach_efficiency),
            x_mem_log_last_peak_distance=float(log_positive(last.previous_peak_distance_ticks)),
            x_mem_log_recent_edge_returns=float(log_positive(z[:,0].sum())),
            x_mem_log_recent_gate_aborts=float(log_positive(z[:,1].sum())),
            x_mem_log_max_probe_seconds=float(log_positive(z[:,2].max())),
            x_mem_log_max_probe_distance=float(log_positive(z[:,3].max())),
            memory_available_ns=int(h.available_ns.max())))
    c=pd.concat([c,pd.DataFrame(memory)],axis=1)
    s=spectral.set_index('candidate_id').reindex(c.candidate_id).reset_index()
    if s.available_ns.isna().any() or (s.available_ns.to_numpy()>c.t_ns.to_numpy()).any():
        raise ValueError('Unknown or future spectrum availability')
    c['spectrum_available_ns']=s.available_ns.to_numpy(np.int64)
    for name in ['occupancy_mean','occupancy_variance','main_lobe_power_fraction','spectral_entropy',
                 'autocorr_at_peak','return_interval_cv','period_candidate_supported','spectrum_known']:
        c['x_spec_'+name]=s[name].astype(float).to_numpy()
    for name in ['arrival_count','dominant_period_seconds','return_interval_mean_seconds']:
        c['x_spec_log_'+name]=log_positive(s[name].to_numpy(float))
    # Context used for matching is available without episode-memory outcomes.
    c['match_atr']=f.atr14.to_numpy(float)
    c['match_distance_bin']=np.searchsorted([1.5,2.0],c.distance_r,side='right')
    c['match_spread_bin']=(c.spread_per_oz>.3).astype(int)
    c['match_phase']=np.searchsorted([12*60,16*60],minutes,side='right')
    return c


def add_past_placebo(frame, train_mask):
    """Past other-zone memory within frozen coarse context; no future donors.

    This is a causal stale-memory control, not an exchangeability/randomization
    test. Missing donors stay explicit. Donor selection never uses outcomes.
    """
    f=frame.copy().sort_values(['t_ns','candidate_id']).reset_index(drop=True)
    train_ids=set(frame.loc[train_mask,'candidate_id'])
    atr=f.loc[f.candidate_id.isin(train_ids),'match_atr']
    cuts=np.quantile(atr,[1/3,2/3])
    f['match_vol_bin']=np.searchsorted(cuts,f.match_atr,side='right')
    keys=['side','match_distance_bin','match_spread_bin','match_phase','match_vol_bin']
    memory=[x for x in f if x.startswith('x_mem_')]
    null=np.full((len(f),len(memory)),np.nan)
    donors=np.full(len(f),-1,np.int64);availability=donors.copy()
    pool=defaultdict(lambda:deque(maxlen=256))
    values=f[memory].to_numpy(float)
    # Update only after all rows at a timestamp have been queried.
    for _,group in f.groupby('t_ns',sort=True):
        for i,r in group.iterrows():
            key=tuple(int(r[k]) for k in keys)
            for j in reversed(pool[key]):
                if f.at[j,'zone_low_nano']!=r.zone_low_nano:
                    null[i]=values[j];donors[i]=int(f.at[j,'candidate_id'])
                    availability[i]=int(f.at[j,'t_ns']);break
        for i,r in group.iterrows():
            pool[tuple(int(r[k]) for k in keys)].append(i)
    for j,name in enumerate(memory):f[name.replace('x_mem_','x_placebo_')]=null[:,j]
    f['placebo_donor_id']=donors;f['placebo_available_ns']=availability
    f['placebo_known']=donors>=0
    f['x_placebo_known']=f.placebo_known.astype(float)
    if (f.loc[f.placebo_known,'placebo_available_ns']>=f.loc[f.placebo_known,'t_ns']).any():
        raise AssertionError('Nonpast placebo donor')
    return f,dict(atr_tertiles=cuts.tolist(),matching_keys=keys,donor_rule='latest strictly past different absolute zone; maximum256 per stratum')
