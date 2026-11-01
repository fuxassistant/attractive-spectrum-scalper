"""Auditable paired action targets; no forecast or trained policy is implied.

Market descriptions, realized counterfactual utilities, and causal action
values are different objects. The latter require conditional estimation.
"""
import numpy as np
import pandas as pd

NS = 1_000_000_000
STATES = ('NEITHER', 'LONG_ONLY', 'SHORT_ONLY', 'BOTH', 'UNKNOWN')


def paired_targets(features, outcomes, *, pending_seconds=2.):
    """Keep incomplete actions and administrative no-fill distinct.

    COMPLETE: measured net terminal utility. EMPTY: zero *modeled* PnL under
    the parent's explicit no-fill assumption, with pending time retained.
    CENSORED: unknown, never zero. Pair statistics need both actions known.
    All future-derived columns are future_* to reject accidental use as X.
    Nanosecond clocks use nullable integer arrays, never a float merge.
    """
    if outcomes.duplicated(['origin_index', 'side']).any():
        raise ValueError('One outcome per origin and side; select one execution scenario')
    if not outcomes.side.isin([1, -1]).all():
        raise ValueError('Only LONG/SHORT action rows')
    if features.origin_index.duplicated().any():
        raise ValueError('Unique origin identifiers required')
    admitted = features.loc[features.origin_admitted].copy().set_index('origin_index')
    if set(admitted.index) != set(outcomes.origin_index):
        raise ValueError('Outcomes must preserve every admitted origin')
    sides = {name: outcomes[outcomes.side == side].set_index('origin_index').reindex(admitted.index)
             for name, side in [('long', 1), ('short', -1)]}
    for f in sides.values():
        if f.status.isna().any():raise ValueError('Missing counterfactual side')
        if not f.signal_ns.equals(admitted.t_ns.rename('signal_ns')):
            if not np.array_equal(f.signal_ns.to_numpy(np.int64), admitted.t_ns.to_numpy(np.int64)):
                raise ValueError('Paired origins must share the exact decision clock')
    if not np.allclose(sides['long'].risk_per_oz,sides['short'].risk_per_oz,rtol=0,atol=0):
        raise ValueError('Paired actions must share frozen risk')
    out = admitted[['date','t_ns','family','level_kind','side','risk_hint']].copy()
    out['risk_nano'] = np.rint(out.risk_hint*NS).astype(np.int64)
    # Equal physical decisions/action contracts do not become independent just
    # because several named templates proposed them.
    counts=out.groupby(['date','t_ns','risk_nano']).t_ns.transform('size')
    out['exact_decision_weight']=1./counts
    for name,f in sides.items():
        full=f.status.eq('COMPLETE');empty=f.status.eq('EMPTY');known=full|empty
        if f.loc[full,'net_per_oz'].isna().any():raise ValueError('Complete without monetary outcome')
        if f.loc[~full,'net_per_oz'].notna().any():raise ValueError('Unknown outcomes must not contain invented PnL')
        out['future_status_'+name]=f.status
        out['future_reason_'+name]=f.reason
        out['future_value_known_'+name]=known
        out['future_filled_complete_'+name]=full
        out['future_u_'+name]=np.where(full,f.net_per_oz,np.where(empty,0.,np.nan))
        out['future_u_'+name+'_r']=out['future_u_'+name]/out.risk_hint
        mature=pd.Series(pd.NA,index=out.index,dtype='Int64')
        mature.loc[full]=f.loc[full,'exit_ns'].astype('int64')
        mature.loc[empty]=out.loc[empty,'t_ns']+int(round(pending_seconds*NS))
        out['future_available_ns_'+name]=mature
        for field in ('entry_ns','trigger_ns','exit_ns'):
            out['future_'+field+'_'+name]=pd.array(f[field],dtype='Int64')
        out['future_duration_seconds_'+name]=np.where(full,(f.exit_ns-f.entry_ns)/NS,
                                                       np.where(empty,pending_seconds,np.nan))
        # These extrema terminate at the action's exit, not at a common horizon.
        out['future_stopped_mfe_r_'+name]=f.mfe_per_oz/out.risk_hint
        out['future_stopped_mae_r_'+name]=f.mae_per_oz/out.risk_hint
    known=out.future_value_known_long & out.future_value_known_short
    filled=out.future_filled_complete_long & out.future_filled_complete_short
    out['future_pair_known']=known
    out['future_both_filled_complete']=filled
    out['future_common_r']=(out.future_u_long_r+out.future_u_short_r)/2
    out['future_direction_advantage_r']=(out.future_u_long_r-out.future_u_short_r)/2
    long=out.future_u_long>0;short=out.future_u_short>0
    out['future_joint_profit_state']=np.select([~known,long&short,long,short],
                                               ['UNKNOWN','BOTH','LONG_ONLY','SHORT_ONLY'],default='NEITHER')
    out['future_hindsight_best_r']=np.where(known,np.maximum(0,np.maximum(out.future_u_long_r,out.future_u_short_r)),np.nan)
    out['future_near_equal_01r']=known & ((out.future_u_long_r-out.future_u_short_r).abs()<=.1)
    maturity=pd.Series(pd.NA,index=out.index,dtype='Int64')
    maturity.loc[known]=np.maximum(out.loc[known,'future_available_ns_long'].to_numpy(np.int64),
                                   out.loc[known,'future_available_ns_short'].to_numpy(np.int64))
    out['future_pair_available_ns']=maturity
    return out.reset_index()


def mature_pair_mask(targets, cutoff_ns):
    if isinstance(cutoff_ns,bool) or not isinstance(cutoff_ns,(int,np.integer)):
        raise ValueError('Exact integer publication cutoff required')
    return (targets.future_pair_known & targets.future_pair_available_ns.le(cutoff_ns).fillna(False)
            & targets.t_ns.lt(cutoff_ns)).astype(bool)


def decision_from_values(common, direction):
    """Diagnostic risk-neutral one-entry gate. Full sequential WAIT has value.

    This helper takes CONDITIONAL expected utilities, not hindsight labels.
    Deterministic ties choose WAIT if nonpositive, then LONG for equal positives.
    """
    c,d=np.broadcast_arrays(np.asarray(common,float),np.asarray(direction,float))
    ql,qs=c+d,c-d
    finite=np.isfinite(ql)&np.isfinite(qs)
    value=np.maximum(ql,qs)
    action=np.where(finite & (value>0),np.where(ql>=qs,1,-1),0)
    return dict(q_long=ql,q_short=qs,entry_value=value,action=action,known=finite)
