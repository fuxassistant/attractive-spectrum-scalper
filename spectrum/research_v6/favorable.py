"""Configurable relabeling of immutable V6 crossing facts, no quote rescan.

MFE-qualified: a 2:1 excursion existed somewhere before the crossing.
Endpoint-qualified: the crossing itself delivers >=2 times full-path adverse.
Unknown never becomes NONE; extrema include costs and entry publication.
"""
import numpy as np
import pandas as pd


def qualify(frame, ratio=2.):
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('positive finite ratio required')
    out = frame.copy()
    observed = out.profile_known.to_numpy(bool)
    known = out.hit_known.to_numpy(bool)
    if (observed & ~known).any():
        raise ValueError('observed profiles must be known')
    for s in ('L', 'S'):
        adverse = np.maximum(-out[f'mae_R_{s}'].to_numpy(float), 0)
        net = out[f'net_R_{s}'].to_numpy(float)
        mfe = out[f'mfe_R_{s}'].to_numpy(float)
        if not np.isfinite(np.c_[adverse[observed], net[observed], mfe[observed]]).all():
            raise ValueError('observed profile nonfinite')
        # Cross multiplication avoids the undefined zero-adverse ratio.
        for kind, value in (('mfe', mfe), ('endpoint', net)):
            margin = value-ratio*adverse
            passed = observed & (value > 0) & (margin >= -1e-10)
            out[f'{kind}_margin_{s}'] = np.where(observed, margin, np.nan)
            out[f'{kind}_qualified_{s}'] = np.where(known, passed.astype(np.int8), -1)
        if ((out[f'endpoint_qualified_{s}']==1) & (out[f'mfe_qualified_{s}']!=1)).any():
            raise ValueError('endpoint must imply MFE qualification')
        out[f'adverse_R_{s}'] = np.where(observed, adverse, np.nan)
    return out


def origin_summary(cells):
    a=cells.sort_values(['t_ns','scale','role']).reset_index(drop=True)
    if a.duplicated(['t_ns','scale','role']).any() or not (a.groupby('t_ns').size()==9).all():
        raise ValueError('complete nine-line rectangular map required')
    n=len(a)//9
    def mat(name): return a[name].to_numpy().reshape(n,9)
    out=a.iloc[::9][['date','t_ns','split']].reset_index(drop=True)
    common=mat('hit_known').all(axis=1)
    out['common_known']=common
    for kind in ('mfe','endpoint'):
        flags=[]
        for s in ('L','S'):
            good=mat(f'{kind}_qualified_{s}')==1
            present=good.any(axis=1)
            flags.append(present)
            out[f'{kind}_{s}']=np.where(common,present.astype(np.int8),-1)
            ords=np.where(good,mat('cross_ordinal'),np.iinfo(np.int64).max)
            col=ords.argmin(axis=1)
            out[f'{kind}_first_id_{s}']=np.where(common & present,mat('v6_event_id')[np.arange(n),col],'')
            for field in ('cross_ns','cross_ordinal','elapsed_seconds','net_R_'+s,'adverse_R_'+s,'risk_before_cross_'+s):
                vals=mat(field)[np.arange(n),col]
                out[f'{kind}_first_{field}_{s}']=np.where(common & present,vals, -1 if field in ('cross_ns','cross_ordinal') else np.nan)
        out[f'{kind}_state']=np.where(common,flags[0].astype(int)+2*flags[1].astype(int),-1)
    # Opportunity amplitude on the selected endpoint-qualified event, not a mixed oracle profile.
    for s in ('L','S'):
        out[f'F_{s}']=np.where(common,np.where(out[f'endpoint_{s}']==1,out[f'endpoint_first_net_R_{s}_{s}'],0),np.nan)
    out['common_strength']=np.minimum(out.F_L,out.F_S)
    out['dominance']=out.F_L-out.F_S
    return out


def adjacent_changes(cells):
    """Separate same-event label changes from event switches on actual 5s clock."""
    results=[]
    for (date,scale,role),g in cells.groupby(['date','scale','role'],sort=False):
        g=g.sort_values('t_ns')
        for s in ('L','S'):
            present=g.profile_known.to_numpy(copy=True)
            adjacent=np.r_[False,np.diff(g.t_ns.to_numpy())==5*10**9]
            valid = present & np.r_[False,present[:-1]] & adjacent
            ids=g.v6_event_id.to_numpy()
            same=np.r_[False,ids[1:]==ids[:-1]]
            y=g[f'endpoint_qualified_{s}'].to_numpy()
            change=np.r_[False,y[1:]!=y[:-1]]
            margin=g[f'endpoint_margin_{s}'].to_numpy()
            dm=np.r_[np.nan,abs(np.diff(margin))]
            for label,mask in (('same_event',valid & same),('event_switch',valid & ~same)):
                results.append(dict(date=date,scale=int(scale),role=int(role),side=s,kind=label,
                    pairs=int(mask.sum()),flips=int((mask&change).sum()),
                    margin_abs_sum=float(np.nansum(dm[mask]))))
            known=g.hit_known.to_numpy()
            pair_known=known & np.r_[False,known[:-1]] & adjacent
            presence_change=present != np.r_[False,present[:-1]]
            mask=pair_known & presence_change
            results.append(dict(date=date,scale=int(scale),role=int(role),side=s,kind='hit_presence_change',
                pairs=int(mask.sum()),flips=int((mask&change).sum()),margin_abs_sum=0.))
    return pd.DataFrame(results)


def probe_feature_names(names):
    micro=[n for n in names if n.startswith('mbo__') and not n.startswith('mbo__fft_')]
    price=[n for n in micro if any(n.startswith('mbo__'+p) for p in ('displacement_','path_','efficiency_'))]
    geometry=[n for n in names if n.startswith('bb') or n in ('minutes_since_london','spread_ticks')]
    flow=[n for n in micro if n not in price and n!='mbo__spread_ticks']
    if len(geometry)!=14 or len(price)!=18 or len(flow)!=37:
        raise ValueError('unexpected frozen Micro schema')
    return geometry+price,flow
