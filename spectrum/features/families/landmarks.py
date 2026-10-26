"""Reusable formulas extracted from spectrum/event_landmarks.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.features.families.response import NS, spectral_features

VERSION = 'event_landmarks_v1'


HORIZONS = (30, 120)


def detect_bursts(move, volume, supported):
    """5s bins, preceding 120 bins; frozen-current threshold crossing + rearm.

    Return contains no future labels. Four events (three intervals) minimum for
    regular recurrence. An unsupported bin resets history and the state machine.
    """
    move, volume = np.asarray(move), np.asarray(volume)
    valid = np.asarray(supported, bool)
    m = pd.Series(np.where(valid, abs(move), np.nan))
    v = pd.Series(np.where(valid, volume, np.nan))
    mt = np.maximum(3., m.shift().rolling(120, min_periods=120).quantile(.95).to_numpy()*2.5)
    vt = np.maximum(10., v.shift().rolling(120, min_periods=120).quantile(.95).to_numpy()*2.5)
    ready = valid & np.isfinite(mt) & np.isfinite(vt)
    elevated = ready & ((abs(move) > mt) | (volume > vt))
    rows = []; quiet = 0; armed = False; history = []
    for i in range(len(move)):
        if not ready[i]:
            quiet = 0; armed = False; history = []; continue
        history = [j for j in history if (i-j)*5 <= 180]
        if not elevated[i]:
            quiet += 1
            if quiet >= 2: armed = True
            continue
        # A fall in rolling threshold alone must not create a fresh burst.
        p = abs(move[i]) > mt[i] and abs(move[i-1]) <= mt[i]
        q = volume[i] > vt[i] and volume[i-1] <= vt[i]
        if armed and (p or q):
            gaps = np.diff(history+[i])*5
            regular = len(gaps) >= 3 and min(gaps) >= 15 and max(gaps) <= 90 and gaps.std()/gaps.mean() <= .5
            kind = ('fresh_impulse' if not history else 'second_in_cluster' if len(history)==1
                    else 'recurrent_regular' if regular else 'recurrent_irregular')
            rows.append(dict(bin=i, event_class=kind, prior_events_180=len(history),
                interval_cv=float(gaps.std()/gaps.mean()) if len(gaps)>1 else 0.,
                interval_cv_known=len(gaps)>1, last_gap_s=int(gaps[-1]) if len(gaps) else 0,
                activity='joint' if p and q else 'price_led' if p else 'volume_led',
                move_threshold=float(mt[i]), volume_threshold=float(vt[i])))
            history.append(i); armed = False
        quiet = 0
    return rows, ready, elevated, mt, vt


def recurrence_labels(event_bins, all_bins, support):
    """Future repeat only after 180s full support; no negative label at EOF/gap."""
    events = np.asarray(event_bins, int); support = np.asarray(support, bool)
    labels = []
    for i in all_bins:
        mature = i+36 < len(support) and bool(support[i:i+37].all())
        count = int(np.sum((events>i)&(events<=i+36))) if mature else 0
        labels.append(dict(recurrence_mature=mature, future_events_180=count,
                           retrospectively_no_repeat=bool(mature and count==0)))
    return labels


def spectral_diagnostic(d, end):
    z=d.iloc[end-512:end+1]
    ret=np.diff(z.mid.to_numpy())/.1
    q=np.diff((z.c_trade_buy-z.c_trade_sell).to_numpy())
    vol=np.diff((z.c_trade_buy+z.c_trade_sell+z.c_trade_unknown).to_numpy())
    book=np.diff((z.c_add_bid-z.c_remove_bid-z.c_add_ask+z.c_remove_ask).to_numpy())
    channels=dict(return_=ret, classified_flow=q, volume=vol, book_flow=book, abs_return=abs(ret))
    channels['return']=channels.pop('return_')
    # Standardize each channel's energy before combining. Use complete 5s bins.
    energies=np.stack([ret[-510:]**2, q[-510:]**2]).reshape(2,102,5).sum(axis=2)
    totals=energies.sum(axis=1,keepdims=True)
    fractions=np.divide(energies,totals,out=np.zeros_like(energies),where=totals>0)
    peak=int(fractions.sum(axis=0).argmax()); left=2+peak*5
    _, original=spectral_features(channels)
    results=dict(return_peak5_share=float(fractions[0].max()), flow_peak5_share=float(fractions[1].max()))
    # Zero removal and linear interpolation with neighboring 5s are sensitivities,
    # neither is a null distribution or evidence that the shock causes coupling.
    variants={'original':channels}
    for name,pad in [('remove5',0),('interpolate15',5)]:
        copy={k:v.copy() for k,v in channels.items()}; a=max(0,left-pad); b=min(512,left+5+pad)
        for k,v in copy.items():
            if name=='remove5': v[a:b]=0
            else: v[a:b]=np.linspace(v[max(0,a-1)],v[min(511,b)],b-a+2)[1:-1]
        variants[name]=copy
    for name,chs in variants.items():
        cross=original if name=='original' else spectral_features(chs)[1]
        for pair in ('classified_flow_to_return','volume_to_abs_return'):
            vals=[cross[f'{pair}_{a}_{b}_coherence'] for a,b in ((4,8),(8,16),(16,32))]
            known=all(cross[f'{pair}_{a}_{b}_known'] for a,b in ((4,8),(8,16),(16,32)))
            results[f'{pair}_{name}']=float(np.mean(vals)); results[f'{pair}_{name}_known']=known
    return results
