"""Reusable formulas extracted from spectrum/cognition/dataset.py; see docs/MIGRATION.json."""
import numpy as np
from spectrum.core.schema import NS

VERSION = 'HOLD_CLOSE_PAIRED_V1'


HISTORY_NAMES = ('return_5m_ticks','return_15m_ticks','return_60m_ticks',
                 'range_5m_ticks','range_15m_ticks','range_60m_ticks',
                 'history_5m_known','history_15m_known','history_60m_known',
                 'quote_move_since_m1_ticks','m1_age_seconds','quote_move_since_m1_ticks_known')


def market_features(day, now):
    """Only completed M1 available at now; never read later rows."""
    i = int(np.searchsorted(day.decision_t, now, side='right')) - 1
    if i < 0: raise ValueError('No available M1 context')
    x = np.asarray(day.x[i], dtype=float).copy()
    history = []; known = []
    for window in (5,15,60):
        lo=i-window+1
        ok = lo>=1 and bool(np.all(day.x[lo-1:i+1,8]>.5)) and bool(np.all(np.diff(day.bar_t[lo-1:i+1])==60*NS))
        history.append((float(day.x[lo:i+1,3].sum()),float(day.x[lo:i+1,1].mean())) if ok else (0.,0.))
        known.append(float(ok))
    q = int(np.searchsorted(day.quote_t,now,side='right'))-1
    mid=(float(day.bid[q])+float(day.ask[q]))/2e9 if q>=0 else float(day.decision_mid[i])
    refq=int(day.cuts[i])-1
    contrast_known=bool(q>=0 and refq>=0 and day.valid[q] and day.valid[refq] and day.epoch[q]==day.epoch[refq] and x[8]>.5)
    contrast=(mid-float(day.decision_mid[i]))/.1 if contrast_known else 0.
    extra=[h[0] for h in history]+[h[1] for h in history]+known+[contrast,(now-int(day.decision_t[i]))/NS,float(contrast_known)]
    return np.r_[x,extra].astype(np.float32), i
