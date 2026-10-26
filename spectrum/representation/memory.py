"""Reusable formulas extracted from spectrum/player_v3/data.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.core.schema import NS
from .schema import FEATURES

MINUTE = 60 * NS


VERSION = 'MIXED_MEMORY_V1'


BASE_NAMES = ['open_minus_previous', 'range', 'body', 'close_change',
    'from_session_open', 'session_range', 'log_volume', 'coverage', 'candle_known',
    'elapsed_minutes', 'session_fraction'] + list(FEATURES) + [n+'__known' for n in FEATURES]


FAST_BASE = FEATURES[:18] + (FEATURES[18], FEATURES[20])


VWAP_NAMES = tuple(f'vwap_proxy_{n}s_{field}' for n in (60, 300, 900, 3600)
                  for field in ('z', 'return_rms_1s', 'volume_rate'))


FAST_NAMES = list(FAST_BASE) + list(VWAP_NAMES)


ANCHOR_VALUES = ['from_current_open_ticks', 'from_previous_open_ticks',
    'from_london_open_ticks', 'from_previous_london_open_ticks',
    'from_previous_close_ticks', 'from_previous_high_ticks', 'from_previous_low_ticks',
    'from_running_high_ticks', 'from_running_low_ticks', 'running_range_ticks',
    'minutes_since_session_start', 'minutes_since_london', 'previous_age_days',
    'previous_same_contract', 'current_open_delay_seconds', 'london_open_delay_seconds',
    'previous_session_coverage', 'events_available', 'events_truncated']


EVENT_CLASSES = ('fresh_impulse', 'second_in_cluster', 'recurrent_irregular', 'recurrent_regular')


EVENT_ACTIVITY = ('price_led', 'volume_led', 'joint')


EVENT_VALUES = ('direction', 'move_ticks', 'volume', 'unknown_fraction',
    'pressure_lower', 'pressure_upper', 'prior_events_180', 'interval_cv',
    'interval_cv_known', 'last_gap_s', 'baseline_volatility', 'spread_after',
    'depth_bid_change_fraction', 'depth_ask_change_fraction', 'return_peak5_share',
    'flow_peak5_share', 'classified_flow_to_return_original',
    'classified_flow_to_return_original_known')


EVENT_NAMES = list(EVENT_VALUES) + list(EVENT_CLASSES) + list(EVENT_ACTIVITY) + [
    'age_minutes', 'current_minus_event_start_ticks', 'current_minus_event_end_ticks']


def bounds(date, contract):
    """NY session boundaries with DST; actual observation gaps remain masked."""
    day = pd.Timestamp(date)
    prior = (day-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    start = pd.Timestamp(prior+' 18:00', tz='America/New_York').value
    end = pd.Timestamp(date+' 17:00', tz='America/New_York').value
    london = pd.Timestamp(date+f' {contract.london_open_hour:02d}:00', tz='Europe/London').value
    ny = pd.Timestamp(date+f' {contract.new_york_close_hour:02d}:00', tz='America/New_York').value
    return start, end, london, ny


def asof_indices(endpoint, publication, decisions, max_age_ns=None):
    """Latest endpoint whose full prefix has been published by each decision.

    Prefix maximum is deliberately conservative at a late/out-of-order proof.
    It preserves the order of the recurrent sequence and cannot back-fill it.
    """
    endpoint = np.asarray(endpoint, np.int64)
    publication = np.maximum.accumulate(np.asarray(publication, np.int64))
    decisions = np.asarray(decisions, np.int64)
    if len(endpoint) == 0:
        return np.zeros(len(decisions), np.int64), np.zeros(len(decisions), bool)
    if np.any(np.diff(endpoint) <= 0):
        raise ValueError('Endpoint clock must increase')
    idx = np.minimum(np.searchsorted(endpoint, decisions, side='right'),
                     np.searchsorted(publication, decisions, side='right'))-1
    known = idx >= 0
    safe = np.maximum(idx, 0)
    if max_age_ns is not None:
        known &= decisions-endpoint[safe] <= max_age_ns
    return safe, known


def minute_values(bars, features, start, end):
    """Same 71 fields as the old player, joined by availability, not just endpoint."""
    b = bars.reset_index(drop=True)
    decisions = np.maximum.accumulate(b.available_ns.to_numpy(np.int64))
    idx, supported = asof_indices(features.t_ns, features.available_ns, decisions, NS+1)
    values = np.column_stack([features[n].to_numpy()[idx] for n in FEATURES])
    known = np.column_stack([features[n+'__known'].to_numpy()[idx] for n in FEATURES])
    known &= supported[:, None]
    values = np.where(known, values, 0.)
    valid = b.valid.to_numpy(bool) & np.isfinite(b[['o','h','l','c']].to_numpy()).all(axis=1)
    anchor = previous = 0.; high = -np.inf; low = np.inf; last_t = start; candles = []
    for i, row in enumerate(b.itertuples()):
        if valid[i]:
            if not anchor:
                anchor = previous = float(row.o)
            high = max(high, row.h); low = min(low, row.l)
            x = [(row.o-previous)/.1, (row.h-row.l)/.1, (row.c-row.o)/.1,
                 (row.c-previous)/.1, (row.c-anchor)/.1, (high-low)/.1,
                 np.log1p(max(0., row.volume)), row.coverage, 1.]
            previous = float(row.c)
        else:
            x = [0.]*7 + [float(row.coverage), 0.]
        candles.append(x+[(int(row.t_ns)-last_t)/MINUTE, (int(row.t_ns)-start)/(end-start)])
        last_t = int(row.t_ns)
    x = np.c_[candles, values, known.astype(float)].astype(np.float32)
    admitted = valid & supported & features.admissible.to_numpy(bool)[idx]
    admitted &= (decisions-b.t_ns.to_numpy()) <= NS
    return x, decisions, admitted


def observed_previous_date(date, calendar, contract):
    """Previous observed weekday session; a missing MBO day is never skipped."""
    for age in range(1, 11):
        d = (pd.Timestamp(date)-pd.Timedelta(days=age)).strftime('%Y-%m-%d')
        if pd.Timestamp(d).weekday() >= 5:
            continue
        _, _, london, ny = bounds(d, contract)
        minutes = np.arange(london//MINUTE, ny//MINUTE, dtype=np.int64)*MINUTE
        if (~calendar.closed(minutes)).sum() >= 60:
            return d
    return None


def _opening(seconds, boundary, decision):
    good = seconds.valid.to_numpy(bool) & np.isfinite(seconds.mid.to_numpy())
    eligible = good & (seconds.t_ns.to_numpy() >= boundary)
    i = np.flatnonzero(eligible)
    if not len(i):
        return 0., np.zeros(len(decision), bool), 0.
    i = i[0]; delay = (int(seconds.t_ns.iloc[i])-boundary)/NS
    known = (decision >= int(seconds.available_ns.iloc[i])) & (delay <= 60)
    return float(seconds.mid.iloc[i]), known, delay


def anchor_values(seconds, bars, decision, previous, start, london, date, previous_date,
                  same_contract, event_available):
    idx, current_known = asof_indices(seconds.t_ns, seconds.available_ns, decision, 2*NS)
    mid = seconds.mid.to_numpy()[idx]
    current_known &= seconds.valid.to_numpy(bool)[idx] & np.isfinite(mid)
    vals = []; masks = []
    def add(v, known=True):
        v = np.broadcast_to(v, len(decision)); k = np.broadcast_to(known, len(decision)) & np.isfinite(v)
        vals.append(np.where(k, v, 0.)); masks.append(k)
    op, ok, delay = _opening(seconds, start, decision)
    lop, lok, ldelay = _opening(seconds, london, decision)
    previous_open = previous_london = previous_close = previous_high = previous_low = 0.
    pok = plok = pstats = False; coverage = 0.
    if previous is not None:
        ps, pb, pstart, plondon, pend = previous
        previous_open, pk, _ = _opening(ps, pstart, np.asarray([decision[0]])); pok = bool(pk[0])
        previous_london, pk, _ = _opening(ps, plondon, np.asarray([decision[0]])); plok = bool(pk[0])
        good = pb.valid.to_numpy(bool) & (pb.available_ns.to_numpy() <= decision[0])
        good &= np.isfinite(pb[['o','h','l','c']].to_numpy()).all(axis=1)
        coverage = float(good.sum()/max(1, (pend-pstart)/MINUTE))
        if good.any():
            v = pb[good]; previous_close = float(v.c.iloc[-1])
            previous_high = float(v.h.max()); previous_low = float(v.l.min()); pstats = coverage >= .8
            # Exact terminal supported second, rather than the penultimate M1
            # close. The final completed minute is included in previous memory.
            pgood=ps.valid.to_numpy(bool)&np.isfinite(ps.mid.to_numpy())
            pgood &= ps.available_ns.to_numpy()<=decision[0]
            if pgood.any():
                previous_close=float(ps.mid.to_numpy()[np.flatnonzero(pgood)[-1]])
    good_bar = bars.valid.to_numpy(bool)
    high = np.maximum.accumulate(np.where(good_bar, bars.h.to_numpy(), -np.inf))
    low = np.minimum.accumulate(np.where(good_bar, bars.l.to_numpy(), np.inf))
    for price, known in ((op, ok), (previous_open, pok), (lop, lok),
        (previous_london, plok), (previous_close, pstats), (previous_high, pstats),
        (previous_low, pstats), (high, np.isfinite(high)), (low, np.isfinite(low))):
        add((mid-price)/.1, current_known & known)
    add((high-low)/.1, np.isfinite(high)&np.isfinite(low))
    add((decision-start)/MINUTE); add((decision-london)/MINUTE)
    add((pd.Timestamp(date)-pd.Timestamp(previous_date)).days if previous_date else 0, previous_date is not None)
    add(float(same_contract)); add(delay, ok); add(ldelay, lok); add(coverage, previous is not None)
    counts = np.searchsorted(event_available, decision, side='right')
    add(counts); add(np.maximum(counts-256, 0))
    return np.c_[np.column_stack(vals), np.column_stack(masks)].astype(np.float32), np.where(current_known, mid, 0.)


def encode_events(events):
    """Explicit allowlist: outcome/recurrence-maturity columns cannot enter X."""
    numeric = events.reindex(columns=EVENT_VALUES).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite causal event field')
    onehot = np.column_stack([events.event_class.eq(k) for k in EVENT_CLASSES] +
                            [events.activity.eq(k) for k in EVENT_ACTIVITY])
    return np.c_[numeric, onehot].astype(np.float32)


def support_segment_starts(t, valid, epoch):
    """Start of the current contiguous supported second run, inclusive.

    An invalid endpoint yields an empty run. No hidden state may carry through
    a missing second, source reset, or invalid observed reference minute.
    """
    t=np.asarray(t);valid=np.asarray(valid,bool);epoch=np.asarray(epoch)
    i=np.arange(len(t),dtype=np.int64)
    reset=np.r_[True,(np.diff(t)!=NS)|(np.diff(epoch)!=0)|~valid[:-1]]
    return np.maximum.accumulate(np.where(~valid,i+1,np.where(reset,i,0)))

class MemoryView:
    """Causal slices over prepared arrays; callers supply arrays and publication cuts."""
    def memory_bounds(self, i, fast_steps=900, slow_steps=2760, event_limit=256):
        now = int(self.decision_t[i]); fr = int(self.fast_cuts[i])
        fl = max(0, fr-fast_steps, int(np.searchsorted(self.fast_t, now-fast_steps*NS, side='right')))
        if fr and hasattr(self,'fast_segment_start'):
            fl=max(fl,int(self.fast_segment_start[fr-1]))
        sr = int(self.slow_cuts[i]); sl = max(0, sr-slow_steps)
        er = int(self.event_cuts[i]); el = max(0, er-event_limit)
        return fl, fr, sl, sr, el, er

    def memory(self, i, fast_steps=900, slow_steps=2760, event_limit=256):
        now = int(self.decision_t[i])
        fl, fr, sl, sr, el, er = self.memory_bounds(i,fast_steps,slow_steps,event_limit)
        event = np.asarray(self.event_x[el:er]).copy()
        age = (now-self.event_available[el:er])/MINUTE
        distance = (self.decision_mid[i]-self.event_prices[el:er])/.1
        event = np.c_[event, age, distance].astype(np.float32)
        return dict(fast=self.fast_x[fl:fr], slow=self.slow_x[sl:sr],
            slow_session=self.slow_session[sl:sr], anchors=self.anchors[i], events=event,
            fast_t=self.fast_t[fl:fr], slow_t=self.slow_t[sl:sr], event_t=self.event_t[el:er],
            event_available=self.event_available[el:er], event_truncated=el,
            decision_t=now, fast_valid=self.fast_valid[fl:fr])
