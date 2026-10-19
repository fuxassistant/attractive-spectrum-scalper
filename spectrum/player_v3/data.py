"""Causal mixed-resolution memory, with exact existing BBO replay execution.

One second tape and one minute tape are stored once per session. Windows are
views selected by publication time, never repeated on disk. Future event outcomes
are not read. All prices sent to a model are differences, never raw GC levels.
"""
from pathlib import Path
import argparse
import json
import time
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.full_history import ObservedCalendar
from spectrum.lab import Context
from spectrum.plugins.volatility_vwap import compute as compute_vwap
from spectrum.player.contract import Contract, NS
from spectrum.player.data import FEATURES, Day as ReplayDay

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


def _load_session(date, symbol, receipts, calendar, contract, verify=True, quotes=False,
                  include_closing_bar=False):
    start, end, london, ny = bounds(date, contract)
    day0 = pd.Timestamp(start, tz='UTC').strftime('%Y-%m-%d')
    candidates = [r for r in receipts if r['status']=='COMPLETE' and
        r['entry']['selection']['symbol']==symbol and r['entry']['selection']['date'] in (day0, date)]
    second_parts = []; bar_parts = []; feature_parts = []; quote_parts = []; lineage = []
    for number, r in enumerate(sorted(candidates, key=lambda r:r['entry']['selection']['date'])):
        root = Path(r['market']); feat = Path(r['features'])/'features.parquet'
        needed = ['seconds.parquet', 'bars.parquet'] + (['quotes.parquet'] if quotes else [])
        if verify:
            for name in needed:
                if sha(root/name) != r['market_payloads'][name]:
                    raise ValueError('Source changed: '+str(root/name))
            if sha(feat) != r['feature_sha256']:
                raise ValueError('Source features changed')
        filt = [('t_ns','>=', start), ('t_ns','<',end)]
        s = pq.read_table(root/'seconds.parquet', filters=filt).to_pandas()
        bfilt=[('t_ns','>=',start),('t_ns','<=' if include_closing_bar else '<',end)]
        b = pq.read_table(root/'bars.parquet', filters=bfilt).to_pandas()
        cols = ['t_ns','available_ns','admissible'] + [n for name in FEATURES for n in (name,name+'__known')]
        f = pq.read_table(feat, columns=cols, filters=filt).to_pandas()
        if not np.array_equal(s.t_ns, f.t_ns):
            raise ValueError('Seconds/features alignment')
        # Epochs from separate reset captures must never appear continuous.
        s['epoch'] = s.epoch.to_numpy(np.int64)+number*1_000_000
        s['valid'] &= ~calendar.closed(s.t_ns.to_numpy())
        b['valid'] &= ~calendar.closed(b.t_ns.to_numpy()-1)
        second_parts.append(s); bar_parts.append(b); feature_parts.append(f)
        if quotes:
            q = pq.read_table(root/'quotes.parquet', filters=filt).to_pandas()
            q['epoch'] = q.epoch.to_numpy(np.int64)+number*1_000_000
            q['valid'] &= ~calendar.closed(q.t_ns.to_numpy())
            quote_parts.append(q)
        lineage.append(dict(key=r['key'], market_payloads={n:r['market_payloads'][n] for n in needed},
                            feature_sha256=r['feature_sha256']))
    if not second_parts or sum(map(len, second_parts)) == 0:
        return None
    s, b, f = [pd.concat(parts, ignore_index=True).sort_values('t_ns', kind='stable').reset_index(drop=True)
               for parts in (second_parts, bar_parts, feature_parts)]
    if np.any(np.diff(s.t_ns) <= 0) or np.any(np.diff(b.t_ns) <= 0):
        raise ValueError('Overlapping captures')
    q = pd.concat(quote_parts, ignore_index=True).sort_values(['available_ns','source_ordinal'], kind='stable') if quotes else None
    return dict(seconds=s, bars=b, features=f, quotes=q, start=start, end=end, london=london, ny=ny, lineage=lineage)


def prepare_days(market, output, dates, contract=None,
                 event_path='reports/event_landmarks_v1/run_003/causal_events.parquet',
                 audit_path='reports/full_mbo_v1/audit_20261016/training_admission_overlay.json', verify=True):
    """Prepare selected dates only; prerequisite preceding context is read automatically."""
    started = time.perf_counter(); contract = contract or Contract(); market = Path(market); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    master = json.loads((market/'manifest.json').read_text()); receipts = master['captures']
    config = json.loads((market/'run_config.json').read_text()); reference = config['reference']
    calendar = ObservedCalendar(reference['path'], reference['sha256'])
    overlay = json.loads(Path(audit_path).read_text())
    eligible = {r['date']: r for r in overlay['captures'] if r.get('audit_eligible', r['eligible'])}
    selected = {r['entry']['selection']['date']: r for r in receipts if
                r['status']=='COMPLETE' and r['key'] in {v['key'] for v in eligible.values()}}
    absent = sorted(set(dates)-set(selected))
    if absent:
        raise ValueError('Requested non-admitted dates: '+str(absent))
    event_path = Path(event_path); event_frame = pq.read_table(event_path).to_pandas()
    event_sha = sha(event_path); results = []
    for date in dates:
        if shutil.disk_usage(output).free < 8*1024**3:
            raise RuntimeError('DISK_RESERVE_BELOW_8_GIB: completed days retained')
        r = selected[date]; symbol = r['entry']['selection']['symbol']
        cur = _load_session(date, symbol, receipts, calendar, contract, verify, True)
        if cur is None:
            raise ValueError('Missing requested session '+date)
        pdte = observed_previous_date(date, calendar, contract)
        same = pdte in selected and selected[pdte]['entry']['selection']['symbol']==symbol
        prev = _load_session(pdte, symbol, receipts, calendar, contract, verify, False, True) if same else None
        s, b, f, q = (cur[n] for n in ('seconds','bars','features','quotes'))
        x, decision, legal = minute_values(b, f, cur['start'], cur['end'])
        legal &= (decision >= cur['london']) & (decision < cur['ny'])
        legal &= decision >= int(r['entry']['selection']['selection_available_ns'])
        slow_x = x.copy(); slow_t = b.t_ns.to_numpy(np.int64); slow_available = decision.copy()
        slow_session = np.ones(len(x), np.int8); previous_bundle = None
        if prev is not None:
            px, pav, _ = minute_values(prev['bars'], prev['features'], prev['start'], prev['end'])
            slow_x = np.concatenate([px, x]); slow_t = np.r_[prev['bars'].t_ns, slow_t]
            slow_available = np.r_[pav, slow_available]; slow_session = np.r_[np.zeros(len(px), np.int8), slow_session]
            previous_bundle = (prev['seconds'],prev['bars'],prev['start'],prev['london'],prev['end'])
        context = Context(s, calendar); compute_vwap(context)
        fv = np.column_stack([f[n].to_numpy() for n in FAST_BASE]+[context.values[n] for n in VWAP_NAMES])
        fk = np.column_stack([f[n+'__known'].to_numpy() for n in FAST_BASE]+[context.masks[n] for n in VWAP_NAMES])
        fast_x = np.c_[np.where(fk, fv, 0.), fk].astype(np.float32)
        lo = prev['start'] if prev is not None else cur['start']
        ev = event_frame[(event_frame.symbol==symbol)&(event_frame.t_ns>=lo)&(event_frame.t_ns<cur['end'])]
        # Do not accidentally include another intervening date on a weekend or missing session.
        ev = ev[ev.date.isin([date]+([pdte] if prev is not None else []))].sort_values('decision_ns')
        event_available = ev.decision_ns.to_numpy(np.int64)
        anchors, mid = anchor_values(s,b,decision,previous_bundle,cur['start'],cur['london'],
                                     date,pdte,same,event_available)
        arrays = dict(x=x, decision_t=decision, bar_t=b.t_ns.to_numpy(np.int64), can_enter=legal,
            quote_t=q.available_ns.to_numpy(np.int64), source_t=q.t_ns.to_numpy(np.int64),
            bid=q.bid_nano.to_numpy(np.int64), ask=q.ask_nano.to_numpy(np.int64),
            valid=q.valid.to_numpy(bool), epoch=q.epoch.to_numpy(np.int64),
            ordinal=np.arange(len(q), dtype=np.int64), source_ordinal=q.source_ordinal.to_numpy(np.int64),
            fast_x=fast_x, fast_t=s.t_ns.to_numpy(np.int64),
            fast_available=np.maximum.accumulate(s.available_ns.to_numpy(np.int64)),
            fast_valid=s.valid.to_numpy(bool), slow_x=slow_x, slow_t=slow_t,
            slow_available=np.maximum.accumulate(slow_available), slow_session=slow_session,
            anchors=anchors, decision_mid=mid, event_x=encode_events(ev),
            event_available=event_available, event_t=ev.t_ns.to_numpy(np.int64),
            event_prices=ev[['event_start_price','event_end_price']].to_numpy(float))
        dest = output/date; dest.mkdir()
        for name, a in arrays.items():
            np.save(dest/(name+'.npy'), a)
        meta = dict(version=VERSION, date=date, symbol=symbol, previous_date=pdte,
            previous_available=prev is not None, previous_same_contract=same,
            start_ns=str(cur['start']), end_ns=str(cur['end']), london_ns=str(cur['london']), ny_close_ns=str(cur['ny']),
            liquidation_deadline_ns=str(cur['end']-contract.terminal_reserve_seconds*NS),
            rows=len(x), quotes=len(q), entry_origins=int(legal.sum()),
            feature_names=BASE_NAMES, fast_names=FAST_NAMES+[n+'__known' for n in FAST_NAMES],
            anchor_names=ANCHOR_VALUES+[n+'__known' for n in ANCHOR_VALUES], event_names=EVENT_NAMES,
            slow_names=BASE_NAMES, current_lineage=cur['lineage'], previous_lineage=prev['lineage'] if prev else [],
            source_event_sha256=event_sha, vwap_basis='midpoint weighted by all T volume, NOT exact trade VWAP',
            session_boundaries='18:00 NY previous date to 17:00 NY; absent OHLC reference minutes remain unavailable',
            files={name+'.npy':sha(dest/(name+'.npy')) for name in arrays})
        save_json(dest/'manifest.json', meta); results.append(meta)
        print(json.dumps(dict(prepared=date, seconds=len(s), minutes=len(x), previous=pdte,
                              previous_available=prev is not None, quotes=len(q))), flush=True)
    result = dict(version=VERSION, status='COMPLETE', days=results, dates=list(dates),
        elapsed_seconds=time.perf_counter()-started, market_manifest_sha256=sha(market/'manifest.json'),
        audit_overlay_sha256=sha(audit_path), reference=reference, event_source_sha256=event_sha,
        implementation_sha256=sha(__file__), session_contract=contract.dict(),
        memory='900 real seconds, previous/current M1 prefixes, explicit masked anchors, 256 causal events',
        no_future='event allowlist only; prefix publication gates; current-day high/low expanding only')
    save_json(output/'manifest.json', result)
    return result


class Day(ReplayDay):
    """Existing gym-compatible day plus bounded, unpadded memory views."""
    def __init__(self, path, verify=None):
        header=json.loads((Path(path)/'manifest.json').read_text())
        if header.get('backing')=='SHARED_CANONICAL_QUOTES_V1':
            from spectrum.playground.data import MonthDay
            MonthDay.__init__(self,path,verify=False if verify is None else verify)
            return
        if header.get('support_contract')=='FINALIZED_SEGMENTS_V2':
            from .finalized_data import FinalizedDay
            FinalizedDay.__init__(self,path,verify=False if verify is None else verify)
            return
        super().__init__(path, True if verify is None else verify)
        self.fast_cuts = np.searchsorted(self.fast_available, self.decision_t, side='right')
        self.slow_cuts = np.searchsorted(self.slow_available, self.decision_t, side='right')
        self.event_cuts = np.searchsorted(self.event_available, self.decision_t, side='right')
        qi=np.searchsorted(self.quote_t,self.fast_available,side='right')-1
        self.fast_epoch=np.asarray(self.epoch[np.maximum(qi,0)])
        supported=np.asarray(self.fast_valid)&(qi>=0)
        self.fast_segment_start=support_segment_starts(self.fast_t,supported,self.fast_epoch)

    def __getattr__(self,name):
        if self.__dict__.get('meta',{}).get('backing')=='SHARED_CANONICAL_QUOTES_V1':
            from spectrum.playground.data import MonthDay
            return MonthDay.__getattr__(self,name)
        raise AttributeError(name)

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


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--market', default='data/derived/full_mbo_v1')
    parser.add_argument('--output', required=True); parser.add_argument('--dates', nargs='+', required=True)
    args = parser.parse_args(); prepare_days(args.market, args.output, args.dates)


if __name__ == '__main__':
    main()
