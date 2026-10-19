"""Passive (limit) execution variants over the same origins as the signatures.

Entry
  T   taker: first supported quote strictly after intent+100ms, within +2s,
      at the ask (long) / bid (short) - identical to CAUSAL_HOLD_CLOSE_BBO_V1
  P   join: a 1-lot limit at the decision-time best bid (long) / ask (short),
      effective at intent+100ms, cancelled after W seconds; fills only by FIFO
      proof from the MBO queue probe. Unfilled -> no trade (NOT_FILLED, $0)
  PX  join, and if unfilled after W cross as taker (first quote after W+100ms)
Exit
  stop, time limit and session deadline: taker at the triggering quote, as in
  the existing kernel (priority deadline > stop > target > time)
  T   take-profit when the executable side reaches the target (trade-through)
  Q   take-profit as a resting limit at entry +- take, placed at fill+100ms,
      filled only by FIFO proof; it competes in time with stop/time exits
Support loss (invalid publication or segment change) while exposed censors.
Fee $0.30 per 1 oz round trip. No market impact. Independent fresh accounts.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from .signatures import DEFAULT_DATA, DEFAULT_OUTPUT as SIGNATURES, MENU_PLANS, TRAIN_MONTHS
from . import queue as qu

VERSION = 'PASSIVE_EXECUTION_V3'
DEFAULT_OUTPUT = 'data/derived/passive_execution_v5'
MARKET = 'data/derived/full_mbo_v1'
LATENCY = 100_000_000
TIMEOUT = 2 * NS
FEE = .30
TICK = 100_000_000  # GC $0.10 per oz
WAITS = (30, 120, 600)
PLANS = (0, 1, 2, 3)


class Quotes:
    def __init__(self, tape):
        self.t = tape.quote_t; self.bid = tape.bid; self.ask = tape.ask
        self.ok = np.asarray(tape.valid, bool) & (tape.bid > 0) & (tape.ask >= tape.bid)
        self.epoch = tape.epoch; self.deadline = tape.deadline_ns
        self.entry_start = tape.entry_start_ns; self.entry_cutoff = tape.entry_cutoff_ns
        self.good = np.flatnonzero(self.ok); self.good_t = self.t[self.good]

    def asof(self, t):
        """Latest supported quote at or before t (decision information)."""
        j = np.searchsorted(self.good_t, t, side='right') - 1
        return self.good[j] if j >= 0 else -1

    def eligible(self, t):
        """Kernel open_at rule: the latest publication at or before t is a valid book."""
        j = np.searchsorted(self.t, t, side='right') - 1
        return j >= 0 and bool(self.ok[j])

    def taker_entry(self, t, side):
        """Kernel rule: first supported quote strictly after t+latency, within t+timeout."""
        if not self.eligible(t):
            return -2, 0
        j = np.searchsorted(self.t, t + LATENCY, side='right')
        while j < len(self.t) and self.t[j] <= t + TIMEOUT:
            if self.ok[j]:
                return j, int(self.ask[j] if side > 0 else self.bid[j])
            j += 1
        return -1, 0


def exit_path(q, side, entry_px, start, plan, tp_fill_ns=0, queue_tp=False, horizon_ns=None):
    """Exit after entry at quote index `start` (inclusive). Returns (status, exit_ns, exit_px, reason).
    Exposure still open at the horizon (kernel: intent + hold + 5s) is censored, never priced."""
    take, stop, hold = MENU_PLANS[plan]
    t0 = int(q.t[start])
    horizon = t0 + (hold + 5) * NS if horizon_ns is None else int(horizon_ns)
    hi = min(np.searchsorted(q.t, horizon, side='left'), len(q.t))
    tp_fill_ns = tp_fill_ns if tp_fill_ns and tp_fill_ns < horizon else 0
    sl = slice(start, hi)
    tt = q.t[sl]; ok = q.ok[sl]; ep = q.epoch[sl]
    px = (q.bid[sl] if side > 0 else q.ask[sl]).astype(np.int64)
    move = side * (px - entry_px)
    lost = (~ok) | (ep != q.epoch[start])
    dead = tt >= q.deadline
    stop_hit = ok & (move <= -stop * TICK)
    take_hit = ok & (move >= take * TICK) & (not queue_tp)
    time_hit = ok & (tt - t0 >= hold * NS)
    events = [(lost, 'support_loss'), (dead & ok, 'session_close'), (stop_hit, 'stop'), (take_hit, 'take'), (time_hit, 'time')]
    first = len(tt); reason = None
    for mask, name in events:
        idx = np.flatnonzero(mask)
        if len(idx) and idx[0] < first:
            first, reason = idx[0], name
    if queue_tp and tp_fill_ns and (reason is None or tp_fill_ns < tt[first] or
                                    (tp_fill_ns == tt[first] and reason == 'time')):
        tp_px = entry_px + side * take * TICK
        return 'COMPLETE', int(tp_fill_ns), int(tp_px), 'take_limit'
    if reason is None:
        return 'CENSORED', int(tt[-1]) if len(tt) else t0, 0, 'horizon_unresolved'
    if reason == 'support_loss':
        return 'CENSORED', int(tt[first]), 0, reason
    return 'COMPLETE', int(tt[first]), int(px[first]), reason


def adverse(q, side, entry_px, entry_ns, seconds=60):
    j = q.asof(entry_ns + seconds * NS)
    if j < 0:
        return np.nan
    return side * ((q.bid[j] + q.ask[j]) / 2 - entry_px) / TICK


def process_day(date, tape_mbo, sig_dir, data):
    """All variants for one day. tape_mbo: concatenated captures of the date."""
    started = time.perf_counter()
    x = pd.read_parquet(Path(sig_dir) / date / 'x.parquet', columns=['origin_id', 'intent_ns'])
    day = MonthDay(Path(data) / date[:7] / date); tape = QuoteTape.from_day(day); q = Quotes(tape)
    rows = []
    # ---- entries -------------------------------------------------------
    entries = []  # (origin_id, intent, side, kind, wait, entry_idx, entry_px, entry_ns, queue_ahead, status)
    p_place, p_side, p_price, p_end, p_ref = [], [], [], [], []
    for oid, intent in zip(x.origin_id, x.intent_ns.astype('int64')):
        for side in (-1, 1):
            j, px = q.taker_entry(intent, side)
            entries.append(dict(origin_id=oid, intent=intent, side=side, entry='T', wait=0, idx=j, px=px,
                                ns=int(q.t[j]) if j >= 0 else 0, ahead=-1,
                                status='FILLED' if j >= 0 else ('NOT_ELIGIBLE' if j == -2 else 'NOT_FILLED')))
            d = q.asof(intent)
            if d < 0 or not q.entry_start <= intent < min(q.entry_cutoff, q.deadline):
                for w in WAITS:
                    for kind in ('P', 'PX'):
                        entries.append(dict(origin_id=oid, intent=intent, side=side, entry=kind, wait=w, idx=-1, px=0,
                                            ns=0, ahead=-1, status='NOT_ELIGIBLE'))
                continue
            price = int(q.bid[d] if side > 0 else q.ask[d])
            p_place.append(intent + LATENCY); p_side.append(side); p_price.append(price)
            p_end.append(intent + LATENCY + max(WAITS) * NS); p_ref.append((oid, intent, side, price))
    st, fill, kind, ahead = qu.probe(tape_mbo, p_place, p_side, p_price, p_end)
    qu.check_fill_proofs(tape_mbo, p_place, p_side, p_price, st, fill)
    for (oid, intent, side, price), s, f, k, a in zip(p_ref, st, fill, kind, ahead):
        place = intent + LATENCY
        # marketable at placement: the opposite side already at/through our price -> taker at that quote
        jm = q.asof(place)
        marketable = jm >= 0 and ((side > 0 and q.ask[jm] <= price) or (side < 0 and q.bid[jm] >= price))
        for w in WAITS:
            filled = s == qu.FILLED and f <= place + w * NS
            if marketable:
                j = np.searchsorted(q.t, place, side='right')
                ok = j < len(q.t) and q.ok[j]
                e = dict(idx=j if ok else -1, px=int(q.ask[j] if side > 0 else q.bid[j]) if ok else 0,
                         ns=int(q.t[j]) if ok else 0, status='FILLED' if ok else 'NOT_FILLED', fill_kind='marketable')
            elif filled:
                j = np.searchsorted(q.t, f, side='left')
                e = dict(idx=j if j < len(q.t) else -1, px=price, ns=int(f), status='FILLED' if j < len(q.t) else 'CENSORED',
                         fill_kind=qu.KIND[int(k)])
            elif s == qu.CENSORED and f <= place + w * NS:
                e = dict(idx=-1, px=0, ns=0, status='CENSORED', fill_kind='none')
            else:
                e = dict(idx=-1, px=0, ns=0, status='NOT_FILLED', fill_kind='none')
            entries.append(dict(origin_id=oid, intent=intent, side=side, entry='P', wait=w, ahead=int(a), **e))
            if e['status'] == 'NOT_FILLED' and q.entry_start <= place + w * NS < min(q.entry_cutoff, q.deadline):
                j, px = q.taker_entry(intent + w * NS, side)  # cross after the wait
                e = dict(idx=j, px=px, ns=int(q.t[j]) if j >= 0 else 0, status='FILLED' if j >= 0 else 'NOT_FILLED',
                         fill_kind='crossed', cross_ns=intent + w * NS)
            entries.append(dict(origin_id=oid, intent=intent, side=side, entry='PX', wait=w, ahead=int(a), **e))
    # ---- take-profit queue probes for every distinct filled entry and plan ----
    tp_key = {}
    for e in entries:
        if e['status'] != 'FILLED':
            continue
        for plan in PLANS:
            take, _, hold = MENU_PLANS[plan]
            key = (e['ns'], e['side'], e['px'], plan)
            if key not in tp_key:
                tp_key[key] = len(tp_key)
    keys = sorted(tp_key, key=tp_key.get)
    tp_st, tp_fill, _, _ = qu.probe(tape_mbo, [k[0] + LATENCY for k in keys], [-k[1] for k in keys],
                                    [k[2] + k[1] * MENU_PLANS[k[3]][0] * TICK for k in keys],
                                    [k[0] + MENU_PLANS[k[3]][2] * NS for k in keys])
    qu.check_fill_proofs(tape_mbo, [k[0] + LATENCY for k in keys], [-k[1] for k in keys],
                         [k[2] + k[1] * MENU_PLANS[k[3]][0] * TICK for k in keys], tp_st, tp_fill)
    tp = {k: (int(tp_fill[i]) if tp_st[i] == qu.FILLED else 0) for k, i in tp_key.items()}
    # ---- compose variants ------------------------------------------------
    paths = {}  # identical fills across waits share one exit path
    for e in entries:
        for plan in PLANS:
            hold = MENU_PLANS[plan][2]
            for exit_kind in ('T', 'Q'):
                if e['entry'] == 'T' and e['wait'] != 0:
                    continue
                variant = f"{e['entry']}{exit_kind}" + (f"_{e['wait']}" if e['entry'] != 'T' else '')
                row = dict(origin_id=e['origin_id'], date=date, side=e['side'], plan_id=plan, variant=variant,
                           entry=e['entry'], exit=exit_kind, wait_s=e['wait'], queue_ahead=e['ahead'],
                           fill_kind=e.get('fill_kind', 'taker'))
                if e['intent'] + (hold + 5) * NS >= q.deadline:
                    rows.append(dict(row, status='EXCLUDED_SESSION_TIME')); continue
                if e['status'] != 'FILLED' or e['idx'] < 0:
                    rows.append(dict(row, status=e['status'], known_ns=e['intent'] + LATENCY + e['wait'] * NS)); continue
                if e['ns'] + hold * NS >= q.deadline:
                    rows.append(dict(row, status='EXCLUDED_SESSION_TIME')); continue
                # kernel horizon is measured from the order intent; a crossed entry from its new intent
                ref = e['intent'] if e['entry'] == 'T' else e.get('cross_ns', e['ns'])
                horizon = ref + (hold + 5) * NS
                key = (e['idx'], e['ns'], e['side'], e['px'], plan, exit_kind, horizon)
                if key not in paths:
                    paths[key] = exit_path(q, e['side'], e['px'], e['idx'], plan,
                                           tp.get((e['ns'], e['side'], e['px'], plan), 0), exit_kind == 'Q', horizon)
                status, xns, xpx, reason = paths[key]
                net = e['side'] * (xpx - e['px']) / 1e9 - FEE if status == 'COMPLETE' else None
                rows.append(dict(row, status=status, entry_ns=e['ns'], entry_px=e['px'], exit_ns=xns, exit_reason=reason,
                                 net_usd=net, known_ns=xns, wait_to_fill_s=(e['ns'] - e['intent']) / NS,
                                 adverse60_ticks=adverse(q, e['side'], e['px'], e['ns'])))
    QUOTE_CACHE.clear()
    frame = pd.DataFrame(rows)
    return frame, dict(date=date, probes=len(p_place), tp_probes=len(keys), rows=len(frame),
                       seconds=time.perf_counter() - started)


def captures_by_date(market=MARKET, months=TRAIN_MONTHS):
    by = defaultdict(list)
    m = json.loads((Path(market) / 'manifest.json').read_text(encoding='utf8'))
    for c in m['captures']:
        d = c['entry']['selection']['date']
        if c['status'] == 'COMPLETE' and d[:7] in months:
            e = c['entry']
            by[d].append(dict(key=c['key'], route=e['route'], first_ordinal=e['first_ordinal'], last_ordinal=e['last_ordinal'],
                              records=e['records'], member_key=e['member_key']))
    return by


def day_tapes(member_key, dates, market=MARKET):
    """Stream one month member once; yield (date, merged MBO tape, capture keys)
    as soon as every capture selected for that date has been read. Captures of
    one date are concatenated in ordinal order; each starts with its own reset."""
    config = json.loads((Path(market) / 'run_config.json').read_text(encoding='utf8'))
    item = next(i for i in config['members'] if i['key'] == member_key)
    by = captures_by_date(market, months=tuple(sorted({d[:7] for d in dates}))); wanted = {d: by[d] for d in dates}
    pending = {d: {c['key'] for c in cs} for d, cs in wanted.items()}
    collected = defaultdict(dict)
    key_date = {c['key']: d for d, cs in wanted.items() for c in cs}
    caps = [c for cs in wanted.values() for c in cs]
    for key, tape in qu.capture_tapes(item['archive'], item['member'], caps):
        d = key_date[key]; collected[d][key] = tape; pending[d].discard(key)
        if pending[d]:
            continue
        order = sorted(collected[d], key=lambda k: next(c['first_ordinal'] for c in wanted[d] if c['key'] == k))
        merged = {n: np.concatenate([collected[d][k][n] for k in order]) for n in collected[d][order[0]]}
        del collected[d]
        yield d, merged, order


def run_member(member_key, dates, sig_dir, data, output, market=MARKET):
    receipts = []
    for d, merged, order in day_tapes(member_key, dates, market):
        dest = Path(output) / f'{d}.parquet'
        frame, r = process_day(d, merged, sig_dir, data)
        frame.to_parquet(dest, index=False)
        r.update(sha256=sha(dest), captures=order, mbo_records=int(len(merged['ts'])))
        save_json(Path(output) / f'{d}.json', r); receipts.append(r)
        print(json.dumps({k: r[k] for k in ('date', 'probes', 'tp_probes', 'mbo_records', 'seconds')}), flush=True)
    return receipts


def _job(args):
    return run_member(*args)


def prepare(sig_dir=SIGNATURES, data=DEFAULT_DATA, output=DEFAULT_OUTPUT, workers=5, months=TRAIN_MONTHS):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(sig_dir) / 'manifest.json').read_text(encoding='utf8'))
    dates = [d['date'] for d in manifest['days'] if d['date'][:7] in months]
    by = captures_by_date(months=months)
    groups = defaultdict(list)
    for d in dates:
        if not (out / f'{d}.json').exists() and by.get(d):
            groups[by[d][0]['member_key']].append(d)
    code = {p: sha(Path(__file__).with_name(p)) for p in ('passive.py', 'queue.py', 'execution.py')}
    save_json(out / 'protocol.json', dict(version=VERSION, waits=list(WAITS), plans=[list(MENU_PLANS[p]) for p in PLANS],
        latency_ns=LATENCY, fee=FEE, code=code, signatures_manifest_sha256=sha(Path(sig_dir) / 'manifest.json'),
        queue=qu.__doc__.strip(), execution=__doc__.strip()))
    started = time.perf_counter(); receipts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_job, [(m, ds, sig_dir, data, output) for m, ds in sorted(groups.items())]):
            receipts.extend(r)
    done = sorted(p.stem for p in out.glob('*.json') if p.stem[:4].isdigit())
    save_json(out / 'manifest.json', dict(status='COMPLETE' if len(done) == len(dates) else 'PARTIAL', version=VERSION,
        days=done, missing=[d for d in dates if d not in done], seconds=time.perf_counter() - started, code=code))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--signatures', default=SIGNATURES)
    p.add_argument('--workers', type=int, default=5); p.add_argument('--months', nargs='+', default=list(TRAIN_MONTHS))
    a = p.parse_args(); prepare(sig_dir=a.signatures, output=a.output, workers=a.workers, months=a.months)
