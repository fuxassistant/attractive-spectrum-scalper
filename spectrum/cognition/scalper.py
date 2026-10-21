"""Continuous scalper: one decision point per second, volatility-scaled exits.

Decisions. Every second boundary t in the London..NY entry window with a
supported book at t. Features use the as-of quote grid at t and MBO seconds
rows available by t; nothing later.

Plans. Adaptive plans express take/stop as MID moves in volatility units
u = sigma600 * sqrt(H) ticks, so a widening spread alone never trips a stop;
the exit itself executes as a taker at the executable side. Two fixed plans
keep the old executable-tick semantics for comparison.

Entries. T: taker, first supported quote strictly after t+100ms within t+2s.
P: join the decision-time best price with a 1-lot limit at t+100ms for 30 s,
FIFO queue kernel V5 (fill-proof invariant enforced); unfilled = EMPTY ($0).

Exits. Session deadline > stop > take > time, at the triggering quote; support
loss or an exposure open at entry+hold+5s is censored, never priced.
Fee $0.30 per 1 oz round trip. Gates (spread, cost cover) are stored as
features/flags and applied by the policy, so their effect can be measured.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from numba import njit
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from .signatures import DEFAULT_DATA, TRAIN_MONTHS
from .passive import Quotes, LATENCY, TIMEOUT, TICK, day_tapes, captures_by_date, MARKET
from .representation import load_seconds, flow_family
from . import opportunity as op
from . import queue as qu

VERSION = 'SCALPER_CONTINUOUS_V1'
DEFAULT_OUTPUT = 'data/derived/scalper_v1'
FEE = .30
WAIT_S = 30
# (name, trigger, hold seconds, take, stop): 'mid' in units of u, 'exec' in executable ticks
PLANS = (('a30s', 'mid', 30, 1., 1.), ('a30w', 'mid', 30, 1.5, .75),
         ('a120s', 'mid', 120, 1., 1.), ('a120w', 'mid', 120, 1.5, .75),
         ('a600s', 'mid', 600, 1., 1.), ('a600w', 'mid', 600, 1.5, .75),
         ('f12_8', 'exec', 120, 12, 8), ('f24_12', 'exec', 300, 24, 12))
MODES = ('T', 'P')
SIDES = {1: 'L', -1: 'S'}
GATE_SPREAD_TICKS = 2.
COST_MULT = 2.   # take move must cover COST_MULT x (spread + fee) in ticks
COMPLETE, CENSORED, EMPTY, NOT_ELIGIBLE, EXCLUDED = 1, 2, 3, 4, 5
RET_WINDOWS = (5, 15, 60, 300)
FLOW_WINDOWS = (5, 30, 60, 300)


@njit(cache=True)
def _right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def taker_entries(qt, ok, t_dec, latency, timeout):
    """Quote index of the kernel taker fill; -2 not eligible, -1 no supported quote in time."""
    out = np.full(len(t_dec), -1, np.int64)
    for k in range(len(t_dec)):
        t = t_dec[k]
        j = _right(qt, t) - 1
        if j < 0 or not ok[j]:
            out[k] = -2
            continue
        j = _right(qt, t + latency)
        while j < len(qt) and qt[j] <= t + timeout:
            if ok[j]:
                out[k] = j
                break
            j += 1
    return out


@njit(cache=True)
def scan_exits(qt, bid, ask, ok, epoch, deadline, start, side, entry_px, mid_mode, take, stop, hold, horizon,
               st, exit_ns, net_ticks):
    """One linear quote scan per trade from its fill quote (inclusive)."""
    n = len(qt)
    for k in range(len(start)):
        j0 = start[k]
        if j0 < 0:
            continue
        s = side[k]; e0 = epoch[j0]; t0 = qt[j0]; mid0 = .5 * (bid[j0] + ask[j0])
        status = CENSORED; xs = t0
        for j in range(j0, n):
            t = qt[j]
            if t >= horizon[k] or (not ok[j]) or epoch[j] != e0:
                xs = t
                break
            px = bid[j] if s > 0 else ask[j]
            mv = s * (.5 * (bid[j] + ask[j]) - mid0) if mid_mode[k] else s * (px - entry_px[k])
            if t >= deadline or mv <= -stop[k] or mv >= take[k] or t - t0 >= hold[k]:
                status = COMPLETE; xs = t; net_ticks[k] = s * (px - entry_px[k]) / 1e8
                break
        st[k] = status; exit_ns[k] = xs


def features(day, q, grid, ks, t_dec, seconds):
    x = op.causal(grid, ks)
    sig = x['sigma600_ticks_per_sqrt_s']; known = x['sigma600_known'] & (sig > 0)
    out = dict(spread_ticks=x['spread_ticks'], sigma600=sig, sigma_known=known.astype(np.int8),
               sigma60_ratio=np.where(known & x['sigma60_known'], x['sigma60_ticks_per_sqrt_s'] / np.maximum(sig, 1e-9), 0.),
               sigma3600_ratio=np.where(known & x['sigma3600_known'], x['sigma3600_ticks_per_sqrt_s'] / np.maximum(sig, 1e-9), 0.),
               minutes_since_london=(t_dec - int(day.meta['london_ns'])) / (60 * NS))
    mid, seg = grid['mid'], grid['segment']
    for w in RET_WINDOWS:
        lo = np.clip(ks - w, 0, None)
        ok = known & (ks - w >= 0) & ~np.isnan(mid[ks]) & ~np.isnan(mid[lo]) & (seg[lo] == seg[ks])
        out[f'ret{w}'] = np.where(ok, (mid[ks] - mid[lo]) / np.maximum(sig * np.sqrt(w), 1e-9), 0.)
        out[f'ret{w}_known'] = ok.astype(np.int8)
    series = pd.Series(mid)
    for w in (60, 300):
        hi_, lo_ = series.rolling(w + 1, min_periods=w + 1).max().to_numpy(), series.rolling(w + 1, min_periods=w + 1).min().to_numpy()
        rng = hi_[ks] - lo_[ks]; ok = np.isfinite(rng) & (rng > 0)
        out[f'range{w}_pos'] = np.where(ok, (mid[ks] - lo_[ks]) / np.where(ok, rng, 1), .5)
        out[f'range{w}_known'] = ok.astype(np.int8)
    f = flow_family(seconds, t_dec, windows=FLOW_WINDOWS, rate=(30, 300))
    for k, v in f.items():
        out[k] = v.astype(np.int8) if v.dtype == bool else v
    return out


def entry_points(q, tape_mbo, t_dec, eligible):
    """Per (mode, side): entry quote index (-1 none), entry price, pre-scan status (0 = run), time reference."""
    n = len(t_dec)
    qt, bid, ask, ok = q.t, q.bid, q.ask, q.ok
    j_dec = np.searchsorted(q.good_t, t_dec, side='right') - 1
    g = q.good[np.clip(j_dec, 0, None)]
    entries = {}
    t_idx = taker_entries(qt, ok, t_dec, LATENCY, TIMEOUT)
    for s in (1, -1):
        start = np.where(t_idx >= 0, t_idx, -1)
        px = np.where(start >= 0, (ask if s > 0 else bid)[np.clip(start, 0, None)], 0)
        st0 = np.where(t_idx == -2, NOT_ELIGIBLE, np.where(t_idx == -1, EMPTY, 0))
        entries[('T', s)] = (start, px, st0, t_dec.astype(np.int64))
    place = t_dec + LATENCY; probes = []
    for s in (1, -1):
        price = np.where(j_dec >= 0, (bid if s > 0 else ask)[g], 0).astype(np.int64)
        probes.append((s, price))
    p_place = np.concatenate([place, place]); p_side = np.r_[np.ones(n, np.int64), -np.ones(n, np.int64)]
    p_price = np.concatenate([probes[0][1], probes[1][1]]); p_end = p_place + WAIT_S * NS
    valid_probe = np.concatenate([eligible, eligible])
    stp, fill, _, _ = qu.probe(tape_mbo, p_place[valid_probe], p_side[valid_probe], p_price[valid_probe], p_end[valid_probe])
    qu.check_fill_proofs(tape_mbo, p_place[valid_probe], p_side[valid_probe], p_price[valid_probe], stp, fill)
    full_st = np.full(2 * n, qu.UNPLACED, np.int8); full_fill = np.zeros(2 * n, np.int64)
    full_st[valid_probe] = stp; full_fill[valid_probe] = fill
    after = np.searchsorted(qt, place, side='right')
    for b, (s, price) in enumerate(probes):
        sl = slice(b * n, (b + 1) * n); pst, pf = full_st[sl], full_fill[sl]
        a_c = np.clip(after, 0, len(qt) - 1)
        marketable = (j_dec >= 0) & eligible & (((s > 0) & (ask[a_c] <= price)) | ((s < 0) & (bid[a_c] >= price))) & (after < len(qt)) & ok[a_c]
        jm = np.clip(np.searchsorted(qt, place, side='right'), 0, len(qt) - 1)
        filled = (pst == qu.FILLED) & ~marketable
        start = np.where(marketable, jm, np.where(filled, np.searchsorted(qt, pf, side='left'), -1))
        start = np.where(start >= len(qt), -1, start)
        px = np.where(marketable, (ask if s > 0 else bid)[jm], price)
        st0 = np.where(~eligible, NOT_ELIGIBLE, np.where(pst == qu.CENSORED, CENSORED,
                       np.where(start >= 0, 0, EMPTY)))
        ref = np.where(start >= 0, qt[np.clip(start, 0, None)], 0)
        entries[('P', s)] = (start, px.astype(np.int64), st0, ref)
    return entries


def labels(q, tape_mbo, t_dec, eligible, sigma, spread):
    """Outcome columns for every mode x side x plan."""
    n = len(t_dec); cols = {}
    qt, bid, ask, ok, ep = q.t, q.bid, q.ask, q.ok, q.epoch
    entries = entry_points(q, tape_mbo, t_dec, eligible)
    for name, trig, hold, take, stop in PLANS:
        if trig == 'mid':
            unit = sigma * np.sqrt(hold) * TICK
            take_n, stop_n = take * unit, stop * unit
        else:
            take_n = np.full(n, take * TICK, float); stop_n = np.full(n, stop * TICK, float)
        for mode in MODES:
            for s in (1, -1):
                start, px, st0, ref = entries[(mode, s)]
                excluded = t_dec + (hold + 5 + (WAIT_S if mode == 'P' else 2)) * NS >= q.deadline
                run = (start >= 0) & ~excluded & (st0 == 0)
                st = np.where(excluded, EXCLUDED, st0).astype(np.int8)
                ex = np.zeros(n, np.int64); net = np.full(n, np.nan)
                idx = np.flatnonzero(run)
                if len(idx):
                    s_st = np.zeros(len(idx), np.int8); s_ex = np.zeros(len(idx), np.int64); s_net = np.full(len(idx), np.nan)
                    scan_exits(qt, bid, ask, ok, ep, q.deadline, start[idx], np.full(len(idx), s, np.int64), px[idx],
                               np.full(len(idx), trig == 'mid'), take_n[idx].astype(float), stop_n[idx].astype(float),
                               np.full(len(idx), hold * NS, np.int64), ref[idx] + (hold + 5) * NS, s_st, s_ex, s_net)
                    st[idx] = s_st; ex[idx] = s_ex; net[idx] = s_net
                key = f'{mode}_{SIDES[s]}_{name}'
                cols[key + '_st'] = st
                cols[key + '_net'] = np.where(st == COMPLETE, net * TICK / 1e9 - FEE, np.where(st == EMPTY, 0., np.nan)).astype(np.float32)
                known_at = np.where(st == COMPLETE, ex, np.where(st == EMPTY, t_dec + LATENCY + (WAIT_S if mode == 'P' else 2) * NS, 0))
                cols[key + '_busy_s'] = np.where(known_at > 0, (known_at - t_dec) / NS, np.nan).astype(np.float32)
    return cols


def build_day(date, tape_mbo, data=DEFAULT_DATA):
    started = time.perf_counter()
    day = MonthDay(Path(data) / date[:7] / date); qtape = QuoteTape.from_day(day); q = Quotes(qtape)
    start, end = int(day.meta['start_ns']), int(day.meta['end_ns'])
    lo, cutoff = int(day.meta['london_ns']), int(day.meta['ny_close_ns'])
    grid = op.second_grid(qtape.quote_t, qtape.bid, qtape.ask, qtape.valid, qtape.epoch, start, end)
    ks = np.arange((lo - start) // NS + 1, (cutoff - start) // NS, dtype=np.int64)
    t_dec = start + ks * NS
    last = np.searchsorted(q.t, t_dec, side='right') - 1
    eligible = (last >= 0) & q.ok[np.clip(last, 0, None)]
    feats = features(day, q, grid, ks, t_dec, load_seconds(day.meta['quote_parts']))
    keep = eligible & (feats['sigma_known'] > 0) & (feats['spread_ticks'] > 0)
    t_dec, ks = t_dec[keep], ks[keep]; feats = {k: v[keep] for k, v in feats.items()}
    cols = labels(q, tape_mbo, t_dec, np.ones(len(t_dec), bool), feats['sigma600'], feats['spread_ticks'])
    frame = pd.DataFrame(dict(date=date, t_ns=t_dec, **{k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == 'f' else v
                                                        for k, v in feats.items()}, **cols))
    QUOTE_CACHE.clear()
    return frame, dict(date=date, decisions=len(frame), seconds=time.perf_counter() - started)


def run_member(member_key, dates, data, output):
    receipts = []
    for d, tape, order in day_tapes(member_key, dates):
        frame, r = build_day(d, tape, data)
        dest = Path(output) / f'{d}.parquet'; frame.to_parquet(dest, index=False)
        r.update(sha256=sha(dest), captures=order); save_json(Path(output) / f'{d}.json', r); receipts.append(r)
        print(json.dumps(r | dict(sha256=r['sha256'][:12])), flush=True)
    return receipts


def _job(args):
    return run_member(*args)


def prepare(data=DEFAULT_DATA, output=DEFAULT_OUTPUT, months=TRAIN_MONTHS, workers=5, limit=None):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    by = captures_by_date(MARKET, tuple(months)); groups = defaultdict(list)
    dates = []
    for m in months:
        man = json.loads((Path(data) / m / 'manifest.json').read_text(encoding='utf8'))
        dates += [r['date'] for r in man['days']]
    dates = sorted(dates)[:limit] if limit else sorted(dates)
    for d in dates:
        if not (out / f'{d}.json').exists() and by.get(d):
            groups[by[d][0]['member_key']].append(d)
    code = {p: sha(Path(__file__).with_name(p)) for p in ('scalper.py', 'queue.py', 'passive.py', 'opportunity.py', 'representation.py')}
    save_json(out / 'protocol.json', dict(version=VERSION, plans=[list(p) for p in PLANS], modes=list(MODES), wait_s=WAIT_S,
        fee=FEE, gate_spread_ticks=GATE_SPREAD_TICKS, cost_mult=COST_MULT, code=code, semantics=__doc__.strip()))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, [(m, ds, data, output) for m, ds in sorted(groups.items())]))
    done = sorted(p.stem for p in out.glob('2*.json'))
    save_json(out / 'manifest.json', dict(status='COMPLETE' if len(done) == len(dates) else 'PARTIAL', version=VERSION,
        days=done, missing=[d for d in dates if d not in done], seconds=time.perf_counter() - started, code=code))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--months', nargs='+', default=list(TRAIN_MONTHS)); p.add_argument('--workers', type=int, default=5)
    p.add_argument('--limit', type=int)
    a = p.parse_args(); prepare(output=a.output, months=a.months, workers=a.workers, limit=a.limit)
