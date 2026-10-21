"""Path measurement after every sampled entry: first-passage ladders (PATH_MEASUREMENT_V1).

Decisions are the scalper's (London..NY, supported book, sigma600 known, spread > 0) at every
STRIDE-th second. Entries are the scalper's (scalper.entry_points): T taker, one entry quote for
both sides; P queue join per side (kernel V5). From each entry quote (inclusive) one numba pass in
quote order records, for every threshold of the row's ladder, the first quote where the mid moved
up by >= thr and down by >= thr (the float arithmetic of scalper.scan_exits), the running opposite
extreme at that quote, snapshots at the bin edges and the first unsupported quote. Nothing here is
a feature: it is all outcome (Y) of the decision at t_ns.

Ladder per row: nine plan levels m * sigma600 * sqrt(H) (m = .75, 1, 1.5; H = 30, 120, 600),
ten volatility levels c * sigma600 and ten absolute levels in ticks. The six adaptive plans are
re-assembled from the ladder (deadline > stop > take > time at one quote; support loss, epoch
change or a quote at/after ref + H + 5 s censors) and must equal data/derived/scalper_v1 exactly.
Per plan: category (take/stop/time), net $, busy s, time in trade, exit bin, favourable mid move at
exit, maximum adverse and favourable mid excursion before the exit.
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
from .passive import Quotes, LATENCY, TICK, day_tapes, captures_by_date, MARKET
from . import opportunity as op
from .scalper import (PLANS, SIDES, WAIT_S, FEE, COMPLETE, CENSORED, EMPTY, EXCLUDED, entry_points,
                      DEFAULT_OUTPUT as SCALPER_DATA)

VERSION = 'PATH_MEASUREMENT_V1'
DEFAULT_OUTPUT = 'data/derived/path_v1'
STRIDE = 5
EDGES = (5, 15, 30, 60, 120, 300, 600)
SPAN = EDGES[-1]
ADAPTIVE = tuple(p for p in PLANS if p[1] == 'mid')
PLAN_LEVELS = tuple((m, h) for h in (30, 120, 600) for m in (.75, 1., 1.5))
VOL_LEVELS = (1, 2, 3, 4, 6, 8, 11, 16, 22, 32)
ABS_LEVELS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
LADDERS = (('T', 0), ('P', 1), ('P', -1))       # T is shared by both sides
END_TAPE, END_SUPPORT, END_DEADLINE, END_SPAN = 1, 2, 3, 4
NONE, TAKE, STOP, TIME, DEADLINE = 0, 1, 2, 3, 4


@njit(cache=True)
def scan_ladder(qt, bid, ask, ok, epoch, deadline, start, thr, edges,
                up_j, dn_j, up_lo, dn_hi, snap_j, snap_hi, snap_lo, end_j, end_kind):
    """thr (n, L) ascending per row, > 0. up_j/dn_j: first quote with mv >= thr / mv <= -thr (-1 none);
    up_lo/dn_hi: running min/max mv up to and including that quote; snap_*: first quote with t - t0 >= edge."""
    n, L = thr.shape
    B = len(edges); N = len(qt)
    for k in range(n):
        j0 = start[k]
        if j0 < 0:
            continue
        e0 = epoch[j0]; t0 = qt[j0]; mid0 = .5 * (bid[j0] + ask[j0])
        hi = -1e300; lo = 1e300; pu = 0; pd = 0; nb = 0
        end_kind[k] = END_TAPE; end_j[k] = N
        for j in range(j0, N):
            t = qt[j]
            if (not ok[j]) or epoch[j] != e0:
                end_kind[k] = END_SUPPORT; end_j[k] = j
                break
            if t >= deadline:
                end_kind[k] = END_DEADLINE; end_j[k] = j
                break
            mv = .5 * (bid[j] + ask[j]) - mid0
            if mv > hi:
                hi = mv
            if mv < lo:
                lo = mv
            while pu < L and hi >= thr[k, pu]:
                up_j[k, pu] = j; up_lo[k, pu] = lo; pu += 1
            while pd < L and lo <= -thr[k, pd]:
                dn_j[k, pd] = j; dn_hi[k, pd] = hi; pd += 1
            while nb < B and t - t0 >= edges[nb]:
                snap_j[k, nb] = j; snap_hi[k, nb] = hi; snap_lo[k, nb] = lo; nb += 1
            if nb == B:
                end_kind[k] = END_SPAN; end_j[k] = j
                break


def thresholds(sigma):
    """(n, L) mid-move thresholds in nano, columns: PLAN_LEVELS, VOL_LEVELS, ABS_LEVELS (unsorted)."""
    cols = [m * (sigma * np.sqrt(h) * TICK) for m, h in PLAN_LEVELS]
    cols += [c * sigma * TICK for c in VOL_LEVELS] + [np.full(len(sigma), float(a * TICK)) for a in ABS_LEVELS]
    return np.stack(cols, 1).astype(np.float64)


def ladder(q, start, thr):
    """Run the scanner with per-row sorted thresholds; outputs in the original column order."""
    n, L = thr.shape; B = len(EDGES)
    order = np.argsort(thr, axis=1, kind='stable'); srt = np.take_along_axis(thr, order, 1)
    out = dict(up_j=np.full((n, L), -1, np.int64), dn_j=np.full((n, L), -1, np.int64),
               up_lo=np.zeros((n, L)), dn_hi=np.zeros((n, L)), snap_j=np.full((n, B), -1, np.int64),
               snap_hi=np.zeros((n, B)), snap_lo=np.zeros((n, B)), end_j=np.full(n, -1, np.int64),
               end_kind=np.zeros(n, np.int8))
    scan_ladder(q.t, q.bid, q.ask, q.ok, q.epoch, q.deadline, start.astype(np.int64), srt,
                np.asarray(EDGES, np.int64) * NS, out['up_j'], out['dn_j'], out['up_lo'], out['dn_hi'],
                out['snap_j'], out['snap_hi'], out['snap_lo'], out['end_j'], out['end_kind'])
    inv = np.argsort(order, axis=1)
    for k in ('up_j', 'dn_j', 'up_lo', 'dn_hi'):
        out[k] = np.take_along_axis(out[k], inv, 1)
    return out


def assemble(qt, bid, ask, start, side, entry_px, lad, col_take, col_stop, hold, horizon):
    """One mid-triggered plan from the ladder, mirroring scan_exits. Returns dict of arrays (index -1 = none)."""
    n = len(start); big = np.iinfo(np.int64).max; r = np.arange(n)
    up, dn = lad['up_j'], lad['dn_j']
    take_j = up[:, col_take] if side > 0 else dn[:, col_take]
    stop_j = dn[:, col_stop] if side > 0 else up[:, col_stop]
    b = EDGES.index(hold); hold_j = lad['snap_j'][:, b]
    dead_j = np.where(lad['end_kind'] == END_DEADLINE, lad['end_j'], -1)
    cand = [np.where(x >= 0, x, big) for x in (dead_j, stop_j, take_j, hold_j)]
    ex = np.minimum.reduce(cand)
    cat = np.select([ex == big, ex == cand[0], ex == cand[1], ex == cand[2]], [NONE, DEADLINE, STOP, TAKE], TIME).astype(np.int8)
    exc = np.clip(np.where(ex == big, 0, ex), 0, len(qt) - 1)
    done = (start >= 0) & (ex != big) & (qt[exc] < horizon)
    px = bid[exc] if side > 0 else ask[exc]
    net_ticks = side * (px - entry_px) / 1e8
    j0 = np.clip(start, 0, None)
    mid0 = .5 * (bid[j0] + ask[j0]); mv_exit = side * (.5 * (bid[exc] + ask[exc]) - mid0) / 1e8
    if side > 0:
        mae_ev = np.select([cat == TAKE, cat == STOP, cat == TIME], [-lad['up_lo'][r, col_take], -mv_exit * 1e8, -lad['snap_lo'][:, b]], np.nan)
        mfe_ev = np.select([cat == TAKE, cat == STOP, cat == TIME], [mv_exit * 1e8, lad['dn_hi'][r, col_stop], lad['snap_hi'][:, b]], np.nan)
    else:
        mae_ev = np.select([cat == TAKE, cat == STOP, cat == TIME], [lad['dn_hi'][r, col_take], -mv_exit * 1e8, lad['snap_hi'][:, b]], np.nan)
        mfe_ev = np.select([cat == TAKE, cat == STOP, cat == TIME], [mv_exit * 1e8, -lad['up_lo'][r, col_stop], -lad['snap_lo'][:, b]], np.nan)
    t0 = qt[j0]
    return dict(done=done, cat=np.where(done, cat, NONE).astype(np.int8), exit_j=np.where(done, ex, -1),
                exit_ns=np.where(done, qt[exc], 0), net_ticks=np.where(done, net_ticks, np.nan),
                tau=np.where(done, (qt[exc] - t0) / NS, np.nan), mv=np.where(done, mv_exit, np.nan),
                mae=np.where(done, mae_ev / 1e8, np.nan), mfe=np.where(done, mfe_ev / 1e8, np.nan))


def exit_bin(tau, hold):
    """Bin index of the exit time: (EDGES[k-1], EDGES[k]], time exits fall in the last bin up to hold."""
    last = EDGES.index(hold)
    return np.where(np.isfinite(tau), np.minimum(np.searchsorted(np.asarray(EDGES, float), np.nan_to_num(tau), side='left'), last), -1).astype(np.int8)


def plan_columns(q, t_dec, entries, lads, sigma):
    """Scalper-compatible st/net/busy_s plus path fields for every mode x side x adaptive plan."""
    n = len(t_dec); cols = {}
    qt, bid, ask = q.t, q.bid, q.ask
    which = {('T', 1): 0, ('T', -1): 0, ('P', 1): 1, ('P', -1): 2}
    for (mode, s), li in which.items():
        start, px, st0, ref = entries[(mode, s)]; lad = lads[li]
        j0 = np.clip(start, 0, None); key0 = f'{mode}_{SIDES[s]}'
        cols[key0 + '_delay'] = np.where(start >= 0, (qt[j0] - t_dec) / NS, np.nan).astype(np.float32)
        cols[key0 + '_cost'] = np.where(start >= 0, s * (px - .5 * (bid[j0] + ask[j0])) / 1e8, np.nan).astype(np.float32)
        cols[key0 + '_spread0'] = np.where(start >= 0, (ask[j0] - bid[j0]) / 1e8, np.nan).astype(np.float32)
        for name, trig, hold, take, stop in ADAPTIVE:
            ct, cs = PLAN_LEVELS.index((take, hold)), PLAN_LEVELS.index((stop, hold))
            excluded = t_dec + (hold + 5 + (WAIT_S if mode == 'P' else 2)) * NS >= q.deadline
            run = (start >= 0) & ~excluded & (st0 == 0)
            a = assemble(qt, bid, ask, np.where(run, start, -1), s, px, lad, ct, cs, hold, ref + (hold + 5) * NS)
            st = np.where(excluded, EXCLUDED, st0).astype(np.int8)
            st = np.where(run, np.where(a['done'], COMPLETE, CENSORED), st).astype(np.int8)
            key = f'{key0}_{name}'; c = st == COMPLETE
            cols[key + '_st'] = st
            cols[key + '_net'] = np.where(c, a['net_ticks'] * TICK / 1e9 - FEE, np.where(st == EMPTY, 0., np.nan)).astype(np.float32)
            known_at = np.where(c, a['exit_ns'], np.where(st == EMPTY, t_dec + LATENCY + (WAIT_S if mode == 'P' else 2) * NS, 0))
            cols[key + '_busy_s'] = np.where(known_at > 0, (known_at - t_dec) / NS, np.nan).astype(np.float32)
            cols[key + '_cat'] = np.where(c, a['cat'], NONE).astype(np.int8)
            cols[key + '_bin'] = np.where(c, exit_bin(a['tau'], hold), -1).astype(np.int8)
            for f in ('tau', 'mv', 'mae', 'mfe'):
                cols[f'{key}_{f}'] = np.where(c, a[f], np.nan).astype(np.float32)
    return cols


def grid_arrays(q, lads, starts):
    """Volatility and absolute levels: first-passage times (s from the entry quote, NaN = not before scan end)."""
    L0 = len(PLAN_LEVELS); out = {}
    up_t, dn_t, end_t, end_k = [], [], [], []
    for lad, start in zip(lads, starts):
        t0 = q.t[np.clip(start, 0, None)]
        def secs(j):
            return np.where(j >= 0, (q.t[np.clip(j, 0, None)] - t0[:, None]) / NS, np.nan).astype(np.float32)
        up_t.append(secs(lad['up_j'][:, L0:])); dn_t.append(secs(lad['dn_j'][:, L0:]))
        e = np.clip(lad['end_j'], 0, len(q.t) - 1)
        end_t.append(np.where(start >= 0, (q.t[e] - t0) / NS, np.nan).astype(np.float32)); end_k.append(lad['end_kind'])
    out['up_t'], out['dn_t'] = np.stack(up_t, 1), np.stack(dn_t, 1)
    out['end_t'], out['end_kind'] = np.stack(end_t, 1), np.stack(end_k, 1)
    return out


def decisions(day, q, qtape, stride=STRIDE):
    start, end = int(day.meta['start_ns']), int(day.meta['end_ns'])
    lo, cutoff = int(day.meta['london_ns']), int(day.meta['ny_close_ns'])
    grid = op.second_grid(qtape.quote_t, qtape.bid, qtape.ask, qtape.valid, qtape.epoch, start, end)
    ks = np.arange((lo - start) // NS + 1, (cutoff - start) // NS, dtype=np.int64)
    t_dec = start + ks * NS
    last = np.searchsorted(q.t, t_dec, side='right') - 1
    eligible = (last >= 0) & q.ok[np.clip(last, 0, None)]
    x = op.causal(grid, ks)
    sig = x['sigma600_ticks_per_sqrt_s']; known = x['sigma600_known'] & (sig > 0)
    keep = eligible & known & (x['spread_ticks'] > 0) & (ks % stride == 0)
    return t_dec[keep], sig[keep], x['spread_ticks'][keep]


def build_day(date, tape_mbo, data=DEFAULT_DATA, stride=STRIDE):
    started = time.perf_counter()
    day = MonthDay(Path(data) / date[:7] / date); qtape = QuoteTape.from_day(day); q = Quotes(qtape)
    t_dec, sigma, spread = decisions(day, q, qtape, stride)
    entries = entry_points(q, tape_mbo, t_dec, np.ones(len(t_dec), bool))
    thr = thresholds(sigma); t_scan = time.perf_counter()
    starts = [entries[('T', 1)][0], entries[('P', 1)][0], entries[('P', -1)][0]]
    lads = [ladder(q, s, thr) for s in starts]
    scan_s = time.perf_counter() - t_scan
    cols = plan_columns(q, t_dec, entries, lads, sigma)
    frame = pd.DataFrame(dict(date=date, t_ns=t_dec, sigma600=sigma.astype(np.float32), spread_ticks=spread.astype(np.float32), **cols))
    grid = grid_arrays(q, lads, starts)
    QUOTE_CACHE.clear()
    return frame, grid, dict(date=date, decisions=len(frame), scan_seconds=scan_s, seconds=time.perf_counter() - started)


def parity(frame, date, scalper_data=SCALPER_DATA):
    """Exact comparison with the scalper dataset on the shared decision rows."""
    keys = [f'{m}_{SIDES[s]}_{p[0]}' for m in ('T', 'P') for s in (1, -1) for p in ADAPTIVE]
    ref = pd.read_parquet(Path(scalper_data) / f'{date}.parquet', columns=['t_ns'] + [k + f for k in keys for f in ('_st', '_net', '_busy_s')])
    m = frame[['t_ns'] + [k + f for k in keys for f in ('_st', '_net', '_busy_s')]].merge(ref, on='t_ns', how='left', suffixes=('', '_ref'), indicator=True)
    out = dict(date=date, rows=len(frame), missing_rows=int((m._merge != 'both').sum()), mismatches={})
    for k in keys:
        st_bad = int((m[k + '_st'] != m[k + '_st_ref']).sum())
        a, b = m[k + '_net'].to_numpy(np.float64), m[k + '_net_ref'].to_numpy(np.float64)
        net_bad = int(((np.isnan(a) != np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 1e-6)).sum())
        a, b = m[k + '_busy_s'].to_numpy(np.float64), m[k + '_busy_s_ref'].to_numpy(np.float64)
        busy_bad = int(((np.isnan(a) != np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 1e-3)).sum())
        if st_bad or net_bad or busy_bad:
            out['mismatches'][k] = dict(st=st_bad, net=net_bad, busy=busy_bad)
    out['exact'] = not out['mismatches'] and not out['missing_rows']
    return out


def run_member(member_key, dates, data, output):
    receipts = []
    for d, tape, order in day_tapes(member_key, dates):
        frame, grid, r = build_day(d, tape, data)
        dest = Path(output) / f'{d}.parquet'; frame.to_parquet(dest, index=False)
        np.savez_compressed(Path(output) / f'{d}.npz', **grid)
        r.update(sha256=sha(dest), captures=order, parity=parity(frame, d))
        save_json(Path(output) / f'{d}.json', r); receipts.append(r)
        print(json.dumps(dict(date=d, n=r['decisions'], s=round(r['seconds'], 1), scan=round(r['scan_seconds'], 1),
                              exact=r['parity']['exact'])), flush=True)
    return receipts


def _job(args):
    return run_member(*args)


def prepare(data=DEFAULT_DATA, output=DEFAULT_OUTPUT, months=TRAIN_MONTHS, workers=5, limit=None):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    by = captures_by_date(MARKET, tuple(months)); groups = defaultdict(list); dates = []
    for m in months:
        man = json.loads((Path(data) / m / 'manifest.json').read_text(encoding='utf8'))
        dates += [r['date'] for r in man['days']]
    dates = sorted(dates)[:limit] if limit else sorted(dates)
    for d in dates:
        if not (out / f'{d}.json').exists() and by.get(d):
            groups[by[d][0]['member_key']].append(d)
    code = {p: sha(Path(__file__).with_name(p)) for p in ('path_scan.py', 'scalper.py', 'queue.py', 'passive.py', 'opportunity.py')}
    save_json(out / 'protocol.json', dict(version=VERSION, stride=STRIDE, edges=list(EDGES), plan_levels=[list(x) for x in PLAN_LEVELS],
        vol_levels=list(VOL_LEVELS), abs_levels=list(ABS_LEVELS), fee=FEE, code=code, semantics=__doc__.strip()))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, [(m, ds, data, output) for m, ds in sorted(groups.items())]))
    done = sorted(p.stem for p in out.glob('2*.json'))
    receipts = [json.loads((out / f'{d}.json').read_text(encoding='utf8')) for d in done]
    save_json(out / 'manifest.json', dict(status='COMPLETE' if len(done) == len(dates) else 'PARTIAL', version=VERSION,
        days=done, missing=[d for d in dates if d not in done], seconds=time.perf_counter() - started, code=code,
        parity_exact_days=sum(r['parity']['exact'] for r in receipts), decisions=sum(r['decisions'] for r in receipts)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--months', nargs='+', default=list(TRAIN_MONTHS)); p.add_argument('--workers', type=int, default=5)
    p.add_argument('--limit', type=int)
    a = p.parse_args(); prepare(output=a.output, months=a.months, workers=a.workers, limit=a.limit)
