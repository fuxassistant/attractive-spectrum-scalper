"""Causal market representation families for signature origins.

Families (each value has a known mask; unknown is never a fabricated zero):
  vol    - trailing sigma 60/600/3600s, spread, session clock    (signature X)
  price  - sigma-normalised returns, range position, session anchors (quote grid)
  flow   - long-window classified trade flow with unknown bounds, volume rate,
           book add/remove pressure, depth imbalance          (MBO seconds)
  micro  - the compact M1 row: 30 second-level book/flow/spectral fields
  event  - recognised event fields and prior-event context      (event job)

Seconds rows are used only when available_ns <= intent. Cumulative counters are
differenced only inside one epoch, with the window start present in that epoch.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.player.data import FEATURES
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from . import opportunity as op
from .signatures import DEFAULT_DATA, DEFAULT_OUTPUT as SIGNATURES

VERSION = 'CAUSAL_REPRESENTATION_V1'
DEFAULT_OUTPUT = 'data/derived/representation_v1'
WINDOWS = (300, 900, 1800, 3600)
COUNTERS = ('c_trade_buy', 'c_trade_sell', 'c_trade_unknown', 'c_add_bid', 'c_add_ask', 'c_remove_bid', 'c_remove_ask')
VOL = ('sigma60_ticks_per_sqrt_s', 'sigma600_ticks_per_sqrt_s', 'sigma3600_ticks_per_sqrt_s', 'spread_ticks',
       'minutes_since_london')


def family_columns(columns):
    """Value columns per family; masks travel with their values."""
    fam = dict(vol=[c for c in VOL], price=[], flow=[], micro=[], event=[])
    for c in columns:
        for name in ('price', 'flow', 'micro'):
            if c.startswith(name + '_'):
                fam[name].append(c)
        if c.startswith('ev_') or c in ('prior_events_300s', 'since_prior_event_s', 'since_prior_event_known',
                                        'event_bank_known', 'is_event'):
            fam['event'].append(c)
    fam['vol'] += ['sigma60_known', 'sigma600_known', 'sigma3600_known']
    return fam


def price_family(grid, k, sigma, sigma_known):
    out = {}; mid = grid['mid']; seg = grid['segment']; n = len(mid)
    here = ~np.isnan(mid[k])
    for w in WINDOWS:
        lo = k - w
        ok = here & (lo >= 0) & sigma_known & (sigma > 0)
        lo_c = np.clip(lo, 0, n - 1)
        ok &= ~np.isnan(mid[lo_c]) & (seg[lo_c] == seg[k])
        out[f'price_ret{w}_sigma'] = np.where(ok, (mid[k] - mid[lo_c]) / np.maximum(sigma * np.sqrt(w), 1e-9), 0.)
        out[f'price_ret{w}_known'] = ok
        pos = np.zeros(len(k)); pk = np.zeros(len(k), bool)
        for i in np.flatnonzero(here & (lo >= 0)):
            path = mid[lo[i]:k[i] + 1]
            if np.isnan(path).any() or (seg[lo[i]:k[i] + 1] != seg[k[i]]).any():
                continue
            hi_, lo_ = path.max(), path.min()
            if hi_ > lo_:
                pos[i] = (mid[k[i]] - lo_) / (hi_ - lo_); pk[i] = True
        out[f'price_range{w}_position'] = pos; out[f'price_range{w}_known'] = pk
    # Session anchors: first valid mid of the session, running extremes so far.
    first = np.flatnonzero(~np.isnan(mid))
    scale = np.maximum(sigma * np.sqrt(3600), 1e-9)
    if len(first):
        f0 = first[0]
        running_hi = np.fmax.accumulate(np.where(np.isnan(mid), -np.inf, mid))
        running_lo = np.fmin.accumulate(np.where(np.isnan(mid), np.inf, mid))
        ok = here & (k >= f0) & sigma_known & (sigma > 0)
        out['price_from_session_open_sigma1h'] = np.where(ok, (mid[k] - mid[f0]) / scale, 0.)
        out['price_from_session_high_sigma1h'] = np.where(ok, (mid[k] - running_hi[k]) / scale, 0.)
        out['price_from_session_low_sigma1h'] = np.where(ok, (mid[k] - running_lo[k]) / scale, 0.)
    else:
        ok = np.zeros(len(k), bool)
        for name in ('open', 'high', 'low'):
            out[f'price_from_session_{name}_sigma1h'] = np.zeros(len(k))
    out['price_session_known'] = ok
    return out


def load_seconds(parts):
    frames = []
    for number, part in enumerate(parts):
        s = pq.read_table(Path(part['root']) / 'seconds.parquet',
                          columns=['t_ns', 'available_ns', 'epoch', 'depth_bid_1', 'depth_bid_4', 'depth_bid_12',
                                   'depth_ask_1', 'depth_ask_4', 'depth_ask_12', 'book_valid'] + list(COUNTERS),
                          filters=[('t_ns', '>=', part['start_ns']), ('t_ns', '<', part['end_ns'])]).to_pandas()
        s['part'] = number
        frames.append(s)
    s = pd.concat(frames, ignore_index=True).sort_values(['t_ns', 'part'], kind='stable')
    return s.drop_duplicates('t_ns', keep='last').reset_index(drop=True)


def flow_family(s, intents):
    """Counters at the latest available row; window start must share its epoch."""
    t = s.t_ns.to_numpy(np.int64); avail = np.maximum.accumulate(s.available_ns.to_numpy(np.int64))
    epoch = s.epoch.to_numpy(np.int64) * 16 + s.part.to_numpy(np.int64)
    c = {n: s[n].to_numpy(float) for n in COUNTERS}
    i = np.searchsorted(avail, intents, side='right') - 1
    here = i >= 0; ic = np.clip(i, 0, None)
    out = {}
    for w in WINDOWS:
        j = np.searchsorted(t, t[ic] - w * NS, side='left')
        jc = np.clip(j, 0, len(t) - 1)
        ok = here & (j < len(t)) & (t[jc] == t[ic] - w * NS) & (epoch[jc] == epoch[ic])
        d = {n: np.where(ok, c[n][ic] - c[n][jc], 0.) for n in COUNTERS}
        b, sl, u = d['c_trade_buy'], d['c_trade_sell'], d['c_trade_unknown']; v = b + sl + u
        vk = ok & (v > 0)
        out[f'flow_pressure{w}_low'] = np.where(vk, (b - sl - u) / np.maximum(v, 1), 0.)
        out[f'flow_pressure{w}_high'] = np.where(vk, (b - sl + u) / np.maximum(v, 1), 0.)
        out[f'flow_unknown{w}_fraction'] = np.where(vk, u / np.maximum(v, 1), 0.)
        out[f'flow_pressure{w}_known'] = vk
        out[f'flow_log_volume{w}'] = np.where(ok, np.log1p(v), 0.)
        book = d['c_add_bid'] + d['c_add_ask'] + d['c_remove_bid'] + d['c_remove_ask']
        bk = ok & (book > 0)
        out[f'flow_book{w}_pressure'] = np.where(bk, ((d['c_add_bid'] - d['c_remove_bid']) -
                                                     (d['c_add_ask'] - d['c_remove_ask'])) / np.maximum(book, 1), 0.)
        out[f'flow_book{w}_known'] = bk
        out[f'flow_volume{w}_known'] = ok
    v300, v3600 = out['flow_log_volume300'], out['flow_log_volume3600']
    rk = out['flow_volume300_known'] & out['flow_volume3600_known'] & (v3600 > 0)
    out['flow_volume_rate_ratio'] = np.where(rk, (np.expm1(v300) / 300) / np.maximum(np.expm1(v3600) / 3600, 1e-9), 0.)
    out['flow_volume_rate_known'] = rk
    valid = here & s.book_valid.to_numpy(bool)[ic]
    for lvl in (1, 4, 12):
        bid, ask = s[f'depth_bid_{lvl}'].to_numpy(float)[ic], s[f'depth_ask_{lvl}'].to_numpy(float)[ic]
        k = valid & (bid + ask > 0)
        out[f'flow_depth{lvl}_imbalance'] = np.where(k, (bid - ask) / np.maximum(bid + ask, 1), 0.)
        out[f'flow_depth{lvl}_known'] = k
    return out


def micro_family(day, intents):
    """Latest completed M1 row whose decision time is not after the intent."""
    x = np.asarray(day.x, dtype=float); dt = np.asarray(day.decision_t)
    i = np.searchsorted(dt, intents, side='right') - 1
    ok = i >= 0; ic = np.clip(i, 0, None)
    names = list(day.meta['feature_names'])
    out = {}
    for f in FEATURES:
        known = ok & (x[ic, names.index(f + '__known')] > .5)
        out['micro_' + f] = np.where(known, x[ic, names.index(f)], 0.)
        out['micro_' + f + '_known'] = known
    out['micro_age_seconds'] = np.where(ok, (intents - dt[ic]) / NS, 0.)
    return out


def prepare_day(sig_day, data, output, code):
    started = time.perf_counter(); date = Path(sig_day).name
    dest = Path(output) / f'{date}.parquet'; receipt = Path(output) / f'{date}.json'
    identity = dict(signature_manifest=sha(Path(sig_day) / 'manifest.json'), code=code)
    if receipt.exists():
        r = json.loads(receipt.read_text(encoding='utf8'))
        if r['identity'] != identity or sha(dest) != r['sha256']:
            raise ValueError('Completed representation changed: ' + date)
        return r
    x = pd.read_parquet(Path(sig_day) / 'x.parquet')
    path = Path(data) / date[:7] / date
    day = MonthDay(path); tape = QuoteTape.from_day(day)
    grid = op.second_grid(tape.quote_t, tape.bid, tape.ask, tape.valid, tape.epoch,
                          int(day.meta['start_ns']), int(day.meta['end_ns']))
    intents = x.intent_ns.to_numpy(np.int64); k = op.origin_index(grid, intents)
    sigma = x.sigma600_ticks_per_sqrt_s.to_numpy(); known = x.sigma600_known.to_numpy(bool)
    cols = dict(origin_id=x.origin_id.to_numpy())
    cols.update(price_family(grid, k, sigma, known))
    cols.update(flow_family(load_seconds(day.meta['quote_parts']), intents))
    cols.update(micro_family(day, intents))
    frame = pd.DataFrame(cols)
    for c in frame.columns:
        if frame[c].dtype == bool:
            frame[c] = frame[c].astype(np.int8)
    values = frame.drop(columns='origin_id').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite representation value ' + date)
    frame.to_parquet(dest, index=False)
    r = dict(status='COMPLETE', version=VERSION, date=date, identity=identity, rows=len(frame),
             columns=len(frame.columns) - 1, sha256=sha(dest), seconds=time.perf_counter() - started)
    save_json(receipt, r); QUOTE_CACHE.clear()
    return r


def _job(args):
    return prepare_day(*args)


def prepare(signatures=SIGNATURES, data=DEFAULT_DATA, output=DEFAULT_OUTPUT, workers=4):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(signatures) / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'COMPLETE':
        raise ValueError('Representation requires complete signatures')
    code = {p.name: sha(p) for p in (Path(__file__), Path(op.__file__))}
    jobs = [(Path(signatures) / d['date'], data, out, code) for d in manifest['days']]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_job, jobs):
            results.append(r); print(json.dumps({k: r[k] for k in ('date', 'rows', 'columns', 'seconds')}), flush=True)
    save_json(out / 'manifest.json', dict(status='COMPLETE', version=VERSION, days=results, code=code,
        signature_manifest_sha256=sha(Path(signatures) / 'manifest.json'), windows=list(WINDOWS),
        semantics='Causal at intent: quote grid as-of boundary, seconds rows by availability, M1 by decision time.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--signatures', default=SIGNATURES); p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--output', default=DEFAULT_OUTPUT); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(); prepare(a.signatures, a.data, a.output, a.workers)
