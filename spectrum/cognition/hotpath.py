"""Hot path: features anchored at the previous session's London open, targets at several scales (HOTPATH_V1).

The hypothesis under test: the interests of market participants that matter today were formed from
yesterday's London open onwards. For every second of day d the "hot path" is
    [London open of the previous trading day, end of that day] + [start of day d, now],
so the window grows through the day. All features are causal (as-of the second boundary; MBO counters
by availability time); every row of every day is anchored at its own previous London open.

Families (distances in units D = sigma600 * sqrt(600) ticks, returns in sigma600 * sqrt(w)):
  path   multi-scale returns of the mid;
  flow   trade imbalance, book pressure, depth imbalance, volume rate;
  levels anchored VWAP since the previous London open, today's VWAP, previous session high/low/close,
         window and today's extremes, volume profile since the previous London open (POC distance,
         position in the 70% value area, volume density at the current price);
  time   minutes since London open (only for global models).
Targets (Y): z_h = (mid(t+h) - mid(t)) / (sigma600 * sqrt(h)), mature iff every second is supported.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from numba import njit
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from .signatures import DEFAULT_DATA
from .representation import load_seconds
from . import opportunity as op

VERSION = 'HOTPATH_V1'
OUTPUT = 'data/derived/hotpath_v1'
GRID = 10
SCALES = (30, 60, 120, 300, 600)
RET_W = (10, 30, 60, 120, 300, 600, 1800)
FLOW_W = (30, 120, 600)
PATH = [f'ret{w}' for w in RET_W]
FLOW = [f'imb{w}' for w in FLOW_W] + ['book30', 'book120', 'depth1', 'depth4', 'vol_rate']
LEVELS = ['d_avwap', 'd_vwap_today', 'd_prev_hi', 'd_prev_lo', 'd_prev_close', 'd_win_hi', 'd_win_lo',
          'd_today_hi', 'd_today_lo', 'd_poc', 'va_pos', 'density']
TIME = ['minutes_london']
FAMILIES = dict(path=PATH, flow=FLOW, levels=LEVELS)


def base_day(date, data=DEFAULT_DATA):
    """Per-second arrays over the whole day file: mid, spread, support, sigma600, volume and book by availability."""
    md = MonthDay(Path(data) / date[:7] / date); q = QuoteTape.from_day(md)
    start, end = int(md.meta['start_ns']), int(md.meta['end_ns'])
    g = op.second_grid(q.quote_t, q.bid, q.ask, q.valid, q.epoch, start, end)
    n = len(g['mid']); k = np.arange(n)
    c = op.causal(g, k)
    s = load_seconds(md.meta['quote_parts'])
    avail = np.maximum.accumulate(s.available_ns.to_numpy(np.int64))
    idx = np.searchsorted(avail, g['boundary'], side='right') - 1; here = idx >= 0; ic = np.clip(idx, 0, None)
    key = np.where(here, s.epoch.to_numpy(np.int64)[ic] * 16 + s.part.to_numpy(np.int64)[ic], -1)
    same = np.r_[False, (key[1:] == key[:-1]) & (key[1:] >= 0)]
    out = dict(t_ns=g['boundary'], mid=g['mid'], spread=g['spread'], seg=g['segment'], step=g['step'],
               sigma=c['sigma600_ticks_per_sqrt_s'], sigma_known=c['sigma600_known'],
               sigma60=c['sigma60_ticks_per_sqrt_s'], sigma3600=c['sigma3600_ticks_per_sqrt_s'], sigma3600_known=c['sigma3600_known'])
    for name in ('c_trade_buy', 'c_trade_sell', 'c_trade_unknown', 'c_add_bid', 'c_add_ask', 'c_remove_bid', 'c_remove_ask'):
        cum = np.where(here, s[name].to_numpy(float)[ic], 0.)
        out[name[2:]] = np.where(same, np.maximum(np.diff(np.r_[0., cum]), 0.), 0.)
    valid = here & s.book_valid.to_numpy(bool)[ic]
    for lvl in (1, 4, 12):
        b, a = s[f'depth_bid_{lvl}'].to_numpy(float)[ic], s[f'depth_ask_{lvl}'].to_numpy(float)[ic]
        out[f'depth{lvl}'] = np.where(valid & (a + b > 0), (b - a) / np.maximum(a + b, 1), 0.)
        out[f'depth{lvl}_total'] = np.where(valid, a + b, np.nan)
    meta = dict(date=date, start_ns=start, end_ns=end, london=int((int(md.meta['london_ns']) - start) // NS),
                ny_close=int((int(md.meta['ny_close_ns']) - start) // NS), n=n)
    QUOTE_CACHE.clear()
    return out, meta


@njit(cache=True)
def profile_scan(price_bin, vol, n_prev, sample, nbins, va_share):
    """Volume-at-price histogram over the concatenated hot path; at each sample (index into the concatenation)
    returns POC bin, value-area low/high bins and volume within +-2 bins of the current price."""
    hist = np.zeros(nbins); poc = -1; total = 0.
    m = len(sample); poc_o = np.full(m, -1); val_o = np.full(m, -1); vah_o = np.full(m, -1); dens_o = np.zeros(m)
    tot_o = np.zeros(m); occ_o = np.zeros(m)
    occupied = 0; si = 0
    for i in range(len(price_bin)):
        b = price_bin[i]
        if b >= 0 and vol[i] > 0:
            if hist[b] == 0:
                occupied += 1
            hist[b] += vol[i]; total += vol[i]
            if poc < 0 or hist[b] > hist[poc]:
                poc = b
        while si < m and sample[si] == i:
            if poc >= 0 and total > 0:
                lo = poc; hi = poc; acc = hist[poc]
                while acc < va_share * total:
                    left = hist[lo - 1] if lo > 0 else -1.
                    right = hist[hi + 1] if hi < nbins - 1 else -1.
                    if left < 0 and right < 0:
                        break
                    if right >= left:
                        hi += 1; acc += right
                    else:
                        lo -= 1; acc += left
                poc_o[si] = poc; val_o[si] = lo; vah_o[si] = hi
                if b >= 0:
                    d = 0.
                    for j in range(max(0, b - 2), min(nbins, b + 3)):
                        d += hist[j]
                    dens_o[si] = d
                tot_o[si] = total; occ_o[si] = occupied
            si += 1
    return poc_o, val_o, vah_o, dens_o, tot_o, occ_o


def window_sum(x, w):
    c = np.r_[0., np.cumsum(x)]; k = np.arange(len(x))
    return c[k + 1] - c[np.clip(k + 1 - w, 0, None)]


def features_day(prev, today, grid=GRID):
    """Rows every `grid` seconds of today's file with causal hot-path features and targets."""
    (bp, mp), (bt, mt) = prev, today
    n = mt['n']; k = np.arange(0, n, grid)
    mid, sig = bt['mid'], bt['sigma']
    ok = np.isfinite(mid[k]) & bt['sigma_known'][k] & (sig[k] > 0)
    k = k[ok]; S = sig[k]; D = S * np.sqrt(600.)
    f = {}
    seg = bt['seg']
    for w in RET_W:
        lo = k - w; good = (lo >= 0)
        loc = np.clip(lo, 0, None)
        good &= np.isfinite(mid[loc]) & (seg[loc] == seg[k])
        f[f'ret{w}'] = np.where(good, (mid[k] - mid[loc]) / (S * np.sqrt(w)), 0.)
    vb, vs, vu = bt['trade_buy'], bt['trade_sell'], bt['trade_unknown']
    for w in FLOW_W:
        b, s_, u = window_sum(vb, w)[k], window_sum(vs, w)[k], window_sum(vu, w)[k]
        f[f'imb{w}'] = np.where(b + s_ + u > 0, (b - s_) / np.maximum(b + s_ + u, 1), 0.)
    for w in (30, 120):
        ab, aa, rb, ra = (window_sum(bt[x], w)[k] for x in ('add_bid', 'add_ask', 'remove_bid', 'remove_ask'))
        tot = ab + aa + rb + ra
        f[f'book{w}'] = np.where(tot > 0, ((ab - rb) - (aa - ra)) / np.maximum(tot, 1), 0.)
    f['depth1'], f['depth4'] = bt['depth1'][k], bt['depth4'][k]
    vol = vb + vs + vu
    v60, v1800 = window_sum(vol, 60)[k], window_sum(vol, 1800)[k]
    f['vol_rate'] = np.log((v60 / 60 + 1e-3) / (v1800 / 1800 + 1e-3))
    # hot path: previous session from its London open + today up to k
    lp = mp['london']
    pm, pv = bp['mid'][lp:], (bp['trade_buy'] + bp['trade_sell'] + bp['trade_unknown'])[lp:]
    pmid = np.where(np.isfinite(pm), pm, np.nan)
    cm = np.r_[pmid, mid]; cv = np.r_[pv, vol]; good = np.isfinite(cm)
    cvx = np.where(good, cv, 0.); cpx = np.where(good, cv * np.nan_to_num(cm), 0.)
    off = len(pm); kk = off + k
    CV, CP = np.cumsum(cvx), np.cumsum(cpx)
    avwap = CP[kk] / np.maximum(CV[kk], 1e-9)
    f['d_avwap'] = np.where(CV[kk] > 0, (mid[k] - avwap) / D, 0.)
    lt = mt['london']; anchor = np.where(k >= lt, off + lt, off)
    va = CV[kk] - np.where(anchor > 0, CV[np.clip(anchor - 1, 0, None)], 0.)
    pa = CP[kk] - np.where(anchor > 0, CP[np.clip(anchor - 1, 0, None)], 0.)
    f['d_vwap_today'] = np.where(va > 0, (mid[k] - pa / np.maximum(va, 1e-9)) / D, 0.)
    ph, pl = np.nanmax(pmid) if np.isfinite(pmid).any() else np.nan, np.nanmin(pmid) if np.isfinite(pmid).any() else np.nan
    pc = pmid[np.isfinite(pmid)][-1] if np.isfinite(pmid).any() else np.nan
    for name, lvl in (('d_prev_hi', ph), ('d_prev_lo', pl), ('d_prev_close', pc)):
        f[name] = np.where(np.isfinite(lvl), (mid[k] - lvl) / D, 0.)
    run_hi = np.fmax.accumulate(np.where(good, cm, -np.inf)); run_lo = np.fmin.accumulate(np.where(good, cm, np.inf))
    f['d_win_hi'] = (mid[k] - run_hi[kk]) / D; f['d_win_lo'] = (mid[k] - run_lo[kk]) / D
    tm = np.where(np.isfinite(mid), mid, np.nan)
    th, tl = np.empty(len(k)), np.empty(len(k))
    hi_from_l = np.fmax.accumulate(np.where(np.arange(n) >= lt, np.nan_to_num(tm, nan=-np.inf), -np.inf))
    lo_from_l = np.fmin.accumulate(np.where(np.arange(n) >= lt, np.nan_to_num(tm, nan=np.inf), np.inf))
    hi_all = np.fmax.accumulate(np.nan_to_num(tm, nan=-np.inf)); lo_all = np.fmin.accumulate(np.nan_to_num(tm, nan=np.inf))
    th = np.where(k >= lt, hi_from_l[k], hi_all[k]); tl = np.where(k >= lt, lo_from_l[k], lo_all[k])
    f['d_today_hi'] = (mid[k] - th) / D; f['d_today_lo'] = (mid[k] - tl) / D
    fin = cm[good]; pmin = np.floor(fin.min()) - 3; nb = int(np.floor(fin.max()) - pmin) + 6
    pbin = np.where(good, np.floor(np.nan_to_num(cm) - pmin), -1).astype(np.int64)
    poc, val, vah, dens, tot, occ = profile_scan(pbin, cv.astype(float), off, kk.astype(np.int64), nb, .70)
    have = poc >= 0
    f['d_poc'] = np.where(have, (mid[k] - (poc + pmin + .5)) / D, 0.)
    width = np.maximum(vah - val + 1, 1)
    f['va_pos'] = np.where(have, ((np.floor(mid[k]) - pmin) - val + .5) / width - .5, 0.)
    f['density'] = np.where(have & (tot > 0), np.log((dens / 5 + 1e-6) / (tot / np.maximum(occ, 1) + 1e-6)), 0.)
    f['minutes_london'] = (k - lt) / 60.
    y = {}
    step_cum = np.r_[0, np.cumsum(bt['step'])]
    for h in SCALES:
        e = k + h; inside = e < n; ec = np.clip(e, 0, n - 1)
        mature = inside & (step_cum[ec + 1] - step_cum[np.clip(k + 1, 0, n)] == h)
        y[f'z{h}'] = np.where(mature, (mid[ec] - mid[k]) / (S * np.sqrt(h)), np.nan)
        y[f'move{h}'] = np.where(mature, mid[ec] - mid[k], np.nan)
    session = (k >= lt) & (k < mt['ny_close'])
    frame = pd.DataFrame(dict(date=mt['date'], t_ns=bt['t_ns'][k], k=k, session=session, sigma=S, spread=bt['spread'][k],
                              **{c: np.asarray(v, np.float32) for c, v in f.items()},
                              **{c: np.asarray(v, np.float32) for c, v in y.items()}))
    return frame


def trading_days(months, data=DEFAULT_DATA):
    days = []
    for m in months:
        man = json.loads((Path(data) / m / 'manifest.json').read_text(encoding='utf8'))
        days += [r['date'] for r in man['days']]
    return sorted(days)


def _job(pair):
    prev, today, output = pair
    out = Path(output) / f'{today}.parquet'
    if out.exists():
        return today, 'exists'
    t0 = time.perf_counter()
    try:
        bp, bt = base_day(prev), base_day(today)
    except Exception as e:  # holidays without quotes
        return today, f'skip: {e!r}'[:200]
    if not np.isfinite(bt[0]['mid']).any() or not np.isfinite(bp[0]['mid'][bp[1]['london']:]).any():
        return today, 'skip: no quotes'
    fr = features_day(bp, bt); fr['prev_date'] = prev
    fr.to_parquet(out, index=False)
    return today, f'{len(fr)} rows {time.perf_counter() - t0:.1f}s'


def prepare(months, output=OUTPUT, workers=6):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    days = trading_days(months)
    receipts = Path('data/derived/path_v1')   # trading sessions = days with at least one scalper decision
    live = [d for d in days if not (receipts / f'{d}.json').exists()
            or json.loads((receipts / f'{d}.json').read_text(encoding='utf8'))['decisions'] > 0]
    pairs = [(live[i - 1], live[i], output) for i in range(1, len(live))]
    save_json(out / 'protocol.json', dict(version=VERSION, grid=GRID, scales=list(SCALES), families=FAMILIES, semantics=__doc__.strip()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for d, msg in pool.map(_job, pairs):
            print(d, msg, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--months', nargs='+', required=True); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); prepare(a.months, workers=a.workers)
