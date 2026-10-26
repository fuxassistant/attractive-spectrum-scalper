"""Reusable formulas extracted from spectrum/cognition/representation.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.core.schema import NS
from spectrum.representation.schema import FEATURES

VERSION = 'CAUSAL_REPRESENTATION_V1'


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


def flow_family(s, intents, windows=WINDOWS, rate=(300, 3600)):
    """Counters at the latest available row; window start must share its epoch."""
    t = s.t_ns.to_numpy(np.int64); avail = np.maximum.accumulate(s.available_ns.to_numpy(np.int64))
    epoch = s.epoch.to_numpy(np.int64) * 16 + s.part.to_numpy(np.int64)
    c = {n: s[n].to_numpy(float) for n in COUNTERS}
    i = np.searchsorted(avail, intents, side='right') - 1
    here = i >= 0; ic = np.clip(i, 0, None)
    out = {}
    for w in windows:
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
    short, long_ = rate
    v300, v3600 = out[f'flow_log_volume{short}'], out[f'flow_log_volume{long_}']
    rk = out[f'flow_volume{short}_known'] & out[f'flow_volume{long_}_known'] & (v3600 > 0)
    out['flow_volume_rate_ratio'] = np.where(rk, (np.expm1(v300) / short) / np.maximum(np.expm1(v3600) / long_, 1e-9), 0.)
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
