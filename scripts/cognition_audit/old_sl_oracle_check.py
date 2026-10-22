"""What the old SL monolith's validation metric rewards, measured on our GC replay.

The old validator (old_monolith_sl.py, validate_soft) books every trade at the best sustained mid inside
[t+60 s, t+360 s] (exit at the extremum, mid to mid, fee $0.30) and tunes the threshold on the same
validation data. Here the same metric is computed for entries that need no model:
  random side at every decision; random side at the 10% most volatile decisions (causal sigma600);
  side = sign of the last 60 s return at those decisions;
against two executable outcomes at the same entries: a taker round trip held 360 s, and the scalper's
volatility-scaled taker plan a600s from data/derived/path_v1.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.cognition import opportunity as op
from spectrum.cognition.path_model import dates_of, SPLIT

FEE, START, H, SUSTAIN, EPS_TICKS = .30, 60, 360, 5, 1.0
USD_PER_TICK = .10


def day(date, rng):
    md = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); q = QuoteTape.from_day(md)
    start, end = int(md.meta['start_ns']), int(md.meta['end_ns'])
    g = op.second_grid(q.quote_t, q.bid, q.ask, q.valid, q.epoch, start, end)
    mid, spread, seg = g['mid'].astype(float), g['spread'].astype(float), g['segment']
    p = pd.read_parquet(Path('data/derived/path_v1') / f'{date}.parquet', columns=['t_ns', 'sigma600', 'T_L_a600s_net', 'T_S_a600s_net'])
    k = ((p.t_ns.to_numpy() - start) // NS).astype(np.int64)
    n = len(mid)
    s = pd.Series(mid)
    smin = s[::-1].rolling(SUSTAIN, min_periods=1).min()[::-1].to_numpy() - EPS_TICKS
    smax = s[::-1].rolling(SUSTAIN, min_periods=1).max()[::-1].to_numpy() + EPS_TICKS
    w = H - START
    fmax = pd.Series(smin)[::-1].rolling(w, min_periods=1).max()[::-1].to_numpy()
    fmin = pd.Series(smax)[::-1].rolling(w, min_periods=1).min()[::-1].to_numpy()
    ok = (k + H < n)
    k = k[ok]; p = p[ok].reset_index(drop=True)
    a, b = k + START, k + H
    good = np.isfinite(mid[k]) & np.isfinite(mid[b]) & np.isfinite(fmax[a]) & np.isfinite(fmin[a]) & (seg[k] == seg[b])
    k, a, b, p = k[good], a[good], b[good], p[good].reset_index(drop=True)
    up = (fmax[a] - mid[k]) * USD_PER_TICK; dn = (mid[k] - fmin[a]) * USD_PER_TICK
    move = (mid[b] - mid[k]) * USD_PER_TICK; cost = (spread[k] + spread[b]) / 2 * USD_PER_TICK
    ret60 = np.where(k >= 60, mid[k] - mid[np.clip(k - 60, 0, None)], 0.)
    rand = np.where(rng.random(len(k)) < .5, 1, -1); mom = np.where(ret60 >= 0, 1, -1)
    vol = p.sigma600.to_numpy() >= np.nanquantile(p.sigma600, .9)
    out = []
    for name, side, sel in (('random, all', rand, np.ones(len(k), bool)), ('random, top-10% sigma', rand, vol),
                            ('momentum 60 s, top-10% sigma', mom, vol)):
        old = np.where(side > 0, up, dn)[sel] - FEE
        hold = (side * move - cost)[sel] - FEE
        plan = np.where(side > 0, p.T_L_a600s_net, p.T_S_a600s_net)[sel]
        out.append(dict(date=date, policy=name, n=int(sel.sum()), old=float(old.sum()), hold=float(np.nansum(hold)),
                        plan=float(np.nansum(plan)), plan_n=int(np.isfinite(plan).sum()), old_win=int((old > 0).sum())))
    QUOTE_CACHE.clear()
    return out


if __name__ == '__main__':
    months = SPLIT['B'] if len(sys.argv) < 2 else tuple(sys.argv[1:])
    rng = np.random.default_rng(5); rows = []
    for d in dates_of(months):
        rows += day(d, rng)
    f = pd.DataFrame(rows); g = f.groupby('policy', sort=False).sum(numeric_only=True)
    res = {pol: dict(attempts=int(r.n), old_metric_per_trade=r.old / r.n, old_win_rate=r.old_win / r.n,
                     taker_hold360_per_trade=r.hold / r.n, scalper_a600s_per_trade=r.plan / max(r.plan_n, 1))
           for pol, r in g.iterrows()}
    print(json.dumps(res, indent=1))
    Path('reports/scalper_mind_v1/old_sl_oracle_check.json').write_text(json.dumps(dict(months=list(months), result=res), indent=2), encoding='utf8')
