"""Sequential 1 oz scalper account on the MBO tape with a passive take-profit.

For every trade the policy actually takes, the entry (taker or queued) and a
resting take-profit limit are simulated on the MBO queue kernel V5 with the
fill-proof invariant; stop, time limit and session deadline exit as takers
on mid-move triggers (scalper.scan_exits with the take disabled). The take
limit rests at the tick-rounded price mid_at_fill +- m_take*u, placed at
fill+100 ms; if it is already marketable it fills at the executable price.
Whichever of {take fill, stop/time/deadline} comes first ends the trade; a
support loss before the take fill censors it. A trade's real exit sets when
the account is free again, so the whole day is simulated in order.

Thresholds are chosen without B: models fit on A1 (Oct-Nov 2025), threshold
by the A2 (Dec 2025) sequential result; final models fit on all of A and the
chosen threshold is applied to B (Jan-Feb 2026), with controls.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from .passive import Quotes, LATENCY, TIMEOUT, TICK, day_tapes, captures_by_date, MARKET
from .signatures import DEFAULT_DATA
from .signature_report import day_boot
from . import queue as qu
from . import scalper as sc
from . import scalper_policy as sp

REPORT = 'reports/scalper_v2'
THRESHOLDS = (0., .05, .10, .20, .30)
A1 = ('2025-10', '2025-11'); A2 = ('2025-12',)


def _probe_one(tape, place, side, price, end):
    hi = min(int(np.searchsorted(tape['ts'], end, side='right')) + 1, len(tape['ts']))
    sub = {k: v[:hi] for k, v in tape.items()}
    st, fill, kind, _ = qu.probe(sub, [place], [side], [price], [end])
    qu.check_fill_proofs(sub, [place], [side], [price], st, fill)
    return int(st[0]), int(fill[0])


def entry(q, tape, t, mode, side):
    """(quote index, entry price, entry ns) or None when no fill (EMPTY)."""
    if mode == 'T':
        j = int(sc.taker_entries(q.t, q.ok, np.array([t], np.int64), LATENCY, TIMEOUT)[0])
        if j < 0:
            return None
        return j, int(q.ask[j] if side > 0 else q.bid[j]), int(q.t[j])
    d = q.asof(t)
    if d < 0:
        return None
    price = int(q.bid[d] if side > 0 else q.ask[d]); place = t + LATENCY
    a = int(np.searchsorted(q.t, place, side='right'))
    if a < len(q.t) and q.ok[a] and ((side > 0 and q.ask[a] <= price) or (side < 0 and q.bid[a] >= price)):
        return a, int(q.ask[a] if side > 0 else q.bid[a]), int(q.t[a])
    st, fill = _probe_one(tape, place, side, price, place + sc.WAIT_S * NS)
    if st != qu.FILLED:
        return None
    j = int(np.searchsorted(q.t, fill, side='left'))
    return (j, price, fill) if j < len(q.t) else None


def trade(q, tape, t, mode, side, plan, sigma, passive_tp):
    """Simulate one chosen trade. Returns (status, net_usd, end_ns)."""
    name, trig, hold, take, stop = plan
    e = entry(q, tape, t, mode, side)
    if e is None:
        return sc.EMPTY, 0., t + LATENCY + (sc.WAIT_S if mode == 'P' else 2) * NS
    j, px, t_fill = e
    unit = sigma * np.sqrt(hold) * TICK
    horizon = int(q.t[j]) + (hold + 5) * NS
    st = np.zeros(1, np.int8); ex = np.zeros(1, np.int64); net = np.full(1, np.nan)
    sc.scan_exits(q.t, q.bid, q.ask, q.ok, q.epoch, q.deadline, np.array([j]), np.array([side]), np.array([px], np.int64),
                  np.array([True]), np.array([np.inf if passive_tp else take * unit]), np.array([stop * unit]),
                  np.array([hold * NS], np.int64), np.array([horizon], np.int64), st, ex, net)
    if passive_tp:
        mid0 = (int(q.bid[j]) + int(q.ask[j])) / 2
        raw = mid0 + side * take * unit
        tp = int(np.ceil(raw / TICK) * TICK) if side > 0 else int(np.floor(raw / TICK) * TICK)
        if side * (tp - px) < TICK:
            tp = px + side * TICK
        place = t_fill + LATENCY; end = min(int(ex[0]), int(q.t[j]) + hold * NS)
        a = int(np.searchsorted(q.t, place, side='right'))
        if a < len(q.t) and q.ok[a] and q.t[a] <= end and ((side > 0 and q.bid[a] >= tp) or (side < 0 and q.ask[a] <= tp)):
            xp = int(q.bid[a] if side > 0 else q.ask[a])
            return sc.COMPLETE, side * (xp - px) / 1e9 - sc.FEE, int(q.t[a])
        if place < end:
            pst, pfill = _probe_one(tape, place, -side, tp, end)
            if pst == qu.FILLED and pfill <= end:
                return sc.COMPLETE, side * (tp - px) / 1e9 - sc.FEE, pfill
    if st[0] == sc.COMPLETE:
        return sc.COMPLETE, float(net[0]) * TICK / 1e9 - sc.FEE, int(ex[0])
    return sc.CENSORED, np.nan, int(ex[0])


def run_day(day_frame, preds, q, tape, threshold, passive_tp, random_q=None, seed=0):
    """Continuous decisions over one day; preds: key -> array aligned with day_frame."""
    keys, legal, score, plans = [], [], [], {}
    for plan in sp.ADAPTIVE:
        name, trig, hold, take, stop = plan
        spread_ok, cost_ok = sp.gates(day_frame, hold, take)
        for mode in sc.MODES:
            for s in (1, -1):
                key = f'{mode}_{sc.SIDES[s]}_{name}'; stc = day_frame[key + '_st'].to_numpy()
                keys.append(key); plans[key] = (mode, s, plan)
                legal.append(spread_ok & cost_ok & np.isin(stc, (sc.COMPLETE, sc.EMPTY, sc.CENSORED)))
                score.append(preds[key])
    legal = np.stack(legal, 1); score = np.where(legal, np.stack(score, 1), -np.inf)
    rng = np.random.default_rng(seed)
    if random_q is not None:
        score = np.where(legal, rng.random(score.shape), -np.inf)
    best = score.argmax(1); best_score = score[np.arange(len(day_frame)), best]
    t = day_frame.t_ns.to_numpy(); sigma = day_frame.sigma600.to_numpy(); busy = -1; trades = []
    for i in range(len(day_frame)):
        if t[i] < busy or not np.isfinite(best_score[i]):
            continue
        if random_q is None and best_score[i] <= threshold:
            continue
        if random_q is not None and rng.random() >= random_q:
            continue
        mode, s, plan = plans[keys[best[i]]]
        status, net, end = trade(q, tape, int(t[i]), mode, s, plan, float(sigma[i]), passive_tp)
        trades.append(dict(t_ns=int(t[i]), key=keys[best[i]], status=status, net=net, hold_s=(end - t[i]) / NS))
        busy = end if status != sc.CENSORED else int(t[i]) + (plan[2] + 35) * NS
    return trades


def _worker(args):
    member, dates, model_path, configs, data = args
    with open(model_path, 'rb') as fh:
        models = pickle.load(fh)
    out = []
    for date, tape, _ in day_tapes(member, dates):
        started = time.perf_counter()
        f = pd.read_parquet(Path(data) / f'{date}.parquet')
        day = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); q = Quotes(QuoteTape.from_day(day))
        X = {s: sp.side_features(f, s) for s in (1, -1)}
        for cfg in configs:
            m = models[cfg['models']]
            preds = {key: mdl.predict(X[side]).astype(np.float32) for key, (side, mdl) in m.items()}
            rq = cfg.get('random_q', {}).get(date) if cfg.get('random_q') is not None else None
            if cfg.get('random_q') is not None and rq is None:
                continue
            tr = run_day(f, preds, q, tape, cfg['threshold'], cfg['passive_tp'], random_q=rq, seed=int(date.replace('-', '')))
            out.append(dict(config=cfg['name'], date=date, trades=tr))
        QUOTE_CACHE.clear()
        print(json.dumps(dict(date=date, configs=len(configs), seconds=round(time.perf_counter() - started, 1))), flush=True)
    return out


def fit_models(f, months):
    """Attempt-value model per (mode, adaptive plan, side), on every TRAIN_STRIDE-th second of the months."""
    sel = f.date.str[:7].isin(months).to_numpy() & (np.arange(len(f)) % sp.TRAIN_STRIDE == 0)
    X = {s: sp.side_features(f, s) for s in (1, -1)}; models = {}
    for name, trig, hold, take, stop in sp.ADAPTIVE:
        for mode in sc.MODES:
            for s in (1, -1):
                key = f'{mode}_{sc.SIDES[s]}_{name}'
                y = f[key + '_net'].to_numpy(np.float64); ok = sel & np.isfinite(y)
                models[key] = (s, sp.model().fit(X[s][ok], y[ok]))
    return models


def run_configs(configs, dates, model_path, data, workers):
    by = captures_by_date(MARKET, tuple(sorted({d[:7] for d in dates}))); groups = defaultdict(list)
    for d in dates:
        groups[by[d][0]['member_key']].append(d)
    results = []
    # one task per (month member, config): a month lives in one member, so splitting
    # by config is what lets independent sequential simulations run in parallel
    tasks = [(m, ds, model_path, [cfg], data) for m, ds in sorted(groups.items()) for cfg in configs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_worker, tasks):
            results.extend(r)
    rows = [dict(config=r['config'], date=r['date'], **t) for r in results for t in r['trades']]
    return pd.DataFrame(rows), sorted(dates)


def summarize(tr, dates):
    out = {}
    for name in sorted(tr.config.unique()) if len(tr) else []:
        g = tr[tr.config == name]; filled = g[g.status == sc.COMPLETE]
        daily = pd.DataFrame(dict(date=dates)).merge(g.groupby('date', as_index=False).net.sum(), on='date', how='left')
        daily['net'] = pd.to_numeric(daily.net, errors='coerce').fillna(0.)
        eq = g.sort_values(['date', 't_ns']).net.fillna(0).cumsum().to_numpy()
        out[name] = dict(trades=len(g), filled=len(filled), net=float(g.net.sum()),
                         per_trade=float(filled.net.mean()) if len(filled) else None,
                         win_rate=float((filled.net > 0).mean()) if len(filled) else None,
                         per_day=len(g) / len(dates), daily=day_boot(daily, 'net'),
                         max_drawdown=float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()),
                         median_hold_s=float(filled.hold_s.median()) if len(filled) else None)
    return out


def random_rates(f, dates, target_per_day, models, threshold):
    """Per-day entry probability giving the policy's trade count, solved on the cheap label-based simulation."""
    rates = {}
    X = {s: sp.side_features(f, s) for s in (1, -1)}
    for d in dates:
        n = target_per_day.get(d, 0)
        if not n:
            continue
        g = f[f.date == d].reset_index(drop=True); preds = {k: np.zeros(len(g), np.float32) for k in models}
        lo, hi = 1e-6, 1.
        for _ in range(14):
            mid = (lo * hi) ** .5
            cnt = _label_count(g, preds, mid)
            lo, hi = (mid, hi) if cnt < n else (lo, mid)
        rates[d] = (lo * hi) ** .5
    return rates


def _label_count(g, preds, qrate, seed=0):
    """Trade count of a random-entry day using label busy times (no MBO)."""
    legal = []; busys = []
    for name, trig, hold, take, stop in sp.ADAPTIVE:
        spread_ok, cost_ok = sp.gates(g, hold, take)
        for mode in sc.MODES:
            for s in (1, -1):
                key = f'{mode}_{sc.SIDES[s]}_{name}'
                legal.append(spread_ok & cost_ok & np.isin(g[key + '_st'].to_numpy(), (sc.COMPLETE, sc.EMPTY, sc.CENSORED)))
                busys.append(np.nan_to_num(g[key + '_busy_s'].to_numpy(), nan=hold + 35))
    legal = np.stack(legal, 1); busys = np.stack(busys, 1); rng = np.random.default_rng(seed)
    score = np.where(legal, rng.random(legal.shape), -np.inf); best = score.argmax(1)
    t = g.t_ns.to_numpy(); busy = -1; n = 0
    for i in range(len(g)):
        if t[i] < busy or not legal[i].any() or rng.random() >= qrate:
            continue
        n += 1; busy = t[i] + int(busys[i, best[i]] * 1e9)
    return n


def experiment(data=sc.DEFAULT_OUTPUT, output=REPORT, workers=5):
    started = time.perf_counter(); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    f = sp.load(data)
    models = {'A1': fit_models(f, A1), 'A': fit_models(f, A1 + A2)}
    model_path = out / 'models.pkl'
    with open(model_path, 'wb') as fh:
        pickle.dump(models, fh)
    a2_dates = sorted(f.date[f.date.str[:7].isin(A2)].unique())
    # ---- 1. threshold selection on A2 with A1 models (passive TP, the deployed exit) ----
    sel_cfg = [dict(name=f'A2_th{th}', models='A1', threshold=th, passive_tp=True) for th in THRESHOLDS]
    tr_sel, _ = run_configs(sel_cfg, a2_dates, model_path, data, workers)
    sel = summarize(tr_sel, a2_dates)
    chosen = max(THRESHOLDS, key=lambda th: (sel.get(f'A2_th{th}', {}).get('net', -1e9), th))
    # ---- 2. B with all-A models and the chosen threshold, plus controls ----
    b_dates = sorted(f.date[f.date >= '2026-01'].unique())
    main = [dict(name='policy_passive_tp', models='A', threshold=chosen, passive_tp=True),
            dict(name='policy_taker_tp', models='A', threshold=chosen, passive_tp=False)]
    tr_b, _ = run_configs(main, b_dates, model_path, data, workers)
    per_day = tr_b[tr_b.config == 'policy_passive_tp'].groupby('date').size().to_dict() if len(tr_b) else {}
    rq = random_rates(f, b_dates, per_day, models['A'], chosen)
    tr_r, _ = run_configs([dict(name='random_passive_tp', models='A', threshold=0., passive_tp=True, random_q=rq)],
                          b_dates, model_path, data, workers)
    tr_all = pd.concat([tr_b, tr_r], ignore_index=True)
    tr_all.to_parquet(out / 'trades_B.parquet', index=False); tr_sel.to_parquet(out / 'trades_A2.parquet', index=False)
    s = dict(version='SCALPER_LIVE_V1', thresholds=list(THRESHOLDS), selection_A2=sel, chosen_threshold=chosen,
             B=summarize(tr_all, b_dates), split=dict(A1=list(A1), A2=list(A2), B=['2026-01', '2026-02']),
             seconds=time.perf_counter() - started, semantics=__doc__.strip())
    save_json(out / 'live_summary.json', s)
    print(json.dumps(dict(chosen=chosen, B={k: (v['trades'], round(v['net'], 1), v['per_trade']) for k, v in s['B'].items()})), flush=True)
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=REPORT); p.add_argument('--workers', type=int, default=5)
    a = p.parse_args(); experiment(output=a.output, workers=a.workers)
