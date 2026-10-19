"""Continuous scalper policy: gates, attempt-value models, one 1 oz account.

Fit on block A (Oct-Dec 2025), read block B (Jan-Feb 2026), both train months.
At every second while flat, the account takes the (entry mode, adaptive plan,
side) with the highest predicted attempt value if it passes both gates and the
prediction is above THRESHOLD; otherwise it waits. Busy until the realised exit
(or the end of an unfilled passive wait). Hyperparameters, gates and threshold
are fixed in code before the B readout.
"""
from pathlib import Path
import argparse
import glob
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from spectrum.ingest import save_json
from .signature_report import day_boot
from .scalper import DEFAULT_OUTPUT as DATA, PLANS, MODES, SIDES, GATE_SPREAD_TICKS, COST_MULT, COMPLETE, EMPTY, CENSORED, FEE

REPORT = 'reports/scalper_v1'
ADAPTIVE = [p for p in PLANS if p[1] == 'mid']
THRESHOLD = 0.
TRAIN_STRIDE = 10   # every 10th second of block A enters training (overlapping paths)
SEED = 61081
DIRECTIONAL = ['ret5', 'ret15', 'ret60', 'ret300'] + [f'flow_book{w}_pressure' for w in (5, 30, 60, 300)] + \
              [f'flow_depth{d}_imbalance' for d in (1, 4, 12)]
SYMMETRIC = ['spread_ticks', 'sigma600', 'sigma60_ratio', 'sigma3600_ratio', 'minutes_since_london', 'flow_volume_rate_ratio'] + \
            [f'flow_log_volume{w}' for w in (5, 30, 60, 300)] + [f'flow_unknown{w}_fraction' for w in (5, 30, 60, 300)] + \
            ['ret5_known', 'ret60_known', 'ret300_known', 'range60_known', 'range300_known']


def block(date):
    return 'A' if date < '2026-01' else 'B'


def load(data=DATA):
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(str(Path(data) / '2*.parquet')))]
    f = pd.concat(frames, ignore_index=True); f['block'] = np.where(f.date < '2026-01', 'A', 'B')
    return f


def side_features(f, side):
    s = float(side); x = {c: f[c].to_numpy(np.float32) for c in SYMMETRIC}
    for c in DIRECTIONAL:
        x[c] = s * f[c].to_numpy(np.float32)
    for w in (5, 30, 60, 300):
        lo, hi = f[f'flow_pressure{w}_low'].to_numpy(np.float32), f[f'flow_pressure{w}_high'].to_numpy(np.float32)
        x[f'pressure{w}_low'], x[f'pressure{w}_high'] = (lo, hi) if side > 0 else (-hi, -lo)
    for w in (60, 300):
        pos = f[f'range{w}_pos'].to_numpy(np.float32); x[f'range{w}'] = pos if side > 0 else 1 - pos
    return pd.DataFrame(x)


def gates(f, hold, take):
    spread = f.spread_ticks.to_numpy()
    unit = f.sigma600.to_numpy() * np.sqrt(hold)
    return (spread <= GATE_SPREAD_TICKS), (take * unit >= COST_MULT * (spread + FEE * 10))


def diagnostics(f):
    """Mean attempt value over all decision seconds, with and without each gate."""
    rows = []
    for name, trig, hold, take, stop in PLANS:
        spread_ok, cost_ok = gates(f, hold, take) if trig == 'mid' else (f.spread_ticks.to_numpy() <= GATE_SPREAD_TICKS,) * 2
        for mode in MODES:
            for b in ('A', 'B'):
                m = (f.block == b).to_numpy()
                v = np.r_[f[f'{mode}_L_{name}_net'].to_numpy()[m], f[f'{mode}_S_{name}_net'].to_numpy()[m]]
                for gname, g in (('none', np.ones(m.sum(), bool)), ('spread', spread_ok[m]), ('spread+cost', (spread_ok & cost_ok)[m])):
                    gg = np.r_[g, g]
                    rows.append(dict(plan=name, mode=mode, block=b, gate=gname, share=float(g.mean()),
                                     attempt=float(np.nanmean(v[gg])) if gg.any() else None, n=int(np.isfinite(v[gg]).sum())))
    return rows


def model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=500,
                                         l2_regularization=1., random_state=SEED)


def fit_predict(f, permute=False):
    """Predicted attempt value for every (mode, plan, side) column, fitted on block A only."""
    a = (f.block == 'A').to_numpy(); train = a & (np.arange(len(f)) % TRAIN_STRIDE == 0)
    rng = np.random.default_rng(SEED); preds = {}
    X = {s: side_features(f, s) for s in (1, -1)}
    for name, trig, hold, take, stop in ADAPTIVE:
        for mode in MODES:
            for s in (1, -1):
                key = f'{mode}_{SIDES[s]}_{name}'
                y = f[key + '_net'].to_numpy(np.float64); ok = train & np.isfinite(y)
                target = rng.permutation(y[ok]) if permute else y[ok]
                preds[key] = model().fit(X[s][ok], target).predict(X[s]).astype(np.float32)
    return preds


def simulate(f, preds, blk, threshold=THRESHOLD, random_rate=None, rng=None):
    """One 1 oz account with continuous decisions. random_rate: matched random entries instead of the model."""
    m = (f.block == blk).to_numpy(); g = f[m].reset_index(drop=True)
    keys, legal, score = [], [], []
    for name, trig, hold, take, stop in ADAPTIVE:
        spread_ok, cost_ok = gates(g, hold, take)
        for mode in MODES:
            for s in (1, -1):
                key = f'{mode}_{SIDES[s]}_{name}'; st = g[key + '_st'].to_numpy()
                keys.append(key); legal.append(spread_ok & cost_ok & np.isin(st, (COMPLETE, EMPTY, CENSORED)))
                score.append(preds[key][m] if preds is not None else np.zeros(m.sum(), np.float32))
    legal = np.stack(legal, 1); score = np.where(legal, np.stack(score, 1), -np.inf)
    if random_rate is not None:
        score = np.where(legal, rng.random(score.shape), -np.inf)
    best = score.argmax(1); best_score = score[np.arange(len(g)), best]
    t = g.t_ns.to_numpy(); dates = g.date.to_numpy(); trades = []; busy = -1; day = None
    nets = {k: g[k + '_net'].to_numpy() for k in keys}; busys = {k: g[k + '_busy_s'].to_numpy() for k in keys}
    sts = {k: g[k + '_st'].to_numpy() for k in keys}
    hold_of = {f'{mo}_{SIDES[s]}_{p[0]}': p[2] for p in ADAPTIVE for mo in MODES for s in (1, -1)}
    for i in range(len(g)):
        if dates[i] != day:
            day = dates[i]; busy = -1
        if t[i] < busy or not np.isfinite(best_score[i]):
            continue
        if random_rate is None and best_score[i] <= threshold:
            continue
        if random_rate is not None and rng.random() >= random_rate:
            continue
        k = keys[best[i]]; st = sts[k][i]
        busy_s = busys[k][i] if np.isfinite(busys[k][i]) else hold_of[k] + 35
        trades.append(dict(date=day, t_ns=int(t[i]), key=k, status=int(st), net=float(nets[k][i]) if st in (COMPLETE, EMPTY) else np.nan,
                           pred=float(best_score[i])))
        busy = t[i] + int(busy_s * 1e9)
    tr = pd.DataFrame(trades)
    days = sorted(set(dates))
    if not len(tr):
        return dict(trades=0, filled=0, net=0., per_trade=None, daily=None, days=len(days)), tr
    filled = tr[tr.status == COMPLETE]
    daily = pd.DataFrame(dict(date=days)).merge(tr.groupby('date', as_index=False).net.sum(), on='date', how='left')
    daily['net'] = pd.to_numeric(daily.net, errors='coerce').fillna(0.)
    eq = tr.net.fillna(0).cumsum().to_numpy(); dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    return dict(trades=len(tr), filled=len(filled), net=float(tr.net.sum()), per_trade=float(filled.net.mean()) if len(filled) else None,
                per_day_trades=len(tr) / len(days), win_rate=float((filled.net > 0).mean()) if len(filled) else None,
                max_drawdown=dd, daily=day_boot(daily, 'net'), days=len(days),
                keys=tr.key.value_counts().head(6).to_dict()), tr


def matched_random(f, blk, target):
    """Random gated entries (random mode/plan/side) with the policy's trade count."""
    if not target:
        return dict(trades=0, note='policy made no trades; nothing to match')
    lo, hi = 1e-7, 1.
    for _ in range(18):
        mid = (lo * hi) ** .5
        r, _ = simulate(f, None, blk, random_rate=mid, rng=np.random.default_rng(SEED))
        lo, hi = (mid, hi) if r['trades'] < target else (lo, mid)
    r, _ = simulate(f, None, blk, random_rate=(lo * hi) ** .5, rng=np.random.default_rng(SEED))
    return r


def experiment(data=DATA, output=REPORT):
    f = load(data); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    s = dict(version='SCALPER_POLICY_V1', rows=len(f), days=int(f.date.nunique()), diagnostics=diagnostics(f),
             threshold=THRESHOLD, gate_spread_ticks=GATE_SPREAD_TICKS, cost_mult=COST_MULT)
    preds = fit_predict(f)
    for blk in ('A', 'B'):
        s[f'policy_{blk}'], tr = simulate(f, preds, blk)
        tr.to_parquet(out / f'trades_{blk}.parquet', index=False)
        for th in (.05, .10, .20):
            s[f'policy_{blk}_threshold_{th}'], _ = simulate(f, preds, blk, threshold=th)
    s['random_B'] = matched_random(f, 'B', s['policy_B']['trades'])
    s['permuted_B'], _ = simulate(f, fit_predict(f, permute=True), 'B')
    save_json(out / 'scalper_summary.json', s)
    for k in ('policy_A', 'policy_B', 'random_B', 'permuted_B'):
        v = s[k]; print(k, v['trades'], round(v['net'], 1), v['per_trade'] and round(v['per_trade'], 3), v.get('daily'), flush=True)
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=REPORT)
    a = p.parse_args(); experiment(output=a.output)
