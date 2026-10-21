"""Scalper mind, stage 1: decomposed consequences, nulls, decisions, adaptation (SCALPER_MIND_V1).

Attempt = decision (5 s grid of path_v1) x entry mode (T taker, P queue) x side x adaptive plan.
Outcome = (filled?, exit bin, category take/stop/time, net $). Three predictors of the outcome:
  brownian  finite-horizon two-barrier Brownian mid with sigma600 (path_null), no learning;
  null      gradient boosting on volatility, spread, time of day, plan geometry and the Brownian
            hazards (strong empirical null: no returns, flow, book or range features);
  full      the same model class with the scalper's causal features, directional ones signed by side.
Discrete-time factorisation: alpha_k = P(event in bin k | none before), beta_k = P(take | event in
bin k), fill = P(queue entry fills | placed) for P. The log-likelihood of one outcome splits exactly
into "when" (alpha), "what" (beta) and "chosen by the counterparty" (fill). Value head V_c =
E[net | category c]. E[G] = fill * sum_c P_c V_c (fill = 1 for T).
Scalar baseline: one regression of net per mode on the full features and geometry, same attempts.
Decisions: one 1 oz account on the 5 s grid; while flat take the legal attempt (scalper spread and
cost gates) with the highest E[G] if above theta, else WAIT; busy until the realised exit.
Adaptation: Hedge over experts {WAIT, scalar@theta, decomposed@theta} on daily results, A2 then B.
Fit on A1 for A2, refit on A (A1+A2) for B. Validation/evaluation months are not read.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from spectrum.ingest import save_json
from .signature_report import day_boot
from .scalper import SIDES, COMPLETE, CENSORED, EMPTY, GATE_SPREAD_TICKS, COST_MULT, FEE, DEFAULT_OUTPUT as SCALPER_DATA
from .scalper_policy import side_features, SYMMETRIC, DIRECTIONAL
from .path_scan import ADAPTIVE, EDGES, TAKE, STOP, TIME, DEFAULT_OUTPUT as PATH_DATA
from . import path_null as pn

VERSION = 'SCALPER_MIND_V1'
REPORT = 'reports/scalper_mind_v1'
OUTPUT = 'data/derived/scalper_mind_v1'
SPLIT = dict(A1=('2025-10', '2025-11'), A2=('2025-12',), B=('2026-01', '2026-02'))
TRAIN_EVERY = 6            # every 6th 5 s decision (30 s grid) enters training
SEED = 61091
THETAS = (0., .05, .1, .2, .3, .5)
MODES = ('T', 'P')
HEDGE_RANGE = 60.          # daily $ clipped to [-R, R] for the Hedge losses
TOP = .01
NULL_BASE = ['spread_ticks', 'sigma600', 'sigma60_ratio', 'sigma3600_ratio', 'minutes_since_london']
GEOM = ['hold', 'take_m', 'stop_m', 'g_ticks', 'r_ticks', 'cover', 'bn_take', 'bn_stop']
BINF = ['bin', 'bin_lo', 'bin_hi', 'bn_alpha', 'bn_beta', 'bn_surv']
SCALPER_COLUMNS = sorted((set(SYMMETRIC) | set(DIRECTIONAL) | {f'flow_pressure{w}_{e}' for w in (5, 30, 60, 300) for e in ('low', 'high')}
                          | {f'range{w}_pos' for w in (60, 300)}) - {'spread_ticks', 'sigma600'})


def dates_of(months, data=PATH_DATA):
    """Trading days of the months (days without a single decision, e.g. holidays, are not sessions)."""
    days = sorted(p.stem for p in Path(data).glob('2*.json') if p.stem[:7] in months)
    return [d for d in days if json.loads((Path(data) / f'{d}.json').read_text(encoding='utf8'))['decisions'] > 0]


def load_day(date, path_data=PATH_DATA, scalper_data=SCALPER_DATA):
    p = pd.read_parquet(Path(path_data) / f'{date}.parquet')
    f = pd.read_parquet(Path(scalper_data) / f'{date}.parquet', columns=['t_ns'] + SCALPER_COLUMNS)
    g = p.merge(f, on='t_ns', how='left', validate='one_to_one')
    assert g.minutes_since_london.notna().all(), date
    return g


def model(kind):
    cls = HistGradientBoostingClassifier if kind == 'clf' else HistGradientBoostingRegressor
    return cls(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=500, l2_regularization=1., random_state=SEED)


def attempts(g, mode):
    """Twelve attempts per decision (side x plan) for one mode: full feature matrix, outcome, Brownian hazards."""
    sig = g.sigma600.to_numpy(np.float64); spread = g.spread_ticks.to_numpy(np.float64)
    Xs = {s: side_features(g, s) for s in (1, -1)}
    names = list(Xs[1].columns) + GEOM; parts = []
    for pi, (name, trig, hold, take, stop) in enumerate(ADAPTIVE):
        u = sig * np.sqrt(hold); edges = [e for e in EDGES if e <= hold]
        alpha, beta, (ft, fs, _) = pn.plan_bins(take * u, stop * u, sig, edges)
        gt, rt = pn.lattice(take * u), pn.lattice(stop * u)
        geom = np.stack([np.full(len(g), hold, float), np.full(len(g), take), np.full(len(g), stop), gt, rt,
                         gt / (spread + FEE * 10), ft, fs], 1)
        gate = (spread <= GATE_SPREAD_TICKS) & (take * u >= COST_MULT * (spread + FEE * 10))
        for s in (1, -1):
            key = f'{mode}_{SIDES[s]}_{name}'; st = g[key + '_st'].to_numpy()
            meta = dict(row=np.arange(len(g)), plan=pi, side=s, hold=hold, st=st, net=g[key + '_net'].to_numpy(np.float64),
                        cat=g[key + '_cat'].to_numpy(), bin=g[key + '_bin'].to_numpy(), busy=g[key + '_busy_s'].to_numpy(np.float64),
                        legal=gate & np.isin(st, (COMPLETE, EMPTY, CENSORED)))
            X = np.concatenate([Xs[s].to_numpy(np.float32), geom.astype(np.float32)], 1)
            parts.append(dict(X=X, meta=meta, alpha=alpha, beta=beta, edges=edges))
    return names, parts


def column_index(names, variant):
    keep = names if variant == 'full' else NULL_BASE + GEOM
    return np.array([names.index(c) for c in keep])


def person_period(X, alpha, beta, edges):
    """Every attempt repeated over its bins with bin and Brownian hazard columns."""
    n, K = alpha.shape
    lo = np.r_[0., edges[:-1]]; hi = np.asarray(edges, float)
    surv = np.cumprod(np.c_[np.ones(n), 1 - alpha[:, :-1]], 1)
    extra = np.stack([np.tile(np.arange(K, dtype=float), n), np.tile(lo, n), np.tile(hi, n), alpha.ravel(), beta.ravel(), surv.ravel()], 1)
    return np.concatenate([np.repeat(X, K, 0), extra.astype(np.float32)], 1)


def targets(meta, K):
    """At-risk mask, event indicator and take indicator per (attempt, bin) for completed attempts."""
    done = meta['st'] == COMPLETE; b = np.where(done, meta['bin'], -1).astype(int); k = np.arange(K)[None, :]
    event = np.isin(meta['cat'], (TAKE, STOP))
    at_risk = done[:, None] & (k <= b[:, None])
    y_event = at_risk & event[:, None] & (k == b[:, None])
    y_take = (meta['cat'] == TAKE)[:, None] & y_event
    return at_risk, y_event, y_take


def training_arrays(dates, mode, every=TRAIN_EVERY):
    """Pooled training matrices over days, sides and plans (full columns; the null uses a column subset)."""
    A = dict(att=[], net=[], filled=[], fillrow=[], pp=[], y_ev=[], ev=[], y_take=[], val=[], y_val=[])
    names = None
    for d in dates:
        g = load_day(d); g = g.iloc[::every].reset_index(drop=True)
        names, parts = attempts(g, mode)
        for p in parts:
            X, m = p['X'], p['meta']; K = p['alpha'].shape[1]
            real = np.isin(m['st'], (COMPLETE, EMPTY))
            A['att'].append(X[real]); A['net'].append(m['net'][real])
            A['filled'].append((m['st'] == COMPLETE)[real])
            at_risk, y_ev, y_take = targets(m, K)
            pp = person_period(X, p['alpha'], p['beta'], p['edges'])
            A['pp'].append(pp[at_risk.ravel()]); A['y_ev'].append(y_ev.ravel()[at_risk.ravel()])
            A['ev'].append(pp[y_ev.ravel()]); A['y_take'].append(y_take.ravel()[y_ev.ravel()])
            done = m['st'] == COMPLETE
            A['val'].append(np.c_[X[done], m['cat'][done].astype(np.float32)]); A['y_val'].append(m['net'][done])
    return names, {k: np.concatenate(v) for k, v in A.items() if v}


def fit(dates, log=print):
    """All models for both modes and both learned variants, plus the scalar baseline."""
    models = {}
    for mode in MODES:
        names, A = training_arrays(dates, mode)
        F = len(names); started = time.perf_counter()
        for variant in ('null', 'full'):
            ci = column_index(names, variant); cpp = np.r_[ci, F + np.arange(len(BINF))]; cv = np.r_[ci, F]
            m = dict(cols=ci)
            m['alpha'] = model('clf').fit(A['pp'][:, cpp], A['y_ev'])
            m['beta'] = model('clf').fit(A['ev'][:, cpp], A['y_take'])
            m['value'] = model('reg').fit(A['val'][:, cv], A['y_val'])
            if mode == 'P':
                m['fill'] = model('clf').fit(A['att'][:, ci], A['filled'])
            models[(mode, variant)] = m
        models[(mode, 'scalar')] = dict(cols=np.arange(F), reg=model('reg').fit(A['att'], A['net']))
        models[(mode, 'names')] = names
        log(json.dumps(dict(mode=mode, attempts=int(len(A['att'])), pp_rows=int(len(A['pp'])), events=int(len(A['ev'])),
                            seconds=round(time.perf_counter() - started, 1))))
    return models


def clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def components(alpha, beta):
    n, K = alpha.shape
    surv = np.cumprod(np.c_[np.ones(n), 1 - alpha], 1)
    p_take = np.sum(surv[:, :-1] * alpha * beta, 1); p_stop = np.sum(surv[:, :-1] * alpha * (1 - beta), 1)
    return p_take, p_stop, surv[:, -1]


def losses(alpha, beta, meta):
    """Per completed attempt: 'when' and 'what' negative log-likelihood of the realised (bin, category)."""
    n, K = alpha.shape; at_risk, y_ev, y_take = targets(meta, K)
    a, b = clip(alpha), clip(beta)
    when = -np.sum(np.where(at_risk, np.where(y_ev, np.log(a), np.log(1 - a)), 0.), 1)
    ev = y_ev.any(1); what = -np.sum(np.where(y_ev, np.where(y_take, np.log(b), np.log(1 - b)), 0.), 1)
    return when, np.where(ev, what, np.nan)


def brier(p_take, p_stop, p_time, cat):
    return (p_take - (cat == TAKE)) ** 2 + (p_stop - (cat == STOP)) ** 2 + (p_time - (cat == TIME)) ** 2


def evaluate_day(date, models, sample_rng):
    """Predictions for every attempt of one day: knowledge aggregates, legal attempts with E[G], calibration sample."""
    g = load_day(date); agg, legal, sample = [], [], []
    for mode in MODES:
        names, parts = attempts(g, mode); F = len(names)
        for p in parts:
            X, m = p['X'], p['meta']; n, K = p['alpha'].shape
            pp = person_period(X, p['alpha'], p['beta'], p['edges'])
            out = dict(eg_scalar=models[(mode, 'scalar')]['reg'].predict(X))
            done = m['st'] == COMPLETE; real = np.isin(m['st'], (COMPLETE, EMPTY))
            row = dict(date=date, mode=mode, plan=ADAPTIVE[m['plan']][0], side=m['side'])
            for variant in ('brownian', 'null', 'full'):
                if variant == 'brownian':
                    alpha, beta = p['alpha'], p['beta']
                else:
                    mm = models[(mode, variant)]; cpp = np.r_[mm['cols'], F + np.arange(len(BINF))]
                    alpha = mm['alpha'].predict_proba(pp[:, cpp])[:, 1].reshape(n, K)
                    beta = mm['beta'].predict_proba(pp[:, cpp])[:, 1].reshape(n, K)
                pt, ps_, pz = components(alpha, beta); when, what = losses(alpha, beta, m)
                br = brier(pt, ps_, pz, m['cat'])
                for legal_only in (False, True):
                    sel = done & (m['legal'] if legal_only else True)
                    agg.append(dict(row, variant=variant, legal=legal_only, attempts=int(sel.sum()),
                                    events=int(np.isfinite(what[sel]).sum()), when=float(when[sel].sum()),
                                    what=float(np.nansum(what[sel])), brier=float(br[sel].sum())))
                out[f'p_take_{variant}'], out[f'p_stop_{variant}'], out[f'p_time_{variant}'] = pt, ps_, pz
                out[f'when_{variant}'], out[f'what_{variant}'] = when, what
                if variant != 'brownian':
                    mm = models[(mode, variant)]; cv = np.r_[mm['cols'], F]
                    V = {c: mm['value'].predict(np.c_[X, np.full(n, c, np.float32)][:, cv]) for c in (TAKE, STOP, TIME)}
                    ef = pt * V[TAKE] + ps_ * V[STOP] + pz * V[TIME]
                    pf = mm['fill'].predict_proba(X[:, mm['cols']])[:, 1] if mode == 'P' else np.ones(n)
                    out[f'eg_{variant}'] = pf * ef; out[f'fill_{variant}'] = pf; out[f'ef_{variant}'] = ef
                    for c, nm in ((TAKE, 'take'), (STOP, 'stop'), (TIME, 'time')):
                        out[f'v_{nm}_{variant}'] = V[c]
                    if mode == 'P':
                        y = m['st'] == COMPLETE; q = clip(pf)
                        fl = -(y * np.log(q) + (1 - y) * np.log(1 - q))
                        for legal_only in (False, True):
                            sel = real & (m['legal'] if legal_only else True)
                            agg.append(dict(row, variant=variant, legal=legal_only, kind='fill', attempts=int(sel.sum()),
                                            fill=float(fl[sel].sum()), filled=int(y[sel].sum())))
            base = dict(date=date, t_ns=g.t_ns.to_numpy()[m['row']], mode=mode, plan=m['plan'], side=m['side'], hold=m['hold'],
                        st=m['st'], net=m['net'], cat=m['cat'], bin=m['bin'], busy=m['busy'])
            full = pd.DataFrame({**base, **out})
            legal.append(full[m['legal']])
            pick = sample_rng.random(n) < .03
            sample.append(full[pick & done])
    return pd.DataFrame(agg), pd.concat(legal, ignore_index=True), pd.concat(sample, ignore_index=True)


def evaluate(dates, models, log=print):
    rng = np.random.default_rng(SEED); A, L, S = [], [], []
    for d in dates:
        t = time.perf_counter(); a, l, s = evaluate_day(d, models, rng); A.append(a); L.append(l); S.append(s)
        log(json.dumps(dict(date=d, legal=len(l), seconds=round(time.perf_counter() - t, 1))))
    return pd.concat(A, ignore_index=True), pd.concat(L, ignore_index=True), pd.concat(S, ignore_index=True)


# ---------- knowledge verdicts ----------

def day_t(diff):
    d = np.asarray(diff, float); d = d[np.isfinite(d)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return dict(days=int(len(d)), mean=float(d.mean()) if len(d) else None, t=None)
    return dict(days=int(len(d)), mean=float(d.mean()), t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))


def knowledge(agg):
    """Per-day mean losses per variant and day-clustered paired improvements (positive = better)."""
    out = {}
    path = agg[agg.get('kind', pd.Series(index=agg.index, dtype=object)).isna()] if 'kind' in agg else agg
    for legal_only in (False, True):
        for mode in MODES:
            a = path[(path['mode'] == mode) & (path.legal == legal_only)]
            day = a.groupby(['date', 'variant'])[['attempts', 'events', 'when', 'what', 'brier']].sum()
            per = pd.DataFrame(dict(when=day.when / day.attempts, what=day.what / day.events, brier=day.brier / day.attempts)).unstack('variant')
            res = dict(attempts=int(a[a.variant == 'full'].attempts.sum()), events=int(a[a.variant == 'full'].events.sum()))
            for comp in ('when', 'what', 'brier'):
                res[comp] = {v: float(per[comp][v].mean()) for v in ('brownian', 'null', 'full')}
                res[comp + '_null_vs_brownian'] = day_t(per[comp]['brownian'] - per[comp]['null'])
                res[comp + '_full_vs_null'] = day_t(per[comp]['null'] - per[comp]['full'])
            if mode == 'P' and 'kind' in agg:
                f = agg[(agg.kind == 'fill') & (agg['mode'] == 'P') & (agg.legal == legal_only)]
                fd = f.groupby(['date', 'variant'])[['attempts', 'fill', 'filled']].sum()
                fl = (fd.fill / fd.attempts).unstack('variant')
                rate = (fd.filled / fd.attempts).unstack('variant')['full']
                res['fill'] = {v: float(fl[v].mean()) for v in ('null', 'full')}
                res['fill_rate'] = float(rate.mean())
                res['fill_full_vs_null'] = day_t(fl['null'] - fl['full'])
            out[f"{mode}_{'legal' if legal_only else 'all'}"] = res
    return out


def calibration(sample):
    """Predicted vs realised take/stop/time shares by plan and mode, and by decile of predicted take share."""
    rows = []
    for (mode, plan), g in sample.groupby(['mode', 'plan']):
        r = dict(mode=mode, plan=ADAPTIVE[plan][0], n=len(g), take=float((g.cat == TAKE).mean()), stop=float((g.cat == STOP).mean()),
                 time=float((g.cat == TIME).mean()))
        for v in ('brownian', 'null', 'full'):
            r.update({f'{k}_{v}': float(g[f'p_{k}_{v}'].mean()) for k in ('take', 'stop', 'time')})
        rows.append(r)
    dec = []
    for v in ('brownian', 'null', 'full'):
        g = sample.assign(q=pd.qcut(sample[f'p_take_{v}'].rank(method='first'), 10, labels=False))
        for q, h in g.groupby('q'):
            dec.append(dict(variant=v, decile=int(q), predicted=float(h[f'p_take_{v}'].mean()), realised=float((h.cat == TAKE).mean())))
    return rows, dec


# ---------- decisions ----------

def best_per_decision(leg, score):
    """Index of the highest-scoring legal attempt per decision (date, t_ns)."""
    o = leg.sort_values(['date', 't_ns', score], ascending=[True, True, False], kind='mergesort')
    return o.drop_duplicates(['date', 't_ns'], keep='first')


def account(best, score, theta=0., rate=None, rng=None):
    """One 1 oz account over the best attempt per decision; rate: take each free decision with this probability."""
    t = best.t_ns.to_numpy(); dates = best.date.to_numpy(); sc = best[score].to_numpy(); busy = best.busy.to_numpy()
    hold = best.hold.to_numpy(); take = rng.random(len(best)) if rate is not None else None
    chosen = []; free = -1; day = None
    for i in range(len(best)):
        if dates[i] != day:
            day = dates[i]; free = -1
        if t[i] < free:
            continue
        if rate is None and not sc[i] > theta:
            continue
        if rate is not None and take[i] >= rate:
            continue
        chosen.append(i); b = busy[i] if np.isfinite(busy[i]) else hold[i] + 35
        free = t[i] + int(b * 1e9)
    tr = best.iloc[chosen]
    return tr


def summary(tr, days):
    daily = pd.DataFrame(dict(date=days)).merge(tr.groupby('date', as_index=False).net.sum(), on='date', how='left')
    daily['net'] = pd.to_numeric(daily.net, errors='coerce').fillna(0.)
    filled = tr[tr.st == COMPLETE]
    eq = np.r_[0, tr.net.fillna(0).cumsum().to_numpy()]
    return dict(trades=int(len(tr)), filled=int(len(filled)), net=float(tr.net.sum()),
                per_trade=float(filled.net.mean()) if len(filled) else None,
                win_rate=float((filled.net > 0).mean()) if len(filled) else None,
                max_drawdown=float((np.maximum.accumulate(eq) - eq).max()), daily=day_boot(daily, 'net') if len(tr) else None,
                per_day=len(tr) / max(len(days), 1)), daily.set_index('date').net


def matched_random(leg, days, target):
    """Random legal attempt per decision, taken with the rate that matches the policy's trade count."""
    if not target:
        return dict(trades=0), pd.Series(0., index=days)
    best = best_per_decision(leg.assign(_r=np.random.default_rng(SEED + 1).random(len(leg))), '_r')
    lo, hi = 1e-6, 1.
    for _ in range(16):
        mid = (lo * hi) ** .5; n = len(account(best, '_r', rate=mid, rng=np.random.default_rng(SEED)))
        lo, hi = (mid, hi) if n < target else (lo, mid)
    return summary(account(best, '_r', rate=(lo * hi) ** .5, rng=np.random.default_rng(SEED)), days)


def selection(leg):
    """Ranking quality per day over legal attempts with a realised result: Spearman and top-1% mean net."""
    real = leg[np.isin(leg.st, (COMPLETE, EMPTY))]; rows = []
    for d, g in real.groupby('date'):
        r = dict(date=d, n=len(g))
        for v in ('scalar', 'null', 'full'):
            s = g[f'eg_{v}'].to_numpy(); q = np.quantile(s, 1 - TOP)
            r[f'top_{v}'] = float(g.net[s >= q].mean()); r[f'rho_{v}'] = float(spearmanr(s, g.net).statistic)
        rows.append(r)
    f = pd.DataFrame(rows)
    return dict(days=len(f), **{f'top_{v}': float(f[f'top_{v}'].mean()) for v in ('scalar', 'null', 'full')},
                **{f'rho_{v}': float(f[f'rho_{v}'].mean()) for v in ('scalar', 'null', 'full')},
                top_full_vs_scalar=day_t(f.top_full - f.top_scalar), rho_full_vs_scalar=day_t(f.rho_full - f.rho_scalar),
                top_full_vs_null=day_t(f.top_full - f.top_null)), f


def experts(leg, days):
    """Daily results of every expert: WAIT, scalar@theta, null@theta, full@theta."""
    res, daily, trades = {}, {'WAIT': pd.Series(0., index=days)}, {}
    res['WAIT'] = dict(trades=0, net=0.)
    for v in ('scalar', 'null', 'full'):
        best = best_per_decision(leg, f'eg_{v}')
        for th in THETAS:
            tr = account(best, f'eg_{v}', th); s, dl = summary(tr, days)
            res[f'{v}@{th}'] = s; daily[f'{v}@{th}'] = dl; trades[f'{v}@{th}'] = tr
    return res, pd.DataFrame(daily), trades


def hedge(daily, R=HEDGE_RANGE):
    """Exponential weights on daily gains clipped to [-R, R]; returns mixed daily gain, weights, regret and bound."""
    g = daily.clip(-R, R).to_numpy(); T, N = g.shape
    eta = np.sqrt(8 * np.log(N) / T); loss = (R - g) / (2 * R)
    cum = np.zeros(N); mixed, W = [], []
    for t in range(T):
        w = np.exp(-eta * (cum - cum.min())); w /= w.sum(); W.append(w)
        mixed.append(float(w @ daily.to_numpy()[t])); cum += loss[t]
    mixed = pd.Series(mixed, index=daily.index)
    best = daily.sum().idxmax()
    return dict(experts=list(daily.columns), days=T, eta=float(eta), net=float(mixed.sum()), best_expert=best,
                best_net=float(daily[best].sum()), regret=float(daily[best].sum() - mixed.sum()),
                bound=float(2 * R * np.sqrt(T * np.log(N) / 2)), final_weights=dict(zip(daily.columns, map(float, W[-1]))),
                weights=pd.DataFrame(W, index=daily.index, columns=daily.columns)), mixed


def ledger(tr):
    """Regret ledger on executed trades: forecast, outcome, exact error decomposition (fill, outcome, value)."""
    t = tr.copy(); f = np.where(t.st == COMPLETE, 1., 0.); v = 'full'
    P = {c: t[f'p_{c}_{v}'].to_numpy() for c in ('take', 'stop', 'time')}; V = {c: t[f'v_{c}_{v}'].to_numpy() for c in P}
    ind = {c: (t.cat.to_numpy() == k).astype(float) for c, k in (('take', TAKE), ('stop', STOP), ('time', TIME))}
    ef = t[f'ef_{v}'].to_numpy(); pf = t[f'fill_{v}'].to_numpy(); net = np.where(f > 0, t.net.to_numpy(), 0.)
    v_real = sum(ind[c] * V[c] for c in P)
    t['err_fill'] = (f - pf) * ef
    t['err_outcome'] = f * sum((ind[c] - P[c]) * V[c] for c in P)
    t['err_value'] = f * (net - v_real)
    t['err_total'] = net - t[f'eg_{v}'].to_numpy()
    t['identity_gap'] = t.err_total - t.err_fill - t.err_outcome - t.err_value
    t['net0'] = net
    return t


def ledger_summary(t):
    comp = ['err_fill', 'err_outcome', 'err_value']
    ss = {c: float((t[c] ** 2).sum()) for c in comp}; tot = sum(ss.values())
    dom = t[comp].abs().idxmax(axis=1).value_counts(normalize=True).to_dict()
    filled = t[t.st == COMPLETE]
    return dict(trades=len(t), identity_max_gap=float(t.identity_gap.abs().max()), forecast=float(t.eg_full.mean()),
                realised=float(t.net0.mean()), bias={c: float(t[c].mean()) for c in comp},
                square_share={c: ss[c] / tot if tot else None for c in comp}, dominant=dom,
                when_nll=float(filled.when_full.mean()) if len(filled) else None,
                what_nll=float(filled.what_full.mean()) if len(filled) else None,
                by_mode={m: dict(trades=int(len(g)), forecast=float(g.eg_full.mean()), realised=float(g.net0.mean()))
                         for m, g in t.groupby('mode')},
                by_outcome=[dict(mode=m, cat=int(c), trades=int(len(g)), net=float(g.net0.mean()),
                                 v_pred=float(np.select([g.cat == TAKE, g.cat == STOP, g.cat == TIME],
                                                        [g.v_take_full, g.v_stop_full, g.v_time_full], np.nan).mean()) if c else None,
                                 err_value=float(g.err_value.mean()))
                            for (m, c), g in t.groupby(['mode', 'cat'])],
                plans={ADAPTIVE[k][0]: int(v) for k, v in t.plan.value_counts().items()})


def refresh_ledger(output=REPORT, data_out=OUTPUT):
    """Recompute the ledger summary from the saved ledger (readout change only; trades unchanged)."""
    path = Path(output) / 'stage1_summary.json'; s = json.loads(path.read_text(encoding='utf8'))
    led = pd.read_parquet(Path(data_out) / 'ledger_B.parquet')
    s['ledger_B'] = dict(policy=s['ledger_B']['policy'], **ledger_summary(led)); save_json(path, s)
    return s['ledger_B']


def adverse_selection(leg):
    """Queue entries: realised value after a fill against the model's view, by predicted fill probability."""
    p = leg[(leg['mode'] == 'P') & np.isin(leg.st, (COMPLETE, EMPTY))].copy()
    p['filled'] = p.st == COMPLETE
    p['q'] = pd.qcut(p.fill_full.rank(method='first'), 5, labels=False)
    rows = []
    for q, g in p.groupby('q'):
        f = g[g.filled]
        rows.append(dict(quintile=int(q), p_fill=float(g.fill_full.mean()), fill_rate=float(g.filled.mean()),
                         ef_pred=float(f.ef_full.mean()), net_after_fill=float(f.net.mean()),
                         stop_share=float((f.cat == STOP).mean()), p_stop_pred=float(f.p_stop_full.mean())))
    t = leg[(leg['mode'] == 'T') & (leg.st == COMPLETE)][['date', 't_ns', 'plan', 'side', 'net']]
    m = p.merge(t, on=['date', 't_ns', 'plan', 'side'], suffixes=('', '_taker'))
    return dict(quintiles=rows, taker_where_queue_filled=float(m[m.filled].net_taker.mean()),
                taker_where_queue_empty=float(m[~m.filled].net_taker.mean()), pairs=int(len(m)))


def experiment(output=REPORT, data_out=OUTPUT, log=print):
    out = Path(output); out.mkdir(parents=True, exist_ok=True); dout = Path(data_out); dout.mkdir(parents=True, exist_ok=True)
    A1, A2, B = (dates_of(SPLIT[k]) for k in ('A1', 'A2', 'B'))
    s = dict(version=VERSION, split=dict(A1=A1, A2=A2, B=B), train_every=TRAIN_EVERY, thetas=list(THETAS), seed=SEED)
    res = {}
    for name, train, test in (('A2', A1, A2), ('B', A1 + A2, B)):
        t0 = time.perf_counter(); models = fit(train, log)
        agg, leg, smp = evaluate(test, models, log)
        leg.to_parquet(dout / f'legal_{name}.parquet', index=False); smp.to_parquet(dout / f'sample_{name}.parquet', index=False)
        agg.to_parquet(dout / f'knowledge_{name}.parquet', index=False)
        res[name] = dict(leg=leg, agg=agg, smp=smp, days=test)
        log(json.dumps(dict(split=name, seconds=round(time.perf_counter() - t0, 1))))
    for name in ('A2', 'B'):
        r = res[name]; s[f'knowledge_{name}'] = knowledge(r['agg'])
        s[f'calibration_{name}'], s[f'deciles_{name}'] = calibration(r['smp'])
        s[f'selection_{name}'], sel = selection(r['leg']); sel.to_csv(out / f'selection_{name}.csv', index=False)
        s[f'experts_{name}'], r['daily'], r['trades'] = experts(r['leg'], r['days'])
        s[f'adverse_{name}'] = adverse_selection(r['leg'])
    sel_a2 = s['experts_A2']; choice = max(sel_a2, key=lambda k: sel_a2[k]['net'])
    s['agent'] = dict(choice=choice, a2_net=sel_a2[choice]['net'], b=s['experts_B'][choice],
                      b_full_theta0=s['experts_B']['full@0.0'], b_scalar_theta0=s['experts_B']['scalar@0.0'])
    tr_b = res['B']['trades']; s['random_B'], _ = matched_random(res['B']['leg'], res['B']['days'], s['experts_B']['full@0.0']['trades'])
    pair = res['B']['daily']['full@0.0'] - res['B']['daily']['scalar@0.0']; s['agent']['full_minus_scalar_theta0_B'] = day_t(pair)
    daily = pd.concat([res['A2']['daily'], res['B']['daily']])
    s['hedge'], mixed = hedge(daily); weights = s['hedge'].pop('weights'); weights.to_csv(out / 'hedge_weights.csv')
    s['hedge']['B_net'] = float(mixed.loc[res['B']['days']].sum()); s['hedge']['A2_net'] = float(mixed.loc[res['A2']['days']].sum())
    s['hedge']['B_best_expert'] = res['B']['daily'].sum().idxmax(); s['hedge']['B_best_net'] = float(res['B']['daily'].sum().max())
    led_key = choice if choice != 'WAIT' else 'full@0.0'
    if not led_key.startswith('full'):
        led_key = 'full@0.0'
    led = ledger(tr_b[led_key]); led.to_parquet(dout / 'ledger_B.parquet', index=False)
    s['ledger_B'] = dict(policy=led_key, **ledger_summary(led))
    s['daring_B'] = {k: s['experts_B'][k]['net'] for k in s['experts_B'] if k.startswith('full@')}
    s['daring_A2'] = {k: s['experts_A2'][k]['net'] for k in s['experts_A2'] if k.startswith('full@')}
    save_json(out / 'stage1_summary.json', s)
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=REPORT)
    a = p.parse_args(); experiment(output=a.output)
