"""Action-value model over representation families (the 'slow' learner).

For each origin and plan the paired sibling outcomes give
    m = (net_long + net_short) / 2   cost and volatility component
    d = (net_long - net_short) / 2   direction component
so Q(long) = m + d, Q(short) = m - d and Q(WAIT) = 0. Fitting on block A and
reading block B is an internal chronological check inside train months.
Hyperparameters are fixed in code before any B readout. Origins are independent
fresh-account experiments here; the sequential single-account test is schemas.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .representation import DEFAULT_OUTPUT as REPRESENTATION, family_columns
from .signature_report import load, day_boot
from .signatures import DEFAULT_OUTPUT as SIGNATURES, MENU_PLANS

SETS = dict(vol=('vol',), price=('vol', 'price'), flow=('vol', 'flow'), micro=('vol', 'micro'),
            event=('vol', 'event'), price_flow=('vol', 'price', 'flow'), all=('vol', 'price', 'flow', 'micro', 'event'))
PLANS = (1, 2, 3)  # 12/8/120 is excluded: a perfect side call still loses on average
SEED = 61041


def dataset(signatures=SIGNATURES, representation=REPRESENTATION):
    _, x, y, b = load(signatures)
    rep = pd.concat([pd.read_parquet(p) for p in sorted(Path(representation).glob('*.parquet'))], ignore_index=True)
    frame = x.merge(rep, on='origin_id', validate='1:1')
    frame['is_event'] = (frame.kind == 'event').astype(float)
    for col in ('ev_event_class', 'ev_activity'):
        for value in sorted(frame[col].dropna().unique()):
            frame[f'ev_{value}'] = (frame[col] == value).astype(float)
    c = b[b.status == 'COMPLETE']
    nets = c.pivot_table(index='origin_id', columns=['plan_id', 'side'], values='net_usd')
    exits = c.pivot_table(index='origin_id', columns=['plan_id', 'side'], values='exit_ns')
    return frame.set_index('origin_id'), nets, exits


def matrix(frame, families):
    cols = []
    fam = family_columns(frame.columns)
    for f in families:
        cols += [c for c in fam[f] if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    if 'event' in families:
        cols += [c for c in frame.columns if c.startswith('ev_') and c not in cols and
                 pd.api.types.is_numeric_dtype(frame[c])]
    cols = list(dict.fromkeys(cols))
    return frame[cols].astype(float).fillna(0.).to_numpy(), cols


def day_weights(dates):
    counts = pd.Series(dates).map(pd.Series(dates).value_counts()).to_numpy(float)
    return 1 / counts


def model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=200,
                                         l2_regularization=1., random_state=SEED)


def fit_predict(X, frame, nets, permute=False):
    """Returns per-plan predictions of m and d for every origin, fitted on A only."""
    a = (frame.block == 'A').to_numpy(); rng = np.random.default_rng(SEED); out = {}
    for p in PLANS:
        if (p, 1) not in nets or (p, -1) not in nets:
            continue
        L = nets[(p, 1)].reindex(frame.index).to_numpy(); S = nets[(p, -1)].reindex(frame.index).to_numpy()
        ok = np.isfinite(L) & np.isfinite(S); train = a & ok
        w = day_weights(frame.date.to_numpy()[train])
        d, m = (L - S) / 2, (L + S) / 2
        target = d[train].copy()
        if permute:
            target = target[rng.permutation(len(target))]
        md, mm = model().fit(X[train], target, sample_weight=w), model().fit(X[train], m[train], sample_weight=w)
        out[p] = dict(d_hat=md.predict(X), m_hat=mm.predict(X), d=d, m=m, ok=ok, L=L, S=S)
    return out


def direction_skill(pred, frame, block):
    rows = {}
    for p, v in pred.items():
        sel = (frame.block == block).to_numpy() & v['ok']
        w = day_weights(frame.date.to_numpy()[sel]); d, dh = v['d'][sel], v['d_hat'][sel]
        hit = np.sign(dh) == np.sign(d)
        rows[str(MENU_PLANS[p])] = dict(n=int(sel.sum()), r2_vs_zero=float(1 - np.average((d - dh) ** 2, weights=w) /
                                                                        np.average(d ** 2, weights=w)),
                                        corr=float(np.corrcoef(d, dh)[0, 1]),
                                        dollar_weighted_hit=float(np.average(hit, weights=w * np.abs(d))))
    return rows


def policy(pred, frame, block, margin=0., flip_side=False):
    """Independent origins: best Q over plans and sides, WAIT when none > margin."""
    sel = (frame.block == block).to_numpy(); rng = np.random.default_rng(SEED + 1)
    best_q = np.full(len(frame), margin); net = np.zeros(len(frame)); act = np.full(len(frame), '', object)
    for p, v in pred.items():
        for side in (1, -1):
            q = v['m_hat'] + side * v['d_hat']
            take = v['ok'] & (q > best_q)
            best_q = np.where(take, q, best_q)
            realised = v['L'] if side > 0 else v['S']
            other = v['S'] if side > 0 else v['L']
            if flip_side:  # same timing/plan, random side: isolates direction skill
                realised = np.where(rng.random(len(frame)) < .5, realised, other)
            net = np.where(take, realised, net)
            act = np.where(take, f'{"long" if side > 0 else "short"}/P{p}', act)
    f = pd.DataFrame(dict(date=frame.date.to_numpy(), net=net, acted=act != ''))[sel]
    traded = f[f.acted]
    return dict(origins=int(sel.sum()), trades=int(len(traded)), net_sum=float(traded.net.sum()),
                net_per_trade=float(traded.net.mean()) if len(traded) else None,
                daily_net=day_boot(f.groupby('date', as_index=False).net.sum(), 'net'),
                actions=pd.Series(act[sel][act[sel] != '']).value_counts().head(8).to_dict())


def ablation(signatures=SIGNATURES, representation=REPRESENTATION):
    frame, nets, _ = dataset(signatures, representation)
    results = {}
    for name, families in SETS.items():
        X, cols = matrix(frame, families)
        pred = fit_predict(X, frame, nets)
        results[name] = dict(features=len(cols), skill_A=direction_skill(pred, frame, 'A'),
                             skill_B=direction_skill(pred, frame, 'B'), policy_A=policy(pred, frame, 'A'),
                             policy_B=policy(pred, frame, 'B'), policy_B_random_side=policy(pred, frame, 'B', flip_side=True))
        print(name, len(cols), json.dumps({k: round(v['r2_vs_zero'], 4) for k, v in results[name]['skill_B'].items()}),
              round(results[name]['policy_B']['net_sum'], 2), results[name]['policy_B']['trades'], flush=True)
    X, cols = matrix(frame, SETS['all'])
    pred = fit_predict(X, frame, nets, permute=True)
    results['all_permuted_direction'] = dict(features=len(cols), skill_B=direction_skill(pred, frame, 'B'),
                                             policy_B=policy(pred, frame, 'B'))
    return results


if __name__ == '__main__':
    import argparse
    from spectrum.ingest import save_json
    p = argparse.ArgumentParser(); p.add_argument('--output', default='reports/schemas_v1/qmodel_ablation.json')
    a = p.parse_args(); Path(a.output).parent.mkdir(parents=True, exist_ok=True); save_json(a.output, ablation())
