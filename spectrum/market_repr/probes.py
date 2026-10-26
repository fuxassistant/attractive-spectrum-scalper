"""Frozen development probes. They consume summaries, not a sequence encoder.

The diagnostic 120s endpoint is independent of existing crossing-path targets.
It measures predictive distinctness, never executable PnL or causal mechanisms.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from .representation import NS, TICK, anchor

BASE = ['ret30_ticks', 'ret300_ticks', 'rv30_ticks', 'range30_ticks', 'range300_ticks',
        'spread_ticks', 'clock_sin', 'clock_cos']
BOOK = ['aggression30', 'unknown_fraction30', 'depth_imbalance', 'volume_decay30', 'log_trade_quantity30',
        'log_depth_bid_12', 'log_depth_ask_12'] + [s + '_' + f for s in ('bid', 'ask') for f in
        ('fill_fraction30', 'withdraw_fraction30', 'refill_fraction30', 'depth_change30')] + [
        'log_' + s + '_' + f for s in ('bid', 'ask') for f in ('near_add30', 'near_fill30', 'near_withdrawal30', 'near_refill30')]
CONTEXT = ['pos_' + v for v in ('day_vwap', 'combined_vwap', 'previous_high', 'previous_low', 'previous_vwap', 'previous_poc', 'day_high', 'day_low')]
CONTEXT += ['day_sigma', 'combined_sigma', 'day_coverage', 'combined_coverage', 'stft_2_8m', 'stft_8_32m', 'stft_32_64m', 'stft_entropy']
FAMILIES = dict(price=BASE, price_book=BASE + BOOK, price_context=BASE + CONTEXT, all=BASE + BOOK + CONTEXT)


def endpoint_outcomes(start, end, deadband_ticks=2):
    # Midpoints live on half ticks. Avoid changing the FLAT label because
    # binary floating point represented an exact 2-tick move as 2+epsilon.
    delta = (np.rint(np.asarray(end) / TICK * 2) - np.rint(np.asarray(start) / TICK * 2)) / 2
    outcome = np.where(delta > deadband_ticks, 2, np.where(delta < -deadband_ticks, 0, 1))
    return delta, outcome


def query_tables(fast, contexts, cfg):
    xs, ys = [], []
    for date, context in contexts.items():
        d = fast[fast.date == date].reset_index(drop=True)
        origin = pd.date_range(date + ' ' + cfg['decision_start_utc'], date + ' ' + cfg['decision_end_utc'],
            tz='UTC', freq=str(cfg['probe']['cadence_seconds']) + 's', inclusive='left').asi8
        av = d.available_ns.to_numpy()
        i = np.searchsorted(av, origin, side='right') - 1
        j = np.searchsorted(av, origin + cfg['probe']['future_seconds'] * NS, side='right') - 1
        assert (i >= 0).all() and (j < len(d)).all()
        row, end = d.iloc[i].reset_index(drop=True), d.iloc[j].reset_index(drop=True)
        known = (row.known & end.known & row.local_segment.eq(end.local_segment) & row.ret300_ticks.notna()).to_numpy()
        c = pd.merge_asof(pd.DataFrame(dict(origin_ns=origin)), context.sort_values('available_ns'),
                          left_on='origin_ns', right_on='available_ns')
        known &= c.known.fillna(False).to_numpy(bool)
        X = row[[n for n in BASE + BOOK if n in row]].copy()
        for v in ('trade_quantity30', 'depth_bid_12', 'depth_ask_12') + tuple(s + '_' + f for s in ('bid', 'ask') for f in ('near_add30', 'near_fill30', 'near_withdrawal30', 'near_refill30')):
            X['log_' + v] = np.log1p(row[v])
        phase = pd.to_datetime(origin, utc=True).tz_convert('Europe/London')
        phase = (phase.hour * 3600 + phase.minute * 60 + phase.second) / 86400 * 2 * np.pi
        X['clock_sin'], X['clock_cos'] = np.sin(phase), np.cos(phase)
        scale = c.bb1_half.clip(lower=.1)
        for name in ('day_vwap', 'combined_vwap', 'previous_high', 'previous_low', 'previous_vwap', 'previous_poc', 'day_high', 'day_low'):
            X['pos_' + name] = (row.mid - c[name]) / scale
        for name in CONTEXT:
            if not name.startswith('pos_'): X[name] = c[name]
        X['date'] = date; X['origin_ns'] = origin
        X['root_id'] = [f'{date}:{int(t)}' for t in origin]
        X['available_ns'] = np.maximum(row.available_ns, c.available_ns)
        threshold = cfg['probe']['direction_deadband_ticks']
        delta, y = endpoint_outcomes(row.mid, end.mid, threshold)
        Y = pd.DataFrame(dict(root_id=X.root_id, date=date, origin_ns=origin,
             mature_ns=origin + cfg['probe']['future_seconds'] * NS,
             origin_quote_ns=row.t_ns, future_quote_ns=end.t_ns,
             delta_ticks=delta, outcome=y, known=known))
        assert (X.available_ns <= X.origin_ns).all()
        xs.append(X); ys.append(Y)
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def metrics(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    onehot = np.eye(3)[y]
    directional = y != 1
    score = p[:, 2] / np.maximum(1e-12, p[:, 0] + p[:, 2])
    return dict(n=len(y), log_loss=float(log_loss(y, p, labels=[0, 1, 2])),
        brier=float(np.mean(np.sum((p - onehot)**2, axis=1))),
        accuracy=float(accuracy_score(y, p.argmax(axis=1))),
        directional_auc=float(roc_auc_score(y[directional] == 2, score[directional])) if len(np.unique(y[directional])) == 2 else None)


def fit_probes(X, Y, cfg):
    usable = Y.known.to_numpy(bool)
    train = usable & X.date.isin(cfg['splits']['train']).to_numpy()
    y = Y.outcome.to_numpy(int)
    predictions = X[['root_id', 'date', 'origin_ns']].copy()
    predictions['known'] = usable
    report = dict(target='Diagnostic three-way midpoint displacement at 120s: DOWN < -2 ticks, FLAT otherwise, UP > 2 ticks. Not a crossing target or economic return.',
        common_cohort='Regular 30s decision grid; known origin/future price, uninterrupted local support and available 300s past. Flat/quiet examples retained.',
        architecture='Small histogram gradient boosting on scalar summaries; no claim that a sequence model consumes the full two-day tensor.',
        tuning='One frozen configuration, no validation tuning or test model selection.',
        class_counts={split: np.bincount(y[usable & X.date.isin(dates).to_numpy()], minlength=3).tolist() for split, dates in cfg['splits'].items()},
        families={k: v for k, v in FAMILIES.items()}, results={}, false_signal_diagnostics={})
    class_prior = np.bincount(y[train], minlength=3).astype(float)
    class_prior /= class_prior.sum()
    for name, columns in {'prior': [] , **FAMILIES}.items():
        if name == 'prior':
            p = np.tile(class_prior, (len(X), 1))
        else:
            model = HistGradientBoostingClassifier(max_iter=cfg['probe']['max_iter'],
                max_leaf_nodes=cfg['probe']['max_leaf_nodes'], min_samples_leaf=cfg['probe']['min_samples_leaf'],
                l2_regularization=cfg['probe']['l2_regularization'], random_state=cfg['probe']['seed'], early_stopping=False)
            model.fit(X.loc[train, columns], y[train])
            p = model.predict_proba(X[columns])
            assert model.classes_.tolist() == [0, 1, 2]
        for j, cls in enumerate(('down', 'flat', 'up')): predictions[name + '_' + cls] = p[:, j]
        result = {}
        for split, dates in {**cfg['splits'], **{d: [d] for d in cfg['splits']['test']}}.items():
            mask = usable & X.date.isin(dates).to_numpy()
            result[split] = metrics(y[mask], p[mask])
        report['results'][name] = result
        mask = usable & X.date.isin(cfg['splits']['test']).to_numpy()
        signal = mask & (np.maximum(p[:, 0], p[:, 2]) >= .65)
        direction = np.where(p[:, 2] > p[:, 0], 1, -1)
        delta = Y.delta_ticks.to_numpy()
        report['false_signal_diagnostics'][name] = dict(threshold=.65, signals=int(signal.sum()),
            opposite=int((signal & (direction * delta < -2)).sum()), flat=int((signal & (np.abs(delta) <= 2)).sum()),
            mean_signed_ticks=float(np.mean((direction * delta)[signal])) if signal.any() else None,
            interpretation='Conditional endpoint diagnostic, overlapping 120s labels; not trades, costs or PnL.')
    # Paired hourly resampling is descriptive; only two test dates exist.
    test = usable & X.date.isin(cfg['splits']['test']).to_numpy()
    a = predictions[['all_down', 'all_flat', 'all_up']].to_numpy()
    b = predictions[['price_down', 'price_flat', 'price_up']].to_numpy()
    losses = -np.log(np.maximum(a[np.arange(len(y)), y], 1e-12)) + np.log(np.maximum(b[np.arange(len(y)), y], 1e-12))
    blocks = pd.DataFrame(dict(hour=X.origin_ns[test] // (3600 * NS), loss=losses[test])).groupby('hour').loss.agg(['sum', 'count'])
    rng = np.random.default_rng(cfg['probe']['seed'])
    samples = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    boot = blocks['sum'].to_numpy()[samples].sum(axis=1) / blocks['count'].to_numpy()[samples].sum(axis=1)
    report['paired_hourly_interval'] = dict(delta_log_loss_all_minus_price=float(losses[test].mean()),
        percentile_95=np.quantile(boot, [.025, .975]).tolist(), blocks=len(blocks),
        limitation='Development description with only two test dates; not an estimate of independent new-regime uncertainty.')
    return predictions, report
