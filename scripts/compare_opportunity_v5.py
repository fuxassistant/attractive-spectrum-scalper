"""Descriptive paired TEST-day audit of the prespecified V5 primary setting.

No threshold, model, seed, or economic plan is selected with TEST labels.  The
two seeds are averaged inside each date before resampling five whole dates.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_opportunity_v5 import ARMS, alarm_metrics, score_channels


REPORT = Path('reports/scientific_pipeline_v5')
CONFIG = Path('configs/research/opportunity_pipeline_v5.json')
BOOTSTRAP_SEED = 20261023
REPLICATIONS = 20000
SIGNAL_METRICS = ('precision', 'safe_precision', 'none_fpr',
                  'event_opportunity_recall', 'safe_recall')
ECONOMIC_METRICS = ('complete_daily_cash', 'complete_daily_trades')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def independent_retention_threshold(score, eligible, fraction):
    """Ascending order statistic, independent of evaluator's descending sort."""
    values = np.asarray(score, np.float64)[np.asarray(eligible, bool)]
    if not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError('Finite CAL scores and a valid retention are required')
    count = int(np.floor(fraction * len(values)))
    if not len(values):
        threshold = 1.01
    elif fraction == 1:
        threshold = 0.
    else:
        index = len(values) - count - 1
        threshold = float(np.nextafter(np.partition(values, index)[index], np.inf))
    actual = int(np.count_nonzero(values >= threshold))
    if actual > count:
        raise AssertionError('Conservative inclusive retention budget exceeded')
    return dict(fraction=float(fraction), calibration_candidates=len(values),
                allowed=count, actual=actual, threshold=threshold,
                conservative_ties=count - actual)


def interval(values, dates):
    values = np.asarray(values, np.float64)
    if values.shape != (len(dates),) or not np.isfinite(values).all():
        raise ValueError('Paired finite day values required')
    if not len(values):
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = values[rng.integers(0, len(values), (REPLICATIONS, len(values)))].mean(1)
    return dict(days=len(values), mean=float(values.mean()),
                percentile_95_interval=np.quantile(boot, [.025, .975]).tolist(),
                positive_days=int((values > 0).sum()),
                by_day=[dict(date=day, effect=float(v)) for day, v in zip(dates, values)],
                replications=REPLICATIONS, bootstrap_seed=BOOTSTRAP_SEED)


def safe_daily(economic, date):
    daily = economic['daily'][date]
    complete = daily['status'] == 'COMPLETE' and daily['total_net_if_complete'] is not None
    return dict(status=daily['status'], projection_complete=bool(economic['completed']),
                complete_daily_cash=float(daily['total_net_if_complete']) if complete else None,
                complete_daily_trades=int(daily['trades']) if complete else None,
                observed_partial_cash=None if complete else float(daily['observed_closed_cash']),
                censored=int(daily['censored']))


def paired_effect(rows, arm, metric, seeds, dates):
    economic = metric in ECONOMIC_METRICS
    field = 'economics' if economic else 'signal'
    sign = -1. if metric == 'none_fpr' else 1.
    lookup = {(r['seed'], r['date']): r for r in rows}
    # Never promote a completed prefix to a full-period economic comparison.
    partial = [dict(seed=s, arm=a, date=d)
               for s in seeds for d in dates for a in ('base', arm)
               if economic and not lookup[s, d]['arms'][a]['economics']['projection_complete']]
    base = dict(metric=metric, contrast=arm + '_vs_base',
                sign='base-minus-arm' if sign < 0 else 'arm-minus-base',
                positive_means='more trades, not an improvement' if metric == 'complete_daily_trades'
                else 'improvement', unit='USD/reference ounce/day' if metric == 'complete_daily_cash'
                else 'trades/day' if economic else 'fractional probability/rate difference')
    if partial:
        return dict(**base, status='SKIPPED_CENSORED_PROJECTION',
                    mean_seed_day_interval=None, per_seed=[],
                    reason='At least one entire TEST account is incomplete; no prefix-day bootstrap.',
                    affected_seed_arm=sorted({(x['seed'], x['arm']) for x in partial}))
    usable = [d for d in dates if all(
        lookup[s, d]['arms'][a][field][metric] is not None
        for s in seeds for a in ('base', arm))]
    per_seed = []
    vectors = []
    for seed in seeds:
        values = [sign * (lookup[seed, day]['arms'][arm][field][metric]
                          - lookup[seed, day]['arms']['base'][field][metric]) for day in usable]
        vectors.append(values)
        per_seed.append(dict(seed=seed, interval=interval(values, usable)))
    if not usable:
        return dict(**base, status='UNDEFINED_ZERO_DENOMINATORS', mean_seed_day_interval=None,
                    per_seed=per_seed, excluded_dates=dates)
    means = np.asarray(vectors, float).mean(axis=0)
    seed_means = np.asarray(vectors, float).mean(axis=1)
    direction = np.sign(seed_means).astype(int)
    return dict(**base, status='COMPLETE' if usable == dates else 'PARTIAL_DEFINED_DAY_COHORT',
                mean_seed_day_interval=interval(means, usable), per_seed=per_seed,
                excluded_dates=[d for d in dates if d not in usable],
                seed_consistency=dict(same_direction=bool(np.all(direction == direction[0])),
                    signs=direction.tolist(), positive_seeds=int((direction > 0).sum()),
                    negative_seeds=int((direction < 0).sum()),
                    means={str(s): float(v) for s, v in zip(seeds, seed_means)}))


def assert_equal(actual, expected, location):
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'Keys differ: {location}')
        for key in actual:
            assert_equal(actual[key], expected[key], location + '.' + key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f'Length differs: {location}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equal(a, b, f'{location}[{i}]')
    elif isinstance(actual, float):
        if expected is None or not np.isclose(actual, expected, rtol=0, atol=1e-14):
            raise AssertionError(f'Value differs: {location}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(f'Value differs: {location}: {actual} != {expected}')


def synthetic_checks():
    # Ties must be excluded conservatively under inclusive >= comparison.
    r = independent_retention_threshold([.2, .4, .4, .9], [True] * 4, .5)
    assert r['actual'] == 1 and r['threshold'] > .4
    assert independent_retention_threshold([], [], .5)['actual'] == 0
    assert independent_retention_threshold([.2, .4], [True] * 2, 1.)['actual'] == 2
    x = np.array([.1, .2, .3, .4, .99])
    eligible = np.array([True, True, True, True, False])
    r = independent_retention_threshold(x, eligible, .5)
    x[-1] = -999.
    assert independent_retention_threshold(x, eligible, .5) == r
    e = dict(completed=False, daily={'a': dict(status='PARTIAL',
        total_net_if_complete=None, observed_closed_cash=3., trades=2, censored=1)})
    assert safe_daily(e, 'a')['complete_daily_cash'] is None
    assert safe_daily(e, 'a')['complete_daily_trades'] is None
    row = dict(seed=17, date='a', arms={a: dict(economics=safe_daily(e, 'a'))
                                      for a in ('base', 'risk')})
    assert paired_effect([row], 'risk', 'complete_daily_cash', [17], ['a'])[
        'mean_seed_day_interval'] is None
    assert interval([1., 3.], ['a', 'b'])['mean'] == 2.
    return dict(status='PASS', cases=['conservative_ties', 'empty_CAL', 'full_retention',
                'non_CAL_mutation', 'censored_cash_and_trades_missing',
                'censored_whole_projection_skips_interval', 'paired_mean'])


def main():
    started = time.perf_counter()
    checks = synthetic_checks()
    config, suite, evaluation = read(CONFIG), read(REPORT / 'suite_summary.json'), read(REPORT / 'evaluation.json')
    root = Path(config['data']['root'])
    manifest = read(root / 'manifest.json')
    bindings = {str(CONFIG): sha(CONFIG), str(REPORT / 'suite_summary.json'): sha(REPORT / 'suite_summary.json'),
                str(REPORT / 'evaluation.json'): sha(REPORT / 'evaluation.json'),
                str(root / 'manifest.json'): sha(root / 'manifest.json')}
    if any(z['status'] != 'COMPLETE' for z in (suite, evaluation, manifest)):
        raise ValueError('Complete suite, evaluation, and dataset required')
    for owner in (suite, evaluation):
        if owner['config_sha256'] != bindings[str(CONFIG)] or owner['dataset_manifest_sha256'] != bindings[str(root / 'manifest.json')]:
            raise ValueError('Configuration/dataset identity mismatch')
    if evaluation['suite_sha256'] != bindings[str(REPORT / 'suite_summary.json')]:
        raise ValueError('Evaluation suite identity mismatch')
    plan, fraction = config['experiment']['primary_plan'], config['experiment']['primary_retention']
    if plan != 'cost2' or fraction != .5:
        raise ValueError('This audit is prespecified for cost2 and CAL retention 0.5')
    dates, seeds = config['splits']['test'], config['experiment']['seeds']
    if len(dates) != 5 or seeds != [17, 29]:
        raise ValueError('Expected five TEST dates and two prespecified seeds')
    parts = []
    for day in manifest['days']:
        path = root / (day['date'] + '_roots.parquet')
        digest = sha(path)
        if digest != day['files'][path.name]:
            raise ValueError('Changed root table: ' + str(path))
        bindings[str(path)] = digest
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    before_available = frame.feature_available.to_numpy(bool, copy=True)
    if len(frame) != manifest['roots'] or not frame.t_ns.is_monotonic_increasing or frame.t_ns.duplicated().any():
        raise ValueError('Malformed full root clock')
    calibration = frame.date.isin(config['splits']['calibration']).to_numpy()
    test = frame.date.isin(dates).to_numpy()
    rows, pooled, thresholds = [], [], []
    index = {r['seed']: r for r in evaluation['runs']}
    if set(index) != set(seeds) or len(index) != len(evaluation['runs']):
        raise ValueError('Evaluation seed cohort differs')
    for run in suite['runs']:
        seed = run['seed']
        out = Path(suite['runs_root']) / run['run_id']
        receipt = read(out / 'receipt.json')
        if receipt['status'] != 'COMPLETE' or receipt['seed'] != seed:
            raise ValueError('Incomplete/wrong fit receipt')
        bindings[str(out / 'receipt.json')] = sha(out / 'receipt.json')
        for name, digest in receipt['sources'].items():
            path = Path(name)
            if sha(path) != digest:
                raise ValueError('Changed declared fit source: ' + str(path))
            bindings[str(path)] = digest
        for name, digest in receipt['outputs'].items():
            path = out / name
            if sha(path) != digest:
                raise ValueError('Changed fit artifact: ' + str(path))
            bindings[str(path)] = digest
        pred = pd.read_parquet(out / 'predictions.parquet')
        for column in ('date', 't_ns', 'feature_available', 'y_crossing'):
            if not np.array_equal(pred[column], frame[column]):
                raise ValueError('Prediction/root mismatch: ' + column)
        if not np.array_equal(pred.root_index, np.arange(len(frame))):
            raise ValueError('Prediction ordering mismatch')
        channels, side = score_channels(pred, seed)
        candidate = channels[plan]['base'] >= config['experiment']['candidate_floor']
        eligible = calibration & frame.feature_available.to_numpy(bool) & candidate
        safe = pred['target_q_safe_cost2'].to_numpy(float)
        if not np.isin(safe, [-1., 0., 1.]).all():
            raise ValueError('Unexpected risk-label mask/value')
        by_day = {d: dict(seed=seed, date=d, arms={}) for d in dates}
        for arm in ARMS:
            published = index[seed]['primary'][arm]
            if published['plan'] != plan or published['fraction'] != fraction:
                raise ValueError('Wrong published primary setting')
            score = channels[plan][arm]
            rule = independent_retention_threshold(score, eligible, fraction)
            assert_equal(rule, published['threshold_rule'], f'{seed}.{arm}.CAL_rule')
            thresholds.append(dict(seed=seed, arm=arm, **rule, independent_check='EXACT'))
            alert = score >= rule['threshold']
            actual = alarm_metrics(frame, side, safe, candidate, alert, test)
            assert_equal(actual, published['test_signal'], f'{seed}.{arm}.TEST_signal')
            assert_equal(alarm_metrics(frame, side, safe, candidate, alert, calibration),
                         published['calibration_signal'], f'{seed}.{arm}.CAL_signal')
            econ = published['test_economics']
            pooled.append(dict(seed=seed, arm=arm, signal=actual, economics=dict(
                status=econ['status'], completed=econ['completed'],
                total_net_if_complete=econ['total_net_if_complete'],
                trades_if_complete=econ['trades'] if econ['completed'] else None,
                halt=econ['halt']), economic_selector= index[seed]['selected'].get(arm)))
            if econ['completed']:
                if not all(econ['daily'][d]['status'] == 'COMPLETE' for d in dates):
                    raise AssertionError('Complete projection has incomplete daily rows')
                if not np.isclose(sum(econ['daily'][d]['total_net_if_complete'] for d in dates),
                                  econ['total_net_if_complete'], rtol=0, atol=1e-8):
                    raise AssertionError('Complete daily cash does not sum to full account')
                if sum(econ['daily'][d]['trades'] for d in dates) != econ['trades']:
                    raise AssertionError('Complete daily trades do not sum to full account')
            for day in dates:
                signal = alarm_metrics(frame, side, safe, candidate, alert, frame.date.eq(day).to_numpy())
                by_day[day]['arms'][arm] = dict(signal=signal, economics=safe_daily(econ, day))
        rows.extend(by_day.values())
    if not np.array_equal(before_available, frame.feature_available):
        raise AssertionError('Metric helper mutated root admission')
    if len(rows) != 10 or {(r['seed'], r['date']) for r in rows} != {(s, d) for s in seeds for d in dates}:
        raise AssertionError('Expected exactly ten unique seed-day rows')
    effects = {arm + '_vs_base': {metric: paired_effect(rows, arm, metric, seeds, dates)
               for metric in SIGNAL_METRICS + ECONOMIC_METRICS} for arm in ARMS if arm != 'base'}
    source_paths = [Path(__file__), Path('scripts/evaluate_opportunity_v5.py'),
                    Path('scripts/audit_opportunity_v4.py'), Path('scripts/audit_economics_v4.py')]
    for path in source_paths:
        bindings[str(path.relative_to(Path.cwd()) if path.is_absolute() else path)] = sha(path)
    result = dict(status='COMPLETE', schema='V5_PRESPECIFIED_PAIRED_PRIMARY_DAY_DIAGNOSTICS',
        setting=dict(plan=plan, calibration_retention=fraction,
                     candidate_floor=config['experiment']['candidate_floor'], seeds=seeds, test_dates=dates),
        definitions=dict(
            admission='Current causal feature_available and frozen base candidate floor only; unknown future labels do not exclude alerts.',
            alert='Inclusive score>=numeric threshold, calibrated once on CAL candidates without labels and frozen on TEST.',
            precision='Correct frozen chosen side / all known alerts; BOTH counts correct for either chosen side.',
            safe_precision='Correct chosen side and cost2 pre-cross safety / all known alerts. Unknown risk is not counted safe.',
            none_fpr='NONE alerts / all admitted known NONE roots, not merely candidate NONE roots.',
            event_opportunity_recall='Correct chosen-side alerts / all admitted known roots with any event (LONG, SHORT, BOTH).',
            safe_recall='Safe correct alerts / all admitted known roots whose frozen chosen side is correct and risk-known safe.',
            economics='Existing continuous one-position account; USD/reference ounce after stated fee/latency. Signals are not trades.',
            censor='Partial daily cash/trades are null as complete outcomes. Any incomplete full TEST account suppresses its paired economic interval.',
            pairing='Arm-minus-base within each seed/date (base-minus-arm for NONE FPR); then equal seed average inside date; then equal-weight whole-date resampling.',
            inference='Descriptive percentile interval from 20000 whole-date bootstrap draws. Only five TEST dates; dates need not be independent. Seeds are not extra days. No multiple-comparison correction.',
            trade_sign='Positive trade-count difference is more activity, not a quality improvement.',
            selector='Primary setting is separate from the CAL economic selector. No TEST winner or threshold is selected.',
            historical_scope='Dates are new relative to V3/V4 but not claimed never inspected historically; base GCQ6 versus new GCZ6 conflates temporal and contract changes.'),
        checks=dict(synthetic=checks, source_output_hashes='PASS', independent_CAL_thresholds='EXACT',
                    pooled_CAL_TEST_metrics='MATCH', root_alignment='EXACT', admission_unmutated=True,
                    complete_daily_economics_sum='PASS'),
        thresholds=thresholds, pooled=pooled, seed_day_rows_first_ten=rows,
        paired_effects=effects, sources_sha256=bindings, seconds=time.perf_counter() - started)
    target = REPORT / 'paired_diagnostics.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE', path=str(target), sha256=sha(target),
                         seconds=result['seconds'], checks=result['checks']), ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
