"""Independent V5 CAL-budget/account audit and raw pre-cross risk proofs.

Uses the frozen V2 account with explicit score and side arrays, not the V5
adapter or V4 project_account. Labels are attached only after each replay.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.verify_opportunity_v3 import BoundInputs, read_json, sha
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.research_v2.evaluation import _aligned, _projection
from spectrum.research_v5.targets import build_filter_targets, verify_risk_targets_raw

PLANS = ('cost2', 'cost4', 'cost6')
ARMS = ('base', 'semantic', 'risk', 'hash_control')
PNAMES = ('p_none', 'p_long', 'p_short', 'p_both')
QNAMES = ('q_event', 'q_correct', 'q_safe_cost2', 'q_safe_cost4', 'q_safe_cost6')
RISK_BUCKETS = ('UNKNOWN_TEACHER', 'NONE', 'WRONG_SIDE', 'CORRECT_RISK_UNKNOWN', 'CORRECT_RISK_FIRST', 'CORRECT_SAFE')
RAW_DAYS = ('2026-07-30', '2026-08-17', '2026-09-01')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def scalar_hash_scores(t_ns, seed):
    """Independent scalar integer SplitMix64; only causal timestamp and seed."""
    mask = (1 << 64)-1
    def one(t):
        z = ((int(t) ^ int(seed))+0x9e3779b97f4a7c15) & mask
        z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
        z ^= z >> 31
        return (float(z >> 11)+.5)/(2.**53)
    return np.fromiter((one(t) for t in t_ns), dtype=np.float64, count=len(t_ns))


def independent_scores(pred, seed):
    p, q = pred[list(PNAMES)].to_numpy(float), pred[list(QNAMES)].to_numpy(float)
    if (not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (p > 1).any()
            or (q < 0).any() or (q > 1).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Finite normalized base and conditional filter probabilities required')
    long, short = p[:, 1]+p[:, 3], p[:, 2]+p[:, 3]
    side = np.where(short > long, -1, 1)
    if not np.array_equal(side, pred.side):
        raise ValueError('Frozen base side changed')
    semantic = q[:, 0]*q[:, 1]
    h = scalar_hash_scores(pred.t_ns.to_numpy(np.int64), seed)
    return {plan: dict(base=np.maximum(long, short), semantic=semantic,
        risk=semantic*q[:, j+2], hash_control=h) for j, plan in enumerate(PLANS)}, side


def independent_retention(score, cal, fraction):
    values = np.asarray(score, float)[np.asarray(cal, bool)]
    if not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError('Finite CAL-only retention query required')
    n = len(values)
    budget = int(np.floor(fraction*n))
    if not n:
        threshold = 1.01
    elif budget == n:
        threshold = 0.
    else:
        # k-th descending value is the (n-k-1)-th ascending order statistic.
        boundary = np.partition(values.copy(), n-budget-1)[n-budget-1]
        threshold = float(np.nextafter(boundary, np.inf))
    actual = int(np.count_nonzero(values >= threshold))
    if actual > budget:
        raise AssertionError('CAL retention budget exceeded')
    return dict(fraction=float(fraction), calibration_candidates=n, allowed=budget,
                actual=actual, threshold=threshold, conservative_ties=budget-actual)


def reference_replay(aligned, score, side, candidate, plan, threshold, days, pending):
    score, side = np.asarray(score, float), np.asarray(side)
    if (score.shape != aligned['t'].shape or side.shape != score.shape or not np.isin(side, [1, -1]).all()
            or not np.isfinite(score).all() or (score < 0).any() or (score > 1).any()):
        raise ValueError('One causal score and frozen side per root required')
    direct = dict(aligned, score=score, side=np.where(side == 1, 0, 1),
                  available=aligned['available'] & np.asarray(candidate, bool))
    return _projection(direct, direct['plans'].index(plan), threshold, days, pending, True)


def verify_saved_projection(saved, summary, ledger):
    for key, value in summary.items():
        if key in ('net', 'daily'):
            continue
        if saved[key] != value:
            raise AssertionError('Independent V2 replay summary differs: '+key)
    if saved['observed_closed_cash'] != summary['net'] or saved['total_net_if_complete'] != (summary['net'] if summary['completed'] else None):
        raise AssertionError('Censored prefix presented as full economics')
    if saved['ledger_sha256'] != digest(ledger) or saved['ledger_rows'] != len(ledger):
        raise AssertionError('Independent frozen-side ledger differs')
    halt_date = summary['halt']['date'] if summary['halt'] else None
    cumulative = 0.
    for date, day in summary['daily'].items():
        observed = halt_date is None or date <= halt_date
        complete = halt_date is None or date < halt_date
        if observed:
            cumulative += day['net']
        expected = dict(status='COMPLETE' if complete else ('PARTIAL' if observed else 'NOT_OBSERVED'),
            observed_closed_cash=day['net'] if observed else None,
            cumulative_closed_cash=cumulative if observed else None,
            total_net_if_complete=day['net'] if complete else None,
            **{k: v if observed else None for k, v in day.items() if k != 'net'})
        if saved['daily'][date] != expected:
            raise AssertionError('Daily censor-safe accounting differs')


def risk_economics(frame, targets, plan, summary, ledger):
    by_time = {int(t): i for i, t in enumerate(frame.t_ns)}
    selected = {key: [] for key in RISK_BUCKETS}
    for item in ledger:
        i = by_time[int(item['t_ns'])]
        y = int(frame.y_crossing.iloc[i])
        if y < 0:
            bucket = 'UNKNOWN_TEACHER'
        elif y == 0:
            bucket = 'NONE'
        elif not (y & (1 if item['side'] == 1 else 2)):
            bucket = 'WRONG_SIDE'
        else:
            safe = int(targets['y_risk_safe_'+plan].iloc[i])
            bucket = 'CORRECT_RISK_UNKNOWN' if safe < 0 else ('CORRECT_SAFE' if safe else 'CORRECT_RISK_FIRST')
        selected[bucket].append(item)
    buckets = {}
    for key, rows in selected.items():
        done = [x for x in rows if x['status'] == 'COMPLETE']
        net = np.asarray([x['net'] for x in done], float)
        buckets[key] = dict(attempts=len(rows), completed_trades=len(done),
            empty=sum(x['status'] == 'EMPTY' for x in rows), censored=sum(x['status'] == 'CENSORED' for x in rows),
            observed_closed_net=float(net.sum()), mean_net=float(net.mean()) if len(net) else None,
            wins=int((net > 0).sum()), losses=int((net < 0).sum()),
            exit_reasons={r: sum(x['reason'] == r for x in done) for r in ('TAKE', 'STOP', 'TIME')})
    held = sum((int(x['exit_ns'])-int(x['entry_ns']))/1e9 for x in ledger if x['status'] == 'COMPLETE')
    if sum(b['completed_trades'] for b in buckets.values()) != summary['trades']:
        raise AssertionError('Post-execution risk buckets lost a completed trade')
    if not np.isclose(sum(b['observed_closed_net'] for b in buckets.values()), summary['net'], atol=1e-8, rtol=0):
        raise AssertionError('Post-execution risk buckets do not reconcile money')
    return dict(completed=summary['completed'], trades=summary['trades'], censored=summary['censored'],
        observed_closed_cash=summary['net'], total_net_if_complete=summary['net'] if summary['completed'] else None,
        completed_trade_holding_seconds=held, halt=summary['halt'], buckets=buckets,
        ledger_sha256=digest(ledger), account_state_matching=False)


def verify_signal_counts(saved, frame, side, safe, candidate, alert, cohort):
    available = frame.feature_available.to_numpy(bool) & cohort
    y = frame.y_crossing.to_numpy(int)
    known = available & (y >= 0)
    chosen = ((y & np.where(side == 1, 1, 2)) != 0) & (y > 0) & known
    hit = np.asarray(alert, bool) & candidate
    values = dict(known_roots=int(known.sum()), unknown_roots=int((available & (y < 0)).sum()),
        causal_candidates=int((candidate & available).sum()), alerts=int((hit & known).sum()),
        unknown_alerts=int((hit & available & (y < 0)).sum()), total_causal_alerts=int((hit & available).sum()),
        true_positives=int((hit & chosen).sum()), none_false_positives=int((hit & known & (y == 0)).sum()),
        wrong_side_alerts=int((hit & known & (y > 0) & ~chosen).sum()),
        safe_correct_alerts=int((hit & chosen & (safe == 1)).sum()),
        unsafe_correct_alerts=int((hit & chosen & (safe == 0)).sum()),
        unknown_risk_correct_alerts=int((hit & chosen & (safe < 0)).sum()))
    values['false_signals'] = values['none_false_positives']+values['wrong_side_alerts']
    if any(saved[k] != v for k, v in values.items()):
        raise AssertionError('Saved root alert counts differ from independent partition')
    return values


def audit():
    config_path = Path('configs/research/opportunity_pipeline_v5.json')
    report = Path('reports/scientific_pipeline_v5')
    c, suite = read_json(config_path), read_json(report/'suite_summary.json')
    saved = read_json(report/'evaluation.json')
    if suite['status'] != 'COMPLETE' or saved['status'] != 'COMPLETE':
        raise RuntimeError('Complete fits and V5 evaluation required')
    if suite['config_sha256'] != sha(config_path) or saved['suite_sha256'] != sha(report/'suite_summary.json'):
        raise ValueError('Evaluation/suite/config binding changed')
    bound = BoundInputs()
    for p in (__file__, 'scripts/verify_opportunity_v3.py', 'spectrum/research_v2/evaluation.py',
              'scripts/evaluate_opportunity_v5.py', config_path, report/'suite_summary.json', report/'evaluation.json'):
        bound.bind(p)
    root = Path(c['data']['root'])
    bound.bind(root/'manifest.json', suite['dataset_manifest_sha256'])
    manifest = read_json(root/'manifest.json')
    frames, trials = [], []
    for daily in manifest['days']:
        if daily['status'] != 'COMPLETE':
            raise ValueError('Incomplete daily dataset')
        for name, collector in (('roots', frames), ('execution', trials)):
            p = root/f"{daily['date']}_{name}.parquet"
            bound.bind(p, daily['files'][p.name])
            collector.append(pd.read_parquet(p))
    f, executions = pd.concat(frames, ignore_index=True), pd.concat(trials, ignore_index=True)
    results, targets_by_seed = {}, {}
    for run in suite['runs']:
        seed, out = run['seed'], Path(suite['runs_root'])/run['run_id']
        receipt = read_json(out/'receipt.json')
        if receipt['status'] != 'COMPLETE':
            raise ValueError('Incomplete filter run')
        bound.bind(out/'receipt.json')
        for path, value in receipt['sources'].items():
            bound.bind(path, value)
        for name, value in receipt['base']['files'].items():
            bound.bind(Path(receipt['base']['root'])/name, value)
        for name in ('predictions.parquet', 'targets.parquet', 'target_receipt.json'):
            bound.bind(out/name, receipt['outputs'][name])
        pred, target = pd.read_parquet(out/'predictions.parquet'), pd.read_parquet(out/'targets.parquet')
        for name in ('date', 't_ns', 'feature_available', 'y_crossing'):
            if not np.array_equal(pred[name], f[name]):
                raise ValueError('Prediction/origin identity differs')
        channels, side = independent_scores(pred, seed)
        for daily in manifest['days']:
            use = f.date.eq(daily['date']).to_numpy()
            pmap = root/f"{daily['date']}_map.parquet"
            bound.bind(pmap, daily['files'][pmap.name])
            rebuilt = build_filter_targets(f.loc[use].reset_index(drop=True), pd.read_parquet(pmap), side[use], c)
            pd.testing.assert_frame_equal(target.loc[use].reset_index(drop=True), rebuilt['frame'], check_exact=True)
            for j, name in enumerate(QNAMES):
                if not np.array_equal(pred.loc[use, 'target_'+name].to_numpy(), rebuilt['labels'][:, j]):
                    raise AssertionError('Saved head label differs from exact conditional target rebuild')
        targets_by_seed[seed] = target
        aligned = _aligned(f, executions, pred[list(PNAMES)].to_numpy(), c)
        candidate = channels['cost2']['base'] >= c['experiment']['candidate_floor']
        roles = {role: f.date.isin(dates).to_numpy() for role, dates in c['splits'].items()}
        cal = roles['calibration'] & f.feature_available.to_numpy() & candidate
        result = next(r for r in saved['runs'] if r['seed'] == seed)
        inspected, primary = [], {}
        for item in result['grid']:
            arm, plan, fraction = item['arm'], item['plan'], item['fraction']
            score = channels[plan][arm]
            rule = independent_retention(score, cal, fraction)
            if rule != item['threshold_rule']:
                raise AssertionError('CAL-only order-statistic retention differs')
            proofs = {}
            for role in ('calibration', 'test'):
                summary, ledger = reference_replay(aligned, score, side, candidate, plan, rule['threshold'],
                    c['splits'][role], c['execution']['pending_timeout_seconds'])
                verify_saved_projection(item[role+'_economics'], summary, ledger)
                counts = verify_signal_counts(item[role+'_signal'], f, side,
                    target['y_risk_safe_'+plan].to_numpy(), candidate, score >= rule['threshold'], roles[role])
                proofs[role] = dict(signal_counts=counts, economics=risk_economics(f, target, plan, summary, ledger))
                if role+'_ledger' in item:
                    bound.bind(item[role+'_ledger'])
                    if read_json(item[role+'_ledger']) != ledger:
                        raise AssertionError('Saved primary ledger differs from independent V2 account')
            record = dict(arm=arm, plan=plan, fraction=fraction, threshold_rule=rule, **proofs)
            inspected.append(record)
            if plan == c['experiment']['primary_plan'] and fraction == c['experiment']['primary_retention']:
                if result['primary'][arm] != item:
                    raise AssertionError('Saved primary arm differs from its preregistered grid cell')
                primary[arm] = record
        expected_keys = {(a, p, float(v)) for a in ARMS for p in PLANS for v in c['experiment']['retention_fractions']}
        actual_keys = {(x['arm'], x['plan'], float(x['fraction'])) for x in inspected}
        if actual_keys != expected_keys or len(inspected) != len(expected_keys):
            raise AssertionError('Missing or duplicated prespecified grid arm')
        selected = {}
        for arm in ('base', 'semantic', 'risk'):
            valid = [x for x in inspected if x['arm'] == arm and x['calibration']['economics']['completed']
                and x['calibration']['economics']['trades'] >= c['experiment']['minimum_calibration_trades']
                and x['calibration']['economics']['total_net_if_complete'] > 0]
            old = result['selected'][arm]
            if valid:
                best = sorted(valid, key=lambda x: (-x['calibration']['economics']['total_net_if_complete'],
                    -x['threshold_rule']['threshold'], float(x['plan'][4:])))[0]
                expected = ('positive_complete_CAL', best['plan'], best['fraction'], best['threshold_rule']['threshold'])
            else:
                expected = ('WAIT_no_positive_complete_CAL_with_50_trades', 'cost2', 0., 1.01)
            if (old['reason'], old['plan'], old['fraction'], old['threshold']) != expected:
                raise AssertionError('CAL economic selection changed')
            chosen = {}
            for role in ('calibration', 'test'):
                summary, ledger = reference_replay(aligned, channels[old['plan']][arm], side, candidate,
                    old['plan'], old['threshold'], c['splits'][role], c['execution']['pending_timeout_seconds'])
                verify_saved_projection(old[role], summary, ledger)
                chosen[role] = risk_economics(f, target, old['plan'], summary, ledger)
            selected[arm] = dict(reason=old['reason'], plan=old['plan'], fraction=old['fraction'], threshold=old['threshold'], **chosen)
        zero = {p: {a: int((candidate & f.feature_available.to_numpy() & (channels[p][a] == 0) & (side == -1)).sum()) for a in ARMS} for p in PLANS}
        results[str(seed)] = dict(status='PASS', exact_grid_account_and_signal_count_parity=True,
            exact_saved_target_rebuild=True, grid=inspected, primary=primary, selected=selected,
            zero_score_short_candidates=zero, matching='CAL causal root-alert count budget only; actual completed trade count and holding time shown separately. TEST counts may drift.')
        print(json.dumps(dict(seed=seed, status='PASS', account_replays=2*len(inspected)+6)), flush=True)
    raw_probes = {}
    for date in RAW_DAYS:
        day_path = Path(c['data']['market_root'])/date[:7]/date
        day = MonthDay(day_path, verify=True)
        bound.bind(day_path/'manifest.json')
        for part in day.meta['quote_parts']:
            bound.bind(Path(part['root'])/'quotes.parquet', part['quote_file']['sha256'])
        q = QuoteTape.from_day(day)
        raw_probes[date] = {}
        for seed, target in targets_by_seed.items():
            subset = target.loc[target.date.eq(date)].reset_index(drop=True)
            raw_probes[date][str(seed)] = verify_risk_targets_raw(q, subset, c, max_probes=12)
        del q, day
        QUOTE_CACHE.clear()
        gc.collect()
    bound.confirm_unchanged()
    report_data = dict(status='PASS', version='INDEPENDENT_V5_FILTER_ACCOUNT_AND_RAW_RISK_AUDIT_V1',
        runs=results, raw_risk_probes=raw_probes,
        proof='Frozen V2 _projection receives the causal score and explicit fixed side directly; V5 adapter and V4 project_account are not used. Labels only bucket an already fixed ledger.',
        limitations=['CAL alert matching is not matching actual holding-time or account states.',
            'Raw risk probes sample known positive-side paths; all saved risk labels are also exactly rebuilt from factual maps.',
            'Partial closed cash after censor is not a complete economic result.',
            'Event/side correctness and pre-cross safety do not guarantee favorable actual exit after latency.'],
        source_hashes={p:v['sha256'] for p,v in bound.files.items() if Path(p).suffix=='.py'},
        input_hashes={p:v['sha256'] for p,v in bound.files.items() if Path(p).suffix!='.py'})
    path = report/'economic_audit.json'
    path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', output=str(path), sha256=sha(path))), flush=True)
    return report_data


if __name__ == '__main__':
    audit()
