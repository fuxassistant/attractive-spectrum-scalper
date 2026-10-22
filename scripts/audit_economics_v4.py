"""Independent CAL policy replay and preregistered TEST economic stress audit.

No production evaluation/account function is imported. Labels are attached only
after chronological decisions. Censoring halts the account and makes total PnL
unknown; preceding closed cash remains a partial observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.verify_opportunity_v3 import BoundInputs, read_json, sha

NS = 1_000_000_000
STRESS_THRESHOLDS = (.35, .50)
PLANS = ('cost2', 'cost4', 'cost6')
BUCKETS = ('UNKNOWN_TEACHER', 'TEACHER_NONE', 'WRONG_SIDE', 'TARGET_POSITIVE_SIDE')
EXPECTED_RUNS = tuple(f'tcn_micro_restored_{weighting}_seed{seed}'
                      for seed in (17, 29) for weighting in ('uniform', 'soft_crossing_id'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_account(frame, executions, config):
    """Align every causal clock and both execution branches; no label filtering."""
    if not len(frame) or frame.feature_available.dtype != bool:
        raise ValueError('Nonempty origins with boolean feature admission required')
    t = frame.t_ns.to_numpy(np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Unique chronological origin clock required')
    plans = [f'cost{float(m):g}' for m in config['risk']['cost_multipliers']]
    if plans != list(PLANS):
        raise ValueError('The fixed cost2/4/6 risk grid is required')
    date = frame.date.astype(str).to_numpy()
    if not np.isin(executions.plan, plans).all() or len(executions) != len(frame)*6:
        raise ValueError('Every risk plan requires both sides of every origin')
    arrays = {}
    for plan in plans:
        f = executions.loc[executions.plan.eq(plan)].sort_values(['t_ns', 'side'], ascending=[True, False])
        if (not np.array_equal(f.t_ns.to_numpy(), np.repeat(t, 2))
                or not np.array_equal(f.side.to_numpy(), np.tile([1, -1], len(t)))
                or not np.array_equal(f.date.astype(str).to_numpy(), np.repeat(date, 2))):
            raise ValueError('Missing/duplicate/misaligned execution branch')
        if not f.status.isin(['INELIGIBLE', 'EMPTY', 'CENSORED', 'COMPLETE']).all():
            raise ValueError('Unknown execution status')
        complete = f.status.eq('COMPLETE').to_numpy()
        numbers = f[['net', 'mae', 'mfe', 'R_trade', 'R_event', 'G_event']].to_numpy(float)
        if not np.isfinite(numbers[complete]).all():
            raise ValueError('Nonfinite completed economics')
        if (not f.loc[complete, 'reason'].isin(['TAKE', 'STOP', 'TIME']).all()
                or (f.entry_ns.to_numpy()[complete] <= f.t_ns.to_numpy()[complete]).any()
                or (f.exit_ns.to_numpy()[complete] <= f.entry_ns.to_numpy()[complete]).any()
                or (f.mae.to_numpy()[complete] > f.net.to_numpy()[complete]+1e-9).any()
                or (f.mfe.to_numpy()[complete] < f.net.to_numpy()[complete]-1e-9).any()):
            raise ValueError('Invalid completed execution clock/excursion')
        for name in ('R_event', 'G_event'):
            if not np.allclose(f[name], np.repeat(frame[name], 2), rtol=0, atol=1e-9, equal_nan=True):
                raise ValueError('Risk plan changed the event reference')
        if 'R_'+plan in frame and not np.allclose(f.R_trade, np.repeat(frame['R_'+plan], 2), rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError('Execution trade risk differs from origin risk')
        arrays[plan] = {name: f[name].to_numpy().reshape(len(t), 2) for name in
            ('status', 'reason', 'net', 'mae', 'mfe', 'entry_ns', 'exit_ns', 'R_trade', 'R_event', 'G_event')}
    # Deliberately exclude y_crossing/maturity/event IDs from account input.
    return dict(t=t, date=date, available=frame.feature_available.to_numpy(), arrays=arrays, plans=plans)


def prepare_probabilities(p, n):
    p = np.asarray(p, float)
    if (p.shape != (n, 4) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Finite normalized probabilities required')
    long, short = p[:, 1]+p[:, 3], p[:, 2]+p[:, 3]
    return np.maximum(long, short), np.where(short > long, -1, 1)


def project_account(book, p, plan, threshold, days, pending_seconds):
    """Independent one-position loop, retaining exact production output schema."""
    score, side = prepare_probabilities(p, len(book['t']))
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(pending_seconds) or pending_seconds <= 0:
        raise ValueError('Valid threshold and pending duration required')
    a = book['arrays'][plan]
    allowed = book['available'] & np.isin(book['date'], days)
    candidates = np.flatnonzero(allowed & (score >= threshold))
    daily = {str(d): dict(trades=0, net=0., ineligible=0, empty=0, censored=0) for d in days}
    counts = dict(trades=0, ineligible=0, empty=0, censored=0, skipped_busy=0)
    reasons = dict(TAKE=0, STOP=0, TIME=0)
    cash = peak = drawdown = 0.
    busy, wins, processed = -1, 0, 0
    halt, ledger = None, []
    for i in candidates:
        processed += 1
        now, date = int(book['t'][i]), str(book['date'][i])
        if now <= busy:
            counts['skipped_busy'] += 1
            continue
        si = 0 if side[i] == 1 else 1
        key = i, si
        status, reason = str(a['status'][key]), str(a['reason'][key])
        if status == 'INELIGIBLE':
            counts['ineligible'] += 1
            daily[date]['ineligible'] += 1
            continue
        item = dict(date=date, t_ns=str(now), side=int(side[i]), probability=float(score[i]), status=status, reason=reason)
        if status == 'EMPTY':
            counts['empty'] += 1
            daily[date]['empty'] += 1
            busy = now+int(round(pending_seconds*NS))
            item.update(busy_until_ns=str(busy), net=None, closed_cash=cash)
        elif status == 'CENSORED':
            counts['censored'] += 1
            daily[date]['censored'] += 1
            halt = dict(date=date, t_ns=str(now), side=int(side[i]), reason=reason)
            item.update(net=None, closed_cash=cash)
            ledger.append(item)
            break
        else:
            net = float(a['net'][key])
            cash += net
            peak = max(peak, cash)
            drawdown = max(drawdown, peak-cash)
            busy = int(a['exit_ns'][key])
            counts['trades'] += 1
            daily[date]['trades'] += 1
            daily[date]['net'] += net
            wins += net > 0
            reasons[reason] += 1
            item.update(entry_ns=str(int(a['entry_ns'][key])), exit_ns=str(busy), net=net,
                mae=float(a['mae'][key]), mfe=float(a['mfe'][key]), R_trade=float(a['R_trade'][key]),
                R_event=float(a['R_event'][key]), G_event=float(a['G_event'][key]), closed_cash=cash,
                closed_drawdown=peak-cash)
        ledger.append(item)
    summary = dict(plan=plan, threshold=float(threshold), completed=halt is None, net=cash,
        closed_max_drawdown=drawdown, wins=int(wins), win_rate=float(wins/counts['trades']) if counts['trades'] else None,
        origin_candidates=len(candidates), causal_feature_origins=int(allowed.sum()),
        unprocessed_after_censor=len(candidates)-processed, halt=halt, exit_reasons=reasons, daily=daily, **counts)
    return summary, ledger


def select_from_calibration(book, p, config):
    policy, pending = config['policy'], config['execution']['pending_timeout_seconds']
    if not policy['one_position'] or policy['minimum_calibration_trades'] < 1:
        raise ValueError('One-position policy and positive minimum trades required')
    grid = []
    for plan in book['plans']:
        for threshold in policy['threshold_candidates']:
            summary, _ = project_account(book, p, plan, threshold, config['splits']['calibration'], pending)
            grid.append(summary)
    candidates = [x for x in grid if x['completed'] and x['trades'] >= policy['minimum_calibration_trades'] and x['net'] > 0]
    if candidates:
        chosen = sorted(candidates, key=lambda x: (-x['net'], -x['threshold'], float(x['plan'][4:])))[0]
        selected = dict(plan=chosen['plan'], threshold=chosen['threshold'], reason='positive_complete_calibration')
    else:
        selected = dict(plan='cost2', threshold=1.01, reason='WAIT_no_positive_complete_calibration_with_minimum_trades')
    return selected, grid


def teacher_bucket(y, side):
    if y < 0:
        return 'UNKNOWN_TEACHER'
    if y == 0:
        return 'TEACHER_NONE'
    return 'TARGET_POSITIVE_SIDE' if (y & (1 if side == 1 else 2)) else 'WRONG_SIDE'


def teacher_buckets(frame, ledger):
    """Attach future teacher labels only AFTER the trading path is fixed."""
    labels = dict(zip(frame.t_ns.astype(np.int64), frame.y_crossing.astype(int)))
    rows = []
    for item in ledger:
        y = labels[int(item['t_ns'])]
        rows.append(dict(**item, y_crossing=y, teacher_bucket=teacher_bucket(y, item['side'])))
    output = {}
    for bucket in BUCKETS:
        attempted = [r for r in rows if r['teacher_bucket'] == bucket]
        complete = [r for r in attempted if r['status'] == 'COMPLETE']
        net = np.array([r['net'] for r in complete], float)
        output[bucket] = dict(attempts=len(attempted), complete_trades=len(complete),
            empty_attempts=sum(r['status'] == 'EMPTY' for r in attempted),
            censored_attempts=sum(r['status'] == 'CENSORED' for r in attempted),
            observed_closed_net=float(net.sum()), mean_net=float(net.mean()) if len(net) else None,
            median_net=float(np.median(net)) if len(net) else None,
            wins=int((net > 0).sum()), losses=int((net < 0).sum()), zeros=int((net == 0).sum()),
            win_rate=float((net > 0).mean()) if len(net) else None,
            gross_positive_net=float(net[net > 0].sum()), gross_negative_net=float(net[net < 0].sum()),
            exit_reasons={reason: sum(r['reason'] == reason for r in complete) for reason in ('TAKE', 'STOP', 'TIME')},
            joint_teacher_classes={str(y): sum(r['y_crossing'] == y for r in complete) for y in (-1, 0, 1, 2, 3)})
    return output


def safe_projection(frame, summary, ledger):
    """Expose partial cash without presenting post-censor zeroes as observations."""
    result = {k: v for k, v in summary.items() if k not in ('net', 'daily')}
    result['status'] = 'COMPLETE' if summary['completed'] else 'INCOMPLETE_CENSORED'
    result['observed_closed_cash'] = summary['net']
    result['total_net_if_complete'] = summary['net'] if summary['completed'] else None
    result['daily'] = {}
    halt_date = summary['halt']['date'] if summary['halt'] else None
    cumulative = 0.
    for date, day in summary['daily'].items():
        observed = halt_date is None or date <= halt_date
        full = halt_date is None or date < halt_date
        if observed:
            cumulative += day['net']
        result['daily'][date] = dict(status='COMPLETE' if full else ('PARTIAL' if observed else 'NOT_OBSERVED'),
            observed_closed_cash=day['net'] if observed else None,
            cumulative_closed_cash=cumulative if observed else None,
            total_net_if_complete=day['net'] if full else None,
            **{k: v if observed else None for k, v in day.items() if k != 'net'})
    result['teacher_buckets_after_execution'] = teacher_buckets(frame, ledger)
    result['ledger_rows'] = len(ledger)
    result['ledger_sha256'] = canonical_sha(ledger)
    result['ledger_first_three'] = ledger[:3]
    result['ledger_last_three'] = ledger[-3:]
    completed = sum(x['complete_trades'] for x in result['teacher_buckets_after_execution'].values())
    money = sum(x['observed_closed_net'] for x in result['teacher_buckets_after_execution'].values())
    if completed != summary['trades'] or not np.isclose(money, summary['net'], rtol=0, atol=1e-8):
        raise AssertionError('Teacher bucket accounting does not reconcile')
    return result


def audit(config_path, runs_root, report_root):
    config = read_json(config_path)
    runs_root, report_root = Path(runs_root), Path(report_root)
    suite_path = report_root/'suite_summary.json'
    suite = read_json(suite_path)
    if suite.get('status') != 'COMPLETE' or {r['run_id'] for r in suite['runs']} != set(EXPECTED_RUNS):
        raise RuntimeError('Wait for the COMPLETE four-arm V4 suite before economic audit')
    baseline = read_json('configs/research/opportunity_pipeline_v3.json')
    if config['experiment']['false_signal_protocol']['fixed_test_execution_stress']['thresholds'] != list(STRESS_THRESHOLDS):
        raise ValueError('Economic stress thresholds differ from the fixed protocol')
    for name in ('event_query', 'execution', 'risk', 'loss', 'policy', 'splits'):
        if config[name] != baseline[name]:
            raise ValueError('V4 changed frozen economic recipe: '+name)
    if config['data']['root'] != baseline['data']['root']:
        raise ValueError('V4 must reuse the frozen V3 dataset')
    bound = BoundInputs()
    for path in (__file__, 'scripts/verify_opportunity_v3.py', config_path, suite_path, 'configs/research/opportunity_pipeline_v3.json'):
        bound.bind(path)
    dataset = Path(config['data']['root'])
    manifest_path = dataset/'manifest.json'
    bound.bind(manifest_path, suite['dataset_manifest_sha256'])
    manifest = read_json(manifest_path)
    if manifest['status'] != 'COMPLETE':
        raise ValueError('Frozen dataset is incomplete')
    for path, digest in manifest['identity']['sources'].items():
        bound.bind(path, digest)
    days = config['splits']['calibration']+config['splits']['test']
    frames, executions = [], []
    for day in manifest['days']:
        if day['date'] not in days:
            continue
        for name, collector in (('roots', frames), ('execution', executions)):
            path = dataset/f"{day['date']}_{name}.parquet"
            bound.bind(path, day['files'][path.name])
            collector.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True).sort_values('t_ns').reset_index(drop=True)
    executions = pd.concat(executions, ignore_index=True)
    if set(frame.date) != set(days):
        raise ValueError('Missing CAL/TEST dates')
    book = prepare_account(frame, executions, config)
    results = {}
    for run_id in EXPECTED_RUNS:
        directory = runs_root/run_id
        receipt = read_json(directory/'receipt.json')
        if receipt.get('status') != 'COMPLETE' or receipt['dataset_manifest_sha256'] != suite['dataset_manifest_sha256']:
            raise ValueError('Incomplete or mismatched V4 run')
        bound.bind(directory/'receipt.json')
        for name in ('config.json', 'predictions.parquet', 'economic_evaluation.json'):
            bound.bind(directory/name, receipt['outputs'][name])
        for path, digest in receipt['sources'].items():
            bound.bind(path, digest)
        c = read_json(directory/'config.json')
        if c['run_id'] != run_id or c['data']['context_mode'] != 'restored' or c['data']['root'] != config['data']['root']:
            raise ValueError('Run identity/context/dataset differs')
        for key in ('event_query', 'execution', 'risk', 'loss', 'policy', 'splits'):
            if c[key] != config[key]:
                raise ValueError('Unmatched run economic recipe')
        prediction = pd.read_parquet(directory/'predictions.parquet', columns=[
            'date', 't_ns', 'feature_available', 'y_crossing', 'p_none', 'p_long', 'p_short', 'p_both'])
        prediction = prediction.loc[prediction.date.isin(days)].reset_index(drop=True)
        for name in ('date', 't_ns', 'feature_available', 'y_crossing'):
            if not np.array_equal(prediction[name], frame[name]):
                raise ValueError('Prediction/root identity differs: '+name)
        p = prediction[['p_none', 'p_long', 'p_short', 'p_both']].to_numpy()
        saved = read_json(directory/'economic_evaluation.json')
        selected, grid = select_from_calibration(book, p, config)
        if selected != saved['summary']['selected'] or grid != saved['calibration_grid']:
            raise AssertionError('Independent CAL grid/selection differs from saved evaluation')
        selected_replay = {}
        for split in ('calibration', 'test'):
            summary, ledger = project_account(book, p, selected['plan'], selected['threshold'],
                config['splits'][split], config['execution']['pending_timeout_seconds'])
            if summary != saved['summary'][split] or ledger != saved['ledgers'][split]:
                raise AssertionError('Independent selected replay differs: '+split)
            selected_replay[split] = safe_projection(frame, summary, ledger)
        stress = []
        for threshold in STRESS_THRESHOLDS:
            for plan in PLANS:
                summary, ledger = project_account(book, p, plan, threshold, config['splits']['test'],
                                                  config['execution']['pending_timeout_seconds'])
                stress.append(safe_projection(frame, summary, ledger))
        results[run_id] = dict(status='PASS', selected_from_calibration=selected,
            exact_calibration_grid_parity=True, calibration_candidates=len(grid),
            exact_selected_calibration_and_test_summary_and_ledger_parity=True,
            selected_replay=selected_replay, preregistered_fixed_threshold_test_stress=stress,
            prediction_sha256=receipt['outputs']['predictions.parquet'],
            saved_economic_evaluation_sha256=receipt['outputs']['economic_evaluation.json'])
        print(json.dumps(dict(run=run_id, status='PASS', selected=selected)), flush=True)
    bound.confirm_unchanged()
    result = dict(status='PASS', version='INDEPENDENT_V4_CAUSAL_ECONOMIC_AUDIT_V1', runs=results,
        fixed_stress=dict(thresholds=list(STRESS_THRESHOLDS), risk_plans=list(PLANS),
            scope='Identical preregistered descriptive stress projections for every arm/seed. No TEST threshold, risk, arm, or seed winner selection.',
            not_deployable_selection=True),
        definitions=dict(admission='Full clock, causal feature_available and predicted side marginal only. Future y/maturity/execution completeness never gates an attempt.',
            teacher_buckets='Attach y only after actual one-position replay: NONE, wrong side of known positive event, target-positive chosen side (includes BOTH), or UNKNOWN. Count completed filled trades separately from empty/censored attempts.',
            positive_teacher='Geometric crossing label is not executed profitability: STOP may precede a qualifying crossing and latency/costs change realized cash.',
            cash='USD per reference ounce under saved taker BBO execution. CENSORED halts the entire path; observed_closed_cash is a partial prefix and total_net_if_complete=null.',
            drawdown='Closed-trade drawdown only; not intratrade mark-to-market or position sizing simulation.',
            pending='EMPTY blocks until intent+2s inclusive; COMPLETE blocks through exit timestamp inclusive; account state persists across dates.',
            small_buckets='All bucket counts and observed money are shown. Empty buckets have null means/rates. No significance or economic readiness is inferred from sparse buckets.'),
        source_hashes={p: v['sha256'] for p, v in bound.files.items() if Path(p).suffix == '.py'},
        input_hashes={p: v['sha256'] for p, v in bound.files.items() if Path(p).suffix != '.py'},
        verified_files=bound.files, no_frozen_source_changes=True)
    out = report_root/'economic_audit.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', output=str(out), sha256=sha(out))), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/research/opportunity_pipeline_v4.json')
    p.add_argument('--runs-root', default='data/derived/scientific_pipeline_v4/runs')
    p.add_argument('--report-root', default='reports/scientific_pipeline_v4')
    a = p.parse_args()
    audit(a.config, a.runs_root, a.report_root)


if __name__ == '__main__':
    main()
