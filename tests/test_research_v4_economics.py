import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_economics_v4 import (PLANS, STRESS_THRESHOLDS, audit, prepare_account,
    project_account, safe_projection, select_from_calibration, teacher_buckets)
from spectrum.research_v2.evaluation import evaluate_risk_grid

NS = 10**9


def sample(ncal=4, ntest=5, spacing=10):
    dates = ['2026-06-01']*ncal+['2026-07-13']*ntest
    t = np.r_[pd.Timestamp('2026-06-01T08:00Z').value+np.arange(ncal)*spacing*NS,
              pd.Timestamp('2026-07-13T08:00Z').value+np.arange(ntest)*spacing*NS]
    frame = pd.DataFrame(dict(date=dates, t_ns=t, feature_available=True,
        y_crossing=np.resize([0, 2, 1, 3, -1], len(t)).astype(np.int8), R_event=1., G_event=2.))
    rows = []
    for plan in PLANS:
        for date, now in zip(dates, t):
            for side in (1, -1):
                net = 1. if side == 1 else -.5
                rows.append(dict(date=date, t_ns=now, plan=plan, side=side, status='COMPLETE',
                    reason='TAKE' if side == 1 else 'STOP', net=net, mae=-.5, mfe=1.,
                    entry_ns=now+200_000_000, exit_ns=now+NS, R_trade=float(plan[4:]), R_event=1., G_event=2.))
    executions = pd.DataFrame(rows)
    p = np.tile([.1, .9, 0., 0.], (len(t), 1))
    config = dict(risk=dict(cost_multipliers=[2, 4, 6]), event_query=dict(gain_multiple=2.),
        execution=dict(pending_timeout_seconds=2., fee_roundtrip=.3),
        splits=dict(calibration=['2026-06-01'], test=['2026-07-13']),
        policy=dict(one_position=True, threshold_candidates=[.35, .5, .65, .8, 1.01], minimum_calibration_trades=1))
    return frame, executions, p, config


def independent(f, e, p, c):
    book = prepare_account(f, e, c)
    selected, grid = select_from_calibration(book, p, c)
    replay = {split: project_account(book, p, selected['plan'], selected['threshold'], dates, 2.)
              for split, dates in c['splits'].items()}
    return selected, grid, replay


def test_independent_selection_full_grid_and_ledgers_exact_production_parity():
    f, e, p, c = sample()
    selected, grid, replay = independent(f, e.sample(frac=1, random_state=7), p, c)
    saved = evaluate_risk_grid(f, e, p, c)
    assert selected == saved['summary']['selected']
    assert grid == saved['calibration_grid']
    for split, (summary, ledger) in replay.items():
        assert summary == saved['summary'][split]
        assert ledger == saved['ledgers'][split]
        json.dumps(safe_projection(f, summary, ledger), allow_nan=False)


def test_future_label_changes_do_not_change_attempts_or_money():
    f, e, p, c = sample()
    original = independent(f, e, p, c)
    f['y_crossing'] = -1
    f['maturity_ns'] = f.t_ns+900*NS
    mutated = independent(f, e, p, c)
    assert original == mutated
    summary, ledger = mutated[2]['test']
    buckets = teacher_buckets(f, ledger)
    assert buckets['UNKNOWN_TEACHER']['complete_trades'] == summary['trades']


def test_unknown_censor_halts_entire_account_and_future_days_are_unobserved():
    f, e, p, c = sample(ncal=1, ntest=4)
    f.loc[4, 'date'] = '2026-07-14'
    e.loc[e.t_ns.eq(f.t_ns.iloc[4]), 'date'] = '2026-07-14'
    c['splits']['test'].append('2026-07-14')
    f.loc[2, 'y_crossing'] = -1
    mask = e.t_ns.eq(f.t_ns.iloc[2]) & e.side.eq(1)
    e.loc[mask, ['status', 'reason', 'net', 'mae', 'mfe', 'exit_ns']] = ['CENSORED', 'SUPPORT', np.nan, np.nan, np.nan, 0]
    _, _, replay = independent(f, e, p, c)
    summary, ledger = replay['test']
    safe = safe_projection(f, summary, ledger)
    assert summary['trades'] == 1 and summary['censored'] == 1
    assert safe['observed_closed_cash'] == 1.
    assert safe['total_net_if_complete'] is None
    assert safe['daily']['2026-07-13']['status'] == 'PARTIAL'
    assert safe['daily']['2026-07-14']['status'] == 'NOT_OBSERVED'
    assert safe['daily']['2026-07-14']['observed_closed_cash'] is None
    assert safe['daily']['2026-07-14']['cumulative_closed_cash'] is None
    assert safe['teacher_buckets_after_execution']['UNKNOWN_TEACHER']['censored_attempts'] == 1
    assert summary == evaluate_risk_grid(f, e, p, c)['summary']['test']


def test_empty_pending_and_exit_clock_are_inclusive_no_concurrent_position():
    f, e, p, c = sample(ncal=1, ntest=7, spacing=1)
    mask = e.t_ns.eq(f.t_ns.iloc[1]) & e.side.eq(1)
    e.loc[mask, ['status', 'reason', 'net', 'mae', 'mfe', 'entry_ns', 'exit_ns']] = ['EMPTY', 'ENTRY_TIMEOUT', np.nan, np.nan, np.nan, 0, 0]
    _, _, replay = independent(f, e, p, c)
    summary, ledger = replay['test']
    assert summary['empty'] == 1 and summary['trades'] == 2 and summary['skipped_busy'] == 4
    assert [r['t_ns'] for r in ledger] == [str(f.t_ns.iloc[i]) for i in (1, 4, 6)]


def test_cal_selection_ignores_changed_test_outcomes_and_scores():
    f, e, p, c = sample()
    a = independent(f, e, p, c)
    e.loc[e.date.eq('2026-07-13'), ['net', 'mfe']] = 10000.
    p[4:] = [0., 0., 1., 0.]
    b = independent(f, e, p, c)
    assert a[:2] == b[:2]
    assert a[2]['test'] != b[2]['test']


def test_teacher_none_can_win_positive_side_can_lose_and_wrong_side_is_separate():
    f, e, p, c = sample(ncal=1, ntest=5)
    f.loc[1:, 'y_crossing'] = np.array([0, 2, 1, 3, -1], np.int8)
    mask = e.t_ns.eq(f.t_ns.iloc[3]) & e.side.eq(1)
    e.loc[mask, ['net', 'mae', 'reason']] = [-2., -2., 'STOP']
    summary, ledger = independent(f, e, p, c)[2]['test']
    buckets = teacher_buckets(f, ledger)
    assert buckets['TEACHER_NONE']['observed_closed_net'] == 1.
    assert buckets['WRONG_SIDE']['complete_trades'] == 1
    assert buckets['TARGET_POSITIVE_SIDE']['losses'] == 1
    assert buckets['TARGET_POSITIVE_SIDE']['complete_trades'] == 2
    assert buckets['UNKNOWN_TEACHER']['complete_trades'] == 1
    assert sum(x['observed_closed_net'] for x in buckets.values()) == summary['net']


def test_fixed_stress_grid_and_both_marginal_long_tie():
    f, e, p, c = sample()
    p[:] = [0., .125, .125, .75]
    book = prepare_account(f, e, c)
    for plan in PLANS:
        for threshold in STRESS_THRESHOLDS:
            summary, ledger = project_account(book, p, plan, threshold, c['splits']['test'], 2.)
            assert summary['trades'] == 5
            assert {r['side'] for r in ledger} == {1}


def test_unavailable_features_skip_without_future_filter_and_ineligible_is_not_empty():
    f, e, p, c = sample(ncal=1, ntest=3)
    f.loc[1, 'feature_available'] = False
    mask = e.t_ns.eq(f.t_ns.iloc[2]) & e.side.eq(1)
    e.loc[mask, 'status'] = 'INELIGIBLE'
    summary, ledger = independent(f, e, p, c)[2]['test']
    assert summary['trades'] == 1 and summary['ineligible'] == 1 and summary['empty'] == 0
    assert len(ledger) == 1


def test_no_positive_complete_calibration_falls_back_to_wait():
    f, e, p, c = sample()
    c['policy']['minimum_calibration_trades'] = 99
    selected, _, replay = independent(f, e, p, c)
    assert selected['threshold'] == 1.01 and selected['plan'] == 'cost2'
    assert replay['test'][0]['trades'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nonfinite', 'changed_gain'])
def test_bad_execution_inputs_fail_closed(mutation):
    f, e, p, c = sample()
    if mutation == 'missing':
        e = e.iloc[:-1]
    elif mutation == 'duplicate':
        e.iloc[1] = e.iloc[0]
    elif mutation == 'nonfinite':
        e.loc[1, 'net'] = np.nan
    else:
        e.loc[1, 'G_event'] = 3.
    with pytest.raises(ValueError):
        prepare_account(f, e, c)


def test_audit_refuses_partial_suite_before_loading_results(monkeypatch):
    reads = []
    def fake_read(path):
        reads.append(str(path))
        if len(reads) == 1:
            return {}
        if len(reads) == 2:
            return dict(status='RUNNING', runs=[])
        raise AssertionError('Partial suite must not read any result payload')
    monkeypatch.setattr('scripts.audit_economics_v4.read_json', fake_read)
    with pytest.raises(RuntimeError, match='COMPLETE four-arm'):
        audit('config.json', 'missing_runs', 'missing_reports')
    assert len(reads) == 2
