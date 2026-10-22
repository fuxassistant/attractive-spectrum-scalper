import numpy as np
import pandas as pd
import pytest

from scripts.audit_economics_v4 import prepare_account
from scripts.audit_economics_v5 import (independent_retention, reference_replay,
    risk_economics, scalar_hash_scores, verify_saved_projection)
from scripts.evaluate_opportunity_v5 import hash_scores, replay, retention_threshold
from spectrum.research_v2.evaluation import _aligned

NS = 10**9


def sample():
    dates = ['2026-08-17']*3+['2026-08-24']*4
    times = np.r_[pd.Timestamp('2026-08-17T08:00Z').value+np.arange(3)*5*NS,
                  pd.Timestamp('2026-08-24T08:00Z').value+np.arange(4)*5*NS]
    frame = pd.DataFrame(dict(date=dates, t_ns=times, feature_available=True,
        y_crossing=np.array([0, 1, 2, 0, 1, 3, -1], np.int8), R_event=1., G_event=2.))
    rows = []
    for plan in ('cost2', 'cost4', 'cost6'):
        for date, t in zip(dates, times):
            for side in (1, -1):
                rows.append(dict(date=date, t_ns=t, plan=plan, side=side, status='COMPLETE',
                    reason='TIME', entry_ns=t+100_000_001, exit_ns=t+NS,
                    net=1. if side == 1 else -1., mae=-1., mfe=1., R_trade=float(plan[4:]), R_event=1., G_event=2.))
    config = dict(risk=dict(cost_multipliers=[2, 4, 6]), execution=dict(pending_timeout_seconds=2., fee_roundtrip=.3),
        event_query=dict(gain_multiple=2.), splits=dict(calibration=['2026-08-17'], test=['2026-08-24']))
    p = np.tile([.1, .2, .6, .1], (len(frame), 1))
    return frame, pd.DataFrame(rows), p, config


def test_scalar_hash_matches_vector_implementation_and_preserves_prefix():
    t = np.array([0, 1, 100, 1780297200000000000, 9223372036854775806], np.int64)
    for seed in (17, 29):
        actual = scalar_hash_scores(t, seed)
        assert np.array_equal(actual, hash_scores(t, seed))
        assert np.array_equal(actual[:3], scalar_hash_scores(t[:3], seed))
        assert ((actual > 0) & (actual < 1)).all()


@pytest.mark.parametrize('fraction', [.1, .25, .5, .75, 1.])
def test_cal_count_budget_and_ties_match_independent_order_statistic(fraction):
    scores = np.array([0., .2, .2, .5, .5, .5, .9, 1., .999])
    cal = np.array([True]*8+[False])
    a = independent_retention(scores, cal, fraction)
    assert a == retention_threshold(scores, cal, fraction)
    assert a['actual'] <= a['allowed']
    changed = scores.copy(); changed[-1] = 0.
    assert a == independent_retention(changed, cal, fraction)


def test_empty_cal_or_zero_budget_does_not_use_test_values():
    assert independent_retention([.9], [False], .5)['threshold'] == 1.01
    q = independent_retention([.9], [True], .5)
    assert q['allowed'] == q['actual'] == 0 and q['threshold'] > .9


@pytest.mark.parametrize('threshold', [0., np.nextafter(0., 1.), .5])
def test_zero_score_short_adapter_matches_explicit_frozen_side_account(threshold):
    f, e, p, c = sample()
    score = np.array([0., .5, .6, 0., np.nextafter(0., 1.), .5, .8])
    side = -np.ones(len(f), np.int8)
    candidate = np.ones(len(f), bool)
    aligned = _aligned(f, e, p, c)
    summary, ledger = reference_replay(aligned, score, side, candidate, 'cost2', threshold, c['splits']['test'], 2.)
    saved, actual_ledger = replay(prepare_account(f, e, c), f, score, side, 'cost2', threshold, c['splits']['test'], 2.)
    assert actual_ledger == ledger
    assert all(x['side'] == -1 for x in ledger)
    verify_saved_projection(saved, summary, ledger)
    if threshold == 0:
        assert len(ledger) == 4 and ledger[0]['probability'] == 0.


def test_candidate_gate_and_future_teacher_changes_never_change_replay():
    f, e, p, c = sample()
    score, side = np.ones(len(f)), np.ones(len(f), np.int8)
    candidate = np.array([True, True, True, False, True, True, True])
    a = reference_replay(_aligned(f, e, p, c), score, side, candidate, 'cost2', .5, c['splits']['test'], 2.)
    f.y_crossing = -1
    b = reference_replay(_aligned(f, e, p, c), score, side, candidate, 'cost2', .5, c['splits']['test'], 2.)
    assert a == b and a[0]['trades'] == 3


def test_censor_and_unknown_teacher_halt_before_unobserved_future_money():
    f, e, p, c = sample()
    f.loc[6, 'date'] = '2026-08-25'; c['splits']['test'].append('2026-08-25')
    e.loc[e.t_ns.eq(f.t_ns.iloc[6]), 'date'] = '2026-08-25'
    f.loc[4, 'y_crossing'] = -1
    bad = e.t_ns.eq(f.t_ns.iloc[4]) & e.side.eq(1)
    e.loc[bad, ['status', 'reason', 'net', 'mae', 'mfe', 'exit_ns']] = ['CENSORED', 'EXIT_TIMEOUT', np.nan, np.nan, np.nan, 0]
    score, side = np.ones(len(f)), np.ones(len(f), np.int8)
    summary, ledger = reference_replay(_aligned(f, e, p, c), score, side, np.ones(len(f), bool), 'cost2', 0., c['splits']['test'], 2.)
    saved, actual = replay(prepare_account(f, e, c), f, score, side, 'cost2', 0., c['splits']['test'], 2.)
    assert actual == ledger
    verify_saved_projection(saved, summary, ledger)
    assert saved['observed_closed_cash'] == 1.
    assert saved['total_net_if_complete'] is None
    assert saved['daily']['2026-08-25']['status'] == 'NOT_OBSERVED'


def test_risk_buckets_are_after_execution_and_do_not_define_money():
    f, e, p, c = sample()
    f.loc[3:, 'y_crossing'] = np.array([0, 2, 1, 3], np.int8)
    target = pd.DataFrame({'y_risk_safe_cost2': [-1, 1, -1, -1, -1, 0, 1]})
    summary, ledger = reference_replay(_aligned(f, e, p, c), np.ones(len(f)), np.ones(len(f), np.int8),
        np.ones(len(f), bool), 'cost2', 0., c['splits']['test'], 2.)
    result = risk_economics(f, target, 'cost2', summary, ledger)
    assert result['buckets']['NONE']['completed_trades'] == 1
    assert result['buckets']['WRONG_SIDE']['completed_trades'] == 1
    assert result['buckets']['CORRECT_RISK_FIRST']['observed_closed_net'] == 1.
    assert result['buckets']['CORRECT_SAFE']['observed_closed_net'] == 1.
    assert sum(x['observed_closed_net'] for x in result['buckets'].values()) == 4.
    assert result['completed_trade_holding_seconds'] == pytest.approx(4*.899999999)
