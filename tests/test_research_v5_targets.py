from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spectrum.research_v5.targets import build_filter_targets, verify_risk_targets_raw

NS = 10**9


def config():
    return dict(event_query=dict(family='BB', scales=[1, 3, 6], deadline_seconds=300., reference_fee=.3),
        execution=dict(fee_roundtrip=.3, max_quote_gap_seconds=60.), risk=dict(cost_multipliers=[2, 4, 6]))


def sample(states=(1,), sides=None, long_mae=-1.8, short_mae=-.5):
    roots, maps = [], []
    for i, y in enumerate(states):
        t = (1000+i*600)*NS
        entry_ns = t+100_000_001
        e, l, s = i*10+1, i*10+4, i*10+6
        roots.append(dict(date='2026-08-24', t_ns=t, entry_ordinal=e, entry_ns=entry_ns,
            R_event=1., G_event=2., R_cost2=1., R_cost4=2., R_cost6=3., y_crossing=y,
            available_ns=entry_ns+300*NS if y >= 0 else 0,
            first_qualifying_event_id_L=f'L{i}' if y in (1, 3) else '',
            first_qualifying_event_id_S=f'S{i}' if y in (2, 3) else '',
            first_qualifying_ordinal_L=l if y in (1, 3) else -1,
            first_qualifying_ordinal_S=s if y in (2, 3) else -1,
            first_qualifying_ns_L=entry_ns+NS if y in (1, 3) else 0,
            first_qualifying_ns_S=entry_ns+2*NS if y in (2, 3) else 0))
        for scale in (1, 3, 6):
            for role in (0, 1, 2):
                which = 'L' if (scale, role) == (1, 0) and y in (1, 3) else (
                    'S' if (scale, role) == (1, 2) and y in (2, 3) else '')
                maps.append(dict(date='2026-08-24', t_ns=t, scale=scale, role=role, ready=True,
                    entry_ordinal=e, entry_ns=entry_ns, cross_ordinal=(l if which == 'L' else s) if which else -1,
                    cross_ns=entry_ns+(1 if which == 'L' else 2)*NS if which else 0,
                    event_id=f'{which}{i}' if which else '',
                    mark_L=2.1 if which == 'L' else (-3.1 if which == 'S' else np.nan),
                    mark_S=2.1 if which == 'S' else (-3.1 if which == 'L' else np.nan),
                    mae_L=long_mae if which == 'L' else (-3.1 if which == 'S' else np.nan),
                    mae_S=short_mae if which == 'S' else (-3.1 if which == 'L' else np.nan), R_event=1., G_event=2.))
    return pd.DataFrame(roots), pd.DataFrame(maps), np.asarray(sides if sides is not None else [1]*len(states), np.int8), config()


def test_conditional_hierarchy_unknown_is_not_none_and_wider_risk_is_monotone():
    f, m, sides, c = sample(states=(-1, 0, 1, 2, 3))
    out = build_filter_targets(f, m, sides, c)
    assert out['labels'].dtype == np.float32
    assert out['labels'].tolist() == [[-1]*5, [0, -1, -1, -1, -1], [1, 1, 0, 1, 1],
                                     [1, 0, -1, -1, -1], [1, 1, 0, 1, 1]]
    assert np.array_equal(out['masks'], out['labels'] >= 0)
    assert out['frame'].risk_context_reason.tolist() == ['UNKNOWN_TEACHER', 'NO_EVENT',
        'QUALIFYING_BEFORE_DEADLINE', 'WRONG_SIDE', 'QUALIFYING_BEFORE_DEADLINE']


def test_short_side_and_both_keep_frozen_chosen_direction():
    f, m, sides, c = sample(states=(2, 3, 1), sides=(-1, -1, -1), short_mae=-2.)
    out = build_filter_targets(f, m, sides, c)
    assert out['labels'].tolist() == [[1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 0, -1, -1, -1]]
    assert out['frame'].chosen_side.tolist() == [-1, -1, -1]


def test_adverse_equality_is_unsafe_and_nearby_strictly_inside_is_safe():
    f, m, sides, c = sample(long_mae=-1.)
    assert build_filter_targets(f, m, sides, c)['labels'][0].tolist() == [1, 1, 0, 1, 1]
    m.loc[m.event_id.eq('L0'), 'mae_L'] = np.nextafter(-1., 0.)
    assert build_filter_targets(f, m, sides, c)['labels'][0].tolist() == [1, 1, 1, 1, 1]


def test_exit_censor_or_postcross_loss_cannot_erase_known_path_risk():
    f, m, sides, c = sample(long_mae=-.5)
    original_f, original_m = f.copy(deep=True), m.copy(deep=True)
    a = build_filter_targets(f, m, sides, c)
    f['status'] = 'CENSORED'
    f['reason'] = 'EXIT_TIMEOUT'
    f['mae'] = -999.
    f['net'] = np.nan
    b = build_filter_targets(f, m, sides, c)
    assert np.array_equal(a['labels'], b['labels'])
    pd.testing.assert_frame_equal(f[original_f.columns], original_f)
    pd.testing.assert_frame_equal(m, original_m)


def test_deadline_equal_witness_masks_risk_only_original_event_and_side_unchanged():
    f, m, sides, c = sample()
    due = int(f.entry_ns.iloc[0])+300*NS
    f.loc[0, 'first_qualifying_ns_L'] = due
    m.loc[m.event_id.eq('L0'), 'cross_ns'] = due
    out = build_filter_targets(f, m, sides, c)
    assert out['labels'][0].tolist() == [1, 1, -1, -1, -1]
    assert out['frame'].risk_context_reason.iloc[0] == 'AT_DEADLINE_TIME_WINS'
    assert out['receipt']['exact_deadline_positive_witnesses_masked'] == 1
    assert out['frame'].risk_available_ns.iloc[0] == 0


def test_conditional_label_availability_waits_for_original_common_maturity():
    f, m, sides, c = sample()
    out = build_filter_targets(f, m, sides, c)['frame'].iloc[0]
    assert out.risk_path_available_ns == f.first_qualifying_ns_L.iloc[0]
    assert out.risk_available_ns == f.available_ns.iloc[0]
    assert out.risk_available_ns > out.risk_path_available_ns


@pytest.mark.parametrize('side', [1, -1])
def test_raw_probe_same_ns_earlier_adverse_and_later_exit_latency_loss(side):
    f, m, sides, c = sample(states=(1 if side == 1 else 2,), sides=(side,), long_mae=-1.8, short_mae=-1.8)
    t = int(f.t_ns.iloc[0])
    qt = np.array([t, t+100_000_001, t+200_000_000, t+200_000_000, t+200_000_000, t+300_000_000], np.int64)
    bid = np.rint(np.array([100., 100., 98.7, 99., 102.6, 96.])*NS).astype(np.int64)
    ask = bid+int(.2*NS)
    if side == -1:
        bid, ask = 200*NS-ask, 200*NS-bid
    q = SimpleNamespace(quote_t=qt, bid=bid, ask=ask, valid=np.ones(len(qt), bool), epoch=np.zeros(len(qt), np.int64))
    letter = 'L' if side == 1 else 'S'
    f.loc[0, 'first_qualifying_ordinal_'+letter] = 4
    f.loc[0, 'first_qualifying_ns_'+letter] = qt[4]
    m.loc[m.event_id.eq(letter+'0'), ['cross_ordinal', 'cross_ns']] = [4, qt[4]]
    out = build_filter_targets(f, m, sides, c)
    proof = verify_risk_targets_raw(q, out, c)
    assert proof['probes'] == 1
    hit = proof['checks'][0]['plans']
    assert hit['cost2']['first_adverse_ordinal'] == 2
    assert hit['cost2']['first_adverse_ns'] == str(qt[4])
    assert hit['cost4']['safe'] and hit['cost6']['safe']
    # The later loss lies outside the first-cross path and must not alter S.
    assert out['labels'][0, 2:].tolist() == [0, 1, 1]


def test_initial_entry_cost_itself_can_be_adverse_first_hit():
    f, m, sides, c = sample(long_mae=-.5)
    f[['R_cost2', 'R_cost4', 'R_cost6']] = [.5, 1., 2.]
    out = build_filter_targets(f, m, sides, c)
    assert out['labels'][0].tolist() == [1, 1, 0, 1, 1]


@pytest.mark.parametrize('change', ['missing_map', 'duplicate_map', 'witness_id', 'ordinal', 'side', 'fee', 'reference', 'positive_mae'])
def test_malformed_teacher_contract_fails_closed(change):
    f, m, sides, c = sample()
    if change == 'missing_map':
        m = m.iloc[:-1]
    elif change == 'duplicate_map':
        m.iloc[1] = m.iloc[0]
    elif change == 'witness_id':
        f.loc[0, 'first_qualifying_event_id_L'] = 'changed'
    elif change == 'ordinal':
        m.loc[m.event_id.eq('L0'), 'cross_ordinal'] = f.entry_ordinal.iloc[0]
    elif change == 'side':
        sides[:] = 0
    elif change == 'fee':
        c['execution']['fee_roundtrip'] = .4
    elif change == 'reference':
        m.loc[0, 'G_event'] = 3.
    else:
        m.loc[m.event_id.eq('L0'), 'mae_L'] = .1
    with pytest.raises(ValueError):
        build_filter_targets(f, m, sides, c)
