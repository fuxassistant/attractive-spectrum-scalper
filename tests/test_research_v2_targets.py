from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
from spectrum.cognition.execution import QuoteTape
from spectrum.research_v2.targets import build_targets
from spectrum.research_v2.simulation import NS
from scripts.crossing_map_analyze import target_frame


def synthetic(constant=False, future_shift=None):
    seconds = np.arange(0, 11_401, dtype=np.int64)
    prices = np.full(len(seconds), 100.) if constant else 100+2.8*np.sin(seconds/41)+.0001*seconds
    if future_shift is not None:
        at, change = future_shift
        prices[seconds > at] += change
    q = QuoteTape(seconds*NS, np.rint(prices*1e9).astype(np.int64), np.rint((prices+.2)*1e9).astype(np.int64),
        np.ones(len(seconds), bool), np.zeros(len(seconds), np.int64), 11_402*NS,
        7_200*NS, 11_000*NS, 11_000*NS)
    day = SimpleNamespace(meta={'date': '2026-03-30', 'quote_parts': [
        {'quote_file': {'sha256': 'synthetic_fixed_source'}, 'start_ns': 0, 'end_ns': 11_402*NS}]})
    return day, q


def cfg():
    return dict(event_query=dict(family='BB', scales=[1, 3, 6], deadline_seconds=300,
        gain_multiple=2, reference_fee=.3, risk_cost_multiple=2, risk_half_multiple=.1, minimum_risk=.5),
        execution=dict(fee_roundtrip=.3, latency_ms=100, pending_timeout_seconds=2, max_quote_gap_seconds=60),
        risk=dict(cost_multipliers=[2., 4., 6.], half_multiple=.1, minimum_risk=.5))


def test_new_query_matches_old_amplitude_function_and_map_strictly_follows_entry():
    day, q = synthetic()
    t = np.arange(8_000, 8_301, 30, dtype=np.int64)*NS
    result = build_targets(day, q, t, cfg())
    legacy = target_frame(result['map'], result['roots'], [1, 3, 6], 300, 2)
    assert np.array_equal(legacy.amplitude_state, result['labels'].state)
    assert np.array_equal(legacy.all_known, result['labels'].all_known)
    known = result['map'].resolved
    assert (result['map'].loc[known, 'cross_ordinal'] > result['map'].loc[known, 'entry_ordinal']).all()
    assert len(result['map']) == len(t)*9 and len(result['executions']) == len(t)*6
    assert result['receipts']['checks']
    assert (result['labels'].loc[result['labels'].all_known, 'available_ns'] > result['labels'].loc[result['labels'].all_known, 't_ns']).all()


def test_risk_grid_changes_neither_event_target_nor_bank_id_and_gain_is_fixed():
    day, q = synthetic()
    t = np.arange(8_000, 8_101, 20, dtype=np.int64)*NS
    a = build_targets(day, q, t, cfg())
    other = cfg()
    other['risk']['cost_multipliers'] = [3., 8.]
    other['risk']['half_multiple'] = .25
    b = build_targets(day, q, t, other)
    pd.testing.assert_frame_equal(a['labels'], b['labels'])
    pd.testing.assert_frame_equal(a['bank'], b['bank'])
    pd.testing.assert_frame_equal(a['map'], b['map'])
    assert a['receipts']['event_query_sha256'] == b['receipts']['event_query_sha256']
    for _, rows in a['executions'].groupby('t_ns'):
        assert rows.G_event.nunique() == 1
        assert rows.groupby('plan').R_trade.first().nunique() > 1


def test_future_prices_cannot_change_origin_scene_risk_or_entry():
    day, q = synthetic()
    t = np.array([8_000, 8_030, 8_060], np.int64)*NS
    a = build_targets(day, q, t, cfg())
    day2, altered = synthetic(future_shift=(8_070, 100.))
    b = build_targets(day2, altered, t, cfg())
    pd.testing.assert_frame_equal(a['scene'], b['scene'])
    for key in ('R_event', 'G_event', 'R_cost2', 'R_cost4', 'R_cost6', 'entry_ordinal', 'entry_ns', 'decision_eligible'):
        pd.testing.assert_series_equal(a['roots'][key], b['roots'][key])


def test_empty_bank_and_unknown_geometry_do_not_fabricate_negative_labels():
    day, q = synthetic(constant=True)
    t = np.array([8_000, 8_030], np.int64)*NS
    result = build_targets(day, q, t, cfg())
    assert result['bank'].empty
    assert not result['map'].resolved.any()
    assert (result['labels'].state == -1).all()
    assert not result['labels'].all_known.any()
    assert (result['executions'].status == 'INELIGIBLE').all()


def test_five_second_grid_contains_identical_thirty_second_queries():
    day, q = synthetic()
    thirty = np.arange(8_010, 8_101, 30, dtype=np.int64)*NS
    five = np.arange(8_010, 8_101, 5, dtype=np.int64)*NS
    a = build_targets(day, q, thirty, cfg())
    b = build_targets(day, q, five, cfg())
    pd.testing.assert_frame_equal(a['labels'], b['labels'].loc[b['labels'].t_ns.isin(thirty)].reset_index(drop=True))
    pd.testing.assert_frame_equal(a['bank'], b['bank'])
