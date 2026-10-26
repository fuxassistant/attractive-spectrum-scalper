from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import spectrum.features.families.flow_context as f


def tape(date='2026-01-06', n=300):
    start = pd.Timestamp(date + ' 08:00', tz='UTC').value
    t = start + np.arange(1, n + 1) * f.MINUTE
    return dict(t=t, available=t + 5, mid=40000 + np.sin(np.arange(n) / 3) * 10 + np.arange(n) * .01,
                volume=10 + np.arange(n) % 7, coverage=np.ones(n), valid=np.ones(n, bool), segment=np.ones(n, int),
                london=start, end=t[-1], symbol='GCH6', date=date)


def test_flow_has_no_episode_dependency_in_columns():
    assert len(f.FAMILIES['flow']) == 22 and len(f.FAMILIES['rich_spectrum']) == 66
    assert not any(x.startswith(('episode', 'reaction')) for x in f.FAMILIES['flow'])
    assert set(f.FAMILIES['flow_episode']) - set(f.FAMILIES['flow']) == set(f.ep.EPISODE + ['reaction5', 'reaction15'])


def test_context_and_spectrum_future_mutation_preserves_prefix():
    d = tape(); prev = tape('2026-01-05'); t = d['t'][130:180] + 10
    a, meta = f.context_features(d, prev, t, np.full(len(t), 40000.), np.ones(len(t)))
    changed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()}
    changed['mid'][180:] = 90000; changed['volume'][180:] = 999999
    b, _ = f.context_features(changed, prev, t, np.full(len(t), 40000.), np.ones(len(t)))
    pd.testing.assert_frame_equal(a, b)
    assert (meta['latest_available_ns'] <= t).all() and (meta['latest_endpoint_ns'] <= t).all()


def test_no_future_minute_or_missing_previous_weekday_borrow():
    d = tape('2026-01-12'); previous = tape('2026-01-08')  # Thu cannot replace absent Fri.
    t = np.array([d['t'][150]])  # minute 150 not yet published at this exact endpoint
    x, m = f.context_features(d, previous, t, [40000.], [1.])
    assert m['latest_endpoint_ns'][0] == d['t'][149]
    assert x.previous_known.iloc[0] == 0 and np.isnan(x.previous_vwap.iloc[0])
    proper = tape('2026-01-09'); proper['symbol'] = 'GCJ6'
    xx, _ = f.context_features(d, proper, t, [40000.], [1.])
    assert xx.previous_known.iloc[0] == 0


def test_stft_gap_epoch_and_unknown_volume_are_not_bridged():
    d = tape(); full = f.trailing_spectra(d)
    assert np.isfinite(full[160]).all()
    d['segment'][150:] = 2
    changed = f.trailing_spectra(d)
    assert np.isnan(changed[160]).all() and np.isfinite(changed[280]).all()
    d['valid'][170] = False
    assert np.isnan(f.trailing_spectra(d)[200]).all()


def test_minute_support_rejects_internal_quote_reset_even_with_good_endpoints():
    day = SimpleNamespace(meta=dict(feature_names=['log_volume', 'coverage', 'candle_known'], london_ns='0', end_ns=str(4*f.MINUTE), symbol='X', date='2026-01-06'),
        x=np.ones((3, 3)), cuts=np.array([2, 5, 7]), execution_valid=np.ones(7, bool),
        execution_segment=np.array([1, 1, 1, 2, 2, 2, 2]), bar_t=np.arange(1, 4)*f.MINUTE,
        decision_t=np.arange(1, 4)*f.MINUTE+5, decision_mid=np.ones(3)*4000)
    d = f.minute_tape(day)
    assert d['valid'].tolist() == [False, False, True]
