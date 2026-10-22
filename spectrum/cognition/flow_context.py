"""Decision-time flow ablation and explicitly compressed day/previous-day context."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import episodes as ep

NS = ep.NS
MINUTE = 60 * NS
FLOW = [c for c in ep.RESPONSE if not c.startswith('reaction')]
RICH = ['ret300', 'range60', 'range300', 'unknown5', 'depth12'] + [f'volume{w}' for w in (5, 30, 60, 300)] + [f'{v}{w}' for w in (60, 300) for v in ('pressure', 'unknown', 'book')]
DAY = ['day_age_minutes', 'context_age_seconds', 'day_coverage', 'day_from_open', 'day_from_high', 'day_from_low', 'day_vwap', 'day_range', 'day_log_volume',
       'previous_known', 'previous_age_days', 'previous_coverage', 'previous_from_open', 'previous_from_close', 'previous_from_high', 'previous_from_low', 'previous_vwap', 'previous_range',
       'combined_vwap', 'combined_range', 'volume_vs_previous']
SPECTRUM = [f'stft_{source}_{field}' for source in ('price', 'volume') for field in ('b0', 'b1', 'b2', 'entropy')]
FAMILIES = dict(baseline=ep.BASE, flow=ep.BASE + FLOW, flow_episode=ep.BASE + ep.EPISODE + ep.RESPONSE,
                rich_flow=ep.BASE + FLOW + RICH, rich_day=ep.BASE + FLOW + RICH + DAY,
                rich_spectrum=ep.BASE + FLOW + RICH + DAY + SPECTRUM)
FAMILIES['shuffled'] = FAMILIES['rich_spectrum']
SIGNED = ep.SIGNED | {'ret300', 'pressure60', 'pressure300', 'book60', 'book300', 'depth12'} | {c for c in DAY if '_from_' in c or c.endswith('_vwap')}


def contract():
    return json.loads((Path(__file__).resolve().parents[2] / 'contracts/flow_context_v2.json').read_text(encoding='utf-8'))


def local_features(df, segment):
    x, meta = ep.features(df, segment)
    for w in (60, 300):
        x[f'range{w}'] = ep.measured(df, f'range{w}_pos', f'range{w}_known') - .5
    x['ret300'] = ep.measured(df, 'ret300', 'ret300_known')
    x['depth12'] = ep.measured(df, 'flow_depth12_imbalance', 'flow_depth12_known')
    for w in (5, 30, 60, 300):
        x[f'volume{w}'] = ep.measured(df, f'flow_log_volume{w}', f'flow_volume{w}_known')
        if w != 30:
            x[f'unknown{w}'] = ep.measured(df, f'flow_unknown{w}_fraction', f'flow_pressure{w}_known')
        if w >= 60:
            x[f'pressure{w}'] = (ep.measured(df, f'flow_pressure{w}_low', f'flow_pressure{w}_known') + ep.measured(df, f'flow_pressure{w}_high', f'flow_pressure{w}_known')) / 2
            x[f'book{w}'] = ep.measured(df, f'flow_book{w}_pressure', f'flow_book{w}_known')
    return x, meta


def orient(x, side):
    out = x.copy()
    for name in SIGNED | {'range60', 'range300'}:
        if name in out:
            out[name] *= side
    out['side'] = side
    return out


def minute_tape(day):
    names = day.meta['feature_names']; x = np.asarray(day.x)
    cuts = np.asarray(day.cuts, np.int64); last = cuts - 1
    epochs = np.asarray(day.execution_segment); qvalid = np.asarray(day.execution_valid, bool)
    changes = np.cumsum(np.r_[True, (epochs[1:] != epochs[:-1]) | ~qvalid[1:] | ~qvalid[:-1]])
    prev_last = np.r_[-1, last[:-1]]
    supported = (last >= 0) & (prev_last >= 0)
    safe, prev_safe = np.maximum(last, 0), np.maximum(prev_last, 0)
    supported &= qvalid[safe] & (changes[safe] == changes[prev_safe])
    return dict(t=np.asarray(day.bar_t, np.int64), available=np.maximum.accumulate(np.asarray(day.decision_t, np.int64)),
                mid=np.asarray(day.decision_mid, float) / .1,
                volume=np.expm1(x[:, names.index('log_volume')].astype(float)),
                coverage=x[:, names.index('coverage')].astype(float),
                valid=(x[:, names.index('candle_known')] > .5) & supported,
                segment=np.where(supported, epochs[safe], -1),
                london=int(day.meta['london_ns']), end=int(day.meta['end_ns']), symbol=day.meta['symbol'], date=day.meta['date'])


def trailing_spectra(tape):
    """Hann spectra at a published completed M1 endpoint, no gap filling."""
    n = len(tape['t']); out = np.full((n, len(SPECTRUM)), np.nan)
    w = 128; freq = np.fft.rfftfreq(w); hann = np.hanning(w)
    bands = [(freq > 1 / hi) & (freq <= 1 / lo) for lo, hi in ((2, 8), (8, 32), (32, 128))]
    valid = tape['valid'] & np.isfinite(tape['mid']) & (tape['mid'] > 0) & (tape['coverage'] >= .95)
    returns = np.r_[np.nan, np.diff(tape['mid'])]
    volume = np.log1p(tape['volume'])
    for i in range(w, n):
        lo = i - w + 1
        if not valid[lo - 1:i + 1].all() or not np.all(np.diff(tape['t'][lo - 1:i + 1]) == MINUTE) or not np.all(tape['segment'][lo - 1:i + 1] == tape['segment'][i]):
            continue
        for j, series in enumerate((returns, volume)):
            a = series[lo:i + 1]; a = (a - a.mean()) * hann
            power = np.abs(np.fft.rfft(a)) ** 2; total = power[1:].sum()
            if total <= 1e-16:
                continue
            p = power[1:] / total
            out[i, j * 4:j * 4 + 3] = [power[b].sum() / total for b in bands]
            out[i, j * 4 + 3] = -np.sum(p * np.log(np.maximum(p, 1e-300))) / np.log(len(p))
    return out


def context_features(current, previous, intents, price_ticks, sigma):
    """Current London prefix and previous London-to-close summaries, as of t.

    Combined VWAP includes current pre-London overnight observations. Coverage
    and previous-known are explicit. Missing prior weekdays/contract rolls never
    borrow a different session or create price levels equal to zero.
    """
    t = np.asarray(intents, np.int64); n = len(t); scale = np.maximum(np.asarray(sigma) * np.sqrt(600), 1e-9)
    out = {c: np.full(n, np.nan) for c in DAY + SPECTRUM}
    for c in ('previous_known',):
        out[c][:] = 0.
    dt = current['t']; av = current['available']; mid = current['mid']; vol = current['volume']
    good = current['valid'] & np.isfinite(mid) & (mid > 0) & np.isfinite(vol) & (vol >= 0)
    idx = np.minimum(np.searchsorted(dt, t, side='right'), np.searchsorted(av, t, side='right')) - 1
    spectra = trailing_spectra(current)
    prev = None
    expected_previous = str((pd.Timestamp(current['date']) - pd.offsets.BDay()).date())
    if previous is not None and previous['symbol'] == current['symbol'] and previous['date'] == expected_previous:
        age = (pd.Timestamp(current['date']) - pd.Timestamp(previous['date'])).days
        mask = (previous['t'] > previous['london']) & (previous['t'] <= previous['end'])
        mask &= previous['valid'] & np.isfinite(previous['mid']) & (previous['mid'] > 0)
        expected = max(1, (previous['end'] - previous['london']) / MINUTE)
        cov = float(np.sum(np.where(mask, previous['coverage'], 0.)) / expected)
        published = previous['available'][mask].max() if mask.any() else np.iinfo(np.int64).max
        out['previous_age_days'][t >= published] = age; out['previous_coverage'][t >= published] = cov
        if 0 < age <= 4 and cov >= contract()['previous_required_coverage'] and mask.any():
            pm, pv = previous['mid'][mask], previous['volume'][mask]
            prev = dict(open=pm[0], close=pm[-1], high=pm.max(), low=pm.min(), volume=pv.sum(), pv=np.dot(pm, pv), available=previous['available'][mask].max())
    for row, k in enumerate(idx):
        if k < 0:
            continue
        price = price_ticks[row]; sc = scale[row]
        out['context_age_seconds'][row] = (t[row] - dt[k]) / NS
        mask = good[:k + 1] & (dt[:k + 1] > current['london'])
        expected = max(1., (min(t[row], dt[k]) - current['london']) / MINUTE)
        out['day_age_minutes'][row] = max(0., (t[row] - current['london']) / MINUTE)
        out['day_coverage'][row] = np.sum(np.where(mask, current['coverage'][:k + 1], 0.)) / expected
        if mask.any():
            mm, vv = mid[:k + 1][mask], vol[:k + 1][mask]
            for field, level in (('open', mm[0]), ('high', mm.max()), ('low', mm.min())):
                out['day_from_' + field][row] = (price - level) / sc
            out['day_range'][row] = (mm.max() - mm.min()) / sc
            out['day_log_volume'][row] = np.log1p(vv.sum())
            if vv.sum() > 0:
                out['day_vwap'][row] = (price - np.dot(mm, vv) / vv.sum()) / sc
        if prev is not None and prev['available'] <= t[row]:
            out['previous_known'][row] = 1.
            for field in ('open', 'close', 'high', 'low'):
                out['previous_from_' + field][row] = (price - prev[field]) / sc
            out['previous_range'][row] = (prev['high'] - prev['low']) / sc
            all_mid, all_vol = mid[:k + 1][good[:k + 1]], vol[:k + 1][good[:k + 1]]
            if prev['volume'] > 0:
                out['previous_vwap'][row] = (price - prev['pv'] / prev['volume']) / sc
                out['volume_vs_previous'][row] = np.log1p(all_vol.sum() / prev['volume'])
            if prev['volume'] + all_vol.sum() > 0:
                out['combined_vwap'][row] = (price - (prev['pv'] + np.dot(all_mid, all_vol)) / (prev['volume'] + all_vol.sum())) / sc
            if len(all_mid):
                out['combined_range'][row] = (max(prev['high'], all_mid.max()) - min(prev['low'], all_mid.min())) / sc
        if t[row] - dt[k] <= 90 * NS:
            for j, name in enumerate(SPECTRUM):
                out[name][row] = spectra[k, j]
    return pd.DataFrame(out), dict(latest_available_ns=np.where(idx >= 0, av[np.maximum(idx, 0)], -1), latest_endpoint_ns=np.where(idx >= 0, dt[np.maximum(idx, 0)], -1))


def factor_probabilities(fill_probability, conditional):
    f = np.asarray(fill_probability); q = np.asarray(conditional)
    if q.shape != (len(f), 3) or not np.allclose(q.sum(axis=1), 1):
        raise ValueError('Conditional probabilities must sum to one')
    return np.c_[f[:, None] * q, 1 - f]
