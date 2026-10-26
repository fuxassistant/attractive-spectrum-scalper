"""Causal scalar projections formerly embedded in the representation probe job."""
import numpy as np
import pandas as pd

BASE = ['ret30_ticks', 'ret300_ticks', 'rv30_ticks', 'range30_ticks', 'range300_ticks',
        'spread_ticks', 'clock_sin', 'clock_cos']
BOOK = ['aggression30', 'unknown_fraction30', 'depth_imbalance', 'volume_decay30', 'log_trade_quantity30',
        'log_depth_bid_12', 'log_depth_ask_12'] + [s + '_' + f for s in ('bid', 'ask') for f in
        ('fill_fraction30', 'withdraw_fraction30', 'refill_fraction30', 'depth_change30')] + [
        'log_' + s + '_' + f for s in ('bid', 'ask') for f in ('near_add30', 'near_fill30', 'near_withdrawal30', 'near_refill30')]
CONTEXT = ['pos_' + v for v in ('day_vwap', 'combined_vwap', 'previous_high', 'previous_low', 'previous_vwap', 'previous_poc', 'day_high', 'day_low')]
CONTEXT += ['day_sigma', 'combined_sigma', 'day_coverage', 'combined_coverage', 'stft_2_8m', 'stft_8_32m', 'stft_32_64m', 'stft_entropy']
FAMILIES = dict(price=BASE, price_book=BASE + BOOK, price_context=BASE + CONTEXT, all=BASE + BOOK + CONTEXT)


def summary_at(fast, context, origins, *, date):
    """Project only published rows; eligibility never consults a future outcome."""
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or not len(origins) or np.any(np.diff(origins) <= 0):
        raise ValueError('Ordered unique decision origins required')
    if not fast.available_ns.is_monotonic_increasing:
        raise ValueError('Ordered fast publication clock required')
    i = np.searchsorted(fast.available_ns.to_numpy(), origins, side='right') - 1
    if np.any(i < 0):
        raise ValueError('Decision precedes available fast history')
    row = fast.iloc[i].reset_index(drop=True)
    c = pd.merge_asof(pd.DataFrame(dict(origin_ns=origins)), context.sort_values('available_ns'),
                      left_on='origin_ns', right_on='available_ns')
    if c.available_ns.isna().any():
        raise ValueError('Decision precedes available context')
    x = row[[n for n in BASE + BOOK if n in row]].copy()
    for name in ('trade_quantity30', 'depth_bid_12', 'depth_ask_12') + tuple(
            side + '_' + field for side in ('bid', 'ask') for field in
            ('near_add30', 'near_fill30', 'near_withdrawal30', 'near_refill30')):
        x['log_' + name] = np.log1p(row[name])
    phase = pd.to_datetime(origins, utc=True).tz_convert('Europe/London')
    phase = (phase.hour*3600 + phase.minute*60 + phase.second)/86400*2*np.pi
    x['clock_sin'], x['clock_cos'] = np.sin(phase), np.cos(phase)
    scale = c.bb1_half.clip(lower=.1)
    for name in ('day_vwap', 'combined_vwap', 'previous_high', 'previous_low',
                 'previous_vwap', 'previous_poc', 'day_high', 'day_low'):
        x['pos_' + name] = (row.mid - c[name])/scale
    for name in CONTEXT:
        if not name.startswith('pos_'):
            x[name] = c[name]
    x['date'], x['origin_ns'] = date, origins
    x['root_id'] = [f'{date}:{int(t)}' for t in origins]
    x['available_ns'] = np.maximum(row.available_ns, c.available_ns)
    x['known'] = row.known & c.known.fillna(False) & row.ret300_ticks.notna()
    return x
