"""Causal, multi-rate market observations. Future outcomes are a separate table.

Times are int64 nanoseconds, prices dollars/oz, quantities GC contracts. Every
bucket has an actual publication watermark. NaN is unknown, zero is measured.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from spectrum.units import NS, MINUTE, GC_TICK_USD as TICK
from spectrum.market.calendar import MarketCalendar
from spectrum.features.contract import reject_future
RAW_FIELDS = ('add', 'remove', 'linked_fill', 'passive_fill', 'aggressor_fill',
              'ambiguous_remove', 'unlinked_remove', 'refill', 'refill_delay_qseconds',
              'removed_age_qseconds', 'removed_age_known_qty', 'near_add', 'near_remove',
              'near_fill', 'near_withdrawal', 'near_ambiguous', 'near_refill')


def anchor(date):
    return int(pd.Timestamp(date + ' 08:00', tz='Europe/London').value)




def scheduled_open(t_ns):
    """The same GC venue schedule used by finalized data and feature streams."""
    return ~MarketCalendar().closed(t_ns)


def grouped_rolling(values, groups, window, op='sum'):
    s = pd.Series(np.asarray(values))
    g = np.asarray(groups)
    if len(s) != len(g) or window < 1:
        raise ValueError('Rolling groups/window mismatch')
    if not len(s):
        return np.empty(0,float)
    # Actual continuity groups are contiguous runs. Compute the rolling window
    # once, then mask windows crossing a run boundary; do not allocate tens of
    # thousands of one-row pandas groups during a closed market.
    starts = np.r_[True,g[1:]!=g[:-1]]
    if len(np.unique(g[starts])) != int(starts.sum()):
        # General public API also accepts recurring, noncontiguous groups.
        return s.groupby(g,sort=False).transform(
            lambda a:getattr(a.rolling(window,min_periods=window),op)()).to_numpy()
    result = getattr(s.rolling(window,min_periods=window),op)().to_numpy()
    run = np.cumsum(starts)
    same = np.zeros(len(s),bool)
    if len(s)>=window:
        same[window-1:]=run[window-1:]==run[:len(s)-window+1]
    return np.where(same,result,np.nan)


def ledger_releases(seconds, complete_quotes):
    """First COMPLETE publication at/after each half-open ledger bucket end.

    An absent publication is not evidence of a measured zero. This watermark
    also releases empty buckets, which have no rows in the sparse price ledger.
    """
    times = complete_quotes.t_ns.to_numpy(np.int64)
    i = np.searchsorted(times, seconds.t_ns.to_numpy(), side='left')
    release = np.full(len(seconds), np.iinfo(np.int64).max, np.int64)
    known = i < len(times)
    release[known] = complete_quotes.available_ns.to_numpy(np.int64)[i[known]]
    return release


def make_fast(seconds, levels, trades):
    """Complete 1s source buckets with near-book mutation facts and T-only volume.

    Inputs must belong to one contract. Canonical epochs remain namespaced by
    capture. Missing values never shorten physical history.
    """
    df = seconds.sort_values('t_ns').reset_index(drop=True).copy()
    assert df.t_ns.is_unique
    df['scheduled_open'] = scheduled_open(df.t_ns)
    df['interval_open'] = scheduled_open(df.t_ns-1)
    df['known'] = df.valid & df.book_valid & df.interval_open
    df['decision_open'] = df.scheduled_open
    # Quote age is a feature, not a reason to erase an otherwise supported
    # physical history. Canonical validity already enforces the 30s contract.
    df['fresh_2s'] = df.known & (df.age_ns <= 2 * NS)
    df['quote_age_seconds'] = df.age_ns / NS
    continuity = df['segment_id'] if 'segment_id' in df else df.epoch
    same = continuity.eq(continuity.shift()) & df.t_ns.diff().eq(NS)
    df['local_segment'] = (~(same & df.known & df.known.shift(fill_value=False))).cumsum()
    # Source totals can be zero only inside a complete, supported clock bucket.
    available = df.available_ns.to_numpy().copy()
    if 'ledger_release_ns' in df:
        available = np.maximum(available, df.ledger_release_ns.to_numpy())
    for side, name in ((1, 'bid'), (-1, 'ask')):
        part = levels[levels.side == side]
        totals = part.groupby('t_ns', sort=True)[list(RAW_FIELDS)].sum()
        # reindex(fill_value=0) preserves int64; fillna after reindex would
        # silently round epoch nanoseconds through float64.
        availability = part.groupby('t_ns').available_ns.max().reindex(df.t_ns, fill_value=0).to_numpy(np.int64)
        available = np.maximum(available, availability)
        totals = totals.reindex(df.t_ns, fill_value=0)
        for field in RAW_FIELDS:
            df[name + '_' + field] = totals[field].to_numpy()
    tr = trades.copy()
    tr['bucket'] = (tr.t_ns // NS + 1) * NS
    tr['buy'] = tr.quantity.where(tr.side == 1, 0)
    tr['sell'] = tr.quantity.where(tr.side == -1, 0)
    tr['unknown'] = tr.quantity.where(tr.side == 0, 0)
    tr['unsupported'] = tr.quantity.where(~tr.supported, 0)
    # Use tick displacement around a fixed numerical origin for stable moments.
    tr['px'] = tr.price_nano / 1e9
    tr['pv'] = (tr.px - 4000.) * tr.quantity
    tr['p2v'] = (tr.px - 4000.)**2 * tr.quantity
    totals = tr.groupby('bucket')[['quantity', 'buy', 'sell', 'unknown', 'unsupported', 'pv', 'p2v']].sum()
    totals = totals.reindex(df.t_ns, fill_value=0)
    for field in totals:
        df['trade_' + field] = totals[field].to_numpy()
    trade_av = tr.groupby('bucket').available_ns.max().reindex(df.t_ns,fill_value=0).to_numpy(np.int64)
    available = np.maximum(available,trade_av)
    df['source_available_ns'] = available
    # The joined prefix cannot be published until ALL preceding buckets are
    # available, including the complete event releasing an interaction bucket.
    df['available_ns'] = np.maximum.accumulate(available)
    df['ledger_known'] = df.known & same & df.known.shift(fill_value=False) & (df.trade_unsupported == 0)
    if 'ledger_complete' in df:
        df['ledger_known'] &= df.ledger_complete
    df['trade_known'] = df.known & (df.trade_unsupported==0) & df.get('trade_complete',df.ledger_known)
    df['spread_ticks'] = (df.ask - df.bid) / TICK
    df['ret1_ticks'] = df.mid.diff().where(same & df.known & df.known.shift(fill_value=False)) / TICK
    df['path1_ticks'] = df.c_path_ticks.diff().where(same & df.known & df.known.shift(fill_value=False))
    df['depth_imbalance'] = (df.depth_bid_12 - df.depth_ask_12) / (df.depth_bid_12 + df.depth_ask_12).replace(0, np.nan)
    # Fixed windows have physical length because source rows are checked at 1s.
    group = df.local_segment
    valid_ledger = grouped_rolling(df.ledger_known.astype(float), group, 30) == 30
    for field in ('quantity', 'buy', 'sell', 'unknown'):
        df['trade_' + field + '30'] = grouped_rolling(df['trade_' + field], group, 30).astype(float)
    for side in ('bid', 'ask'):
        for field in ('near_add', 'near_remove', 'near_fill', 'near_withdrawal', 'near_refill', 'near_ambiguous'):
            df[side + '_' + field + '30'] = grouped_rolling(df[side + '_' + field], group, 30)
    df['ret30_ticks'] = (df.mid - df.mid.groupby(group).shift(30)) / TICK
    df['ret300_ticks'] = (df.mid - df.mid.groupby(group).shift(300)) / TICK
    df['path30_ticks'] = grouped_rolling(df.path1_ticks, group, 30)
    df['rv30_ticks'] = np.sqrt(grouped_rolling(df.ret1_ticks**2, group, 30))
    df['range30_ticks'] = (grouped_rolling(df.mid, group, 30, 'max') - grouped_rolling(df.mid, group, 30, 'min')) / TICK
    df['range300_ticks'] = (grouped_rolling(df.mid, group, 300, 'max') - grouped_rolling(df.mid, group, 300, 'min')) / TICK
    known_side = df.trade_buy30 + df.trade_sell30
    df['aggression30'] = (df.trade_buy30 - df.trade_sell30) / known_side.replace(0, np.nan)
    df['unknown_fraction30'] = df.trade_unknown30 / df.trade_quantity30.replace(0, np.nan)
    for side in ('bid', 'ask'):
        remove = df[side + '_near_remove30']
        df[side + '_fill_fraction30'] = df[side + '_near_fill30'] / remove.replace(0, np.nan)
        df[side + '_withdraw_fraction30'] = df[side + '_near_withdrawal30'] / remove.replace(0, np.nan)
        df[side + '_refill_fraction30'] = df[side + '_near_refill30'] / df[side + '_near_fill30'].replace(0, np.nan)
        df[side + '_depth_change30'] = df['depth_' + side + '_12'] - df['depth_' + side + '_12'].groupby(group).shift(30)
    prior_vol = df.trade_quantity30.groupby(group).shift(30)
    df['volume_decay30'] = df.trade_quantity30 / prior_vol.replace(0, np.nan)
    df['mechanism_known'] = valid_ledger & df.ret30_ticks.notna() & (df.unknown_fraction30 < .25)
    # Descriptive tags, never mutually exclusive classes or presumed causes.
    effort = df.trade_quantity30 >= 20
    stalled = df.ret30_ticks.abs() <= 2
    refill = np.maximum(df.bid_near_refill30, df.ask_near_refill30)
    fill = df.bid_near_fill30 + df.ask_near_fill30
    withdrawal = df.bid_near_withdrawal30 + df.ask_near_withdrawal30
    df['REFILL_UNDER_PRESSURE'] = df.mechanism_known & effort & stalled & (refill >= 5) & (df.aggression30.abs() >= .35)
    df['EXECUTED_DEPLETION'] = df.mechanism_known & effort & (df.ret30_ticks.abs() >= 4) & (fill >= 10) & (fill >= .25 * withdrawal)
    df['WITHDRAWAL'] = df.mechanism_known & (withdrawal >= 50) & (withdrawal > 4 * fill) & (df.ret30_ticks.abs() >= 4)
    df['AGGRESSION_DECAY'] = df.mechanism_known & stalled & (prior_vol >= 30) & (df.volume_decay30 <= .35)
    return df.copy()


def make_minutes(fast, trades, source_bars=None, *, bb_ddof=1):
    """Completed M1 observations, exact trade VWAP, trailing BB and spectra."""
    if bb_ddof not in (0,1):
        raise ValueError('BB requires an explicit population/sample definition')
    d = fast.copy()
    # seconds are interval ends, including the exact final second in the minute.
    d['minute'] = ((d.t_ns - 1) // MINUTE + 1) * MINUTE
    g = d.groupby('minute', sort=True)
    out = g.agg(t_ns=('t_ns', 'max'), available_ns=('available_ns', 'max'),
                open=('mid', 'first'), high=('mid', 'max'), low=('mid', 'min'), close=('mid', 'last'),
                coverage=('known', 'mean'), seconds=('t_ns', 'size'),
                epoch=('epoch', 'last'), epochs=('epoch', 'nunique'),
                scheduled_open=('scheduled_open', 'max'), volume=('trade_quantity', 'sum'),
                buy=('trade_buy', 'sum'), sell=('trade_sell', 'sum'), unknown=('trade_unknown', 'sum'),
                unsupported=('trade_unsupported', 'sum'), pv=('trade_pv', 'sum'), p2v=('trade_p2v', 'sum'))
    out['t_ns'] = out.index
    out = out.reset_index(drop=True)
    out['interval_open'] = scheduled_open(out.t_ns-1)
    out['scheduled_open'] = scheduled_open(out.t_ns)
    # Incomplete last-minute rows are not published early.
    out = out[out.t_ns <= int(fast.t_ns.max())].reset_index(drop=True)
    # Retain internal incomplete minutes as unknown physical-time rows. The
    # first subsequent published second proves that the bucket has ended.
    ix = np.searchsorted(fast.t_ns.to_numpy(), out.t_ns.to_numpy())
    out['available_ns'] = np.maximum(out.available_ns, fast.available_ns.to_numpy()[ix])
    out['known'] = (out.coverage >= .95) & (out.epochs == 1) & (out.unsupported == 0) & out.interval_open
    out['known'] &= out.seconds == 60
    if source_bars is not None:
        source = source_bars.set_index('t_ns').reindex(out.t_ns)
        # Preserve true intrasecond extrema from the all-event canonical tape.
        for name, original in (('open', 'o'), ('high', 'h'), ('low', 'l'), ('close', 'c')):
            out[name] = source[original].to_numpy()
        out['fast_coverage'] = out.coverage
        out['coverage'] = source.coverage.fillna(0).to_numpy()
        out['known'] = source.valid.eq(True).to_numpy() & out.interval_open
        sa = source_bars.set_index('t_ns').available_ns.reindex(out.t_ns, fill_value=0).to_numpy(np.int64)
        out['available_ns'] = np.maximum(out.available_ns, sa)
    # Read exact T prints directly so a partial last second of a UTC capture
    # is not silently lost from the next completed minute's VWAP.
    tr = trades.copy()
    tr['minute'] = (tr.t_ns // MINUTE + 1) * MINUTE
    for name, side in (('buy', 1), ('sell', -1), ('unknown', 0)):
        tr[name] = tr.quantity.where(tr.side == side, 0)
    tr['unsupported'] = tr.quantity.where(~tr.supported, 0)
    price_offset = tr.price_nano / NS - 4000.
    tr['pv'] = price_offset * tr.quantity
    tr['p2v'] = price_offset**2 * tr.quantity
    totals = tr.groupby('minute')[['quantity', 'buy', 'sell', 'unknown', 'unsupported', 'pv', 'p2v']].sum().reindex(out.t_ns, fill_value=0)
    for name in totals:
        out['volume' if name == 'quantity' else name] = totals[name].to_numpy()
    out['known'] &= out.unsupported == 0
    trade_av = tr.groupby('minute').available_ns.max().reindex(out.t_ns,fill_value=0).to_numpy(np.int64)
    out['available_ns'] = np.maximum(out.available_ns,trade_av)
    out['available_ns'] = np.maximum.accumulate(out.available_ns)
    out['trade_vwap_minute'] = 4000. + out.pv / out.volume.replace(0, np.nan)
    out['midpoint_proxy_minute'] = out.close
    out['trade_delta'] = out.buy - out.sell
    london = pd.to_datetime(out.t_ns, utc=True).dt.tz_convert('Europe/London')
    phase = (london.dt.hour * 3600 + london.dt.minute * 60) / 86400 * (2 * np.pi)
    out['clock_sin'], out['clock_cos'] = np.sin(phase), np.cos(phase)
    continuous = out.t_ns.diff().eq(MINUTE) & out.epoch.eq(out.epoch.shift()) & out.known & out.known.shift(fill_value=False)
    out['segment'] = (~continuous).cumsum()
    # Completed native candles, explicitly distinct from developing quote BB.
    for scale in (1, 3, 6):
        sample = out[(out.t_ns % (scale * MINUTE) == 0)].copy()
        sg = (~(sample.known & sample.known.shift(fill_value=False) & sample.segment.eq(sample.segment.shift()) & sample.t_ns.diff().eq(scale * MINUTE))).cumsum()
        sample['bb' + str(scale)] = grouped_rolling(sample.close, sg, 20, 'mean')
        sample['bb' + str(scale) + '_half'] = grouped_rolling(sample.close, sg, 20, 'std') * np.sqrt(19 / (20-bb_ddof)) * 2
        cols = ['bb' + str(scale), 'bb' + str(scale) + '_half']
        sample['bb_segment'] = sample.segment
        aligned = pd.merge_asof(out[['available_ns']], sample[['available_ns', 'bb_segment'] + cols].sort_values('available_ns'), on='available_ns')
        out[cols] = aligned[cols].to_numpy()
        # A stale pre-gap band must not masquerade as a fresh supported one.
        out.loc[~out.known | out.segment.ne(aligned.bb_segment), cols] = np.nan
    n = 128
    freq = np.fft.rfftfreq(n)
    bands = [(freq > 1 / hi) & (freq <= 1 / lo) for lo, hi in ((2, 8), (8, 32), (32, 64))]
    ret = out.close.diff().to_numpy()
    stft = np.full((len(out), 4), np.nan)
    seg = out.segment.to_numpy()
    for i in range(n, len(out)):
        if seg[i] != seg[i - n] or not out.known.iloc[i]: continue
        a = ret[i - n + 1:i + 1]
        if not np.isfinite(a).all(): continue
        power = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(n)))**2
        total = power[1:].sum()
        if total > 1e-15:
            p = power[1:] / total
            stft[i] = [*(power[b].sum() / total for b in bands), -np.sum(p * np.log(np.maximum(p, 1e-300))) / np.log(len(p))]
    out[['stft_2_8m', 'stft_8_32m', 'stft_32_64m', 'stft_entropy']] = stft
    for name,band in zip(('stft_2_8m','stft_8_32m','stft_32_64m'),bands):
        out[name+'_bin_share'] = float(band.sum()/len(freq[1:]))
        out[name+'_relative_density'] = out[name]/out[name+'_bin_share']
        w=np.hanning(n)
        expected=np.maximum(0.,w@w-abs(np.fft.rfft(w))**2/n)
        expected[0]=0
        reference=float(expected[band].sum()/expected.sum())
        out[name+'_white_reference']=reference
        out[name+'_relative_white_power']=out[name]/reference
    out.attrs['bb_geometry'] = dict(version='COMPLETED_NATIVE_BB_V2',period=20,k=2,ddof=bb_ddof,
                                    clock='completed_native',scales=[1,3,6])
    # No 64-128m inference from <2 cycles. Energy is unsigned.
    return out


def anchored_context(minutes, trades, date, previous_date, *, current_anchor_ns=None,
                     previous_anchor_ns=None, end_ns=None):
    """Two London anchors; prior-cycle zones become known at publication time."""
    current = anchor(date) if current_anchor_ns is None else int(current_anchor_ns)
    previous = anchor(previous_date) if previous_anchor_ns is None else int(previous_anchor_ns)
    end = current+24*3600*NS if end_ns is None else int(end_ns)
    if not previous<current<end:
        raise ValueError('Context anchor order')
    m = minutes[(minutes.t_ns > previous) & (minutes.t_ns <= end)].copy().reset_index(drop=True)
    for name, start in (('combined', previous), ('day', current)):
        active = m.t_ns > start
        q = m.volume.where(active, 0).cumsum()
        pv = m.pv.where(active, 0).cumsum()
        p2v = m.p2v.where(active, 0).cumsum()
        mean = pv / q.replace(0, np.nan)
        m[name + '_vwap'] = 4000. + mean
        m[name + '_sigma'] = np.sqrt(np.maximum(0, p2v / q.replace(0, np.nan) - mean**2))
        proxy_supported = active & m.midpoint_proxy_minute.notna()
        proxy_q = m.volume.where(proxy_supported, 0).cumsum()
        proxy_pv = (m.midpoint_proxy_minute * m.volume).where(proxy_supported, 0).cumsum()
        m[name + '_midpoint_proxy'] = proxy_pv / proxy_q.replace(0, np.nan)
        # Exact-vs-proxy comparisons must use the same volume population. The
        # primary exact VWAP above still includes all observed T executions.
        matched_pv = m.pv.where(proxy_supported, 0).cumsum()
        m[name + '_trade_vwap_on_proxy_support'] = 4000. + matched_pv / proxy_q.replace(0, np.nan)
        m[name + '_proxy_volume_share'] = proxy_q / q.replace(0, np.nan)
        m[name + '_high'] = m.high.where(active & m.known).cummax()
        m[name + '_low'] = m.low.where(active & m.known).cummin()
        # Fixed-volume bins, cumulative only. All T prints, not T+F.
        tape = trades[(trades.t_ns >= start) & (trades.t_ns < end)].sort_values(['t_ns', 'ordinal'])
        ti = tape.t_ns.to_numpy(); bins = ((tape.price_nano.to_numpy() // 100_000_000) // 5) * 5
        qty = tape.quantity.to_numpy(); supported = tape.supported.to_numpy()
        profile = {}; cursor = 0; poc = []
        for row in m.itertuples():
            end_index = int(np.searchsorted(ti, row.t_ns, side='left'))
            for k in range(cursor, end_index):
                if supported[k]: profile[int(bins[k])] = profile.get(int(bins[k]), 0) + int(qty[k])
            cursor = end_index
            best = min(profile, key=lambda p: (-profile[p], p)) if profile else None
            poc.append(np.nan if best is None else (best + 2.5) * TICK)
        m[name + '_poc'] = poc
        expected = scheduled_open(m.t_ns - 1) & active.to_numpy()
        known = (m.known.to_numpy() & expected).cumsum()
        count = expected.cumsum()
        m[name + '_coverage'] = np.divide(known, count, out=np.zeros(len(m), float), where=count > 0)
    prior = m[m.t_ns <= current]
    if len(prior) == 0: raise ValueError('Missing previous London cycle')
    mature = int(max(current, prior.available_ns.max()))
    previous_values = dict(high=prior.high.where(prior.known).max(), low=prior.low.where(prior.known).min(),
                           vwap=prior.combined_vwap.iloc[-1], poc=prior.combined_poc.iloc[-1])
    for kind, value in previous_values.items():
        m['previous_' + kind] = np.where(m.available_ns >= mature, value, np.nan)
    m['previous_recognized_ns'] = np.where(m.available_ns >= mature, mature, 0)
    m['context_start_ns'] = previous
    return m


FAST_VALUES = ['mid', 'spread_ticks', 'depth_bid_1', 'depth_ask_1', 'depth_bid_4', 'depth_ask_4',
               'depth_bid_12', 'depth_ask_12', 'trade_quantity', 'trade_buy', 'trade_sell', 'trade_unknown',
               'ret1_ticks', 'path1_ticks', 'depth_imbalance', 'quote_age_seconds', 'fresh_2s'] + [side + '_' + field for side in ('bid', 'ask') for field in RAW_FIELDS]
DAY_VALUES = ['open', 'high', 'low', 'close', 'volume', 'trade_delta', 'unknown', 'trade_vwap_minute',
              'clock_sin', 'clock_cos', 'bb1', 'bb1_half', 'bb3', 'bb3_half', 'bb6', 'bb6_half',
              'stft_2_8m', 'stft_8_32m', 'stft_32_64m', 'stft_entropy'] + [a + '_' + b for a in ('day', 'combined') for b in ('vwap', 'sigma', 'high', 'low', 'poc', 'coverage')] + ['previous_' + x for x in ('high', 'low', 'vwap', 'poc')]
OPTIONAL_DAY_VALUES = [n+'_relative_white_power' for n in ('stft_2_8m','stft_8_32m','stft_32_64m')]
DAY_VALUES += OPTIONAL_DAY_VALUES


@dataclass
class Observation:
    t_ns: int
    fast: dict
    day: dict
    zones: pd.DataFrame
    events: pd.DataFrame
    receipt: dict


class MarketObservationStore:
    """As-of loader actually exposing both physical time scales and all masks."""
    def __init__(self, fast, context, zones, events, *, symbol, previous_date, date):
        for data in (fast, context, zones, events):
            reject_future(data.columns, data)
        self.fast, self.context = fast, context
        self.zones, self.events = zones, events
        self.symbol, self.previous_date, self.date = symbol, previous_date, date
        for data in (fast, context):
            assert data.available_ns.is_monotonic_increasing

    @staticmethod
    def _tensor(frame, columns, origin):
        reject_future(frame.columns, frame)
        a = frame[columns].to_numpy(float)
        mask = np.isfinite(a) & frame.known.to_numpy()[:, None]
        for j,name in enumerate(columns):
            if name+'__known' in frame:
                mask[:,j] = np.isfinite(a[:,j]) & frame[name+'__known'].to_numpy(bool)
        if 'ledger_known' in frame:
            for j, name in enumerate(columns):
                if name.startswith(('bid_', 'ask_')):
                    mask[:, j] &= frame.ledger_known.to_numpy()
                elif name.startswith('trade_'):
                    mask[:,j] &= frame.get('trade_known',frame.ledger_known).to_numpy()
        # Raw dollar prices and all dimensional values remain available; a
        # model may choose a separately declared causal normalization.
        return dict(values=np.where(mask, a, 0), known=mask,
                    t_ns=frame.t_ns.to_numpy(np.int64), available_ns=frame.available_ns.to_numpy(np.int64),
                    age_seconds=(origin - frame.t_ns.to_numpy()) / NS,
                    scheduled_open=frame.scheduled_open.to_numpy(bool), names=list(columns))

    @staticmethod
    def _normalize(tensor, mid, scale):
        a = tensor['values'].astype(float).copy()
        price_names = {'mid', 'open', 'high', 'low', 'close', 'trade_vwap_minute', 'bb1', 'bb3', 'bb6'}
        for j, name in enumerate(tensor['names']):
            if name in price_names or name.endswith(('_vwap', '_high', '_low', '_poc')) or name.startswith('previous_'):
                a[:, j] = (a[:, j] - mid) / scale
            elif name.endswith('_ticks'):
                a[:, j] /= scale / TICK
            elif name.endswith(('_half', '_sigma')):
                a[:, j] /= scale
            elif name.startswith(('depth_bid', 'depth_ask', 'trade_', 'bid_', 'ask_')) or name in ('volume', 'unknown'):
                a[:, j] = np.sign(a[:, j]) * np.log1p(np.abs(a[:, j]))
        tensor['normalized'] = np.where(tensor['known'], a, 0).astype(np.float32)
        tensor['normalization'] = dict(reference_mid=mid, reference_scale_dollars=scale,
            rule='price displacement / current completed BB1 halfwidth (floor 1 tick); signed log1p quantities; ticks / scale_ticks')
        return tensor

    def observe(self, t_ns):
        t_ns = int(t_ns)
        if t_ns < anchor(self.date): raise ValueError('Before declared decision day')
        start = anchor(self.previous_date)
        return self._observe(t_ns,start)

    def _observe(self,t_ns,start):
        fast = self.fast[(self.fast.available_ns <= t_ns) & (self.fast.t_ns > t_ns - 900 * NS)]
        day = self.context[(self.context.available_ns <= t_ns) & (self.context.t_ns > start)]
        zones = self.zones[self.zones.available_ns <= t_ns].groupby('zone_id', sort=False).tail(1).copy()
        ev = self.events[self.events.available_ns <= t_ns]
        truncated = len(ev) > 256
        ev = ev.tail(256).copy()
        for frame in (fast, day, zones, ev):
            assert len(frame) == 0 or frame.available_ns.max() <= t_ns
        latest = fast[fast.known]
        mid = float(latest.mid.iloc[-1]) if len(latest) else 0.
        scales = day.loc[day.known, 'bb1_half'].dropna()
        scale = max(TICK, float(scales.iloc[-1])) if len(scales) else TICK
        fast_tensor = self._normalize(self._tensor(fast, FAST_VALUES, t_ns), mid, scale)
        # Historical V1 payloads lack the new spectral references. Retain their
        # declared 36-channel view; never fabricate the three missing channels.
        day_names = [n for n in DAY_VALUES if n not in OPTIONAL_DAY_VALUES or n in day]
        day_tensor = self._normalize(self._tensor(day, day_names, t_ns), mid, scale)
        return Observation(t_ns, fast_tensor, day_tensor, zones, ev,
            dict(symbol=self.symbol, context_start_ns=start, last_day_ns=int(day.t_ns.max()) if len(day) else None,
                 fast_rows=len(fast), day_rows=len(day), day_truncated=False, events_truncated=truncated,
                 first_day_ns=int(day.t_ns.min()) if len(day) else None,
                 decision_ns=t_ns, previous_date=self.previous_date, day_known_fraction=float(day.known.mean()) if len(day) else 0,
                 source_to_tensor='fast 1s + complete M1 two-day prefix + as-of zones + last 256 event tokens',
                 future_outcomes_present=False))
