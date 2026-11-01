"""Past-only scales, volume anchors, and physical-time addresses for episodes."""
import numpy as np
import pandas as pd

from experiments.regime_language_v1.measures import TimeAtPrice

NS = 1_000_000_000
MINUTE = 60 * NS


def context_features(frame):
    """Each row is usable strictly AFTER its published available_ns.

    Rolling VWAP uses accepted-trade moments, not close times volume. Windows
    with missing minutes, closures, or epoch changes are explicitly unknown.
    ATR and rolling anchors never bridge these breaks.
    """
    f = frame.copy().reset_index(drop=True)
    valid = f.known & f.interval_open & f.epoch.ge(0)
    broken = (~valid | ~valid.shift(fill_value=False)
              | f.epoch.ne(f.epoch.shift()) | f.t_ns.diff().ne(MINUTE))
    segment = broken.cumsum()
    for name in ('atr', 'vwap60', 'sigma60', 'vwap240', 'sigma240',
                 'slope60', 'slope240'):
        f[name] = np.nan
    for _, ids in f.loc[valid].groupby(segment[valid]).groups.items():
        x = f.loc[ids]
        tr = pd.concat([x.high - x.low, (x.high - x.close.shift()).abs(),
                        (x.low - x.close.shift()).abs()], axis=1).max(axis=1)
        # First bar has no preceding close in this uninterrupted segment.
        tr.iloc[0] = np.nan
        f.loc[ids, 'atr'] = tr.rolling(14).mean()
        for minutes in (60, 240):
            vol = x.volume.rolling(minutes).sum()
            mean = x.pv.rolling(minutes).sum() / vol.replace(0, np.nan)
            var = x.p2v.rolling(minutes).sum() / vol.replace(0, np.nan) - mean ** 2
            f.loc[ids, 'vwap' + str(minutes)] = 4000. + mean
            f.loc[ids, 'sigma' + str(minutes)] = np.sqrt(var.clip(lower=0))
            f.loc[ids, 'slope' + str(minutes)] = mean - mean.shift(15)
    f['risk'] = f.atr.clip(1.2, 6.)
    f['scale_known'] = valid & f.risk.notna() & f.bb1.notna() & f.bb1_half.notna()
    return f


def quote_context(quotes, context):
    qt = quotes.available_ns.to_numpy(np.int64)
    av = context.available_ns.to_numpy(np.int64)
    ids = av.searchsorted(qt, side='left') - 1
    safe = np.maximum(ids, 0)
    good = ((ids >= 0) & context.scale_known.to_numpy(bool)[safe]
            & ((qt - context.t_ns.to_numpy(np.int64)[safe]) <= 120 * NS)
            & (context.epoch.to_numpy(np.int64)[safe] == quotes.epoch.to_numpy(np.int64)))
    risk = np.where(good, context.risk.to_numpy(float)[safe], np.nan)
    # Thresholds are exact nano-price integers (rounded upwards).
    nano = np.where(good, np.ceil(np.nan_to_num(risk) * NS), 0).astype(np.int64)
    return ids.astype(np.int64), nano


def frozen_levels(quotes, context, start_ns, end_ns):
    """A level is published at the block start from the preceding 30 minutes.

    There is no use of the current block's future occupancy. The mode is a
    midpoint time address, not a claim about orders or participants' motives.
    """
    clock = TimeAtPrice(quotes)
    av = context.available_ns.to_numpy(np.int64)
    rows = []
    for block, t in enumerate(range(start_ns, end_ns, 30 * MINUTE)):
        ci = int(av.searchsorted(t, side='left')) - 1
        if ci < 0:
            continue
        c = context.iloc[ci]
        context_known = bool(c.scale_known and 0 <= t - int(c.t_ns) <= 120 * NS)
        for kind, shift in [('tpo', 0.), ('shifted_tpo', 2.5)]:
            p = clock.facets(t - 30 * MINUTE, t, 12, 5, shift)
            left = int(quotes.available_ns.searchsorted(t - 30 * MINUTE, 'right')) - 1
            right = int(quotes.available_ns.searchsorted(t, 'left'))
            q = quotes.iloc[max(0, left):right]
            same_epoch = len(q) > 0 and q.epoch.nunique() == 1 and int(q.epoch.iloc[-1]) == int(c.epoch)
            known = bool(context_known and p['known'] and same_epoch)
            row = dict(block=block, kind=kind, start_ns=t, end_ns=min(t + 30 * MINUTE, end_ns),
                       known=known, level=p['mode_price'], risk=float(c.risk) if context_known else None,
                       context_index=ci, context_release_ns=int(c.available_ns), epoch=int(c.epoch),
                       coverage=p['coverage'], concentration=p['largest_bin_fraction'],
                       half_overlap=p['half_profile_overlap'],
                       time_center_migration=p['second_half_mean'] - p['first_half_mean']
                       if p['known'] else None,
                       profile_low=p['bin_low_nano'].tolist(),
                       profile_seconds=p['bin_seconds'].tolist())
            rows.append(row)
            if kind == 'tpo':
                control = dict(row, kind='grid', level=float(np.floor(p['mode_price'] + .5)) if p['mode_price'] is not None else None)
                rows.append(control)
    return rows


def coarse_texture(quotes, start, end):
    """One-second held samples distinguish a mesoscopic path from native jitter.

    Support is separately checked on all native publications, not only samples.
    """
    qt = quotes.available_ns.to_numpy(np.int64)
    if not len(qt) or start < qt[0] or end < start:
        return dict(texture_known=False)
    a = max(0, int(qt.searchsorted(start, 'right')) - 1)
    b = int(qt.searchsorted(end, 'right'))
    q = quotes.iloc[a:b]
    if q.empty:
        return dict(texture_known=False)
    good = (q.valid & q.epoch.ge(0) & q.bid_nano.gt(0) & q.ask_nano.gt(q.bid_nano))
    supported = bool(good.all() and q.epoch.nunique() == 1
                     and q.available_ns.diff().dropna().le(30 * NS).all()
                     and 0 <= start - int(q.available_ns.iloc[0]) <= 30 * NS
                     and end - int(q.available_ns.iloc[-1]) <= 30 * NS)
    grid = np.unique(np.r_[start, np.arange(start + NS, end, NS, dtype=np.int64), end])
    ids = qt.searchsorted(grid, 'right') - 1
    mid = (quotes.bid_nano.to_numpy(np.int64)[ids] + quotes.ask_nano.to_numpy(np.int64)[ids]) / (2 * NS)
    path = float(np.abs(np.diff(mid)).sum())
    native_mid = (q.bid_nano.to_numpy(np.int64) + q.ask_nano.to_numpy(np.int64)) / (2 * NS)
    return dict(texture_known=supported, path_1s=path,
                efficiency_1s=float(abs(mid[-1] - mid[0]) / path) if path else 0.,
                native_path=float(np.abs(np.diff(native_mid)).sum()))
