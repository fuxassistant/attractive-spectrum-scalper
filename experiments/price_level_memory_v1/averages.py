"""Matched-publication-clock midpoint TWA and exact accepted-print VWAP.

This experimental diagnostic deliberately does not call a close-price SMA a
time integral, nor a close*volume proxy an exact trade VWAP. Windows are [L,R),
on ``available_ns``. The quote used at a trade is the last published quote at
or before its availability, possibly the final quote *after* that same event.
It must not be interpreted as the pre-aggressor quote or as execution alpha.

Callers prove source lineage and complete coverage. Explicit shared bounds are
required: a last observed print is not proof that a sparse stream is complete.
Unknown support is reported, never silently imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = 1_000_000_000


def weighted_decomposition(prices, weights):
    """Exact same-sample mean/variance identities with population conventions.

    weighted_mean - mean = Cov(price, weight) / mean(weight)
    weighted_var - var = Cov(weight, (price-mean)**2) / mean(weight)
                         - (weighted_mean-mean)**2.
    Zero covariance means equal centers, not necessarily equal widths.
    """
    p, w = np.asarray(prices, float), np.asarray(weights, float)
    if p.ndim != 1 or p.shape != w.shape or not len(p):
        raise ValueError("Nonempty aligned one-dimensional prices and weights required")
    if not np.isfinite(p).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Finite prices and nonnegative weights with positive total required")
    origin = p[0]
    z = p - origin
    mean_offset = z.mean()
    centered = z - mean_offset
    mean_weight = w.mean()
    covariance = np.mean(centered * (w - mean_weight))
    weighted_offset = np.dot(w, z) / w.sum()
    gap = weighted_offset - mean_offset
    variance = np.mean(centered ** 2)
    weighted_variance = np.dot(w, (z - weighted_offset) ** 2) / w.sum()
    variance_covariance = np.mean((w - mean_weight) * centered ** 2)
    return dict(mean=float(origin + mean_offset), weighted_mean=float(origin + weighted_offset),
                mean_weight=float(mean_weight), price_weight_covariance=float(covariance),
                center_gap=float(gap), covariance_center_gap=float(covariance / mean_weight),
                population_variance=float(variance), weighted_population_variance=float(weighted_variance),
                squared_deviation_weight_covariance=float(variance_covariance),
                variance_gap=float(weighted_variance - variance),
                covariance_variance_gap=float(variance_covariance / mean_weight - gap ** 2),
                effective_samples=float(w.sum() ** 2 / np.dot(w, w)))


def _clock(frame, name):
    if "available_ns" not in frame or frame.available_ns.dtype.kind not in "iu":
        raise ValueError(f"{name} requires integer-nanosecond available_ns")
    clock = frame.available_ns.to_numpy(np.int64)
    if len(clock) and (np.diff(clock) < 0).any():
        raise ValueError(f"{name} publication clock must be monotone")
    if "t_ns" in frame:
        if frame.t_ns.dtype.kind not in "iu" or (clock < frame.t_ns.to_numpy(np.int64)).any():
            raise ValueError(f"{name} availability cannot precede its integer source clock")
    return clock


def _prefix(x):
    return np.r_[0., np.cumsum(x, dtype=np.float64)]


def _count(points, left, right, *, include_left=True):
    return int(np.searchsorted(points, right, "left") - np.searchsorted(points, left, "left" if include_left else "right"))


def _variance(second, mean):
    variance = second - mean ** 2
    tolerance = 1e-9 * max(1., abs(second))
    return max(0., variance) if variance >= -tolerance else np.nan


def compare_windows(quotes, trades, query_ns, window_seconds=(1200, 3600),
                    max_gap_seconds=30, coverage_start_ns=None, coverage_end_ns=None):
    """Compare exact native midpoint-time and accepted-print statistics.

    Quote fields: available_ns, bid_nano, ask_nano, valid, epoch.
    Trade fields: available_ns, price_nano, quantity, supported, epoch.
    Optional t_ns fields are checked for availability ordering. Raw and quote
    epoch IDs need not share numbering; each stream must be continuous itself.
    All means/standard deviations are USD/oz, all windows use the same [L,R).
    One row is returned per query and requested window, ordered query/window.
    """
    q_required = {"available_ns", "bid_nano", "ask_nano", "valid", "epoch"}
    t_required = {"available_ns", "price_nano", "quantity", "supported", "epoch"}
    if not q_required.issubset(quotes) or not t_required.issubset(trades):
        raise ValueError("Missing native quote or accepted-T fields")
    query = np.asarray(query_ns)
    if query.ndim != 1 or query.dtype.kind not in "iu":
        raise ValueError("Queries must be one-dimensional integer nanoseconds")
    query = query.astype(np.int64, copy=False)
    windows = tuple(window_seconds)
    if not windows or any(not isinstance(w, (int, float)) or not np.isfinite(w) or w <= 0 for w in windows):
        raise ValueError("Positive finite windows in seconds required")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Positive finite quote-age bound required")
    qt_all, tt = _clock(quotes, "Quotes"), _clock(trades, "Trades")
    if not len(qt_all):
        raise ValueError("At least one published quote is needed")
    qgood_all = (quotes.valid.to_numpy(bool) & (quotes.epoch.to_numpy(np.int64) >= 0)
                 & (quotes.bid_nano.to_numpy(np.int64) > 0)
                 & (quotes.ask_nano.to_numpy(np.int64) > quotes.bid_nano.to_numpy(np.int64)))
    # A publication may carry several records: only its final state has a
    # positive duration, while any invalid record still invalidates support.
    keep = np.r_[np.flatnonzero(np.diff(qt_all) > 0), len(qt_all) - 1]
    qt = qt_all[keep]
    qgood = qgood_all[keep]
    qm = (quotes.bid_nano.to_numpy(np.int64)[keep] / 1e9 + quotes.ask_nano.to_numpy(np.int64)[keep] / 1e9) / 2
    origin = float(qm[np.flatnonzero(qgood)[0]]) if qgood.any() else 0.
    z = np.where(qgood, qm - origin, 0.)
    durations = np.diff(qt) / NS
    integrals = [_prefix(durations * value[:-1]) for value in (z, z * z)]
    qbad_times = qt_all[~qgood_all]
    qe = quotes.epoch.to_numpy(np.int64)
    qepoch_changes = qt_all[1:][qe[1:] != qe[:-1]]
    gap_ns = int(round(max_gap_seconds * NS))
    long_gaps = np.diff(qt) > gap_ns
    stale_starts = qt[:-1][long_gaps] + gap_ns
    stale_ends = qt[1:][long_gaps]

    def integral(at):
        j = int(np.searchsorted(qt, at, "right") - 1)
        if j < 0:
            return np.nan, np.nan
        tail = (int(at) - int(qt[j])) / NS
        return integrals[0][j] + z[j] * tail, integrals[1][j] + z[j] * z[j] * tail

    rawp = trades.price_nano.to_numpy(np.int64) / 1e9
    raww = trades.quantity.to_numpy(float)
    tgood = (trades.supported.to_numpy(bool) & (trades.epoch.to_numpy(np.int64) >= 0)
             & np.isfinite(raww) & (raww > 0) & np.isfinite(rawp) & (rawp > 0))
    te = trades.epoch.to_numpy(np.int64)
    tepoch_changes = tt[1:][te[1:] != te[:-1]]
    j = np.searchsorted(qt, tt, "right") - 1
    safe = np.maximum(j, 0)
    mid_known = (j >= 0) & qgood[safe] & ((tt - qt[safe]) <= gap_ns)
    mid_z = np.where(mid_known, qm[safe] - origin, 0.)
    p = np.where(tgood, rawp - origin, 0.)
    w = np.where(tgood, raww, 0.)
    sums = {name: _prefix(values) for name, values in dict(
        w=w, w2=w*w, p=p, p2=p*p, wp=w*p, wp2=w*p*p,
        mid=mid_z, wmid=w*mid_z, bad=(~tgood).astype(float), badmid=(~mid_known).astype(float)).items()}
    if coverage_start_ns is None:
        raise ValueError('Explicit verified coverage bounds are required')
    if coverage_end_ns is None:
        raise ValueError('Explicit verified coverage bounds are required')
    if not isinstance(coverage_start_ns, (int, np.integer)) or not isinstance(coverage_end_ns, (int, np.integer)):
        raise ValueError("Coverage bounds must be integer nanoseconds")
    rows = []
    for right in query:
        for width in windows:
            left = int(right) - int(round(width * NS))
            duration = (int(right) - left) / NS
            a, b = np.searchsorted(tt, [left, right], "left")
            s = {name: values[b] - values[a] for name, values in sums.items()}
            n = int(b - a)
            shared_coverage = coverage_start_ns <= left and right <= coverage_end_ns
            left_quote = int(np.searchsorted(qt, left, "right") - 1)
            stale = (np.searchsorted(stale_starts, right, "left") > np.searchsorted(stale_ends, left, "right")) or right > int(qt[-1]) + gap_ns
            quote_known = bool(shared_coverage and left_quote >= 0 and qgood[max(left_quote, 0)]
                               and not stale and not _count(qbad_times, left, right)
                               and not _count(qepoch_changes, left, right, include_left=False))
            trade_known = bool(shared_coverage and n > 0 and s["w"] > 0 and s["bad"] == 0
                               and not _count(tepoch_changes, left, right, include_left=False))
            row = dict(query_ns=int(right), left_ns=left, window_seconds=float(width),
                quote_known=quote_known, trade_known=trade_known, known=False,
                trade_count=n, trade_quantity=float(s["w"]), unsupported_prints=int(s["bad"]),
                missing_publication_mid_prints=int(s["badmid"]), coverage_known=bool(shared_coverage),
                quote_stale=bool(stale), twa_mid=np.nan, twa_mid_sigma_population=np.nan,
                vwap_trade=np.nan, vwap_trade_sigma_population=np.nan,
                equal_trade_price_mean=np.nan, equal_trade_price_sigma_population=np.nan,
                equal_trade_publication_mid=np.nan, volume_weighted_publication_mid=np.nan,
                trade_size_weighting_gap=np.nan, trade_price_basis_gap=np.nan, trade_timing_gap=np.nan,
                weighted_mid_gap=np.nan, volume_weighted_trade_basis_gap=np.nan,
                trade_price_size_covariance=np.nan, mean_trade_size=np.nan,
                decomposition_residual=np.nan, effective_trade_count=np.nan)
            if quote_known:
                x1, x2 = integral(left)
                y1, y2 = integral(right)
                mean = (y1 - x1) / duration
                var = _variance((y2 - x2) / duration, mean)
                row.update(twa_mid=origin + mean, twa_mid_sigma_population=float(np.sqrt(var)))
                if not np.isfinite(var):
                    row["quote_known"] = False
            if trade_known:
                mean, vw = s["p"] / n, s["wp"] / s["w"]
                var = _variance(s["p2"] / n, mean)
                vwvar = _variance(s["wp2"] / s["w"], vw)
                row.update(vwap_trade=origin + vw, vwap_trade_sigma_population=float(np.sqrt(vwvar)),
                    equal_trade_price_mean=origin + mean, equal_trade_price_sigma_population=float(np.sqrt(var)),
                    trade_size_weighting_gap=vw - mean,
                    trade_price_size_covariance=s["wp"] / n - mean * (s["w"] / n),
                    mean_trade_size=s["w"] / n, effective_trade_count=s["w"]**2 / s["w2"])
                if not np.isfinite(var) or not np.isfinite(vwvar):
                    row["trade_known"] = False
            known = bool(row["quote_known"] and row["trade_known"] and s["badmid"] == 0)
            if known:
                midpoint_mean = origin + s["mid"] / n
                weighted_mid = origin + s["wmid"] / s["w"]
                row.update(equal_trade_publication_mid=midpoint_mean,
                    volume_weighted_publication_mid=weighted_mid,
                    trade_price_basis_gap=row["equal_trade_price_mean"] - midpoint_mean,
                    trade_timing_gap=midpoint_mean - row["twa_mid"],
                    weighted_mid_gap=weighted_mid - row["twa_mid"],
                    volume_weighted_trade_basis_gap=row["vwap_trade"] - weighted_mid)
                row["decomposition_residual"] = (row["vwap_trade"] - row["twa_mid"]
                    - row["trade_size_weighting_gap"] - row["trade_price_basis_gap"] - row["trade_timing_gap"])
            row["known"] = known
            rows.append(row)
    return pd.DataFrame(rows)
