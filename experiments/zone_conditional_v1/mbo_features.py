"""Causal, global MBO context on frozen zone decisions.

These features do not locate activity at the remembered zone. Canonical seconds
retain global counters and current near-book depth; price-local sidecars are a
separate, incomplete-date descriptive source. T is counted once. The canonical
matched-F counter pools passive/aggressor/unknown roles, and its unmatched
residual includes ambiguous decrements. Neither is a cancellation-intent label.

For a query q, T is the latest fully published one-second endpoint. Counter
differences describe (T-30s,T], not the fractional interval (q-30s,q]. Both clocks
and the endpoint lag are returned. A historical window is frozen at confirmed
departure; later observations cannot repair its missingness.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

NS = 1_000_000_000
TICK_NANO = 100_000_000
COUNTERS = (
    "path_ticks", "up_ticks", "down_ticks", "trade_buy", "trade_sell",
    "trade_unknown", "trade_count", "add_bid", "add_ask", "remove_bid",
    "remove_ask", "matched_fill_remove_bid", "matched_fill_remove_ask",
    "post_trade_add_bid", "post_trade_add_ask", "event_count",
)
WINDOW_FEATURES = (
    "known", "trade_toward_log", "trade_away_log", "trade_unknown_log",
    "trade_total_log", "trade_imbalance", "trade_unknown_fraction",
    "trade_count_log", "event_count_log", "native_quote_count_log",
    "support_add_log", "opposition_add_log", "support_remove_log",
    "opposition_remove_log", "support_matched_fill_log",
    "opposition_matched_fill_log", "support_unmatched_remove_log",
    "opposition_unmatched_remove_log", "support_replenishment_log",
    "opposition_replenishment_log", "book_net_pressure",
    "matched_remove_fraction", "replenishment_fraction",
    "depth_1_toward_imbalance", "depth_4_toward_imbalance",
    "depth_12_toward_imbalance", "depth_12_total_log", "spread_ticks",
    "toward_progress_ticks", "path_ticks_log", "toward_path_efficiency",
    "endpoint_age_seconds",
)
# Matched generic activity controls belong in the baseline, not exclusively in
# the MBO ablation. The remaining fields add direction, flow composition or book.
BASE_CONTEXT_COLUMNS = tuple("mbo_now_" + name for name in (
    "known", "trade_total_log", "trade_count_log", "event_count_log",
    "native_quote_count_log", "path_ticks_log", "toward_progress_ticks",
    "toward_path_efficiency", "spread_ticks", "endpoint_age_seconds",
))
ALL_FEATURE_COLUMNS = tuple("mbo_" + tag + "_" + name
                            for tag in ("dep", "now") for name in WINDOW_FEATURES)
MBO_FEATURE_COLUMNS = tuple(n for n in ALL_FEATURE_COLUMNS if n not in BASE_CONTEXT_COLUMNS)


def _ns(value):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError("Clocks must be exact integer UTC nanoseconds")
    return int(value)


def _clock(frame, name):
    if name not in frame or not pd.api.types.is_integer_dtype(frame[name].dtype):
        raise TypeError(f"{name} must be an integer column")
    return frame[name].to_numpy(np.int64)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def _prefix(mask):
    return np.r_[0, np.cumsum(mask, dtype=np.int64)]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class GlobalMBOContext:
    """One verified route/day; callers verify source manifest/payload identities.

    Native quotes are required even when all sampled seconds look valid: a brief
    invalid publication or reset between snapshots must invalidate the window.
    ``coverage_end_ns`` is an explicit source bound, never inferred from future
    data. Route selection is checked at the decision, not at historical recall.
    """

    def __init__(self, seconds, quotes, *, selection_available_ns, coverage_end_ns,
                 window_seconds=30, max_quote_gap_seconds=30,
                 max_snapshot_age_seconds=2):
        self.selection = _ns(selection_available_ns)
        self.coverage_end = _ns(coverage_end_ns)
        if not isinstance(window_seconds, (int, np.integer)) or isinstance(window_seconds, bool) or window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer")
        if not np.isfinite([max_quote_gap_seconds, max_snapshot_age_seconds]).all() or min(max_quote_gap_seconds, max_snapshot_age_seconds) <= 0:
            raise ValueError("Support/age limits must be positive finite seconds")
        self.window_ns = int(window_seconds) * NS
        self.max_gap = int(max_quote_gap_seconds * NS)
        self.max_age = int(max_snapshot_age_seconds * NS)
        self.s = seconds.reset_index(drop=True)
        self.st = _clock(self.s, "t_ns")
        self.sa = _clock(self.s, "available_ns")
        self.se = _clock(self.s, "epoch")
        self.qt = _clock(quotes, "available_ns")
        self.qe = _clock(quotes, "epoch")
        if len(self.st) and ((np.diff(self.st) <= 0).any() or (np.diff(self.sa) < 0).any() or (self.sa < self.st).any()):
            raise ValueError("Seconds must have ascending physical and nondecreasing publication clocks")
        if len(self.qt) and (np.diff(self.qt) < 0).any():
            raise ValueError("Native publication clock regressed")
        self.counter = self.s[["c_" + n for n in COUNTERS]].to_numpy(float)
        sg = self.s.valid.to_numpy(bool) & self.s.book_valid.to_numpy(bool) & (self.se >= 0)
        sg &= np.isfinite(self.counter).all(axis=1) & (self.counter >= 0).all(axis=1)
        self.sbad = _prefix(~sg)
        self.sbreak = _prefix(np.r_[False, (np.diff(self.st) != NS) | (self.se[1:] != self.se[:-1])])
        self.cbreak = _prefix(np.r_[False, (np.diff(self.counter, axis=0) < -1e-8).any(axis=1)])
        qg = quotes.valid.to_numpy(bool) & (self.qe >= 0)
        qg &= (quotes.ask_nano.to_numpy(np.int64) > quotes.bid_nano.to_numpy(np.int64)) & (quotes.bid_nano.to_numpy(np.int64) > 0)
        self.qbad = _prefix(~qg)
        self.qbreak = _prefix(np.r_[False, (np.diff(self.qt) > self.max_gap) | (self.qe[1:] != self.qe[:-1])])
        self._cache = {}

    def window(self, query_ns, side):
        """Compute one past-only window; metadata is never a model feature."""
        query = _ns(query_ns)
        if side not in (-1, 1):
            raise ValueError("side must be +1 toward long or -1 toward short")
        key = query, int(side)
        if key in self._cache:
            return self._cache[key]
        x = {name: np.nan for name in WINDOW_FEATURES}
        x["known"] = 0.
        meta = dict(query_ns=query, window_start_ns=-1, window_end_ns=-1,
                    max_source_available_ns=-1, source_epoch=-1, supported_seconds=0.,
                    status="NO_PUBLISHED_ENDPOINT", source_scope="GLOBAL_ROUTE_ACTIVITY",
                    interval_convention="(physical_start,physical_end]", endpoint_age_seconds=np.nan,
                    native_quote_updates=np.nan, trade_total_qty=np.nan,
                    classified_trade_qty=np.nan, unknown_trade_qty=np.nan,
                    addition_qty=np.nan, removal_qty=np.nan, matched_fill_qty=np.nan,
                    replenishment_qty=np.nan, global_counter_deltas=None)

        def fail(reason):
            meta["status"] = reason
            self._cache[key] = x, meta
            return x, meta

        if query > self.coverage_end:
            return fail("AFTER_SOURCE_COVERAGE")
        b = int(np.searchsorted(self.sa, query, side="right") - 1)
        if b < 0:
            return fail("NO_PUBLISHED_ENDPOINT")
        end = int(self.st[b]); start = end - self.window_ns
        meta.update(window_start_ns=start, window_end_ns=end,
                    max_source_available_ns=int(self.sa[b]), source_epoch=int(self.se[b]),
                    endpoint_age_seconds=(query - end) / NS)
        if query - end > self.max_age:
            return fail("STALE_PHYSICAL_ENDPOINT")
        a = int(np.searchsorted(self.st, start))
        if a > b or self.st[a] != start:
            return fail("MISSING_LEFT_ENDPOINT")
        if self.sbad[b + 1] - self.sbad[a] or self.sbreak[b + 1] - self.sbreak[a + 1]:
            return fail("SECOND_SUPPORT_OR_CONTINUITY")
        if self.cbreak[b + 1] - self.cbreak[a + 1]:
            return fail("COUNTER_REGRESSION")
        qa = int(np.searchsorted(self.qt, start, side="right") - 1)
        qb = int(np.searchsorted(self.qt, query, side="right") - 1)
        if qa < 0 or qb < qa or query - self.qt[qb] > self.max_gap:
            return fail("NATIVE_SUPPORT_OR_AGE")
        if (self.qbad[qb + 1] - self.qbad[qa] or self.qbreak[qb + 1] - self.qbreak[qa + 1]
                or self.qe[qa] != self.se[a] or self.qe[qb] != self.se[b]):
            return fail("NATIVE_SUPPORT_OR_CONTINUITY")
        d = dict(zip(COUNTERS, self.counter[b] - self.counter[a]))
        for book_side in ("bid", "ask"):
            if d["matched_fill_remove_" + book_side] > d["remove_" + book_side] + 1e-8:
                return fail("MATCHED_EXCEEDS_REMOVAL")
            if d["post_trade_add_" + book_side] > d["add_" + book_side] + 1e-8:
                return fail("REPLENISHMENT_EXCEEDS_ADDITION")
        row = self.s.iloc[b]; previous = self.s.iloc[a]
        depth_values = [row[f"depth_{s}_{k}"] for s in ("bid", "ask") for k in (1, 4, 12)]
        if (not np.isfinite([row.mid, previous.mid, row.bid, row.ask, *depth_values]).all()
                or min(depth_values) < 0 or row.ask <= row.bid or row.bid <= 0 or previous.bid <= 0):
            return fail("INVALID_SNAPSHOT_GEOMETRY")
        toward, away = (d["trade_buy"], d["trade_sell"]) if side == 1 else (d["trade_sell"], d["trade_buy"])
        total = toward + away + d["trade_unknown"]
        native_count = int(np.searchsorted(self.qt, end, side="right") - np.searchsorted(self.qt, start, side="right"))
        x.update(known=1., trade_toward_log=np.log1p(toward), trade_away_log=np.log1p(away),
                 trade_unknown_log=np.log1p(d["trade_unknown"]), trade_total_log=np.log1p(total),
                 trade_imbalance=_ratio(toward-away, toward+away),
                 trade_unknown_fraction=_ratio(d["trade_unknown"], total),
                 trade_count_log=np.log1p(d["trade_count"]), event_count_log=np.log1p(d["event_count"]),
                 native_quote_count_log=np.log1p(native_count), endpoint_age_seconds=(query-end)/NS)
        support, opposition = ("bid", "ask") if side == 1 else ("ask", "bid")
        for label, book_side in (("support", support), ("opposition", opposition)):
            values = dict(add=d["add_"+book_side], remove=d["remove_"+book_side],
                          matched_fill=d["matched_fill_remove_"+book_side],
                          unmatched_remove=max(0., d["remove_"+book_side]-d["matched_fill_remove_"+book_side]),
                          replenishment=d["post_trade_add_"+book_side])
            x.update({label+"_"+n+"_log": np.log1p(v) for n, v in values.items()})
        adds = d["add_bid"]+d["add_ask"]; removes = d["remove_bid"]+d["remove_ask"]
        x["book_net_pressure"] = _ratio((d["add_"+support]-d["remove_"+support])-(d["add_"+opposition]-d["remove_"+opposition]), adds+removes)
        x["matched_remove_fraction"] = _ratio(d["matched_fill_remove_bid"]+d["matched_fill_remove_ask"], removes)
        x["replenishment_fraction"] = _ratio(d["post_trade_add_bid"]+d["post_trade_add_ask"], adds)
        for ticks in (1, 4, 12):
            bid, ask = row[f"depth_bid_{ticks}"], row[f"depth_ask_{ticks}"]
            x[f"depth_{ticks}_toward_imbalance"] = _ratio(side*(bid-ask), bid+ask)
        x["depth_12_total_log"] = np.log1p(row.depth_bid_12+row.depth_ask_12)
        x["spread_ticks"] = (row.ask-row.bid) / (TICK_NANO/NS)
        x["toward_progress_ticks"] = side*(row.mid-previous.mid) / (TICK_NANO/NS)
        x["path_ticks_log"] = np.log1p(d["path_ticks"])
        x["toward_path_efficiency"] = _ratio(x["toward_progress_ticks"], d["path_ticks"])
        meta.update(status="KNOWN", supported_seconds=self.window_ns/NS,
                    max_source_available_ns=max(int(self.sa[b]), int(self.qt[qb])),
                    native_quote_updates=float(native_count), trade_total_qty=total,
                    classified_trade_qty=toward+away, unknown_trade_qty=d["trade_unknown"],
                    addition_qty=adds, removal_qty=removes,
                    matched_fill_qty=d["matched_fill_remove_bid"]+d["matched_fill_remove_ask"],
                    replenishment_qty=d["post_trade_add_bid"]+d["post_trade_add_ask"],
                    global_counter_deltas=d)
        self._cache[key] = x, meta
        return x, meta

    def candidates(self, candidates):
        for field in ("t_ns", "episode_available_ns", "candidate_id"):
            _clock(candidates, field)
        if not candidates.side.isin([-1, 1]).all():
            raise ValueError("Candidates require direction +/-1")
        if (candidates.episode_available_ns > candidates.t_ns).any():
            raise ValueError("Historical departure cannot be later than decision")
        if (candidates.t_ns < self.selection).any():
            raise ValueError("Route is not available at candidate decision")
        rows = []
        for r in candidates.itertuples():
            output = {"candidate_id": int(r.candidate_id)}
            for tag, query in (("dep", r.episode_available_ns), ("now", r.t_ns)):
                x, meta = self.window(query, int(r.side))
                output.update({"mbo_"+tag+"_"+n: v for n, v in x.items()})
                output.update({"mbometa_"+tag+"_"+n: v for n, v in meta.items()})
            rows.append(output)
        return pd.DataFrame(rows, index=candidates.index) if rows else pd.DataFrame(columns=["candidate_id", *ALL_FEATURE_COLUMNS], index=candidates.index)


def global_mbo_features(candidates, seconds, quotes, *, selection_available_ns,
                        coverage_end_ns, window_seconds=30,
                        max_quote_gap_seconds=30, max_snapshot_age_seconds=2):
    """Return row-aligned global context; no labels or future prices are read."""
    return GlobalMBOContext(seconds, quotes, selection_available_ns=selection_available_ns,
                            coverage_end_ns=coverage_end_ns, window_seconds=window_seconds,
                            max_quote_gap_seconds=max_quote_gap_seconds,
                            max_snapshot_age_seconds=max_snapshot_age_seconds).candidates(candidates)

