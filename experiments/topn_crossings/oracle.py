"""Exact outcome representatives, with explicit exports of future-only research Y.

The max for a fixed exit is attained at minimum entry ask (LONG) or maximum
entry bid (SHORT). This avoids materializing every origin x line x side row.
Every retained quote origin is nevertheless considered; no early crossing is
skipped. Money is delegated to spectrum.economics.
"""
import numpy as np
import pandas as pd

from spectrum.crossings.bank import ROLES, SCALES
from spectrum.economics import future_crossing_marks
from spectrum.future.crossings import future_support_end, ORIGIN_KEYS
from spectrum.leakage import export_future_values, mark_future
from spectrum.market.calendar import MarketCalendar

NS = 1_000_000_000


def make_windows(t, bank, coverage_end, mode):
    """All windows are half-open; event boundaries never split a publication."""
    start, end = int(t[0]), int(coverage_end)
    if mode.startswith('time_'):
        minutes = int(mode.split('_')[1].removesuffix('m'))
        width = minutes * 60 * NS
        shift = 15 * 60 * NS if 'shift15' in mode else 0
        first = ((start - shift) // width + 1) * width + shift
        bounds = np.r_[start, np.arange(first, end, width, dtype=np.int64), end]
        partial = np.diff(bounds) < width
    else:
        kind, number = mode.split('_')
        count = int(number)
        events = bank if kind == 'events' else bank[bank.opportunity]
        # K distinct publication times, not K line aliases at the same tick.
        ticks = np.unique(events.t_ns.to_numpy(np.int64))
        ticks = ticks[(ticks >= start) & (ticks < end)]
        inner = ticks[count::count]
        bounds = np.unique(np.r_[start, inner, end]).astype(np.int64)
        partial = np.zeros(len(bounds)-1, bool)
        if len(partial):
            partial[-1] = len(ticks) % count != 0 or len(ticks) < count
    horizon_ns = 60 * 60 * NS if mode.endswith('_h60') else 0
    return pd.DataFrame(dict(window=np.arange(len(bounds)-1), start=bounds[:-1],
                             end=bounds[1:], partial=partial, horizon_ns=horizon_ns,
                             maturity_ns=np.minimum(bounds[1:]+horizon_ns, end)))


def prepare(tape, coverage_end, calendar=True):
    t = tape['t']
    valid = np.isfinite(tape['bid']) & np.isfinite(tape['ask']) & (tape['bid'] > 0) & (tape['ask'] > tape['bid'])
    if calendar:
        valid &= ~MarketCalendar().closed(t)
    ends = {s: export_future_values(future_support_end(t, tape['support'][s], coverage_end)) for s in SCALES}
    return valid, ends


def outcome_representatives(bank, tape, windows, valid, support_ends, fee=None):
    """Best LONG and SHORT origin for each (window, raw event), all quotes considered.

    Working tables are private oracle state; serialize via future_labels().
    Counters refer to origin-line pairs, before LONG/SHORT duplication.
    """
    t = tape['t']; bid = tape['bid']; ask = tape['ask']
    wid = np.searchsorted(windows.end.to_numpy(), t, side='right')
    in_window = wid < len(windows)
    w = np.minimum(wid, len(windows)-1)
    eligible = valid & in_window & (t >= windows.start.to_numpy()[w])
    end = windows.end.to_numpy()[w]
    horizon = windows.horizon_ns.to_numpy()[w] if 'horizon_ns' in windows else np.zeros(len(t),np.int64)
    limit = np.where(horizon > 0, t+horizon+1, end)
    counts = {key: np.zeros(len(windows), np.int64) for key in
              ('origins', 'queries', 'not_ready', 'crossed', 'window_censored', 'support_end')}
    counts['origins'] = np.bincount(w[eligible], minlength=len(windows))
    parts = []
    for scale in SCALES:
        support = tape['support'][scale]
        ready = eligible & (support >= 0)
        for role in range(3):
            events = bank[(bank.scale == scale) & (bank.role == role)].reset_index(drop=True)
            et = events.t_ns.to_numpy(np.int64)
            ix = np.searchsorted(et, t, side='right')
            safe = np.minimum(ix, max(len(et)-1, 0))
            found = ready & (ix < len(et))
            if len(et):
                found &= ((events.support.to_numpy()[safe] == support)
                          & (et[safe] <= support_ends[scale]) & (et[safe] < limit))
            masks = dict(queries=eligible, not_ready=eligible & ~ready, crossed=found,
                         window_censored=ready & ~found & (support_ends[scale] >= limit),
                         support_end=ready & ~found & (support_ends[scale] < limit))
            for key, mask in masks.items():
                counts[key] += np.bincount(w[mask], minlength=len(windows))
            indices = np.flatnonzero(found)
            if not len(indices):
                continue
            frame = pd.DataFrame(dict(window=w[indices], event=ix[indices], ask=ask[indices], bid=bid[indices]), index=indices)
            grouped = frame.groupby(['window', 'event'], sort=True)
            # idxmin/idxmax retain earliest origin among exact-price ties.
            li = grouped.ask.idxmin().to_numpy(np.int64)
            si = grouped.bid.idxmax().to_numpy(np.int64)
            group_keys = grouped.size()
            ww = group_keys.index.get_level_values(0).to_numpy(np.int64)
            ee = group_keys.index.get_level_values(1).to_numpy(np.int64)
            ev = events.iloc[ee]
            # Both candidate entry arrays are priced through the canonical economic layer.
            def money(origin):
                q = pd.DataFrame(dict(t_ns=t[origin], scale=scale, role=ROLES[role],
                    origin_bid=bid[origin], origin_ask=ask[origin], future_status='CROSSED',
                    future_bid=ev.bid.to_numpy(), future_ask=ev.ask.to_numpy()))
                marks = future_crossing_marks(mark_future(q, __name__, ORIGIN_KEYS), fee)
                return (export_future_values(marks.future_long_usd), export_future_values(marks.future_short_usd))
            long_net = money(li)[0]; short_net = money(si)[1]
            # Equal rounded score: earlier entry, then LONG.
            lr = np.round(long_net, 8); sr = np.round(short_net, 8)
            choose_long = (lr > sr) | ((lr == sr) & (t[li] <= t[si]))
            origin = np.where(choose_long, li, si)
            parts.append(pd.DataFrame(dict(window=ww, event_id=ev.event_id.to_numpy(),
                cluster_id=ev.cluster_event_id.to_numpy(), ordinal=ev.ordinal.to_numpy(),
                scale=scale, role=ROLES[role], side=np.where(choose_long, 'LONG', 'SHORT'),
                entry_ns=t[origin], exit_ns=ev.t_ns.to_numpy(), entry_bid=bid[origin], entry_ask=ask[origin],
                exit_bid=ev.bid.to_numpy(), exit_ask=ev.ask.to_numpy(), line=ev.line.to_numpy(),
                net=np.where(choose_long, long_net, short_net),
                best_long_net=long_net, best_short_net=short_net,
                best_long_entry_ns=t[li], best_short_entry_ns=t[si],
                candidate_origins=group_keys.to_numpy(), opportunity=ev.opportunity.to_numpy(),
                cluster_rank=ev.cluster_rank.to_numpy(), support=ev.support.to_numpy(),
                window_start=windows.start.to_numpy()[ww], window_end=windows.end.to_numpy()[ww],
                exit_limit_ns=limit[origin],
                maturity_ns=(windows.maturity_ns.to_numpy() if 'maturity_ns' in windows else windows.end.to_numpy())[ww])))
    summary = windows.copy()
    for key, value in counts.items():
        summary[key] = value
    if not parts:
        return pd.DataFrame(), summary
    result = pd.concat(parts, ignore_index=True)
    result['score'] = result.net.round(8)
    result = result.sort_values(['window', 'score', 'entry_ns', 'exit_ns', 'scale', 'role', 'side', 'event_id'],
                                ascending=[True, False, True, True, True, True, True, True], kind='stable')
    result['rank'] = 0
    positive = result.score > 0
    result.loc[positive, 'rank'] = result[positive].groupby('window').cumcount() + 1
    return result.reset_index(drop=True), summary


def summarize_window(rows, max_rank=32):
    positive = rows[rows.score > 0] if len(rows) else rows
    gains = positive.net.to_numpy() if len(positive) else np.array([])
    total = float(gains.sum())
    result = dict(outcomes=len(rows), positive_outcomes=len(positive), oracle_mass=total,
                  n50=int(np.searchsorted(np.cumsum(gains), total*.5)+1) if total else 0,
                  n80=int(np.searchsorted(np.cumsum(gains), total*.8)+1) if total else 0,
                  top1_share=float(gains[0]/total) if total else None,
                  top10_share=float(gains[:10].sum()/total) if total else None,
                  top10_flicker_fraction=None, top10_same_entry_fraction=None,
                  top10_same_exit_fraction=None, top10_hold_median_s=None)
    controls = {}
    if len(positive):
        controls['event'] = positive
        controls['physical_exit'] = positive.drop_duplicates('ordinal')
        controls['cluster'] = positive.drop_duplicates('cluster_id')
        controls['unique_entry_greedy'] = positive.drop_duplicates('entry_ns')
        kept = []
        for row in positive.itertuples():
            if all(row.exit_ns <= a or row.entry_ns >= b for a, b in kept):
                kept.append((row.entry_ns, row.exit_ns))
            if len(kept) == max_rank:
                break
        idx = [(int(r.entry_ns), int(r.exit_ns)) for r in positive.itertuples()]
        # Select exactly the greedily accepted rows, not aliases with the same interval.
        wanted = set(kept)
        controls['nonoverlap_greedy'] = positive.loc[[i in wanted for i in idx]].drop_duplicates(['entry_ns', 'exit_ns']).head(max_rank)
        top = positive.head(10)
        result.update(top10_flicker_fraction=float((~top.opportunity).mean()),
                      top10_same_entry_fraction=1-float(top.entry_ns.nunique()/len(top)),
                      top10_same_exit_fraction=1-float(top.ordinal.nunique()/len(top)),
                      top10_hold_median_s=float(((top.exit_ns-top.entry_ns)/NS).median()))
    curves = []
    for name in ('event', 'physical_exit', 'cluster', 'unique_entry_greedy', 'nonoverlap_greedy'):
        values = controls[name].net.to_numpy() if name in controls else np.array([])
        for k in range(1, max_rank+1):
            curves.append(dict(control=name, k=k, mass=float(values[:k].sum()),
                               selected=min(k, len(values)), coverage_of_event_mass=float(values[:k].sum()/total) if total else None))
    return result, curves


def future_labels(frame):
    """All saved oracle fields, including rank and chosen origin, are future Y."""
    return mark_future(frame.rename(columns={c:'future_'+c for c in frame.columns}), __name__)


def read_labels(path):
    return mark_future(pd.read_parquet(path), __name__)


def write_labels(frame, path):
    marked = future_labels(frame)
    # Serialization explicitly leaves the typed boundary; future_* names and
    # read_labels restore it. Do not export this table to an X builder.
    exported = pd.DataFrame({c:export_future_values(marked[c]) for c in marked.columns})
    exported.to_parquet(path, index=False)


def select_top_n(labels, n):
    """Independent event-top slice with relative economic weights, not confidence.

    This selector applies to one partition or combined partitions. Exact
    origin-event matching has its own non-nested k assignments instead.
    """
    if n < 0:
        raise ValueError('n must be nonnegative')
    if not hasattr(labels, 'future_source'):
        raise TypeError('Expected marked future labels from read_labels')
    raw=pd.DataFrame({c:export_future_values(labels[c]) for c in labels.columns})
    selected=raw[(raw.future_rank>0)&(raw.future_rank<=n)].copy()
    keys=[c for c in ('future_day','future_mode','future_window') if c in selected]
    denominator=selected.groupby(keys).future_net.transform('sum')
    selected['future_weight_within_top_n']=selected.future_net/denominator
    selected['future_top_n']=n
    return mark_future(selected,__name__)
