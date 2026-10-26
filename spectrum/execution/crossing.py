"""Causal first-line-crossing exits with a fixed gain and variable risk budget.

The decision risk is supplied by causal features. Future quotes determine only
the saved consequences, never admission at the origin. Every exit has the same
latency and support requirements as an entry. A support loss is censored rather
than assigned zero PnL. Prices and PnL are USD per reference ounce.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit

NS = 1_000_000_000
STATUSES = ('INELIGIBLE', 'EMPTY', 'CENSORED', 'COMPLETE')
REASONS = ('ORIGIN', 'SESSION', 'RISK', 'ENTRY_TIMEOUT', 'SUPPORT',
           'TAKE', 'STOP', 'TIME', 'EXIT_TIMEOUT', 'COVERAGE')


@dataclass(frozen=True)
class SimulationConfig:
    fee_roundtrip: float = .30
    latency_ms: float = 100.
    pending_timeout_seconds: float = 2.
    max_quote_gap_seconds: float = 60.
    deadline_seconds: float = 300.

    def __post_init__(self):
        numeric = np.array(list(self.__dict__.values()), float)
        if not np.isfinite(numeric).all() or self.fee_roundtrip < 0 or self.latency_ms < 0:
            raise ValueError('Finite nonnegative cost and latency required')
        if min(self.pending_timeout_seconds, self.max_quote_gap_seconds,
               self.deadline_seconds) <= 0:
            raise ValueError('Strictly positive duration and barrier scales required')
        if self.latency_ms / 1000 >= self.pending_timeout_seconds:
            raise ValueError('Fill timeout must exceed latency')

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, cls):
            return config
        values = {}
        for key in ('execution', 'event_query'):
            values.update({k: v for k, v in config.get(key, {}).items() if k in cls.__dataclass_fields__})
        values.update({k: v for k, v in config.items() if k in cls.__dataclass_fields__})
        return cls(**values)


@njit(cache=True)
def _good(j, bid, ask, valid, epoch):
    return valid[j] and epoch[j] >= 0 and bid[j] > 0 and ask[j] > bid[j]


@njit(cache=True)
def _supported_step(j, qt, bid, ask, valid, epoch, token, gap):
    return (_good(j, bid, ask, valid, epoch) and epoch[j] == token
            and (j == 0 or qt[j] - qt[j-1] <= gap))


@njit(cache=True)
def _net(side, bid, ask, e, j, fee):
    if side == 1:
        return (bid[j] - ask[e]) / 1e9 - fee
    return (bid[e] - ask[j]) / 1e9 - fee


@njit(cache=True)
def _simulate(qt, bid, ask, valid, epoch, coverage, start, session_end,
              origins, risks, take_ordinals, fee, latency, pending, gap, duration):
    n = len(origins)
    ints = np.full((2*n, 7), -1, np.int64)  # status, reason, decision, entry, trigger, exit, request_ns
    values = np.full((2*n, 3), np.nan)  # net, MAE, MFE through actual exit
    for i in range(n):
        t = origins[i]
        d = np.searchsorted(qt, t, side='right') - 1
        reason = 0
        if t < start or t + 2*pending + duration >= session_end:
            reason = 1
        elif not np.isfinite(risks[i]) or risks[i] <= 0:
            reason = 2
        elif d < 0 or not _good(d, bid, ask, valid, epoch) or t-qt[d] > gap:
            reason = 0
        else:
            reason = -1
        for si in range(2):
            r = 2*i + si
            ints[r, 0] = 0
            ints[r, 1] = reason
            ints[r, 2] = d
        if reason >= 0:
            continue
        e = np.searchsorted(qt, t+latency, side='right')
        status = 3
        entry_prefix = min(e, np.searchsorted(qt, t+pending, side='right')-1)
        for j in range(d+1, entry_prefix+1):
            if not _supported_step(j, qt, bid, ask, valid, epoch, epoch[d], gap):
                status = 2
                reason = 4
                break
        if status == 3 and (e == len(qt) or qt[e] > t+pending):
            status = 1
            reason = 3
            if coverage <= t+pending:
                status = 2
                reason = 9
            elif t+pending-qt[entry_prefix] > gap:
                status = 2
                reason = 4
        if status != 3:
            for si in range(2):
                ints[2*i+si, 0] = status
                ints[2*i+si, 1] = reason
            continue
        for si in range(2):
            r = 2*i + si
            side = 1 if si == 0 else -1
            ints[r, 0] = 2
            ints[r, 1] = 9
            ints[r, 3] = e
            low = np.inf
            high = -np.inf
            trigger = -1
            request_ns = -1
            last_scanned = e-1
            due = qt[e]+duration
            reason = 9
            for j in range(e, len(qt)):
                # A real clock timer fires even when no quote is published at
                # the deadline. Publications at the deadline cannot beat TIME.
                if qt[j] >= due:
                    request_ns = due
                    reason = 7
                    break
                if qt[j] >= session_end:
                    reason = 1
                    break
                if not _supported_step(j, qt, bid, ask, valid, epoch, epoch[e], gap):
                    reason = 4
                    break
                mark = _net(side, bid, ask, e, j, fee)
                low = min(low, mark)
                high = max(high, mark)
                last_scanned = j
                if mark <= -risks[i]:
                    trigger = j
                    request_ns = qt[j]
                    reason = 6
                    break
                if j == take_ordinals[i, si]:
                    trigger = j
                    request_ns = qt[j]
                    reason = 5
                    break
            if request_ns < 0 and reason == 9 and coverage > due and last_scanned >= e and due-qt[last_scanned] <= gap:
                request_ns = due
                reason = 7
            ints[r, 1] = reason
            if request_ns < 0:
                continue
            ints[r, 4] = trigger
            ints[r, 6] = request_ns
            x = np.searchsorted(qt, request_ns+latency, side='right')
            scan_to = min(x, np.searchsorted(qt, request_ns+pending, side='right')-1)
            supported = True
            for j in range(last_scanned+1, scan_to+1):
                if not _supported_step(j, qt, bid, ask, valid, epoch, epoch[e], gap):
                    supported = False
                    ints[r, 1] = 4
                    break
                mark = _net(side, bid, ask, e, j, fee)
                low = min(low, mark)
                high = max(high, mark)
            if not supported:
                continue
            if x == len(qt) or qt[x] > request_ns+pending:
                ints[r, 1] = 8 if coverage > request_ns+pending else 9
                if coverage > request_ns+pending and request_ns+pending-qt[scan_to] > gap:
                    ints[r, 1] = 4
                continue
            if qt[x] >= session_end:
                ints[r, 1] = 1
                continue
            ints[r, 0] = 3
            ints[r, 5] = x
            values[r, 0] = _net(side, bid, ask, e, x, fee)
            values[r, 1] = low
            values[r, 2] = high
    return ints, values


def simulate_tape(q, origins_ns, risk, take_ordinals, config):
    """Return paired independent fixed-plan trials; this is not an account curve.

    Exact publication ordinals disambiguate equal nanosecond timestamps. Causal
    admission uses only origin support, finite risk, and the known session clock.
    COMPLETE means observed execution under the stated taker quote assumption.
    No depth/queue/market-impact fill claim is made.
    """
    c = SimulationConfig.from_dict(config)
    t = np.asarray(origins_ns, np.int64)
    r = np.asarray(risk, float)
    if t.ndim != 1 or r.shape != t.shape:
        raise ValueError('One risk for each one-dimensional origin')
    take = np.asarray(take_ordinals, np.int64)
    if take.shape != (len(t), 2) or np.any(take < -1) or np.any(take >= len(q.quote_t)):
        raise ValueError('Two valid first qualifying crossing ordinals per origin required')
    expected_entry = np.searchsorted(q.quote_t, t+int(round(c.latency_ms*1e6)), side='right')
    if np.any((take >= 0) & (take <= expected_entry[:, None])):
        raise ValueError('A qualifying crossing must be strictly after the entry publication')
    qt = np.asarray(q.quote_t, np.int64)
    if len(qt) == 0 or np.any(qt[1:] < qt[:-1]):
        raise ValueError('Nonempty monotonically published quote tape required')
    z, v = _simulate(qt, q.bid, q.ask, q.valid, q.epoch, int(q.coverage_end_ns),
        int(q.entry_start_ns), int(q.deadline_ns), t, r, take, c.fee_roundtrip,
        int(round(c.latency_ms*1e6)), int(round(c.pending_timeout_seconds*NS)),
        int(round(c.max_quote_gap_seconds*NS)),
        int(round(c.deadline_seconds*NS)))
    out = pd.DataFrame(dict(t_ns=np.repeat(t, 2), side=np.tile([1, -1], len(t)),
        risk_R=np.repeat(r, 2), status=np.asarray(STATUSES)[z[:, 0]],
        reason=np.asarray(REASONS)[z[:, 1]], net=v[:, 0], mae=v[:, 1], mfe=v[:, 2]))
    for col, name in enumerate(('decision', 'entry', 'trigger', 'exit'), start=2):
        out[name+'_ordinal'] = z[:, col]
        out[name+'_ns'] = np.where(z[:, col] >= 0, qt[np.maximum(z[:, col], 0)], 0)
    out['trigger_ns'] = np.maximum(z[:, 6], 0)
    return out


@njit(cache=True)
def _entry_indices(qt, bid, ask, valid, epoch, coverage, start, deadline, origins, latency, pending, gap):
    """Entry alignment for factual targets; does not impose future hold maturity."""
    out = np.full((len(origins), 4), -1, np.int64)
    for i in range(len(origins)):
        t = origins[i]
        d = np.searchsorted(qt, t, side='right')-1
        out[i, 0] = 0
        out[i, 1] = 0
        out[i, 2] = d
        if t < start or t >= deadline:
            out[i, 1] = 1
            continue
        if d < 0 or not _good(d, bid, ask, valid, epoch) or t-qt[d] > gap:
            continue
        e = np.searchsorted(qt, t+latency, side='right')
        last = min(e, np.searchsorted(qt, t+pending, side='right')-1)
        supported = True
        for j in range(d+1, last+1):
            if not _supported_step(j, qt, bid, ask, valid, epoch, epoch[d], gap):
                supported = False
                break
        if not supported:
            out[i, 0] = 2
            out[i, 1] = 4
            continue
        if e == len(qt) or qt[e] > t+pending:
            out[i, 0] = 1
            out[i, 1] = 3
            if coverage <= t+pending:
                out[i, 0] = 2
                out[i, 1] = 9
            elif t+pending-qt[last] > gap:
                out[i, 0] = 2
                out[i, 1] = 4
            continue
        if qt[e] >= deadline:
            out[i, 1] = 1
            continue
        out[i, 0] = 3
        out[i, 1] = -1
        out[i, 3] = e
    return out


def align_entries(q, origins_ns, config):
    """Return causal/pending entry facts without conditioning on future labels."""
    c = SimulationConfig.from_dict(config)
    t = np.asarray(origins_ns, np.int64)
    z = _entry_indices(q.quote_t, q.bid, q.ask, q.valid, q.epoch,
        q.coverage_end_ns, q.entry_start_ns, q.deadline_ns, t,
        int(round(c.latency_ms*1e6)), int(round(c.pending_timeout_seconds*NS)),
        int(round(c.max_quote_gap_seconds*NS)))
    out = pd.DataFrame(dict(t_ns=t, entry_status=np.asarray(STATUSES)[z[:, 0]],
        entry_reason=np.where(z[:, 1] >= 0, np.asarray(REASONS)[np.maximum(z[:, 1], 0)], 'FILLED'),
        decision_ordinal=z[:, 2], entry_ordinal=z[:, 3]))
    for name, column in [('decision', 2), ('entry', 3)]:
        out[name+'_ns'] = np.where(z[:, column] >= 0, q.quote_t[np.maximum(z[:, column], 0)], 0)
    return out


def causal_first_qualifying(q, entry_ordinal, bank, gain, fee, origin_support, deadline_seconds=300):
    """Slow online seen-line implementation used to verify the precomputed query.

    Read each crossing in publication order, consume a line only at its first
    event in the entry's geometry support, and never inspect a later event to
    choose an earlier decision. Return the first qualifying ordinal per side.
    """
    found = np.full(2, -1, np.int64)
    if entry_ordinal < 0 or not np.isfinite(gain):
        return found
    seen = set()
    deadline = int(q.quote_t[entry_ordinal]+deadline_seconds*NS)
    future = bank.loc[(bank.ordinal > entry_ordinal) & (bank.t_ns <= deadline)]
    for row in future.sort_values(['ordinal', 'scale', 'role']).itertuples():
        line = (int(row.scale), int(row.role))
        if line in seen or int(row.support) != int(origin_support.get(int(row.scale), -1)):
            continue
        seen.add(line)
        marks = ((q.bid[row.ordinal]-q.ask[entry_ordinal])/1e9-fee,
                 (q.bid[entry_ordinal]-q.ask[row.ordinal])/1e9-fee)
        for si in range(2):
            if found[si] < 0 and marks[si] >= gain:
                found[si] = row.ordinal
    return found


def scalar_reference(q, origin_ns, risk, side, take_ordinal, config):
    """Independent NumPy full-quote benchmark for one trial, including censoring.

    This intentionally does not call the Numba kernel or its helper functions.
    """
    c = SimulationConfig.from_dict(config)
    t = int(origin_ns)
    qt = np.asarray(q.quote_t)
    pending = int(round(c.pending_timeout_seconds*NS))
    latency = int(round(c.latency_ms*1e6))
    gap = int(round(c.max_quote_gap_seconds*NS))
    duration = int(round(c.deadline_seconds*NS))
    d = int(np.searchsorted(qt, t, 'right')-1)
    out = dict(status='INELIGIBLE', reason='ORIGIN', decision_ordinal=d,
               entry_ordinal=-1, trigger_ordinal=-1, exit_ordinal=-1,
               trigger_ns=0,
               net=np.nan, mae=np.nan, mfe=np.nan)
    if t < q.entry_start_ns or t+2*pending+duration >= q.deadline_ns:
        out['reason'] = 'SESSION'
        return out
    if not np.isfinite(risk) or risk <= 0:
        out['reason'] = 'RISK'
        return out
    if d < 0 or not q.valid[d] or q.epoch[d] < 0 or q.bid[d] <= 0 or q.ask[d] <= q.bid[d] or t-qt[d] > gap:
        return out
    e = int(np.searchsorted(qt, t+latency, 'right'))
    entry_prefix = min(e+1, int(np.searchsorted(qt, t+pending, 'right')))
    entry_good = (q.valid[d:entry_prefix] & (q.epoch[d:entry_prefix] == q.epoch[d])
                  & (q.bid[d:entry_prefix] > 0) & (q.ask[d:entry_prefix] > q.bid[d:entry_prefix]))
    if not entry_good.all() or (np.diff(qt[d:entry_prefix]) > gap).any():
        out.update(status='CENSORED', reason='SUPPORT')
        return out
    if e == len(qt) or qt[e] > t+pending:
        out.update(status='EMPTY', reason='ENTRY_TIMEOUT')
        if q.coverage_end_ns <= t+pending:
            out.update(status='CENSORED', reason='COVERAGE')
        elif t+pending-qt[entry_prefix-1] > gap:
            out.update(status='CENSORED', reason='SUPPORT')
        return out
    due = int(qt[e]+duration)
    limit = min(len(qt), int(np.searchsorted(qt, due+pending, 'right'))+1)
    ix = np.arange(d, limit)
    good = (q.valid[ix] & (q.epoch[ix] == q.epoch[d]) & (q.bid[ix] > 0)
            & (q.ask[ix] > q.bid[ix]))
    good[1:] &= np.diff(qt[d:limit]) <= gap
    bad = np.flatnonzero(~good)
    end = int(d+bad[0]) if len(bad) else limit
    out.update(status='CENSORED', reason='SUPPORT')
    if e >= end:
        return out
    out['entry_ordinal'] = e
    ix = np.arange(e, end)
    mark = ((q.bid[ix]-q.ask[e])/1e9-c.fee_roundtrip if side == 1
            else (q.bid[e]-q.ask[ix])/1e9-c.fee_roundtrip)
    hits = np.flatnonzero((qt[ix] < due) & ((mark <= -risk) | (ix == take_ordinal)))
    if len(hits):
        h = int(hits[0])
        j = e+h
        request_ns = int(qt[j])
        reason = 'STOP' if mark[h] <= -risk else 'TAKE'
    else:
        if end < len(qt) and qt[end] < due:
            out['reason'] = 'SUPPORT'
            return out
        if end == len(qt) and (q.coverage_end_ns <= due or due-qt[end-1] > gap):
            out['reason'] = 'COVERAGE'
            return out
        j = -1
        request_ns = due
        reason = 'TIME'
    out.update(trigger_ordinal=j, trigger_ns=request_ns, reason=reason)
    x = int(np.searchsorted(qt, request_ns+latency, 'right'))
    pending_last = min(x, int(np.searchsorted(qt, request_ns+pending, 'right'))-1)
    if pending_last >= end:
        out['reason'] = 'SUPPORT'
        return out
    if x == len(qt) or qt[x] > request_ns+pending:
        out['reason'] = 'EXIT_TIMEOUT' if q.coverage_end_ns > request_ns+pending else 'COVERAGE'
        if q.coverage_end_ns > request_ns+pending and request_ns+pending-qt[pending_last] > gap:
            out['reason'] = 'SUPPORT'
        return out
    if qt[x] >= q.deadline_ns:
        out['reason'] = 'SESSION'
        return out
    if x >= end:
        out['reason'] = 'SUPPORT'
        return out
    path = mark[:x-e+1]
    out.update(status='COMPLETE', exit_ordinal=x, net=float(path[-1]),
               mae=float(path.min()), mfe=float(path.max()))
    return out
