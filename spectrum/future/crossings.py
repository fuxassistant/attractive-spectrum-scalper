"""Future band crossings seen from each origin. Everything here is future.

The crossing bank itself is past: each crossing is a fact at its own t_ns.
Looking forward from an origin t, the fact for every line (scale x
lower/middle/upper) is the first bank crossing strictly after t within the
same band segment, or the reason it is unknown: HORIZON (bands lasted past
t + horizon) or SUPPORT_END (bands ended first). NOT_READY means no band at t.
future_maturity_ns is the moment the answer became past.

Every future column is named future_*; model input rejects such names.
"""
import numpy as np
import pandas as pd
from spectrum.crossings.bank import ROLES, SCALES
from spectrum.units import MINUTE
from spectrum.leakage import future_output

FUTURE_STATUSES = ('NOT_READY', 'CROSSED', 'HORIZON', 'SUPPORT_END')


@future_output()
def future_support_end(t, support, coverage_end_ns):
    """End of each band segment (known only after it ends), capped by a 60 s quote silence."""
    return _support_end(t, support, coverage_end_ns)


def _support_end(t, support, coverage_end_ns):
    out = np.zeros(len(t), np.int64)
    bounds = np.r_[0, np.flatnonzero(support[1:] != support[:-1]) + 1, len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if support[a] < 0:
            continue
        out[a:b] = min(int(t[b]) if b < len(t) else int(coverage_end_ns), int(t[b-1]) + MINUTE)
    return out


ORIGIN_KEYS = ('t_ns', 'scale', 'role', 'origin_bid', 'origin_ask')


@future_output(keys=ORIGIN_KEYS)
def future_crossings(bank, tape, origins_ns, horizon_ns, coverage_end_ns):
    """One row per origin and line; bank, tape come from crossings.build_bank or open_bank.

    Pass bank[bank.opportunity] (or bank[bank.line_visit]) to ask for the next
    new opportunity of each line instead of the next crossing, flicker included.
    That answers "when is the next setup", not "when is the line touched": an
    earlier raw touch may exist, so price exits and fills on the full bank.
    """
    t = np.asarray(origins_ns, np.int64)
    j = np.searchsorted(tape['t'], t, side='right') - 1
    if (j < 0).any():
        raise ValueError('Origin precedes the quote source')
    limit = t + int(horizon_ns)
    frames = []
    for scale in SCALES:
        support = tape['support'][scale]
        ready = (support[j] >= 0) & (t - tape['t'][j] <= MINUTE)
        end = _support_end(tape['t'], support, coverage_end_ns)[j]
        for role in range(3):
            events = bank[(bank.scale == scale) & (bank.role == role)]
            event_t = events.t_ns.to_numpy(np.int64)
            ix = np.searchsorted(event_t, t, side='right')
            at = np.minimum(ix, max(len(events) - 1, 0))
            found = ready & (ix < len(events))
            if len(events):
                found &= (events.support.to_numpy()[at] == support[j]) & (event_t[at] <= end) & (event_t[at] <= limit)
            status = np.where(~ready, 0, np.where(found, 1, np.where(end > limit, 2, 3)))

            def pick(column, empty):
                if not len(events):
                    return np.full(len(t), empty)
                return np.where(found, events[column].to_numpy()[at], empty)

            cross_ns = pick('t_ns', 0)
            frames.append(pd.DataFrame(dict(
                t_ns=t, scale=scale, role=ROLES[role],
                origin_bid=tape['bid'][j], origin_ask=tape['ask'][j],
                future_status=np.asarray(FUTURE_STATUSES)[status], future_cross_ns=cross_ns,
                future_ordinal=pick('ordinal', -1), future_direction=pick('direction', 0),
                future_event_id=pick('event_id', ''), future_line=pick('line', np.nan),
                future_bid=pick('bid', np.nan), future_ask=pick('ask', np.nan),
                future_maturity_ns=np.select([status == 1, status == 2, status == 3], [cross_ns, limit, end], 0))))
    return pd.concat(frames, ignore_index=True).sort_values(['t_ns', 'scale', 'role'], ignore_index=True)
