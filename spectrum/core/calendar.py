"""Declared operational entry support for the bounded August 2026 GC pilot.

This is not a general exchange holiday calendar. The fixed source capture is
weekdays 2026-08-03 through 2026-08-28, with an intentional epoch reset at each
UTC midnight. CME GC regular hours use America/Chicago: Sunday 17:00 to Friday
16:00, with the daily 16:00-17:00 break. The inspected CME 2026 U.S. holiday list
has no holiday window within these pilot dates.

One common 31-minute reserve serves all model/target arms: maximum holding time
30 minutes + entry wait <=2 seconds + stressed exit latency .5 seconds + up to
30 seconds until the next supported quote. Unexpected support loss still censors
and halts execution; this calendar never inspects future observations or labels.

Sources:
https://www.cmegroup.com/market-regulation/files/gold-futures-and-options-fact-card.pdf
https://www.cmegroup.com/trading-hours.html
"""
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR
RESERVE_NS = 31 * MINUTE
MONTH_START_NS = int(pd.Timestamp('2026-08-01', tz='UTC').value)
MONTH_END_NS = int(pd.Timestamp('2026-09-01', tz='UTC').value)
CAPTURE_START_NS = int(pd.Timestamp('2026-08-03', tz='UTC').value)
CAPTURE_END_NS = int(pd.Timestamp('2026-08-29', tz='UTC').value)


def operational_gate(origin_ns, phase_end_ns):
    """Return a causal boolean entry mask, independent of risk, Y and quote tape.

    Inputs may include any August timestamps; non-captured days return false.
    Timestamps outside August are rejected, to prevent treating this bounded
    convention as verified holiday/session support for another period.
    """
    origins = np.asarray(origin_ns, dtype=np.int64)
    if origins.ndim != 1:
        raise ValueError('Operational origins must be a one-dimensional ns array')
    end = int(phase_end_ns)
    if not MONTH_START_NS < end <= MONTH_END_NS:
        raise ValueError('Operational calendar supports August 2026 phase ends only')
    if np.any((origins < MONTH_START_NS) | (origins >= MONTH_END_NS)):
        raise ValueError('Operational calendar supports August 2026 origins only')
    if not len(origins):
        return np.empty(0, bool)
    utc = pd.to_datetime(origins, unit='ns', utc=True)
    chicago = utc.tz_convert('America/Chicago')
    local_midnight = chicago.normalize().asi8
    tod = origins - local_midnight
    dow = chicago.dayofweek.to_numpy()
    exchange_open = (((dow <= 3) & ((tod < 16*HOUR) | (tod >= 17*HOUR))) |
                     ((dow == 4) & (tod < 16*HOUR)) |
                     ((dow == 6) & (tod >= 17*HOUR)))
    capture_open = ((origins >= CAPTURE_START_NS) & (origins < CAPTURE_END_NS) &
                    (utc.dayofweek.to_numpy() < 5))
    next_close = local_midnight + 16*HOUR + np.where(tod >= 17*HOUR, DAY, 0)
    next_capture_boundary = utc.normalize().asi8 + DAY
    deadline = np.minimum(np.minimum(next_close, next_capture_boundary), end)
    return exchange_open & capture_open & (deadline - origins >= RESERVE_NS)
