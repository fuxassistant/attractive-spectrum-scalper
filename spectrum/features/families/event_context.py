"""Reusable formulas extracted from spectrum/cognition/signatures.py; see docs/MIGRATION.json."""
import numpy as np
from spectrum.core.schema import NS
EVENT_LOOKBACK_SECONDS = 300

def event_context(intent, event_times, bank_known=True):
    """Strictly earlier recognised events only; the origin's own event is excluded.

    Without a certified event job for the day, counts are masked, not a known zero.
    """
    intent = np.asarray(intent, np.int64); event_times = np.asarray(event_times, np.int64)
    j = np.searchsorted(event_times, intent, side='left')
    lo = np.searchsorted(event_times, intent - EVENT_LOOKBACK_SECONDS * NS, side='left')
    previous = event_times[np.clip(j - 1, 0, None)] if len(event_times) else np.zeros(len(intent), np.int64)
    known = (j > 0) & bank_known
    return dict(prior_events_300s=np.where(bank_known, j - lo, 0).astype(float),
                since_prior_event_s=np.where(known, (intent - previous) / NS, 0.), since_prior_event_known=known,
                event_bank_known=np.full(len(intent), bool(bank_known)))
