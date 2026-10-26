"""The one place where model input X meets the future Y: joined by time, kept apart."""
import numpy as np
from spectrum.features.contract import FeatureBatch
from spectrum.leakage import FutureFrame, FutureLeak, export_future_values, future_join


@future_join
def future_examples(x, y, cutoff_ns=None, max_x_age_ns=None):
    """Pair each future row with the X published by its origin.

    x is a FeatureBatch; y is a FutureFrame with t_ns (origin) and
    future_maturity_ns. The X row is the latest endpoint already published at
    the origin (available_ns <= origin; as-of on the publication clock, never
    on t_ns; late publications are allowed). With max_x_age_ns an older X
    (origin - its t_ns) leaves the pair unusable. A pair is usable when that
    X exists and the answer matured strictly after the origin; with cutoff_ns
    (e.g. the start of a test period) answers maturing later are dropped, so
    no training label overlaps the period it is tested on.
    Returns dict(x=values, known, names, x_available_ns, y=FutureFrame, usable).
    """
    if not isinstance(x, FeatureBatch):
        raise TypeError('X must be a FeatureBatch')
    if not isinstance(y, FutureFrame) or 'future_maturity_ns' not in y or 't_ns' not in y:
        raise TypeError('Y must be a marked FutureFrame with t_ns and future_maturity_ns')
    available = np.asarray(x.available_ns)
    origin = y.t_ns.to_numpy(np.int64)
    maturity = export_future_values(y.future_maturity_ns).astype(np.int64)
    if not len(available):
        return dict(x=np.zeros((len(y), len(x.names)), dtype=x.values.dtype),
                    known=np.zeros((len(y), len(x.names)), bool), names=x.names,
                    x_available_ns=np.full(len(y), -1, np.int64),
                    y=y.reset_index(drop=True), usable=np.zeros(len(y), bool))
    # rows by publication; the latest endpoint among those published so far
    order = np.argsort(available, kind='stable')
    latest = np.maximum.accumulate(order)
    position = np.searchsorted(available[order], origin, side='right') - 1
    has_x = position >= 0
    safe = latest[np.maximum(position, 0)]
    if np.any(available[safe][has_x] > origin[has_x]):
        raise FutureLeak('X published after its origin')
    usable = has_x & (maturity > origin)
    if max_x_age_ns is not None:
        usable &= origin - np.asarray(x.t_ns)[safe] <= int(max_x_age_ns)
    if cutoff_ns is not None:
        usable &= maturity <= int(cutoff_ns)
    values = np.where(has_x[:, None], x.values[safe], 0.)
    known = has_x[:, None] & x.known[safe]
    return dict(x=values, known=known, names=x.names, x_available_ns=np.where(has_x, available[safe], -1),
                y=y.reset_index(drop=True), usable=usable)
