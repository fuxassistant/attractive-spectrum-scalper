"""Publish representation columns through the common FeatureBatch contract."""
import numpy as np
from spectrum.features.contract import FeatureBatch, reject_future


def to_feature_batch(frame, units, known=None):
    """Convert NaN-is-unknown tables into value + mask storage.

    Representation tables keep NaN for unknown and 0 for a measured zero. Here
    NaN becomes storage 0 with a false mask. ``known`` maps a column to a
    boolean column (e.g. mechanism_known) that can only narrow its mask.
    Infinite values are rejected rather than reinterpreted as unknown.
    """
    names = tuple(units)
    reject_future(frame.columns, frame)
    values, masks = {}, {}
    for name in names:
        value = frame[name].to_numpy(float)
        if np.isinf(value).any():
            raise ValueError('Infinite value in representation column ' + name)
        mask = ~np.isnan(value)
        if known and name in known:
            mask &= frame[known[name]].to_numpy(bool)
        values[name], masks[name] = np.where(mask, value, 0.), mask
    return FeatureBatch.from_columns(frame.t_ns.to_numpy(np.int64), frame.available_ns.to_numpy(np.int64),
                                     values, masks, units)
