"""Dataframe convenience adapter; calculations live only in FeatureStream."""
import numpy as np
import pandas as pd
from .schema import NS, MINUTE
from .feature_stream import FeatureStream
from .feature_families import WINDOWS


def _ordered(frame, required, name):
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    times = frame.t_ns.to_numpy(np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be nonempty and strictly ordered")


def rows(frame):
    names = tuple(frame.columns)
    for values in frame.itertuples(index=False, name=None):
        yield dict(zip(names, values))


def build_features(snapshots, bars, tick_size=.1, *, snapshot_ns=2*NS, families=None):
    """Replay snapshot/bar history through the live feature implementation.

    Legacy frames without available_ns explicitly assume availability at close;
    newly reconstructed frames carry the actual publication watermark instead.
    Names, masks and numeric output are identical to FeatureStream for the same
    input sequence. No target or model is imported here.
    """
    _ordered(snapshots, ('t_ns','epoch','valid','age_ns','bid','ask','mid'), 'snapshots')
    _ordered(bars, ('t_ns','o','h','l','c','volume','coverage','epoch','valid'), 'bars')
    stream = FeatureStream(tick_size=tick_size, snapshot_ns=snapshot_ns, families=families)
    source = iter(rows(snapshots))
    current = next(source, None)
    origins, values = [], []
    for bar in rows(bars):
        while current is not None and current['t_ns'] <= bar['t_ns']:
            stream.push_snapshot(current)
            current = next(source, None)
        frame = stream.push_bar(bar)
        origins.append(frame.origin())
        values.append(frame.values)
    manifest = stream.manifest(len(origins))
    manifest['input_availability'] = 'explicit' if 'available_ns' in bars and 'available_ns' in snapshots else 'legacy close-time assumption'
    return pd.DataFrame(origins), np.stack(values), manifest
