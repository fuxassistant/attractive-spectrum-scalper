"""Same bounded-state feature machine for replay, warmup and live input."""
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
import math
import numpy as np
from .schema import NS, MINUTE
from .price_history import PriceHistory, NAN
from .feature_registry import Registry
from .feature_families import WINDOWS, default_families


@dataclass(frozen=True)
class Window:
    supported: bool
    progress: float
    left: object
    right: object

    def delta(self, name):
        if not self.supported:
            return NAN
        value = self.right.get('c_'+name, NAN) - self.left.get('c_'+name, NAN)
        if math.isfinite(value) and value < -1e-7:
            raise ValueError('cumulative counter regressed: '+name)
        return max(value, 0.) if math.isfinite(value) else NAN


@dataclass(frozen=True)
class Context:
    t_ns: int
    available_ns: int
    tick_size: float
    scale: float
    price_ready: bool
    bar: object
    price: object
    snapshot: object
    snapshot_valid: bool
    quote_age_seconds: float
    windows: tuple
    observations: object

    def current(self, name):
        return self.snapshot.get(name, NAN)

    def window(self, seconds):
        return self.observations[seconds]


@dataclass(frozen=True)
class FeatureFrame:
    t_ns: int
    available_ns: int
    epoch: int
    bar_valid: bool
    price_ready: bool
    snapshot_valid: bool
    snap_row: int
    scale: float
    values: np.ndarray

    def origin(self):
        return {name: getattr(self, name) for name in (
            't_ns', 'available_ns', 'epoch', 'bar_valid', 'price_ready', 'snapshot_valid', 'snap_row', 'scale')}


class FeatureStream:
    def __init__(self, *, tick_size=.1, snapshot_ns=2*NS, windows=WINDOWS, families=None):
        if not math.isfinite(tick_size) or tick_size <= 0 or snapshot_ns <= 0:
            raise ValueError('Invalid feature price/clock unit')
        self.windows = tuple(windows)
        if not self.windows or len(set(self.windows)) != len(self.windows) or any(
                type(w) is not int or w <= 0 or (w*NS) % snapshot_ns for w in self.windows):
            raise ValueError('Window endpoints must align to the snapshot clock')
        self.tick_size, self.snapshot_ns = tick_size, snapshot_ns
        self.registry = Registry(default_families(self.windows) if families is None else families)
        self.history = PriceHistory(tick_size)
        self.snapshots = deque()
        self.snapshot_count = 0
        self.last_bar_ns = None
        self.failed = False

    def push_snapshot(self, snapshot):
        if self.failed:
            raise RuntimeError('Feature stream failed; restore before continuing')
        snapshot = dict(snapshot)
        t = int(snapshot['t_ns'])
        if self.snapshots and t <= self.snapshots[-1]['t_ns']:
            raise ValueError('Snapshots must increase strictly')
        if self.last_bar_ns is not None and t <= self.last_bar_ns:
            raise ValueError('Late snapshot would revise a published feature frame')
        snapshot['available_ns'] = int(snapshot.get('available_ns', t))
        if snapshot['available_ns'] < t:
            raise ValueError('Snapshot cannot be available before its close')
        self.snapshots.append(snapshot)
        self.snapshot_count += 1
        cutoff = t - max(self.windows)*NS - self.snapshot_ns
        while len(self.snapshots) > 1 and self.snapshots[0]['t_ns'] < cutoff:
            self.snapshots.popleft()

    def push_bar(self, raw_bar):
        if self.failed:
            raise RuntimeError('Feature stream failed; restore before continuing')
        bar = dict(raw_bar)
        t, epoch = int(bar['t_ns']), int(bar['epoch'])
        available = int(bar.get('available_ns', t))
        if t % MINUTE or (self.last_bar_ns is not None and t <= self.last_bar_ns):
            raise ValueError('Finalized M1 bars must increase on minute boundaries')
        if available < t or not 0 <= bar['coverage'] <= 1:
            raise ValueError('Invalid bar availability/coverage')
        if self.snapshots and self.snapshots[-1]['t_ns'] > t:
            raise ValueError('Consume each bar before snapshots from its future')
        bar['valid'] = bool(bar['valid'] and all(math.isfinite(bar[k]) for k in ('o','h','l','c')))
        current = self.snapshots[-1] if self.snapshots else {}
        if current.get('available_ns', t) > available:
            raise ValueError('Bar published before its snapshot dependency')
        price, scale, ready = self.history.update(bar)
        supported = bool(current and current['valid'] and current['epoch'] == epoch and t-current['t_ns'] <= self.snapshot_ns)
        windows = {}
        points = {point['t_ns']: point for point in self.snapshots}
        for seconds in self.windows:
            start = t-seconds*NS
            left = points.get(start, {})
            good = supported and current.get('t_ns') == t and bool(left)
            if good:
                good = all(at in points and points[at]['valid'] and points[at]['epoch'] == epoch
                           for at in range(start, t+1, self.snapshot_ns))
            progress = (current.get('mid', NAN)-left.get('mid', NAN))/self.tick_size if good else NAN
            windows[seconds] = Window(good, progress, MappingProxyType(left), MappingProxyType(current))
        age = (current['age_ns']+t-current['t_ns'])/NS if current else NAN
        context = Context(t, available, self.tick_size, scale, ready, MappingProxyType(bar),
                          MappingProxyType(price), MappingProxyType(current),
                          supported, age, self.windows, MappingProxyType(windows))
        try:
            values = self.registry.compute(context)
        except Exception:
            self.failed = True
            raise
        self.last_bar_ns = t
        return FeatureFrame(t, available, epoch, bar['valid'], ready, supported,
                            self.snapshot_count-1, scale, values)

    def manifest(self, rows=0):
        accumulated, arms = [], {}
        for family, indices in self.registry.indices.items():
            accumulated = accumulated + indices
            arms[family] = list(accumulated)
        return dict(contract='MBO_STREAM_FEATURES_V2', feature_names=self.registry.names,
                    families=self.registry.indices, arms=arms, fields=self.registry.specs,
                    registry=self.registry.contract, rows=rows, columns=len(self.registry.names),
                    tick_size=self.tick_size, snapshot_ns=self.snapshot_ns, windows_seconds=self.windows,
                    missing='zero plus a separate known mask; no origin deletion',
                    availability='available_ns is publication watermark, t_ns is bucket close',
                    history='bounded snapshots, 61 M1 closes, 60 TR, 60 native closes, day/week/session summaries',
                    y_access=False)

    def state_dict(self):
        if self.failed:
            raise RuntimeError('Cannot checkpoint failed feature state')
        return dict(tick_size=self.tick_size, snapshot_ns=self.snapshot_ns, windows=list(self.windows),
                    registry=self.registry.contract, snapshots=list(self.snapshots),
                    snapshot_count=self.snapshot_count, last_bar_ns=self.last_bar_ns, history=self.history.state_dict())

    def load_state_dict(self, state):
        if (state['tick_size'] != self.tick_size or state['snapshot_ns'] != self.snapshot_ns or
            state['windows'] != list(self.windows) or state['registry'] != self.registry.contract):
            raise ValueError('Feature checkpoint schema/configuration drift')
        self.snapshots = deque(state['snapshots'])
        self.snapshot_count, self.last_bar_ns = state['snapshot_count'], state['last_bar_ns']
        self.history = PriceHistory(self.tick_size)
        self.history.load_state_dict(state['history'])
        self.failed = False
