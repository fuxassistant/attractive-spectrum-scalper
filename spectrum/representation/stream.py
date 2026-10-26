"""Attach interaction facts to MboStream's existing book, including restart.

There is one order book. The collector borrows BookView only during a complete
publication callback. Physical buckets keep source time; delivery delays advance
available_ns and never backdate an observation.
"""
from collections import deque
from types import SimpleNamespace
import numpy as np
from spectrum.core import MboStream
from .raw import InteractionCollector


class _SideView:
    def __init__(self, book, side):
        self.book, self.side = book, side

    def get(self, price):
        quantity = self.book.level_quantity(self.side, price)
        return SimpleNamespace(size=quantity) if quantity else None

    def items(self):
        return ((price, SimpleNamespace(size=size)) for price, size, _ in self.book.levels(self.side))


class InteractionMetric:
    """MboStream extension; snapshots contain data-only collector state."""
    name = 'interaction_ledger'
    version = '1'
    fields = ('interaction_publications',)

    def __init__(self, emit_levels, emit_trades):
        self._clock = lambda: 0

        def levels(rows):
            for row in rows:
                row['available_ns'] = max(row['available_ns'], self._clock() or 0)
            emit_levels(rows)

        def trades(row):
            row['available_ns'] = max(row['available_ns'], self._clock() or 0)
            emit_trades(row)

        self.collector = InteractionCollector(levels, trades)

    def bind(self, stream):
        self._clock = lambda: stream.received_ns

    def update(self, event, book):
        borrowed = SimpleNamespace(bids=_SideView(book, 'B'), asks=_SideView(book, 'A'))
        self.collector.feed(event, borrowed)

    def values(self, t_ns):
        return (float(self.collector.stats['events']),)

    def reset(self):
        # The collector sees the same unsupported event and reconciles its
        # shadow stock there; a sampling support reset must not erase raw facts.
        pass

    def state_dict(self):
        c = self.collector
        return dict(bucket=c.bucket, stats=dict(c.stats),
                    rows=[[s, p, v.tolist()] for (s, p), v in c.rows.items()],
                    recent=[[s, p, list(v)] for (s, p), v in c.recent.items()],
                    birth=[[oid, s, p, t] for (oid, (s, p)), t in c.birth.items()],
                    shadow=[[s, p, q] for (s, p), q in c.shadow.items()])

    def load_state_dict(self, state):
        c = self.collector
        c.bucket, c.stats = state['bucket'], dict(state['stats'])
        c.rows = {(s, p): np.asarray(v, float) for s, p, v in state['rows']}
        c.recent = {(s, p): deque(tuple(x) for x in v) for s, p, v in state['recent']}
        c.birth = {(oid, (s, p)): t for oid, s, p, t in state['birth']}
        c.shadow = {(s, p): q for s, p, q in state['shadow']}


def interaction_stream(config, *, emit_levels, emit_trades, checkpoint=None, **callbacks):
    """Create/restore the shared MBO stream with the interaction family attached."""
    metric = InteractionMetric(emit_levels, emit_trades)
    if checkpoint is None:
        stream = MboStream(config, metrics=(metric,), features=False, **callbacks)
    else:
        stream = MboStream.restore(checkpoint, metrics=(metric,), expected_config=config, **callbacks)
    metric.bind(stream)
    return stream
