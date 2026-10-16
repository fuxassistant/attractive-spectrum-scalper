"""All-event measurements and clock buckets; no files, dataframe or model."""
from dataclasses import asdict, replace
import numpy as np
from .schema import COUNTERS, DEPTH, NS, MINUTE, NANO, TICK
from .vendor.l3 import BookFragment, F_SNAPSHOT
from .extensions import BookView
from .event_facts import decode_event


def depth_vector(engine, book, undo=(), tick_nano=100_000_000):
    """Near-book geometry, optionally undoing an unpublished event group."""
    if not book.valid:
        return np.full(len(DEPTH), np.nan)
    corrections = {}
    for delta in undo:
        if delta.action not in ('A', 'C', 'M'):
            continue
        for side, price, quantity in (
            (delta.prior_side, delta.prior_price_nano, delta.prior_size),
            (delta.resulting_side, delta.resulting_price_nano, -delta.resulting_size),
        ):
            if quantity:
                key = side, price
                corrections[key] = corrections.get(key, 0) + quantity
    values, ladders = [], []
    for side, best in (('B', book.best_bid_nano), ('A', book.best_ask_nano)):
        levels = engine.bids if side == 'B' else engine.asks
        sign = -1 if side == 'B' else 1
        sizes = []
        for k in range(12):
            price = best + sign * k * tick_nano
            level = levels.get(price)
            sizes.append((0 if level is None else level.size) + corrections.get((side, price), 0))
        if min(sizes) < 0:
            raise ValueError('Negative reconstructed pre-event depth')
        values.extend(sum(sizes[:k]) for k in (1, 4, 12))
        ladders.append(sizes)
    for sizes in (ladders[1], ladders[0]):
        for quantity in (1, 5, 10):
            remaining, cost = quantity, 0.
            for tick, size in enumerate(sizes):
                filled = min(size, remaining)
                cost += filled * tick * tick_nano / NANO
                remaining -= filled
                if not remaining:
                    break
            values.append(cost / quantity if remaining == 0 else np.nan)
    return np.array(values)


class FlowCounters:
    """Observed price effort, T volume and order mutation facts in one epoch."""
    def __init__(self, tick_size=TICK):
        self.tick_size = tick_size
        self.values = np.zeros(len(COUNTERS))
        self.recent = {}

    def reset(self):
        self.values[:] = 0
        self.recent.clear()

    def update(self, event, previous, continuous):
        c, book, t = self.values, event.book, event.available_at_ns
        if book.valid and previous.valid and continuous:
            change = (book.midpoint_nano - previous.midpoint_nano) / (NANO * self.tick_size)
            c[:3] += abs(change), max(0, change), max(0, -change)
        facts = decode_event(event)
        hits = {}
        for trade in facts.trades:
            c[3 if trade.side == 'B' else 4 if trade.side == 'A' else 5] += trade.quantity
            c[6] += 1
        for m in facts.mutations:
            if not m.accepted:
                continue
            side = 0 if m.side == 'B' else 1
            if m.quantity < 0:
                c[9+side] += -m.quantity
                c[11+side] += m.linked_fill
                if m.linked_fill:
                    key = m.side, m.price_nano
                    hits[key] = hits.get(key, 0) + m.linked_fill
            else:
                c[7+side] += m.quantity
                key = m.side, m.price_nano
                budgets = [(stamp,q) for stamp,q in self.recent.get(key,())
                           if 0 <= t-stamp <= 2*NS and q>0]
                remaining=m.quantity;updated=[]
                for stamp,q in budgets:
                    restored=min(remaining,q) if t>stamp else 0
                    remaining-=restored;q-=restored
                    c[13+side]+=restored
                    if q:updated.append((stamp,q))
                self.recent[key]=updated
        # Capped matched-fill replenishment proxy; detailed gross/net recovery
        # is separately measured by PriceFlowWorld, with side and price retained.
        for key, quantity in hits.items():
            self.recent[key]=[(stamp,q) for stamp,q in self.recent.get(key,()) if 0<=t-stamp<=2*NS and q>0]+[(t,quantity)]
        c[15] += 1
        if len(self.recent) > 1000:
            self.recent = {k:[(stamp,q) for stamp,q in v if t-stamp<=2*NS and q>0]
                           for k,v in self.recent.items() if any(t-stamp<=2*NS and q>0 for stamp,q in v)}
        return facts



class MapAccumulator:
    """Exclusive watermark closes (start,end] buckets exactly once.

    Sinks have one append(tuple) method. Extra event metrics declare fields,
    update(event, engine), values(t_ns), reset(), state_dict/load_state_dict.
    They see only complete publications. Sampling occurs before the next update.
    """
    def __init__(self, quotes, snapshots, bars, fragment=0, period_ns=2*NS,
                 max_age_ns=30*NS, tick_nano=100_000_000, metrics=()):
        if period_ns <= 0 or MINUTE % period_ns or max_age_ns <= 0:
            raise ValueError('Invalid bucket/support clock')
        self.quotes, self.snapshots, self.bars = quotes, snapshots, bars
        self.fragment, self.period, self.max_age = fragment, period_ns, max_age_ns
        self.tick_nano, self.metrics = tick_nano, tuple(metrics)
        self.flow = FlowCounters(tick_nano / NANO)
        self.next_grid = self.cursor = self.last_ns = self.book = None
        self.epoch = fragment * 1_000_000
        self.bar = None
        self.reason_counts = {}
        self.last_source_epoch = None

    @property
    def c(self):
        return self.flow.values

    def _new_bar(self, t):
        supported = self.book is not None and self.book.valid and t - self.last_ns <= self.max_age
        price = self.book.midpoint_nano / NANO if supported else np.nan
        self.bar = [price, price, price, price, 0., self.epoch, True, float(self.c[3:6].sum())]

    def _integrate(self, t):
        if self.cursor is not None and t < self.cursor:
            raise ValueError('Watermark/receive-time regression')
        if self.cursor is not None and self.book is not None and self.book.valid:
            self.bar[4] += max(0, min(t, self.last_ns + self.max_age) - self.cursor)
        self.cursor = t

    def _snapshot(self, t, depth, available_ns):
        age, b = t - self.last_ns, self.book
        good = b.valid and age <= self.max_age
        prices = [b.best_bid_nano/NANO, b.best_ask_nano/NANO, b.midpoint_nano/NANO,
                  b.bid_size_l1, b.ask_size_l1] if good else [np.nan] * 5
        extras = []
        for metric in self.metrics:
            values = tuple(metric.values(t))
            if len(values) != len(metric.fields):
                raise ValueError('Event metric output differs from declared fields')
            extras.extend(values if good else [np.nan] * len(values))
        self.snapshots.append((t, self.epoch, good, b.valid, age, *prices,
                               *(depth if good else [np.nan] * len(DEPTH)),
                               *self.c, available_ns, *extras))

    def advance(self, watermark_ns, depth, available_ns=None):
        """Only timestamps strictly below watermark are final."""
        if self.next_grid is None:
            return
        available_ns = watermark_ns if available_ns is None else available_ns
        while self.next_grid < watermark_ns:
            at = self.next_grid
            self._integrate(at)
            self._snapshot(at, depth, available_ns)
            if at % MINUTE == 0:
                o, h, l, c, coverage, epoch, consistent, volume0 = self.bar
                coverage = min(1., coverage / MINUTE)
                good = consistent and coverage >= .95 and np.isfinite([o, h, l, c]).all()
                volume = max(0, float(self.c[3:6].sum()) - volume0)
                self.bars.append((at, o, h, l, c, volume, coverage, epoch, good, available_ns))
                self._new_bar(at)
            self.next_grid += self.period
        self._integrate(watermark_ns)

    def _break_support(self):
        self.epoch += 1
        self.flow.reset()
        for metric in self.metrics:
            metric.reset()
        if self.bar is not None:
            self.bar[6], self.bar[7] = False, 0.

    def on_event(self, event, engine, reset_depth=None, available_ns=None):
        t, book = int(event.available_at_ns), event.book
        available_ns = t if available_ns is None else available_ns
        if self.last_ns is not None and t < self.last_ns:
            raise ValueError('Receive-time regression')
        if self.next_grid is None:
            self.cursor = self.last_ns = t
            self.next_grid = (t // self.period + 1) * self.period
            self.book = event.book_before
            self._new_bar(t)
        if self.next_grid < t:
            prior = reset_depth if reset_depth is not None else depth_vector(
                engine, event.book_before, event.deltas, self.tick_nano)
            self.advance(t, prior, available_ns)
        else:
            self._integrate(t)
        broken = (t - self.last_ns > self.max_age or
                  self.last_source_epoch not in (None, book.continuity_epoch) or
                  self.book.valid != book.valid)
        if broken:
            self._break_support()
        facts = self.flow.update(event, self.book, not broken)
        if self.metrics:
            view = BookView(engine)
            for metric in self.metrics:
                update_facts = getattr(metric, 'update_facts', None)
                if update_facts is not None:
                    update_facts(facts, view)
                else:
                    metric.update(event, view)
        if book.valid:
            price = book.midpoint_nano / NANO
            if not np.isfinite(self.bar[0]):
                self.bar[:4] = [price] * 4
            else:
                self.bar[1] = max(self.bar[1], price)
                self.bar[2] = min(self.bar[2], price)
                self.bar[3] = price
        self.last_ns, self.book = t, book
        self.last_source_epoch = book.continuity_epoch
        self.quotes.append((t, book.best_bid_nano/NANO if book.valid else np.nan,
                            book.best_ask_nano/NANO if book.valid else np.nan,
                            book.valid, self.epoch, event.last_source_ordinal, available_ns))
        self.reason_counts[book.reason] = self.reason_counts.get(book.reason, 0) + 1

    def mark_gap(self, at_ns, reason, source_ordinal, available_ns=None):
        """Explicit transport loss; a source reset/snapshot must restore the book."""
        self._break_support()
        self.last_source_epoch = None
        if self.book is not None:
            self.book = replace(self.book, valid=False, reason=reason)
            self.last_ns = at_ns
            self.quotes.append((at_ns, np.nan, np.nan, False, self.epoch, source_ordinal,
                                at_ns if available_ns is None else available_ns))
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def finish(self, end_ns, engine):
        # Compatibility for historical consumers; strict Core.finish rejects
        # incomplete groups instead of pretending that EOF completed F_LAST.
        if self.last_ns is not None:
            self.advance(end_ns + 1, depth_vector(engine, self.book, engine.pending, self.tick_nano))

    def state_dict(self):
        return dict(next_grid=self.next_grid, cursor=self.cursor, last_ns=self.last_ns,
                    epoch=self.epoch, bar=self.bar, book=asdict(self.book) if self.book else None,
                    last_source_epoch=self.last_source_epoch, counters=self.c.tolist(),
                    recent=[[a, b, list(value)] for (a, b), value in self.flow.recent.items()],
                    reason_counts=self.reason_counts,
                    metrics=[metric.state_dict() for metric in self.metrics])

    def load_state_dict(self, state):
        for name in ('next_grid', 'cursor', 'last_ns', 'epoch', 'bar', 'last_source_epoch', 'reason_counts'):
            setattr(self, name, state[name])
        self.book = BookFragment(**state['book']) if state['book'] else None
        self.flow.values[:] = state['counters']
        self.flow.recent = {(side, price): [tuple(v) for v in budgets] for side, price, budgets in state['recent']}
        for metric, saved in zip(self.metrics, state['metrics'], strict=True):
            metric.load_state_dict(saved)
