"""Price/lifecycle-aware facts of one COMPLETE publication, never inferred trades.

F links are evidence for resting decrements; they do not mutate the book and
do not add a second T volume. Ambiguous decrements remain explicitly unknown.
"""
from dataclasses import dataclass, replace
from .vendor.l3 import F_SNAPSHOT, F_MAYBE_BAD_BOOK, F_BAD_TS_RECV

VERSION = 'GC_EVENT_FACTS_V1'


def stock_supported(fragment):
    """Reconstructed orders can be known while a usable two-sided price is not."""
    return fragment.valid or fragment.reason in ('INCOMPLETE_BOOK', 'CROSSED_BOOK')


@dataclass(frozen=True, slots=True)
class Mutation:
    ordinal: int
    order_id: int
    side: str
    price_nano: int
    quantity: int
    action: str
    version: int
    accepted: bool
    kind: str
    linked_fill: int = 0
    ambiguous_remove: int = 0
    passive_fill: int = 0
    aggressor_fill: int = 0


@dataclass(frozen=True, slots=True)
class Trade:
    ordinal: int
    order_id: int
    side: str
    price_nano: int
    quantity: int


@dataclass(frozen=True, slots=True)
class Fill:
    ordinal: int
    order_id: int
    side: str
    price_nano: int
    quantity: int
    role: str
    resting_side: str
    queue_price_nano: int
    version: int | None
    linked: int = 0


@dataclass(frozen=True, slots=True)
class EventFacts:
    event: object
    mutations: tuple
    trades: tuple
    fills: tuple
    links: tuple
    supported: bool
    unresolved_fill: int
    legacy_wrong_price_qty: int
    legacy_changed_qty: int

    @property
    def has_snapshot(self):
        return any(d.flags & F_SNAPSHOT or d.action == 'R' for d in self.event.deltas)

    @property
    def price_supported(self):
        return self.supported and self.event.book_before.valid and self.event.book.valid


_cached_event = _cached_facts = None


def decode_event(event):
    """Shared identity cache retains ONE event; facts themselves are stateless."""
    global _cached_event, _cached_facts
    if event is _cached_event:
        return _cached_facts
    mutations, trades, fills = [], [], []
    versions = {}
    last_trade = None
    supported = (stock_supported(event.book_before) and stock_supported(event.book)
                 and event.book_before.continuity_epoch == event.book.continuity_epoch
                 and not any(d.flags & (F_SNAPSHOT | F_MAYBE_BAD_BOOK | F_BAD_TS_RECV)
                             or d.action == 'R' for d in event.deltas))
    for d in event.deltas:
        if d.flags & F_SNAPSHOT or d.action == 'R':
            continue
        v = versions.setdefault(d.order_id, 0)
        if d.action == 'T' and d.accepted and d.record_size > 0:
            last_trade = Trade(d.source_ordinal, d.order_id, d.side,
                               d.record_price_nano, d.record_size)
            trades.append(last_trade)
        elif d.action == 'F' and d.accepted and d.record_size > 0:
            role = 'UNKNOWN'
            resting_side=d.side if d.side in ('B','A') else d.prior_side
            if last_trade is not None and last_trade.price_nano == d.record_price_nano:
                if last_trade.order_id and d.order_id == last_trade.order_id:
                    role = 'AGGRESSOR_EXISTING'
                    if resting_side not in ('B','A') and last_trade.side in ('B','A'):
                        resting_side=last_trade.side
                elif last_trade.side in ('B', 'A') and resting_side in ('B','A') and resting_side != last_trade.side:
                    role = 'PASSIVE'
            # An aggressor can execute at a price different from its own limit.
            queue_price = (d.prior_price_nano if role == 'AGGRESSOR_EXISTING'
                           and d.prior_size else d.record_price_nano)
            version = v if d.prior_size else None
            if role=='AGGRESSOR_EXISTING' and not d.prior_size:
                prior_removes=[m for m in mutations if m.order_id==d.order_id and m.quantity<0
                               and m.side==resting_side and m.accepted]
                if prior_removes:
                    last_version=max(m.version for m in prior_removes)
                    prices={m.price_nano for m in prior_removes if m.version==last_version}
                    if len(prices)==1:queue_price=next(iter(prices));version=last_version
            fills.append(Fill(d.source_ordinal, d.order_id, d.side, d.record_price_nano,
                              d.record_size, role, resting_side, queue_price, version))
        if d.action not in ('A', 'C', 'M'):
            continue
        moved = (d.prior_side, d.prior_price_nano) != (d.resulting_side, d.resulting_price_nano)
        removed = d.prior_size if moved else max(0, d.prior_size - d.resulting_size)
        added = d.resulting_size if moved else max(0, d.resulting_size - d.prior_size)
        if removed and d.prior_side in ('B', 'A'):
            mutations.append(Mutation(d.source_ordinal, d.order_id, d.prior_side,
                d.prior_price_nano, -removed, d.action, v, d.accepted,
                'REPRICE' if moved and d.resulting_size else 'DECREMENT'))
        if moved and (d.prior_size or d.action == 'A'):
            versions[d.order_id] = v + 1
        if added and d.resulting_side in ('B', 'A'):
            mutations.append(Mutation(d.source_ordinal, d.order_id, d.resulting_side,
                d.resulting_price_nano, added, d.action, versions[d.order_id], d.accepted,
                'REPRICE' if d.prior_size and moved else 'ADDITION'))
        if not d.resulting_size and d.prior_size:
            versions[d.order_id] = v + 1
    links = []
    for fi, f in enumerate(fills):
        if f.resting_side not in ('B','A'):
            sides={m.side for m in mutations if m.quantity<0 and m.order_id==f.order_id
                   and m.price_nano==f.queue_price_nano and (f.version is None or m.version==f.version)}
            if len(sides)==1:
                f=replace(f,resting_side=next(iter(sides)));fills[fi]=f
        candidates = [i for i, m in enumerate(mutations)
            if m.accepted and m.quantity < 0
            and m.order_id == f.order_id and m.side == f.resting_side
            and m.price_nano == f.queue_price_nano
            and (f.version is None or m.version == f.version)
            and -m.quantity > m.linked_fill]
        quantity = sum(-mutations[i].quantity - mutations[i].linked_fill for i in candidates)
        if len(candidates) > 1 and f.quantity < quantity:
            for i in candidates:
                m = mutations[i]
                mutations[i] = replace(m, ambiguous_remove=-m.quantity-m.linked_fill)
            continue
        # Multiple candidates are identifiable only when all are fully covered.
        if len(candidates) == 1 or candidates and f.quantity >= quantity:
            remaining = f.quantity
            for i in candidates:
                m = mutations[i]
                q = min(remaining, -m.quantity - m.linked_fill)
                mutations[i] = replace(m, linked_fill=m.linked_fill+q,
                    ambiguous_remove=max(0, m.ambiguous_remove-q),
                    passive_fill=m.passive_fill + (q if f.role == 'PASSIVE' else 0),
                    aggressor_fill=m.aggressor_fill + (q if f.role == 'AGGRESSOR_EXISTING' else 0))
                links.append((f.ordinal, m.ordinal, m.side, m.price_nano, q, f.role))
                remaining -= q
            fills[fi] = replace(f, linked=f.quantity-remaining)
    # Quantify the exact old greedy attribution without using it as new truth.
    old_pool, old_allocation = {}, {}
    for f in fills:
        old_pool[f.order_id] = old_pool.get(f.order_id, 0) + f.quantity
    wrong = changed = 0
    for m in mutations:
        if m.quantity >= 0 or not m.accepted:
            continue
        old = min(-m.quantity, old_pool.get(m.order_id, 0))
        old_pool[m.order_id] = old_pool.get(m.order_id, 0)-old
        old_allocation[(m.ordinal, m.side, m.price_nano)] = old
        changed += abs(old-m.linked_fill)
        if old and not any(f.order_id == m.order_id and f.resting_side == m.side
                          and f.queue_price_nano == m.price_nano for f in fills):
            wrong += old
    result = EventFacts(event, tuple(mutations), tuple(trades), tuple(fills), tuple(links),
                        bool(supported), sum(f.quantity-f.linked for f in fills), wrong, changed)
    _cached_event, _cached_facts = event, result
    return result
