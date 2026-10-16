"""Optional stateful event measurements, upstream of snapshot feature families."""
from typing import Protocol


class BookView:
    """Read interface valid during update(), never retain this live view."""
    __slots__ = ('_engine',)

    def __init__(self, engine):
        self._engine = engine

    def level_quantity(self, side, price_nano):
        if side not in ('B', 'A'):
            raise ValueError('Book side must be B or A')
        level = (self._engine.bids if side == 'B' else self._engine.asks).get(price_nano)
        return level.size if level is not None else 0

    def order(self, order_id):
        order = self._engine.orders.get(order_id)
        return None if order is None else (order.side, order.price_nano, order.size)

    def levels(self, side):
        if side not in ('B', 'A'):
            raise ValueError('Book side must be B or A')
        levels = self._engine.bids if side == 'B' else self._engine.asks
        return ((price, levels[price].size, levels[price].count)
                for price in sorted(levels, reverse=side == 'B'))


class EventMetric(Protocol):
    """Legacy update is required. Optional update_facts(facts, book) reuses
    the group-local immutable EventFacts decoded by the common accumulator.
    BookView remains borrowed; it must not be retained after the callback.
    """
    name: str
    version: str
    fields: tuple[str, ...]

    def update(self, event, book: BookView) -> None: ...
    def values(self, t_ns: int) -> tuple[float, ...]: ...
    def reset(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
