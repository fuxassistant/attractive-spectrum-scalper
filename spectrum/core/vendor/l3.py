"""Standalone single-route L3 book with an immutable full MBO delta tape."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from typing import Iterable, Mapping

F_MAYBE_BAD_BOOK = 1 << 2
F_BAD_TS_RECV = 1 << 3
F_SNAPSHOT = 1 << 5
F_TOB = 1 << 6
F_LAST = 1 << 7
UNDEFINED_PRICE = (1 << 63) - 1


def _uid(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MboRecord:
    source_ordinal: int
    publisher_id: int
    instrument_id: int
    channel_id: int
    ts_event_ns: int
    ts_recv_ns: int
    sequence: int
    order_id: int
    price_nano: int
    size: int
    flags: int
    action: str
    side: str

    @property
    def is_last(self) -> bool:
        return bool(self.flags & F_LAST)

    @property
    def is_snapshot(self) -> bool:
        return bool(self.flags & F_SNAPSHOT)


@dataclass(frozen=True, slots=True)
class MboDelta:
    source_ordinal: int
    ts_event_ns: int
    ts_recv_ns: int
    sequence: int
    action: str
    side: str
    order_id: int
    flags: int
    record_price_nano: int
    record_size: int
    accepted: bool
    reason: str
    prior_side: str
    prior_price_nano: int
    prior_size: int
    resulting_side: str
    resulting_price_nano: int
    resulting_size: int
    resting_size_delta: int


@dataclass(frozen=True, slots=True)
class BookFragment:
    continuity_epoch: int
    valid: bool
    reason: str
    best_bid_nano: int
    best_ask_nano: int
    bid_size_l1: int
    ask_size_l1: int
    midpoint_nano: int
    microprice_nano: int
    imbalance_l1: float
    active_order_count: int


@dataclass(frozen=True, slots=True)
class CompletedMboEvent:
    event_uid: str
    sequence: int
    event_ts_ns: int
    available_at_ns: int
    first_source_ordinal: int
    last_source_ordinal: int
    deltas: tuple[MboDelta, ...]
    book_before: BookFragment
    book: BookFragment
    bbo_changed: bool
    accepted_book_mutations: int
    deep_book_only_mutations: int


@dataclass(slots=True)
class _Order:
    side: str
    price_nano: int
    size: int


@dataclass(slots=True)
class _Level:
    size: int = 0
    count: int = 0


class L3EventEngine:
    """Applies every MBO record and publishes only complete F_LAST groups."""

    def __init__(self, *, publisher_id: int, instrument_id: int, channel_id: int,
                 event_uid_mode: str = "CANONICAL_CONTENT") -> None:
        if event_uid_mode not in {"CANONICAL_CONTENT", "SOURCE_COORDINATE"}:
            raise ValueError("unsupported event UID mode")
        self.route = (int(publisher_id), int(instrument_id), int(channel_id))
        self.event_uid_mode = event_uid_mode
        self.orders: dict[int, _Order] = {}
        self.bids: dict[int, _Level] = {}
        self.asks: dict[int, _Level] = {}
        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []
        self.pending: list[MboDelta] = []
        self.pending_bbo_before: tuple[int, int, int, int] | None = None
        self.pending_book_before: BookFragment | None = None
        self.continuity_epoch = 0
        self.book_ready = False
        self.snapshot_in_progress = False
        self.quality_reason = "WAITING_FOR_SNAPSHOT"
        self.last_source_ordinal = -1
        self.last_sequence = 0
        self.selected_records = 0
        self.completed_events = 0

    def append(self, record: MboRecord) -> CompletedMboEvent | None:
        if (record.publisher_id, record.instrument_id, record.channel_id) != self.route:
            raise ValueError("record route mismatch")
        if record.source_ordinal <= self.last_source_ordinal:
            raise ValueError("source ordinal regression")
        if not self.pending:
            self.pending_bbo_before = self._bbo_key()
            self.pending_book_before = self._fragment_state(0)
        self.last_source_ordinal = record.source_ordinal
        self.selected_records += 1
        if record.flags & F_MAYBE_BAD_BOOK:
            self._invalidate("MAYBE_BAD_BOOK")
        if record.action != "R" and not record.is_snapshot and record.sequence and self.last_sequence and record.sequence < self.last_sequence:
            self._invalidate("SEQUENCE_REGRESSION")
        if record.action != "R" and not record.is_snapshot and record.sequence:
            self.last_sequence = record.sequence
        if record.action == "R":
            if self.selected_records > 1 and (self.book_ready or self.orders):
                self.continuity_epoch += 1
            self._clear_book()
            self.last_sequence = 0
            self.snapshot_in_progress = record.is_snapshot
            self.book_ready = not record.is_snapshot
            self.quality_reason = "SNAPSHOT_IN_PROGRESS" if record.is_snapshot else "OK"
        elif record.is_snapshot and not self.snapshot_in_progress:
            self.snapshot_in_progress = True
            self.book_ready = False
            self.quality_reason = "SNAPSHOT_IN_PROGRESS"
        self.pending.append(self._apply(record))
        if not record.is_last:
            return None
        self._finalize_snapshot(record)
        self.completed_events += 1
        fragment = self._fragment(record)
        before = self.pending_book_before
        if before is None:
            raise ValueError("missing pre-event book fragment")
        after = self._bbo_key()
        changed = after != self.pending_bbo_before
        deltas = tuple(self.pending)
        mutations = sum(value.accepted and value.action in {"A", "C", "M"} for value in deltas)
        if self.event_uid_mode == "CANONICAL_CONTENT":
            body = {
                "contract": "FULL_MBO_EVENT_GROUP_V1", "route": self.route,
                "sequence": record.sequence, "event_ts_ns": record.ts_event_ns,
                "available_at_ns": record.ts_recv_ns,
                "first_source_ordinal": deltas[0].source_ordinal,
                "last_source_ordinal": deltas[-1].source_ordinal,
                "deltas": [asdict(value) for value in deltas], "book_before": asdict(before), "book": asdict(fragment),
                "bbo_changed": changed, "accepted_book_mutations": mutations,
                "deep_book_only_mutations": mutations if not changed else 0,
            }
            event_uid = _uid(body)
        else:
            coordinate = "|".join(str(value) for value in (
                "FULL_MBO_SOURCE_COORDINATE_V1", *self.route, deltas[0].source_ordinal,
                deltas[-1].source_ordinal, record.sequence, record.ts_event_ns, record.ts_recv_ns,
            ))
            event_uid = hashlib.sha256(coordinate.encode()).hexdigest()
        result = CompletedMboEvent(
            event_uid, record.sequence, record.ts_event_ns, record.ts_recv_ns,
            deltas[0].source_ordinal, deltas[-1].source_ordinal, deltas, before, fragment,
            changed, mutations, mutations if not changed else 0,
        )
        self.pending = []
        self.pending_bbo_before = None
        self.pending_book_before = None
        return result

    def append_many(self, records: Iterable[MboRecord]) -> tuple[CompletedMboEvent, ...]:
        result = []
        for record in records:
            event = self.append(record)
            if event is not None:
                result.append(event)
        return tuple(result)

    def _apply(self, record: MboRecord) -> MboDelta:
        prior = self.orders.get(record.order_id)
        prior_side = "" if prior is None else prior.side
        prior_price = 0 if prior is None else prior.price_nano
        prior_size = 0 if prior is None else prior.size
        accepted, reason = True, "OK"
        if record.action == "R":
            reason = "BOOK_RESET"
        elif record.action in {"T", "F", "N"}:
            reason = "NON_RESTING_FACT"
        elif record.side not in {"A", "B"}:
            raise ValueError("book mutation lacks side")
        elif record.price_nano == UNDEFINED_PRICE or record.size < 0:
            raise ValueError("book mutation has invalid price/size")
        elif record.flags & F_TOB:
            accepted, reason = False, "TOB_RECORD_IN_MBO_STREAM"
            self._invalidate(reason)
        elif record.action == "A":
            if record.size <= 0:
                accepted, reason = False, "NONPOSITIVE_ADD_SIZE"; self._invalidate(reason)
            elif prior is not None:
                raise ValueError("duplicate add")
            else:
                self.orders[record.order_id] = _Order(record.side, record.price_nano, record.size)
                self._add_level(record.side, record.price_nano, record.size)
        elif record.action == "C":
            if prior is None:
                accepted, reason = False, "UNKNOWN_CANCEL"; self._invalidate(reason)
            else:
                removed = min(record.size, prior.size)
                self._remove_level(prior.side, prior.price_nano, removed, removed == prior.size)
                prior.size -= removed
                if prior.size == 0:
                    self.orders.pop(record.order_id)
                if record.size > removed:
                    accepted, reason = False, "OVER_CANCEL"; self._invalidate(reason)
        elif record.action == "M":
            if record.size <= 0:
                if prior is None:
                    accepted, reason = False, "NONPOSITIVE_UNKNOWN_MODIFY"; self._invalidate(reason)
                else:
                    self._remove_order(record.order_id, prior)
            elif prior is None:
                self.orders[record.order_id] = _Order(record.side, record.price_nano, record.size)
                self._add_level(record.side, record.price_nano, record.size)
                reason = "UNKNOWN_MODIFY_CREATED_ORDER"
            elif prior.side != record.side:
                accepted, reason = False, "MODIFY_SIDE_CHANGE"; self._invalidate(reason)
            elif prior.price_nano != record.price_nano:
                self._remove_order(record.order_id, prior)
                self.orders[record.order_id] = _Order(record.side, record.price_nano, record.size)
                self._add_level(record.side, record.price_nano, record.size)
            else:
                delta = record.size - prior.size
                if delta > 0:
                    self._levels(prior.side)[prior.price_nano].size += delta
                elif delta < 0:
                    self._remove_level(prior.side, prior.price_nano, -delta, False)
                prior.size = record.size
        else:
            raise ValueError(f"unsupported MBO action: {record.action}")
        resulting = self.orders.get(record.order_id)
        resulting_side = "" if resulting is None else resulting.side
        resulting_price = 0 if resulting is None else resulting.price_nano
        resulting_size = 0 if resulting is None else resulting.size
        return MboDelta(
            record.source_ordinal, record.ts_event_ns, record.ts_recv_ns, record.sequence,
            record.action, record.side, record.order_id, record.flags, record.price_nano,
            record.size, accepted, reason,
            prior_side, prior_price, prior_size, resulting_side, resulting_price,
            resulting_size, resulting_size - prior_size,
        )

    def _levels(self, side: str) -> dict[int, _Level]:
        return self.bids if side == "B" else self.asks

    def _add_level(self, side: str, price: int, size: int) -> None:
        levels = self._levels(side)
        if price not in levels:
            levels[price] = _Level()
            heapq.heappush(self.bid_heap if side == "B" else self.ask_heap, -price if side == "B" else price)
        levels[price].size += size
        levels[price].count += 1

    def _remove_level(self, side: str, price: int, size: int, remove_order: bool) -> None:
        level = self._levels(side).get(price)
        if level is None or size > level.size:
            raise ValueError("level underflow")
        level.size -= size
        if remove_order:
            level.count -= 1
        if level.size == 0 or level.count == 0:
            self._levels(side).pop(price, None)

    def _remove_order(self, order_id: int, order: _Order) -> None:
        self._remove_level(order.side, order.price_nano, order.size, True)
        self.orders.pop(order_id)

    def _best(self, side: str) -> tuple[int, _Level] | None:
        heap, levels = (self.bid_heap, self.bids) if side == "B" else (self.ask_heap, self.asks)
        while heap:
            price = -heap[0] if side == "B" else heap[0]
            level = levels.get(price)
            if level is not None and level.size > 0:
                return price, level
            heapq.heappop(heap)
        return None

    def _bbo_key(self) -> tuple[int, int, int, int]:
        bid, ask = self._best("B"), self._best("A")
        return (0 if bid is None else bid[0], 0 if ask is None else ask[0], 0 if bid is None else bid[1].size, 0 if ask is None else ask[1].size)

    def _fragment(self, record: MboRecord) -> BookFragment:
        return self._fragment_state(record.flags)

    def _fragment_state(self, flags: int) -> BookFragment:
        bid, ask = self._best("B"), self._best("A")
        crossed = bid is not None and ask is not None and bid[0] >= ask[0]
        valid = self.book_ready and not self.snapshot_in_progress and bid is not None and ask is not None and not crossed and not (flags & (F_MAYBE_BAD_BOOK | F_BAD_TS_RECV))
        reason = "OK" if valid else self.quality_reason
        if not valid and reason == "OK":
            reason = "BAD_TS_RECV" if flags & F_BAD_TS_RECV else "CROSSED_BOOK" if crossed else "INCOMPLETE_BOOK"
        bp, bs = (0, 0) if bid is None else (bid[0], bid[1].size)
        ap, ass = (0, 0) if ask is None else (ask[0], ask[1].size)
        denominator = bs + ass
        return BookFragment(
            self.continuity_epoch, bool(valid), reason, bp if valid else 0, ap if valid else 0,
            bs if valid else 0, ass if valid else 0, (bp + ap) // 2 if valid else 0,
            (ap * bs + bp * ass) // denominator if valid and denominator else 0,
            (bs - ass) / denominator if valid and denominator else 0.0, len(self.orders),
        )

    def _invalidate(self, reason: str) -> None:
        self.book_ready = False
        self.quality_reason = reason

    def _finalize_snapshot(self, record: MboRecord) -> None:
        """Legacy behavior; corrected lifecycle is versioned in l3_v2."""
        if record.is_snapshot and self.snapshot_in_progress:
            self.snapshot_in_progress = False
            self.book_ready = True
            self.quality_reason = "OK"

    def _clear_book(self) -> None:
        self.orders.clear(); self.bids.clear(); self.asks.clear(); self.bid_heap.clear(); self.ask_heap.clear()

    def checkpoint(self) -> dict[str, object]:
        if self.pending:
            raise ValueError("checkpoint requires an F_LAST boundary")
        body = {
            "contract": "STANDALONE_L3_EVENT_ENGINE_V1", "route": self.route,
            "orders": [[key, value.side, value.price_nano, value.size] for key, value in sorted(self.orders.items())],
            "state": {
                "continuity_epoch": self.continuity_epoch, "book_ready": self.book_ready,
                "snapshot_in_progress": self.snapshot_in_progress, "quality_reason": self.quality_reason,
                "last_source_ordinal": self.last_source_ordinal, "last_sequence": self.last_sequence,
                "selected_records": self.selected_records, "completed_events": self.completed_events,
            },
        }
        return {**body, "checkpoint_uid": _uid(body)}

    @classmethod
    def restore(cls, checkpoint: Mapping[str, object]) -> "L3EventEngine":
        body = {key: value for key, value in checkpoint.items() if key != "checkpoint_uid"}
        if checkpoint.get("checkpoint_uid") != _uid(body) or body.get("contract") != "STANDALONE_L3_EVENT_ENGINE_V1":
            raise ValueError("checkpoint drift")
        route = body["route"]
        result = cls(publisher_id=int(route[0]), instrument_id=int(route[1]), channel_id=int(route[2]))
        for raw in body["orders"]:
            order_id, side, price, size = int(raw[0]), str(raw[1]), int(raw[2]), int(raw[3])
            result.orders[order_id] = _Order(side, price, size)
            result._add_level(side, price, size)
        for key, value in body["state"].items():
            setattr(result, key, value)
        return result
