"""One MBO owner; immutable committed queries, bounded batches and restart."""
from dataclasses import dataclass
from .core import MboStream, StreamConfig


@dataclass(frozen=True)
class MarketView:
    available_ns: int
    source_ordinal: int
    epoch: int
    bid_nano: int
    ask_nano: int
    bid_size: int
    ask_size: int
    known: bool
    reason: str

    @property
    def mid2_nano(self):
        return self.bid_nano + self.ask_nano


class MarketReplay:
    def __init__(self, config, *, on_quote=None, on_snapshot=None, on_bar=None):
        self.committed_ordinal = -1
        def quote(row):
            self.committed_ordinal = row[5]
            if on_quote is not None:on_quote(row)
        self.core = MboStream(config, features=False, on_quote=quote,
                              on_snapshot=on_snapshot, on_bar=on_bar)

    def feed(self, record, *, received_at_ns=None):
        self.core.feed(record, received_at_ns=received_at_ns)

    def feed_batch(self, records, *, stop_available_ns=None):
        if stop_available_ns is not None and not hasattr(records,'__getitem__'):
            raise TypeError('A bounded cutoff requires an indexable batch; retain records[consumed:]')
        count = 0
        for r in records:
            if stop_available_ns is not None and r.ts_recv_ns > stop_available_ns:
                break
            self.feed(r)
            count += 1
        return count

    def advance(self, watermark_ns, *, received_at_ns=None):
        self.core.advance(watermark_ns,received_at_ns=received_at_ns)

    def mark_gap(self, at_ns, reason='TRANSPORT_GAP', *, received_at_ns=None):
        self.core.mark_gap(at_ns,reason,received_at_ns=received_at_ns)

    def finish(self):
        """Finalize only the observed prefix; pending groups must be resumed."""
        self.core.finish()

    def query(self, at_ns=None):
        b = self.core.accumulator.book
        if b is None:
            raise LookupError('No complete publication')
        t = self.core.received_ns if at_ns is None else at_ns
        if t < self.core.received_ns:
            raise ValueError('Query cannot rewind the source')
        known = b.valid and t-self.core.accumulator.last_ns <= self.core.config.max_quote_age_ns
        # The accumulator owns the last committed book even during partial F_LAST.
        return MarketView(t, self.committed_ordinal,
                          self.core.accumulator.epoch, b.best_bid_nano, b.best_ask_nano,
                          b.bid_size_l1, b.ask_size_l1, known,
                          b.reason if known or not b.valid else 'STALE_QUOTE')

    def checkpoint(self):
        from .core.state import seal
        return seal(dict(contract='SPECTRUM_REPLAY_V1',committed_ordinal=self.committed_ordinal,core=self.core.checkpoint()))

    @classmethod
    def restore(cls, checkpoint, **callbacks):
        from .core.state import unseal
        state=unseal(checkpoint)
        if state['contract']!='SPECTRUM_REPLAY_V1':raise ValueError('Replay contract drift')
        obj = cls.__new__(cls)
        obj.committed_ordinal=state['committed_ordinal']
        on_quote=callbacks.pop('on_quote',None)
        def quote(row):
            obj.committed_ordinal=row[5]
            if on_quote is not None:on_quote(row)
        obj.core = MboStream.restore(state['core'], on_quote=quote, **callbacks)
        return obj
