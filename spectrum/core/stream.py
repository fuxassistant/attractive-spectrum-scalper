"""Single-route event kernel shared by historical replay and live adapters."""
from dataclasses import asdict
import heapq
from .aggregate import MapAccumulator, depth_vector
from .schema import StreamConfig, SNAP_SCHEMA, BAR_SCHEMA, NS, MINUTE
from .feature_stream import FeatureStream
from .feature_families import WINDOWS
from .state import engine_state, restore_engine, seal, unseal, implementation_id
from .vendor.l3_v2 import L3EventEngineV2

CONTRACT = 'MBO_STREAM_CORE_V3'


class CallbackSink:
    def __init__(self, callback=None):
        self.callback = callback
        self.count = 0

    def append(self, row):
        self.count += 1
        if self.callback is not None:
            self.callback(row)


class MboStream:
    """feed() never treats a transport batch as an exchange publication.

    advance(w) is an explicit promise: no record with receive time < w can
    arrive later. A raw record advances this frontier only to its own timestamp.
    Equal-timestamp groups stay ordered; a bucket at that time remains open.
    Callbacks are synchronous; a failing callback poisons the instance, requiring
    restore. A checkpoint contains state, not already delivered sink output.
    """
    def __init__(self, config, *, on_quote=None, on_snapshot=None, on_bar=None,
                 on_features=None, features=True, families=None, metrics=(), feature_windows=WINDOWS):
        self.config = config
        self.engine = L3EventEngineV2(publisher_id=config.publisher_id,
                                    instrument_id=config.instrument_id, channel_id=config.channel_id)
        self.metrics = tuple(metrics)
        self.metric_contract = []
        names = [name for name, _ in SNAP_SCHEMA]
        for metric in self.metrics:
            if not metric.name or not metric.version:
                raise ValueError('Event metrics need name/version')
            fields = list(metric.fields)
            if not fields or any(field in names for field in fields) or len(set(fields)) != len(fields):
                raise ValueError('Duplicate/empty event metric fields')
            names.extend(fields)
            self.metric_contract.append(dict(name=metric.name, version=metric.version, fields=fields))
        if len({item['name'] for item in self.metric_contract}) != len(self.metric_contract):
            raise ValueError('Duplicate event metric name')
        self.snapshot_names = tuple(names)
        self.feature_stream = FeatureStream(tick_size=config.tick_size, snapshot_ns=config.snapshot_ns,
                                            windows=feature_windows, families=families) if features else None
        self.on_snapshot, self.on_bar, self.on_features = on_snapshot, on_bar, on_features
        self.quote_sink = CallbackSink(on_quote)
        self.snapshot_sink = CallbackSink(self._snapshot)
        self.bar_sink = CallbackSink(self._bar)
        self.accumulator = MapAccumulator(self.quote_sink, self.snapshot_sink, self.bar_sink,
            config.fragment, config.snapshot_ns, config.max_quote_age_ns, config.tick_nano, self.metrics)
        self.watermark_ns = self.last_record_ns = self.reset_depth = None
        self.received_ns = None
        self.failed = self.closed = False

    def _snapshot(self, row):
        if self.feature_stream is not None:
            self.feature_stream.push_snapshot(dict(zip(self.snapshot_names, row, strict=True)))
        if self.on_snapshot is not None:
            self.on_snapshot(row)

    def _bar(self, row):
        if self.feature_stream is not None:
            frame = self.feature_stream.push_bar(dict(zip((name for name, _ in BAR_SCHEMA), row, strict=True)))
            if self.on_features is not None:
                self.on_features(frame)
        if self.on_bar is not None:
            self.on_bar(row)

    def _check_active(self):
        if self.failed or self.closed:
            raise RuntimeError('Stream is failed/closed; restore or create a new stream')

    def _depth(self):
        if self.reset_depth is not None:
            return self.reset_depth
        if self.accumulator.book is None:
            return None
        return depth_vector(self.engine, self.accumulator.book, self.engine.pending, self.config.tick_nano)

    def _availability(self, source_ns, received_at_ns):
        if received_at_ns is not None and (type(received_at_ns) is not int or received_at_ns < source_ns):
            raise ValueError('Delivery time must be integer UTC ns no earlier than source time')
        if received_at_ns is not None and self.received_ns is not None and received_at_ns < self.received_ns:
            raise ValueError('Delivery time regression')
        return max(source_ns, received_at_ns or source_ns, self.received_ns or source_ns)

    def advance(self, watermark_ns, *, received_at_ns=None):
        self._check_active()
        if type(watermark_ns) is not int or watermark_ns < 0:
            raise ValueError('Watermark must be nonnegative integer UTC nanoseconds')
        if self.watermark_ns is not None and watermark_ns < self.watermark_ns:
            raise ValueError('Watermark regression')
        available = self._availability(watermark_ns, received_at_ns)
        try:
            if self.accumulator.next_grid is not None:
                self.accumulator.advance(watermark_ns, self._depth(), available)
            self.watermark_ns = watermark_ns
            self.received_ns = available
        except Exception:
            self.failed = True
            raise

    def feed(self, record, *, received_at_ns=None):
        self._check_active()
        if record.ts_recv_ns < 0:
            raise ValueError('Receive timestamp must be nonnegative UTC nanoseconds')
        if (record.publisher_id, record.instrument_id, record.channel_id) != self.config.route:
            raise ValueError('record route mismatch')
        if record.source_ordinal <= self.engine.last_source_ordinal:
            raise ValueError('source ordinal regression/duplicate')
        if self.watermark_ns is not None and record.ts_recv_ns < self.watermark_ns:
            raise ValueError('Receive-time regression or record behind committed watermark')
        if len(self.engine.pending) >= self.config.max_pending_records:
            self.failed = True
            raise ValueError('Unfinished F_LAST group exceeds configured bound')
        available = self._availability(record.ts_recv_ns, received_at_ns)
        try:
            # Compute depth only when a clock point will actually be emitted.
            if self.accumulator.next_grid is not None and self.accumulator.next_grid < record.ts_recv_ns:
                self.advance(int(record.ts_recv_ns), received_at_ns=available)
            else:
                self.watermark_ns = int(record.ts_recv_ns)
                self.received_ns = available
            if record.action == 'R' and self.reset_depth is None:
                self.reset_depth = self._depth()
            event = self.engine.append(record)
            self.last_record_ns = int(record.ts_recv_ns)
            if event is not None:
                self.accumulator.on_event(event, self.engine, self.reset_depth, available)
                self.reset_depth = None
                if self.engine.completed_events % 10_000 == 0:
                    self._compact_heaps()
        except Exception:
            self.failed = True
            raise

    def feed_many(self, records, *, received_at_ns=None):
        for record in records:
            self.feed(record, received_at_ns=received_at_ns)

    def _compact_heaps(self):
        for heap, levels, sign in ((self.engine.bid_heap, self.engine.bids, -1),
                                   (self.engine.ask_heap, self.engine.asks, 1)):
            if len(heap) > 4 * len(levels) + 1024:
                heap[:] = [sign * price for price in levels]
                heapq.heapify(heap)

    def mark_gap(self, at_ns, reason='TRANSPORT_GAP', *, received_at_ns=None):
        """Discard an incomplete group and require source book recovery.

        Source ordinals remain monotone across this control event. A reconnect
        with a fresh numbering scheme needs an adapter-owned monotone offset.
        """
        self.advance(at_ns, received_at_ns=received_at_ns)
        ordinal = self.engine.last_source_ordinal
        try:
            self.accumulator.mark_gap(at_ns, reason, ordinal, self.received_ns)
        except Exception:
            self.failed = True
            raise
        self.engine = L3EventEngineV2(publisher_id=self.config.publisher_id,
            instrument_id=self.config.instrument_id, channel_id=self.config.channel_id)
        self.engine.last_source_ordinal = ordinal
        self.reset_depth = None

    def finish(self, end_ns=None):
        self._check_active()
        if self.engine.pending:
            raise ValueError('EOF inside F_LAST group: checkpoint to continue or declare a gap')
        if end_ns is None:
            # EOF proves only the observed prefix. Extending to the next minute
            # would invent a quiet future suffix. A wider capture boundary must
            # be explicitly supplied by the source adapter.
            end_ns = max(self.last_record_ns or 0, self.watermark_ns or 0)
        self.advance(int(end_ns)+1)
        self.closed = True

    def checkpoint(self):
        if self.failed:
            raise RuntimeError('Cannot checkpoint failed state')
        return seal(dict(contract=CONTRACT, implementation=implementation_id(), config=asdict(self.config), engine=engine_state(self.engine),
                         accumulator=self.accumulator.state_dict(), metrics=self.metric_contract,
                         features=self.feature_stream.state_dict() if self.feature_stream else None,
                         watermark_ns=self.watermark_ns, last_record_ns=self.last_record_ns,
                         received_ns=self.received_ns,
                         reset_depth=self.reset_depth.tolist() if self.reset_depth is not None else None,
                         closed=self.closed,
                         emitted=[self.quote_sink.count, self.snapshot_sink.count, self.bar_sink.count]))

    @classmethod
    def restore(cls, checkpoint, *, on_quote=None, on_snapshot=None, on_bar=None,
                on_features=None, families=None, metrics=(), expected_config=None):
        import numpy as np
        state = unseal(checkpoint)
        if state['contract'] != CONTRACT or state['implementation'] != implementation_id():
            raise ValueError('Stream checkpoint version drift')
        config = StreamConfig(**state['config'])
        if expected_config is not None and config != expected_config:
            raise ValueError('Stream checkpoint route/configuration drift')
        result = cls(config, on_quote=on_quote, on_snapshot=on_snapshot, on_bar=on_bar,
                     on_features=on_features, features=state['features'] is not None,
                     families=families, metrics=metrics,
                     feature_windows=tuple(state['features']['windows']) if state['features'] else WINDOWS)
        if result.metric_contract != state['metrics']:
            raise ValueError('Event metric checkpoint schema drift')
        result.engine = restore_engine(state['engine'])
        result.accumulator.load_state_dict(state['accumulator'])
        if result.feature_stream:
            result.feature_stream.load_state_dict(state['features'])
        result.watermark_ns, result.last_record_ns = state['watermark_ns'], state['last_record_ns']
        result.received_ns = state['received_ns']
        result.reset_depth = np.array(state['reset_depth']) if state['reset_depth'] is not None else None
        result.closed = state['closed']
        for sink, count in zip((result.quote_sink, result.snapshot_sink, result.bar_sink), state['emitted']):
            sink.count = count
        return result
