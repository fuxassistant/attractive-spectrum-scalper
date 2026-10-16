"""Wire-independent clocks, configuration and emitted row layouts."""
from dataclasses import asdict, dataclass
import hashlib
import json

NS = 1_000_000_000
MINUTE = 60 * NS
NANO = 1_000_000_000
TICK = .1

COUNTERS = (
    'path_ticks', 'up_ticks', 'down_ticks', 'trade_buy', 'trade_sell',
    'trade_unknown', 'trade_count', 'add_bid', 'add_ask', 'remove_bid',
    'remove_ask', 'matched_fill_remove_bid', 'matched_fill_remove_ask',
    'post_trade_add_bid', 'post_trade_add_ask', 'event_count',
)
DEPTH = (
    'depth_bid_1', 'depth_bid_4', 'depth_bid_12',
    'depth_ask_1', 'depth_ask_4', 'depth_ask_12',
    'sweep_buy_1', 'sweep_buy_5', 'sweep_buy_10',
    'sweep_sell_1', 'sweep_sell_5', 'sweep_sell_10',
)
QUOTE_SCHEMA = [
    ('t_ns', 'int64'), ('bid', 'float64'), ('ask', 'float64'),
    ('valid', 'bool'), ('epoch', 'int64'), ('source_ordinal', 'int64'),
    ('available_ns', 'int64'),
]
SNAP_SCHEMA = [
    ('t_ns', 'int64'), ('epoch', 'int64'), ('valid', 'bool'),
    ('book_valid', 'bool'), ('age_ns', 'int64'),
] + [(s, 'float64') for s in (
    'bid', 'ask', 'mid', 'bid_size', 'ask_size', *DEPTH,
    *('c_' + name for name in COUNTERS),
)] + [('available_ns', 'int64')]
BAR_SCHEMA = [('t_ns', 'int64')] + [
    (s, 'float64') for s in ('o', 'h', 'l', 'c', 'volume', 'coverage')
] + [('epoch', 'int64'), ('valid', 'bool'), ('available_ns', 'int64')]


@dataclass(frozen=True)
class StreamConfig:
    publisher_id: int
    instrument_id: int
    channel_id: int
    tick_nano: int = 100_000_000
    snapshot_ns: int = 2 * NS
    max_quote_age_ns: int = 30 * NS
    fragment: int = 0
    max_pending_records: int = 100_000

    def __post_init__(self):
        values = asdict(self)
        if any(type(value) is not int for value in values.values()):
            raise TypeError('Stream configuration uses integer coordinates')
        if min(self.route) < 0 or self.fragment < 0 or self.tick_nano <= 0:
            raise ValueError('Invalid instrument route, fragment or tick')
        if self.snapshot_ns <= 0 or MINUTE % self.snapshot_ns:
            raise ValueError('Snapshot period must divide one minute exactly')
        if self.max_quote_age_ns <= 0 or self.max_pending_records <= 0:
            raise ValueError('Invalid support or pending-group bound')

    @property
    def route(self):
        return self.publisher_id, self.instrument_id, self.channel_id

    @property
    def tick_size(self):
        return self.tick_nano / NANO


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
