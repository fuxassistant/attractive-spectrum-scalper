"""Data-only checkpoints, including an unfinished F_LAST group; never pickle."""
from dataclasses import asdict
import math
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
from .schema import digest
from .vendor.l3 import BookFragment, MboDelta, _Order
from .vendor.l3_v2 import L3EventEngineV2

ENGINE_FIELDS = (
    'continuity_epoch', 'book_ready', 'snapshot_in_progress', 'quality_reason',
    'last_source_ordinal', 'last_sequence', 'selected_records', 'completed_events',
    'snapshot_fault', 'snapshot_rows_seen',
)


@lru_cache(maxsize=1)
def implementation_id():
    root = Path(__file__).parent
    names = ('schema.py', 'aggregate.py', 'event_facts.py', 'state.py', 'stream.py', 'extensions.py',
             'feature_stream.py', 'feature_registry.py', 'feature_families.py', 'price_history.py',
             'vendor/l3.py', 'vendor/l3_v2.py')
    return digest({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


def engine_state(engine):
    return dict(route=list(engine.route), event_uid_mode=engine.event_uid_mode,
                orders=[[key, value.side, value.price_nano, value.size] for key, value in sorted(engine.orders.items())],
                scalars={key: getattr(engine, key) for key in ENGINE_FIELDS},
                pending=[asdict(delta) for delta in engine.pending],
                before=asdict(engine.pending_book_before) if engine.pending_book_before else None,
                bbo_before=engine.pending_bbo_before)


def restore_engine(state):
    route = state['route']
    engine = L3EventEngineV2(publisher_id=route[0], instrument_id=route[1], channel_id=route[2],
                           event_uid_mode=state['event_uid_mode'])
    for key, side, price, size in state['orders']:
        engine.orders[key] = _Order(side, price, size)
        engine._add_level(side, price, size)
    for key in ENGINE_FIELDS:
        setattr(engine, key, state['scalars'][key])
    engine.pending = [MboDelta(**delta) for delta in state['pending']]
    engine.pending_book_before = BookFragment(**state['before']) if state['before'] else None
    engine.pending_bbo_before = tuple(state['bbo_before']) if state['bbo_before'] else None
    return engine


def encode(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {'$float': str(value)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'$float'}:
            return float(value['$float'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def seal(state):
    body = encode(state)
    return {'state': body, 'sha256': digest(body)}


def unseal(checkpoint):
    if digest(checkpoint['state']) != checkpoint['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return decode(checkpoint['state'])
