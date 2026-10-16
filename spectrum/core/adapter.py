"""File adapter over the same MboStream used for online delivery.

The adapter owns decoding, source identity and Parquet; the core owns all
market state, event publication, clocks, history and calculations.
"""
from pathlib import Path
import json
import time
import numpy as np
from .schema import (StreamConfig, NS, NANO, TICK, COUNTERS, DEPTH,
                     QUOTE_SCHEMA, SNAP_SCHEMA, BAR_SCHEMA)
from .storage import Sink, sha
from .aggregate import MapAccumulator, depth_vector
from .stream import MboStream
from .vendor.glbx_dbn import GlbxZipMboSource
from .vendor.l3 import MboRecord


def records_from_block(block, route, ordinal_offset=0):
    """A transport block only supplies records and stable source coordinates."""
    selected = ((block['publisher_id'] == route[0]) &
                (block['instrument_id'] == route[1]) &
                (block['channel_id'] == route[2]))
    for index in np.flatnonzero(selected):
        row = block[index]
        yield MboRecord(ordinal_offset + int(index), *route,
                        int(row['ts_event']), int(row['ts_recv']), int(row['sequence']),
                        int(row['order_id']), int(row['price']), int(row['size']),
                        int(row['flags']), bytes(row['action']).decode(), bytes(row['side']).decode())


def build_day(archive, member, out, route=(1, 41271, 22), fragment=0,
              end_ns=None, block_records=262144):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent
    names = ('adapter.py', 'aggregate.py', 'schema.py', 'storage.py', 'stream.py', 'state.py')
    files = [root/name for name in names] + list((root/'vendor').glob('*.py'))
    code = {str(path.relative_to(root)): sha(path) for path in files}
    config = StreamConfig(*route, fragment=fragment)
    protocol = dict(contract='MBO_STREAM_SOURCE_V3', archive=str(Path(archive).resolve()),
                    member=member, route=list(route), fragment=fragment, end_ns=end_ns, code=code,
                    period_ns=config.snapshot_ns, max_quote_age_ns=config.max_quote_age_ns,
                    availability='exclusive receive watermark; separate bucket close and available_ns',
                    price='all completed F_LAST midpoint',
                    trade_side='provider T B=buy,A=sell,N=unknown; no double counting F',
                    flow='whole-book observed A/C/M; same-group F matched removals; last-hit-capped later additions')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old['protocol'] != protocol:
            raise ValueError('Source adapter changed: new output required')
        for name, expected in old['payloads'].items():
            if sha(out/name) != expected:
                raise ValueError('Source artifact changed')
        return old
    if list(out.glob('*.parquet')):
        raise ValueError('Unfinished source build; use fresh output directory')
    quotes = Sink(out/'quotes.parquet', QUOTE_SCHEMA)
    snapshots = Sink(out/'snapshots.parquet', SNAP_SCHEMA)
    bars = Sink(out/'bars.parquet', BAR_SCHEMA)
    core = MboStream(config, on_quote=quotes.append, on_snapshot=snapshots.append,
                     on_bar=bars.append, features=False)
    source = GlbxZipMboSource(archive, member, block_records=block_records)
    count, started = 0, time.perf_counter()
    try:
        for block in source:
            core.feed_many(records_from_block(block, route, count))
            count += len(block)
            progress = dict(raw=count, selected=core.engine.selected_records, events=quotes.count,
                            seconds=time.perf_counter()-started)
            (out/'progress.json').write_text(json.dumps(progress), encoding='utf8')
        if not source.compressed_sha256 or not core.engine.selected_records:
            raise ValueError('Missing route or incompletely consumed source')
        core.finish(end_ns)
    finally:
        for sink in (quotes, snapshots, bars):
            sink.close()
    receipt = dict(protocol=protocol, raw_records=count, selected_records=core.engine.selected_records,
                   quote_events=quotes.count, snapshots=snapshots.count, bars=bars.count,
                   seconds=time.perf_counter()-started, pending_unpublished_records=len(core.engine.pending),
                   source_sha256=source.compressed_sha256, book_reasons=core.accumulator.reason_counts,
                   payloads={name: sha(out/name) for name in ('quotes.parquet', 'snapshots.parquet', 'bars.parquet')})
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    return receipt
