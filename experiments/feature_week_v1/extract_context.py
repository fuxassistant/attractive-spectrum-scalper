"""Bounded, immutable exact interaction extraction for the week readiness audit.

The route is declared from Friday's existing selection before Sunday's open.
No new contract is selected using facts from the following Monday.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from spectrum.core import StreamConfig
from spectrum.io import save_json, sha
from spectrum.market.ingest import PACKED, packed_records
from spectrum.market.source import DbnSource
from spectrum.representation.stream import interaction_stream

CATALOG = ROOT / 'data/derived/full_mbo_v1'
OUTPUT = ROOT / 'data/derived/feature_week_v1'
DATES = ('2026-08-21', '2026-08-23', '2026-08-24', '2026-08-25',
         '2026-08-26', '2026-08-27', '2026-08-28')
WEEK = DATES[2:]
ROUTE = (1, 41271, 22)


def source_for(date):
    catalog = json.loads((CATALOG / 'manifest.json').read_text())
    items = [r for r in catalog['captures'] if r['status'] == 'COMPLETE'
             and r['entry']['selection']['date'] == date
             and tuple(r['entry']['route']) == ROUTE]
    if len(items) != 1:
        raise ValueError(f'Expected one complete fixed-route capture: {date}')
    return items[0]


class Writer:
    def __init__(self, path):
        self.path, self.rows, self.writer, self.n = Path(path), [], None, 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= 20000:
            self.flush()

    def add_many(self, rows):
        self.rows.extend(rows)
        if len(self.rows) >= 20000:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pandas(pd.DataFrame(self.rows), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='zstd')
        self.writer.write_table(table)
        self.n += len(self.rows)
        self.rows = []

    def close(self):
        self.flush()
        if self.writer:
            self.writer.close()


def extract(date):
    source = source_for(date)
    entry = source['entry']
    config = json.loads((CATALOG / 'run_config.json').read_text())
    item = next(x for x in config['members'] if x['key'] == entry['member_key'])
    out = OUTPUT / 'raw' / date
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save_json(out / 'manifest.json', dict(status='RUNNING', pid=os.getpid(), date=date))
    levels, trades = Writer(out / 'levels_1s.partial.parquet'), Writer(out / 'trades.partial.parquet')
    # UTC captures start with an explicit source reset, not a guessed book.
    fragment = int(Path(source['market']).name.split('_')[-1]) + 1
    stream = interaction_stream(StreamConfig(*ROUTE, snapshot_ns=1_000_000_000, fragment=fragment),
                                emit_levels=levels.add_many, emit_trades=trades.add)
    collector = stream.metrics[0].collector
    src = DbnSource(item['archive'], item['member'])
    offset = selected = 0
    digest = hashlib.sha256()
    rss_peak = 0
    try:
        for block in src:
            ids = np.flatnonzero((block['publisher_id'] == ROUTE[0])
                                & (block['instrument_id'] == ROUTE[1])
                                & (block['channel_id'] == ROUTE[2]))
            ids = ids[(ids + offset >= entry['first_ordinal']) & (ids + offset <= entry['last_ordinal'])]
            packed = np.empty(len(ids), PACKED)
            packed['source_ordinal'], packed['record'] = ids + offset, block[ids]
            digest.update(packed.tobytes())
            stream.feed_many(packed_records(packed, ROUTE))
            selected += len(ids)
            offset += len(block)
            rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
            save_json(out / 'progress.json', dict(pid=os.getpid(), date=date, selected=selected,
                      total=entry['records'], seconds=time.perf_counter()-start,
                      events=collector.stats['events'], rss_peak_sampled=rss_peak))
        if selected != entry['records'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Raw partition identity differs')
        if src.sha256 != item['sha256'] or stream.engine.pending:
            raise ValueError('Source member identity or publication completion differs')
        collector.finish()  # No extrapolated final partial second.
        levels.close()
        trades.close()
        for name in ('levels_1s', 'trades'):
            (out / (name + '.partial.parquet')).replace(out / (name + '.parquet'))
        result = dict(status='COMPLETE', date=date, pid=os.getpid(), route=list(ROUTE), symbol='GCZ6',
                      route_declared_from='2026-08-21', rows=dict(levels=levels.n, trades=trades.n),
                      stats=collector.stats, elapsed_seconds=time.perf_counter()-start,
                      rss_peak_sampled=rss_peak,
                      source=dict(archive=item['archive'], member=item['member'], member_sha256=src.sha256,
                                  partition_sha256=digest.hexdigest(), records=selected,
                                  first_ordinal=entry['first_ordinal'], last_ordinal=entry['last_ordinal']),
                      outputs={n: sha(out / n) for n in ('levels_1s.parquet', 'trades.parquet')},
                      implementation={str(p.relative_to(ROOT)): sha(p) for p in
                                      [Path(__file__), ROOT / 'spectrum/representation/raw.py',
                                       ROOT / 'spectrum/representation/stream.py', ROOT / 'spectrum/core/stream.py']})
        save_json(out / 'manifest.json', result)
        return dict(date=date, seconds=result['elapsed_seconds'], rows=result['rows'],
                    stock_residual=collector.stats['max_stock_residual'])
    except Exception as exc:
        levels.close()
        trades.close()
        save_json(out / 'manifest.json', dict(status='FAILED', date=date, error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dates', nargs='+', default=['2026-08-21', '2026-08-23'])
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(extract, d) for d in a.dates]):
            print(json.dumps(f.result()), flush=True)
