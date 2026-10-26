"""Admission checks of a new finalized release before it is published.

Restored from the archive branch (spectrum/finalized_audit.py and
spectrum/data_semantics_audit.py): only the checks finalize_market calls.
Benchmarks, plots and HTML reports stay in the archive.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from spectrum.market.finalized import FinalizedReplay, read_json
from spectrum.market.source import DbnSource

F_LAST = 128
BBO_COLUMNS = ('bid_nano', 'ask_nano', 'bid_size', 'ask_size')


class ReferenceBook:
    """Small independent dictionary book; shares no code with the production kernel."""

    def __init__(self, expected):
        self.orders = {}
        self.levels = {b'B': Counter(), b'A': Counter()}
        self.expected = expected
        self.rows = self.checks = 0
        self.errors = []
        self.unknown_cancels = self.unknown_modifies = 0

    def remove(self, order_id, size=None):
        side, price, old_size = self.orders[order_id]
        quantity = old_size if size is None else min(size, old_size)
        self.levels[side][price] -= quantity
        if not self.levels[side][price]:
            del self.levels[side][price]
        if quantity == old_size:
            del self.orders[order_id]
        else:
            self.orders[order_id] = (side, price, old_size - quantity)

    def feed(self, record, ordinal):
        self.rows += 1
        action, side = bytes(record['action']), bytes(record['side'])
        order_id, price, size = (int(record[k]) for k in ('order_id', 'price', 'size'))
        if action == b'R':
            self.orders.clear()
            self.levels = {b'B': Counter(), b'A': Counter()}
        elif action in (b'A', b'M'):
            if action == b'A' and order_id in self.orders:
                self.errors.append(dict(ordinal=str(ordinal), reason='DUPLICATE_ADD'))
            if action == b'M' and order_id not in self.orders:
                self.unknown_modifies += 1
            if order_id in self.orders:
                self.remove(order_id)
            if size > 0 and side in self.levels:
                self.orders[order_id] = (side, price, size)
                self.levels[side][price] += size
        elif action == b'C':
            if order_id in self.orders:
                self.remove(order_id, size)
            else:
                self.unknown_cancels += 1
        expected = self.expected.get(ordinal)
        if expected is not None and expected['valid']:
            bid = max(self.levels[b'B'], default=0)
            ask = min(self.levels[b'A'], default=0)
            actual = (bid, ask, self.levels[b'B'].get(bid, 0), self.levels[b'A'].get(ask, 0))
            target = tuple(int(expected[k]) for k in BBO_COLUMNS)
            self.checks += 1
            if actual != target:
                self.errors.append(dict(ordinal=str(ordinal), actual=actual, expected=target))


def compare_canonical(root, legacy):
    """Every publication identity and every valid BBO equals the earlier canonical tape."""
    new = FinalizedReplay(root, columns=['source_ordinal', 'available_ns', 'valid', *BBO_COLUMNS])
    old = pq.ParquetFile(Path(legacy) / 'quotes.parquet')
    if old.metadata.num_rows != new.spec['rows']:
        raise ValueError('Publication count mismatch')
    pending, offset, checked, valid_checked = None, 0, 0, 0
    for old_batch in old.iter_batches(batch_size=32768, columns=new.columns):
        at = 0
        while at < len(old_batch):
            if pending is None or offset == len(pending):
                pending, offset = new.read_batch(), 0
            n = min(len(old_batch) - at, len(pending) - offset)
            a, b = old_batch.slice(at, n), pending.slice(offset, n)
            for column in ('source_ordinal', 'available_ns', 'valid'):
                if not np.array_equal(a.column(column).to_numpy(zero_copy_only=False), b[column].to_numpy()):
                    raise ValueError('Canonical mismatch ' + column)
            valid = b['valid'].to_numpy()
            for column in BBO_COLUMNS:
                if not np.array_equal(a.column(column).to_numpy()[valid], b[column].to_numpy()[valid]):
                    raise ValueError('Canonical BBO mismatch ' + column)
            checked += n
            valid_checked += int(valid.sum())
            at += n
            offset += n
    return dict(status='PASS', publications=checked, valid_bbo_checked=valid_checked,
                limitation='Canonical comparison shares production L3 logic; independent prefix probe is separate')


def independent_prefix(root, config_path, limit=20000):
    """Replay the raw prefix through ReferenceBook and compare the first published BBOs."""
    manifest = read_json(Path(root) / 'manifest.json')
    config = read_json(config_path)
    member = next(x for x in config['members'] if x['key'] == manifest['source_identity']['key'])
    quotes = FinalizedReplay(root, columns=['source_ordinal', 'valid', *BBO_COLUMNS]).read_batch(limit).to_pandas()
    book = ReferenceBook({int(row['source_ordinal']): row for row in quotes.to_dict('records')})
    route = manifest['selection']['route']
    offset, last = 0, None
    for block in DbnSource(member['archive'], member['member']):
        selected = np.flatnonzero((block['publisher_id'] == route[0]) & (block['instrument_id'] == route[1])
                                  & (block['channel_id'] == route[2]))
        for i in selected:
            book.feed(block[i], int(i + offset))
            last = block[i]
            if book.rows >= limit and int(last['flags']) & F_LAST:
                break
        if book.rows >= limit and last is not None and int(last['flags']) & F_LAST:
            break
        offset += len(block)
    if book.errors:
        raise ValueError('Independent prefix mismatch ' + repr(book.errors[:3]))
    return dict(status='PASS', raw_records=book.rows, valid_publications_checked=book.checks,
                unknown_modifies=book.unknown_modifies, unknown_cancels=book.unknown_cancels,
                limitation='Bounded prefix from explicit reset, not full independent reconstruction')
