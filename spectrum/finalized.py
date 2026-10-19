"""Immutable, indexed, columnar market replay. No reconstruction in the reader."""
from pathlib import Path
import bisect
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = 'FINALIZED_MARKET_V2'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class FinalizedReplay:
    """A cursor addresses a row group and an offset, not a replayed prefix.

    Full payload verification is an explicit administrative operation. Open
    checks version, publication status, and file sizes only. The published
    directory is an immutable trust boundary, not protection from silent disk
    corruption (run verify after copying or suspected modification).
    """
    def __init__(self, root, layer='quotes', columns=None):
        self.root = Path(root)
        self.manifest = read_json(self.root/'manifest.json')
        if self.manifest.get('contract') != VERSION or self.manifest.get('status') != 'COMPLETE':
            raise ValueError('Finalized replay requires a complete compatible publication')
        self.identity = self.manifest['identity']
        self.layer = layer
        self.columns = columns
        self.spec = self.manifest['layers'][layer]
        path = self.root/self.spec['file']
        if path.stat().st_size != self.spec['bytes']:
            raise ValueError('Replay size changed; run verify')
        self.file = pq.ParquetFile(path)
        self.groups = self.spec['groups']
        self._ends = [g['last_available_ns'] for g in self.groups]
        self.group = self.offset = 0
        self._cache_group = -1
        self._cache = None
        self._current = None
        self._query_ns = None

    def _table(self):
        if self._cache_group != self.group:
            self._cache = self.file.read_row_group(self.group, columns=self.columns)
            self._cache_group = self.group
        return self._cache

    def read_batch(self, max_rows=65536):
        if max_rows <= 0:
            raise ValueError('max_rows must be positive')
        if self.group >= len(self.groups):
            return None
        table = self._table()
        n = min(max_rows, len(table)-self.offset)
        result = table.slice(self.offset, n)
        self.offset += n
        if self.offset == len(table):
            self.group += 1
            self.offset = 0
        return result

    def batches(self, start_ns=None, end_ns=None):
        if start_ns is not None:
            self.seek(start_ns)
        while (table := self.read_batch()) is not None:
            if end_ns is not None:
                times = table['available_ns'].to_numpy()
                n = int(np.searchsorted(times, end_ns, side='left'))
                if n < len(table):
                    if n:
                        yield table.slice(0, n)
                    return
            yield table

    def seek(self, available_ns):
        """First row at or after available_ns, preserving equal-time order."""
        self.group = bisect.bisect_left(self._ends, int(available_ns))
        self.offset = 0
        self._current = None
        self._query_ns = None
        if self.group < len(self.groups):
            times = self.file.read_row_group(self.group, columns=['available_ns'])['available_ns'].to_numpy()
            self.offset = int(np.searchsorted(times, available_ns, side='left'))
        return self.checkpoint()

    def advance(self, until_ns):
        """Consume complete publications through a monotone query watermark.

        The returned row is the last observation, not a fabricated new quote.
        Age is informational; no quality threshold is inferred here.
        """
        if self._query_ns is not None and until_ns < self._query_ns:
            raise ValueError('Replay cannot rewind; use seek explicitly')
        if self.columns is not None and 'available_ns' not in self.columns:
            raise ValueError('advance requires available_ns in projected columns')
        while self.group < len(self.groups):
            table=self._table()
            n=int(np.searchsorted(table['available_ns'].to_numpy(),until_ns,side='right'))
            if n>self.offset:
                self._current=table.slice(n-1,1)
                self.offset=n
            if self.offset==len(table):
                self.group+=1;self.offset=0
            else:break
        self._query_ns=int(until_ns)
        if self._current is None:return None
        row=self._current.to_pylist()[0]
        row.update(query_ns=int(until_ns),quote_age_ns=int(until_ns)-int(row['available_ns']))
        return row

    def checkpoint(self):
        return dict(identity=self.identity, layer=self.layer, group=self.group, offset=self.offset,query_ns=self._query_ns)

    @classmethod
    def restore(cls, root, cursor, columns=None):
        obj = cls(root, cursor['layer'], columns)
        if obj.identity != cursor['identity']:
            raise ValueError('Replay identity changed')
        g, o = cursor['group'], cursor['offset']
        if type(g) is not int or type(o) is not int or not 0 <= g <= len(obj.groups):
            raise ValueError('Invalid cursor')
        if o < 0 or (g == len(obj.groups) and o != 0) or (g < len(obj.groups) and o >= obj.groups[g]['rows']):
            raise ValueError('Invalid cursor offset')
        obj.group, obj.offset = g, o
        obj._query_ns=cursor.get('query_ns')
        if obj._query_ns is not None and (g or o):
            previous_group=g if o else g-1
            previous=obj.file.read_row_group(previous_group,columns=columns)
            obj._current=previous.slice(o-1 if o else len(previous)-1,1)
        return obj

    def read_window(self, start_ns, end_ns):
        if self.columns is not None and 'available_ns' not in self.columns:
            reader=FinalizedReplay(self.root,self.layer,[*self.columns,'available_ns'])
            return reader.read_window(start_ns,end_ns).select(self.columns)
        parts = list(self.batches(start_ns, end_ns))
        return pa.concat_tables(parts) if parts else pa.Table.from_batches([], schema=self.file.schema_arrow if self.columns is None else pa.schema([self.file.schema_arrow.field(n) for n in self.columns]))


def verify(root):
    """Explicit expensive verification, never called implicitly by replay."""
    from .core.storage import sha
    from .core.schema import digest
    root = Path(root)
    m = read_json(root/'manifest.json')
    if m.get('contract') != VERSION or m.get('status') != 'COMPLETE':
        raise ValueError('Not a complete publication')
    body = {k:v for k,v in m.items() if k != 'identity'}
    if digest(body) != m['identity']:
        raise ValueError('Manifest identity mismatch')
    for name, spec in m['payloads'].items():
        path = root/name
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError('Payload mismatch: '+name)
    for layer in m['layers']:
        reader = FinalizedReplay(root, layer)
        if reader.file.metadata.num_rows != reader.spec['rows']:
            raise ValueError('Row count mismatch: '+layer)
    return dict(status='PASS', identity=m['identity'], files=len(m['payloads']))


def restore_l3(root, available_ns):
    """Apply prevalidated resulting order mutations; no raw decoder or QA."""
    import zstandard
    root = Path(root)
    m = read_json(root/'manifest.json')
    eligible = [c for c in m['checkpoints'] if c['available_ns'] <= available_ns]
    orders = {}
    after = -1
    if eligible:
        c = eligible[-1]
        payload = zstandard.ZstdDecompressor().decompress((root/c['file']).read_bytes())
        state = json.loads(payload)
        orders = {int(oid):(side, price, size) for oid,side,price,size in state['orders']}
        after = c['source_ordinal']
        start = c['available_ns']
    else:
        start = 0
    reader = FinalizedReplay(root, 'l3')
    for table in reader.batches(start, int(available_ns)+1):
        columns = {n:table[n].to_numpy() for n in table.column_names}
        for i in range(len(table)):
            if columns['source_ordinal'][i] <= after:
                continue
            action = columns['action'][i]
            if action == ord('R'):
                orders.clear()
            if not columns['accepted'][i] or action not in (ord('A'),ord('M'),ord('C')):
                continue
            oid = int(columns['order_id'][i])
            size = int(columns['resulting_size'][i])
            if size:
                orders[oid] = (chr(columns['resulting_side'][i]), int(columns['resulting_price_nano'][i]), size)
            else:
                orders.pop(oid, None)
    return orders


class FinalizedSession:
    """Small adapter shared by feature jobs and gym input preparation."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = FinalizedReplay(root).manifest
        self.bounds = self.manifest['session']

    def quotes(self, activity_only=False, columns=None):
        reader = FinalizedReplay(self.root, columns=columns)
        if activity_only:
            return reader.batches(self.bounds['london_ns'], self.bounds['ny_close_ns'])
        return reader.batches()

    def seconds(self):
        return FinalizedReplay(self.root, 'seconds').read_window(0, 2**63-1).to_pandas()

    def segments(self):
        return read_json(self.root/'segments.json')

    def features(self, families=('motion','flow','spectral')):
        from .lab import compute
        return compute(self.seconds(), families=families)
