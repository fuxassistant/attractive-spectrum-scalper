"""Replay committed exact BBO from verified Parquet, with a resumable cursor."""
from pathlib import Path
import json
import pyarrow.parquet as pq
from .core.storage import sha


class TapeReplay:
    def __init__(self,capture,*,verify=True):
        self.root=Path(capture)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.identity=sha(self.root/'manifest.json')
        if self.manifest['status']!='COMPLETE':raise ValueError('Incomplete capture')
        if verify and sha(self.root/'quotes.parquet')!=self.manifest['payloads']['quotes.parquet']:
            raise ValueError('Quote payload drift')
        self.cursor=0;self.current=None;self._rows=self._iterate();self._pending=None

    def _iterate(self):
        for batch in pq.ParquetFile(self.root/'quotes.parquet').iter_batches(batch_size=32768):
            names=batch.schema.names
            for row in zip(*(batch.column(i).to_pylist() for i in range(batch.num_columns))):
                yield dict(zip(names,row))

    def __iter__(self):return self
    def __next__(self):
        row=self._pending if self._pending is not None else next(self._rows)
        self._pending=None;self.current=row;self.cursor+=1
        return dict(row)

    def advance(self,until_ns):
        if self.current and until_ns<self.current['available_ns']:raise ValueError('Replay cannot rewind')
        while True:
            if self._pending is None:
                try:self._pending=next(self._rows)
                except StopIteration:break
            if self._pending['available_ns']>until_ns:break
            next(self)
        if self.current is None:return None
        result=dict(self.current)
        age=until_ns-result['t_ns']
        result.update(query_ns=int(until_ns),quote_age_ns=int(age),
                      admissible_at_query=bool(result.get('valid',False) and 0<=age<=30*10**9))
        return result

    def checkpoint(self):return dict(identity=self.identity,cursor=self.cursor)

    @classmethod
    def restore(cls,capture,state):
        obj=cls(capture)
        if obj.identity!=state['identity']:raise ValueError('Replay input drift')
        for _ in range(state['cursor']):next(obj)
        return obj
