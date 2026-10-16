"""Optional Parquet output. The streaming kernel never imports this module."""
from pathlib import Path
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class Sink:
    def __init__(self, path, schema, size=32768):
        if size <= 0:
            raise ValueError('Output buffer must be positive')
        self.names = [name for name, _ in schema]
        self.buffer = np.empty(size, np.dtype(schema))
        self.n = self.count = 0
        self.path = Path(path)
        self.writer = pq.ParquetWriter(self.path, pa.schema([
            (name, pa.from_numpy_dtype(np.dtype(dtype))) for name, dtype in schema
        ]), compression='zstd')

    def append(self, row):
        self.buffer[self.n] = tuple(row)
        self.n += 1
        self.count += 1
        if self.n == len(self.buffer):
            self.flush()

    def flush(self):
        if self.n:
            arrays = [pa.array(self.buffer[name][:self.n]) for name in self.names]
            self.writer.write_table(pa.Table.from_arrays(arrays, names=self.names))
            self.n = 0

    def close(self):
        self.flush()
        self.writer.close()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
