"""Bounded original DBN records: files or ZIP members, integer source order."""
from pathlib import Path
from contextlib import ExitStack
from zipfile import ZipFile
import hashlib
import struct
import numpy as np
import zstandard as zstd
from .core.vendor.glbx_dbn import MBO_DTYPE, _HashingReader, _read_exact


class DbnSource:
    def __init__(self, path, member=None, block_records=65536):
        self.path = Path(path)
        self.member = member
        self.block_records = block_records
        self.sha256 = None
        self.count = 0

    def __iter__(self):
        self.sha256 = None
        self.count = 0
        h = hashlib.sha256()
        with ExitStack() as stack:
            if self.member:
                archive = stack.enter_context(ZipFile(self.path))
                raw = stack.enter_context(archive.open(self.member))
            else:
                raw = stack.enter_context(self.path.open('rb'))
            reader = stack.enter_context(zstd.ZstdDecompressor().stream_reader(_HashingReader(raw, h), read_across_frames=True))
            header = _read_exact(reader, 8)
            if header[:3] != b'DBN':
                raise ValueError('Expected DBN header')
            _read_exact(reader, struct.unpack_from('<I', header, 4)[0])
            carry = b''
            while chunk := reader.read(self.block_records * 56):
                payload = carry + chunk
                n = len(payload) // 56 * 56
                carry = payload[n:]
                block = np.frombuffer(payload[:n], dtype=MBO_DTYPE)
                if np.any(block['length'] != 14) or np.any(block['rtype'] != 160):
                    raise ValueError('Non-MBO/fixed-width record')
                self.count += len(block)
                yield block
            if carry:
                raise ValueError('Truncated MBO record')
        self.sha256 = h.hexdigest()
