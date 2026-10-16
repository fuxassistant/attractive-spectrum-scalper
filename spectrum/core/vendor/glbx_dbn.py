"""Bounded fixed-width GLBX MBO reader; no feature or target semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Iterator
from zipfile import ZipFile

import numpy as np
import zstandard as zstd

MBO_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"),
    ("instrument_id", "<u4"), ("ts_event", "<u8"), ("order_id", "<u8"),
    ("price", "<i8"), ("size", "<u4"), ("flags", "u1"),
    ("channel_id", "u1"), ("action", "S1"), ("side", "S1"),
    ("ts_recv", "<u8"), ("ts_in_delta", "<i4"), ("sequence", "<u4"),
], align=False)
MBO_RECORD_BYTES = 56
MBO_RTYPE = 160


class _HashingReader:
    def __init__(self, source: object, digest: object) -> None:
        self.source, self.digest = source, digest

    def read(self, size: int = -1) -> bytes:
        value = self.source.read(size)
        if value:
            self.digest.update(value)
        return value

    def readinto(self, buffer: object) -> int:
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count

    def readable(self) -> bool:
        return True


def _read_exact(reader: object, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = reader.read(size - len(result))
        if not chunk:
            raise ValueError("truncated DBN metadata")
        result.extend(chunk)
    return bytes(result)


class GlbxZipMboSource:
    """Streams one DBN+zstd ZIP member as validated bounded NumPy blocks."""

    def __init__(self, zip_path: str | Path, member: str, *, block_records: int = 262_144) -> None:
        if block_records < 1_024:
            raise ValueError("block_records must be >= 1024")
        self.zip_path = Path(zip_path).resolve()
        self.member = member
        self.block_records = int(block_records)
        self.compressed_sha256 = ""
        self.metadata_bytes = 0
        self.record_count = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        digest = hashlib.sha256()
        self.record_count = 0
        with ZipFile(self.zip_path) as archive, archive.open(self.member) as raw:
            source = _HashingReader(raw, digest)
            with zstd.ZstdDecompressor().stream_reader(source, read_across_frames=True) as reader:
                header = _read_exact(reader, 8)
                if header[:3] != b"DBN":
                    raise ValueError("member is not DBN")
                self.metadata_bytes = struct.unpack_from("<I", header, 4)[0]
                _read_exact(reader, self.metadata_bytes)
                carry = b""
                target = self.block_records * MBO_RECORD_BYTES
                while True:
                    chunk = reader.read(target)
                    if not chunk:
                        break
                    payload = carry + chunk
                    complete = len(payload) // MBO_RECORD_BYTES * MBO_RECORD_BYTES
                    carry = payload[complete:]
                    if not complete:
                        continue
                    block = np.frombuffer(payload[:complete], dtype=MBO_DTYPE)
                    if not np.all(block["length"] == MBO_RECORD_BYTES // 4):
                        raise ValueError("invalid fixed MBO record length")
                    if not np.all(block["rtype"] == MBO_RTYPE):
                        raise ValueError("non-MBO record in payload")
                    self.record_count += len(block)
                    yield block
                if carry:
                    raise ValueError("truncated fixed-width MBO payload")
        self.compressed_sha256 = digest.hexdigest()


def benchmark_source(source: GlbxZipMboSource, *, max_records: int) -> dict[str, object]:
    started = time.perf_counter()
    rows = blocks = bytes_uncompressed = 0
    for block in source:
        rows += len(block)
        blocks += 1
        bytes_uncompressed += block.nbytes
        if rows >= max_records:
            break
    elapsed = time.perf_counter() - started
    return {
        "contract": "GLBX_DBN_SOURCE_BENCHMARK_V1", "records": rows, "blocks": blocks,
        "elapsed_seconds": elapsed, "records_per_second": rows / elapsed,
        "uncompressed_mib_per_second": bytes_uncompressed / elapsed / 2**20,
        "full_member_consumed": bool(source.compressed_sha256),
        "compressed_sha256": source.compressed_sha256 or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("member")
    parser.add_argument("--max-records", type=int, default=1_000_000)
    parser.add_argument("--block-records", type=int, default=262_144)
    args = parser.parse_args()
    print(json.dumps(benchmark_source(
        GlbxZipMboSource(args.zip_path, args.member, block_records=args.block_records),
        max_records=args.max_records,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
