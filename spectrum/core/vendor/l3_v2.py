"""Versioned reconstruction: corrected snapshot lifecycle, unchanged order math."""
from __future__ import annotations

from dataclasses import replace
import hashlib

from .l3 import L3EventEngine, MboRecord, _uid


CONTRACT = 'L3_RECONSTRUCTION_V2'


class L3EventEngineV2(L3EventEngine):
    def __init__(self, **kwargs):
        kwargs.setdefault('event_uid_mode', 'SOURCE_COORDINATE')
        super().__init__(**kwargs)
        self.snapshot_fault: str | None = None
        self.snapshot_rows_seen = False
        self._snapshot_record = False

    def append(self, record: MboRecord):
        # Only a new explicit clear starts a recovery attempt. A clean ending
        # record must not erase a fault detected earlier in the same snapshot.
        if record.action == 'R':
            self.snapshot_fault = None
            self.snapshot_rows_seen = False
        self._snapshot_record = record.is_snapshot
        if record.is_snapshot:
            self.snapshot_rows_seen = True
        result = super().append(record)
        self._snapshot_record = False
        if result is not None:
            return replace(result, event_uid=hashlib.sha256((CONTRACT+'|'+result.event_uid).encode()).hexdigest())
        return None

    def _invalidate(self, reason: str) -> None:
        super()._invalidate(reason)
        if self.snapshot_in_progress or self._snapshot_record:
            self.snapshot_fault = self.snapshot_fault or reason

    def _finalize_snapshot(self, record: MboRecord) -> None:
        # Called only at F_LAST; the closing record need not carry F_SNAPSHOT.
        if self.snapshot_in_progress and self.snapshot_rows_seen:
            self.snapshot_in_progress = False
            self.book_ready = self.snapshot_fault is None
            self.quality_reason = self.snapshot_fault or 'OK'
            self.snapshot_rows_seen = False

    def checkpoint(self):
        body = super().checkpoint()
        body.pop('checkpoint_uid')
        body['contract'] = CONTRACT
        body['event_uid_mode'] = self.event_uid_mode
        body['snapshot_fault'] = self.snapshot_fault
        body['snapshot_rows_seen'] = self.snapshot_rows_seen
        return {**body, 'checkpoint_uid': _uid(body)}

    @classmethod
    def restore(cls, checkpoint):
        body = {k:v for k,v in checkpoint.items() if k != 'checkpoint_uid'}
        if body.get('contract') != CONTRACT or checkpoint.get('checkpoint_uid') != _uid(body):
            raise ValueError('V2 checkpoint drift')
        route = body['route']
        result = cls(publisher_id=route[0], instrument_id=route[1], channel_id=route[2], event_uid_mode=body['event_uid_mode'])
        # Reuse only the verified level construction, not the V1 state contract.
        from .l3 import _Order
        for oid, side, price, size in body['orders']:
            result.orders[oid] = _Order(side, price, size)
            result._add_level(side, price, size)
        for key, value in body['state'].items():
            setattr(result, key, value)
        result.snapshot_fault = body['snapshot_fault']
        result.snapshot_rows_seen = body['snapshot_rows_seen']
        return result
