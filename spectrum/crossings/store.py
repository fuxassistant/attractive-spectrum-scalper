"""Immutable crossing bank storage: past crossing facts with cluster markers plus the reduced quote tape.

The tape (publication time, bid, ask, band segment per scale) is what
spectrum.future needs to look forward from any origin without rebuilding.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.io import save_json, sha
from .bank import SCALES, build_bank, geometry_definition

VERSION = 'CROSSING_BANK_LIBRARY_V4'


def publish_bank(quotes, output, *, source_id: str, ddof: int = 1, cooldown_ns=None, coverage_end_ns=None, meta=None) -> dict:
    if not source_id:
        raise ValueError('Explicit source identity required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'manifest.json', {'status': 'RUNNING', 'source_id': source_id})
    try:
        bank, tape = build_bank(quotes, source_id, ddof=ddof, cooldown_ns=cooldown_ns)
        if bank.event_id.duplicated().any():
            raise ValueError('Crossing identities must be unique')
        bank.to_parquet(root / 'events.parquet', index=False)
        pd.DataFrame(dict(t_ns=tape['t'], bid=tape['bid'], ask=tape['ask'],
                          **{f'support_{s}': tape['support'][s] for s in SCALES})).to_parquet(root / 'tape.parquet', index=False)
        end = int(quotes.quote_t[-1]) + 1 if coverage_end_ns is None else int(coverage_end_ns)
        manifest = dict(version=VERSION, status='COMPLETE', source_id=source_id, meta=meta or {},
                        geometry=geometry_definition(ddof, cooldown_ns), events=len(bank), quotes=len(quotes.quote_t),
                        tape_rows=len(tape['t']), coverage_end_ns=end,
                        files={name: sha(root / name) for name in ('events.parquet', 'tape.parquet')},
                        semantics='Past crossing facts (bid OR ask rule) at their own t_ns, flicker included; '
                                  'line_visit/opportunity/cluster markers use only earlier crossings; no future, no money.')
        save_json(root / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        save_json(root / 'manifest.json', dict(status='FAILED', error=repr(exc), source_id=source_id))
        raise


def open_bank(root):
    """Return (bank, tape, manifest); tape is shaped like build_bank's second result."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION or manifest.get('status') != 'COMPLETE':
        raise ValueError('A complete library crossing bank is required')
    for name, digest in manifest['files'].items():
        if sha(root / name) != digest:
            raise ValueError('Crossing bank payload changed: ' + name)
    bank = pd.read_parquet(root / 'events.parquet')
    frame = pd.read_parquet(root / 'tape.parquet')
    if len(bank) != manifest['events'] or bank.event_id.duplicated().any() or len(frame) != manifest['tape_rows']:
        raise ValueError('Crossing bank shape/identity mismatch')
    tape = dict(t=frame.t_ns.to_numpy(np.int64), bid=frame.bid.to_numpy(), ask=frame.ask.to_numpy(),
                support={s: frame[f'support_{s}'].to_numpy(np.int64) for s in SCALES})
    return bank, tape, manifest
