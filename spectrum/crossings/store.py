"""Immutable factual bank storage, independent of model, policy and research reports."""
import json
from pathlib import Path
import pandas as pd
from spectrum.io import save_json, sha
from .bank import build_bank, geometry_definition


def publish_bank(quotes, output, *, source_id: str, ddof: int = 1) -> dict:
    if not source_id:
        raise ValueError('Explicit source identity required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'manifest.json', {'status': 'RUNNING', 'source_id': source_id})
    try:
        bank, _ = build_bank(quotes, source_id, ddof=ddof)
        if bank.event_id.duplicated().any():
            raise ValueError('Crossing identities must be unique')
        path = root / 'events.parquet'
        bank.to_parquet(path, index=False)
        manifest = dict(version='CROSSING_BANK_LIBRARY_V1', status='COMPLETE', source_id=source_id,
                        geometry=geometry_definition(ddof), events=len(bank), quotes=len(quotes.quote_t),
                        files={'events.parquet': sha(path)},
                        semantics='Factual full-publication bank. Future queries and policy are separate consumers.')
        save_json(root / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        save_json(root / 'manifest.json', dict(status='FAILED', error=repr(exc), source_id=source_id))
        raise


def open_bank(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 'CROSSING_BANK_LIBRARY_V1' or manifest.get('status') != 'COMPLETE':
        raise ValueError('A complete library crossing bank is required')
    path = root / 'events.parquet'
    if sha(path) != manifest['files']['events.parquet']:
        raise ValueError('Crossing bank payload changed')
    bank = pd.read_parquet(path)
    if len(bank) != manifest['events'] or bank.event_id.duplicated().any():
        raise ValueError('Crossing bank shape/identity mismatch')
    return bank, manifest
