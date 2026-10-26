"""Public dataset entry point for the later sequence encoder/player."""
import json
from pathlib import Path
from datetime import date as CalendarDate
import pandas as pd
from spectrum.io import save_json, sha
from spectrum.representation.representation import MarketObservationStore
from spectrum.features.contract import reject_future


def open_observation_store(root, date):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('Observation dataset is not complete')
    item = next((x for x in manifest['inventory'] if x['date'] == date), None)
    if item is None or item['status'] != 'MATERIALIZED':
        raise ValueError('Requested date lacks an admitted previous-day context')
    out = root / 'observations'
    if manifest.get('version') == 'OBSERVATION_STORE_LIBRARY_V1':
        for suffix in ('fast', 'context', 'zones', 'events'):
            name = date + '_' + suffix + '.parquet'
            if sha(out / name) != manifest['files'][name]:
                raise ValueError('Observation payload changed: ' + name)
    return MarketObservationStore(
        pd.read_parquet(out / (date + '_fast.parquet')),
        pd.read_parquet(out / (date + '_context.parquet')),
        pd.read_parquet(out / (date + '_zones.parquet')),
        pd.read_parquet(out / (date + '_events.parquet')),
        symbol=item['symbol'], previous_date=item['previous'], date=date)


def publish_observation_store(root, *, date, previous_date, symbol, fast, context,
                              zones, events, source_identity):
    """Publish one immutable observation dataset from explicitly supplied tables.

    No experiment dates, report directories or historical player loaders are
    involved. Existing historical manifests remain readable through open().
    """
    current = CalendarDate.fromisoformat(date)
    previous = CalendarDate.fromisoformat(previous_date)
    if previous >= current or not symbol or not source_identity:
        raise ValueError('Earlier context date, symbol and explicit source identity required')
    for frame in (fast, context, zones, events):
        reject_future(frame.columns, frame)
        if 'available_ns' not in frame:
            raise ValueError('Every observation table requires its publication clock')
        if 't_ns' in frame and (frame.available_ns < frame.t_ns).any():
            raise ValueError('Observation publication precedes its endpoint')
    if not fast.available_ns.is_monotonic_increasing or not context.available_ns.is_monotonic_increasing:
        raise ValueError('Observation publication clocks must be ordered')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    out = root / 'observations'
    out.mkdir()
    save_json(root / 'manifest.json', dict(status='RUNNING', date=date))
    try:
        files = {}
        for suffix, frame in (('fast', fast), ('context', context), ('zones', zones), ('events', events)):
            name = date + '_' + suffix + '.parquet'
            frame.to_parquet(out / name, index=False)
            files[name] = sha(out / name)
        manifest = dict(version='OBSERVATION_STORE_LIBRARY_V1', status='COMPLETE',
                        source_identity=source_identity, files=files,
                        inventory=[dict(date=date, previous=previous_date, symbol=symbol, status='MATERIALIZED')],
                        semantics='Causal observations only; future outcomes are a separate dataset')
        save_json(root / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        save_json(root / 'manifest.json', dict(status='FAILED', date=date, error=repr(exc)))
        raise
