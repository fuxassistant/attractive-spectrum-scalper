"""Public dataset entry point for the later sequence encoder/player."""
import json
from pathlib import Path
import pandas as pd
from .representation import MarketObservationStore


def open_observation_store(root, date):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('Observation dataset is not complete')
    item = next((x for x in manifest['inventory'] if x['date'] == date), None)
    if item is None or item['status'] != 'MATERIALIZED':
        raise ValueError('Requested date lacks an admitted previous-day context')
    out = root / 'observations'
    return MarketObservationStore(
        pd.read_parquet(out / (date + '_fast.parquet')),
        pd.read_parquet(out / (date + '_context.parquet')),
        pd.read_parquet(out / (date + '_zones.parquet')),
        pd.read_parquet(out / (date + '_events.parquet')),
        symbol=item['symbol'], previous_date=item['previous'], date=date)
