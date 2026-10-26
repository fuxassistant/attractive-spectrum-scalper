"""Recompute named May feature samples without modifying historical datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.features.pipeline import compute, implementation_identity
from spectrum.market.calendar import MarketCalendar
from baseline_data import sha


SAMPLES = (
    ('features_v1', 'market_v1', 'capture_000'),
    ('features_v1', 'market_v1', 'capture_002'),
    ('features_v2', 'market_v2', 'capture_006'),
    ('features_v2', 'market_v2', 'capture_008'),
    ('features_may_extension_v1', 'market_may_extension_v1', 'capture_012'),
    ('features_may_extension_v1', 'market_may_extension_v1', 'capture_014'),
    ('features_may_roll_v1', 'market_may_roll_v1', 'capture_024'),
    ('features_may_roll_v1', 'market_may_roll_v1', 'capture_026'),
)


def certify(data_root):
    root = Path(data_root)
    results = []
    for feature_set, market_set, capture in SAMPLES:
        feature = root / 'derived' / feature_set / capture
        market = root / 'derived' / market_set / capture
        fm = json.loads((feature / 'manifest.json').read_text())
        mm = json.loads((market / 'manifest.json').read_text())
        identities = {'feature_manifest': sha(feature / 'manifest.json'),
                      'market_manifest': sha(market / 'manifest.json'),
                      'seconds': sha(market / 'seconds.parquet'), 'features': sha(feature / 'features.parquet')}
        if identities['market_manifest'] != fm['market_manifest_sha256'] or identities['seconds'] != fm['source_seconds_sha256'] or identities['features'] != fm['feature_sha256']:
            raise ValueError('Sample source lineage mismatch: ' + str(feature))
        if identities['seconds'] != mm['payloads']['seconds.parquet']:
            raise ValueError('Market seconds checksum mismatch')
        seconds = pq.read_table(market / 'seconds.parquet').to_pandas()
        saved = pq.read_table(feature / 'features.parquet')
        context = compute(seconds, tuple(fm['families']))
        columns = {name: seconds[name].to_numpy() for name in ('t_ns', 'available_ns', 'epoch')}
        columns.update(scheduled_closed=context.closed, admissible=context.admitted)
        for name, values in context.values.items():
            columns[name] = values
            columns[name + '__known'] = context.masks[name]
        if set(saved.column_names) != set(columns) or len(saved) != len(seconds) or fm['units'] != context.units:
            raise ValueError('Feature schema, units or row count mismatch')
        digests = {}; errors = []
        for name in saved.column_names:
            old = saved[name].combine_chunks().to_numpy(zero_copy_only=False)
            new = np.asarray(columns[name])
            if old.dtype != new.dtype or old.tobytes() != new.tobytes():
                errors.append(name)
            digests[name] = hashlib.sha256(new.tobytes()).hexdigest()
        if errors:
            raise ValueError('Non-equivalent sample columns: ' + repr(errors))
        result = dict(dataset='derived/' + feature_set, capture=capture, market='derived/' + market_set + '/' + capture,
                      status='PASS_BITWISE_COLUMN_VALUES', rows=len(seconds), columns=len(columns),
                      first_utc=pd.Timestamp(int(seconds.t_ns.iloc[0]), tz='UTC').isoformat(),
                      last_utc=pd.Timestamp(int(seconds.t_ns.iloc[-1]), tz='UTC').isoformat(),
                      original_implementation=fm['implementation_sha256'], original_calendar=fm['calendar'],
                      hashes=identities, column_sha256=digests)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('hashes', 'column_sha256')}), flush=True)
    return dict(version='FEATURE_EQUIVALENCE_CERTIFICATE_V1', baseline='0.3.0', date='2026-10-27',
                status='PASS_NAMED_SAMPLES_ONLY', current_implementation=implementation_identity(),
                current_calendar=MarketCalendar().contract(), environment_receipt='environment.json',
                units_and_masks_checked=True, parquet_file_bytes_reproduced=False,
                historical_manifests_modified=False, full_history_equivalence=False,
                excluded='Every unlisted capture, full_mbo_v1/features, other seasons, different feature families, execution/target/policy admission',
                samples=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='baselines/0.3.0/feature-equivalence.json')
    args = parser.parse_args()
    receipt = certify(args.data_root)
    Path(args.output).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8', newline='\n')
