"""Bind the materialized bank's contracts and marker counts to the data passport."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq
from spectrum.crossings.bank import geometry_definition
from spectrum.crossings.store import VERSION
from baseline_data import sha


def certify(data_root, receipt_root):
    data, receipts = Path(data_root), Path(receipt_root)
    bank = data / 'derived/bb_crossing_bank_v3'
    index = {r['path']: r['sha256'] for r in
             map(json.loads, (receipts / 'data-files.jsonl').read_text().splitlines())}
    def bound(path):
        if sha(path) != index[path.relative_to(data).as_posix()]:
            raise ValueError('Passport drift: ' + str(path))
    bound(bank / 'manifest.json')
    top = json.loads((bank / 'manifest.json').read_text())
    geometry = json.loads(json.dumps(geometry_definition()))
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(top['status'] == 'COMPLETE' and top['geometry'] == geometry, 'Bank geometry/status mismatch')
    totals = dict(events=0, line_visit=0, opportunity=0)
    days = []
    for entry in top['entries']:
        day = bank / entry['date']
        bound(day / 'manifest.json')
        manifest = json.loads((day / 'manifest.json').read_text())
        require(manifest['version'] == VERSION and manifest['status'] == 'COMPLETE', 'Day version/status mismatch')
        require(manifest['geometry'] == geometry, 'Day geometry mismatch')
        for name, digest in manifest['files'].items():
            bound(day / name)
            require(index[(day / name).relative_to(data).as_posix()] == digest, 'Day payload receipt mismatch')
        events = pq.read_table(day / 'events.parquet', columns=['line_visit', 'opportunity'])
        require(events.num_rows == manifest['events'] == entry['events'], 'Event row count mismatch')
        require(pq.ParquetFile(day / 'tape.parquet').metadata.num_rows == manifest['tape_rows'], 'Tape row count mismatch')
        totals['events'] += events.num_rows
        for name in ('line_visit', 'opportunity'):
            totals[name] += int(events[name].to_numpy().sum())
        days.append(entry['date'])
    require(len(set(days)) == len(days) == top['days'], 'Day coverage mismatch')
    require(totals['events'] == top['events'], 'Total event count mismatch')
    return dict(status='PASS_STORED_CONTRACT_AND_PAYLOADS', date='2026-10-27',
                dataset='derived/bb_crossing_bank_v3', days=len(days),
                first_day=min(days), last_day=max(days), totals=totals,
                storage_version=VERSION, geometry=geometry,
                manifest_sha256=sha(bank / 'manifest.json'),
                replay_performed=False, trading_or_training_admission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--receipt-root', default='baselines/0.3.0')
    parser.add_argument('--output', default='baselines/0.3.0/bank-compatibility.json')
    args = parser.parse_args()
    result = certify(args.data_root, args.receipt_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(result))
