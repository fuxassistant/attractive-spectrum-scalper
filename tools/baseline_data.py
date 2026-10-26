"""Read-only inventory and verification of local baseline data (no replay or relabeling)."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: ' + str(path))
    return h.hexdigest()


def files_under(root):
    def fail(error):
        raise error
    files = []
    for directory, _, names in os.walk(root, onerror=fail):
        for name in names:
            p = Path(directory) / name
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError('External link in inventory: ' + str(p))
            files.append(p)
    return sorted(files)


def disposition(name):
    if name == 'bb_crossing_bank_v3':
        return 'CURRENT_FACT_BANK', 'CROSSING_BANK_LIBRARY_V4; current geometry, no learner admission'
    if name in ('features_v1', 'features_v2', 'features_may_extension_v1', 'features_may_roll_v1'):
        return 'LEGACY_FEATURES_SAMPLE_CERTIFIED', 'Only named samples in feature-equivalence.json are certified; original identities retained'
    if name in ('market_v1', 'market_v2', 'market_may_extension_v1', 'market_may_roll_v1'):
        return 'FEATURE_SOURCE_TAPES', 'Frozen source tapes for the May feature snapshots'
    if name == 'full_mbo_v1':
        return 'HISTORICAL_MBO_AND_FEATURES', '291 capture receipts; multi-month features exist but are NOT certified for current code'
    if name == 'basket_multimonth_v1':
        return 'BANK_SOURCE_CATALOG', 'Day selections and references used by the bank; historical derived views are not current X'
    if name.startswith('bb_crossing_bank_'):
        return 'SUPERSEDED_BANK', 'Historical geometry/storage; do not open as the current V4 bank'
    if name.startswith('raw_'):
        return 'HISTORICAL_RAW_PARTITIONS', 'Preserved partitions, not an independently repeated MBO replay'
    if name.startswith('finalized_'):
        return 'PARTIAL_OR_PILOT', 'Inspect original receipt statuses; not admitted as full-history baseline data'
    return 'ARCHIVED_RESEARCH', 'Retained for archaeology; no current feature, target, policy or training admission'


def inventory(root, output):
    root, output = Path(root).resolve(), Path(output)
    scopes = ('derived', 'legacy')
    output.mkdir(parents=True, exist_ok=True)
    rows = []; datasets = []
    for scope in scopes:
        if not (root / scope).is_dir():
            raise FileNotFoundError(root / scope)
        for dataset in sorted((root / scope).iterdir()):
            if not dataset.is_dir():
                raise ValueError('Unclassified file at scope root: ' + str(dataset))
            parts = []; total = 0
            for p in files_under(dataset):
                row = dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                line = json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n'
                rows.append(line); parts.append(line); total += row['bytes']
            status, note = disposition(dataset.name)
            receipts = []
            for filename in ('manifest.json', 'catalog.json', 'run.json', 'training_catalog.json'):
                p = dataset / filename
                if p.exists():
                    value = json.loads(p.read_text(encoding='utf8'))
                    receipts.append(dict(path=p.relative_to(root).as_posix(), sha256=sha(p),
                                         declared_status=value.get('status'),
                                         declared_version=value.get('version', value.get('contract'))))
            datasets.append(dict(path=dataset.relative_to(root).as_posix(), status=status, note=note,
                                 files=len(parts), bytes=total, receipts=receipts,
                                 tree_sha256=hashlib.sha256(''.join(parts).encode()).hexdigest()))
            print(json.dumps(dict(dataset=dataset.name, files=len(parts), bytes=total)), flush=True)
    index = output / 'data-files.jsonl'
    index.write_text(''.join(rows), encoding='utf8', newline='\n')
    passport = dict(version='BASELINE_DATA_PASSPORT_V1', baseline='0.3.0', date='2026-10-27',
                    source_code_commit='5439815', data_root='data', inventory_scopes=list(scopes),
                    excluded_scopes=['raw', 'labels', 'research'],
                    inventory_file=index.name, inventory_sha256=sha(index),
                    files=len(rows), bytes=sum(d['bytes'] for d in datasets), datasets=datasets,
                    verification='All listed local payload bytes hashed; original raw archives outside scopes not rehashed; no MBO replay',
                    training_admission=False, feature_certificate='feature-equivalence.json')
    (output / 'data-passport.json').write_text(json.dumps(passport, indent=2) + '\n', encoding='utf8', newline='\n')
    return passport


def verify(root, passport_path):
    root, passport_path = Path(root).resolve(), Path(passport_path)
    passport = json.loads(passport_path.read_text(encoding='utf8'))
    index = passport_path.parent / passport['inventory_file']
    if sha(index) != passport['inventory_sha256']:
        raise ValueError('Inventory digest mismatch')
    seen = set(); total = 0; groups = {}
    for line in index.read_text(encoding='utf8').splitlines(keepends=True):
        row = json.loads(line); p = (root / row['path']).resolve()
        if not p.is_relative_to(root) or row['path'] in seen:
            raise ValueError('Unsafe or duplicate inventory path')
        seen.add(row['path'])
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Data drift: ' + row['path'])
        total += row['bytes']
        key = '/'.join(Path(row['path']).parts[:2])
        groups.setdefault(key, []).append(line)
    actual = {p.relative_to(root).as_posix() for s in passport['inventory_scopes'] for p in files_under(root / s)}
    actual_datasets = {p.relative_to(root).as_posix() for s in passport['inventory_scopes']
                       for p in (root / s).iterdir() if p.is_dir()}
    if actual_datasets != {d['path'] for d in passport['datasets']}:
        raise ValueError('Dataset coverage changed')
    if actual != seen or len(seen) != passport['files'] or total != passport['bytes']:
        raise ValueError('Inventory coverage changed')
    for dataset in passport['datasets']:
        if not (root / dataset['path']).is_dir():
            raise ValueError('Missing dataset: ' + dataset['path'])
        lines = groups.pop(dataset['path'], [])
        if len(lines) != dataset['files'] or hashlib.sha256(''.join(lines).encode()).hexdigest() != dataset['tree_sha256']:
            raise ValueError('Dataset identity mismatch: ' + dataset['path'])
    if groups:
        raise ValueError('Unclassified datasets')
    return dict(status='PASS', files=len(seen), bytes=total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'verify'))
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='baselines/0.3.0')
    args = parser.parse_args()
    result = inventory(args.data_root, args.output) if args.command == 'inventory' else verify(args.data_root, Path(args.output) / 'data-passport.json')
    print(json.dumps({k: v for k, v in result.items() if k != 'datasets'}))
