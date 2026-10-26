"""Baseline receipts are checked without requiring the private market data."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from baseline_data import inventory, verify

BASE = ROOT / 'baselines' / '0.3.0'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


@pytest.fixture
def snapshot(tmp_path):
    data = tmp_path / 'data'
    for scope in ('derived', 'legacy'):
        (data / scope / 'sample').mkdir(parents=True)
    (data / 'derived/sample/payload').write_bytes(b'original')
    output = tmp_path / 'receipt'
    inventory(data, output)
    return data, output / 'data-passport.json'


def test_inventory_round_trip_including_empty_dataset(snapshot):
    assert verify(*snapshot) == {'status': 'PASS', 'files': 1, 'bytes': 8}


@pytest.mark.parametrize('change', ['modify', 'add', 'remove', 'remove_empty_dataset', 'add_empty_dataset', 'tamper_index'])
def test_inventory_detects_drift(snapshot, change):
    data, passport = snapshot
    payload = data / 'derived/sample/payload'
    if change == 'modify':
        payload.write_bytes(b'modified')  # same size, different content
    elif change == 'add':
        (payload.parent / 'extra').write_bytes(b'new')
    elif change == 'remove':
        payload.unlink()
    elif change == 'remove_empty_dataset':
        (data / 'legacy/sample').rmdir()
    elif change == 'add_empty_dataset':
        (data / 'derived/new_empty').mkdir()
    else:
        with (passport.parent / 'data-files.jsonl').open('ab') as stream:
            stream.write(b'\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(data, passport)


@pytest.mark.parametrize('platform', ['windows', 'linux'])
def test_platform_lock_covers_project_and_wheel_hashes(platform):
    lock = (ROOT / f'requirements/baseline-{platform}-py312.lock').read_text()
    pins = dict(re.findall(r'^([\w.-]+)==([^\s]+)', lock, re.M))
    wheels = read(BASE / f'wheels-{platform}.json')
    assert len(pins) == len(wheels) == 20
    assert pins == {w['name']: w['version'] for w in wheels}
    assert set(re.findall(r'--hash=sha256:([a-f0-9]{64})', lock)) == {w['sha256'] for w in wheels}
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    for requirement in (project['project']['dependencies'] + project['build-system']['requires']
                        + project['project']['optional-dependencies']['dev']):
        name, version = requirement.split('==')
        assert pins[name] == version
    assert project['project']['requires-python'] == '>=3.12,<3.13'
    assert (ROOT / '.python-version').read_text().strip() == '3.12.14'


def test_committed_passport_is_internally_complete():
    passport = read(BASE / 'data-passport.json')
    index = (BASE / passport['inventory_file']).read_bytes()
    assert hashlib.sha256(index).hexdigest() == passport['inventory_sha256']
    groups = {}
    paths = set()
    rows = []
    for line in index.splitlines(keepends=True):
        row = json.loads(line)
        assert row['path'] not in paths
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
        paths.add(row['path'])
        assert re.fullmatch('[a-f0-9]{64}', row['sha256'])
        rows.append(row)
        groups.setdefault('/'.join(row['path'].split('/')[:2]), []).append(line)
    assert len(rows) == passport['files']
    assert sum(r['bytes'] for r in rows) == passport['bytes']
    for dataset in passport['datasets']:
        lines = groups.pop(dataset['path'], [])
        assert len(lines) == dataset['files']
        assert sum(json.loads(line)['bytes'] for line in lines) == dataset['bytes']
        assert hashlib.sha256(b''.join(lines)).hexdigest() == dataset['tree_sha256']
    assert not groups
    assert passport['training_admission'] is False


def test_certificate_binds_samples_to_inventory_and_current_code():
    from spectrum.features.pipeline import implementation_identity
    from spectrum.market.calendar import MarketCalendar
    cert = read(BASE / 'feature-equivalence.json')
    index = {r['path']: r['sha256'] for r in
             map(json.loads, (BASE / 'data-files.jsonl').read_text().splitlines())}
    assert cert['current_implementation'] == implementation_identity()
    assert cert['current_calendar'] == MarketCalendar().contract()
    assert len(cert['samples']) == 8
    assert not cert['full_history_equivalence']
    assert not cert['historical_manifests_modified']
    for sample in cert['samples']:
        assert sample['status'] == 'PASS_BITWISE_COLUMN_VALUES'
        assert sample['columns'] == len(sample['column_sha256']) == 369
        feature = sample['dataset'] + '/' + sample['capture']
        for key, path in {'features': feature + '/features.parquet',
                          'feature_manifest': feature + '/manifest.json',
                          'market_manifest': sample['market'] + '/manifest.json',
                          'seconds': sample['market'] + '/seconds.parquet'}.items():
            assert sample['hashes'][key] == index[path]


def test_runtime_matches_migration_receipt():
    expected = read(ROOT / 'docs/MIGRATION.json')['current_runtime_sha256']
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'spectrum').rglob('*') if p.suffix in ('.py', '.json')}
    assert actual == expected


def test_bank_contract_receipt_matches_current_geometry():
    from spectrum.crossings.bank import geometry_definition
    from spectrum.crossings.store import VERSION
    bank = read(BASE / 'bank-compatibility.json')
    assert bank['storage_version'] == VERSION
    assert bank['geometry'] == json.loads(json.dumps(geometry_definition()))
    assert bank['days'] == 238
    assert not bank['replay_performed'] and not bank['trading_or_training_admission']
    index = {r['path']: r['sha256'] for r in
             map(json.loads, (BASE / 'data-files.jsonl').read_text().splitlines())}
    assert bank['manifest_sha256'] == index[bank['dataset'] + '/manifest.json']


def test_validation_binds_frozen_receipts():
    validation = read(BASE / 'validation.json')
    for path, digest in validation['evidence_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert validation['remote_ci'] == 'NOT_RUN'
    assert validation['runtime_payload_files'] == 106
    assert validation['installed_outside_source'] is True
