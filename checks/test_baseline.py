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


def evolution_chain(version):
    """Follow immutable parent receipts back to the historical 0.3 baseline."""
    chain = []
    seen = set()
    while version != '0.3.0':
        assert version not in seen
        seen.add(version)
        path = ROOT / 'docs' / ('EVOLUTION_'+version.replace('.', '_')+'.json')
        receipt = read(path)
        assert receipt['version'] == version
        if chain:
            child = chain[-1]
            assert child['parent_evolution_path'] == path.relative_to(ROOT).as_posix()
            assert child['parent_evolution_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert child['parent_feature_implementation'] == receipt['current_feature_implementation']
        chain.append(receipt)
        version = receipt['parent_version']
    assert chain[-1]['parent_migration_sha256'] == hashlib.sha256((ROOT/'docs/MIGRATION.json').read_bytes()).hexdigest()
    return chain


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


def test_certificate_binds_samples_to_inventory_and_declared_version():
    from spectrum.features.pipeline import implementation_identity
    from spectrum.market.calendar import MarketCalendar
    cert = read(BASE / 'feature-equivalence.json')
    index = {r['path']: r['sha256'] for r in
             map(json.loads, (BASE / 'data-files.jsonl').read_text().splitlines())}
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    if version==cert['baseline']:
        assert cert['current_implementation'] == implementation_identity()
    else:
        chain=evolution_chain(version)
        assert chain[-1]['parent_version']==cert['baseline']
        assert chain[-1]['parent_feature_implementation']==cert['current_implementation']
        assert chain[0]['current_feature_implementation']==implementation_identity()
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
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    evolution=evolution_chain(version)[0]
    expected = evolution['current_runtime_sha256']
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'spectrum').rglob('*') if p.suffix in ('.py', '.json')}
    assert actual == expected


def test_patch_041_binds_parent_and_only_declared_runtime_changes():
    import spectrum
    current=read(ROOT/'docs/EVOLUTION_0_4_1.json')
    parent=read(ROOT/current['parent_evolution_path'])
    assert current['version'] == spectrum.__version__ == '0.4.1'
    assert parent['version'] == current['parent_version'] == '0.4.0'
    project=(ROOT/'pyproject.toml').read_bytes()
    assert hashlib.sha256(project).hexdigest() == current['current_project_sha256']
    historical=current['parent_project_snapshot']
    historical_bytes=historical['utf8_text'].encode('utf-8')
    assert hashlib.sha256(historical_bytes).hexdigest() == historical['sha256']
    historical_project=tomllib.loads(historical['utf8_text'])
    assert historical_project['project']['version'] == '0.4.0'
    historical_project['project']['version']='0.4.1'
    assert historical_project == tomllib.loads(project.decode('utf-8'))
    assert current['current_feature_implementation'] == parent['current_feature_implementation']
    previous=parent['current_runtime_sha256']
    actual=current['current_runtime_sha256']
    assert actual.keys() == previous.keys()
    changed={name for name in actual if actual[name] != previous[name]}
    assert changed == {'spectrum/execution/barriers.py', 'spectrum/__init__.py'}
    assert changed == set(current['changed_runtime_files'])
    for name in changed:
        assert current['changed_runtime_files'][name] == {'parent_sha256':previous[name], 'current_sha256':actual[name]}
    for name, expected in current['preserved_evidence_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    assert not current['training_admission'] and not current['economic_admission']


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
    historical=read(ROOT/'docs/EVOLUTION_0_4_0.json')['historical_evidence']
    for path, digest in validation['evidence_sha256'].items():
        evidence=ROOT/historical.get(path,path)
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest
    assert validation['remote_ci'] == 'NOT_RUN'
    assert validation['runtime_payload_files'] == 106
    assert validation['installed_outside_source'] is True
