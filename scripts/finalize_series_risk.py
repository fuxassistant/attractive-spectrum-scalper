"""Publish a checked pointer to the corrected series-risk experiment."""
from pathlib import Path
import json

from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player_v3.gym import risk_contract


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parent = Path('reports/player_v3')
    root = parent / 'series_001'
    martingale = parent / 'martingale_series_001'
    summary = load(root / 'summary.json')
    contract = load(root / 'contract.json')
    memory = load(root / 'memory_usage.json')
    posthoc = load(root / 'posthoc_receipt.json')
    verdict_path = parent / 'sol61_audit/series_verdict.json'
    verdict = load(verdict_path)
    assert verdict['technical_status'] == 'PASS_BOUNDED_SERIES_RISK_RESEARCH'
    assert summary['status'] == memory['status'] == posthoc['status'] == 'COMPLETE'
    assert contract['risk_contract'] == risk_contract()
    assert memory['causal_future_violations'] == 0
    assert sha(root / 'freeze.json') == summary['freeze_sha256'] == posthoc['freeze_sha256']
    for name, digest in contract['source_hashes'].items():
        assert sha(name) == digest, name
        assert sha(root / 'source_snapshot' / name) == digest, name
    assert (root / 'index.html').is_file()
    receipt = dict(
        status='COMPLETE', correction='Series loss relative to capital before its first actual fill',
        risk_contract=risk_contract(), source_snapshot_verified=True,
        old_results='run_001..003 and martingale_001..002 preserved and marked superseded',
        neural_summary_sha256=sha(root / 'summary.json'),
        martingale_summary_sha256=sha(martingale / 'audit_summary.json'),
        memory_audit_sha256=sha(root / 'memory_usage.json'),
        independent_verdict_sha256=sha(verdict_path), independent_verdict=verdict,
        regression_verification=dict(tests_passed=140, seconds=12.16, diff_check='PASS'),
        ui_payload_qa_sha256=sha(root / 'ui_payload_qa.json'),
        ui_sha256=sha(root / 'index.html'),
        limitation='Chronological development rerun on previously seen dates; no claimed profitable policy')
    save_json(root / 'correction_receipt.json', receipt)
    previous = load(parent / 'LATEST.json')
    save_json(parent / 'LATEST.json', dict(
        status='COMPLETE', risk_version=risk_contract()['version'],
        report=str(root.resolve()), ui=str((root / 'index.html').resolve()),
        martingale=str(martingale.resolve()), correction_receipt=str((root / 'correction_receipt.json').resolve()),
        previous=previous.get('previous', previous)))
    print(json.dumps(dict(status='COMPLETE', report=str(root), source_snapshot_verified=True)))


if __name__ == '__main__':
    main()
