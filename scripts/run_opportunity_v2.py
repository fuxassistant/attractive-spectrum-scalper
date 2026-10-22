"""Frozen six-model development comparison, then one prespecified seed replication.

python -m scripts.run_opportunity_v2
No checkpoint, family, risk or threshold is selected from TEST outcomes.
Completed run directories are immutable and verified before reuse.
"""
from __future__ import annotations

import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from spectrum.research_v2.learning import load_dataset, train_run, sha


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def choose_validation(runs):
    """Selection cannot inspect test, calibration, or economic fields."""
    if not runs:
        raise ValueError('No development models')
    ranked = [(float(s['evaluation']['validation']['model']['joint_nll']), s['run_id']) for s in runs]
    if not all(np.isfinite(v) for v, _ in ranked):
        raise ValueError('Nonfinite validation selection score')
    return min(ranked)[1]


def validate_dataset_identity(manifest, config):
    from scripts.prepare_opportunity_v2 import identity
    if manifest.get('identity') != identity(config):
        raise ValueError('Dataset recipe/source differs from the effective run configuration')


def dataset_diagnostics(store, manifest, config):
    f = store.frame
    by_day, by_role = [], {}
    starts = store.segment_starts(config['data']['cadence_seconds'])
    full_window = np.arange(len(f)) - starts + 1 >= config['data']['sequence_length']
    family_cols = {k: np.array([store.feature_names.index(n) for n in names]) for k, names in store.groups.items()}
    for date, ids in f.groupby('date', sort=False).indices.items():
        part = f.iloc[ids]
        known = part.feature_available.to_numpy() & (part.y_crossing.to_numpy() >= 0)
        row = dict(date=date, role=next(r for r, ds in config['splits'].items() if date in ds),
            roots=len(ids), x_available=int(part.feature_available.sum()), target_known=int((part.y_crossing >= 0).sum()),
            admitted=int(known.sum()), states=np.bincount(part.y_crossing.to_numpy()[known], minlength=4).tolist(),
            full_window_fraction=float(full_window[ids][known].mean()),
            finite_fraction={k: float(np.isfinite(store.raw[np.asarray(ids)[known]][:, columns]).mean()) for k, columns in family_cols.items()})
        by_day.append(row)
    for role, dates in config['splits'].items():
        rows = [d for d in by_day if d['date'] in dates]
        by_role[role] = dict(days=len(rows), roots=sum(d['roots'] for d in rows), admitted=sum(d['admitted'] for d in rows),
                            states=np.sum([d['states'] for d in rows], axis=0).tolist())
    return dict(status='COMPLETE', manifest_sha256=store.source_receipt['manifest_sha256'],
        date_count=len(manifest['dates']), roots=len(f), by_role=by_role, by_day=by_day,
        families={k: dict(features=len(v), model_input_channels=2*len(v)+1) for k, v in store.groups.items()},
        feature_names=store.feature_names,
        feature_finite_fraction=np.isfinite(store.raw).mean(axis=0).tolist(),
        context='64 five-second roots, TCN RF61=300 seconds; longer current/previous London summaries and M1 spectra are compressed, not full paths',
        event='One fixed BB first-line crossing query and common known cohort for every family and risk plan',
        historical_exposure=config['experiment']['historical_exposure'])


def verify_run(out, config, sources, manifest_sha):
    receipt = read(out / 'receipt.json')
    if receipt.get('status') != 'COMPLETE' or read(out / 'config.json') != config:
        raise ValueError('Existing run incomplete or config changed: ' + str(out))
    if receipt['sources'] != sources or receipt['dataset_manifest_sha256'] != manifest_sha:
        raise ValueError('Existing run source or dataset identity changed: ' + str(out))
    for name, expected in receipt['outputs'].items():
        if sha(out / name) != expected:
            raise ValueError('Completed run payload changed: ' + str(out / name))
    return read(out / 'summary.json')


def run_one(store, execution, config, runs_root, sources):
    from spectrum.research_v2.evaluation import evaluate_risk_grid
    out = runs_root / config['run_id']
    if out.exists():
        return verify_run(out, config, sources, store.source_receipt['manifest_sha256'])
    out.mkdir(parents=True)
    save(out / 'config.json', config)
    receipt = dict(status='RUNNING', run_id=config['run_id'], config_sha256=sha(out / 'config.json'),
        dataset_manifest_sha256=store.source_receipt['manifest_sha256'], sources=sources,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_note='Working source SHA is authoritative, including uncommitted implementation',
        libraries={n: version(n) for n in ('numpy', 'pandas', 'torch', 'numba', 'scikit-learn')}, cuda=torch.version.cuda)
    save(out / 'receipt.json', receipt)
    print(json.dumps(dict(stage='START', run=config['run_id'])), flush=True)
    result = train_run(store, config, out)
    p = result['predictions'][['p_none', 'p_long', 'p_short', 'p_both']].to_numpy()
    economics = evaluate_risk_grid(store.frame, execution, p, config)
    save(out / 'economic_evaluation.json', economics)
    result['summary'].update(status='COMPLETE', economic_evaluation='economic_evaluation.json',
                             selected_policy=economics['summary'])
    save(out / 'summary.json', result['summary'])
    if any(sha(path) != expected for path, expected in sources.items()):
        raise ValueError('Bound source changed during run: retained incomplete')
    receipt.update(status='COMPLETE', outputs={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'receipt.json'})
    save(out / 'receipt.json', receipt)
    print(json.dumps(dict(stage='COMPLETE', run=config['run_id'], best_update=result['summary']['training']['best_update'],
        validation_nll=result['summary']['evaluation']['validation']['model']['joint_nll'])), flush=True)
    torch.cuda.empty_cache()
    return result['summary']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/research/opportunity_pipeline_v2.json')
    parser.add_argument('--runs-root', default='data/derived/scientific_pipeline_v2/runs')
    parser.add_argument('--report-root', default='reports/scientific_pipeline_v2')
    args = parser.parse_args()
    config = read(args.config)
    report, runs_root = Path(args.report_root), Path(args.runs_root)
    report.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), *[Path('spectrum/research_v2') / n for n in ('learning.py', 'evaluation.py')],
             *[Path('spectrum/research') / n for n in ('models.py', 'losses.py', 'training.py')]]
    sources = {str(p.resolve()): sha(p) for p in paths}
    store = load_dataset(config['data']['root'])
    manifest = read(Path(config['data']['root']) / 'manifest.json')
    validate_dataset_identity(manifest, config)
    save(report / 'dataset_diagnostics.json', dataset_diagnostics(store, manifest, config))
    execution = pd.concat([pd.read_parquet(Path(config['data']['root']) / (d + '_execution.parquet')) for d in manifest['dates']], ignore_index=True)
    suite = dict(status='RUNNING', dataset_root=config['data']['root'], runs_root=str(runs_root),
        config_sha256=sha(args.config), dataset_manifest_sha256=store.source_receipt['manifest_sha256'],
        runs=[], primary_run_id=None, confirmation_run_id=None, selection=None,
        primary_seed=config['experiment']['development_seed'], confirmation_role='Prespecified seed replication; not a second test-selected winner')
    save(report / 'suite_summary.json', suite)
    summaries, configs = [], {}
    started = time.perf_counter()
    for family in config['experiment']['families']:
        for architecture in config['experiment']['architectures']:
            c = copy.deepcopy(config)
            c['data']['family'], c['model']['architecture'] = family, architecture
            c['seed'] = config['experiment']['development_seed']
            c['run_id'] = f'{architecture}_{family}_seed{c["seed"]}'
            result = run_one(store, execution, c, runs_root, sources)
            configs[c['run_id']] = c
            summaries.append(result)
            suite['runs'].append(dict(run_id=c['run_id'], role='development'))
            save(report / 'suite_summary.json', suite)
    if len({s['scaler_sha256'] for s in summaries}) != 1 or any(s['cohorts'] != summaries[0]['cohorts'] for s in summaries):
        raise ValueError('Unmatched feature-family cohorts or scalers')
    selected = choose_validation(summaries)
    selection = dict(criterion='validation_joint_nll', selected_run_id=selected,
        validation_scores={s['run_id']: s['evaluation']['validation']['model']['joint_nll'] for s in summaries},
        test_and_economics_used_for_selection=False, tie_break='lexicographic run_id',
        source_sha256=sources, dataset_manifest_sha256=store.source_receipt['manifest_sha256'])
    # A durable selection receipt precedes the confirmation fit.
    save(report / 'model_selection.json', selection)
    suite.update(primary_run_id=selected, selection=selection)
    save(report / 'suite_summary.json', suite)
    confirmation = copy.deepcopy(configs[selected])
    confirmation['seed'] = config['experiment']['confirmation_seed']
    confirmation['run_id'] = f'{confirmation["model"]["architecture"]}_{confirmation["data"]["family"]}_seed{confirmation["seed"]}'
    run_one(store, execution, confirmation, runs_root, sources)
    suite['runs'].append(dict(run_id=confirmation['run_id'], role='confirmation'))
    suite.update(status='COMPLETE', confirmation_run_id=confirmation['run_id'], wall_seconds=time.perf_counter()-started)
    save(report / 'suite_summary.json', suite)
    print(json.dumps(dict(stage='SUITE_COMPLETE', primary=selected, confirmation=confirmation['run_id'])), flush=True)


if __name__ == '__main__':
    main()
