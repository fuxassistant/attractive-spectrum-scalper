"""Run a frozen configurable scientific opportunity experiment.

python -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json
python -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json --run-id mlp_execution_bce_seed17 --set model.architecture=mlp --skip-controls

Prepared execution outcomes are reused only if their effective recipe matches.
Changing execution parameters requires prepare_research_execution and a new cache.
"""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

# Required before CUDA creates a deterministic GEMM context.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from spectrum.research.controls import make_control_data
from spectrum.research.data import load_market, build_sequence_indices, prepare_sequences
from spectrum.research.training import (validate_config, load_execution, split_indices,
    fit_model, predict, metrics, select_threshold, account_projection, sha, save_json)


def override(config, expression):
    key, value = expression.split('=', 1)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    parts, destination = key.split('.'), config
    for part in parts[:-1]:
        if part not in destination or not isinstance(destination[part], dict):
            raise ValueError('Unknown config path: ' + key)
        destination = destination[part]
    # Loss options are optional, but spelling mistakes elsewhere must fail.
    optional = {'loss': {'epsilon', 'brier_lambda'}}
    if parts[-1] not in destination and parts[-1] not in optional.get(parts[0], set()):
        raise ValueError('Unknown config key: ' + key)
    destination[parts[-1]] = value


def controls(config, x_train, y_train, directory):
    c = config['controls']
    train = make_control_data(config['seed'] + 100, c['synthetic_count'])
    test = make_control_data(config['seed'] + 200, c['synthetic_test_count'])
    validation = make_control_data(config['seed'] + 300, c['synthetic_test_count'])
    cfg = copy.deepcopy(config)
    cfg['run_id'] = 'positive_control'
    cfg['training'].update(max_epochs=c['max_epochs'], patience=12)
    model, history, receipt = fit_model(train['x'], train['y'], validation['x'], validation['y'], cfg, directory / 'positive')
    positive = dict(training=receipt, train_receipt=train['receipts'], test_receipt=test['receipts'],
                    validation_receipt=validation['receipts'], test=metrics(test['y'], predict(model, test['x'], cfg)))
    cfg['run_id'] = 'shuffled_control'
    shuffled = np.random.default_rng(c['shuffle_seed']).permutation(train['y'])
    model, _, receipt = fit_model(train['x'], shuffled, validation['x'], validation['y'], cfg, directory / 'shuffled')
    negative = dict(training=receipt, test=metrics(test['y'], predict(model, test['x'], cfg)),
                    semantics='Only synthetic TRAIN labels permuted; held-out causal process stays intact. This checks fabricated learnability, not market null significance.')
    rng = np.random.default_rng(config['seed'])
    ids = rng.choice(len(y_train), min(c['memorize_count'], len(y_train)), replace=False)
    cfg = copy.deepcopy(config)
    cfg['run_id'] = 'memorize_real_train'
    cfg['model']['dropout'] = 0.
    cfg['training'].update(max_epochs=c['memorize_epochs'], patience=c['memorize_epochs'], weight_decay=0., min_delta=0.)
    model, _, receipt = fit_model(x_train[ids], y_train[ids], x_train[ids], y_train[ids], cfg, directory / 'memorize')
    memorize = dict(training=receipt, train_indices=ids.tolist(), fit=metrics(y_train[ids], predict(model, x_train[ids], cfg)),
                    semantics='Intentional same-batch train/evaluation with dropout and weight decay disabled; capacity/backprop check, not generalization.')
    result = dict(positive=positive, shuffled=negative, memorize=memorize,
                  checks=dict(positive_accuracy_at_least_80pct=positive['test']['accuracy'] >= .8,
                              positive_exceeds_shuffled_by_30pct=positive['test']['accuracy'] - negative['test']['accuracy'] >= .3,
                              real_batch_accuracy_at_least_95pct=memorize['fit']['accuracy'] >= .95))
    save_json(directory / 'controls.json', result)
    del model
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--skip-controls', action='store_true')
    parser.add_argument('--output-root', default='data/derived/scientific_pipeline_v1/runs')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for expression in args.overrides:
        override(config, expression)
    if args.run_id:
        config['run_id'] = args.run_id
    validate_config(config)
    if not config['run_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in config['run_id']):
        raise ValueError('run_id must be a safe directory name')
    out = Path(args.output_root) / config['run_id']
    if out.exists():
        raise ValueError('Run directories are immutable: choose a new run_id')
    out.mkdir(parents=True)
    save_json(out / 'config.json', config)
    source_paths = sorted(Path('spectrum/research').glob('*.py')) + [Path(__file__)]
    receipt = dict(status='RUNNING', config_sha256=sha(out / 'config.json'),
                   sources={str(p): sha(p) for p in source_paths},
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   source_note='Working source SHA is authoritative, including uncommitted implementation',
                   libraries={n: version(n) for n in ('numpy', 'pandas', 'torch', 'numba', 'scikit-learn')},
                   cuda=torch.version.cuda, requested_config_path=args.config)
    save_json(out / 'receipt.json', receipt)
    loaded = load_market(config)
    frame = loaded['frame']
    execution, y_execution, execution_receipt = load_execution(config, frame)
    y = y_execution if config['target']['kind'] == 'execution_profit' else loaded['y_crossing']
    split = split_indices(frame, y, config)
    seqidx = build_sequence_indices(frame, config['data']['sequence_length'], config['data']['cadence_seconds'])
    x, scaler = prepare_sequences(loaded['x_raw'], seqidx, split['train'])
    save_json(out / 'scaler.json', scaler)
    save_json(out / 'data_receipt.json', loaded['receipts'])
    save_json(out / 'execution_receipt.json', execution_receipt)
    cohorts = {}
    for role, days in config['splits'].items():
        rows = frame.date.isin(days).to_numpy()
        known = rows & frame.feature_available.to_numpy(bool) & (y >= 0)
        cohorts[role] = dict(dates=days, roots=int(rows.sum()), feature_available=int((rows & frame.feature_available.to_numpy(bool)).sum()),
                             target_known=int((rows & (y >= 0)).sum()), admitted=int(known.sum()),
                             state_counts=np.bincount(y[known], minlength=4).tolist(),
                             execution_status_L=execution.loc[rows, 'status_L'].value_counts().to_dict(),
                             execution_status_S=execution.loc[rows, 'status_S'].value_counts().to_dict())
    save_json(out / 'cohorts.json', cohorts)
    print(json.dumps(dict(stage='data', run=config['run_id'], cohorts=cohorts, shape=list(x.shape))), flush=True)
    if not args.skip_controls:
        control_result = controls(config, x[split['train']], y[split['train']], out)
        if not all(control_result['checks'].values()):
            raise ValueError('Learnability control failed; market training withheld. Inspect controls.json')
    model, history, train_receipt = fit_model(x[split['train']], y[split['train']], x[split['validation']], y[split['validation']], config, out / 'market')
    p = predict(model, x, config)
    # A half-count prior avoids log(0) on classes absent in a small train split.
    prior = (np.bincount(y[split['train']], minlength=4) + .5) / (len(split['train']) + 2.)
    prior_p = np.tile(prior, (len(frame), 1))
    evaluation = {role: dict(model=metrics(y[ids], p[ids]), prior=metrics(y[ids], prior_p[ids])) for role, ids in split.items()}
    threshold, calibration_grid = select_threshold(frame, execution, p, config)
    projections, ledgers = {}, {}
    for role in ('calibration', 'test'):
        projections[role], ledgers[role] = account_projection(frame, execution, p, config['splits'][role], threshold, config)
        # Frozen forced threshold is a diagnostic of opportunity ranking, not a
        # second chosen strategy. Report even when calibrated policy chooses WAIT.
        projections[role + '_forced_05'], ledgers[role + '_forced_05'] = account_projection(frame, execution, p, config['splits'][role], .5, config)
    prior_threshold, prior_grid = select_threshold(frame, execution, prior_p, config)
    prior_test, prior_ledger = account_projection(frame, execution, prior_p, config['splits']['test'], prior_threshold, config)
    predictions = frame[['date', 't_ns', 'feature_available']].copy()
    predictions['target'] = y
    predictions['execution_target'] = y_execution
    predictions['crossing_target'] = loaded['y_crossing']
    predictions['split'] = [next(k for k, days in config['splits'].items() if d in days) for d in frame.date]
    for k, name in enumerate(('none', 'long', 'short', 'both')):
        predictions['p_' + name] = p[:, k]
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    save_json(out / 'ledgers.json', ledgers)
    same = (y_execution >= 0) & (loaded['y_crossing'] >= 0) & frame.feature_available.to_numpy(bool)
    confusion = np.zeros((4, 4), int)
    np.add.at(confusion, (loaded['y_crossing'][same], y_execution[same]), 1)
    diagnostics = dict(crossing_rows_execution_columns=confusion.tolist(), common_count=int(same.sum()),
                        direct_X_channels=len(loaded['feature_names']), normalized_input_channels=x.shape[-1],
                        complete_window_fraction=float((seqidx[:, 0] >= 0).mean()),
                        risk_quantiles=np.nanquantile(frame.risk_R, [.1, .5, .9]).tolist(),
                        target_change_rate={})
    for name, labels in [('execution', y_execution), ('crossing', loaded['y_crossing'])]:
        adjacent = (frame.date.to_numpy()[1:] == frame.date.to_numpy()[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)
        diagnostics['target_change_rate'][name] = float((labels[1:][adjacent] != labels[:-1][adjacent]).mean())
    summary = dict(run_id=config['run_id'], status='COMPLETE', cohorts=cohorts, training=train_receipt,
                   evaluation=evaluation, selected_threshold=threshold, calibration_grid=calibration_grid,
                   projections=projections, prior_policy=dict(threshold=prior_threshold, calibration_grid=prior_grid, test=prior_test),
                   diagnostics=diagnostics, limitations=config['interpretation'],
                   controls='controls.json' if not args.skip_controls else 'SKIPPED: comparison run; use baseline controls')
    save_json(out / 'summary.json', summary)
    # Source changes during a run invalidate scientific completion.
    if any(sha(path) != expected for path, expected in receipt['sources'].items()):
        raise ValueError('Source changed during training; run retained as RUNNING, not admitted')
    receipt.update(status='COMPLETE', outputs={str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file() and path.name != 'receipt.json'})
    save_json(out / 'receipt.json', receipt)
    print(json.dumps(dict(stage='COMPLETE', run=config['run_id'], best_epoch=train_receipt['best_epoch'],
                          test=evaluation['test'], selected_threshold=threshold, account=projections['test'])), flush=True)


if __name__ == '__main__':
    main()
