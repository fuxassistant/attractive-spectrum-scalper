"""Matched 2x2 context/weighting experiment, all arms replicated at seed29.
No TEST arm/seed winner selection. Checkpoints use unweighted VAL joint NLL.
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
import pandas as pd
import torch
from spectrum.research_v3.learning import load_dataset, train_run, sha
from scripts.run_opportunity_v2 import save, read, verify_run


def experiment_configs(config):
    for seed in config['experiment']['seeds']:
        for context in config['experiment']['contexts']:
            for weighting in config['experiment']['weightings']:
                c = copy.deepcopy(config)
                c['seed'] = seed
                c['data']['context_mode'] = context
                c['training']['weighting'] = weighting
                c['run_id'] = f'tcn_micro_{context}_{weighting}_seed{seed}'
                yield c


def validate_recipe(config):
    old = read('configs/research/opportunity_pipeline_v2.json')
    for key in ('event_query', 'execution', 'risk', 'loss', 'model', 'policy'):
        if config[key] != old[key]:
            raise ValueError('Frozen v2 recipe changed: ' + key)
    for key in old['training']:
        if key != 'weighting' and config['training'][key] != old['training'][key]:
            raise ValueError('Unpaired training parameter: ' + key)
    if config['data']['family'] != 'micro' or config['data']['sequence_length'] != 64:
        raise ValueError('Fixed Micro/64 required')
    if config['experiment']['seeds'] != [17, 29] or config['experiment']['contexts'] != ['legacy','restored'] or config['experiment']['weightings'] != ['uniform','crossing_id']:
        raise ValueError('Exactly the preregistered eight arms required')


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
    parser.add_argument('--config', default='configs/research/opportunity_pipeline_v3.json')
    parser.add_argument('--runs-root', default='data/derived/scientific_pipeline_v3/runs')
    parser.add_argument('--report-root', default='reports/scientific_pipeline_v3')
    args = parser.parse_args()
    config = read(args.config)
    validate_recipe(config)
    report, runs_root = Path(args.report_root), Path(args.runs_root)
    report.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('scripts/run_opportunity_v2.py'), Path(args.config),
             *[Path('spectrum/research_v3') / n for n in ('learning.py', 'features.py')],
             *[Path('spectrum/research_v2') / n for n in ('learning.py', 'evaluation.py')],
             *[Path('spectrum/research') / n for n in ('models.py', 'losses.py', 'training.py')]]
    sources = {str(p.resolve()): sha(p) for p in paths}
    store = load_dataset(config['data']['root'])
    manifest = read(Path(config['data']['root']) / 'manifest.json')
    from scripts.prepare_opportunity_v3 import identity
    if manifest.get('identity') != identity(config):
        raise ValueError('Dataset recipe/source identity differs from configuration')
    execution = pd.concat([pd.read_parquet(Path(config['data']['root']) / (d + '_execution.parquet')) for d in manifest['dates']], ignore_index=True)
    suite = dict(status='RUNNING', dataset_root=config['data']['root'], runs_root=str(runs_root),
        config_sha256=sha(args.config), dataset_manifest_sha256=store.source_receipt['manifest_sha256'],
        runs=[], selection='No arm/seed winner selection; all eight factorial arms reported',
        primary_seed=17, replication_seed=29, fixed_visual_cases=config['experiment']['fixed_visual_cases'])
    save(report / 'suite_summary.json', suite)
    summaries=[]
    started=time.perf_counter()
    for c in experiment_configs(config):
        result=run_one(store, execution, c, runs_root, sources)
        summaries.append(result)
        suite['runs'].append(dict(run_id=c['run_id'], context=c['data']['context_mode'],
                                 weighting=c['training']['weighting'], seed=c['seed'],
                                 role='primary' if c['seed']==17 else 'replication'))
        save(report / 'suite_summary.json', suite)
    if len({s['scaler_sha256'] for s in summaries}) != 1 or any(s['cohorts'] != summaries[0]['cohorts'] for s in summaries):
        raise ValueError('Unmatched cohorts or scalers')
    for seed in (17,29):
        group=[s for s in summaries if s['run_id'].endswith('seed'+str(seed))]
        for key in ('train_metric_probe_sha256','gradient_probe_sha256','initial_state_sha256'):
            if len({s['training'][key] for s in group}) != 1:
                raise ValueError('Unmatched seed-level RNG/probe: '+key)
        histories = [read(runs_root / s['run_id'] / 'history.json') for s in group]
        checkpoints = set.intersection(*[{r['update'] for r in h} for h in histories])
        for update in checkpoints:
            hashes = {next(r['sampled_root_stream_sha256'] for r in h if r['update'] == update) for h in histories}
            if len(hashes) != 1:
                raise ValueError('Unmatched sampled TRAIN-root prefix at update '+str(update))
    suite.update(status='COMPLETE', wall_seconds=time.perf_counter()-started,
                 matched_scaler_and_cohort=True, matched_initialization_and_probes=True)
    save(report / 'suite_summary.json', suite)
    print(json.dumps(dict(stage='SUITE_COMPLETE', runs=len(summaries))), flush=True)


if __name__ == '__main__':
    main()
