"""Frozen restored/uniform TCN plus a separately fitted causal risk filter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch

from spectrum.research.models import build_model
from spectrum.research.losses import joint_probabilities
from spectrum.research_v3.learning import load_dataset, _batch, sha, logit_gradients
from scripts.prepare_opportunity_v3 import identity

PNAMES = ['p_none', 'p_long', 'p_short', 'p_both']
QNAMES = ['q_event', 'q_correct', 'q_safe_cost2', 'q_safe_cost4', 'q_safe_cost6']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def validate_recipe(c):
    old = read('configs/research/opportunity_pipeline_v4.json')
    if c['schema'] != 'scientific_opportunity_pipeline_v5':
        raise ValueError('V5 recipe required')
    for key in ('event_query', 'execution', 'risk', 'loss', 'model'):
        if c[key] != old[key]:
            raise ValueError('Frozen base recipe changed: '+key)
    if c['data']['context_mode'] != 'restored' or c['data']['family'] != 'micro':
        raise ValueError('Restored Micro base required')
    if c['data']['sequence_length'] != 64 or c['data']['cadence_seconds'] != 5:
        raise ValueError('Frozen base clock required')
    dates = sum([c['splits'][k] for k in ('train', 'validation', 'calibration', 'test')], [])
    if dates != sorted(set(dates)) or dates != c['data']['dates'] or dates[0] <= '2026-07-24':
        raise ValueError('New disjoint chronological blocks required')
    if c['experiment']['seeds'] != [17, 29] or c['filter']['base_updates'] != 0:
        raise ValueError('Two frozen base checkpoints required')
    if sha(c['data']['source_inventory']) != c['data']['source_inventory_sha256']:
        raise ValueError('Coverage selection changed')
    for seed in c['experiment']['seeds']:
        b = c['base_models'][str(seed)]
        for name, digest in b['files'].items():
            if sha(Path(b['root'])/name) != digest:
                raise ValueError('Frozen base changed: '+name)
        cfg = read(Path(b['root'])/'config.json')
        if cfg['training']['weighting'] != 'uniform' or cfg['data']['context_mode'] != 'restored':
            raise ValueError('Wrong frozen base')
        if max(cfg['splits']['validation']) >= min(c['splits']['train']):
            raise ValueError('Meta training overlaps base checkpoint selection')


@torch.no_grad()
def frozen_features(store, c, seed):
    """Only raw causal X and explicitly whitelisted decision risk enter X_meta."""
    b = c['base_models'][str(seed)]
    checkpoint = torch.load(Path(b['root'])/'checkpoint.pt', map_location='cpu', weights_only=True)
    base_config, scaler = read(Path(b['root'])/'config.json'), read(Path(b['root'])/'scaler.json')
    if checkpoint['scaler_sha256'] != sha(Path(b['root'])/'scaler.json'):
        raise ValueError('Frozen checkpoint/scaler mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint['input_dim'], checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.requires_grad_(False)
    p, z, h, current = [], [], [], []
    n = len(store.frame)
    for start in range(0, n, 512):
        ids = np.arange(start, min(start+512, n))
        batch = _batch(store, ids, base_config, scaler)
        xb = torch.as_tensor(batch, device=device)
        hidden = model.encoder(model.projection(xb).transpose(1, 2)).transpose(1, 2)[:, -1]
        logits = model.output(hidden)
        probability = joint_probabilities(logits, 'factorized')
        # Check the extraction against the unchanged public forward interface.
        if start == 0 and not torch.equal(logits, model(xb)):
            raise AssertionError('Frozen hidden extraction changed base prediction')
        p.append(probability.cpu().numpy()); z.append(logits.cpu().numpy())
        h.append(hidden.cpu().numpy()); current.append(batch[:, -1].copy())
    p, z, h, current = map(np.concatenate, (p, z, h, current))
    side = np.where(p[:, 2]+p[:, 3] > p[:, 1]+p[:, 3], -1, 1).astype(np.int8)
    f = store.frame
    gain = f.G_event.to_numpy(float)
    causal = np.column_stack([f['R_'+plan].to_numpy(float) for plan in ('cost2', 'cost4', 'cost6')]
                            + [f.decision_spread.to_numpy(float)+c['event_query']['reference_fee']])
    ratios = np.divide(causal, gain[:, None], out=np.zeros_like(causal), where=np.isfinite(gain[:, None]) & (gain[:, None] > 0))
    ratios = np.where(np.isfinite(ratios), ratios, 0).astype(np.float32)
    x = np.column_stack((current, h, p, side, ratios)).astype(np.float32)
    if x.shape[1] != 182 or not np.isfinite(x).all():
        raise ValueError('Expected finite 182-channel filter X')
    columns = list(store.groups['micro'])
    names = ['current_value__'+v for v in columns] + ['current_known__'+v for v in columns]
    names += ['current_supported_clock'] + ['frozen_hidden_'+str(i) for i in range(32)]
    names += PNAMES + ['chosen_side'] + ['R_cost2_over_G', 'R_cost4_over_G', 'R_cost6_over_G', 'cost_over_G']
    return x, p, z, side, names


def bound_sources(cpath, manifest):
    paths = [Path(__file__), Path(cpath), Path('spectrum/research_v5/learning.py'),
             Path('spectrum/research_v5/targets.py'), Path('spectrum/research_v5/__init__.py'),
             Path('spectrum/research_v3/learning.py'), Path('spectrum/research_v2/learning.py'),
             Path('spectrum/research/models.py'), Path('spectrum/research/losses.py')]
    paths += [Path(p) for p in manifest['identity']['sources']]
    return {str(p.resolve()): sha(p) for p in sorted(set(paths), key=str)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/research/opportunity_pipeline_v5.json')
    args = parser.parse_args()
    c = read(args.config)
    validate_recipe(c)
    torch.set_num_threads(c['filter']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = Path(c['data']['root'])
    manifest = read(root/'manifest.json')
    if manifest['identity'] != identity(c):
        raise ValueError('Dataset identity mismatch')
    store = load_dataset(root)
    maps = pd.concat([pd.read_parquet(root/(d+'_map.parquet')) for d in c['data']['dates']], ignore_index=True)
    sources = bound_sources(args.config, manifest)
    sources[str((root/'manifest.json').resolve())] = sha(root/'manifest.json')
    outroot = Path('data/derived/scientific_pipeline_v5/runs')
    report = Path('reports/scientific_pipeline_v5')
    from spectrum.research_v5.targets import build_filter_targets
    from spectrum.research_v5.learning import train_filter
    runs = []
    for seed in c['experiment']['seeds']:
        out = outroot/f'filter_seed{seed}'
        if out.exists():
            r = read(out/'receipt.json')
            if r['status'] != 'COMPLETE' or r['sources'] != sources:
                raise ValueError('Existing incomplete or differently bound filter run')
            for path, digest in r['outputs'].items():
                if sha(out/path) != digest:
                    raise ValueError('Changed filter output')
            runs.append(dict(seed=seed, run_id=out.name, summary=read(out/'summary.json')))
            continue
        out.mkdir(parents=True)
        save(out/'receipt.json', dict(status='RUNNING', sources=sources, seed=seed, base=c['base_models'][str(seed)]))
        started = time.perf_counter()
        print(json.dumps(dict(stage='FROZEN_INFERENCE', seed=seed)), flush=True)
        x, p, z, side, names = frozen_features(store, c, seed)
        target = build_filter_targets(store.frame, maps, side, c)
        target['frame'].to_parquet(out/'targets.parquet', index=False)
        save(out/'target_receipt.json', target['receipt'])
        # Target adapter exposes exactly five masked binary labels; all labels
        # and future ID/path metadata stay outside the frozen_features function.
        labels = target['labels']
        eligible = store.frame.feature_available.to_numpy(bool)
        split = {k: np.flatnonzero(eligible & store.frame.date.isin(days).to_numpy())
                 for k, days in c['splits'].items()}
        save(out/'feature_schema.json', dict(names=names, features=len(names), base_frozen=True,
            theoretical_max_history_seconds=300, long_day_context=False, split_indices={k:len(v) for k,v in split.items()}))
        result = train_filter(x, labels, split['train'], split['validation'], c, out, seed)
        (out/'filter_predictions.parquet').write_bytes((out/'predictions.parquet').read_bytes())
        q = result['predictions']
        frame = store.frame[['date', 't_ns', 'feature_available', 'y_crossing', 'physical_segment_token', 'physical_support_available']].copy()
        frame['root_index'] = np.arange(len(frame)); frame['side'] = side
        for j, name in enumerate(PNAMES):
            frame[name] = p[:, j]
            frame['base_logit_'+str(j)] = z[:, j]
        for j, name in enumerate(QNAMES):
            frame[name] = q[:, j]
            frame['target_'+name] = labels[:, j]
        gradients, error = logit_gradients(z, store.frame.y_crossing.to_numpy(), c['loss'])
        for j in range(4):
            frame['base_diagnostic_grad_'+str(j)] = gradients[:, j]
        frame.to_parquet(out/'predictions.parquet', index=False)
        summary = dict(result['summary'], seed=seed, run_id=out.name, total_seconds=time.perf_counter()-started,
                       base_frozen=True, base_gradient_analytic_error=error, base_updates=0,
                       base_gradient_note='Hypothetical original-loss diagnostic only; no base backward/update executed',
                       split_coverage={k:dict(dates=c['splits'][k], available=len(v),
                           known_available=int((labels[v,0]>=0).sum())) for k,v in split.items()})
        summary['status']='COMPLETE'
        save(out/'summary.json', summary)
        if any(sha(pth) != digest for pth,digest in sources.items()):
            raise ValueError('Bound source changed during fit')
        validate_recipe(c)
        save(out/'receipt.json', dict(status='COMPLETE', sources=sources, seed=seed, base=c['base_models'][str(seed)],
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != 'receipt.json'}))
        runs.append(dict(seed=seed, run_id=out.name, summary=summary))
        print(json.dumps(dict(stage='COMPLETE', seed=seed, seconds=summary['total_seconds'])), flush=True)
        del x, p, z, q, target, labels, frame
        torch.cuda.empty_cache()
    save(report/'suite_summary.json', dict(status='COMPLETE', config_sha256=sha(args.config),
        dataset_root=str(root), dataset_manifest_sha256=sha(root/'manifest.json'), runs_root=str(outroot), runs=runs,
        original_target_loss_unchanged=True, base_models_frozen=True))


if __name__ == '__main__':
    main()
