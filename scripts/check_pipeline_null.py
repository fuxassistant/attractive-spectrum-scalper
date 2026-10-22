"""Independent per-split label-permutation null for the same training stack.

The original train-only shuffle still selects on genuine validation labels.
This stricter null deliberately breaks X/Y association in every split. It is
one diagnostic randomization, not a market significance test or leakage proof.
"""
import copy
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from spectrum.research.controls import make_control_data
from spectrum.research.training import fit_model, predict, metrics, save_json, sha


def main():
    config_path = Path('configs/research/opportunity_pipeline_v1.json')
    c = json.loads(config_path.read_text(encoding='utf-8'))
    c['run_id'] = 'independent_split_shuffle_seed1701'
    c['training'].update(max_epochs=40, patience=12)
    out = Path('data/derived/scientific_pipeline_v1/null_independent')
    if out.exists():
        raise ValueError('Immutable null run already exists')
    out.mkdir(parents=True)
    save_json(out / 'config.json', c)
    rngs = [np.random.default_rng(s) for s in (1701, 1702, 1703)]
    arrays = [make_control_data(seed, count) for seed, count in ((117, 768), (317, 384), (217, 384))]
    labels = [rng.permutation(a['y']) for rng, a in zip(rngs, arrays)]
    model, history, receipt = fit_model(arrays[0]['x'], labels[0], arrays[1]['x'], labels[1], c, out)
    result = dict(status='COMPLETE', training=receipt, test=metrics(labels[2], predict(model, arrays[2]['x'], c)),
                  permutations=[1701, 1702, 1703], synthetic_receipts=[a['receipts'] for a in arrays],
                  selection='minimum NLL on independently shuffled validation labels; test labels independently shuffled too',
                  scope='One synthetic IID association-breaking control; not a market permutation p-value or a proof of no leakage',
                  source_sha256={str(p): sha(p) for p in [Path(__file__), config_path, *Path('spectrum/research').glob('*.py')]})
    save_json(out / 'summary.json', result)
    report = Path('reports/scientific_pipeline_v1')
    report.mkdir(parents=True, exist_ok=True)
    save_json(report / 'independent_null.json', result)
    print(json.dumps(result['test']), flush=True)


if __name__ == '__main__':
    main()
