"""Build complete versioned feature families, fixed crossing labels and risk trials.

All clock roots survive. Completeness is published only after every selected day
and every source/output hash is checked. Day receipts allow an interrupted build
to resume without admitting partial Parquet or arrays.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import psutil

from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.research.training import sha, save_json
from spectrum.research_v2.features import build_features
from spectrum.research_v2.targets import build_targets

_VERIFIED = {}


def identity(c):
    files = ['scripts/prepare_opportunity_v2.py', 'spectrum/research_v2/features.py',
             'spectrum/research_v2/targets.py', 'spectrum/research_v2/simulation.py',
             'spectrum/cognition/bb_crossing.py', 'spectrum/cognition/crossing_map.py',
             'spectrum/cognition/flow_context.py', 'spectrum/cognition/execution.py',
             'spectrum/playground/data.py', 'scripts/crossing_map_analyze.py',
             'scripts/bb_crossing_session.py']
    data_keys = ('dates', 'market_root', 'source_root', 'admission_overlay', 'source_inventory',
                 'source_inventory_sha256', 'cadence_seconds', 'max_feature_age_seconds')
    return dict(version='MULTIBLOCK_CROSSING_FEATURE_DATASET_V2',
                data={k: c['data'][k] for k in data_keys}, event_query=c['event_query'],
                execution=c['execution'], risk=c['risk'], splits=c['splits'],
                sources={p: sha(p) for p in files})


def build_day(args):
    date, c, ids = args
    out = Path(c['data']['root'])
    receipt_path = out / (date + '.json')
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text(encoding='utf-8'))
        if r['identity'] != ids or r['status'] != 'COMPLETE':
            raise ValueError('Cache identity mismatch: preserve the old dataset and choose a new root')
        for p, h in r['files'].items():
            if sha(out / p) != h:
                raise ValueError('Changed output: ' + p)
        for p, h in r['source_sha256'].items():
            if p not in _VERIFIED:
                _VERIFIED[p] = sha(p)
            if _VERIFIED[p] != h:
                raise ValueError('Changed source: ' + p)
        print(json.dumps(dict(date=date, status='VERIFIED_CACHE', roots=r['roots'])), flush=True)
        return r
    if shutil.disk_usage(out).free < 10 * 1024**3:
        raise RuntimeError('Reserve at least 10 GiB; no partial data admission')
    start = time.perf_counter()
    day_path = Path(c['data']['market_root']) / date[:7] / date
    day = MonthDay(day_path, verify=True)
    raw_hashes = {str(day_path / 'manifest.json'): sha(day_path / 'manifest.json')}
    for part in day.meta['quote_parts']:
        path = str(Path(part['root']) / 'quotes.parquet')
        if path not in _VERIFIED:
            _VERIFIED[path] = sha(path)
        if _VERIFIED[path] != part['quote_file']['sha256']:
            raise ValueError('Raw quote source changed')
        raw_hashes[path] = _VERIFIED[path]
    for name, spec in day.meta['files'].items():
        raw_hashes[str(day_path / name)] = spec['sha256']  # MonthDay verified the bytes
    q = QuoteTape.from_day(day)
    t = np.arange(q.entry_start_ns, q.deadline_ns, int(c['data']['cadence_seconds'] * 1e9), dtype=np.int64)
    targets = build_targets(day, q, t, c)
    features = build_features(day, q, t, targets['scene'], c)
    roots = targets['roots'].copy()
    labels = targets['labels']
    if not np.array_equal(roots.t_ns, t) or not np.array_equal(labels.t_ns, t):
        raise ValueError('Target root alignment changed')
    for name in labels:
        if name not in roots:
            roots[name] = labels[name].to_numpy()
        else:
            numeric = np.issubdtype(roots[name].dtype, np.number)
            same = (np.array_equal(roots[name], labels[name], equal_nan=True)
                    if numeric else roots[name].equals(labels[name]))
            if not same:
                raise ValueError('Conflicting root and label column: ' + name)
    roots['feature_available'] = features['feature_available']
    roots['segment_token'] = features['segment_token']
    if 'y_crossing' not in roots or len(features['x_raw']) != len(roots):
        raise ValueError('Missing aligned joint target')
    if 'date' not in roots:
        roots.insert(0, 'date', date)
    paths = {}
    frames = dict(roots=roots, scene=targets['scene'], bank=targets['bank'],
                  map=targets['map'], execution=targets['executions'])
    for name, frame in frames.items():
        path = out / f'{date}_{name}.parquet'
        frame.to_parquet(path, index=False)
        paths[path.name] = sha(path)
    path = out / f'{date}_x.npy'
    np.save(path, features['x_raw'].astype(np.float32))
    paths[path.name] = sha(path)
    raw_hashes.update(features['receipts'].get('source_sha256', {}))
    known = roots.feature_available.to_numpy(bool) & (roots.y_crossing.to_numpy() >= 0)
    receipt = dict(status='COMPLETE', date=date, symbol=day.meta['symbol'], identity=ids,
        roots=len(roots), feature_available=int(roots.feature_available.sum()), admitted=int(known.sum()),
        classes=np.bincount(roots.y_crossing.to_numpy()[known], minlength=4).tolist(),
        feature_names=features['feature_names'], groups=features['groups'],
        roots_file=f'{date}_roots.parquet', x_file=f'{date}_x.npy', files=paths,
        source_sha256=raw_hashes, feature_receipt=features['receipts'], target_receipt=targets['receipts'],
        quotes=len(q.quote_t), seconds=time.perf_counter() - start,
        process_rss_mb=psutil.Process().memory_info().rss / 1024**2)
    save_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ('date', 'roots', 'admitted', 'classes', 'seconds', 'process_rss_mb')}), flush=True)
    del q, day, roots, labels, targets, features, frames
    QUOTE_CACHE.clear()
    gc.collect()
    return receipt


def finalize(c, ids):
    out = Path(c['data']['root'])
    receipts = []
    for date in c['data']['dates']:
        p = out / (date + '.json')
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        if r['status'] != 'COMPLETE' or r['identity'] != ids:
            raise ValueError('Unmatched daily build')
        for name, h in r['files'].items():
            if sha(out / name) != h:
                raise ValueError('Changed daily payload')
        receipts.append(r)
    names, groups = receipts[0]['feature_names'], receipts[0]['groups']
    if any(r['feature_names'] != names or r['groups'] != groups for r in receipts):
        raise ValueError('Feature schema drift between days')
    manifest = dict(status='COMPLETE', identity=ids, dates=c['data']['dates'], splits=c['splits'],
        feature_names=names, groups=groups, days=receipts,
        roots=sum(r['roots'] for r in receipts), admitted=sum(r['admitted'] for r in receipts),
        family_storage='Three complete logical datasets share immutable NxF payloads, origin keys and target bank; no duplicated windows',
        window='Batch-only gather, no saved NxTxF; 5s origins, current and prior London compressed context',
        semantics='BB crossing event fixed across risk plans; R_event distinct from trade risk; masks, source support and label maturity explicit')
    save_json(out / 'manifest.json', manifest)
    family_dir = out / 'families'
    family_dir.mkdir(exist_ok=True)
    for family, columns in groups.items():
        save_json(family_dir / (family + '.json'), dict(status='COMPLETE', family=family,
            parent_manifest='../manifest.json', parent_manifest_sha256=sha(out / 'manifest.json'),
            feature_names=columns, feature_indices=[names.index(n) for n in columns],
            splits=c['splits'], days=[dict(date=r['date'], x_file='../' + r['x_file'], roots_file='../' + r['roots_file'],
                x_sha256=r['files'][r['x_file']], roots_sha256=r['files'][r['roots_file']]) for r in receipts]))
    save_json(out / 'progress.json', dict(status='COMPLETE', completed=len(receipts), expected=len(receipts), roots=manifest['roots']))
    return manifest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/research/opportunity_pipeline_v2.json')
    p.add_argument('--date', action='append')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--finalize-only', action='store_true')
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if sha(c['data']['source_inventory']) != c['data']['source_inventory_sha256']:
        raise ValueError('Coverage-only source selection changed')
    out = Path(c['data']['root'])
    out.mkdir(parents=True, exist_ok=True)
    ids = identity(c)
    if not a.finalize_only:
        dates = a.date or c['data']['dates']
        if not set(dates) <= set(c['data']['dates']) or a.workers not in (1, 2):
            raise ValueError('Select declared dates and at most two quote loaders')
        jobs = [(d, c, ids) for d in dates]
        if a.workers == 1:
            for job in jobs:
                build_day(job)
        else:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                for _ in pool.map(build_day, jobs):
                    pass
    result = finalize(c, ids)
    print(json.dumps(dict(status='COMPLETE' if result else 'PARTIAL_NOT_ADMITTED', dates=len(c['data']['dates']),
                          roots=result['roots'] if result else None)), flush=True)


if __name__ == '__main__':
    main()
