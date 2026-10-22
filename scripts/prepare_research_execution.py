"""Materialize versioned paired execution outcomes and raw-quote checks.

Usage: python -m scripts.prepare_research_execution --config CONFIG --output DIR
The reusable cache binds only execution/data parameters, while each invocation
records its full configuration hash. Model/loss edits do not recompute outcomes.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.research.simulation import SimulationConfig, simulate_tape, scalar_reference


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    c = SimulationConfig.from_dict(config)
    data = config.get('data', {})
    dates = data.get('dates', [f'2026-02-{d:02}' for d in range(2, 7)])
    roots_dir = Path(data.get('roots_dir', 'data/derived/crossing_map_target_v1'))
    market_dir = Path(data.get('market_dir', 'data/derived/basket_multimonth_v1'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), Path('spectrum/research/simulation.py'),
                  Path('spectrum/cognition/execution.py'), Path('spectrum/playground/data.py'),
                  Path('spectrum/player_v3/data.py')]
    identity = dict(version='CAUSAL_PAIRED_FIXED_PLAN_CLOCK_EXECUTION_V2',
        execution_config=asdict(c), dates=dates, roots_dir=str(roots_dir.resolve()),
        market_dir=str(market_dir.resolve()), code={str(p): sha(p) for p in code_paths},
        libraries={name: version(name) for name in ('numpy', 'pandas', 'numba', 'pyarrow')})
    config_file_sha = sha(args.config)
    receipts = []
    verified_quotes = {}
    source_manifest = roots_dir/'manifest.json'
    root_manifest_sha = sha(source_manifest)
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source.get('status') != 'COMPLETE':
        raise ValueError('Only a complete root dataset may be consumed')
    for date in dates:
        root_path = roots_dir/(date+'_roots.parquet')
        root_sha = sha(root_path)
        source_receipt = json.loads((roots_dir/(date+'.json')).read_text(encoding='utf-8'))
        if source_receipt['files'][root_path.name] != root_sha:
            raise ValueError('Changed causal risk roots')
        receipt_path = out/(date+'.json')
        source_identity = dict(root_sha256=root_sha, root_manifest_sha256=root_manifest_sha)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['identity'] != identity or receipt['root_source'] != source_identity:
                raise ValueError('Cached execution identity changed; choose a new output directory')
            for path, expected in receipt['raw_sources'].items():
                if path not in verified_quotes:
                    verified_quotes[path] = sha(path)
                if verified_quotes[path] != expected:
                    raise ValueError('Cached execution source changed: '+path)
            if sha(out/receipt['file']) != receipt['sha256']:
                raise ValueError('Cached execution file changed')
            receipts.append(receipt)
            print(json.dumps(dict(date=date, state='VERIFIED_CACHE', rows=receipt['rows'])), flush=True)
            continue
        day_path = market_dir/date[:7]/date
        day = MonthDay(day_path, verify=True)
        raw_sources = {str(day_path/'manifest.json'): sha(day_path/'manifest.json')}
        for part in day.meta['quote_parts']:
            path = str(Path(part['root'])/'quotes.parquet')
            if path not in verified_quotes:
                verified_quotes[path] = sha(path)
            if verified_quotes[path] != part['quote_file']['sha256']:
                raise ValueError('Raw quote hash changed')
            raw_sources[path] = verified_quotes[path]
        for name, spec in day.meta['files'].items():
            raw_sources[str(day_path/name)] = spec['sha256']
        roots = pd.read_parquet(root_path)
        q = QuoteTape.from_day(day)
        frame = simulate_tape(q, roots.t_ns.to_numpy(np.int64), roots.risk_R.to_numpy(float), c)
        frame.insert(0, 'date', date)
        checks = []
        for i in np.linspace(0, len(roots)-1, 10, dtype=int):
            for side in (1, -1):
                t = int(roots.t_ns.iloc[i])
                expected = scalar_reference(q, t, float(roots.risk_R.iloc[i]), side, c)
                actual = frame.loc[(frame.t_ns == t) & (frame.side == side)].iloc[0]
                max_error = 0.
                for key, value in expected.items():
                    if isinstance(value, float):
                        if not np.isclose(actual[key], value, atol=1e-10, rtol=0, equal_nan=True):
                            raise ValueError(f'Raw benchmark mismatch {date} {t} {side} {key}')
                        if np.isfinite(value):
                            max_error = max(max_error, abs(float(actual[key])-value))
                    elif actual[key] != value:
                        raise ValueError(f'Raw benchmark mismatch {date} {t} {side} {key}: {actual[key]} != {value}')
                checks.append(dict(t_ns=str(t), side=side, status=expected['status'], max_money_error=max_error))
        path = out/(date+'.parquet')
        frame.to_parquet(path, index=False)
        receipt = dict(date=date, identity=identity, root_source=source_identity,
            raw_sources=raw_sources, full_config_sha256_at_build=config_file_sha,
            file=path.name, sha256=sha(path), quote_rows=len(q.quote_t), rows=len(frame),
            status_counts={str(k): int(v) for k, v in frame.status.value_counts().items()},
            reason_counts={str(k): int(v) for k, v in frame.reason.value_counts().items()},
            checks=checks)
        save(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps(dict(date=date, state='COMPLETE', rows=len(frame), status=receipt['status_counts'],
                              raw_checks=len(checks))), flush=True)
        del frame, q, roots, day
        QUOTE_CACHE.clear()
        gc.collect()
    manifest = dict(status='COMPLETE', identity=identity, days=receipts,
        full_config_sha256_current_invocation=config_file_sha,
        observations=sum(r['rows'] for r in receipts),
        independent_raw_checks=sum(len(r['checks']) for r in receipts),
        semantics={'target': 'Paired outcomes of exactly the same causal fixed plan',
                   'censor': 'Unknown outcomes carry NaN; never loss/zero labels',
                   'execution': 'One reference ounce, taker BBO fills with explicit latency; no queue/depth/impact claim',
                   'path': 'MAE/MFE includes entry cost and all publications through the actual exit',
                   'clock': 'TIME request_ns equals entry_ns plus duration exactly; TIME wins same-ns quotes; trigger_ordinal=-1 for timer',
                   'cache': 'Model/loss changes reuse this effective-execution cache; full config hash recorded per invocation'})
    save(out/'manifest.json', manifest)
    print(json.dumps(dict(state='COMPLETE', observations=manifest['observations'],
                          independent_raw_checks=manifest['independent_raw_checks'])), flush=True)


if __name__ == '__main__':
    main()
