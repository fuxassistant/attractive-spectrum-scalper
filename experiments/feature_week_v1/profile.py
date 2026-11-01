"""Reproducible full observation profile; raw books are never split arbitrarily.

Benchmarks read the same admitted source captures and regenerate all six second
families, exact interaction observations, completed M1 features and context.
Action-conditioned research wrappers are not falsely called a universal X table.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_context import ROOT, OUTPUT, DATES, WEEK, source_for
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from spectrum.io import save_json, sha
from spectrum.core.features import build_features
from spectrum.features.pipeline import compute, REGISTRY, implementation_identity
from spectrum.representation.representation import (
    ledger_releases, make_fast, make_minutes, anchored_context, anchor,
    FAST_VALUES, DAY_VALUES, NS, MINUTE)
from spectrum.representation.zones import zone_and_event_tables
from spectrum.representation.store import publish_observation_store, open_observation_store

FAMILIES = tuple(REGISTRY)


def raw_for(date):
    root = OUTPUT / 'raw' / date
    if not (root / 'manifest.json').exists():
        root = ROOT / 'data/derived/market_representation_v1/raw' / date
    return root


def checked_inputs(date):
    source = source_for(date)
    market, raw = Path(source['market']), raw_for(date)
    mm = json.loads((market / 'manifest.json').read_text())
    rm = json.loads((raw / 'manifest.json').read_text())
    if mm['status'] != 'COMPLETE' or rm['status'] != 'COMPLETE':
        raise ValueError('Incomplete source')
    if source['entry']['sha256'] != rm['source']['partition_sha256']:
        raise ValueError('Exact facts belong to a different source partition')
    checks = {}
    for base, hashes, names in ((market, mm['payloads'], ('seconds.parquet', 'bars.parquet', 'quotes.parquet')),
                                (raw, rm['outputs'], ('levels_1s.parquet', 'trades.parquet'))):
        for name in names:
            h = sha(base / name)
            if h != hashes[name]:
                raise ValueError(f'Payload drift: {base / name}')
            checks[str(base / name)] = h
    return market, raw, checks


def exact_moments(seconds, trades, release):
    """T-only central moments; physical buckets and publication clocks preserved."""
    tr = trades.copy()
    tr['bucket'] = (tr.t_ns // NS + 1) * NS
    tr['price'] = tr.price_nano / NS
    tr['reference'] = tr.groupby('bucket').price.transform('first')
    tr['qdx'] = tr.quantity * (tr.price - tr.reference)
    tr['qdx2'] = tr.quantity * (tr.price - tr.reference)**2
    tr['unsupported'] = tr.quantity.where(~tr.supported, 0)
    grouped = tr.groupby('bucket').agg(q=('quantity', 'sum'), reference=('reference', 'first'),
              dx=('qdx', 'sum'), dx2=('qdx2', 'sum'), unsupported=('unsupported', 'sum'),
              available=('available_ns', 'max'))
    aligned = grouped.reindex(seconds.t_ns, fill_value=0)
    q = aligned.q.to_numpy(float)
    offset = np.divide(aligned.dx.to_numpy(float), q, out=np.zeros(len(q)), where=q > 0)
    m2 = aligned.dx2.to_numpy() - q * offset**2
    if not np.isfinite(m2).all() or np.any(m2 < -1e-8):
        raise ValueError('Negative central trade variance')
    frame = seconds.copy()
    frame['trade_volume'] = q
    frame['trade_price_mean'] = np.where(q > 0, aligned.reference.to_numpy() + offset, 0)
    frame['trade_price_m2'] = np.maximum(m2, 0)
    frame['trade_moments_valid'] = (aligned.unsupported.to_numpy() == 0) & (release < np.iinfo(np.int64).max)
    frame['trade_moments_available_ns'] = np.maximum(release, aligned.available.to_numpy(np.int64))
    frame['available_ns'] = np.maximum.accumulate(np.maximum(frame.available_ns, frame.trade_moments_available_ns))
    return frame


def write_frame(frame, path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression='zstd', row_group_size=32768)
    if not pq.read_table(path).equals(table):
        raise ValueError(f'Parquet roundtrip differs: {path}')
    return sha(path)


def generate_day(date, run):
    start = time.perf_counter()
    out = Path(run) / 'days' / date
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'manifest.json', dict(status='RUNNING', pid=os.getpid(), date=date))
    market, raw, identities = checked_inputs(date)
    inputs_checked = time.perf_counter()
    seconds = pd.read_parquet(market / 'seconds.parquet')
    quotes = pd.read_parquet(market / 'quotes.parquet', columns=['t_ns', 'available_ns'])
    release = ledger_releases(seconds, quotes)
    del quotes
    levels, trades = pd.read_parquet(raw / 'levels_1s.parquet'), pd.read_parquet(raw / 'trades.parquet')
    frame = exact_moments(seconds, trades, release)
    fast_source = seconds.copy()
    fast_source['ledger_release_ns'] = release
    fast = make_fast(fast_source, levels, trades)
    del levels, trades, fast_source
    prepared = time.perf_counter()
    c = compute(frame, FAMILIES)
    calculated = time.perf_counter()
    columns = {k: frame[k].to_numpy() for k in ('t_ns', 'available_ns', 'epoch')}
    columns.update(admissible=c.admitted, scheduled_closed=c.closed)
    for name in c.values:
        columns[name] = c.values[name]
        columns[name + '__known'] = c.masks[name]
    features = pd.DataFrame(columns)
    files = {name: write_frame(data, out / name) for name, data in
             [('input_seconds.parquet', frame), ('features.parquet', features), ('fast.parquet', fast)]}
    result = dict(status='COMPLETE', date=date, pid=os.getpid(), rows=len(frame),
                  features=len(c.values), families=list(FAMILIES), units=c.units,
                  coverage={name: float(mask.mean()) for name, mask in c.masks.items()},
                  source=identities, files=files, implementation_sha256=implementation_identity(),
                  seconds=dict(source_validation=inputs_checked-start, prepare=prepared-inputs_checked,
                               formulas=calculated-prepared, write_verify=time.perf_counter()-calculated,
                               total=time.perf_counter()-start),
                  rss_sample=psutil.Process().memory_info().rss,
                  working_set_peak=getattr(psutil.Process().memory_info(), 'peak_wset', None))
    save_json(out / 'manifest.json', result)
    return result


def assemble(run):
    """Ordered context phase. Explicit preceding captures are an input dependency."""
    start = time.perf_counter()
    run = Path(run)
    fast = pd.concat([pd.read_parquet(run / 'days' / d / 'fast.parquet') for d in DATES], ignore_index=True)
    trades = pd.concat([pd.read_parquet(raw_for(d) / 'trades.parquet') for d in DATES], ignore_index=True)
    bars = pd.concat([pd.read_parquet(Path(source_for(d)['market']) / 'bars.parquet') for d in DATES], ignore_index=True)
    seconds = pd.concat([pd.read_parquet(Path(source_for(d)['market']) / 'seconds.parquet') for d in DATES], ignore_index=True)
    origins, values, core_manifest = build_features(seconds, bars, snapshot_ns=NS)
    origins = origins.rename(columns={n:'origin_'+n for n in origins if n in core_manifest['feature_names']})
    core = pd.concat([origins.reset_index(drop=True), pd.DataFrame(values, columns=core_manifest['feature_names'])], axis=1)
    # This fixture starts on Friday, so the preceding ISO week is incomplete.
    # A range of a partial week is not a known range of the whole previous week.
    masked_previous_week = []
    for name in core_manifest['feature_names']:
        if name.startswith('previous_week_') and not name.endswith('_known'):
            core[name] = 0.
            if name + '_known' in core:
                core[name + '_known'] = 0.
            masked_previous_week.append(name)
    core_manifest['fixture_dependency_masks'] = dict(previous_week=masked_previous_week,
        reason='Fixture has only Friday/Sunday of the preceding ISO week; no whole-week claim')
    write_frame(core, run / 'core_m1.parquet')
    save_json(run / 'core_m1_manifest.json', core_manifest)
    del core, origins, values, seconds
    gc.collect()
    minutes = make_minutes(fast, trades, bars)
    write_frame(minutes, run / 'minutes.parquet')
    inventory = []
    for i, date in enumerate(WEEK):
        previous = '2026-08-21' if i == 0 else WEEK[i-1]
        context = anchored_context(minutes, trades, date, previous)
        local = fast[(fast.t_ns > anchor(date)-900*NS) & (fast.t_ns <= anchor(date)+17*3600*NS)].copy()
        levels = pd.read_parquet(raw_for(date) / 'levels_1s.parquet')
        zones, events = zone_and_event_tables(local, context, levels, date, 'GCZ6')
        destination = run / 'stores' / date
        publish_observation_store(destination, date=date, previous_date=previous, symbol='GCZ6',
                                  fast=local, context=context, zones=zones, events=events,
                                  source_identity=dict(profile='FEATURE_WEEK_V1', dates=list(DATES),
                                      minute_sha256=sha(run / 'minutes.parquet'),
                                      input_receipts={d: sha(run / 'days' / d / 'manifest.json') for d in DATES}))
        store = open_observation_store(destination, date)
        queries = []
        for hour in (8, 12, 17):
            t = int(pd.Timestamp(f'{date} {hour:02}:00', tz='UTC').value)
            observation = store.observe(t)
            for tensor in (observation.fast, observation.day):
                if len(tensor['available_ns']) and np.any(tensor['available_ns'] > t):
                    raise ValueError('Loader leaks a later publication')
            queries.append(observation.receipt)
        inventory.append(dict(date=date, previous=previous, fast_rows=len(local), context_rows=len(context),
                              zones=len(zones), events=len(events), queries=queries))
        del context, levels, zones, events, store
    return dict(elapsed_seconds=time.perf_counter()-start, inventory=inventory,
                core_columns=core_manifest['columns'], fast_channels=len(FAST_VALUES), day_channels=len(DAY_VALUES))


def run_profile(name, workers):
    run = OUTPUT / name
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / 'manifest.json', dict(status='RUNNING', workers=workers, pid=os.getpid()))
    start = time.perf_counter()
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(generate_day, d, str(run)) for d in DATES]
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                print(json.dumps(dict(stage='day', date=r['date'], features=r['features'], seconds=r['seconds'])), flush=True)
        local_elapsed = time.perf_counter()-start
        save_json(run / 'day_stage.json', dict(status='COMPLETE', workers=workers,
                   elapsed_seconds=local_elapsed, dates=list(DATES)))
        context = assemble(run)
        result = dict(status='COMPLETE', version='FEATURE_WEEK_V1', workers=workers, dates=list(DATES),
                      target_week=list(WEEK), days_elapsed_seconds=local_elapsed, context=context,
                      total_elapsed_seconds=time.perf_counter()-start,
                      day_results=[dict(date=r['date'], rows=r['rows'], features=r['features'],
                                        seconds=r['seconds'], working_set_peak=r['working_set_peak'])
                                   for r in sorted(results, key=lambda r:r['date'])],
                      scope='Six second families, M1 core, exact fast ledger, M1 BB/spectrum, anchored context, zones/events',
                      cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False),
                      source_policy='Bound immutable canonical captures and exact sidecars; separate raw extraction timing',
                      implementation={str(p.relative_to(ROOT)):sha(p) for p in
                                      [Path(__file__),HERE/'extract_context.py']},
                      benchmark_caveat='Warm filesystem caches; includes hashes, preparation, formulas, writes and read-back; one run per worker setting, no confidence interval')
        save_json(run / 'manifest.json', result)
        print(json.dumps(dict(stage='complete', run=name, days_seconds=local_elapsed,
                              context_seconds=context['elapsed_seconds'], total_seconds=result['total_elapsed_seconds'])), flush=True)
    except Exception as exc:
        save_json(run / 'manifest.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--smoke-date')
    a = p.parse_args()
    if a.smoke_date:
        r = generate_day(a.smoke_date, str(OUTPUT / a.name))
        print(json.dumps({k:r[k] for k in ('date', 'rows', 'features', 'seconds')}), flush=True)
    else:
        run_profile(a.name, a.workers)
