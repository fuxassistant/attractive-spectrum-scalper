"""Independent saved-data/source replay for three declared scientific V3 days.

This read-only audit never admits partial daily payloads and never calls the V3
feature wrapper. Frozen V2 reconstructs X, labels and execution; separate clock
and full-quote support calculations verify the additional V3 metadata.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.research_v2.features import build_features as build_v2_features
from spectrum.research_v2.targets import build_targets as build_v2_targets

NS = 1_000_000_000
DEFAULT_DATES = ('2026-06-01', '2026-06-25', '2026-07-13')
METADATA = ('t_ns', 'available_ns', 'epoch', 'admissible', 'source_day_degraded')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class BoundInputs:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        before = path.stat()
        identity = (before.st_size, before.st_mtime_ns)
        if key not in self.files:
            digest = sha(path)
            after = path.stat()
            if identity != (after.st_size, after.st_mtime_ns):
                raise ValueError('Input changed while hashing: ' + key)
            self.files[key] = dict(sha256=digest, size=identity[0], mtime_ns=str(identity[1]))
        item = self.files[key]
        if (item['size'], int(item['mtime_ns'])) != identity:
            raise ValueError('Input changed during verification: ' + key)
        if expected is not None and item['sha256'] != expected:
            raise ValueError('Bound SHA mismatch: ' + key)
        return item['sha256']

    def confirm_unchanged(self):
        for key in self.files:
            self.bind(key)


def exact_array(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f'{name}: shape or dtype differs: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}')
    equal = np.array_equal(actual, expected, equal_nan=True) if np.issubdtype(actual.dtype, np.number) else np.array_equal(actual, expected)
    if not equal:
        raise AssertionError(name + ': values differ')


def exact_frame(actual, expected, name):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_exact=True, check_dtype=True)
    except AssertionError as exc:
        raise AssertionError(name + ': ' + str(exc)) from exc
    return dict(rows=len(expected), columns=list(expected.columns), exact=True)


def independent_quote_support(q, clocks, gap_ns):
    """One full-publication scan; no production quote_state invocation."""
    qt = np.asarray(q.quote_t, np.int64)
    if len(qt) == 0 or np.any(qt[1:] < qt[:-1]):
        raise ValueError('Nonempty ordered quote tape required')
    valid = np.asarray(q.valid, bool) & (q.epoch >= 0) & (q.bid > 0) & (q.ask > q.bid)
    transitions = np.empty(len(qt), bool)
    transitions[0] = True
    transitions[1:] = ((~valid[1:]) | (~valid[:-1]) | (q.epoch[1:] != q.epoch[:-1])
                       | ((qt[1:] - qt[:-1]) > gap_ns))
    physical = np.cumsum(transitions, dtype=np.int64)
    clocks = np.asarray(clocks, np.int64)
    ordinal = np.searchsorted(qt, clocks, side='right') - 1
    safe = np.maximum(ordinal, 0)
    supported = (ordinal >= 0) & valid[safe] & ((clocks - qt[safe]) <= gap_ns)
    return dict(supported=supported, token=np.where(supported, physical[safe], -1), ordinal=ordinal,
                quote_transitions=int(transitions.sum()), invalid_quote_rows=int((~valid).sum()))


def independent_source_proof(source, origins, age_limit):
    """Recover endpoint/publication selection and original source segmentation."""
    endpoints = source.t_ns.to_numpy(np.int64)
    if len(endpoints) == 0 or np.any(np.diff(endpoints) <= 0):
        raise ValueError('Strictly ordered source endpoints required')
    publication = np.maximum.accumulate(source.available_ns.to_numpy(np.int64))
    # The selected row must satisfy BOTH clocks. Publication is prefix closed.
    endpoint_count = np.searchsorted(endpoints, origins, side='right')
    publication_count = np.searchsorted(publication, origins, side='right')
    selected = np.minimum(endpoint_count, publication_count) - 1
    safe = np.maximum(selected, 0)
    age = (origins - endpoints[safe]) / NS
    source_good = source.admissible.to_numpy(bool) & ~source.source_day_degraded.to_numpy(bool)
    ready = (selected >= 0) & source_good[safe] & (age >= 0) & (age <= age_limit)
    epoch = source.epoch.to_numpy(np.int64)
    part = source.source_part.to_numpy(np.int64)
    boundary = np.ones(len(source), bool)
    boundary[1:] = ((np.diff(endpoints) != NS) | (epoch[1:] != epoch[:-1]) |
                    (part[1:] != part[:-1]) | ~source_good[1:] | ~source_good[:-1])
    segment = np.cumsum(boundary, dtype=np.int64)
    return dict(selected=selected, ready=ready, endpoint=np.where(selected >= 0, endpoints[safe], -1),
                published=np.where(selected >= 0, publication[safe], -1),
                age=np.where(selected >= 0, age, np.nan), source_segment=np.where(ready, segment[safe], -1),
                late_source_rows=int(((source.available_ns.to_numpy(np.int64) - endpoints) > NS).sum()))


def load_source_metadata(day, config):
    catalog = read_json(Path(config['data']['source_root']) / 'manifest.json')
    captures = {x['key']: x for x in catalog['captures'] if x.get('status') == 'COMPLETE'}
    frames = []
    for number, part in enumerate(day.meta['quote_parts']):
        path = Path(captures[part['key']]['features']) / 'features.parquet'
        frame = pq.read_table(path, columns=list(METADATA),
            filters=[('t_ns', '>=', int(part['start_ns'])), ('t_ns', '<', int(part['end_ns']))]).to_pandas()
        frame['source_part'] = number
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values('t_ns').reset_index(drop=True)


def verify_clocks(q, source, roots, x, config):
    t = roots.t_ns.to_numpy(np.int64)
    age_limit = float(config['data']['max_feature_age_seconds'])
    proof = independent_source_proof(source, t, age_limit)
    state = independent_quote_support(q, np.r_[t, proof['published']],
        int(round(config['execution']['max_quote_gap_seconds'] * NS)))
    n = len(t)
    physical = state['token'][:n]
    supported = state['supported'][:n]
    same_support = supported & state['supported'][n:] & (physical == state['token'][n:])
    admitted = proof['ready'] & same_support
    exact_array(roots.feature_available.to_numpy(bool), admitted, 'independent admission')
    exact_array(roots.physical_support_available.to_numpy(bool), supported, 'physical support')
    exact_array(roots.physical_segment_token.to_numpy(np.int64), physical, 'physical token full scan')
    exact_array(roots.source_endpoint_ns.to_numpy(np.int64), proof['endpoint'], 'source endpoint')
    exact_array(roots.source_available_ns.to_numpy(np.int64), proof['published'], 'source publication')
    exact_array(roots.source_age_seconds.to_numpy(float), proof['age'], 'source endpoint age')
    selected = proof['selected'] >= 0
    if np.any(selected & ((proof['endpoint'] > t) | (proof['published'] > t))):
        raise AssertionError('Source endpoint/publication lookahead')
    if np.any(admitted & ((proof['age'] < 0) | (proof['age'] > age_limit))):
        raise AssertionError('Admitted stale source')
    boundary = np.ones(n, bool)
    boundary[1:] = (~admitted[1:] | ~admitted[:-1] |
        (proof['source_segment'][1:] != proof['source_segment'][:-1]) |
        (physical[1:] != physical[:-1]) |
        (np.diff(t) != int(round(config['data']['cadence_seconds'] * NS))))
    legacy = np.where(admitted, np.cumsum(boundary, dtype=np.int64), -1)
    exact_array(roots.legacy_segment_token.to_numpy(np.int64), legacy, 'legacy continuity independent')
    exact_array(roots.segment_token.to_numpy(np.int64), legacy, 'legacy alias')
    observed = np.isfinite(x) & admitted[:, None]
    if observed[~supported].any() or observed[~admitted].any():
        raise AssertionError('Observed feature outside admission/support')
    both = admitted[:-1] & admitted[1:]
    same_physical = supported[:-1] & supported[1:] & (physical[:-1] == physical[1:])
    removed = both & same_physical & (legacy[:-1] != legacy[1:])
    witnesses = []
    for i in np.flatnonzero(removed)[:5]:
        witnesses.append(dict(previous_origin_ns=str(t[i]), origin_ns=str(t[i+1]),
            physical_token=int(physical[i]), previous_legacy_token=int(legacy[i]), legacy_token=int(legacy[i+1]),
            source_endpoint_ns=str(proof['endpoint'][i+1]), source_publication_ns=str(proof['published'][i+1])))
    return dict(status='PASS', source_rows=len(source), selected_source_rows=int(selected.sum()),
        admitted=int(admitted.sum()), physical_support_origins=int(supported.sum()),
        missing_x_inside_physical_support=int((supported & ~admitted).sum()),
        current_available_legacy_resets_without_physical_break=int(removed.sum()),
        physical_breaks_between_available_origins=int((both & (physical[:-1] != physical[1:])).sum()),
        source_ready_but_crosses_physical_boundary=int((proof['ready'] & ~same_support).sum()),
        finite_observed_cells=int(observed.sum()), unavailable_rows_have_zero_observed_cells=True,
        max_admitted_endpoint_age_seconds=float(proof['age'][admitted].max()) if admitted.any() else None,
        max_selected_lookahead_ns=int(max(0, np.max(proof['published'][selected] - t[selected]))) if selected.any() else 0,
        late_source_rows=proof['late_source_rows'], quote_transitions=state['quote_transitions'],
        invalid_quote_rows=state['invalid_quote_rows'], witnesses=witnesses)


def verify_day(date, config, receipt, bound):
    started = time.perf_counter()
    root = Path(config['data']['root'])
    day_path = Path(config['data']['market_root']) / date[:7] / date
    day = MonthDay(day_path, verify=True)
    q = QuoteTape.from_day(day)
    roots = pd.read_parquet(root / receipt['roots_file'])
    x = np.load(root / receipt['x_file'], allow_pickle=False)
    t = roots.t_ns.to_numpy(np.int64)
    expected_t = np.arange(q.entry_start_ns, q.deadline_ns,
        int(config['data']['cadence_seconds'] * NS), dtype=np.int64)
    exact_array(t, expected_t, 'complete full-clock grid')
    rebuilt = build_v2_targets(day, q, t, config)
    comparisons = {}
    for name in ('roots', 'labels'):
        comparisons[name] = exact_frame(roots[list(rebuilt[name].columns)], rebuilt[name], name)
    for name, rebuilt_name in (('scene', 'scene'), ('bank', 'bank'), ('map', 'map'), ('execution', 'executions')):
        saved = pd.read_parquet(root / f'{date}_{name}.parquet')
        comparisons[name] = exact_frame(saved, rebuilt[rebuilt_name], name)
        del saved
    v2 = build_v2_features(day, q, t, rebuilt['scene'], config)
    exact_array(x, v2['x_raw'], 'all saved V2 X values and NaN masks')
    exact_array(roots.feature_available.to_numpy(bool), v2['feature_available'], 'V2 admission')
    exact_array(roots.legacy_segment_token.to_numpy(np.int64), v2['segment_token'], 'V2 legacy token')
    if receipt['feature_names'] != v2['feature_names'] or receipt['groups'] != v2['groups']:
        raise AssertionError('Feature family/schema differs from V2')
    for path, digest in v2['receipts']['source_sha256'].items():
        bound.bind(path, digest)
    source = load_source_metadata(day, config)
    clocks = verify_clocks(q, source, roots, x, config)
    executions = rebuilt['executions']
    plans = [f'cost{k:g}' for k in config['risk']['cost_multipliers']]
    risks = np.column_stack([roots['R_' + p].to_numpy(float) for p in plans])
    finite = np.isfinite(risks).all(axis=1)
    if (np.diff(risks[finite], axis=1) < 0).any():
        raise AssertionError('Risk budget is not monotonically increasing')
    gain = executions.pivot(index=['t_ns', 'side'], columns='plan', values='G_event')[plans].to_numpy()
    if not np.array_equal(gain, np.repeat(gain[:, :1], len(plans), axis=1), equal_nan=True):
        raise AssertionError('Risk plan redefined the event gain')
    statuses = executions.pivot(index=['t_ns', 'side'], columns='plan', values='status')[plans]
    triggers = executions.pivot(index=['t_ns', 'side'], columns='plan', values='trigger_ns')[plans]
    complete = statuses.eq('COMPLETE').all(axis=1).to_numpy()
    if (np.diff(triggers.to_numpy()[complete], axis=1) < 0).any():
        raise AssertionError('Wider stop triggered earlier on common completed trials')
    checks = rebuilt['receipts']['checks']
    result = dict(date=date, split=next(k for k, dates in config['splits'].items() if date in dates),
        status='PASS', symbol=day.meta['symbol'], roots=len(roots), quotes=len(q.quote_t),
        saved_x_shape=list(x.shape), feature_family_widths={k: len(v) for k, v in v2['groups'].items()},
        exact_v2_X_and_admission=True, exact_nan_masks=True, exact_target_and_execution_tables=comparisons,
        crossing_geometry_namespaces=rebuilt['receipts']['geometry_namespaces'],
        bank_events=len(rebuilt['bank']), map_cells=len(rebuilt['map']), execution_trials=len(executions),
        clock_and_support=clocks, frozen_scalar_checks=checks,
        risk_ordering=dict(common_complete_origin_side_rows=int(complete.sum()), nondecreasing_trade_risk=True,
            fixed_event_gain_across_plans=True, nondecreasing_trigger_time_on_common_complete=True),
        states=rebuilt['receipts']['states'], execution_statuses=rebuilt['receipts']['execution_statuses'],
        source_sha256=receipt['source_sha256'], dataset_files_sha256=receipt['files'],
        seconds=time.perf_counter() - started)
    del day, q, source, v2, rebuilt, roots, x, executions
    QUOTE_CACHE.clear()
    gc.collect()
    print(json.dumps({k: result[k] for k in ('date', 'status', 'roots', 'quotes', 'seconds')}), flush=True)
    return result


def run(config_path, dates, output):
    started = time.perf_counter()
    config = read_json(config_path)
    if len(set(dates)) != len(dates) or not set(dates) <= set(config['data']['dates']):
        raise ValueError('Unique declared dates required')
    root = Path(config['data']['root'])
    # Preflight ALL selected receipts before loading any payload or quote tape.
    receipts = []
    for date in dates:
        path = root / (date + '.json')
        if not path.exists():
            raise RuntimeError('Selected day is not COMPLETE yet: ' + date)
        receipt = read_json(path)
        if receipt.get('status') != 'COMPLETE' or receipt.get('date') != date:
            raise RuntimeError('Partial/mismatched daily receipt: ' + date)
        receipts.append(receipt)
    bound = BoundInputs()
    bound.bind(config_path)
    bound.bind(__file__)
    bound.bind(config['data']['source_inventory'], config['data']['source_inventory_sha256'])
    for receipt in receipts:
        identity = receipt['identity']
        for key in ('event_query', 'execution', 'risk', 'splits'):
            if identity[key] != config[key]:
                raise ValueError('Current config differs from daily binding: ' + key)
        for key, value in identity['data'].items():
            if config['data'][key] != value:
                raise ValueError('Data config differs from daily binding: ' + key)
        for path, digest in identity['sources'].items():
            bound.bind(path, digest)
        bound.bind(root / (receipt['date'] + '.json'))
        for path, digest in receipt['files'].items():
            bound.bind(root / path, digest)
        for path, digest in receipt['source_sha256'].items():
            bound.bind(path, digest)
    days = [verify_day(date, config, receipt, bound) for date, receipt in zip(dates, receipts)]
    bound.confirm_unchanged()
    report = dict(status='PASS', version='SCIENTIFIC_V3_INDEPENDENT_SOURCE_REPLAY_V1',
        config_path=str(Path(config_path).resolve()), config_sha256=sha(config_path),
        verifier_sha256=sha(__file__), selected_dates=list(dates), days=days,
        totals=dict(roots=sum(d['roots'] for d in days), quotes=sum(d['quotes'] for d in days),
            bank_events=sum(d['bank_events'] for d in days), map_cells=sum(d['map_cells'] for d in days),
            execution_trials=sum(d['execution_trials'] for d in days),
            frozen_scalar_checks=sum(len(d['frozen_scalar_checks']) for d in days)),
        bound_files=bound.files, verified_unique_files=len(bound.files), seconds=time.perf_counter() - started,
        interpretation='Exact saved-data/source parity and causal metadata verification on three declared days. This is not an economic or uninspected-holdout claim.',
        method='V2 full-day targets/execution/features replay; independently calculated publication/asof/admission and full raw quote support tokens; no V3 wrapper called; only COMPLETE days read.',
        unchanged_bound_sources=True, no_dataset_writes=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', output=str(output), totals=report['totals'], seconds=report['seconds'])), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/research/opportunity_pipeline_v3.json')
    p.add_argument('--date', action='append')
    p.add_argument('--output', default='reports/scientific_pipeline_v3/causal_verification.json')
    a = p.parse_args()
    run(a.config, a.date or DEFAULT_DATES, a.output)


if __name__ == '__main__':
    main()
