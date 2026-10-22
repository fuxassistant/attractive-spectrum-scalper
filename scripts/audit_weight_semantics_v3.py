"""TRAIN-only observational audit of what inverse crossing-ID weights emphasize.

Future fields are teacher diagnostics only. No prediction, checkpoint, score,
validation/calibration/test root payload, or economic outcome is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.verify_opportunity_v3 import BoundInputs, exact_array, read_json, sha

STATE = {0: 'NONE', 1: 'LONG_ONLY', 2: 'SHORT_ONLY', 3: 'BOTH'}
NS = 1_000_000_000


def stats(values, weights=None):
    x = np.asarray(values, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[good], w[good]
    if not len(x):
        return dict(count=0, mean=None, q10=None, median=None, q90=None)
    order = np.argsort(x, kind='stable')
    ordered, cumulative = x[order], np.cumsum(w[order])
    quantiles = ordered[np.minimum(np.searchsorted(cumulative, np.array([.1, .5, .9]) * cumulative[-1]), len(x)-1)]
    return dict(count=len(x), mean=float(np.average(x, weights=w)), q10=float(quantiles[0]),
                median=float(quantiles[1]), q90=float(quantiles[2]), minimum=float(x.min()), maximum=float(x.max()))


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return None
    ra, rb = pd.Series(a[good]).rank().to_numpy(), pd.Series(b[good]).rank().to_numpy()
    return None if np.std(ra) == 0 or np.std(rb) == 0 else float(np.corrcoef(ra, rb)[0, 1])


def summarize_roots(frame, columns):
    return {c: stats(frame[c]) for c in columns}


def audit(run_dir, output):
    run_dir = Path(run_dir)
    bound = BoundInputs()
    bound.bind(__file__)
    bound.bind('scripts/verify_opportunity_v3.py')
    receipt_path = run_dir / 'receipt.json'
    receipt = read_json(receipt_path)
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('A COMPLETE frozen run is required')
    bound.bind(receipt_path)
    for name in ('config.json', 'weighting.json', 'train_weights.parquet'):
        bound.bind(run_dir/name, receipt['outputs'][name])
    for path, digest in receipt['sources'].items():
        bound.bind(path, digest)
    config = read_json(run_dir/'config.json')
    weighting = read_json(run_dir/'weighting.json')
    if config['training']['weighting'] != 'crossing_id' or weighting['scheme'] != 'crossing_id':
        raise ValueError('This audit requires the frozen crossing-ID weighting arm')
    dataset = Path(config['data']['root'])
    manifest_path = dataset/'manifest.json'
    bound.bind(manifest_path, receipt['dataset_manifest_sha256'])
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('The dataset must be COMPLETE')
    for path, digest in manifest['identity']['sources'].items():
        bound.bind(path, digest)
    train_dates = config['splits']['train']
    frame_parts, offset = [], 0
    for daily in manifest['days']:
        if daily['date'] in train_dates:
            path = dataset/daily['roots_file']
            bound.bind(path, daily['files'][daily['roots_file']])
            frame = pd.read_parquet(path)
            if len(frame) != daily['roots'] or not frame.date.eq(daily['date']).all():
                raise ValueError('Daily TRAIN root identity changed')
            frame['root_index'] = offset + np.arange(len(frame), dtype=np.int64)
            frame_parts.append(frame)
        offset += daily['roots']
    roots = pd.concat(frame_parts, ignore_index=True)
    roots = roots.loc[roots.feature_available & roots.y_crossing.ge(0)].reset_index(drop=True)
    saved = pd.read_parquet(run_dir/'train_weights.parquet')
    for column in ('root_index', 'date', 't_ns', 'y_crossing', 'first_qualifying_event_id_L', 'first_qualifying_event_id_S'):
        exact_array(saved[column].to_numpy(), roots[column].to_numpy(), 'weight cohort '+column)
    roots['weight'] = saved.weight.to_numpy()
    roots['root_row'] = np.arange(len(roots))
    memberships = []
    for side, label in (('L', 1), ('S', 2)):
        f = roots.loc[roots.y_crossing.isin([label, 3])].copy()
        f['side'] = side
        f['event_id'] = f['first_qualifying_event_id_'+side]
        f['cross_ns'] = f['first_qualifying_ns_'+side]
        f['delay_s'] = (f.cross_ns-f.entry_ns)/NS
        f['gain_margin'] = f['gain_margin_'+side]
        f['gain_margin_R'] = f['gain_margin_R_'+side]
        f['max_cross_net_R'] = f['max_cross_net_R_'+side]
        if f.event_id.eq('').any() or (f.delay_s <= 0).any() or (f.delay_s > 300).any():
            raise ValueError('Invalid positive witness identity/clock')
        memberships.append(f[['root_row', 'date', 't_ns', 'entry_ns', 'y_crossing', 'weight', 'side',
            'event_id', 'cross_ns', 'delay_s', 'gain_margin', 'gain_margin_R', 'max_cross_net_R']])
    members = pd.concat(memberships, ignore_index=True)
    groups = members.groupby('event_id', sort=True)
    events = groups.agg(multiplicity=('root_row', 'size'), unique_roots=('root_row', 'nunique'),
        first_origin_ns=('t_ns', 'min'), last_origin_ns=('t_ns', 'max'), cross_times=('cross_ns', 'nunique'),
        dates=('date', 'nunique'), sides=('side', 'nunique'))
    if not events.cross_times.eq(1).all() or not events.dates.eq(1).all():
        raise ValueError('A physical witness ID changed date or crossing timestamp')
    if not events.multiplicity.eq(events.unique_roots).all():
        raise ValueError('Unexpected duplicate membership for one root and physical ID')
    cadence = float(config['data']['cadence_seconds'])
    events['origin_span_s'] = (events.last_origin_ns-events.first_origin_ns)/NS
    events['count_times_cadence_s'] = events.multiplicity * cadence
    events['span_grid_slots'] = events.origin_span_s/cadence+1
    events['admitted_membership_density'] = events.multiplicity/events.span_grid_slots
    counts = events.multiplicity.to_dict()
    count_sha = hashlib.sha256(json.dumps(sorted(counts.items()), separators=(',', ':')).encode()).hexdigest()
    if count_sha != weighting['crossing_id_count_table_sha256']:
        raise ValueError('Reconstructed crossing-ID multiplicities differ')
    for column in ('multiplicity', 'origin_span_s', 'count_times_cadence_s', 'admitted_membership_density'):
        members[column] = members.event_id.map(events[column])
    members['inverse'] = 1./members.multiplicity
    inverse = members.groupby('root_row').inverse.mean()
    raw = np.ones(len(roots), float)
    raw[inverse.index.to_numpy()] = inverse.to_numpy()
    expected = raw.copy()
    for label in range(4):
        mask = roots.y_crossing.eq(label).to_numpy()
        expected[mask] *= mask.sum()/raw[mask].sum()
    exact_array(roots.weight.to_numpy(), expected, 'saved normalized inverse-ID weight')
    roots['effective_multiplicity'] = 1./raw
    roots['G_over_R'] = roots.G_event/roots.R_event
    root_metrics = ('multiplicity', 'origin_span_s', 'count_times_cadence_s', 'admitted_membership_density',
                    'delay_s', 'gain_margin', 'gain_margin_R', 'max_cross_net_R')
    averaged = members.groupby('root_row')[list(root_metrics)].mean()
    for column in root_metrics:
        roots[column] = roots.root_row.map(averaged[column])
    roots['minimum_side_multiplicity'] = roots.root_row.map(members.groupby('root_row').multiplicity.min())
    roots['maximum_side_multiplicity'] = roots.root_row.map(members.groupby('root_row').multiplicity.max())
    columns = ('weight', 'effective_multiplicity', 'multiplicity', 'minimum_side_multiplicity',
        'maximum_side_multiplicity', 'origin_span_s', 'count_times_cadence_s', 'admitted_membership_density',
        'delay_s', 'G_event', 'R_event', 'G_over_R', 'gain_margin', 'gain_margin_R', 'max_cross_net_R')
    classes = {}
    for label in (1, 2, 3):
        f = roots.loc[roots.y_crossing.eq(label)]
        quantiles = np.quantile(f.weight, [0, .25, .5, .75, .9, .99, 1.])
        bins, previous = [], -np.inf
        for hi in np.unique(quantiles):
            selected = f.loc[(f.weight > previous) & (f.weight <= hi)]
            if len(selected):
                bins.append(dict(lower_exclusive=None if not np.isfinite(previous) else float(previous),
                    upper_inclusive=float(hi), roots=len(selected), root_fraction=len(selected)/len(f),
                    optimization_mass_fraction=float(selected.weight.sum()/f.weight.sum()),
                    metrics=summarize_roots(selected, columns)))
            previous = hi
        q10, q90 = np.quantile(f.weight, [.1, .9])
        cuts = {}
        conditions = dict(high_weight_ge_q90=f.weight >= q90, low_weight_le_q10=f.weight <= q10,
            side_mean_multiplicity_le3=f.multiplicity <= 3, side_mean_origin_span_le10s=f.origin_span_s <= 10,
            side_mean_first_cross_delay_le30s=f.delay_s <= 30)
        for name, condition in conditions.items():
            selected = f.loc[condition]
            cuts[name] = dict(roots=len(selected), root_fraction=len(selected)/len(f),
                optimization_mass_fraction=float(selected.weight.sum()/f.weight.sum()),
                metrics=summarize_roots(selected, columns))
        classes[STATE[label]] = dict(roots=len(f), optimization_mass=float(f.weight.sum()),
            kish_effective_sample_size=float(f.weight.sum()**2/np.square(f.weight).sum()),
            uniform_root={c: stats(f[c]) for c in columns if c != 'weight'},
            weighted_optimization={c: stats(f[c], f.weight) for c in columns if c != 'weight'},
            weight_quantile_boundaries=quantiles.tolist(), quantile_bins=bins, tail_and_duration_cuts=cuts,
            spearman_weight_vs={c: correlation(f.weight, f[c]) for c in root_metrics},
            positive_side_memberships=int(members.y_crossing.eq(label).sum()))
    # Bookkeeping allocation only: BOTH's one joint-root loss is not split in
    # training. This decomposition exposes how the raw mean-inverse is formed.
    denominator = members.root_row.map(members.groupby('root_row').inverse.sum())
    members['allocated_mass'] = members.weight*members.inverse/denominator
    event_mass = members.groupby('event_id').allocated_mass.sum()
    classes['NONE'] = dict(roots=int(roots.y_crossing.eq(0).sum()), weight_is_one=bool(roots.loc[roots.y_crossing.eq(0), 'weight'].eq(1).all()))
    id_distribution = dict(unique_ids=len(events), positive_side_memberships=len(members),
        ids_with_long_and_short_memberships=int(events.sides.eq(2).sum()),
        multiplicity=stats(events.multiplicity), origin_span_s=stats(events.origin_span_s),
        admitted_membership_density=stats(events.admitted_membership_density),
        noncontiguous_id_membership_fraction=float(events.admitted_membership_density.lt(1).mean()),
        inverse_decomposed_optimization_mass_per_id=stats(event_mass),
        allocation_caveat='For reporting, allocate a root weight to its sides proportional to inverse multiplicities. Actual training weights the single joint-root loss; this is not a separate side-head gradient allocation.')
    observations = {}
    for name, values in classes.items():
        if name == 'NONE':
            continue
        high = values['tail_and_duration_cuts']['high_weight_ge_q90']
        observations[name] = dict(
            uniform_to_weighted_medians={c: [values['uniform_root'][c]['median'],
                values['weighted_optimization'][c]['median']]
                for c in ('multiplicity', 'origin_span_s', 'delay_s', 'gain_margin_R', 'G_event')},
            high_weight_tail_root_fraction=high['root_fraction'],
            high_weight_tail_optimization_mass_fraction=high['optimization_mass_fraction'],
            weight_span_rank_correlation=values['spearman_weight_vs']['origin_span_s'],
            weight_delay_rank_correlation=values['spearman_weight_vs']['delay_s'])
    bound.confirm_unchanged()
    report = dict(status='PASS', version='SCIENTIFIC_V3_TRAIN_WEIGHT_SEMANTICS_V1',
        run_id=receipt['run_id'], train_dates=train_dates, train_roots=len(roots),
        dataset_manifest_sha256=receipt['dataset_manifest_sha256'], weight_file_sha256=sha(run_dir/'train_weights.parquet'),
        reconstructed_weights_exact=True, count_table_sha256=count_sha, classes=classes,
        observed_comparison=observations,
        id_distribution=id_distribution, verified_files=bound.files,
        field_semantics=dict(multiplicity='Count of admitted mature TRAIN (root,positive side) memberships for the same immutable first-qualifying physical crossing ID; pooled across classes and sides. The same physical crossing can qualify LONG from one earlier entry and SHORT from another; its shared count intentionally includes both.',
            origin_span_s='Last minus first admitted TRAIN origin for that ID. It is the observed preimage span, not a trade holding time, full physical opportunity duration, or count*5 seconds.',
            count_times_cadence_s='Multiplicity * 5s is grid exposure proxy. Even a contiguous group has span=(count-1)*5s; missing/interleaved memberships may make span larger.',
            admitted_membership_density='Count / (span/5s + 1): occupancy of the regular-grid envelope between first/last admitted memberships; not a probability of opportunity.',
            delay_s='First qualifying crossing timestamp minus entry timestamp for this root/side; different roots sharing the same crossing ID generally have different delays.',
            gain_margin='Maximum net mark across observed first crossings of all nine BB lines within 300s, minus fixed G_event. It is NOT the margin at the first qualifying crossing, full-path MFE, or executed profit.',
            gain_margin_R='gain_margin / R_event. Includes the target reference fee .30 and entry BBO spread through the net mark.',
            G_over_R='G_event/R_event is fixed by design at 2, independently of the execution stop-risk plan.',
            both='BOTH root metrics average the two positive side memberships unless named minimum/maximum/effective. Effective multiplicity is harmonic mean(mL,mS), because raw weight is mean(1/mL,1/mS).',
            quantiles='Boundaries are class-specific weight quantiles. Equal weights are kept together, so bin sizes and >=q90 tail proportions may differ from nominal fractions. Stats use weighted empirical step quantiles.'),
        interpretation=dict(formula='NONE=1; L/S inverse ID multiplicity; BOTH average inverses; normalize each joint class to preserve its original total mass. The weight multiplies the entire root BCE plus conditional direction CE.',
            estimand_change='Within each positive class the optimization measure changes from uniform decision origins toward lower-multiplicity witness IDs. It does not simply delete identical examples or guarantee independent/equal event weights.',
            observational_only='Relations between weight, observed origin span, delay and margin are descriptive TRAIN associations. No causal explanation, score advantage, profitability, or model-quality conclusion follows.',
            short_and_rare='Use low-multiplicity or short observed admitted-preimage span. This audit cannot identify physically rare regimes or prove short trade duration from group count alone.'),
        no_test_scores_read=True, no_nontrain_root_payloads_read=True, no_model_or_dataset_changes=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', train_roots=len(roots), positive_ids=len(events), output=str(output), sha256=sha(output))), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir', default='data/derived/scientific_pipeline_v3/runs/tcn_micro_legacy_crossing_id_seed17')
    p.add_argument('--output', default='reports/scientific_pipeline_v3/weight_semantics.json')
    a = p.parse_args()
    audit(a.run_dir, a.output)


if __name__ == '__main__':
    main()
