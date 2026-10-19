"""Frozen, paired memory interventions with the account held constant.

This audit is post-training only. It enriches saved replay diagnostics, never
changes model parameters or selects a checkpoint. Attention weights are linked
back to exact causal source events; their interpretation remains descriptive.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import Contract
from spectrum.player_v2.policy import distribution_numpy, greedy
from spectrum.player_v3.data import Day, EVENT_CLASSES
from spectrum.player_v3.policy import Standardizer, PreparedDay, make_model, encode
from spectrum.player_v3.experiment import rollout


BRANCHES = ('fast', 'previous', 'anchors', 'events')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0, mean=None, minimum=None, maximum=None, p05=None, median=None, p95=None)
    return dict(count=len(a), mean=float(a.mean()), minimum=float(a.min()), maximum=float(a.max()),
                p05=float(np.quantile(a, .05)), median=float(np.median(a)), p95=float(np.quantile(a, .95)))


def _class(day, event_index):
    names = day.meta['event_names']
    values = np.asarray(day.event_x[event_index])
    on = [name for name in EVENT_CLASSES if values[names.index(name)] > .5]
    if len(on) != 1:
        raise ValueError(f'Event class is not one-hot at {day.meta["date"]}/{event_index}')
    return on[0]


def _enrich_replay(path, day):
    """Attach exact top-1 identity without interpreting a masked branch as use."""
    frame = pd.read_parquet(path)
    n = len(frame)
    if not np.array_equal(frame.t_ns.to_numpy(np.int64), day.decision_t[:n]):
        raise ValueError('Replay decision clocks differ from prepared data: ' + str(path))
    clock_before = frame.t_ns.to_numpy(np.int64).copy()
    before_hash = sha(path)
    top_ns = np.zeros(n, np.int64)
    top_available = np.zeros(n, np.int64)
    top_weight = np.zeros(n, np.float32)
    top_class = np.full(n, '', object)
    top_index = np.full(n, -1, np.int64)
    active = np.zeros(n, bool)
    checks = Counter()
    class_counts, active_classes = Counter(), Counter()
    attention_path = path.parent / 'memory_attention.npz'
    if attention_path.exists():
        with np.load(attention_path) as z:
            indices = z['indices'].astype(np.int64)
            weights = z['event_attention']
            counts = z['counts']
            router = z['branch_weights']
            if len(indices) != len(weights) or len(indices) != len(counts):
                raise ValueError('Attention row alignment failed: ' + str(attention_path))
            if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
                raise ValueError('Invalid attention origin indices: ' + str(attention_path))
            for j, i in enumerate(indices):
                fl, fr, sl, sr, el, er = day.memory_bounds(int(i))
                now = int(day.decision_t[i])
                if not np.array_equal(counts[j, :3], [fr-fl, sr-sl, er-el]):
                    raise ValueError('Memory counts changed: ' + str(path))
                for name, times in (('fast', day.fast_available[fl:fr]),
                                    ('slow', day.slow_available[sl:sr]),
                                    ('events', day.event_available[el:er]),
                                    ('fast_endpoint', day.fast_t[fl:fr]),
                                    ('slow_endpoint', day.slow_t[sl:sr]),
                                    ('event_endpoint', day.event_t[el:er])):
                    checks[name + '_rows_checked'] += len(times)
                    checks[name + '_future_violations'] += int(np.sum(times > now))
                checks['origins_checked'] += 1
                length = er-el
                if np.any(np.abs(weights[j, length:]) > 1e-6):
                    raise ValueError('Padded event received nonzero attention: ' + str(path))
                if not length:
                    continue
                valid_weights = weights[j, :length]
                if not np.isfinite(valid_weights).all() or np.any(valid_weights < 0):
                    raise ValueError('Invalid attention weights: ' + str(path))
                if not np.isclose(valid_weights.sum(), 1., atol=2e-3):
                    raise ValueError('Event attention does not sum to one: ' + str(path))
                at = el + int(np.argmax(valid_weights))
                cls = _class(day, at)
                top_ns[i] = int(day.event_t[at])
                top_available[i] = int(day.event_available[at])
                top_weight[i] = float(np.max(valid_weights))
                top_class[i] = cls
                top_index[i] = at
                active[i] = bool(router[j, 3] > 0)
                class_counts[cls] += 1
                if active[i]:
                    active_classes[cls] += 1
                checks['top_events_checked'] += 1
                checks['top_event_future_violations'] += int(top_available[i] > now)
    violations = sum(value for key, value in checks.items() if key.endswith('future_violations'))
    if violations:
        raise ValueError(f'{violations} future memory violations at {path}')
    frame['event_top1_ns'] = top_ns
    frame['event_top1_available_ns'] = top_available
    frame['event_top1_weight'] = top_weight
    frame['event_top1_class'] = top_class
    frame['event_top1_index'] = top_index
    frame['event_top1_active_branch'] = active
    # Rollout encodes only admitted origins. Outside them, zeros in the market
    # logits produced a numerical ~.5 placeholder, not an evaluated policy.
    evaluated = frame.can_enter.to_numpy(bool)
    frame['policy_evaluated'] = evaluated
    cleared = {}
    for column in ('intent', 'p_long'):
        if column in frame:
            cleared[column] = int(frame.loc[~evaluated, column].notna().sum())
            frame.loc[~evaluated, column] = np.nan
    if not np.array_equal(clock_before, frame.t_ns.to_numpy(np.int64)):
        raise AssertionError('Diagnostic enrichment changed clocks')
    temp = path.with_name(path.stem + '.memory_audit.tmp.parquet')
    frame.to_parquet(temp, index=False)
    temp.replace(path)
    return dict(path=str(path), source_sha256_before=before_hash, enriched_sha256=sha(path),
                stable_decision_timestamps=n, checks=dict(checks), top_event_classes=dict(class_counts),
                top_event_classes_active_branch=dict(active_classes),
                unevaluated_policy_placeholders_cleared=cleared, policy_evaluated_origins=int(evaluated.sum()),
                attention_sha256=sha(attention_path) if attention_path.exists() else None)


def _probabilities(model, prepared, indices, accounts, masks, device, ablation=None):
    with torch.no_grad(), torch.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.float16):
        h = encode(model, prepared, indices, ablation=ablation)
        base = model.head(h).float().cpu().numpy()
        aw = model.account_head.weight.float().cpu().numpy()
    rows = []
    for b, account, mask in zip(base, accounts, masks):
        joint, intent, gate, direction, plans = distribution_numpy(b + aw @ account, mask)
        rows.append(dict(intent=float(intent), gate=float(gate), p_long=float(direction[0]),
                         action=greedy(gate, direction, plans), joint=joint))
    return rows


def _aggregate_probes(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    result = []
    for (variant, seed, branch), group in frame.groupby(['variant', 'seed', 'ablation'], sort=True):
        result.append(dict(variant=variant, seed=int(seed), ablation=branch, probes=len(group),
            mean_intent_delta=float(group.intent_delta.mean()),
            mean_abs_intent_delta=float(group.intent_delta.abs().mean()),
            max_abs_intent_delta=float(group.intent_delta.abs().max()),
            p95_abs_intent_delta=float(group.intent_delta.abs().quantile(.95)),
            mean_abs_p_long_delta=float(group.p_long_delta.abs().mean()),
            max_abs_p_long_delta=float(group.p_long_delta.abs().max()),
            action_change_fraction=float(group.action_changed.mean()),
            participation_change_fraction=float(group.participation_changed.mean()),
            mean_joint_total_variation=float(group.joint_total_variation.mean()),
            max_joint_total_variation=float(group.joint_total_variation.max())))
    return result


def _associations(replay_root, prepared_by_date):
    """Rank associations at legal origins; no target/outcome is involved."""
    pairs, weights = {}, {name: [] for name in ('fast', 'slow', 'anchors', 'events')}
    samples = []
    for path in sorted(replay_root.glob('*/decisions.parquet')):
        day = prepared_by_date[path.parent.name].day
        frame = pd.read_parquet(path)
        legal = frame.legal_entry.to_numpy(bool) if 'legal_entry' in frame else frame.can_enter.to_numpy(bool)
        ix = np.flatnonzero(legal)
        gate = frame.gate.to_numpy(float)[ix]
        for name in weights:
            column = 'branch_' + name
            if column in frame:
                weights[name].extend(frame[column].to_numpy(float)[ix].tolist())
        for name in ('from_current_open_ticks', 'from_previous_open_ticks', 'from_london_open_ticks', 'from_previous_close_ticks'):
            j = day.meta['anchor_names'].index(name)
            k = day.meta['anchor_names'].index(name + '__known')
            known = np.asarray(day.anchors[ix, k]) > 0
            pairs.setdefault('abs_' + name, []).extend(zip(gate[known], np.abs(day.anchors[ix[known], j])))
        pairs.setdefault('event_count', []).extend(zip(gate, frame.event_count.to_numpy(float)[ix]))
        name = 'vwap_proxy_900s_return_rms_1s'
        j = day.meta['fast_names'].index(name)
        k = day.meta['fast_names'].index(name + '__known')
        for at, g in zip(ix, gate):
            fl, fr, *_ = day.memory_bounds(int(at))
            if fr > fl and day.fast_x[fr-1, k] > 0:
                pairs.setdefault('fast_900s_return_rms_1s', []).append((g, float(day.fast_x[fr-1, j])))
        samples.append(dict(date=day.meta['date'], legal_origins=len(ix)))
    assoc = []
    for name, values in pairs.items():
        a = np.asarray(values, dtype=float).reshape(-1, 2)
        a = a[np.isfinite(a).all(axis=1)]
        corr = None
        if len(a) >= 3 and len(np.unique(a[:, 0])) > 1 and len(np.unique(a[:, 1])) > 1:
            corr = float(pd.Series(a[:, 0]).rank().corr(pd.Series(a[:, 1]).rank()))
        assoc.append(dict(feature=name, samples=len(a), spearman_gate=corr,
                          constant_gate=len(np.unique(a[:, 0])) <= 1 if len(a) else True,
                          constant_feature=len(np.unique(a[:, 1])) <= 1 if len(a) else True))
    return dict(legal_origin_days=samples, branch_weights={k: _summary(v) for k, v in weights.items()},
                descriptive_gate_associations=assoc)


def audit_memory(data_root, report_root, max_origins_per_day=16, device=None):
    """Audit completed frozen results and return/write memory_usage.json.

    Execute only after the training process has released the GPU. Paired probes
    hold the full-policy account and action mask fixed. ``previous`` removes
    previous-day M1 only; previous-day anchors/events remain available.
    """
    started = time.perf_counter()
    data_root, root = Path(data_root).resolve(), Path(report_root).resolve()
    summary = _load(root / 'summary.json')
    if summary.get('status') != 'COMPLETE':
        raise RuntimeError('Training must be COMPLETE before memory audit')
    contract = _load(root / 'contract.json')
    from spectrum.player_v3.gym import risk_contract
    if contract.get('risk_contract') != risk_contract():
        raise ValueError('Risk contract mismatch; use the preserved historical source for old fixed-floor runs')
    for name, digest in contract['source_hashes'].items():
        if sha(name) != digest:
            raise ValueError('Frozen implementation changed before memory audit: ' + name)
    if sha(data_root / 'manifest.json') != contract['data_manifest_sha256']:
        raise ValueError('Prepared dataset manifest changed')
    if sha(root / 'freeze.json') != summary['freeze_sha256']:
        raise ValueError('Frozen selection receipt changed')
    torch.set_num_threads(4)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.76)
    valid_fields = {field.name for field in fields(Contract)}
    c = Contract(**{k: v for k, v in contract['contract'].items() if k in valid_fields})
    scaler = Standardizer.load(root / 'normalization.json')
    dates = contract['split']['evaluation']
    days = {date: Day(data_root / date) for date in dates}
    prepared = {date: PreparedDay(day, scaler) for date, day in days.items()}
    enrichment = []
    for path in sorted((root / 'replays').rglob('decisions.parquet')):
        date = path.parent.name
        if date not in days:
            declared_dates = set(sum(contract['split'].values(), []))
            if date not in declared_dates:
                raise ValueError('Unexpected replay date outside declared splits: ' + date)
            # Additional frozen stochastic/epoch exports may include validation
            # days. Their diagnostics are enriched, never used to select probes.
            days[date] = Day(data_root / date)
        lineage = _load(path.parent / 'lineage.json')
        if lineage['day_manifest_sha256'] != sha(days[date].path / 'manifest.json'):
            raise ValueError('Replay lineage changed: ' + str(path))
        enrichment.append(_enrich_replay(path, days[date]))
    all_probes, models = [], []
    for item in summary['models']:
        variant, seed = item['variant'], int(item['seed'])
        checkpoint = Path(item['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if sha(checkpoint) != item['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed: ' + str(checkpoint))
        model = make_model(variant).to(device)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        model.eval()
        # The ordinary full-policy replay reconstructs the exact clean account
        # prefix. It does not write files, tune parameters, or inspect outcomes
        # to choose probe origins.
        batches, replay_metric = rollout(model, [prepared[date] for date in dates], c)
        reproduction = []
        origin_count = 0
        day_samples = []
        for batch in batches:
            p = batch['prepared']
            if not len(batch['mask']):
                continue
            legal = np.flatnonzero(batch['mask'][:, 1:].any(axis=1))
            if not len(legal):
                continue
            pick = np.unique(np.linspace(0, len(legal)-1, min(max_origins_per_day, len(legal)), dtype=int))
            indices = legal[pick]
            accounts, masks = batch['account'][indices], batch['mask'][indices]
            full = _probabilities(model, p, indices, accounts, masks, device)
            origin_count += len(indices)
            day_samples.append(dict(date=p.day.meta['date'], legal_clean_origins=len(legal), probes=len(indices),
                                    decision_ns=[str(int(p.day.decision_t[i])) for i in indices]))
            saved_path = root / 'replays' / f'{variant}_s{seed}' / p.day.meta['date'] / 'decisions.parquet'
            saved = pd.read_parquet(saved_path)
            for i, ref in zip(indices, full):
                reproduction.append(abs(ref['intent'] - float(saved.intent.iloc[i])))
            if variant == 'baseline':
                continue
            for branch in BRANCHES:
                altered = _probabilities(model, p, indices, accounts, masks, device, ablation=branch)
                for i, ref, alt in zip(indices, full, altered):
                    all_probes.append(dict(variant=variant, seed=seed, date=p.day.meta['date'], decision_index=int(i),
                        t_ns=str(int(p.day.decision_t[i])), ablation='previous_M1' if branch == 'previous' else branch,
                        full_intent=ref['intent'], ablated_intent=alt['intent'], intent_delta=alt['intent']-ref['intent'],
                        full_p_long=ref['p_long'], ablated_p_long=alt['p_long'], p_long_delta=alt['p_long']-ref['p_long'],
                        full_action=ref['action'], ablated_action=alt['action'], action_changed=ref['action'] != alt['action'],
                        participation_changed=(ref['action'] != 0) != (alt['action'] != 0),
                        joint_total_variation=float(np.abs(ref['joint']-alt['joint']).sum()/2),
                        account_held_fixed=True, mask_held_fixed=True))
        observations = _associations(root / 'replays' / f'{variant}_s{seed}', prepared)
        models.append(dict(variant=variant, seed=seed, probes=origin_count, probe_days=day_samples,
            reproduced_clean_prefix_steps=replay_metric['learning_prefix_steps'],
            saved_vs_paired_full_intent_abs_error=_summary(reproduction), **observations))
        if sha(checkpoint) != item['checkpoint_sha256']:
            raise AssertionError('Audit modified checkpoint')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    probe_path = root / 'memory_probes.parquet'
    pd.DataFrame(all_probes).to_parquet(probe_path, index=False)
    violations = sum(sum(v for k, v in item['checks'].items() if k.endswith('future_violations')) for item in enrichment)
    result = dict(status='COMPLETE', elapsed_seconds=time.perf_counter()-started,
        protocol=dict(selection='up to 16 evenly spaced executable origins in each clean full-policy day prefix; no outcome-based sampling',
            maximum_origins_per_day=max_origins_per_day, account='same full-policy account and action mask in each paired intervention',
            previous_M1='only previous-session M1 recurrent prefix removed; prior-session anchors and events retained',
            ablation='inference-only frozen branch intervention, no retraining or checkpoint selection',
            interpretation='Routing/attention weights and gate correlations are descriptive. Paired deltas measure functional sensitivity, not economic value or market causality.',
            full_replay_comparison='The separately saved ablation equity paths include changed account histories; these paired probes intentionally do not.',
            clocks='exact int64 source clocks; JSON/parquet probe clocks are decimal strings; no event outcome columns are loaded'),
        causal_stable_timestamps_count=sum(r['stable_decision_timestamps'] for r in enrichment),
        causal_future_violations=violations, models=models, paired_ablations=_aggregate_probes(all_probes),
        replay_enrichment=enrichment, probe_rows=len(all_probes), probe_sha256=sha(probe_path),
        contract_sha256=sha(root / 'contract.json'), freeze_sha256=sha(root / 'freeze.json'),
        normalization_sha256=sha(root / 'normalization.json'), audit_source_sha256=sha(__file__))
    save_json(root / 'memory_usage.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device')
    parser.add_argument('--max-origins', type=int, default=16)
    args = parser.parse_args()
    result = audit_memory(args.data, args.report, args.max_origins, args.device)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'probe_rows', 'causal_future_violations')}, indent=2))
