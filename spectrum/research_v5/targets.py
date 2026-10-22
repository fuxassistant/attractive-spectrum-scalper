"""Conditional filter targets without changing the frozen crossing teacher.

E = event exists; C = frozen-base chosen side is correct conditional on E;
S_k = the first qualifying crossing is reached before adverse net -R_k,
conditional on E=C=1. Actual exit fills and post-crossing latency excursions
do not define S_k. Every future field in this module is teacher/audit metadata.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

NS = 1_000_000_000
PLANS = ('cost2', 'cost4', 'cost6')
LABEL_COLUMNS = ('y_event', 'y_side_correct', *(f'y_risk_safe_{p}' for p in PLANS))
MASK_COLUMNS = ('event_known', 'side_known', *(f'risk_known_{p}' for p in PLANS))


def _settings(config):
    query, execution = config['event_query'], config['execution']
    if query['family'] != 'BB' or list(query['scales']) != [1, 3, 6]:
        raise ValueError('Frozen nine-line BB1/3/6 teacher required')
    if [float(v) for v in config['risk']['cost_multipliers']] != [2., 4., 6.]:
        raise ValueError('Frozen cost2/4/6 risk plans required')
    if float(query['reference_fee']) != float(execution['fee_roundtrip']):
        raise ValueError('Teacher and execution fees must agree for prefix-risk equivalence')
    duration = float(query['deadline_seconds'])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Finite positive horizon required')
    return int(round(duration*NS)), float(query['reference_fee'])


def build_filter_targets(frame, line_map, sides, config):
    """Return aligned frame, float32[N,5] labels, bool masks and a receipt.

    `sides` is the frozen base model's causal choice (+1 LONG, -1 SHORT),
    supplied for every clock root. Feature admission is not folded into target
    masks: the caller additionally applies the unchanged causal X admission.
    Unknown conditional targets are -1, never converted into negatives.
    """
    duration, fee = _settings(config)
    required = {'date', 't_ns', 'entry_ordinal', 'entry_ns', 'R_event', 'G_event',
        'y_crossing', 'available_ns', *(f'R_{p}' for p in PLANS)}
    required |= {f'first_qualifying_{what}_{side}' for what in ('event_id', 'ordinal', 'ns') for side in ('L', 'S')}
    if not required <= set(frame) or not len(frame):
        raise ValueError('Complete frozen root/teacher fields required: '+str(sorted(required-set(frame))))
    t = frame.t_ns.to_numpy(np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Unique chronological roots required')
    sides = np.asarray(sides)
    if sides.shape != t.shape or not np.issubdtype(sides.dtype, np.integer) or not np.isin(sides, [1, -1]).all():
        raise ValueError('One integer frozen-base side +/-1 per root required')
    y = frame.y_crossing.to_numpy(np.int8)
    if not np.isin(y, [-1, 0, 1, 2, 3]).all():
        raise ValueError('Frozen crossing teacher state must be -1..3')
    need_map = {'date', 't_ns', 'scale', 'role', 'ready', 'entry_ordinal', 'entry_ns',
        'cross_ordinal', 'cross_ns', 'event_id', 'mark_L', 'mark_S', 'mae_L', 'mae_S', 'G_event', 'R_event'}
    if not need_map <= set(line_map):
        raise ValueError('Missing factual line-map fields: '+str(sorted(need_map-set(line_map))))
    mapped = line_map.sort_values(['t_ns', 'scale', 'role']).reset_index(drop=True)
    n, width = len(frame), 9
    if (len(mapped) != n*width or not np.array_equal(mapped.t_ns, np.repeat(t, width))
            or not np.array_equal(mapped.date.astype(str), np.repeat(frame.date.astype(str), width))
            or not np.array_equal(mapped.scale, np.tile(np.repeat([1, 3, 6], 3), n))
            or not np.array_equal(mapped.role, np.tile([0, 1, 2], 3*n))):
        raise ValueError('Exactly nine unique aligned scale/line cells per root required')
    def matrix(name):
        return mapped[name].to_numpy().reshape(n, width)
    entry = frame.entry_ordinal.to_numpy(np.int64)
    entry_ns = frame.entry_ns.to_numpy(np.int64)
    for name in ('entry_ordinal', 'entry_ns'):
        if not np.array_equal(matrix(name), np.repeat(frame[name].to_numpy()[:, None], width, axis=1)):
            raise ValueError('Map/root entry identity differs')
    for name in ('R_event', 'G_event'):
        if not np.allclose(matrix(name), frame[name].to_numpy()[:, None], rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError('Map/root fixed event reference differs')
    cross, cross_ns = matrix('cross_ordinal'), matrix('cross_ns')
    if np.any((cross >= 0) & ((cross <= entry[:, None]) | (cross_ns < entry_ns[:, None]))):
        raise ValueError('A resolved crossing must be strictly after entry by publication ordinal')
    due = entry_ns+duration
    mark = np.where(sides[:, None] == 1, matrix('mark_L'), matrix('mark_S'))
    mae = np.where(sides[:, None] == 1, matrix('mae_L'), matrix('mae_S'))
    qualifying = (matrix('ready').astype(bool) & (cross >= 0) & (cross_ns <= due[:, None])
                  & np.isfinite(mark) & (mark >= frame.G_event.to_numpy()[:, None]))
    choices = np.where(qualifying, cross, np.iinfo(np.int64).max)
    column = choices.argmin(axis=1)
    found = qualifying.any(axis=1)
    row = np.arange(n)
    witness_ordinal = np.where(found, cross[row, column], -1).astype(np.int64)
    witness_ns = np.where(found, cross_ns[row, column], 0).astype(np.int64)
    witness_id = np.where(found, matrix('event_id')[row, column], '')
    witness_mark = np.where(found, mark[row, column], np.nan)
    prefix_mae = np.where(found, mae[row, column], np.nan)
    event_known = y >= 0
    event = np.where(event_known, y > 0, -1).astype(np.int8)
    side_known = event_known & (y > 0)
    bit = np.where(sides == 1, 1, 2)
    correct = (y & bit) != 0
    side_correct = np.where(side_known, correct, -1).astype(np.int8)
    positive_side = event_known & correct
    if not np.array_equal(found[event_known], positive_side[event_known]):
        raise ValueError('Chosen-side positive teacher disagrees with first-line map')
    for suffix, derived in (('ordinal', witness_ordinal), ('ns', witness_ns), ('event_id', witness_id)):
        expected = np.where(sides == 1, frame[f'first_qualifying_{suffix}_L'], frame[f'first_qualifying_{suffix}_S'])
        if not np.array_equal(np.asarray(expected)[positive_side], derived[positive_side]):
            raise ValueError('First qualifying witness differs from the frozen root field: '+suffix)
    if np.any(positive_side & ((entry < 0) | (entry_ns <= t) | ~np.isfinite(frame.R_event) | (frame.R_event <= 0)
            | ~np.isfinite(frame.G_event) | (frame.G_event <= 0) | ~np.isfinite(prefix_mae)
            | (prefix_mae >= 0) | (prefix_mae > witness_mark))):
        raise ValueError('Invalid witnessed positive path or entry-cost prefix')
    if any(not isinstance(v, str) or not v for v in witness_id[positive_side]):
        raise ValueError('A known positive side requires an immutable witness ID')
    at_deadline = positive_side & (witness_ns == due)
    risk_known = positive_side & (witness_ns < due)
    risks = frame[[f'R_{p}' for p in PLANS]].to_numpy(float)
    if (not np.isfinite(risks[risk_known]).all() or (risks[risk_known] <= 0).any()
            or (np.diff(risks[risk_known], axis=1) < 0).any()):
        raise ValueError('Finite positive nondecreasing causal risk budgets required')
    original_available = frame.available_ns.to_numpy(np.int64)
    if np.any(event_known & (original_available < t)):
        raise ValueError('Known original label availability cannot precede origin')
    out = pd.DataFrame(dict(date=frame.date.astype(str).to_numpy(), t_ns=t, chosen_side=sides.astype(np.int8),
        y_crossing=y, y_event=event, event_known=event_known, y_side_correct=side_correct, side_known=side_known,
        entry_ordinal=entry, entry_ns=entry_ns, deadline_ns=due,
        witness_event_id=witness_id, witness_ordinal=witness_ordinal, witness_ns=witness_ns,
        witness_delay_seconds=np.where(found, (witness_ns-entry_ns)/NS, np.nan),
        witness_mark_net=witness_mark, witness_prefix_mae_net=prefix_mae,
        R_event=frame.R_event.to_numpy(), G_event=frame.G_event.to_numpy(),
        event_available_ns=np.where(event_known, original_available, 0),
        side_available_ns=np.where(side_known, original_available, 0),
        risk_path_available_ns=np.where(found, witness_ns, 0),
        risk_available_ns=np.where(risk_known, np.maximum(original_available, witness_ns), 0),
        risk_context_reason=np.select([~event_known, y == 0, side_known & ~correct, at_deadline],
            ['UNKNOWN_TEACHER', 'NO_EVENT', 'WRONG_SIDE', 'AT_DEADLINE_TIME_WINS'], default='QUALIFYING_BEFORE_DEADLINE')))
    for i, plan in enumerate(PLANS):
        safe = prefix_mae > -risks[:, i]
        out[f'R_{plan}'] = risks[:, i]
        out[f'risk_known_{plan}'] = risk_known
        out[f'y_risk_safe_{plan}'] = np.where(risk_known, safe, -1).astype(np.int8)
    labels = out[list(LABEL_COLUMNS)].to_numpy(np.float32)
    masks = out[list(MASK_COLUMNS)].to_numpy(bool)
    if not np.array_equal(masks, labels >= 0):
        raise AssertionError('Conditional label/mask disagreement')
    if (np.diff(labels[risk_known, 2:], axis=1) < 0).any():
        raise AssertionError('Wider stop cannot become less safe on the same pre-cross path')
    receipt = dict(version='CONDITIONAL_EVENT_SIDE_PRE_CROSS_RISK_V5', roots=n,
        label_columns=list(LABEL_COLUMNS), mask_columns=list(MASK_COLUMNS),
        known_per_head=masks.sum(axis=0).tolist(), positives_per_head=(labels == 1).sum(axis=0).tolist(),
        exact_deadline_positive_witnesses_masked=int(at_deadline.sum()),
        conditional_masks='E known iff original Y known; C known iff E=1; risk known iff E=C=1 and qualifying crossing strictly before deadline. Caller also applies causal feature_available.',
        risk_definition='safe iff min net over entry through first qualifying crossing is strictly greater than -R_trade; equality is unsafe.',
        strict_ordinal_equivalence='Map MAE includes the witness quote. Its net is >=G_event>0, while the entry liquidation mark is negative. The witness cannot create the minimum; inclusive map MAE equals strict ordinal prefix [entry,witness). Same-ns earlier ordinals remain in prefix.',
        deadline='Original crossing teacher allows <=entry+H; execution TIME wins at equality. Those exact-deadline witnesses are excluded only from auxiliary risk masks, not relabelled in E/C.',
        publication='risk_path_available_ns is witness time; conditional risk_available_ns also waits for original common teacher maturity.',
        original_crossing_target_unchanged=True, execution_status_not_used=True, post_cross_latency_not_used=True,
        reference_fee=fee, source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return dict(frame=out, labels=labels, masks=masks, receipt=receipt)


def verify_risk_targets_raw(q, targets, config, positions=None, max_probes=12):
    """Independent full-BBO probes of strict-prefix MAE and adverse first hit."""
    duration, fee = _settings(config)
    frame = targets['frame'] if isinstance(targets, dict) else targets
    eligible = np.flatnonzero(frame.risk_known_cost2.to_numpy(bool))
    if positions is None:
        positions = eligible[np.linspace(0, len(eligible)-1, min(max_probes, len(eligible)), dtype=int)] if len(eligible) else []
    checks = []
    gap = int(round(config['execution'].get('max_quote_gap_seconds', 60.)*NS))
    for i in np.asarray(positions, dtype=np.int64):
        r = frame.iloc[int(i)]
        if not r.risk_known_cost2:
            raise ValueError('Raw risk probes require known conditional risk')
        e, x = int(r.entry_ordinal), int(r.witness_ordinal)
        if not (0 <= e < x < len(q.quote_t)) or q.quote_t[e] != r.entry_ns or q.quote_t[x] != r.witness_ns:
            raise ValueError('Raw entry/witness ordinal clock differs')
        if q.quote_t[x] >= q.quote_t[e]+duration:
            raise ValueError('Risk witness is not strictly before the execution timer')
        section = slice(e, x+1)
        good = (q.valid[section] & (q.epoch[section] == q.epoch[e]) & (q.epoch[section] >= 0)
                & (q.bid[section] > 0) & (q.ask[section] > q.bid[section]))
        if not good.all() or (np.diff(q.quote_t[section]) > gap).any():
            raise ValueError('Raw witness path crossed physical support loss')
        net = ((q.bid[section]-q.ask[e])/NS-fee if r.chosen_side == 1
               else (q.bid[e]-q.ask[section])/NS-fee)
        strict_mae, inclusive_mae = float(net[:-1].min()), float(net.min())
        if (strict_mae != inclusive_mae or not np.isclose(strict_mae, r.witness_prefix_mae_net, rtol=0, atol=1e-10)
                or not np.isclose(net[-1], r.witness_mark_net, rtol=0, atol=1e-10)
                or net[-1] < r.G_event):
            raise AssertionError('Strict full-quote prefix differs from factual witness map')
        hits = {}
        for plan in PLANS:
            adverse = np.flatnonzero(net[:-1] <= -float(r['R_'+plan]))
            first = e+int(adverse[0]) if len(adverse) else -1
            safe = not len(adverse)
            if int(r['y_risk_safe_'+plan]) != int(safe):
                raise AssertionError('Raw adverse-first-hit order differs from risk label')
            hits[plan] = dict(safe=bool(safe), first_adverse_ordinal=first,
                first_adverse_ns=str(int(q.quote_t[first])) if first >= 0 else None)
        checks.append(dict(t_ns=str(int(r.t_ns)), chosen_side=int(r.chosen_side), witness_id=r.witness_event_id,
            entry_ordinal=e, witness_ordinal=x, witness_ns=str(int(r.witness_ns)),
            strict_prefix_mae_net=strict_mae, max_money_error=abs(strict_mae-r.witness_prefix_mae_net), plans=hits))
    return dict(status='PASS', kind='INDEPENDENT_STRICT_ORDINAL_PREFIX_AND_FIRST_ADVERSE_HIT', probes=len(checks), checks=checks)
