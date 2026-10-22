"""Immutable BB crossing labels and separately parameterized risk execution.

Every future column belongs to labels/map/executions. The scene contains only
causally available origin state. The physical geometry ID excludes the query,
risk grid, model and loss. No VWAP approximation is constructed here.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from numba import njit
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.crossing_map import geometry_id, lookup_line
from .simulation import NS, SimulationConfig, align_entries, simulate_tape, scalar_reference, causal_first_qualifying


@njit(cache=True)
def _map_marks(bid, ask, k, blocks, entries, crosses):
    values = np.full((len(entries), 6), np.nan)
    for i in range(len(entries)):
        e = entries[i]
        x = crosses[i]
        if e < 0 or x < 0:
            continue
        ext = bb.range_extrema(bid, ask, k, blocks, e, x)
        values[i, 0] = (bid[x]-ask[e])/1e9
        values[i, 1] = (bid[e]-ask[x])/1e9
        values[i, 2] = (ext[0]-ask[e])/1e9
        values[i, 3] = (bid[e]+ext[3])/1e9
        values[i, 4] = (-ext[1]-ask[e])/1e9
        values[i, 5] = (bid[e]-ext[2])/1e9
    return values


def _config(config):
    query = dict(family='BB', scales=[1, 3, 6], deadline_seconds=300., gain_multiple=2.,
                 reference_fee=.30, risk_cost_multiple=2., risk_half_multiple=.10, minimum_risk=.50)
    query.update(config.get('event_query', {}))
    if query['family'] != 'BB':
        raise ValueError('V2 materializes exact BB only; no synthetic VWAP')
    scales = list(query['scales'])
    if 1 not in scales or len(scales) != len(set(scales)) or not set(scales) <= {1, 3, 6}:
        raise ValueError('Use unique BB scales from 1/3/6 including M1 for causal risk')
    query['scales'] = sorted(scales)
    for name in ('deadline_seconds', 'gain_multiple', 'risk_cost_multiple', 'minimum_risk'):
        if not np.isfinite(query[name]) or query[name] <= 0:
            raise ValueError('Invalid event query '+name)
    for name in ('reference_fee', 'risk_half_multiple'):
        if not np.isfinite(query[name]) or query[name] < 0:
            raise ValueError('Invalid event query '+name)
    risk = dict(cost_multipliers=[2., 4., 6.], half_multiple=.1, minimum_risk=.5,
                gain_reference='fixed_event_G_for_all_plans')
    risk.update(config.get('risk', {}))
    if risk['gain_reference'] != 'fixed_event_G_for_all_plans':
        raise ValueError('Execution risk must not redefine the fixed event gain')
    multipliers = np.asarray(risk['cost_multipliers'], float)
    if len(multipliers) == 0 or not np.isfinite(multipliers).all() or (multipliers <= 0).any() or len(np.unique(multipliers)) != len(multipliers):
        raise ValueError('Unique positive risk cost multipliers required')
    if not np.isfinite(risk['half_multiple']) or risk['half_multiple'] < 0 or not np.isfinite(risk['minimum_risk']) or risk['minimum_risk'] <= 0:
        raise ValueError('Invalid causal volatility risk')
    execution = SimulationConfig.from_dict(config)
    if execution.deadline_seconds != query['deadline_seconds']:
        raise ValueError('Event and execution use the same horizon')
    if execution.max_quote_gap_seconds != 60:
        raise ValueError('Native BB geometry currently binds the 60-second support rule')
    if config.get('execution', {}).get('exit_mode', 'first_qualifying_crossing') != 'first_qualifying_crossing':
        raise ValueError('V2 execution requires first_qualifying_crossing')
    return query, risk, execution


def _summarize_map(line_map, roots, query):
    """Common-maturity target, equivalent to the frozen amplitude definition."""
    n = len(roots)
    width = len(query['scales'])*3
    a = line_map.sort_values(['t_ns', 'scale', 'role']).reset_index(drop=True)
    if len(a) != n*width:
        raise ValueError('A complete rectangular line map is required')
    def matrix(name):
        return a[name].to_numpy().reshape(n, width)
    entry = roots.entry_ns.to_numpy(np.int64)
    due = entry+int(round(query['deadline_seconds']*NS))
    observed = (matrix('cross_ordinal') >= 0) & (matrix('cross_ns') <= due[:, None])
    mature = matrix('ready') & (observed | (matrix('support_end_ns') > due[:, None]))
    all_known = mature.all(axis=1) & np.isfinite(roots.R_event.to_numpy()) & (entry > 0)
    labels = roots[['date', 't_ns', 'entry_ns', 'R_event', 'G_event']].copy()
    labels['all_known'] = all_known
    labels['event_known'] = all_known
    maturity = np.where(observed, matrix('cross_ns'), due[:, None])
    labels['maturity_ns'] = np.where(all_known, maturity.max(axis=1), 0)
    labels['available_ns'] = labels.maturity_ns
    firsts = np.full((n, 2), -1, np.int64)
    flags = []
    for si, side in enumerate(('L', 'S')):
        mark = matrix('mark_'+side)
        good = observed & (mark >= roots.G_event.to_numpy()[:, None])
        flags.append(good.any(axis=1))
        candidates = np.where(good, matrix('cross_ordinal'), np.iinfo(np.int64).max)
        column = candidates.argmin(axis=1)
        first = candidates[np.arange(n), column]
        first = np.where(good.any(axis=1), first, -1)
        firsts[:, si] = first
        labels['positive_witness_'+side] = good.any(axis=1)
        labels['first_qualifying_ordinal_'+side] = first
        labels['first_qualifying_ns_'+side] = np.where(first >= 0, matrix('cross_ns')[np.arange(n), column], 0)
        labels['first_qualifying_event_id_'+side] = np.where(first >= 0, matrix('event_id')[np.arange(n), column], '')
        present = observed & np.isfinite(mark)
        highest = np.where(present, mark, -np.inf).max(axis=1)
        highest = np.where(present.any(axis=1), highest, np.nan)
        labels['max_cross_net_'+side] = highest
        labels['max_cross_net_R_'+side] = highest/roots.R_event.to_numpy()
        labels['gain_margin_'+side] = highest-roots.G_event.to_numpy()
        labels['gain_margin_R_'+side] = labels['gain_margin_'+side]/roots.R_event.to_numpy()
    state = flags[0].astype(np.int8)+2*flags[1].astype(np.int8)
    labels['state'] = np.where(all_known, state, -1).astype(np.int8)
    labels['y_crossing'] = labels.state
    labels['event'] = np.where(all_known, state > 0, -1).astype(np.int8)
    labels['direction_known'] = all_known & (state > 0)
    labels['direction'] = np.where(all_known & (state > 0), state-1, -1).astype(np.int8)
    labels['deadline_ns'] = due
    labels['deadline_seconds'] = query['deadline_seconds']
    labels['gain_multiple'] = query['gain_multiple']
    labels['reference_fee'] = query['reference_fee']
    labels['observed_cells'] = observed.sum(axis=1)
    labels['mature_cells'] = mature.sum(axis=1)
    return labels, firsts


def build_targets(day, q, origins_ns, config):
    """Build one day's factual map, fixed Y and three independently run plans.

    Returned future tables must never be admitted as X. `decision_eligible` is
    causal; `entry_status`, `all_known`, support ends and all execution fields
    are consequences and cannot filter policy predictions prospectively.
    """
    query, risk_config, execution = _config(config)
    t = np.asarray(origins_ns, np.int64)
    if t.ndim != 1 or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError('Strictly increasing one-dimensional decision origins required')
    if not len(t):
        raise ValueError('At least one origin required')
    date = str(day.meta['date'])
    roots = align_entries(q, t, config)
    roots.insert(0, 'date', date)
    decision = roots.decision_ordinal.to_numpy(np.int64)
    ds = np.maximum(decision, 0)
    good = q.valid & (q.epoch >= 0) & (q.bid > 0) & (q.ask > q.bid)
    roots['decision_spread'] = np.where(decision >= 0, (q.ask[ds]-q.bid[ds])/1e9, np.nan)
    scene = pd.DataFrame(dict(t_ns=t, available_ns=roots.decision_ns.to_numpy(),
        mid=np.where(decision >= 0, (q.bid[ds].astype(float)+q.ask[ds])/2e9, np.nan),
        decision_spread=roots.decision_spread.to_numpy()))
    k = bb.reduce_quotes(q)
    qt = q.quote_t[k]
    price = (q.bid[k].astype(float)+q.ask[k])/2e9
    blocks = bb.extremum_blocks(q.bid, q.ask, k)
    origin_j = np.searchsorted(qt, t, side='right')-1
    os = np.maximum(origin_j, 0)
    entry = roots.entry_ordinal.to_numpy(np.int64)
    eok = entry >= 0
    es = np.maximum(entry, 0)
    bank_frames = []
    maps = []
    checks = []
    origin_support = {}
    parts = day.meta['quote_parts']
    for scale in query['scales']:
        middle, half, support = bb.native_geometry(qt, price, good[k], q.epoch[k], scale)
        origin_geometry = (origin_j >= 0) & (t-qt[os] <= 60*NS)
        origin_support[scale] = np.where(origin_geometry, support[os], -1)
        scene[f'BB{scale}_middle'] = np.where(origin_geometry, middle[os], np.nan)
        scene[f'BB{scale}_half'] = np.where(origin_geometry, half[os], np.nan)
        scene[f'BB{scale}_support'] = origin_support[scale]
        if scale == 1:
            costs = roots.decision_spread.to_numpy()+query['reference_fee']
            roots['R_event'] = np.maximum.reduce([query['risk_cost_multiple']*costs,
                query['risk_half_multiple']*scene.BB1_half.to_numpy(), np.full(len(t), query['minimum_risk'])])
            roots['risk_R'] = roots.R_event
            roots['G_event'] = query['gain_multiple']*roots.R_event
            scene['R_event'] = roots.R_event.to_numpy()
        end = bb.support_ends(qt, support, min(q.deadline_ns, q.coverage_end_ns))[os]
        ready = ((origin_j >= 0) & (support[os] >= 0) & eok & (q.quote_t[es] < end)
                 & (t-qt[os] <= 60*NS))
        row, role, direction, cause, dp, dl = bb.crossing_rows(qt, price, middle, half, support)
        source_id = geometry_id('BB', scale, parts)
        bank = pd.DataFrame(dict(family='BB', scale=scale, ordinal=k[row], t_ns=qt[row],
            role=role, direction=direction, support=support[row], mid=price[row],
            line=middle[row]+(role-1)*half[row], cause=[bb.CAUSES[c] for c in cause]))
        bank['event_id'] = [f'{source_id}/{o}/{r}/{d}' for o, r, d in zip(bank.ordinal, bank.role, bank.direction)]
        bank_frames.append(bank)
        for line in range(3):
            b = bank.loc[bank.role == line].reset_index(drop=True)
            ords = b.ordinal.to_numpy(np.int64)
            crossing = lookup_line(ords, entry, ready, end, q.quote_t)
            known = crossing >= 0
            event_ids = np.full(len(t), '', object)
            directions = np.zeros(len(t), np.int8)
            if len(b):
                ix = np.searchsorted(ords, crossing[known], side='left')
                event_ids[known] = b.event_id.to_numpy()[ix]
                directions[known] = b.direction.to_numpy()[ix]
            values = _map_marks(q.bid, q.ask, k, blocks, entry, crossing)-query['reference_fee']
            cross_ns = np.where(known, q.quote_t[np.maximum(crossing, 0)], 0)
            frame = pd.DataFrame(dict(date=date, t_ns=t, family='BB', scale=scale, role=line,
                entry_ordinal=entry, entry_ns=roots.entry_ns.to_numpy(), ready=ready,
                support_end_ns=np.where(ready, end, 0), cross_ordinal=crossing, cross_ns=cross_ns,
                event_id=event_ids, cross_direction=directions, resolved=known,
                within_horizon=known & (cross_ns <= roots.entry_ns.to_numpy()+int(round(query['deadline_seconds']*NS))),
                delay_s=np.where(known, (cross_ns-roots.entry_ns.to_numpy())/NS, np.nan)))
            frame[['mark_L', 'mark_S', 'mae_L', 'mae_S', 'mfe_L', 'mfe_S']] = values
            frame['R_event'] = roots.R_event.to_numpy()
            frame['risk_R'] = roots.R_event.to_numpy()
            frame['G_event'] = roots.G_event.to_numpy()
            maps.append(frame)
            selected = np.flatnonzero(known)
            if len(selected):
                i = int(selected[len(selected)//2])
                e, x = int(entry[i]), int(crossing[i])
                direct = np.array([(q.bid[x]-q.ask[e])/1e9, (q.bid[e]-q.ask[x])/1e9,
                    (q.bid[e:x+1].min()-q.ask[e])/1e9, (q.bid[e]-q.ask[e:x+1].max())/1e9,
                    (q.bid[e:x+1].max()-q.ask[e])/1e9, (q.bid[e]-q.ask[e:x+1].min())/1e9])-query['reference_fee']
                error = float(np.max(np.abs(direct-values[i])))
                if error > 1e-9:
                    raise ValueError('Full-quote crossing profile mismatch')
                checks.append(dict(kind='map_raw', scale=scale, role=line, t_ns=str(t[i]), max_money_error=error))
    bank = pd.concat(bank_frames, ignore_index=True).sort_values(['ordinal', 'scale', 'role']).reset_index(drop=True)
    line_map = pd.concat(maps, ignore_index=True).sort_values(['t_ns', 'scale', 'role']).reset_index(drop=True)
    labels, firsts = _summarize_map(line_map, roots, query)
    due = roots.entry_ns.to_numpy()+int(round(query['deadline_seconds']*NS))
    width = len(query['scales'])*3
    observed = line_map.within_horizon.to_numpy().reshape(len(t), width)
    cross = line_map.cross_ordinal.to_numpy().reshape(len(t), width)
    executable_firsts = np.full((len(t), 2), -1, np.int64)
    for si, side in enumerate(('L', 'S')):
        adjusted = line_map['mark_'+side].to_numpy().reshape(len(t), width) + query['reference_fee']-execution.fee_roundtrip
        candidates = np.where(observed & (adjusted >= roots.G_event.to_numpy()[:, None]), cross, np.iinfo(np.int64).max)
        earliest = candidates.min(axis=1)
        executable_firsts[:, si] = np.where(earliest < np.iinfo(np.int64).max, earliest, -1)
    causal_origin = ((decision >= 0) & good[ds] & (t-q.quote_t[ds] <= int(execution.max_quote_gap_seconds*NS))
        & np.isfinite(roots.R_event.to_numpy()) & (t >= q.entry_start_ns)
        & (t+int((2*execution.pending_timeout_seconds+execution.deadline_seconds)*NS) < q.deadline_ns))
    roots['decision_eligible'] = causal_origin
    roots['geometry_ready'] = np.logical_and.reduce([scene[f'BB{s}_support'].to_numpy() >= 0 for s in query['scales']])
    executions = []
    candidates = np.flatnonzero(causal_origin)
    probes = candidates[np.linspace(0, len(candidates)-1, min(3, len(candidates)), dtype=int)] if len(candidates) else []
    for i in probes:
        online = causal_first_qualifying(q, int(entry[i]), bank, float(roots.G_event.iloc[i]),
            execution.fee_roundtrip, {s: int(origin_support[s][i]) for s in query['scales']}, query['deadline_seconds'])
        if not np.array_equal(online, executable_firsts[i]):
            raise ValueError('Precomputed crossing disagrees with online seen-line rule')
        checks.append(dict(kind='online_seen_lines', t_ns=str(t[i]), sides=2))
    for multiplier in risk_config['cost_multipliers']:
        plan = f'cost{multiplier:g}'
        risks = np.maximum.reduce([float(multiplier)*(roots.decision_spread.to_numpy()+query['reference_fee']),
            risk_config['half_multiple']*scene.BB1_half.to_numpy(), np.full(len(t), risk_config['minimum_risk'])])
        roots['R_'+plan] = risks
        frame = simulate_tape(q, t, risks, executable_firsts, config)
        frame.insert(0, 'date', date)
        frame.insert(1, 'plan', plan)
        frame['R_trade'] = frame.risk_R
        frame['R_event'] = np.repeat(roots.R_event.to_numpy(), 2)
        frame['G_event'] = np.repeat(roots.G_event.to_numpy(), 2)
        frame['cost_multiplier'] = float(multiplier)
        executions.append(frame)
        for i in probes:
            for si, side in enumerate((1, -1)):
                expected = scalar_reference(q, int(t[i]), float(risks[i]), side, int(executable_firsts[i, si]), config)
                actual = frame.iloc[2*i+si]
                error = 0.
                for key, value in expected.items():
                    if isinstance(value, float):
                        if not np.isclose(actual[key], value, atol=1e-10, rtol=0, equal_nan=True):
                            raise ValueError('Independent execution mismatch: '+key)
                        if np.isfinite(value):
                            error = max(error, abs(float(actual[key])-value))
                    elif actual[key] != value:
                        raise ValueError('Independent execution mismatch: '+key)
                checks.append(dict(kind='execution_raw', plan=plan, side=side, t_ns=str(t[i]), max_money_error=error))
    executions = pd.concat(executions, ignore_index=True)
    query_id = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    receipts = dict(version='BB_FIXED_EVENT_VARIABLE_STOP_V2', date=date, event_query=query,
        event_query_sha256=query_id, geometry_namespaces={str(s): geometry_id('BB', s, parts) for s in query['scales']},
        quote_rows=len(q.quote_t), reduced_rows=len(k), roots=len(t), bank_events=len(bank), map_cells=len(line_map),
        common_mature=int(labels.all_known.sum()), states={str(a): int(b) for a, b in labels.state.value_counts().items()},
        execution_statuses={str(a): int(b) for a, b in executions.status.value_counts().items()}, checks=checks,
        semantics={'target': 'Fixed first-line amplitude event; common complete maturity; unknown remains -1',
            'risk': 'R_trade changes only stop budget; G_event is identical for every risk plan',
            'take': 'First qualified crossing among first crossings of each original-support line, strictly after entry',
            'timer': 'TIME request at entry+H; TIME wins same-ns crossing; actual exit has latency',
            'x_boundary': 'Only scene/origin causal fields are potential X; future map, maturity, IDs and outcomes are excluded',
            'units': 'USD per reference ounce, taker BBO assumption; no depth/queue/impact claim'})
    return dict(roots=roots, scene=scene, bank=bank, map=line_map, labels=labels,
                executions=executions, receipts=receipts)
