"""As-of multiview adapter with explicit previous-session and active-story retention."""
from dataclasses import dataclass
import json
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd

from spectrum.io import sha
from spectrum.representation.session import session_bounds
from spectrum.representation.session_store import open_session_store
from .tokens import (motion_tokens, zone_tokens, quote_arrays, integer_clock,
                     TERMINAL, NS, TICK)
from .measures import enrich, encode, path_measures, annotate_zone_history
from .similarity import StoryIndex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@dataclass
class SessionStories:
    session_id: str
    previous_session_id: str
    symbol: str
    start_ns: int
    end_ns: int
    observed_through_ns: int
    quotes: pd.DataFrame
    fast: pd.DataFrame
    context: pd.DataFrame
    motion: pd.DataFrame
    zones: pd.DataFrame

    @cached_property
    def tokens(self):
        return pd.concat([self.motion, self.zones], ignore_index=True)


def build_session(quotes, *, session_id, previous_session_id, symbol, start_ns, end_ns,
                  scales, fast=None, context=None, observed_through_ns=None):
    start_ns, end_ns = integer_clock(start_ns), integer_clock(end_ns)
    through = end_ns if observed_through_ns is None else integer_clock(observed_through_ns)
    if not start_ns <= through <= end_ns: raise ValueError('Prefix outside session')
    if len({s['name'] for s in scales}) != len(scales) or not scales:
        raise ValueError('Unique nonempty scales required')
    for scale in scales:
        for key in ('amplitude_ticks','time_limit_seconds','update_seconds','zone_ticks',
                    'departure_ticks','confirmation_seconds','flow_seconds'):
            if isinstance(scale[key], bool) or int(scale[key]) != scale[key] or scale[key] <= 0:
                raise ValueError('Positive integer scale definitions required')
        if scale['confirmation_seconds'] > 30: raise ValueError('Gate exceeds quote support')
    q = quotes[quotes.available_ns.between(start_ns, through)].reset_index(drop=True)
    motions, zones = [], []
    for scale in scales:
        motions.append(motion_tokens(q, session_id, scale, end_ns=through, session_end_ns=end_ns))
        zones.append(annotate_zone_history(zone_tokens(q, session_id, scale, end_ns=through, session_end_ns=end_ns)))
    return SessionStories(session_id, previous_session_id, symbol, start_ns, end_ns, through,
        q, fast, context, pd.concat(motions, ignore_index=True), pd.concat(zones, ignore_index=True))


def load_session(root, scales):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    preparation = root.parents[2]/'prepared'/'manifest.json'
    if sha(preparation) != manifest['source_identity']['preparation_sha256']:
        raise ValueError('Session handoff preparation receipt drift')
    store = open_session_store(root)  # verifies all payloads and native quote source digests
    start, end = session_bounds(manifest['date'])
    session = build_session(store.quotes, session_id=manifest['date'],
        previous_session_id=manifest['previous_date'], symbol=manifest['symbol'],
        start_ns=start, end_ns=end, scales=scales, fast=store.fast, context=store.context)
    hashes = {str(root/'manifest.json'):sha(root/'manifest.json'), str(preparation):sha(preparation)}
    hashes.update({str(root/name):value for name,value in manifest['files'].items()})
    hashes.update({x['path']:x['sha256'] for x in manifest['quote_sources']})
    return session, hashes


def live_states(session, clock):
    """Virtual as-of snapshots, not edits to previously emitted immutable updates."""
    rows = []
    qt, price, good, _ = quote_arrays(session.quotes)
    i = int(np.searchsorted(qt, clock, side='right'))-1
    if i < 0 or not good[i] or clock-qt[i] > 30*NS or clock >= session.end_ns:
        return pd.DataFrame(columns=session.tokens.columns)
    for frame in (session.motion, session.zones):
        prefix = frame[frame.available_ns <= clock]
        latest = prefix.drop_duplicates('story_id', keep='last')
        active = latest[~latest.phase.isin(TERMINAL | {'OBSERVED_JUMP'})]
        for _, r in active.iterrows():
            r = r.copy()
            r['available_ns'] = clock; r['source_index'] = i
            r['source_ordinal'] = int(session.quotes.source_ordinal.iloc[i])
            r['mid_nano'] = int(price[i]); r['kind'] = 'LIVE_STATE'; r['phase'] = 'FORMING'
            r['token_id'] = str(r.story_id)+':asof:'+str(clock)
            r['elapsed_seconds'] = (clock-int(r.start_ns))/NS
            if r.family == 'motion':
                p = price[int(r.start_index):i+1]
                # All discontinuities terminate stories before these snapshots.
                r['low_nano'] = int(p.min()); r['high_nano'] = int(p.max())
                change = np.diff(p)
                r['path_nano'] = int(np.abs(change).sum())
                r['up_nano'] = int(np.maximum(0, change).sum())
                r['down_nano'] = int(np.maximum(0, -change).sum())
                signs = np.sign(change[change != 0])
                r['turns'] = int(np.count_nonzero(np.diff(signs)))
                if r.direction:
                    j = int(np.argmax(p) if r.direction > 0 else np.argmin(p))
                    r['extreme_nano'] = int(p[j])
                    r['extreme_ns'] = max(int(r.start_ns), int(qt[int(r.start_index)+j]))
            else:
                # Snapshot direction is current geometry; the last edge/gate
                # update can describe an older side after an unqualified jump.
                r['direction'] = 1 if price[i] >= r.zone_high_nano else -1 if price[i] < r.zone_low_nano else 0
                if r.pending_start_ns >= 0:
                    r['pending_seconds'] = (clock-int(r.pending_start_ns))/NS
            rows.append(r)
    result = pd.DataFrame(rows)
    for name in (n for n in result if n.endswith('_ns')):
        # Construct from original scalars: casting an already inferred float
        # column back to int would not recover the lost low nanosecond bits.
        result[name] = pd.array([row.get(name, pd.NA) for row in rows], dtype='Int64')
    return result


def price_memory(sessions, clock, current_mid, budget):
    """Same absolute bands across sessions; visit identity never crosses a close/gap."""
    result = []
    for session in sessions:
        t = min(clock, session.end_ns)
        f = session.zones[session.zones.available_ns <= t]
        active = live_states(session, t)
        active = active[active.family.eq('zone')]
        # All completed visits plus one current state per active visit.
        latest = (pd.concat([f, active], ignore_index=True) if len(active) else f.copy()).drop_duplicates('story_id', keep='last')
        latest = latest[~latest.phase.eq('OBSERVED_JUMP')]
        for scale, group in latest.groupby('scale', sort=False):
            addresses = group[['zone_low_nano','zone_high_nano']].drop_duplicates()
            addresses['distance'] = np.abs((addresses.zone_low_nano+addresses.zone_high_nano)/2-current_mid)
            chosen = addresses.sort_values(['distance','zone_low_nano']).head(budget).zone_low_nano
            selected = group[group.zone_low_nano.isin(chosen)].copy()
            measures = path_measures(selected.reset_index(drop=True), session.quotes)
            selected['dwell'] = measures.zone_dwell_seconds.to_numpy()
            selected['observed_path'] = measures.path_ticks.to_numpy()
            for low, visits in selected.groupby('zone_low_nano', sort=True):
                ev = f[(f.scale == scale) & (f.zone_low_nano == low)]
                completed = visits.phase.eq('CONFIRMED')
                entry_known = visits.entry_direction.ne(0)
                result.append(dict(session_id=session.session_id, scale=scale, zone_low_nano=int(low),
                    zone_high_nano=int(visits.zone_high_nano.iloc[0]),
                    offset_ticks=((int(low)+int(visits.zone_high_nano.iloc[0]))/2-current_mid)/TICK,
                    dwell_seconds=float(visits.dwell.sum()), observed_path_ticks=float(visits.observed_path.sum()),
                    arrivals=int(ev.kind.eq('ARRIVAL').sum()), restarts=int(ev.kind.eq('PRESENT_ON_RESTART').sum()),
                    confirmed_visits=int(completed.sum()), censored_visits=int(visits.phase.eq('CENSORED').sum()),
                    pass_count=int((completed & entry_known & visits.direction.eq(visits.entry_direction)).sum()),
                    reject_count=int((completed & entry_known & visits.direction.ne(visits.entry_direction)).sum()),
                    edge_returns=int(ev.kind.eq('EDGE_RETURN').sum()), gate_aborts=int(ev.kind.eq('GATE_ABORT').sum()),
                    active_visits=int((~visits.phase.isin(TERMINAL)).sum()),
                    last_update_age_seconds=(clock-int(visits.available_ns.max()))/NS,
                    available_ns=int(visits.available_ns.max()),
                    definition='Fixed-band visits between confirmed exits; not independent trade opportunities'))
    return pd.DataFrame(result)


class ScalperAdapter:
    def __init__(self, sessions, config):
        self.sessions = {s.session_id:s for s in sessions}
        if len({s.symbol for s in sessions}) != 1: raise ValueError('No cross-contract memory')
        self.config = config
        self.index = StoryIndex(sessions)

    def observe(self, session_id, clock):
        clock = integer_clock(clock)
        current = self.sessions[session_id]
        if not current.start_ns <= clock <= current.observed_through_ns:
            raise ValueError('Query outside observed session prefix')
        previous = self.sessions.get(current.previous_session_id)
        if previous is not None and previous.observed_through_ns < previous.end_ns:
            raise ValueError('Prior session not observed through its declared close')
        if previous is not None and previous.end_ns > current.start_ns:
            raise ValueError('Previous-session chronology overlaps')
        budget = self.config['retention']
        all_current = current.tokens
        eligible = all_current[all_current.available_ns <= clock]
        selected = eligible.groupby(['family','scale'], sort=False).tail(budget['current_updates_per_family_scale'])
        completed = eligible[eligible.phase.isin(['CONFIRMED','TIME_LIMIT'])]
        completed = completed.groupby(['family','scale'], sort=False).tail(budget['current_completed_stories_per_family_scale'])
        selected = pd.concat([selected, completed], ignore_index=True).drop_duplicates('token_id')
        active = live_states(current, clock)
        selected = pd.concat([selected, active], ignore_index=True)
        previous_available = 0
        if previous is not None:
            past = previous.tokens
            previous_available = len(past)
            # One terminal state per story, with a separate budget in every channel.
            past = past.drop_duplicates('story_id', keep='last')
            past = past.groupby(['family','scale'], sort=False).tail(budget['previous_stories_per_family_scale'])
            selected = pd.concat([past, selected], ignore_index=True)
        matches = []
        for token in active[active.family.eq('motion')].itertuples(index=False):
            matches.extend(self.index.query(pd.Series(token._asdict()), current,
                previous.session_id if previous else None,
                per_session=self.config['similarity']['neighbors_per_session']))
        retrieved = {tid for m in matches for tid in m['context_token_ids']}
        context_ids = {tid for m in matches for tid in m['query_context_token_ids']}
        supplement = self.index.frame[self.index.frame.token_id.isin(retrieved | context_ids)]
        selected = pd.concat([selected, supplement], ignore_index=True).drop_duplicates('token_id', keep='first')
        frames = []
        for date, part in selected.groupby('session_id', sort=False):
            session = self.sessions[date]
            frames.append(enrich(part, session.quotes, session.fast))
        table = pd.concat(frames, ignore_index=True)
        table['_snapshot_order'] = table.kind.eq('LIVE_STATE').astype(int)
        # Equal nanoseconds can contain several native publications. Their
        # source order takes precedence over an arbitrary channel sort.
        table = table.sort_values(['available_ns','session_id','source_index','_snapshot_order',
            'family','scale','sequence_ordinal'], kind='stable').drop(columns='_snapshot_order').reset_index(drop=True)
        # Links refer to the selected observed sequence only; no future successor lookup.
        groups = table.groupby(['session_id','family','scale','story_id'], sort=False)
        table['previous_selected_update_id'] = groups.token_id.shift(1).fillna('')
        table['next_selected_observed_update_id'] = groups.token_id.shift(-1).fillna('')
        qt, price, good, _ = quote_arrays(current.quotes)
        i = int(np.searchsorted(qt, clock, side='right'))-1
        market_known = bool(i >= 0 and good[i] and clock-qt[i] <= 30*NS and clock < current.end_ns)
        mid = int(price[i]) if i>=0 else 0
        table['age_seconds'] = (clock-table.available_ns)/NS
        table['offset_current_ticks'] = (table.mid_nano-mid)/TICK if market_known else np.nan
        table['is_previous_session'] = table.session_id.ne(session_id).astype(float)
        table['is_retrieved'] = table.token_id.isin(retrieved).astype(float)
        memory = price_memory([s for s in (previous, current) if s is not None], clock, mid,
                              budget['price_nodes_per_session_scale']) if market_known else pd.DataFrame()
        context = pd.DataFrame()
        if current.context is not None:
            context = current.context[current.context.available_ns <= clock].copy()
        encoded = encode(table)
        return dict(tokens=table, model=encoded, price_memory=memory, similarity=matches, context=context,
            market=dict(known=market_known, mid_nano=mid if market_known else None, available_ns=int(qt[i]) if i>=0 else None),
            receipt=dict(version=self.config['version'], session_id=session_id, clock_ns=clock,
                current_available_updates=len(eligible), previous_available_updates=previous_available,
                previous_session_id=current.previous_session_id, previous_session_loaded=previous is not None,
                current_retained=int(table.session_id.eq(session_id).sum()),
                previous_retained=int(table.session_id.ne(session_id).sum()), active_retained=len(active),
                current_completed_reserved=len(completed),
                retrieved_tokens=len(retrieved), known_values=int(encoded['known'].sum()), numeric_values=encoded['known'].size,
                source_scope='Native BBO path plus separately published complete 1s book/T context',
                trade_admission=False, active_budget_policy='All active stories; recent and prior-session budgets are separate'))
