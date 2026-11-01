"""Immutable multiscale updates on native publications, not retrospectively timed pivots.

All integer clocks/prices stay integer. A turn is published at its threshold
crossing, with the earlier extreme retained only as an already observed fact.
The existing zone engine is reused without changing its frozen implementation.
"""
import numpy as np
import pandas as pd
from numba import njit

from experiments.zone_episode_lab.episodes import build_episodes, NS, TICK

MOTION_KINDS = {1: 'START', 2: 'DIRECTION_SET', 3: 'PROGRESS',
                4: 'TURN_CONFIRMED', 5: 'TIME_LIMIT', 6: 'CENSORED'}
MOTION_COLUMNS = ['available_ns', 'kind_code', 'story_number', 'segment',
                  'start_ns', 'start_index', 'source_index', 'start_mid_nano',
                  'mid_nano', 'low_nano', 'high_nano', 'path_nano', 'turns',
                  'up_nano', 'down_nano', 'direction', 'extreme_ns', 'extreme_nano']
TERMINAL = {'CONFIRMED', 'TIME_LIMIT', 'CENSORED'}


def integer_clock(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('Integer nanosecond clock required')
    return int(value)


def validate_quotes(quotes):
    for name in ('available_ns', 'bid_nano', 'ask_nano', 'epoch', 'source_ordinal'):
        if name not in quotes or quotes[name].dtype.kind not in 'iu':
            raise ValueError('Integer quote fields required: ' + name)
    if quotes.empty or not quotes.available_ns.is_monotonic_increasing:
        raise ValueError('Nonempty ordered native quote tape required')
    if quotes.valid.isna().any() or quotes.valid.dtype.kind != 'b':
        raise ValueError('Boolean validity required')


def quote_arrays(quotes):
    return (quotes.available_ns.to_numpy(np.int64),
            (quotes.bid_nano.to_numpy(np.int64) + quotes.ask_nano.to_numpy(np.int64)) // 2,
            (quotes.valid & quotes.bid_nano.gt(0) & quotes.ask_nano.gt(quotes.bid_nano)
             & quotes.epoch.ge(0)).to_numpy(bool), quotes.epoch.to_numpy(np.int64))


@njit(cache=True)
def _motion(qt, price, good, epoch, amplitude, limit, heartbeat, gap, end, close_end):
    rows = []
    active = False
    serial = 0
    segment = 0
    direction = 0
    start = si = anchor = low = high = path = turns = up = down = extreme_t = extreme = 0
    last_sign = 0
    next_update = 0
    for i in range(len(qt) + 1):
        terminal = i == len(qt)
        now = end if terminal else qt[i]
        continuous = (i > 0 and good[i-1] and not terminal and good[i]
                      and epoch[i] == epoch[i-1] and now - qt[i-1] <= gap)
        if active:
            until = min(now, qt[i-1] + gap)
            while active and min(next_update, start + limit) <= until:
                due = min(next_update, start + limit)
                code = 5 if due == start + limit else 3
                p = price[i-1]
                rows.append((due, code, serial, segment, start, si, i-1, anchor,
                             p, low, high, path, turns, up, down, direction, extreme_t, extreme))
                if code == 5:
                    serial += 1
                    start = due; si = i-1; anchor = p; low = p; high = p
                    path = turns = up = down = direction = last_sign = 0
                    extreme = p; extreme_t = due
                    rows.append((due, 1, serial, segment, start, si, i-1, anchor,
                                 p, low, high, path, turns, up, down, direction, extreme_t, extreme))
                next_update = due + heartbeat
            broken = (not terminal and not continuous) or now - qt[i-1] > gap or (terminal and close_end)
            if broken:
                at = min(now, qt[i-1] + gap)
                rows.append((at, 6, serial, segment, start, si, i-1, anchor,
                             price[i-1], low, high, path, turns, up, down, direction, extreme_t, extreme))
                active = False
        if terminal:
            break
        if not good[i]:
            continue
        p = price[i]
        if not active:
            active = True; serial += 1; segment += 1
            start = now; si = i; anchor = p; low = p; high = p
            path = turns = up = down = direction = last_sign = 0
            extreme = p; extreme_t = now; next_update = now + heartbeat
            rows.append((now, 1, serial, segment, start, si, i, anchor, p,
                         low, high, path, turns, up, down, direction, extreme_t, extreme))
            continue
        delta = p - price[i-1]
        path += abs(delta); up += max(0, delta); down += max(0, -delta)
        sign = 1 if delta > 0 else -1 if delta < 0 else 0
        if sign:
            if last_sign and sign != last_sign:
                turns += 1
            last_sign = sign
        low = min(low, p); high = max(high, p)
        code = 0
        if direction == 0 and abs(p-anchor) >= amplitude:
            direction = 1 if p > anchor else -1
            extreme = p; extreme_t = now; code = 2
        elif direction:
            if direction * (p-extreme) > 0:
                extreme = p; extreme_t = now
            elif direction * (extreme-p) >= amplitude:
                code = 4
        if code:
            rows.append((now, code, serial, segment, start, si, i, anchor, p,
                         low, high, path, turns, up, down, direction, extreme_t, extreme))
        if code == 4:
            serial += 1
            start = now; si = i; anchor = p; low = p; high = p
            path = turns = up = down = direction = last_sign = 0
            extreme = p; extreme_t = now; next_update = now + heartbeat
            rows.append((now, 1, serial, segment, start, si, i, anchor, p,
                         low, high, path, turns, up, down, direction, extreme_t, extreme))
    return rows


def motion_tokens(quotes, session_id, scale, *, end_ns, session_end_ns, gap_seconds=30):
    validate_quotes(quotes)
    end = integer_clock(end_ns)
    q = quotes[quotes.available_ns <= end].reset_index(drop=True)
    qt, price, good, epoch = quote_arrays(q)
    rows = _motion(qt, price, good, epoch, int(scale['amplitude_ticks']*TICK),
                   int(scale['time_limit_seconds']*NS), int(scale['update_seconds']*NS),
                   int(gap_seconds*NS), end, end == session_end_ns)
    f = pd.DataFrame(rows, columns=MOTION_COLUMNS, dtype='int64')
    f['kind'] = f.kind_code.map(MOTION_KINDS)
    f['phase'] = f.kind.map({'START':'OBSERVED_START', 'DIRECTION_SET':'FORMING',
                            'PROGRESS':'FORMING', 'TURN_CONFIRMED':'CONFIRMED',
                            'TIME_LIMIT':'TIME_LIMIT', 'CENSORED':'CENSORED'})
    f['family'] = 'motion'
    f['zone_low_nano'] = 0
    f['zone_high_nano'] = 0
    f['entry_direction'] = 0
    return _identity(f, quotes, session_id, scale)


def zone_tokens(quotes, session_id, scale, *, end_ns, session_end_ns, gap_seconds=30):
    validate_quotes(quotes)
    end = integer_clock(end_ns)
    q = quotes[quotes.available_ns <= end].reset_index(drop=True).copy()
    # Capture-local source ordinals can restart at a verified file handoff.
    # A separate native index makes simultaneous publications unambiguous.
    q['source_ordinal'] = np.arange(len(q), dtype=np.int64)
    events, _ = build_episodes(q, end_ns=end, bin_ticks=scale['zone_ticks'],
                              departure_ticks=scale['departure_ticks'],
                              confirmation_seconds=scale['confirmation_seconds'],
                              max_quote_age_seconds=gap_seconds)
    # The historical export sorts by event code. At a one-quote side switch,
    # abort the old gate BEFORE starting the new one (codes 6 then 5).
    events['_within_publication'] = events.event_code.replace({6: 4.5})
    events = events.sort_values(['available_ns','source_ordinal','_within_publication'], kind='stable')
    events = events.drop(columns='_within_publication')
    last_good = bool(q.valid.iloc[-1] and q.bid_nano.iloc[-1] > 0
                     and q.ask_nano.iloc[-1] > q.bid_nano.iloc[-1] and q.epoch.iloc[-1] >= 0)
    if end < session_end_ns and last_good and end-int(q.available_ns.iloc[-1]) <= gap_seconds*NS:
        # The old batch engine adds EOF censors. An as-of prefix is not a
        # newly observed session close; remove only that terminal artifact.
        terminal = (events.event_type.eq('CENSORED') & events.available_ns.eq(end)
                    & events.source_ordinal.eq(len(q)-1))
        events = events[~terminal].copy()
    f = events.rename(columns={'episode_id':'story_number', 'source_ordinal':'source_index',
                               'event_type':'kind', 'side':'direction'}).copy()
    starts = f[f.kind.isin(['ARRIVAL', 'PRESENT_ON_RESTART'])].set_index('story_number')
    has_start = f.story_number.isin(starts.index).to_numpy()
    # map(...).fillna(...) would transit through float64 for JUMP_OVER and
    # round real nanosecond clocks above 2**53, including otherwise known rows.
    for target, source, fallback in [('start_ns','available_ns','available_ns'),
                                     ('start_index','source_index','source_index'),
                                     ('start_mid_nano','mid_nano','mid_nano')]:
        value = starts[source].reindex(f.story_number.to_numpy(), fill_value=0).to_numpy(np.int64)
        f[target] = np.where(has_start, value, f[fallback].to_numpy(np.int64))
    f['entry_direction'] = starts.direction.reindex(f.story_number.to_numpy(), fill_value=0).to_numpy(np.int64)
    # Jumps have no visit episode; each has its own identity and no fictitious dwell.
    jump = f.kind.eq('JUMP_OVER')
    f.loc[jump, 'story_number'] = -1-np.arange(int(jump.sum()))
    f['phase'] = f.kind.map({'ARRIVAL':'OBSERVED_START', 'PRESENT_ON_RESTART':'OBSERVED_START',
        'EDGE_EXIT':'FORMING', 'EDGE_RETURN':'FORMING', 'GATE_START':'FORMING',
        'GATE_ABORT':'ABORTED', 'DEPARTURE_CONFIRMED':'CONFIRMED',
        'CENSORED':'CENSORED', 'JUMP_OVER':'OBSERVED_JUMP'})
    f['family'] = 'zone'
    f['zone_high_nano'] = f.zone_low_nano + int(scale['zone_ticks']*TICK)
    f['segment'] = q.epoch.to_numpy()[f.start_index.to_numpy()]
    return _identity(f, quotes, session_id, scale)


def _identity(f, quotes, session_id, scale):
    f = f.reset_index(drop=True)
    f['session_id'] = session_id
    f['scale'] = scale['name']
    f['amplitude_ticks'] = scale['amplitude_ticks']
    f['time_limit_seconds'] = scale['time_limit_seconds']
    f['flow_seconds'] = scale['flow_seconds']
    f['zone_ticks'] = scale['zone_ticks']
    f['departure_ticks'] = scale['departure_ticks']
    f['confirmation_seconds'] = scale['confirmation_seconds']
    prefix = session_id + ':' + str(f.family.iloc[0]) + ':' + scale['name']
    f['story_id'] = prefix + ':' + f.story_number.astype(str)
    f['token_id'] = prefix + ':u' + f.index.astype(str)
    f['sequence_ordinal'] = np.arange(len(f), dtype=np.int64)
    f['source_ordinal'] = quotes.source_ordinal.to_numpy()[f.source_index.to_numpy()]
    f['epoch'] = quotes.epoch.to_numpy()[f.source_index.to_numpy()]
    f['elapsed_seconds'] = (f.available_ns-f.start_ns)/NS
    f['coordinate_version'] = 'ABSOLUTE_ZERO_GRID_V1' if f.family.iloc[0] == 'zone' else 'CAUSAL_TURN_TIME_LIMIT_V1'
    # Optional timestamps must also stay integer when channels are concatenated.
    for name in ('extreme_ns','duration_ns'):
        if name in f: f[name] = f[name].astype('Int64')
    return f
