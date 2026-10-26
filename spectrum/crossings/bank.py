"""Quote-clock native Bollinger bands and the factual bank of their crossings.

A line is crossed when bid OR ask changes side of it (canon_v1 crossing_rule).
Bands use only the past, so every crossing is a past fact at its own t_ns
(publication clock) and may be used as of that time. Flicker stays in the
bank; crossing_clusters marks line visits and opportunities from earlier
crossings only. Looking forward from an origin is future: spectrum.future.
Money: spectrum.economics.
"""
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
from numba import njit

NS = 1_000_000_000
MINUTE = 60 * NS
SCALES = (1, 3, 6, 9)
ROLES = ('lower', 'middle', 'upper')


def identity(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


def default_cooldown_ns():
    from spectrum.canon import load_canon
    return int(load_canon().value('crossing_opportunity_cooldown_ns'))


def geometry_definition(ddof=1, cooldown_ns=None):
    """Only geometry semantics/code; independent of query/model/profile code."""
    functions = (reduce_quotes, native_geometry.py_func, crossing_rows.py_func, crossing_clusters.py_func)
    return dict(version='QUOTE_NATIVE_BID_OR_ASK_CLUSTER_V3',scales=SCALES,period=20,k=2,ddof=ddof,
        minute_ns=MINUTE,tolerance_usd=1e-9,max_gap_ns=MINUTE,
        cooldown_ns=default_cooldown_ns() if cooldown_ns is None else int(cooldown_ns),
        code={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions})


def source_identity(meta, ddof=1, cooldown_ns=None):
    """Crossing facts depend only on the quotes of a session and the geometry."""
    return identity(geometry_definition(ddof, cooldown_ns),meta['date'],meta['symbol'],
        [(p['quote_file']['sha256'],p['start_ns'],p['end_ns']) for p in meta['quote_parts']])


def reduce_quotes(q):
    """Keep every BBO/support change and first publication of each native minute.

    Unchanged BBO publications remain in q for exact execution. Omitting them
    here cannot change a band, crossing, or native close. Retain both sides of
    a >60s silence so a missing interval cannot disappear during compression.
    """
    t = q.quote_t
    change = ((q.bid[1:] != q.bid[:-1]) | (q.ask[1:] != q.ask[:-1]) |
              (q.valid[1:] != q.valid[:-1]) | (q.epoch[1:] != q.epoch[:-1]) |
              (t[1:] // MINUTE != t[:-1] // MINUTE))
    gap = np.flatnonzero(np.diff(t) > MINUTE)
    keep = np.r_[True, change]
    # The last quote before each minute also certifies internal continuity;
    # sparse retained price changes alone must not manufacture a >60s gap.
    minute_edges = np.flatnonzero(t[1:] // MINUTE != t[:-1] // MINUTE)
    keep[minute_edges] = True
    keep[gap] = True; keep[gap + 1] = True; keep[-1] = True
    return np.flatnonzero(keep)


@njit(cache=True)
def native_geometry(t, price, valid, epoch, minutes, ddof=1):
    """19 completed native closes plus current tick. No fabricated warmup."""
    n = len(t); width = minutes * MINUTE
    middle = np.full(n, np.nan); half = np.full(n, np.nan)
    support = np.full(n, -1, np.int64)
    history = np.zeros(19); count = 0; cursor = 0
    bucket = -1; complete = False; token = -1; was_ready = False
    for i in range(n):
        b = t[i] // width
        good = valid[i] and np.isfinite(price[i]) and price[i] > 0
        broken = (i == 0 or not good or not valid[i-1] or epoch[i] != epoch[i-1]
                  or t[i] - t[i-1] > MINUTE)
        if broken:
            count = 0; cursor = 0; complete = False; was_ready = False
            bucket = b
        if not good:
            continue
        if b != bucket:
            if b == bucket + 1 and complete:
                history[cursor] = price[i-1]
                cursor = (cursor + 1) % 19
                count = min(count + 1, 19)
            elif b != bucket + 1:
                count = 0; cursor = 0; was_ready = False
            # First bucket after a support break is partial; the next starts
            # at a continuously supported rollover and can be completed.
            complete = True; bucket = b
        if count < 19:
            was_ready = False
            continue
        # Center before squaring: no catastrophic cancellation near GC 5000.
        center = price[i]; total = 0.; squares = 0.
        for j in range(19):
            v = history[j] - center; total += v; squares += v*v
        mean = center + total / 20
        variance = max(0., (squares - total * total / 20) / (20-ddof))
        h = 2 * np.sqrt(variance)
        if h <= 1e-10:
            was_ready = False
            continue
        if not was_ready:
            token += 1
        middle[i] = mean; half[i] = h; support[i] = token; was_ready = True
    return middle, half, support


@njit(cache=True)
def crossing_rows(t, bid, ask, middle, half, support):
    """A line is crossed at a publication where bid OR ask changes side of it.

    Side is price >= line, separately for bid and ask (USD/oz). Each supported
    publication is compared with the previous one of the same support segment;
    the first publication of a segment only sets the sides. Direction is +1 when
    a side moved up through the line, -1 down. A band line moving through a
    still quote is a crossing as well.
    """
    rows = []; roles = []; directions = []
    bid_above = np.zeros(3, np.bool_); ask_above = np.zeros(3, np.bool_)
    started = False
    for i in range(len(t)):
        if support[i] < 0 or i == 0 or support[i] != support[i-1]:
            started = False
        if support[i] < 0:
            continue
        for r in range(3):
            line = middle[i] + (r-1) * half[i]
            b = bid[i] >= line; a = ask[i] >= line
            if started and (b != bid_above[r] or a != ask_above[r]):
                rows.append(i); roles.append(r)
                directions.append(1 if (b and not bid_above[r]) or (a and not ask_above[r]) else -1)
            bid_above[r] = b; ask_above[r] = a
        started = True
    return np.asarray(rows, np.int64), np.asarray(roles, np.int64), np.asarray(directions, np.int64)


@njit(cache=True)
def crossing_clusters(t, role, ordinal, support, cooldown_ns):
    """Markers of one scale's crossings, ordered by (ordinal, role). Past only.

    Every crossing stays a fact; the markers only group flicker. For each
    crossing of line r: since_same_line_ns is the silence on r since its
    previous crossing (-1 for the first of the segment); line_visit says that
    another line of the scale was crossed since then (first crossings count as
    visits); opportunity = line_visit OR silence >= cooldown_ns. A cluster is an
    opportunity plus the later crossings of the same line until the next one;
    opener is its row and rank the position inside it. Crossings at one
    publication are not 'between' each other.
    """
    n = len(t)
    silence = np.full(n, -1, np.int64); visit = np.zeros(n, np.bool_)
    opportunity = np.zeros(n, np.bool_); opener = np.zeros(n, np.int64); rank = np.zeros(n, np.int64)
    last_t = np.full(3, -1, np.int64); other_since = np.ones(3, np.bool_)
    last_opener = np.full(3, -1, np.int64); last_rank = np.zeros(3, np.int64)
    segment = -2; i = 0
    while i < n:
        if support[i] != segment:
            segment = support[i]
            last_t[:] = -1; other_since[:] = True; last_opener[:] = -1; last_rank[:] = 0
        j = i
        while j < n and ordinal[j] == ordinal[i] and support[j] == segment:
            j += 1
        crossed = np.zeros(3, np.bool_)
        for e in range(i, j):
            r = role[e]; crossed[r] = True
            visit[e] = other_since[r]
            if last_t[r] >= 0:
                silence[e] = t[e] - last_t[r]
            opportunity[e] = visit[e] or silence[e] >= cooldown_ns
            if opportunity[e]:
                last_opener[r] = e; last_rank[r] = 0
            else:
                last_rank[r] += 1
            opener[e] = last_opener[r]; rank[e] = last_rank[r]
            last_t[r] = t[e]
        any_crossed = crossed.any()
        for r in range(3):
            other_since[r] = (not crossed[r]) and (other_since[r] or any_crossed)
        i = j
    return silence, visit, opportunity, opener, rank


def build_bank(q, source_id, ddof=1, cooldown_ns=None):
    """Crossing facts of every native scale; bands are built from the midpoint.

    Flicker stays in the bank; crossing_clusters marks line visits and
    opportunities (cooldown from canon crossing_opportunity_cooldown_ns).
    """
    cooldown_ns = default_cooldown_ns() if cooldown_ns is None else int(cooldown_ns)
    k = reduce_quotes(q); t = q.quote_t[k]
    bid = q.bid[k] / 1e9; ask = q.ask[k] / 1e9
    price = (q.bid[k].astype(float) + q.ask[k]) / 2e9
    valid = q.valid[k] & (q.bid[k] > 0) & (q.ask[k] > q.bid[k])
    supports = {}; frames = []
    for s in SCALES:
        middle, half, support = native_geometry(t, price, valid, q.epoch[k], s, ddof)
        supports[s] = support
        row, role, direction = crossing_rows(t, bid, ask, middle, half, support)
        order = np.lexsort((role, row)); row, role, direction = row[order], role[order], direction[order]
        ordinal = k[row]
        events = np.asarray([f'{source_id}/{int(o)}/{s}/{ROLES[r]}/{int(d)}' for o, r, d in zip(ordinal, role, direction)], object)
        silence, visit, opportunity, opener, rank = crossing_clusters(t[row], role, ordinal, support[row], cooldown_ns)
        frames.append(pd.DataFrame(dict(event_id=events, scale=s, role=role, direction=direction,
            t_ns=t[row], ordinal=ordinal, reduced_row=row, support=support[row],
            line=middle[row]+(role-1)*half[row], bid=bid[row], ask=ask[row],
            since_same_line_ns=silence, line_visit=visit, opportunity=opportunity,
            cluster_event_id=events[opener] if len(events) else events, cluster_rank=rank)))
    bank = pd.concat(frames, ignore_index=True).sort_values(['ordinal','scale','role']).reset_index(drop=True)
    return bank, dict(t=t, bid=bid, ask=ask, support=supports)
