"""Quote-clock native Bollinger bank. Future event metadata never enters X."""
from pathlib import Path
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
CAUSES = ('PRICE', 'LINE', 'BOTH_SUFFICIENT', 'JOINT_NEEDED')
GEOMETRY = [f'bb{s}_{k}' for s in SCALES for k in ('position', 'log_half_ticks', 'spread_half', 'phase')]


def contract():
    return json.loads((Path(__file__).resolve().parents[2] / 'contracts/bb_crossing_bank_v1.json').read_text())


def identity(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


def geometry_definition(ddof=1):
    """Only geometry semantics/code; independent of query/model/profile code."""
    functions = (reduce_quotes, native_geometry.py_func, crossing_rows.py_func)
    return dict(version='QUOTE_NATIVE_STRICT_CROSS_V1',scales=SCALES,period=20,k=2,ddof=ddof,
        minute_ns=MINUTE,tolerance_usd=1e-9,max_gap_ns=MINUTE,
        code={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions})


def source_identity(meta, ddof=1):
    """Quote data and support masks bind a physical session, never its queries."""
    return identity(geometry_definition(ddof),meta['date'],meta['symbol'],
        [(p['quote_file']['sha256'],p['start_ns'],p['end_ns']) for p in meta['quote_parts']],
        {key:meta['files'][key]['sha256'] for key in ('execution_valid.npy','execution_segment.npy')})


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
def crossing_rows(t, price, middle, half, support):
    """Strict side change; zero carries the last strict side and anchor."""
    rows = []; roles = []; directions = []; causes = []; dp = []; dl = []
    side = np.zeros(3, np.int64); old_p = np.zeros(3); old_line = np.zeros(3)
    for i in range(len(t)):
        if support[i] < 0 or i == 0 or support[i] != support[i-1]:
            side[:] = 0
        if support[i] < 0:
            continue
        for r in range(3):
            line = middle[i] + (r-1) * half[i]; delta = price[i]-line
            sign = 1 if delta > 1e-9 else -1 if delta < -1e-9 else 0
            if sign != 0:
                if side[r] != 0 and sign != side[r]:
                    p_only = (price[i]-old_line[r])*sign > 1e-9
                    l_only = (old_p[r]-line)*sign > 1e-9
                    cause = 2 if p_only and l_only else 0 if p_only else 1 if l_only else 3
                    rows.append(i); roles.append(r); directions.append(sign); causes.append(cause)
                    dp.append(price[i]-old_p[r]); dl.append(line-old_line[r])
                side[r] = sign; old_p[r] = price[i]; old_line[r] = line
    return (np.asarray(rows, np.int64), np.asarray(roles, np.int64),
            np.asarray(directions, np.int64), np.asarray(causes, np.int64),
            np.asarray(dp), np.asarray(dl))


def build_bank(q, source_id, ddof=1):
    k = reduce_quotes(q); t = q.quote_t[k]
    price = (q.bid[k].astype(float) + q.ask[k]) / 2e9
    valid = q.valid[k] & (q.bid[k] > 0) & (q.ask[k] > q.bid[k])
    geometries = {}; frames = []
    for s in SCALES:
        middle, half, support = native_geometry(t, price, valid, q.epoch[k], s, ddof)
        geometries[s] = dict(middle=middle, half=half, support=support)
        row, role, direction, cause, dp, dl = crossing_rows(t, price, middle, half, support)
        ordinal = k[row]
        packet = [f'{source_id}/{int(o)}' for o in ordinal]
        events = [f'{p}/{s}/{ROLES[r]}/{int(d)}' for p, r, d in zip(packet, role, direction)]
        frames.append(pd.DataFrame(dict(event_id=events, packet_id=packet, scale=s, role=role,
            direction=direction, cause=[CAUSES[c] for c in cause], t_ns=t[row], ordinal=ordinal,
            reduced_row=row, support=support[row], mid=price[row], line=middle[row]+(role-1)*half[row],
            price_change=dp, line_change=dl, bid=q.bid[ordinal]/1e9, ask=q.ask[ordinal]/1e9)))
    bank = pd.concat(frames, ignore_index=True).sort_values(['ordinal','scale','role']).reset_index(drop=True)
    return bank, dict(k=k, t=t, price=price, geometries=geometries)


def packets(bank, scale=None):
    f = bank if scale is None else bank.loc[bank.scale == scale]
    f = f.copy()
    bit = f.role.to_numpy(int)
    if scale is None:
        bit = bit + f.scale.map({s: 3*i for i, s in enumerate(SCALES)}).to_numpy(int)
    f['bit'] = 1 << bit; f['label'] = bit
    p = f.groupby('ordinal', sort=False).agg(t_ns=('t_ns','first'), reduced_row=('reduced_row','first'),
        support=('support','first'), packet_id=('packet_id','first'), line_mask=('bit','sum'),
        label=('label','first'), count=('label','size')).reset_index()
    p.loc[p['count'] > 1, 'label'] = 12 if scale is None else 3
    p['packet_id'] += f'/M{scale}' if scale else '/ALL'
    return p.drop(columns='count')


def support_ends(t, support, coverage_end):
    """Exclusive first unsupported observation, capped by a 60s silence limit."""
    out = np.zeros(len(t), np.int64)
    bounds = np.r_[0, np.flatnonzero(support[1:] != support[:-1]) + 1, len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if support[a] < 0:
            continue
        end = min(int(t[b]) if b < len(t) else int(coverage_end), int(t[b-1]) + MINUTE)
        out[a:b] = end
    return out


def make_targets(bank, tape, origins, coverage_end, horizon_seconds=1800):
    t = np.asarray(origins, np.int64); j = np.searchsorted(tape['t'], t, side='right')-1
    if (j < 0).any():
        raise ValueError('Origin precedes source')
    out = pd.DataFrame(dict(t_ns=t)); ends = {}; knowns = {}
    for s in SCALES:
        g = tape['geometries'][s]; ready = (g['support'][j] >= 0) & (t-tape['t'][j] <= MINUTE)
        end = support_ends(tape['t'], g['support'], coverage_end)[j]
        ends[s] = end; knowns[s] = ready
        out[f'bb{s}_position'] = np.where(ready, (tape['price'][j]-g['middle'][j])/g['half'][j], np.nan)
        out[f'bb{s}_log_half_ticks'] = np.where(ready, np.log1p(g['half'][j]/.1), np.nan)
        out[f'bb{s}_phase'] = (t % (s*MINUTE)) / (s*MINUTE)
        p = packets(bank, s)
        _assign(out, f'm{s}', t, j, p, end, ready, horizon_seconds, g['support'])
    ready = np.logical_and.reduce([knowns[s] for s in SCALES])
    end = np.minimum.reduce([ends[s] for s in SCALES])
    _assign(out, 'global', t, j, packets(bank), end, ready, horizon_seconds)
    return out


def _assign(out, key, t, origin_j, p, end, ready, horizon, support=None):
    n = len(t); limit = t + horizon*NS; size = 14 if key == 'global' else 5
    y = np.full(n, -1, np.int16); ev_t = np.zeros(n, np.int64); ordinal = np.full(n, -1, np.int64)
    ids = np.full(n, '', object); masks = np.zeros(n, np.int32)
    if len(p):
        ix = np.searchsorted(p.t_ns.to_numpy(), t, side='right'); safe = np.minimum(ix, len(p)-1)
        event_t = p.t_ns.to_numpy()[safe]
        has = ready & (ix < len(p)) & (event_t < end) & (event_t <= limit)
        if support is not None:
            has &= p.support.to_numpy()[safe] == support[origin_j]
        y[has] = p.label.to_numpy()[safe[has]]; ev_t[has] = event_t[has]
        ordinal[has] = p.ordinal.to_numpy()[safe[has]]; ids[has] = p.packet_id.to_numpy()[safe[has]]
        masks[has] = p.line_mask.to_numpy()[safe[has]]
    no = ready & (y < 0) & (end > limit)
    y[no] = size-1
    ids[no] = [f'NO/{key}/{int(x//(horizon*NS))}' for x in t[no]]
    out[key+'_y'] = y; out[key+'_event_ns'] = ev_t; out[key+'_ordinal'] = ordinal
    out[key+'_id'] = ids; out[key+'_mask'] = masks; out[key+'_support_until_ns'] = end
    out[key+'_maturity_ns'] = np.where(ev_t > 0, ev_t, np.where(no, limit, 0))
    out[key+'_delay_s'] = np.where(ev_t > 0, (ev_t-t)/NS, np.nan)


def event_weights(ids):
    ids = pd.Series(np.asarray(ids))
    if (ids == '').any():
        raise ValueError('Unknown labels cannot receive event weights')
    w = 1 / ids.map(ids.value_counts()).to_numpy(float)
    return w / w.mean()


def ce_gradient(probability, label, weight=None):
    p = np.asarray(probability, float); y = np.asarray(label, int)
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Only mature labels enter CE')
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return g if weight is None else g * np.asarray(weight)[:, None]


@njit(cache=True)
def execution_breaks(t, valid, epoch):
    end = np.empty(len(t), np.int64); end[-1] = len(t)
    for i in range(len(t)-2, -1, -1):
        bad = (not valid[i+1] or epoch[i+1] != epoch[i] or t[i+1]-t[i] > MINUTE)
        end[i] = i+1 if bad else end[i+1]
    return end


def extremum_blocks(bid, ask, k, block=256):
    values = np.c_[bid[k], -bid[k], ask[k], -ask[k]].astype(float)
    padded = np.full(((len(k)+block-1)//block*block, 4), np.inf)
    padded[:len(k)] = values
    return padded.reshape(-1, block, 4).min(axis=1)


@njit(cache=True)
def range_extrema(bid, ask, k, blocks, left, right, block=256):
    v = np.array([float(bid[left]), -float(bid[left]), float(ask[left]), -float(ask[left])])
    a = np.searchsorted(k, left, side='left'); b = np.searchsorted(k, right, side='right')
    while a < b:
        if a % block == 0 and a + block <= b:
            for j in range(4): v[j] = min(v[j], blocks[a//block, j])
            a += block
        else:
            r = k[a]
            v[0] = min(v[0], bid[r]); v[1] = min(v[1], -bid[r])
            v[2] = min(v[2], ask[r]); v[3] = min(v[3], -ask[r]); a += 1
    return v


@njit(cache=True)
def trade_profile(t, bid, ask, valid, epoch, breaks, reduced, blocks, origins, cross_t, cross_ord, deadline, fee=.30):
    """Independent one-oz branches. Crossing arrays are one scale's packets.

    Outputs: entry/final/cross ordinal; state 0 EMPTY, 1 resolved, -1 censored;
    exact mark and lagged execution; quote-prefix extrema. No selection by Y.
    """
    n = len(origins); ints = np.full((n, 5), -1, np.int64); vals = np.full((n, 12), np.nan)
    for i in range(n):
        origin = origins[i]; j = np.searchsorted(t, origin, side='right')-1
        e = np.searchsorted(t, origin+100_000_000, side='right')
        ints[i, 3] = 0
        if j < 0 or not valid[j] or e >= len(t) or t[e] > origin+2*NS or e >= breaks[j]:
            continue
        ints[i, 0] = e; ints[i, 3] = -1
        cap = min(t[e]+1800*NS, deadline)
        k = np.searchsorted(cross_t, t[e], side='right')
        c = -1
        if k < len(cross_t) and cross_t[k] <= cap:
            c = cross_ord[k]
        request = t[c] if c >= 0 else cap
        x = np.searchsorted(t, request+100_000_000, side='right')
        ints[i, 1] = x if x < len(t) else -1; ints[i, 2] = c; ints[i, 4] = request
        stop = min(x, breaks[e]-1, len(t)-1)
        ext = range_extrema(bid, ask, reduced, blocks, e, stop)
        vals[i, 6] = (ext[0]-ask[e])/1e9-fee
        vals[i, 7] = (bid[e]+ext[3])/1e9-fee
        vals[i, 10] = (-ext[1]-ask[e])/1e9-fee
        vals[i, 11] = (bid[e]-ext[2])/1e9-fee
        if c >= e and c <= stop:
            ext = range_extrema(bid, ask, reduced, blocks, e, c)
            vals[i, 0] = (bid[c]-ask[e])/1e9-fee
            vals[i, 1] = (bid[e]-ask[c])/1e9-fee
            vals[i, 4] = (ext[0]-ask[e])/1e9-fee
            vals[i, 5] = (bid[e]+ext[3])/1e9-fee
            vals[i, 8] = (-ext[1]-ask[e])/1e9-fee
            vals[i, 9] = (bid[e]-ext[2])/1e9-fee
        if x < len(t) and x < breaks[e] and t[x] <= request+2*NS:
            ints[i, 3] = 1
            vals[i, 2] = (bid[x]-ask[e])/1e9-fee
            vals[i, 3] = (bid[e]-ask[x])/1e9-fee
    return ints, vals
