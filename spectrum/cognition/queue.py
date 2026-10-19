"""FIFO queue probes for hypothetical 1-lot limit orders over the full MBO tape.

Book mutations follow the repository L3 engine: A adds, C reduces, M modifies,
R clears; T (aggressor side) and F (resting side) are facts that do not change
the book. Every resting order carries a priority number; it is renewed on add,
on a price change and on a size increase (CME FIFO), and kept on a size
decrease. A probe placed at time t gets the next priority at the first record
with ts_recv >= t, so every order already resting at its price is ahead.

A probe fills, at its own price, on the first record that proves a FIFO match
would have reached it:
  1 BEHIND  - F on its side and price for an order with a later priority, and
              the queue ahead is exhausted: ahead = lots resting ahead at
              placement minus their cancels/reductions minus fills already
              reported for them but not yet removed by C/M. A behind-fill while
              ahead > 0 is a FIFO violation; it is counted and never fills;
  2 WORSE   - F on its side at a price worse than the probe's;
  3 THROUGH - T at a price strictly through the probe (below a bid, above an ask).
It never consumes liquidity from the replayed market (no-impact assumption)
and never fills by mere touch. A reset or a bad-book flag censors it; it
expires when no proof arrives before its cancel time.
"""
from pathlib import Path
import numpy as np
from numba import njit, types
from numba.typed import Dict
from spectrum.core.vendor.glbx_dbn import GlbxZipMboSource

VERSION = 'MBO_FIFO_QUEUE_PROBE_V2'
FILLED, EXPIRED, CENSORED, UNPLACED = 1, 2, 3, 4
KIND = {0: 'none', 1: 'behind', 2: 'worse', 3: 'through'}
F_MAYBE_BAD_BOOK = 1 << 2
_A, _C, _M, _R, _T, _F, _B, _S = (ord(c) for c in 'ACMRTFBA')


def compact(records):
    """Fixed-width DBN rows -> contiguous typed columns for the kernel."""
    return dict(ts=records['ts_recv'].astype(np.int64), action=records['action'].view(np.uint8).copy(),
                side=records['side'].view(np.uint8).copy(), price=records['price'].astype(np.int64),
                size=records['size'].astype(np.int64), order_id=records['order_id'].view(np.int64).copy(),
                flags=records['flags'].astype(np.uint8))


@njit(cache=True)
def _run(ts, action, side, price, size, oid, flags, p_place, p_side, p_price, p_end,
         out_status, out_fill, out_kind, out_ahead, out_viol):
    n = len(ts); m = len(p_place)
    slot = Dict.empty(key_type=types.int64, value_type=types.int64)
    s_prio = np.zeros(n + 1, np.int64); s_side = np.zeros(n + 1, np.int8)
    s_price = np.zeros(n + 1, np.int64); s_size = np.zeros(n + 1, np.int64)
    s_pend = np.zeros(n + 1, np.int64)  # reported fills not yet removed by C/M
    ahead = np.zeros(m, np.int64); consumed = np.zeros(m, np.int64)
    bid_lvl = Dict.empty(key_type=types.int64, value_type=types.int64)
    ask_lvl = Dict.empty(key_type=types.int64, value_type=types.int64)
    free = 0; counter = 0; nxt = 0
    active = np.empty(m, np.int64); n_active = 0; prio = np.zeros(m, np.int64)
    for i in range(n):
        t = ts[i]
        # place every probe whose effective time has arrived, before this record
        while nxt < m and p_place[nxt] <= t:
            counter += 1; prio[nxt] = counter
            lvl = bid_lvl if p_side[nxt] > 0 else ask_lvl
            out_ahead[nxt] = lvl[p_price[nxt]] if p_price[nxt] in lvl else 0
            ahead[nxt] = out_ahead[nxt]
            out_status[nxt] = EXPIRED
            active[n_active] = nxt; n_active += 1; nxt += 1
        a = action[i]
        if a == _R or (flags[i] & F_MAYBE_BAD_BOOK):
            for k in range(n_active):
                j = active[k]
                if p_end[j] >= t:
                    out_status[j] = CENSORED; out_fill[j] = t
            n_active = 0
            if a == _R:
                slot.clear(); bid_lvl.clear(); ask_lvl.clear()
                continue
            # a flagged mutation still applies to the book, as in the L3 engine
        if a == _T or a == _F:
            px = price[i]; sd = 1 if side[i] == _B else (-1 if side[i] == _S else 0)
            k = 0
            while k < n_active:
                j = active[k]
                done = 0
                if t > p_end[j]:
                    done = EXPIRED
                elif a == _T:
                    if (p_side[j] > 0 and px < p_price[j]) or (p_side[j] < 0 and px > p_price[j]):
                        done = FILLED; out_kind[j] = 3
                elif sd == p_side[j]:
                    if (p_side[j] > 0 and px < p_price[j]) or (p_side[j] < 0 and px > p_price[j]):
                        done = FILLED; out_kind[j] = 2
                    elif px == p_price[j] and oid[i] in slot:
                        q = slot[oid[i]]
                        if s_prio[q] < prio[j]:
                            consumed[j] += size[i]
                        elif ahead[j] - consumed[j] <= 0:
                            done = FILLED; out_kind[j] = 1
                        else:
                            out_viol[j] += 1
                if done:
                    out_status[j] = done
                    if done == FILLED:
                        out_fill[j] = t
                    n_active -= 1; active[k] = active[n_active]
                else:
                    k += 1
            if a == _F and oid[i] in slot:
                s_pend[slot[oid[i]]] += size[i]
            continue
        o = oid[i]
        sd = 1 if side[i] == _B else -1
        if a == _A:
            free += 1; slot[o] = free; counter += 1
            s_prio[free] = counter; s_side[free] = sd; s_price[free] = price[i]; s_size[free] = size[i]
            lvl = bid_lvl if sd > 0 else ask_lvl
            lvl[price[i]] = (lvl[price[i]] if price[i] in lvl else 0) + size[i]
        elif a == _C:
            if o in slot:
                q = slot[o]; removed = min(size[i], s_size[q])
                settled = min(removed, s_pend[q]); s_pend[q] -= settled
                for k in range(n_active):
                    j = active[k]
                    if p_side[j] == s_side[q] and p_price[j] == s_price[q] and s_prio[q] < prio[j]:
                        ahead[j] -= removed; consumed[j] -= settled
                lvl = bid_lvl if s_side[q] > 0 else ask_lvl
                lvl[s_price[q]] = lvl[s_price[q]] - removed
                s_size[q] -= removed
                if s_size[q] == 0:
                    slot.pop(o)
        elif a == _M:
            if o in slot:
                q = slot[o]
                keeps = price[i] == s_price[q] and 0 < size[i] <= s_size[q]
                for k in range(n_active):
                    j = active[k]
                    if p_side[j] == s_side[q] and p_price[j] == s_price[q] and s_prio[q] < prio[j]:
                        ahead[j] -= s_size[q]; consumed[j] -= s_pend[q]
                        if keeps:
                            ahead[j] += size[i]
                s_pend[q] = 0
                lvl = bid_lvl if s_side[q] > 0 else ask_lvl
                lvl[s_price[q]] = lvl[s_price[q]] - s_size[q]
                if size[i] <= 0:
                    slot.pop(o); continue
                if not keeps:
                    counter += 1; s_prio[q] = counter  # priority lost
                s_price[q] = price[i]; s_size[q] = size[i]; s_side[q] = sd
            elif size[i] > 0:
                free += 1; slot[o] = free; counter += 1; q = free
                s_prio[q] = counter; s_side[q] = sd; s_price[q] = price[i]; s_size[q] = size[i]
            else:
                continue
            lvl = bid_lvl if sd > 0 else ask_lvl
            lvl[price[i]] = (lvl[price[i]] if price[i] in lvl else 0) + size[i]
    for k in range(n_active):
        j = active[k]
        if out_status[j] == EXPIRED and p_end[j] > ts[n - 1]:
            out_status[j] = CENSORED; out_fill[j] = ts[n - 1]  # tape ended while working


def probe(tape, place_ns, side, price_nano, end_ns):
    """Probes in any order; returns status, fill_ns, kind, queue size at placement."""
    place_ns = np.asarray(place_ns, np.int64); order = np.argsort(place_ns, kind='stable')
    m = len(place_ns)
    status = np.full(m, UNPLACED, np.int8); fill = np.zeros(m, np.int64)
    kind = np.zeros(m, np.int8); ahead = np.zeros(m, np.int64); viol = np.zeros(m, np.int64)
    if m and len(tape['ts']):
        _run(tape['ts'], tape['action'], tape['side'], tape['price'], tape['size'], tape['order_id'], tape['flags'],
             place_ns[order], np.asarray(side, np.int64)[order], np.asarray(price_nano, np.int64)[order],
             np.asarray(end_ns, np.int64)[order], status, fill, kind, ahead, viol)
    inv = np.empty(m, np.int64); inv[order] = np.arange(m)
    probe.violations = viol[inv]
    return status[inv], fill[inv], kind[inv], ahead[inv]


def capture_tapes(archive, member, captures, block_records=1_048_576):
    """Stream one month member once; yield (capture_key, compact tape) as each
    selected capture completes. Selection reproduces staging: route and the
    capture's member-global ordinal range."""
    wanted = sorted(captures, key=lambda c: c['first_ordinal'])
    parts = {c['key']: [] for c in wanted}; ordinal = 0; done = set()
    for block in GlbxZipMboSource(archive, member, block_records=block_records):
        lo, hi = ordinal, ordinal + len(block)
        for c in wanted:
            if c['key'] in done or c['last_ordinal'] < lo or c['first_ordinal'] >= hi:
                continue
            a, b = max(c['first_ordinal'], lo) - lo, min(c['last_ordinal'] + 1, hi) - lo
            r = block[a:b]; p, ins, ch = c['route']
            parts[c['key']].append(r[(r['publisher_id'] == p) & (r['instrument_id'] == ins) & (r['channel_id'] == ch)])
            if c['last_ordinal'] < hi:
                rec = np.concatenate(parts.pop(c['key'])); done.add(c['key'])
                if len(rec) != c['records']:
                    raise ValueError(f"Capture record count mismatch {c['key']}: {len(rec)} != {c['records']}")
                yield c['key'], compact(rec)
        ordinal = hi
        if len(done) == len(wanted):
            return
    if len(done) != len(wanted):
        raise ValueError('Member ended before all captures were read')
