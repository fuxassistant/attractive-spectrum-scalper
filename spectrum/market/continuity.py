"""Verified handoff across file snapshots, without inventing a quiet EOF suffix.

A daily snapshot may repeat the identical complete order book. Only a verified
identity permits retaining history. Real recovery, gaps and differing books do
not qualify. The historical source ordinal is namespaced by the adapter; source
timestamps and sequence numbers remain untouched.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from spectrum.core import MboStream
from spectrum.core.schema import SNAP_SCHEMA, BAR_SCHEMA, NS, COUNTERS
from spectrum.market.replay import MarketReplay
from spectrum.core.schema import digest


def book_identity(engine):
    return digest([[oid, o.side, o.price_nano, o.size] for oid, o in sorted(engine.orders.items())])


@dataclass
class Handoff:
    seconds: pd.DataFrame
    bars: pd.DataFrame
    proof: dict


def verified_handoff(checkpoint, records, *, next_config, boundary_ns, prefix_end_ns):
    """Replay a bounded next-capture prefix from both reconstructed books.

    ``records`` starts at its declared source R snapshot and includes a complete
    publication strictly after prefix_end_ns. The caller binds both sources to
    their immutable hashes. No source gap may be hidden by this operation.
    """
    if prefix_end_ns < boundary_ns + 3*NS:
        raise ValueError('Need a prefix exceeding the 2s replenishment memory')
    carried_rows, carried_bars, fresh_rows = [], [], []
    carried = MarketReplay.restore(checkpoint, on_snapshot=carried_rows.append, on_bar=carried_bars.append)
    old = carried.core
    if old.config.route != next_config.route or old.config.snapshot_ns != next_config.snapshot_ns:
        raise ValueError('Handoff route/clock mismatch')
    if old.engine.pending or old.last_record_ns is None or not old.accumulator.book.valid:
        raise ValueError('Previous book is not a complete supported prefix')
    if not 0 < boundary_ns-old.last_record_ns < old.config.max_quote_age_ns:
        raise ValueError('A source gap or trading closure cannot be joined')
    fresh = MboStream(next_config, features=False, on_snapshot=fresh_rows.append)
    original_epoch = int(old.accumulator.epoch)
    ordinal_offset = old.engine.last_source_ordinal + 1
    bootstrap = 0
    accepted = False
    stopped = False
    last = None
    bootstrap_last = None
    initial_identity = book_identity(old.engine)
    for r in records:
        if not accepted and r.is_snapshot:
            if (bootstrap == 0 and r.action != 'R') or r.ts_recv_ns != boundary_ns:
                raise ValueError('Expected an explicit same-time file snapshot')
            fresh.feed(r)
            bootstrap += 1
            bootstrap_last = r
            continue
        if not accepted:
            if not bootstrap or fresh.engine.pending or not fresh.engine.book_ready:
                raise ValueError('Next snapshot is not complete')
            if initial_identity != book_identity(fresh.engine):
                raise ValueError('Full order book differs across file boundary')
            if r.ts_recv_ns <= boundary_ns:
                raise ValueError('Need a source frontier strictly after the boundary')
            accepted = True
        fresh.feed(r)
        carried.feed(replace(r, source_ordinal=r.source_ordinal+ordinal_offset))
        last = r
        if r.ts_recv_ns > prefix_end_ns and not fresh.engine.pending and not old.engine.pending:
            stopped = True
            break
    if not stopped or not accepted:
        raise ValueError('Incomplete handoff evidence; never flush at EOF')
    if book_identity(old.engine) != book_identity(fresh.engine):
        raise ValueError('Continuation diverged from the independently rebuilt book')
    a = pd.DataFrame(carried_rows, columns=[n for n,_ in SNAP_SCHEMA])
    b = pd.DataFrame(fresh_rows, columns=[n for n,_ in SNAP_SCHEMA])
    if a.epoch.nunique()!=1 or b.epoch.nunique()!=1:
        raise ValueError('Real support break inside the verification prefix')
    common = a.merge(b, on='t_ns', suffixes=('_carry','_reset'))
    # By the end of the prefix, finite refill memory and both books must agree.
    tail = common[common.t_ns >= prefix_end_ns-2*NS]
    if len(tail) < 2 or not tail.valid_carry.all() or not tail.valid_reset.all():
        raise ValueError('Handoff verification tail lacks support')
    columns = [n for n,_ in SNAP_SCHEMA if n not in ('t_ns','epoch') and not n.startswith('c_')]
    for n in columns:
        if not np.allclose(tail[n+'_carry'],tail[n+'_reset'],rtol=0,atol=1e-10,equal_nan=True):
            raise ValueError('Independent prefix disagrees: '+n)
    offsets = {}
    for n in COUNTERS:
        name = 'c_'+n
        delta = tail[name+'_carry'].to_numpy()-tail[name+'_reset'].to_numpy()
        if not np.allclose(delta,delta[-1],rtol=0,atol=1e-7):
            raise ValueError('Counter offset did not stabilize: '+name)
        offsets[name] = float(delta[-1])
    proof = dict(version='IDENTICAL_BOOK_HANDOFF_V1',boundary_ns=int(boundary_ns),
                 available_ns=int(last.ts_recv_ns),prefix_end_ns=int(prefix_end_ns),
                 book_sha256=initial_identity,bootstrap_records=bootstrap,
                 accepted_available_ns=int(bootstrap_last.ts_recv_ns),
                 bootstrap_last_source_ordinal=int(bootstrap_last.source_ordinal),
                 previous_epoch=original_epoch,next_epoch=int(b.epoch.iloc[0]),
                 ordinal_offset=int(ordinal_offset),last_prefix_source_ordinal=int(last.source_ordinal),
                 counter_offsets=offsets,book_end_sha256=book_identity(old.engine),
                 endpoint_recovered=bool(a.t_ns.eq(boundary_ns).sum()==1))
    return Handoff(a,pd.DataFrame(carried_bars,columns=[n for n,_ in BAR_SCHEMA]),proof)
