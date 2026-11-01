"""Causal price-local MBO evidence, separate from intentions or future outcomes.

The existing sidecars retain price/side/second activity and exact T records.
They do NOT retain mutation action, order identity, repricing, level stock, or
per-row support epochs. A missing row means no retained activity, not zero depth.
Use event_evidence on complete source publications when lifecycle detail matters.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.core.event_facts import decode_event
from spectrum.representation.raw import FIELDS

NS = 1_000_000_000
TICK_NANO = 100_000_000
LEVEL_CHANNELS = (*FIELDS, "linked_unknown_role_fill")
TRADE_CHANNELS = ("buy_trade_qty", "sell_trade_qty", "unknown_trade_qty")
SOURCES = (
    "https://databento.com/docs/conventions/common-fields-enums-types",
    "https://databento.com/docs/datasets/glbx-mdp3",
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ns(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    raise TypeError("Use integer UTC nanoseconds; floating timestamps lose precision")


def _level_checks(levels):
    if not set(FIELDS).issubset(levels.columns):
        raise ValueError("Missing original interaction channels")
    amounts = levels[list(FIELDS)].to_numpy(float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Interaction quantities must be finite and nonnegative")
    if not levels.side.isin([-1, 1]).all():
        raise ValueError("Level side must identify bid or ask")
    if (levels.available_ns < levels.t_ns).any():
        raise ValueError("A completed second cannot be available before its end")
    residual = levels.remove - levels.linked_fill - levels.ambiguous_remove - levels.unlinked_remove
    if (residual.abs() > 1e-8).any():
        raise ValueError("Removal partition does not reconcile")
    role_residual = levels.linked_fill - levels.passive_fill - levels.aggressor_fill
    if (role_residual < -1e-8).any():
        raise ValueError("Fill roles exceed linked quantity")
    if (levels.refill > levels["add"] + 1e-8).any():
        raise ValueError("Refill exceeds observed addition")
    return role_residual.clip(lower=0.)


def load_evidence(raw_dir, start_ns, end_ns, asof_ns=None):
    """Read verified sidecars for physical interval [start,end), known by asof.

    Level t_ns labels the END of [t_ns-1s,t_ns). Only fully contained buckets
    are retained, so a fractional-second left boundary cannot leak a prefix.
    Completed buckets may arrive after their physical end; availability is
    checked independently. Exact trades keep source/publication ordinals.
    Returns ``(levels, trades, receipt)`` without changing any source artifact.
    """
    start, end = _ns(start_ns), _ns(end_ns)
    cutoff = end if asof_ns is None else _ns(asof_ns)
    if end <= start or cutoff < start:
        raise ValueError("Positive interval and asof at or after start required")
    raw = Path(raw_dir)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Only COMPLETE sidecars can supply evidence")
    if manifest.get("stats", {}).get("max_stock_residual") != 0:
        raise ValueError("Source stock reconciliation was not verified")
    hashes = {}
    for name in ("levels_1s.parquet", "trades.parquet"):
        expected = manifest.get("outputs", {}).get(name)
        actual = _sha(raw / name)
        if not expected or expected != actual:
            raise ValueError(f"Sidecar identity mismatch: {name}")
        hashes[name] = actual
    levels = pq.read_table(raw / "levels_1s.parquet", filters=[
        ("t_ns", ">=", start + NS), ("t_ns", "<=", end),
        ("available_ns", "<=", cutoff)]).to_pandas()
    trades = pq.read_table(raw / "trades.parquet", filters=[
        ("t_ns", ">=", start), ("t_ns", "<", end),
        ("available_ns", "<=", cutoff)]).to_pandas()
    levels["linked_unknown_role_fill"] = _level_checks(levels)
    if (trades.available_ns < trades.t_ns).any() or not trades.side.isin([-1, 0, 1]).all():
        raise ValueError("Invalid trade clock or side")
    if (trades.quantity <= 0).any() or not np.isfinite(trades.quantity.to_numpy(float)).all():
        raise ValueError("Trades require positive finite quantity")
    if trades.ordinal.duplicated().any():
        raise ValueError("Trade source ordinals must be unique; do not count F again")
    for side, name in zip((1, -1, 0), TRADE_CHANNELS):
        trades[name] = np.where(trades.side == side, trades.quantity, 0)
    receipt = {
        "status": "VERIFIED_SIDECAR_WINDOW", "raw_dir": str(raw.resolve()),
        "start_ns": start, "end_ns": end, "asof_ns": cutoff,
        "manifest_sha256": _sha(manifest_path), "payload_sha256": hashes,
        "source": manifest.get("source"), "route": manifest.get("route"),
        "level_rows": len(levels), "trade_rows": len(trades),
        "trade_qty": int(trades.quantity.sum()),
        "unsupported_trade_rows": int((~trades.supported).sum()),
        "unsupported_trade_qty": int(trades.loc[~trades.supported, "quantity"].sum()),
        "max_available_ns": max(int(levels.available_ns.max()) if len(levels) else 0,
                                int(trades.available_ns.max()) if len(trades) else 0),
        "semantics": {
            "trade_volume": "T counted once; F is attribution, never additional traded volume",
            "unlinked_remove": "Decrement without a usable linked F; not proof of cancellation intent or no execution",
            "linked_roles": "Passive/aggressor-existing/unknown are disjoint parts of linked fill",
            "levels": "Price-local activity, not resting depth; absence is not zero depth",
            "repricing": "NOT_RETAINED in sidecars; requires full publication mutations",
            "support": "Stock reconciliation verified globally; per-level-row epoch and price support NOT_RETAINED",
            "bucket": "Only fully contained physical seconds and available<=asof are retained",
            "refill": "Same-price additions matched to earlier linked fills within2s, not same participant",
        },
        "provider_sources": list(SOURCES),
    }
    return levels, trades, receipt


def summarize_price_evidence(levels, trades, bin_ticks=5):
    """Absolute anchored bins: channels coexist; they do not establish causation.

    In particular bid_passive_fill is resting-bid execution evidence, whereas
    bid_aggressor_fill is an already-resting buy order acting as aggressor.
    trade_sell_qty is reported sell-aggressor T volume and must not be added to F.
    """
    if not isinstance(bin_ticks, (int, np.integer)) or isinstance(bin_ticks, bool) or bin_ticks <= 0:
        raise ValueError("Positive integer tick bin width required")
    width = int(bin_ticks) * TICK_NANO
    bins = np.union1d(levels.price_nano.to_numpy(np.int64) // width * width,
                      trades.price_nano.to_numpy(np.int64) // width * width)
    out = pd.DataFrame(index=pd.Index(bins, name="price_bin_nano"))
    level_work = levels.copy()
    level_work["linked_unknown_role_fill"] = _level_checks(level_work)
    level_work["price_bin_nano"] = level_work.price_nano.to_numpy(np.int64) // width * width
    for side, prefix in ((1, "bid"), (-1, "ask")):
        grouped = level_work.loc[level_work.side == side].groupby("price_bin_nano")[list(LEVEL_CHANNELS)].sum()
        for col in LEVEL_CHANNELS:
            out[prefix + "_" + col] = grouped[col].reindex(out.index).fillna(0.)
    trade_work = trades.copy()
    trade_work["price_bin_nano"] = trade_work.price_nano.to_numpy(np.int64) // width * width
    for side, name in ((1, "trade_buy_qty"), (-1, "trade_sell_qty"), (0, "trade_unknown_qty")):
        amount = trade_work.loc[trade_work.side == side].groupby("price_bin_nano").quantity.sum()
        out[name] = amount.reindex(out.index).fillna(0).astype(np.int64)
    bad = trade_work.loc[~trade_work.supported].groupby("price_bin_nano").quantity.sum()
    out["trade_unsupported_qty"] = bad.reindex(out.index).fillna(0).astype(np.int64)
    availability = pd.concat([level_work[["price_bin_nano", "available_ns"]],
                              trade_work[["price_bin_nano", "available_ns"]]])
    out["source_max_available_ns"] = availability.groupby("price_bin_nano").available_ns.max().reindex(out.index).astype(np.int64)
    out["price_low"] = out.index.to_numpy() / 1e9
    out["price_high"] = (out.index.to_numpy() + width) / 1e9
    return out.reset_index()


def event_evidence(event, received_at_ns=None):
    """Preserve lifecycle evidence from ONE completed F_LAST publication.

    Raw actions alone cannot distinguish execution from cancellation: the book
    is mutated by A/M/C while T/F separately report executions. Price/lifecycle
    links use the existing audited decode_event. Unsupported attributions are
    None, not zero. This function never infers participant intention.
    """
    if not event.deltas or not event.deltas[-1].flags & 128:
        raise ValueError("Only complete F_LAST publications can be interpreted")
    available = int(event.available_at_ns)
    if received_at_ns is not None:
        received = _ns(received_at_ns)
        if received < available:
            raise ValueError("Delivery clock cannot precede source availability")
        available = received
    facts = decode_event(event)
    by_ordinal = {d.source_ordinal: d for d in event.deltas}
    metadata = dict(event_uid=event.event_uid, available_ns=available,
                    source_available_ns=int(event.available_at_ns), event_ts_ns=int(event.event_ts_ns),
                    first_source_ordinal=int(event.first_source_ordinal),
                    last_source_ordinal=int(event.last_source_ordinal),
                    epoch=int(event.book.continuity_epoch), stock_supported=bool(facts.supported),
                    price_supported=bool(facts.price_supported),
                    before_bid_nano=int(event.book_before.best_bid_nano),
                    before_ask_nano=int(event.book_before.best_ask_nano),
                    after_bid_nano=int(event.book.best_bid_nano),
                    after_ask_nano=int(event.book.best_ask_nano),
                    unresolved_fill_qty=int(facts.unresolved_fill))
    rows = []
    for m in facts.mutations:
        d = by_ordinal[m.ordinal]
        row = dict(asdict(m), available_ns=available, epoch=metadata["epoch"],
                   stock_supported=metadata["stock_supported"], price_supported=metadata["price_supported"],
                   prior_price_nano=int(d.prior_price_nano), resulting_price_nano=int(d.resulting_price_nano),
                   prior_size=int(d.prior_size), resulting_size=int(d.resulting_size))
        known = facts.supported and m.accepted
        remove = max(0, -m.quantity)
        unlinked = remove - m.linked_fill - m.ambiguous_remove
        if known and (unlinked < 0 or m.passive_fill + m.aggressor_fill > m.linked_fill):
            raise ValueError("Event removal attribution does not reconcile")
        row.update(attribution_known=bool(known), added_qty=max(0, m.quantity), removed_qty=remove,
                   linked_unknown_role_fill=m.linked_fill-m.passive_fill-m.aggressor_fill if known else None,
                   unlinked_remove=unlinked if known else None,
                   reprice_out_qty=remove if m.kind == "REPRICE" else 0,
                   reprice_in_qty=max(0, m.quantity) if m.kind == "REPRICE" else 0,
                   unlinked_reprice_out_qty=unlinked if known and m.kind == "REPRICE" else (0 if known else None),
                   unlinked_cancel_qty=unlinked if known and m.action == "C" and m.kind != "REPRICE" else (0 if known else None),
                   unlinked_modify_qty=unlinked if known and m.action == "M" and m.kind != "REPRICE" else (0 if known else None))
        if not known:
            for key in ("linked_fill", "passive_fill", "aggressor_fill", "ambiguous_remove"):
                row[key] = None
        rows.append(row)
    trades = [dict(asdict(t), available_ns=available, epoch=metadata["epoch"],
                   stock_supported=metadata["stock_supported"], price_supported=metadata["price_supported"])
              for t in facts.trades]
    fills = [dict(asdict(f), available_ns=available, attribution_known=metadata["stock_supported"])
             for f in facts.fills]
    if not facts.supported:
        for f in fills:
            f["role"], f["linked"] = "UNKNOWN", None
    links = [dict(fill_ordinal=f, mutation_ordinal=m, side=s, price_nano=p,
                  quantity=q, role=role, available_ns=available,
                  attribution_known=metadata["stock_supported"])
             for f, m, s, p, q, role in facts.links]
    return {"event": metadata, "mutations": rows, "trades": trades, "fills": fills, "links": links}


def _apply_subset_record(engine, record):
    """Reuse verified per-order math; no subset-book BBO is ever published."""
    if record.action == "R":
        engine._clear_book()
    return engine._apply(record)


def _canonical_fragment(row):
    from spectrum.core.vendor.l3 import BookFragment
    bid, ask = int(row.bid_nano), int(row.ask_nano)
    bq, aq = int(row.bid_size), int(row.ask_size)
    return BookFragment(int(row.epoch), bool(row.valid),
                        "OK" if row.valid else "CANONICAL_PRICE_UNSUPPORTED",
                        bid, ask, bq, aq, (bid + ask) // 2, 0, 0., -1)


def _complete_subset_event(deltas, previous_quote, quote, partition_sha256):
    """Attach independently verified full-book BBO to exact selected-order deltas."""
    from spectrum.core.vendor.l3 import CompletedMboEvent
    last = deltas[-1]
    if not last.flags & 128 or last.source_ordinal != int(quote.source_ordinal):
        raise ValueError("Raw group does not end at canonical publication ordinal")
    if last.ts_recv_ns != int(quote.available_ns):
        raise ValueError("Raw publication clock differs from canonical quote clock")
    before, after = _canonical_fragment(previous_quote), _canonical_fragment(quote)
    changed = (before.best_bid_nano, before.best_ask_nano, before.bid_size_l1, before.ask_size_l1) != (
        after.best_bid_nano, after.best_ask_nano, after.bid_size_l1, after.ask_size_l1)
    mutations = sum(d.accepted and d.action in ("A", "M", "C") for d in deltas)
    uid = hashlib.sha256(f"SELECTED_ORDER_PREFIX_V1|{partition_sha256}|{deltas[0].source_ordinal}|{last.source_ordinal}".encode()).hexdigest()
    return CompletedMboEvent(uid, last.sequence, last.ts_event_ns, last.ts_recv_ns,
        deltas[0].source_ordinal, last.source_ordinal, tuple(deltas), before, after,
        changed, mutations, 0 if changed else mutations)


def extract_raw_scenes(raw_dir, canonical_market_dir, windows, output_dir):
    """Two raw scans reconstruct complete publications for small named scenes.

    Pass1 verifies the complete raw member/partition and retains scene groups.
    Pass2 reconstructs every order referenced in those groups from the explicit
    UTC reset through each scene. Other orders cannot change these order states.
    All window records are kept, and full-book pre/post BBO comes from the hashed
    canonical quote tape. No absolute level depth is inferred from this subset.

    windows: dictionaries with id,start_ns,end_ns,asof_ns. Selection is by COMPLETE
    publication availability in [start_ns,asof_ns], inclusive at asof; end_ns is
    the annotated physical scene end. Thus delayed closing publications remain
    observable at asof and are never backdated to the nominal minute boundary.
    """
    import time
    from spectrum.core.vendor.l3_v2 import L3EventEngineV2
    from spectrum.market.ingest import PACKED, packed_records
    from spectrum.market.source import DbnSource

    began = time.perf_counter()
    raw, canonical, out = Path(raw_dir), Path(canonical_market_dir), Path(output_dir)
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    cm_path = canonical / "manifest.json"
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or cm.get("status") != "COMPLETE":
        raise ValueError("Verified COMPLETE raw and canonical manifests required")
    source = manifest["source"]
    if source["partition_sha256"] != cm["source_partition_sha256"]:
        raise ValueError("Raw sidecar and canonical state are from different partitions")
    quote_hash = _sha(canonical / "quotes.parquet")
    if quote_hash != cm["payloads"]["quotes.parquet"]:
        raise ValueError("Canonical quote payload identity mismatch")
    route = tuple(manifest["route"])
    if tuple(cm["input"]["route"]) != route:
        raise ValueError("Canonical route differs from original source route")
    specs = []
    for w in windows:
        name = w["id"]
        if not isinstance(name, str) or not name or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in name):
            raise ValueError("Scene id must be a safe simple name")
        start, end, cutoff = (_ns(w[k]) for k in ("start_ns", "end_ns", "asof_ns"))
        if not start < end <= cutoff:
            raise ValueError("Scene start < physical end <= asof required")
        specs.append(dict(id=name, start_ns=start, end_ns=end, asof_ns=cutoff))
    specs.sort(key=lambda w: w["start_ns"])
    if not specs or len({w["id"] for w in specs}) != len(specs):
        raise ValueError("At least one uniquely named scene required")
    if any(a["asof_ns"] >= b["start_ns"] for a, b in zip(specs, specs[1:])):
        raise ValueError("Scenes must not overlap")
    q = pq.read_table(canonical / "quotes.parquet", columns=["available_ns", "source_ordinal",
        "epoch", "bid_nano", "ask_nano", "bid_size", "ask_size", "valid"],
        filters=[("available_ns", "<=", specs[-1]["asof_ns"])]).to_pandas()
    clocks = q.available_ns.to_numpy(np.int64)
    for w in specs:
        lo, hi = (int(np.searchsorted(clocks, w["start_ns"], "left")),
                  int(np.searchsorted(clocks, w["asof_ns"], "right")))
        if lo == 0 or hi <= lo:
            raise ValueError("Scene requires a known preceding publication and at least one complete event")
        w.update(previous=q.iloc[lo - 1], quotes=q.iloc[lo:hi].reset_index(drop=True),
                 lower_ordinal=int(q.iloc[lo - 1].source_ordinal),
                 upper_ordinal=int(q.iloc[hi - 1].source_ordinal), blocks=[])
    first, last = int(source["first_ordinal"]), int(source["last_ordinal"])

    def selected_blocks(src):
        offset = 0
        for block in src:
            ids = np.flatnonzero((block["publisher_id"] == route[0]) &
                (block["instrument_id"] == route[1]) & (block["channel_id"] == route[2]))
            ids = ids[(ids + offset >= first) & (ids + offset <= last)]
            packed = np.empty(len(ids), dtype=PACKED)
            packed["source_ordinal"], packed["record"] = ids + offset, block[ids]
            offset += len(block)
            yield packed

    scan_times = []
    s1 = DbnSource(source["archive"], source["member"])
    digest, n = hashlib.sha256(), 0
    scan_start = time.perf_counter()
    for packed in selected_blocks(s1):
        digest.update(packed.tobytes()); n += len(packed)
        for w in specs:
            selected = packed[(packed["source_ordinal"] > w["lower_ordinal"]) &
                              (packed["source_ordinal"] <= w["upper_ordinal"])]
            if len(selected):
                w["blocks"].append(selected.copy())
    scan_times.append(time.perf_counter() - scan_start)
    if s1.sha256 != source["member_sha256"] or digest.hexdigest() != source["partition_sha256"] or n != source["records"]:
        raise ValueError("Full original DBN member or selected partition identity mismatch")
    for w in specs:
        w["raw"] = np.concatenate(w.pop("blocks"))
    targets = np.unique(np.concatenate([w["raw"]["record"]["order_id"] for w in specs]))
    engine = L3EventEngineV2(publisher_id=route[0], instrument_id=route[1], channel_id=route[2])
    results = {w["id"]: dict(events=[], mutations=[], trades=[], fills=[], links=[], records=[]) for w in specs}
    cursors, pending = {w["id"]: 0 for w in specs}, {w["id"]: [] for w in specs}
    prefix_updates = 0
    s2 = DbnSource(source["archive"], source["member"])
    digest2, n2 = hashlib.sha256(), 0
    scan_start = time.perf_counter()
    for packed in selected_blocks(s2):
        digest2.update(packed.tobytes()); n2 += len(packed)
        rec, ordinals = packed["record"], packed["source_ordinal"]
        inside = np.zeros(len(packed), dtype=bool)
        for w in specs:
            inside |= (ordinals > w["lower_ordinal"]) & (ordinals <= w["upper_ordinal"])
        mutate = np.isin(rec["action"], [b"A", b"M", b"C"])
        relevant = ((np.isin(rec["order_id"], targets) & mutate) | (rec["action"] == b"R"))
        keep = (inside | relevant) & (ordinals <= specs[-1]["upper_ordinal"])
        for record in packed_records(packed[keep], route):
            delta = _apply_subset_record(engine, record)
            window = next((w for w in specs if w["lower_ordinal"] < record.source_ordinal <= w["upper_ordinal"]), None)
            if window is None:
                prefix_updates += 1
                continue
            name = window["id"]
            results[name]["records"].append(asdict(record))
            pending[name].append(delta)
            if not record.is_last:
                continue
            index = cursors[name]
            quote = window["quotes"].iloc[index]
            previous = window["previous"] if index == 0 else window["quotes"].iloc[index - 1]
            event = _complete_subset_event(pending[name], previous, quote, source["partition_sha256"])
            evidence = event_evidence(event)
            results[name]["events"].append(dict(evidence["event"], window_id=name))
            for kind in ("mutations", "trades", "fills", "links"):
                results[name][kind].extend(dict(row, window_id=name, event_uid=event.event_uid,
                    first_source_ordinal=event.first_source_ordinal, last_source_ordinal=event.last_source_ordinal)
                    for row in evidence[kind])
            pending[name].clear(); cursors[name] += 1
    scan_times.append(time.perf_counter() - scan_start)
    if s2.sha256 != s1.sha256 or digest2.hexdigest() != digest.hexdigest() or n2 != n:
        raise ValueError("Original source changed between the two verified scans")
    if any(pending.values()) or any(cursors[w["id"]] != len(w["quotes"]) for w in specs):
        raise ValueError("Scene publication count or F_LAST completeness differs")
    out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for w in specs:
        destination = out / w["id"]
        destination.mkdir()
        tables = {k: pd.DataFrame(v) for k, v in results[w["id"]].items()}
        if tables["trades"].ordinal.duplicated().any():
            raise ValueError("Duplicate source T record")
        if tables["mutations"][["ordinal", "side", "price_nano", "quantity"]].duplicated().any():
            raise ValueError("Duplicate source mutation")
        files = {}
        for kind, table in tables.items():
            path = destination / (kind + ".parquet")
            table.to_parquet(path, index=False)
            files[path.name] = _sha(path)
        known = tables["mutations"].loc[tables["mutations"].attribution_known]
        bid = known.loc[(known.side == "B") & known.removed_qty.gt(0)]
        trade = tables["trades"]
        receipt = {k: w[k] for k in ("id", "start_ns", "end_ns", "asof_ns", "lower_ordinal", "upper_ordinal")}
        receipt.update(status="COMPLETE", publication_count=len(tables["events"]),
            raw_records=len(tables["records"]), first_source_ordinal=int(w["raw"]["source_ordinal"][0]),
            max_available_ns=int(tables["events"].available_ns.max()),
            unsupported_events=int((~tables["events"].stock_supported).sum()),
            unsupported_price_events=int((~tables["events"].price_supported).sum()),
            source_ordinal_unique=True, all_f_last_groups_complete=True,
            trade_qty=int(trade.quantity.sum()),
            sell_trade_qty=int(trade.loc[trade.side == "A", "quantity"].sum()),
            buy_trade_qty=int(trade.loc[trade.side == "B", "quantity"].sum()),
            unknown_trade_qty=int(trade.loc[~trade.side.isin(["A", "B"]), "quantity"].sum()),
            bid_removed_qty=int(bid.removed_qty.sum()), bid_passive_fill_qty=int(bid.passive_fill.sum()),
            bid_aggressor_existing_fill_qty=int(bid.aggressor_fill.sum()),
            bid_linked_unknown_role_qty=int(bid.linked_unknown_role_fill.sum()),
            bid_ambiguous_remove_qty=int(bid.ambiguous_remove.sum()),
            bid_unlinked_cancel_qty=int(bid.unlinked_cancel_qty.sum()),
            bid_unlinked_modify_qty=int(bid.unlinked_modify_qty.sum()),
            bid_unlinked_reprice_qty=int(bid.unlinked_reprice_out_qty.sum()),
            payload_sha256=files, full_depth_reconstructed=False,
            semantics="All scene records and source order histories retained; full-book BBO from canonical quote tape. Absolute depth unavailable; T volume is not added to linked F removals. Unlinked action classification is source mechanics, not participant intent.")
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipts.append(receipt)
    receipt = dict(status="COMPLETE", contract="SELECTED_ORDER_PREFIX_RECONSTRUCTION_V1",
        source=source, route=list(route), canonical_manifest_sha256=_sha(cm_path), canonical_quotes_sha256=quote_hash,
        raw_sidecar_manifest_sha256=_sha(raw / "manifest.json"), full_member_verified_twice=True,
        selected_partition_verified_twice=True, selected_order_ids=len(targets), prefix_order_updates=prefix_updates,
        pass_seconds=scan_times, elapsed_seconds=time.perf_counter()-began,
        full_depth_reconstructed=False, windows=receipts,
        implementation_sha256={str(p): _sha(p) for p in [Path(__file__),
            Path(decode_event.__code__.co_filename), Path(engine._apply.__code__.co_filename)]})
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
