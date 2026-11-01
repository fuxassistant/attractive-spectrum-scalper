"""Separate descriptive price-local evidence; never a trained feature family."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .mbo_features import GlobalMBOContext, NS, TICK_NANO, _ns, _clock, _sha

LOCAL_LEVEL_FIELDS = ("add", "remove", "linked_fill", "passive_fill", "aggressor_fill",
                      "linked_unknown_role_fill", "ambiguous_remove", "unlinked_remove", "refill")
LOCAL_VALUE_COLUMNS = (
    "observed_trade_buy_qty", "observed_trade_sell_qty", "observed_trade_unknown_qty",
    "observed_trade_toward_qty", "observed_trade_away_qty", "observed_trade_total_qty",
    *("observed_"+side+"_"+name+"_qty" for side in ("support", "opposition")
      for name in LOCAL_LEVEL_FIELDS),
)


def local_mbo_diagnostics(candidates, context, raw_dir=None, *, expected_route,
                          expected_source_partition_sha256):
    """Price-local descriptive rows, deliberately excluded from feature exports.

    Sparse zero is allowed only after independent native support and whole-price
    conservation against canonical counters. Level buckets describe [start,end)
    whereas canonical differences describe (start,end]; native publications at
    either exact boundary prevent the level-conservation proof. Exact T uses
    (start,end] and is reconciled by side and record count. Raw and canonical
    epoch identifiers are different namespaces; only within-source continuity
    is compared. Missing sidecar dates return explicit unknown rows.

    Optional ``departure_mid_nano`` adds a fixed corridor from the zone to the
    confirmed departure point. It is not recomputed using future/current price.
    Returns (long_dataframe, source_receipt), never updates an existing source.
    """
    from experiments.price_level_memory_v1.mbo_evidence import _level_checks

    if not isinstance(context, GlobalMBOContext):
        raise TypeError("Use a GlobalMBOContext for the identical canonical route/day")
    # Also applies candidate route-availability and causal-history guards.
    context.candidates(candidates)
    source_present = raw_dir is not None
    receipt = dict(status="MISSING_LOCAL_SIDECARS", source_present=source_present,
                   local_training_admission="NOT_INCLUDED_IN_MODEL",
                   fields=list(LOCAL_VALUE_COLUMNS),
                   semantics={
                       "T": "Exact T counted once; F is not additional trade volume",
                       "F": "Passive/active/unknown roles are disjoint attribution of linked removals",
                       "unlinked": "Removal without usable F link; includes no claim about cancellation intent",
                       "refill": "Capped same-price addition after earlier linked fill within2s",
                       "zero": "Requires native support, publication completeness, and all-price conservation",
                       "unknown": "Missing source or failed completeness proof is never an assumed zero",
                   })
    levels = trades = None
    if source_present:
        raw = Path(raw_dir); manifest = json.loads((raw/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or manifest.get("stats", {}).get("max_stock_residual") != 0:
            raise ValueError("Local sidecar source must be COMPLETE with reconciled stock")
        if list(manifest.get("route", [])) != list(expected_route):
            raise ValueError("Local sidecar route differs from canonical route")
        if manifest.get("source", {}).get("partition_sha256") != expected_source_partition_sha256:
            raise ValueError("Local and canonical source partitions differ")
        hashes = {}
        for name in ("levels_1s.parquet", "trades.parquet"):
            hashes[name] = _sha(raw/name)
            if hashes[name] != manifest.get("outputs", {}).get(name):
                raise ValueError("Local sidecar payload hash mismatch: " + name)
        levels = pd.read_parquet(raw/"levels_1s.parquet")
        trades = pd.read_parquet(raw/"trades.parquet")
        for frame in (levels, trades):
            for name in ("t_ns", "available_ns", "price_nano"):
                _clock(frame, name)
            if (frame.available_ns < frame.t_ns).any() or (frame.price_nano <= 0).any():
                raise ValueError("Invalid local evidence clock/price")
        levels["linked_unknown_role_fill"] = _level_checks(levels)
        if levels.duplicated(["t_ns", "side", "price_nano"]).any():
            raise ValueError("Duplicate local activity bucket/price/side")
        _clock(trades, "ordinal"); _clock(trades, "epoch")
        if (trades.ordinal.duplicated().any() or not trades.side.isin([-1, 0, 1]).all()
                or not np.isfinite(trades.quantity).all() or (trades.quantity <= 0).any()):
            raise ValueError("Exact T records must be unique positive trades")
        levels = levels.sort_values("t_ns", kind="stable").reset_index(drop=True)
        trades = trades.sort_values("t_ns", kind="stable").reset_index(drop=True)
        lt = levels.t_ns.to_numpy(np.int64); tt = trades.t_ns.to_numpy(np.int64)
        receipt.update(status="VERIFIED_DESCRIPTIVE_LOCAL_SIDECARS", raw_dir=str(raw.resolve()),
                       manifest_sha256=_sha(raw/"manifest.json"), payload_sha256=hashes,
                       route=list(expected_route), source_partition_sha256=expected_source_partition_sha256,
                       level_rows=len(levels), trade_rows=len(trades),
                       helper_source_sha256=_sha(Path(__file__).parents[1]/"price_level_memory_v1"/"mbo_evidence.py"))
    rows = []
    cache = {}
    for r in candidates.itertuples():
        zone_lo, zone_hi = _ns(r.zone_low_nano), _ns(r.zone_high_nano)
        if zone_lo >= zone_hi:
            raise ValueError("Price zone must be a positive half-open interval")
        regions = [("zone", zone_lo, zone_hi)]
        if hasattr(r, "departure_mid_nano"):
            departure = _ns(r.departure_mid_nano)
            if departure <= 0:
                raise ValueError("Departure corridor needs a positive, known departure price")
            regions.append(("departure_corridor", min(zone_lo, departure), max(zone_hi, departure+TICK_NANO)))
        for tag, query in (("dep", r.episode_available_ns), ("now", r.t_ns)):
            x, meta = context.window(query, int(r.side)); start = meta["window_start_ns"]; end = meta["window_end_ns"]
            key = int(query), int(r.side)
            if key not in cache:
                l = t = None; tk = lk = False; boundaries_clear = False; released = False
                if source_present and start >= 0 and end > start:
                    l = levels.iloc[np.searchsorted(lt, start+NS, side="left"):np.searchsorted(lt, end, side="right")]
                    t = trades.iloc[np.searchsorted(tt, start, side="right"):np.searchsorted(tt, end, side="right")]
                    l = l.loc[l.available_ns <= query]; t = t.loc[t.available_ns <= query]
                    # Even an empty bucket needs a publication watermark.
                    ends = np.arange(start+NS, end+1, NS, dtype=np.int64)
                    positions = np.searchsorted(context.qt, ends, side="left")
                    released = bool((positions < len(context.qt)).all())
                    if released:
                        released = bool((context.qt[positions] <= query).all())
                    bounds = np.array([start, end], dtype=np.int64)
                    bp = np.searchsorted(context.qt, bounds, side="left")
                    inside = bp < len(context.qt)
                    boundaries_clear = not bool((context.qt[bp[inside]] == bounds[inside]).any())
                    if x["known"] and released:
                        d = meta["global_counter_deltas"]
                        totals = {side: float(t.loc[t.side == side, "quantity"].sum()) for side in (-1, 0, 1)}
                        trade_supported = (bool(t.supported.all()) and bool((t.epoch >= 0).all())
                                           and t.epoch.nunique() <= 1)
                        tk = bool(trade_supported and len(t) == d["trade_count"]
                                  and totals[1] == d["trade_buy"] and totals[-1] == d["trade_sell"]
                                  and totals[0] == d["trade_unknown"])
                        lk = bool(boundaries_clear)
                        for side, book_side in ((1, "bid"), (-1, "ask")):
                            ll = l.loc[l.side == side]
                            for raw_field, counter_field in (("add", "add_"), ("remove", "remove_"),
                                                            ("linked_fill", "matched_fill_remove_"),
                                                            ("refill", "post_trade_add_")):
                                lk &= bool(float(ll[raw_field].sum()) == d[counter_field+book_side])
                cache[key] = l, t, tk, lk, boundaries_clear, released
            l, t, tk, lk, boundaries_clear, released = cache[key]
            for region, lo, hi in regions:
                item = dict(candidate_id=int(r.candidate_id), window=tag, region=region,
                            query_ns=int(query), physical_start_ns=start, physical_end_ns=end,
                            region_low_nano=lo, region_high_nano=hi, side=int(r.side),
                            source_present=source_present, native_window_known=bool(x["known"]),
                            trade_window_known=tk, level_window_known=lk, known=tk and lk,
                            publication_complete=released, native_boundaries_clear=boundaries_clear,
                            status="KNOWN_LOCAL_ACTIVITY" if tk and lk else "MISSING_LOCAL_SOURCE" if not source_present else "INCOMPLETE_LOCAL_PROOF",
                            global_window_status=meta["status"], source_epoch=meta["source_epoch"],
                            observed_trade_rows=0, observed_level_rows=0, max_source_available_ns=-1,
                            denominator_trade_total_qty=np.nan, denominator_remove_qty=np.nan,
                            denominator_add_qty=np.nan,
                            **{name: np.nan for name in LOCAL_VALUE_COLUMNS})
                if l is not None:
                    ll = l.loc[(l.price_nano >= lo) & (l.price_nano < hi)]
                    tr = t.loc[(t.price_nano >= lo) & (t.price_nano < hi)]
                    item.update(observed_trade_rows=len(tr), observed_level_rows=len(ll),
                                max_source_available_ns=max(int(ll.available_ns.max()) if len(ll) else -1,
                                                            int(tr.available_ns.max()) if len(tr) else -1))
                    if tk or len(tr):
                        for side, label in ((1, "buy"), (-1, "sell"), (0, "unknown")):
                            subset = tr.loc[tr.side == side]
                            item[f"observed_trade_{label}_qty"] = float(subset.quantity.sum()) if tk or len(subset) else np.nan
                        if tk:
                            item["observed_trade_total_qty"] = float(tr.quantity.sum())
                            item["denominator_trade_total_qty"] = float(tr.quantity.sum())
                        item["observed_trade_toward_qty"] = item["observed_trade_buy_qty" if r.side == 1 else "observed_trade_sell_qty"]
                        item["observed_trade_away_qty"] = item["observed_trade_sell_qty" if r.side == 1 else "observed_trade_buy_qty"]
                    for side, label in ((int(r.side), "support"), (-int(r.side), "opposition")):
                        subset = ll.loc[ll.side == side]
                        if lk or len(subset):
                            for name in LOCAL_LEVEL_FIELDS:
                                item[f"observed_{label}_{name}_qty"] = float(subset[name].sum())
                    if lk:
                        item["denominator_remove_qty"] = float(ll.remove.sum())
                        item["denominator_add_qty"] = float(ll["add"].sum())
                rows.append(item)
    frame = pd.DataFrame(rows)
    receipt["rows"] = len(frame)
    receipt["known_rows"] = int(frame.known.sum()) if len(frame) else 0
    receipt["trade_known_rows"] = int(frame.trade_window_known.sum()) if len(frame) else 0
    receipt["level_known_rows"] = int(frame.level_window_known.sum()) if len(frame) else 0
    return frame, receipt


def run_local_audit(conditional_run, destination=None):
    """Materialize the separate seven-day descriptive audit without model edits."""
    import time

    begun = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    run = Path(conditional_run).resolve()
    parent_hash = _sha(run/"manifest.json")
    parent = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    if parent["status"] != "COMPLETE_DIAGNOSTIC_ONLY":
        raise ValueError("The parent conditional run must be complete")
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    source = root/config["source_run"]
    if _sha(source/"manifest.json") != parent["source_manifest_sha256"]:
        raise ValueError("Frozen episode source manifest changed")
    sm = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    destination = Path(destination).resolve() if destination else run/"local"
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []; frames = []; inventory = []
    for r in sm["receipts"]:
        date = r["date"]; day = source/"days"/date
        cp = day/"candidates.parquet"
        if _sha(cp) != r["artifact_sha256"]["candidates.parquet"]:
            raise ValueError("Frozen candidate identity changed")
        c = pd.read_parquet(cp); c = c.loc[~c.inside].copy()
        options = [root/"data/derived"/name/"raw"/date for name in
                   ("feature_week_v2", "feature_week_v1", "market_representation_v1")]
        available = [p for p in options if (p/"levels_1s.parquet").exists() and (p/"trades.parquet").exists()]
        if len(available) > 1:
            raise ValueError("Multiple local source versions need explicit selection")
        inv = dict(date=date, split=r["split"], candidates=len(c), source_present=bool(available),
                   local_training_admission="NOT_INCLUDED_IN_MODEL", known=False,
                   candidate_source_sha256=_sha(cp))
        inventory.append(inv)
        if not available:
            inv["status"] = "MISSING_LOCAL_SOURCE"
            continue
        market = Path(r["source"])
        if _sha(market/"manifest.json") != r["source_manifest_sha256"]:
            raise ValueError("Canonical market manifest changed")
        mm = json.loads((market/"manifest.json").read_text(encoding="utf-8"))
        payloads = {}
        for name in ("quotes.parquet", "seconds.parquet"):
            payloads[name] = _sha(market/name)
            if payloads[name] != mm["payloads"][name]:
                raise ValueError("Canonical market payload changed")
        ep = day/"episodes.parquet"
        if _sha(ep) != r["artifact_sha256"]["episodes.parquet"]:
            raise ValueError("Recorded departure identity changed")
        e = pd.read_parquet(ep, columns=["episode_id", "end_ns", "end_mid_nano", "completed"])
        c = c.merge(e, on="episode_id", how="left", validate="many_to_one")
        if not c.completed.all() or not (c.end_ns == c.episode_available_ns).all():
            raise ValueError("Candidate is not bound to its recorded confirmed departure")
        c = c.rename(columns={"end_mid_nano": "departure_mid_nano"})
        context = GlobalMBOContext(pd.read_parquet(market/"seconds.parquet"),
                                  pd.read_parquet(market/"quotes.parquet"),
                                  selection_available_ns=int(mm["input"]["selection"]["selection_available_ns"]),
                                  coverage_end_ns=int(mm["input"]["end_ns"])+1,
                                  window_seconds=config["mbo"]["window_seconds"])
        frame, receipt = local_mbo_diagnostics(c, context, available[0], expected_route=config["route"],
                                              expected_source_partition_sha256=mm["source_partition_sha256"])
        frame["date"] = date; frame["split"] = r["split"]
        receipt.update(date=date, canonical_market=str(market), canonical_manifest_sha256=_sha(market/"manifest.json"),
                       canonical_payload_sha256=payloads, episode_sha256=_sha(ep), candidate_sha256=_sha(cp))
        inv.update(status="DESCRIPTIVE_SOURCE_AVAILABLE", known=bool(frame.known.all()),
                   rows=len(frame), known_rows=int(frame.known.sum()), raw_dir=str(available[0]))
        frames.append(frame); receipts.append(receipt)
        print(json.dumps(dict(date=date, rows=len(frame), known=int(frame.known.sum())), ensure_ascii=True), flush=True)
        del context
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_parquet(destination/"local_mbo_diagnostics.parquet", index=False)
    pd.DataFrame(inventory).to_parquet(destination/"source_inventory.parquet", index=False)
    if _sha(run/"manifest.json") != parent_hash:
        raise ValueError("Parent manifest changed during the separate audit")
    manifest = dict(status="COMPLETE_DESCRIPTIVE_ONLY", parent_manifest_sha256=parent_hash,
                    source_episode_manifest_sha256=parent["source_manifest_sha256"],
                    implementation_sha256={str(Path(__file__).relative_to(root)): _sha(__file__),
                                           "experiments/zone_conditional_v1/mbo_features.py": _sha(Path(__file__).with_name("mbo_features.py")),
                                           "experiments/price_level_memory_v1/mbo_evidence.py": _sha(root/"experiments/price_level_memory_v1/mbo_evidence.py")},
                    days_with_source=len(receipts), days_without_source=len(inventory)-len(receipts),
                    missing_source_candidates=sum(i["candidates"] for i in inventory if not i["source_present"]),
                    rows=len(combined), known_rows=int(combined.known.sum()) if len(combined) else 0,
                    local_training_admission="NOT_INCLUDED_IN_MODEL", elapsed_seconds=time.perf_counter()-begun,
                    receipts=receipts, artifacts={name: _sha(destination/name) for name in
                                                ("local_mbo_diagnostics.parquet", "source_inventory.parquet")})
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("conditional_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_local_audit(args.conditional_run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("receipts", "implementation_sha256", "artifacts")}, indent=2))
