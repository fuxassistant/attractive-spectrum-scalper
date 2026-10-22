"""Freeze V5 dates from observed source coverage only, before building labels.

Reads manifests, coverage columns of prepared minute X, and Parquet footers.
Does not load quote/trade tapes, targets, predictions, or trading outcomes.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.parquet as pq

SPLITS = {
    "train": ["2026-07-30", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11"],
    "validation": ["2026-08-12", "2026-08-13", "2026-08-14"],
    "calibration": ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"],
    "test": ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-09-01"],
}
CONTEXT_ONLY = ["2026-07-29", "2026-08-03", "2026-08-31"]
SYMBOL, ROUTE = "GCZ6", [1, 41271, 22]
PREPARED = Path("data/derived/basket_multimonth_v1")
FULL = Path("data/derived/full_mbo_v1")
OVERLAY = Path("reports/full_mbo_v1/audit_20261016/training_admission_overlay.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def previous_weekday(value):
    result = date.fromisoformat(value)-timedelta(days=1)
    while result.weekday() >= 5:
        result -= timedelta(days=1)
    return result.isoformat()


def sufficient_context(current, previous):
    return (current["symbol"] == previous["symbol"] == SYMBOL and
        current["full_observed_window"] and previous["full_observed_window"] and
        current["current_audit_eligible"] and previous["current_audit_eligible"])


def inspect_day(value, overlay, hashes, footers):
    directory = PREPARED/value[:7]/value
    manifest_path, diagnostic_path = directory/"manifest.json", directory/"diagnostics.json"
    if not manifest_path.exists():
        return dict(date=value, status="MISSING", full_observed_window=False)
    meta, diagnostic = read(manifest_path), read(diagnostic_path)
    for path in (manifest_path, diagnostic_path):
        hashes[str(path)] = sha(path)
    if meta.get("status") != "COMPLETE":
        raise ValueError("Incomplete prepared source: "+value)
    for name in ("bar_t.npy", "x.npy"):
        digest = sha(directory/name)
        if digest != meta["files"][name]["sha256"]:
            raise ValueError("Changed prepared coverage payload")
        hashes[str(directory/name)] = digest
    t = np.load(directory/"bar_t.npy", mmap_mode="r")
    x = np.load(directory/"x.npy", mmap_mode="r")
    active = (t >= int(meta["london_ns"])) & (t < int(meta["ny_close_ns"]))
    known = x[active, meta["feature_names"].index("candle_known")]
    coverage = x[active, meta["feature_names"].index("coverage")]
    expected = (int(meta["ny_close_ns"])-int(meta["london_ns"]))//60_000_000_000
    full = int(active.sum()) == expected and np.all(known == 1) and np.all(coverage == 1)
    captures = [overlay[part["key"]] for part in meta["quote_parts"]
                if overlay[part["key"]]["date"] == value and overlay[part["key"]]["symbol"] == meta["symbol"]]
    if len(captures) != 1:
        raise ValueError("Ambiguous current feature capture")
    capture = captures[0]
    feature_root = Path(capture["features"])
    market_root = FULL/"market"/capture["key"]
    fm, mm = read(feature_root/"manifest.json"), read(market_root/"manifest.json")
    for path in (feature_root/"manifest.json", market_root/"manifest.json"):
        hashes[str(path)] = sha(path)
    if fm["status"] != "COMPLETE" or mm["status"] != "COMPLETE" or fm["feature_sha256"] != capture["feature_sha256"]:
        raise ValueError("Incomplete feature source or overlay identity mismatch")
    if fm["market_manifest_sha256"] != sha(market_root/"manifest.json"):
        raise ValueError("Feature-to-market manifest binding mismatch")
    schemas = {}
    for path, expected_rows, payload_sha in ((feature_root/"features.parquet", fm["rows"], fm["feature_sha256"]),
        (market_root/"seconds.parquet", mm["seconds_rows"], mm["payloads"]["seconds.parquet"])):
        footer = pq.ParquetFile(path)
        if footer.metadata.num_rows != expected_rows:
            raise ValueError("Source footer/manifest row count mismatch")
        schemas[path.name] = footer.schema_arrow.names
        footers[str(path)] = dict(rows=footer.metadata.num_rows, columns=footer.metadata.num_columns,
            bytes=path.stat().st_size, declared_sha256=payload_sha, payload_sha_checked_now=False,
            schema_sha256=hashlib.sha256(str(footer.schema_arrow).encode()).hexdigest())
    names = schemas["features.parquet"]
    values = [name.removesuffix("__known") for name in names if name.endswith("__known")]
    if len(values) != 182 or not set(values).issubset(names) or len(names) != 373:
        raise ValueError("Canonical full-MBO feature schema changed")
    quotes = []
    for part in meta["quote_parts"]:
        path = Path(part["root"])/"quotes.parquet"
        footer = pq.ParquetFile(path)
        if path.stat().st_size != part["quote_file"]["bytes"]:
            raise ValueError("Quote payload size changed")
        pm_path = Path(part["root"])/"manifest.json"
        hashes[str(pm_path)] = sha(pm_path)
        pm = read(pm_path)
        if pm["status"] != "COMPLETE" or pm["payloads"]["quotes.parquet"] != part["quote_file"]["sha256"] or pm["quotes"] != footer.metadata.num_rows:
            raise ValueError("Quote footer/lineage mismatch")
        footers[str(path)] = dict(rows=footer.metadata.num_rows, columns=footer.metadata.num_columns,
            bytes=path.stat().st_size, declared_sha256=part["quote_file"]["sha256"], payload_sha_checked_now=False)
        quotes.append(dict(path=str(path), key=part["key"], route=part["route"],
            capture_date=overlay[part["key"]]["date"], capture_audit_eligible=overlay[part["key"]]["audit_eligible"],
            capture_reasons=overlay[part["key"]]["reasons"], **footers[str(path)]))
    return dict(date=value, status="COMPLETE", symbol=meta["symbol"], route=meta["quote_parts"][-1]["route"],
        full_observed_window=bool(full), current_audit_eligible=bool(capture["audit_eligible"]),
        prepared_manifest=str(manifest_path), prepared_manifest_sha256=hashes[str(manifest_path)],
        capture=capture["key"], feature_path=str(feature_root/"features.parquet"),
        feature_sha256=fm["feature_sha256"], feature_names=values,
        source_feature_rows=fm["rows"], source_feature_columns=len(names), source_feature_values=len(values),
        observed_minutes=int(active.sum()), known_minutes=int((known == 1).sum()), expected_minutes=int(expected),
        minimum_minute_coverage=float(coverage.min()), quotes=meta["quotes"], active_quotes=diagnostic["quotes"],
        invalid_publications=diagnostic["invalid_publications"], max_quote_gap_seconds=diagnostic["max_pause_seconds"],
        legal_m1_origins=diagnostic["legal_minute_origins"], audit_reference_session_coverage=capture["audit_session_coverage"],
        expected_5s_origins=(int(meta["liquidation_deadline_ns"])-int(meta["london_ns"]))//5_000_000_000,
        quote_parts=quotes, prepared_bytes=sum(path.stat().st_size for path in directory.iterdir() if path.is_file()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="reports/scientific_pipeline_v5/source_inventory.json")
    args = parser.parse_args()
    started = time.perf_counter()
    dates = [day for days in SPLITS.values() for day in days]
    if dates != sorted(set(dates)) or len(dates) != 20 or any(day <= "2026-07-24" for day in dates):
        raise ValueError("Frozen new chronological roles changed")
    sources = {str(path): sha(path) for path in (Path(__file__), FULL/"manifest.json", FULL/"training_catalog.json",
        FULL/"status.json", OVERLAY, PREPARED/"catalog.json", PREPARED/"preparation_contract.json",
        PREPARED/"source_verification.json", Path("spectrum/cognition/dataset.py"),
        Path("reports/scientific_pipeline_v3/source_inventory.json"))}
    master, catalog = read(FULL/"manifest.json"), read(PREPARED/"catalog.json")
    if master["status"] != "COMPLETE" or catalog["status"] != "COMPLETE":
        raise ValueError("Full source/preparation incomplete")
    overlay = {row["key"]: row for row in read(OVERLAY)["captures"]}
    inspected = sorted(set(dates+CONTEXT_ONLY+["2026-07-27", "2026-07-28", "2026-07-31", "2026-08-28"]))
    footers, days = {}, {}
    for day in inspected:
        days[day] = inspect_day(day, overlay, sources, footers)
    selected = []
    for role, values in SPLITS.items():
        for value in values:
            row = dict(days[value]); previous = previous_weekday(value)
            if previous not in days or not sufficient_context(row, days[previous]) or row["route"] != ROUTE:
                raise ValueError("Current/previous source admission or full coverage failed: "+value)
            row.update(split=role, previous_weekday=previous, previous_audit_eligible=True,
                previous_full_observed_window=True, previous_symbol=days[previous]["symbol"])
            selected.append(row)
    if any(row["invalid_publications"] != 0 for row in selected):
        raise ValueError("Unexpected invalid current publications in selected full-window block")
    consumed = selected+[days[day] for day in CONTEXT_ONLY]
    if any(part["route"] != ROUTE for row in consumed for part in row["quote_parts"]):
        raise ValueError("Quote parts cross the frozen contract route")
    quote_paths = {part["path"] for row in consumed for part in row["quote_parts"]}
    feature_paths = {row["feature_path"] for row in consumed}
    second_paths = {str(FULL/"market"/row["capture"]/"seconds.parquet") for row in consumed}
    common_schema = selected[0]["feature_names"]
    if any(row["feature_names"] != common_schema for row in consumed):
        raise ValueError("Selected/context source feature order differs")
    disk = shutil.disk_usage(Path.cwd())
    old_inventory = read("reports/scientific_pipeline_v3/source_inventory.json")
    overlap = sorted(set(dates) & set(old_inventory["dates"]))
    if overlap:
        raise ValueError("Selected dates overlap scientific V3/V4")
    # Runtime measurement only; do not inspect old labels, scores or results.
    old_dataset_path = Path("data/derived/scientific_pipeline_v3/dataset/manifest.json")
    old_dataset = read(old_dataset_path); sources[str(old_dataset_path)] = sha(old_dataset_path)
    old_seconds = sum(day["seconds"] for day in old_dataset["days"])
    old_quotes = sum(day["quotes"] for day in old_dataset["days"])
    old_output_bytes = sum((old_dataset_path.parent/name).stat().st_size for day in old_dataset["days"] for name in day["files"])
    expected_roots = sum(row["expected_5s_origins"] for row in selected)
    estimated_seconds = old_seconds*sum(row["quotes"] for row in selected)/old_quotes
    result = dict(status="COVERAGE_ONLY_FROZEN_SELECTION", schema="SCIENTIFIC_SOURCE_INVENTORY_V5",
        dates=dates, splits=SPLITS, initial_context_date="2026-07-29", context_only_dates=CONTEXT_ONLY,
        symbol=SYMBOL, route=ROUTE, selection_rule="Fixed before V5 labels: one GCZ6 contract; current and immediate previous weekday are overlay-admitted, COMPLETE, 840/840 known London-to-NY minutes with coverage1.0. No outcomes/predictions or model performance used for selection.",
        previous_context_rule="Immediately previous weekday only; never skip back across missing/partial days. Context-only session must have complete current coverage; its own previous day is not recursively required.",
        excluded_dates={"2026-07-27": "GCQ6 rather than fixed GCZ6", "2026-07-28": "GCQ6 rather than fixed GCZ6",
            "2026-07-29": "First GCZ6 selected session; previous weekday GCQ6. Warmup only for July30.",
            "2026-07-31": "Explicit source gap; no prepared session", "2026-08-03": "Previous weekday July31 absent. Complete current session retained for August4 context.",
            "2026-08-28": "813 invalid publications; 839/840 known minutes and minimum minute coverage0.91668868. Overlay remains eligible, but strict full-window selection excludes it.",
            "2026-08-31": "Current840/840 complete but previous weekdayAugust28 partial. Current session retained as September1 context."},
        contract_boundaries=dict(previous_scientific_symbol="GCQ6", previous_route=[1,42011464,22],
            last_observed_GCQ6="2026-07-28", first_observed_GCZ6="2026-07-29", in_selected_window_roll=False,
            contract_shift_from_frozen_backbone=True, note="Source-only selection permits a new-contract domain shift; do not mix absolute contract price levels or carry state across the roll."),
        source_sha256=sources, days=selected, context_days=[days[day] for day in CONTEXT_ONLY],
        exclusion_evidence={day:days[day] for day in ("2026-07-27","2026-07-28","2026-07-31","2026-08-28")},
        source_payload_footers=footers, source_gaps=master["source_gaps"],
        totals=dict(days=20, full_session_quotes=sum(row["quotes"] for row in selected),
            active_quotes=sum(row["active_quotes"] for row in selected), invalid_publications=0,
            max_quote_gap_seconds=max(row["max_quote_gap_seconds"] for row in selected),
            expected_origin_roots_at_5s_if_same_session_clock=expected_roots,
            source_feature_rows=sum(row["source_feature_rows"] for row in selected),
            unique_quote_files_including_context=len(quote_paths), unique_quote_bytes_including_context=sum(footers[p]["bytes"] for p in quote_paths),
            unique_features_bytes_including_context=sum(footers[p]["bytes"] for p in feature_paths),
            unique_seconds_bytes_including_context=sum(footers[p]["bytes"] for p in second_paths),
            prepared_session_bytes_including_context=sum(row["prepared_bytes"] for row in consumed)),
        feature_schema=dict(values=182, known_masks=182, metadata_columns=9, columns=373, value_names=common_schema),
        resources=dict(free_bytes=disk.free, free_GiB=disk.free/1024**3, minimum_reserve_GiB=10,
            old_v3_measured_build_seconds=old_seconds, old_v3_processed_quotes=old_quotes,
            rough_serial_build_seconds_by_quote_ratio=estimated_seconds,
            rough_new_dataset_bytes_by_root_ratio=int(old_output_bytes*expected_roots/old_dataset["roots"]),
            runtime_caveat="Projection from V3 measured builder timing only; compilation, disk cache, integrity hashing and new filter artifacts add time. No model fitting/runtime is estimated from outcomes.",
            inventory_seconds=time.perf_counter()-started),
        historical_exposure=dict(scientific_v3_v4_overlap_dates=overlap, new_relative_to_scientific_v3_v4=True,
            never_inspected_holdout_claim=False, evidence="spectrum/cognition/dataset.py assigns2026-08 to test; historical execution extent is not audited here. September1 lies outside that May-August recipe."),
        limits=["Metadata/coverage arrays and Parquet footers were checked; source quote/feature payload SHAs are declared lineage, not rehashed raw bytes. Builder must verify actual consumed payloads.",
            "Current and previous trading-date overlays are strict. Sunday context-only quote parts may be overlay-ineligible for entries; this does not invalidate valid causal warmup quotes.",
            "Complete observed minutes are an availability criterion, not an authoritative exchange calendar or a guarantee every feature is known.",
            "Quote pauses below60s may still cause2s execution timeout/censor. Preserve all support transitions, UNKNOWN labels and restored history masks.",
            "Twenty new source days support7/3/5/5 chronological roles; five TEST days limit inference and are not independent observations.",
            "Raw trade/depth events are represented by canonical causal182-feature values and masks; this inventory does not regenerate raw MBO or add exact trade-price VWAP."])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=result["status"], output=str(output), sha256=sha(output), totals=result["totals"], resources=result["resources"])))


if __name__ == "__main__":
    main()
