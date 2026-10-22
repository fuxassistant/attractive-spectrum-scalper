"""Saved-data admission and actual source-hash audit for the frozen V5 block.

Only known/unknown label admission is counted; no return, class performance,
future target value, or prediction is used to change source dates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_opportunity_v3 import context_diagnostics, read, sha, stats


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/research/opportunity_pipeline_v5.json")
    parser.add_argument("--causal-proof", default="reports/scientific_pipeline_v5/causal_verification.json")
    parser.add_argument("--output", default="reports/scientific_pipeline_v5/data_admission.json")
    args = parser.parse_args()
    started = time.perf_counter()
    config, proof = read(args.config), read(args.causal_proof)
    root = Path(config["data"]["root"])
    manifest_path = root/"manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "COMPLETE" or proof["status"] != "PASS" or proof["config_sha256"] != sha(args.config):
        raise ValueError("Completed dataset and matching independent replay required")
    if manifest["dates"] != config["data"]["dates"] or manifest["splits"] != config["splits"]:
        raise ValueError("Frozen source dates changed")
    if sha(config["data"]["source_inventory"]) != config["data"]["source_inventory_sha256"]:
        raise ValueError("Source inventory changed")
    names, groups = manifest["feature_names"], dict(manifest["groups"])
    if len(names) != 225 or {key:len(value) for key,value in groups.items()} != dict(geometry=14,micro=70,rich=225):
        raise ValueError("Expected frozen 14/70/225 schema")
    groups.update(micro_only=[n for n in groups["micro"] if n not in groups["geometry"]],
        fast_spectra=[n for n in names if n.startswith("mbo__fft_")],
        day_context=[n for n in names if n.startswith("context__") and not n.startswith("context__stft_")],
        long_stft=[n for n in names if n.startswith("context__stft_")])
    columns = {key:[names.index(n) for n in value] for key,value in groups.items()}
    sources = {str(path):sha(path) for path in (Path(__file__), Path("scripts/audit_opportunity_v3.py"),
        Path(args.config), Path(args.causal_proof), manifest_path, Path(config["data"]["source_inventory"]))}
    expected_sources, payloads, frames, day_rows, finite_counts = {}, {}, [], [], []
    required = ["date","t_ns","feature_available","y_crossing","physical_support_available",
        "physical_segment_token","legacy_segment_token","segment_token","source_endpoint_ns",
        "source_available_ns","source_age_seconds"]
    for day in manifest["days"]:
        receipt = read(root/(day["date"]+".json"))
        if receipt != day or day["status"] != "COMPLETE":
            raise ValueError("Daily receipt differs from completed manifest")
        sources[str(root/(day["date"]+".json"))] = sha(root/(day["date"]+".json"))
        for path, digest in {**day["identity"]["sources"], **day["source_sha256"]}.items():
            canonical = str(Path(path).resolve())
            if canonical in expected_sources and expected_sources[canonical] != digest:
                raise ValueError("Conflicting source identity")
            expected_sources[canonical] = digest
        for name, digest in day["files"].items():
            path = root/name
            if sha(path) != digest:
                raise ValueError("Changed dataset output")
            payloads[str(path)] = digest
        frame = pd.read_parquet(root/day["roots_file"], columns=required)
        x = np.load(root/day["x_file"], mmap_mode="r")
        if x.shape != (len(frame),225) or x.dtype != np.float32 or np.isinf(x).any():
            raise ValueError("Malformed feature matrix")
        admitted = frame.feature_available.to_numpy(bool)
        known = admitted & (frame.y_crossing.to_numpy() >= 0)
        support = frame.physical_support_available.to_numpy(bool)
        if np.any(admitted & ~support):
            raise ValueError("Current X admitted outside physical quote support")
        t, endpoint, publication = (frame[n].to_numpy(np.int64) for n in ("t_ns","source_endpoint_ns","source_available_ns"))
        age = frame.source_age_seconds.to_numpy(float)
        if np.any(endpoint[admitted] > t[admitted]) or np.any(publication[admitted] > t[admitted]) or np.any(age[admitted] < 0) or np.any(age[admitted] > config["data"]["max_feature_age_seconds"]):
            raise ValueError("Source clock/lookahead/staleness violation")
        if not np.allclose(age[admitted], (t[admitted]-endpoint[admitted])/1e9, rtol=0, atol=1e-12):
            raise ValueError("Saved source age differs from endpoint age")
        finite = np.isfinite(x)
        counts = finite[admitted].sum(axis=0)
        finite_counts.append(counts)
        families = {}
        for family, cols in columns.items():
            seen = finite[admitted][:,cols]
            families[family] = dict(features=len(cols), admitted_finite_cells=int(seen.sum()),
                missing_fraction_among_admitted=float(1-seen.mean()),
                full_clock_observed_cell_fraction=float(seen.sum()/(len(frame)*len(cols))),
                all_features_known_roots=int(seen.all(axis=1).sum()),
                all_features_known_fraction=float(seen.all(axis=1).mean()))
        context = day["feature_receipt"]
        if context["max_asof_lookahead_ns"] > 0 or context["max_context_lookahead_ns"] > 0:
            raise ValueError("Builder reported future context publication")
        previous_known = x[admitted,names.index("context__previous_known")]
        day_rows.append(dict(date=day["date"], split=next(k for k,v in config["splits"].items() if day["date"] in v),
            roots=len(frame), feature_available=int(admitted.sum()), known_feature_roots=int(known.sum()),
            unknown_feature_roots=int((admitted & ~known).sum()), physical_support_roots=int(support.sum()),
            missing_X_inside_supported_clock=int((support & ~admitted).sum()),
            source_age_seconds=stats(age[admitted]), families=families,
            previous_date=context["previous_date"], previous_source_status=context["previous_source_status"],
            previous_context_known_fraction_among_admitted=float((previous_known == 1).mean()),
            day_age_minutes=stats(x[admitted,names.index("context__day_age_minutes")]),
            current_day_coverage=stats(x[admitted,names.index("context__day_coverage")]),
            previous_day_coverage=stats(x[admitted,names.index("context__previous_coverage")]),
            source_clock_saved_checks="PASS"))
        frames.append(frame)
        del x, finite
    validated, total_bytes = {}, 0
    for path, expected in expected_sources.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError("Actual consumed source bytes differ: "+path)
        validated[path] = actual
        total_bytes += Path(path).stat().st_size
    frame = pd.concat(frames, ignore_index=True)
    context = context_diagnostics(frame, config["splits"], 5,61,64)
    for mode in context["modes"].values():
        for block in (mode["by_split"],mode["by_day"]):
            for value in block.values():
                value.pop("clock_token_histogram",None)
    by_split = {}
    for role, dates in config["splits"].items():
        rows = [r for r in day_rows if r["date"] in dates]
        n, admitted = sum(r["roots"] for r in rows), sum(r["feature_available"] for r in rows)
        by_split[role] = dict(days=len(rows), roots=n, feature_available=admitted,
            known_feature_roots=sum(r["known_feature_roots"] for r in rows),
            unknown_feature_roots=sum(r["unknown_feature_roots"] for r in rows),
            physical_support_roots=sum(r["physical_support_roots"] for r in rows),
            families={family:dict(features=len(cols),
                missing_fraction_among_admitted=1-sum(r["families"][family]["admitted_finite_cells"] for r in rows)/(admitted*len(cols)),
                full_clock_observed_cell_fraction=sum(r["families"][family]["admitted_finite_cells"] for r in rows)/(n*len(cols)),
                all_features_known_fraction=sum(r["families"][family]["all_features_known_roots"] for r in rows)/admitted)
                for family,cols in columns.items()})
    available = int(frame.feature_available.sum())
    per_feature = [dict(name=name, finite_admitted_roots=int(count),
        missing_fraction_among_admitted=1-float(count/available))
        for name,count in zip(names,np.sum(finite_counts,axis=0))]
    result = dict(status="PASS", config_sha256=sources[str(Path(args.config))], source_hashes=sources,
        dataset_payload_hashes=payloads, source_validation=dict(status="ACTUAL_BYTES_VERIFIED_ALL20_DAYS",
            unique_files=len(validated), bytes_hashed=total_bytes, actual_source_sha256=validated,
            quote_payloads=sum(Path(p).name=="quotes.parquet" for p in validated),
            feature_payloads=sum(Path(p).name=="features.parquet" for p in validated),
            scope="Every unique consumed source and implementation SHA from all20 completed day receipts, plus every saved dataset payload. Raw source bytes streamed for SHA only; quotes were parsed only by the separate three-day replay."),
        independent_source_replay=dict(path=args.causal_proof,sha256=sources[str(Path(args.causal_proof))],
            dates=proof["selected_dates"],totals=proof["totals"],verified_unique_files=proof["verified_unique_files"],status=proof["status"]),
        families={k:len(v) for k,v in groups.items()}, by_split=by_split, by_day=day_rows,
        per_feature=per_feature, context_availability=context,
        utilization=dict(frozen_backbone_feature_family="micro", frozen_backbone_values=70,
            frozen_backbone_input_channels=141, family_available_on_disk=225,
            detailed_history_receptive_field_tokens=61,detailed_history_maximum_span_seconds=300,
            omitted_from_frozen_backbone=dict(fast_spectra=126,day_context=21,long_stft=8),
            filter_input=config["filter"]["input"],
            note="Stored rich225 includes causal current/previous London context, fast spectral channels and long STFT. Frozen Micro70 backbone and the V5 filter do not consume the155 extra rich channels. Complete current+previous-day storage is not a learned two-day sequence."),
        mask_contract="Observation mask is finite(raw feature) AND common feature_available; missing finite geometry/context on unavailable rows must also be masked. Restored histories retain supported missing-X clock positions, never forward-fill. Known target masks only bound scoring/training, not physical history.",
        no_source_date_changes=True,seconds=time.perf_counter()-started)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(status="PASS",output=str(output),sha256=sha(output),source_files=len(validated),roots=len(frame),seconds=result["seconds"])))


if __name__ == "__main__":
    main()
