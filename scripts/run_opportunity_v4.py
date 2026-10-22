"""Four fresh restored-context fits with prespecified bounded soft-ID weights.

No arm, seed, checkpoint or policy is selected on TEST. This is a controlled
follow-up on the already inspected V3 dataset, not a new blind holdout.
"""
from __future__ import annotations

import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

from spectrum.research_v4.learning import load_dataset, train_run, sha, validate_weight_parameters
from scripts.run_opportunity_v2 import save, read, verify_run


CAL_THRESHOLD_RULE = "Sort CAL NONE scores descending s[0..n-1], k=floor(0.05*n); threshold=np.nextafter(s[k],+inf) in float64"


def experiment_configs(config):
    for seed in config["experiment"]["seeds"]:
        for weighting in config["experiment"]["weightings"]:
            arm = copy.deepcopy(config)
            arm["seed"] = seed
            arm["data"]["context_mode"] = "restored"
            arm["training"]["weighting"] = weighting
            arm["run_id"] = f"tcn_micro_restored_{weighting}_seed{seed}"
            yield arm


def validate_false_signal_protocol(protocol):
    expected = dict(fixed_thresholds=[0.35, 0.5, 0.65, 0.8],
        score_channels={"event": "1-p_none", "max_side": "max(p_long+p_both,p_short+p_both)"},
        side_tie_break="LONG",
        event_true_positive="y_crossing>0",
        side_true_positive="chosen side is a positive bit of y_crossing; BOTH qualifies either side",
        wrong_side="y_crossing>0 and chosen side is absent from positive target bits",
        primary_side_recall_denominator="All known X-admitted positive event roots; common across models")
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("Prespecified false-signal protocol changed: " + key)
    calibration = protocol["calibration_none_fpr"]
    if (calibration.get("budget") != 0.05 or calibration.get("score_dtype") != "float64"
            or calibration.get("selection") != CAL_THRESHOLD_RULE
            or calibration.get("cohort") != "Known feature-available CAL NONE roots only"
            or calibration.get("comparison") != "float64 score >= frozen float64 threshold; inclusive and conservative under ties"
            or calibration.get("empty_negative_cohort") != {"threshold": 1.01, "reason": "insufficient_negative_denominator"}):
        raise ValueError("CAL-only conservative NONE-FPR protocol changed")
    if protocol["fixed_test_execution_stress"]["thresholds"] != [0.35, 0.5]:
        raise ValueError("Prespecified descriptive TEST stress thresholds changed")


def validate_recipe(config):
    if config.get("schema") != "scientific_opportunity_pipeline_v4":
        raise ValueError("Expected V4 configuration schema")
    old = read("configs/research/opportunity_pipeline_v3.json")
    for key in ("event_query", "execution", "risk", "loss", "model", "policy", "splits"):
        if config[key] != old[key]:
            raise ValueError("Frozen V3 recipe changed: " + key)
    expected_data = {**old["data"], "context_mode": "restored"}
    if config["data"] != expected_data:
        raise ValueError("Exactly the frozen V3 dataset, admission and restored Micro context required")
    old_training = {key: value for key, value in old["training"].items() if key != "weighting"}
    actual_training = {key: value for key, value in config["training"].items() if key not in {"weighting", "id_weighting"}}
    if old_training != actual_training:
        raise ValueError("Unpaired training hyperparameters")
    validate_weight_parameters(config["training"]["id_weighting"])
    if config["training"]["weighting"] not in {"uniform", "soft_crossing_id"}:
        raise ValueError("Unsupported V4 weighting arm")
    experiment = config["experiment"]
    if (experiment["seeds"] != [17, 29] or experiment["contexts"] != ["restored"]
            or experiment["weightings"] != ["uniform", "soft_crossing_id"]):
        raise ValueError("Exactly the four prespecified V4 arms required")
    if experiment["fixed_visual_cases"] != old["experiment"]["fixed_visual_cases"]:
        raise ValueError("Prespecified fixed visual windows changed")
    validate_false_signal_protocol(experiment["false_signal_protocol"])


def source_receipt(config_path, manifest):
    """Bind the wrapper, frozen dependencies, recipe and dataset-builder sources."""
    paths = {Path(__file__), Path(config_path), Path("scripts/run_opportunity_v2.py"),
             Path("configs/research/opportunity_pipeline_v3.json"),
             Path("spectrum/research_v4/learning.py"),
             *[Path("spectrum/research_v3") / name for name in ("learning.py", "features.py")],
             *[Path("spectrum/research_v2") / name for name in ("learning.py", "evaluation.py")],
             *[Path("spectrum/research") / name for name in ("models.py", "losses.py", "training.py")]}
    paths.update(Path(name) for name in manifest["identity"]["sources"])
    return {str(path.resolve()): sha(path) for path in sorted(paths, key=str)}


def run_one(store, execution, config, runs_root, sources):
    """The frozen V3 receipt/economic boundary with an explicit V4 trainer."""
    from spectrum.research_v2.evaluation import evaluate_risk_grid
    out = runs_root / config["run_id"]
    if out.exists():
        return verify_run(out, config, sources, store.source_receipt["manifest_sha256"])
    out.mkdir(parents=True)
    save(out / "config.json", config)
    receipt = dict(status="RUNNING", run_id=config["run_id"], config_sha256=sha(out / "config.json"),
        dataset_manifest_sha256=store.source_receipt["manifest_sha256"], sources=sources,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_note="Working source SHA is authoritative, including uncommitted implementation",
        libraries={name: version(name) for name in ("numpy", "pandas", "torch", "numba", "scikit-learn")},
        cuda=torch.version.cuda)
    save(out / "receipt.json", receipt)
    print(json.dumps(dict(stage="START", run=config["run_id"])), flush=True)
    result = train_run(store, config, out)
    p = result["predictions"][["p_none", "p_long", "p_short", "p_both"]].to_numpy()
    economics = evaluate_risk_grid(store.frame, execution, p, config)
    save(out / "economic_evaluation.json", economics)
    result["summary"].update(status="COMPLETE", economic_evaluation="economic_evaluation.json",
                             selected_policy=economics["summary"])
    save(out / "summary.json", result["summary"])
    if any(sha(path) != expected for path, expected in sources.items()):
        raise ValueError("Bound source changed during run: retained incomplete")
    receipt.update(status="COMPLETE", outputs={p.name: sha(p) for p in out.iterdir()
                                               if p.is_file() and p.name != "receipt.json"})
    save(out / "receipt.json", receipt)
    print(json.dumps(dict(stage="COMPLETE", run=config["run_id"],
        best_update=result["summary"]["training"]["best_update"],
        validation_nll=result["summary"]["evaluation"]["validation"]["model"]["joint_nll"])), flush=True)
    torch.cuda.empty_cache()
    return result["summary"]


def verify_paired_invariants(summaries, runs_root):
    expected={f"tcn_micro_restored_{weighting}_seed{seed}" for seed in (17,29) for weighting in ("uniform","soft_crossing_id")}
    if len(summaries)!=4 or {s["run_id"] for s in summaries}!=expected:
        raise ValueError("Four distinct prescribed completed arms required")
    if (len({s["scaler_sha256"] for s in summaries}) != 1
            or any(s["cohorts"] != summaries[0]["cohorts"] for s in summaries)):
        raise ValueError("Unmatched cohorts or scalers")
    for seed in (17,29):
        group=[s for s in summaries if s["run_id"].endswith("seed"+str(seed))]
        for key in ("train_metric_probe_sha256", "gradient_probe_sha256", "initial_state_sha256"):
            if len({s["training"][key] for s in group}) != 1:
                raise ValueError("Unmatched seed-level RNG/probe: "+key)
        histories=[read(Path(runs_root)/s["run_id"]/"history.json") for s in group]
        checkpoints=set.intersection(*[{row["update"] for row in history} for history in histories])
        for update in checkpoints:
            prefixes={next(row["sampled_root_stream_sha256"] for row in history if row["update"]==update)
                      for history in histories}
            if len(prefixes)!=1:
                raise ValueError("Unmatched sampled TRAIN-root prefix at update "+str(update))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/research/opportunity_pipeline_v4.json")
    parser.add_argument("--runs-root", default="data/derived/scientific_pipeline_v4/runs")
    parser.add_argument("--report-root", default="reports/scientific_pipeline_v4")
    args=parser.parse_args()
    config=read(args.config)
    validate_recipe(config)
    report, runs_root=Path(args.report_root), Path(args.runs_root)
    report.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    store=load_dataset(config["data"]["root"])
    manifest=read(Path(config["data"]["root"])/"manifest.json")
    from scripts.prepare_opportunity_v3 import identity
    if manifest.get("identity") != identity(config):
        raise ValueError("Frozen dataset identity differs from V4 recipe")
    sources=source_receipt(args.config,manifest)
    execution=pd.concat([pd.read_parquet(Path(config["data"]["root"])/(date+"_execution.parquet"))
                         for date in manifest["dates"]],ignore_index=True)
    suite=dict(status="RUNNING", dataset_root=config["data"]["root"], runs_root=str(runs_root),
        config_sha256=sha(args.config), dataset_manifest_sha256=store.source_receipt["manifest_sha256"],
        runs=[], selection="No arm/seed winner selection; all four fresh fits reported", primary_seed=17,
        replication_seed=29, fixed_visual_cases=config["experiment"]["fixed_visual_cases"],
        false_signal_protocol=config["experiment"]["false_signal_protocol"],
        historical_exposure=config["experiment"]["historical_exposure"],
        prior_control_runs_root="data/derived/scientific_pipeline_v3/runs")
    save(report/"suite_summary.json",suite)
    summaries=[]
    started=time.perf_counter()
    for arm in experiment_configs(config):
        summaries.append(run_one(store,execution,arm,runs_root,sources))
        suite["runs"].append(dict(run_id=arm["run_id"], context="restored", weighting=arm["training"]["weighting"],
                                  seed=arm["seed"],role="primary" if arm["seed"]==17 else "replication"))
        save(report/"suite_summary.json",suite)
    verify_paired_invariants(summaries,runs_root)
    suite.update(status="COMPLETE", wall_seconds=time.perf_counter()-started,
                 matched_scaler_and_cohort=True, matched_initialization_and_probes=True,
                 matched_sampled_root_prefixes=True)
    save(report/"suite_summary.json",suite)
    print(json.dumps(dict(stage="SUITE_COMPLETE",runs=len(summaries))),flush=True)


if __name__=="__main__":
    main()
