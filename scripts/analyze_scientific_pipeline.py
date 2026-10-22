"""Saved-data diagnostics for the scientific opportunity pipeline.

Reads only small verified Parquet/JSON sources and the causal feature allowlist.
No fitting, quote-tape loading, or policy selection occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum.research.data import load_market, RICH_FEATURES, GEOMETRY_FEATURES

NS = 1_000_000_000
STATES = ("NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, q10=None, q50=None, q90=None, min=None, max=None)
    q = np.quantile(a, [.1, .5, .9])
    return dict(n=len(a), mean=float(a.mean()), q10=float(q[0]), q50=float(q[1]),
                q90=float(q[2]), min=float(a.min()), max=float(a.max()))


def counts(values):
    return {str(k): int(v) for k, v in pd.Series(values).value_counts().items()}


def runs(frame, y, event=False):
    """Observed constant-label runs; unknown values/date/gaps break a run."""
    t = frame.t_ns.to_numpy(np.int64)
    dates = frame.date.to_numpy()
    states = np.where(y < 0, -1, (y > 0).astype(int)) if event else y
    all_runs, internal = [], []
    i = 0
    while i < len(states):
        if states[i] < 0:
            i += 1
            continue
        start = i
        while i + 1 < len(states) and states[i + 1] == states[start] and dates[i + 1] == dates[i] and t[i + 1] - t[i] == 30 * NS:
            i += 1
        n = i - start + 1
        left_change = start > 0 and states[start - 1] >= 0 and dates[start - 1] == dates[start] and t[start] - t[start - 1] == 30 * NS
        right_change = i + 1 < len(states) and states[i + 1] >= 0 and dates[i + 1] == dates[i] and t[i + 1] - t[i] == 30 * NS
        all_runs.append(n * 30)
        if left_change and right_change:
            internal.append(n * 30)
        i += 1
    return dict(all_observed_runs_seconds=quantiles(all_runs),
                bounded_by_known_changes_seconds=quantiles(internal),
                one_root_run_fraction=float(np.mean(np.asarray(all_runs) == 30)) if all_runs else None)


def label_stats(frame, y):
    known = y >= 0
    pair = known[1:] & known[:-1] & (np.diff(frame.t_ns.to_numpy()) == 30 * NS)
    pair &= frame.date.to_numpy()[1:] == frame.date.to_numpy()[:-1]
    return dict(roots=len(y), known=int(known.sum()), unknown=int((~known).sum()),
                classes={name: int((y == i).sum()) for i, name in enumerate(STATES)},
                event_fraction=float((y[known] > 0).mean()) if known.any() else None,
                consecutive_known_pairs=int(pair.sum()),
                state_flip_30s=float((y[1:] != y[:-1])[pair].mean()) if pair.any() else None,
                event_flip_30s=float(((y[1:] > 0) != (y[:-1] > 0))[pair].mean()) if pair.any() else None,
                state_runs=runs(frame, y), event_runs=runs(frame, y, True))


def pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def label_lags(frame, y):
    result = {}
    for lag in (30, 60, 300, 900, 1800):
        left = pd.DataFrame(dict(date=frame.date, t_ns=frame.t_ns, y=y))
        right = left.rename(columns={"y": "previous_y"}).copy()
        right.t_ns += lag * NS
        pair = left.merge(right, on=["date", "t_ns"], how="inner", validate="one_to_one")
        pair = pair.loc[(pair.y >= 0) & (pair.previous_y >= 0)]
        a = pair.y.to_numpy()
        b = pair.previous_y.to_numpy()
        result[str(lag)] = dict(pairs=len(pair), state_agreement=float((a == b).mean()) if len(pair) else None,
                                event_agreement=float(((a > 0) == (b > 0)).mean()) if len(pair) else None,
                                event_phi=pearson((a > 0).astype(float), (b > 0).astype(float)))
    return result


def feature_group(name):
    if name.startswith("bb"):
        return "bb_geometry"
    if name.startswith("stft_"):
        return "stft"
    if name.startswith(("day_", "previous_", "combined_", "context_")) or name == "volume_vs_previous":
        return "london_context"
    return "local_flow_and_baseline"


def feature_stats(raw, names, admitted):
    values = raw[admitted]
    finite = np.isfinite(values)
    groups = {}
    for family in sorted({feature_group(n) for n in names}):
        use = np.array([feature_group(n) == family for n in names])
        f = finite[:, use]
        groups[family] = dict(features=int(use.sum()), admitted_roots=len(values),
                              missing_cell_fraction=float((~f).mean()) if len(f) else None,
                              roots_with_any_missing_fraction=float((~f).any(axis=1).mean()) if len(f) else None)
    by_feature = {name: float((~finite[:, i]).mean()) if len(values) else None for i, name in enumerate(names)}
    constant = []
    for i, name in enumerate(names):
        a = values[:, i][finite[:, i]]
        if len(a) and np.max(a) == np.min(a):
            constant.append(name)
    return dict(admitted_roots=int(admitted.sum()), absent_feature_origins=int((~admitted).sum()),
                by_family=groups, nonfinite_fraction_by_feature=by_feature, constant_observed_features=constant)


def execution_stats(f, gain, stop):
    complete = f.loc[f.status == "COMPLETE"].copy()
    complete["holding_s"] = (complete.exit_ns - complete.entry_ns) / NS
    complete["net_R"] = complete.net / complete.risk_R
    reasons = {}
    for reason in ("TAKE", "STOP", "TIME"):
        a = complete.loc[complete.reason == reason]
        nr = a.net_R.to_numpy()
        nominal = gain if reason == "TAKE" else -stop if reason == "STOP" else None
        reasons[reason] = dict(count=len(a), net_R=quantiles(nr), holding_seconds=quantiles(a.holding_s),
                               positive_net_fraction=float((nr > 0).mean()) if len(nr) else None,
                               net_minus_nominal_R=quantiles(nr - nominal) if nominal is not None else None)
        if reason == "TAKE":
            reasons[reason]["exit_below_take_threshold_fraction"] = float((nr < gain - 1e-9).mean()) if len(nr) else None
        if reason == "STOP":
            reasons[reason]["exit_beyond_stop_threshold_fraction"] = float((nr < -stop - 1e-9).mean()) if len(nr) else None
    return dict(rows=len(f), status_counts=counts(f.status), complete_reasons=counts(complete.reason),
                complete_holding_seconds=quantiles(complete.holding_s), complete_net_R=quantiles(complete.net_R),
                holding_under_seconds={str(s): float((complete.holding_s < s).mean()) if len(complete) else None for s in (1, 5, 10, 30, 60)},
                by_reason=reasons)


def cost_stats(frame, fee, gain, stop):
    risk = frame.risk_R.to_numpy(float)
    decision = (frame.decision_spread.to_numpy(float) + fee) / risk
    entry = (frame.entry_spread.to_numpy(float) + fee) / risk
    valid = np.isfinite(entry) & np.isfinite(risk) & (risk > 0)
    remaining = stop - entry[valid]
    favorable = gain + entry[valid]
    ok = remaining > 0
    ratio = favorable[ok] / remaining[ok]
    # The cached risk was fixed with .30 cost before this experiment.
    cost_floor = 2 * (frame.decision_spread.to_numpy(float) + .30)
    known = np.isfinite(risk)
    return dict(decision_cost_over_R=quantiles(decision), entry_cost_over_R=quantiles(entry),
                risk_R_usd=quantiles(risk), frozen_cost_floor_equal_fraction=float(np.isclose(risk[known], cost_floor[known]).mean()),
                remaining_stop_distance_R=quantiles(remaining), required_take_distance_R=quantiles(favorable),
                take_to_stop_liquidation_quote_distance_ratio=quantiles(ratio),
                initial_cost_at_or_beyond_stop_fraction=float((remaining <= 0).mean()) if len(remaining) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/opportunity_pipeline_v1.json")
    parser.add_argument("--output", default="reports/scientific_pipeline_v1/data_diagnostics.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    market = load_market(config)
    frame = market["frame"]
    names = market["feature_names"]
    if names != list(RICH_FEATURES + GEOMETRY_FEATURES):
        raise ValueError("This diagnostic is the preregistered 81-feature rich inventory")
    raw = market["x_raw"]
    directory = Path(config["data"]["execution_dir"])
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise ValueError("Incomplete simulation")
    sources = dict(market["receipts"]["source_sha256"])
    sources[str(Path(args.config).resolve())] = sha(args.config)
    sources[str((directory / "manifest.json").resolve())] = sha(directory / "manifest.json")
    pieces = []
    for rec in manifest["days"]:
        if rec["date"] not in config["data"]["dates"]:
            continue
        path = directory / (rec["date"] + ".parquet")
        digest = sha(path)
        if digest != rec["sha256"]:
            raise ValueError("Changed saved execution")
        sources[str(path.resolve())] = digest
        pieces.append(pd.read_parquet(path))
    execution = pd.concat(pieces, ignore_index=True)
    keys = ["date", "t_ns"]
    long = execution.loc[execution.side == 1].set_index(keys)
    short = execution.loc[execution.side == -1].set_index(keys)
    origin_index = pd.MultiIndex.from_frame(frame[keys])
    long, short = long.reindex(origin_index), short.reindex(origin_index)
    if long.status.isna().any() or short.status.isna().any():
        raise ValueError("Missing paired simulation rows")
    entered = long.entry_ns.to_numpy() > 0
    if not np.array_equal(long.entry_ns.to_numpy()[entered], frame.entry_ns.to_numpy()[entered]):
        raise ValueError("Cached entry spread belongs to a different latency/entry; save current entry BBO before computing cost distances")
    if not np.allclose(long.risk_R.to_numpy(), frame.risk_R.to_numpy(), equal_nan=True):
        raise ValueError("Execution risk differs from causal source risk")
    for name, value in manifest["identity"]["execution_config"].items():
        configured = config.get("execution", {}).get(name, config.get("target", {}).get(name, value))
        if value != configured:
            raise ValueError(f"Execution config differs from saved run: {name}")
    known = (long.status.to_numpy() == "COMPLETE") & (short.status.to_numpy() == "COMPLETE")
    threshold = float(config["target"].get("minimum_net_R", 0)) * frame.risk_R.to_numpy()
    y = np.where(known, (long.net.to_numpy() > threshold).astype(int) + 2 * (short.net.to_numpy() > threshold).astype(int), -1)
    admitted = frame.feature_available.to_numpy(bool)
    y_admitted = np.where(admitted, y, -1)
    gain = manifest["identity"]["execution_config"]["gain_multiple"]
    stop = manifest["identity"]["execution_config"]["stop_multiple"]
    fee = manifest["identity"]["execution_config"]["fee_roundtrip"]
    daily = {}
    for date in config["data"]["dates"]:
        use = frame.date.to_numpy() == date
        f = frame.loc[use].reset_index(drop=True)
        daily[date] = dict(labels_all_clock_roots=label_stats(f, y[use]),
                           labels_feature_admitted=label_stats(f, y_admitted[use]),
                           execution=execution_stats(execution.loc[execution.date == date], gain, stop),
                           costs=cost_stats(f, fee, gain, stop), features=feature_stats(raw[use], names, admitted[use]))
    common = (y >= 0) & (market["y_crossing"] >= 0) & admitted
    crossing_y = market["y_crossing"]
    comparison = dict(common_roots=int(common.sum()),
                      event_agreement=float(((y[common] > 0) == (crossing_y[common] > 0)).mean()),
                      state_agreement=float((y[common] == crossing_y[common]).mean()),
                      crossing_event_execution_none=int(((crossing_y > 0) & (y == 0) & common).sum()),
                      crossing_none_execution_event=int(((crossing_y == 0) & (y > 0) & common).sum()))
    result = dict(status="COMPLETE_DIAGNOSTIC_NOT_POLICY_ADMISSION", source_sha256=sources,
                  analysis_script_sha256=sha(__file__),
                  definitions=dict(
                      execution_target="Joint 0=NONE,1=LONG_ONLY,2=SHORT_ONLY,3=BOTH when both branches COMPLETE; success net > minimum_net_R*causal R; all other statuses unknown",
                      label_runs="Observed equal labels sampled every30s; run measure=count*30s, not a continuously observed economic duration. Unknown/date/gap ends a run. Internal runs have known labels at both boundaries.",
                      lags="Within the same day and exact timestamp lag; endpoints must both have mature labels and admitted causal features. No circular shift, cross-day pairs, fitted model or predictive interpretation.",
                      overshoot="Executed net minus nominal +G or -R, divided by causal R. Combines first threshold overshoot and close latency; it is not pure latency slippage because trigger price is not saved.",
                      costs="C=(entry spread+roundtrip fee)/R is initial negative liquidation mark. Relative liquidation quote must improve by G/R+C to take; adverse allowance is stop_multiple-C. Quote-side move, not an assumption of constant spread midpoint dynamics.",
                      feature_missing="Per-feature and family fractions use source-admitted roots; missing source origins reported separately, not filled or dropped silently.",
                      execution_counts="Two independent alternative side trials per origin; these outcomes are not summable account equity.",
                      risk="Frozen source R=max(2*(decision spread+.30),.10*decision BB1 half,.50); no thresholds tuned in this diagnostic.",
                      sampling="Five previously inspected days; descriptive diagnostics, no inference of independent sample count or out-of-sample trading value."),
                  daily=daily,
                  totals=dict(labels_all_clock_roots=label_stats(frame, y),
                              labels_feature_admitted=label_stats(frame, y_admitted),
                              execution=execution_stats(execution, gain, stop), costs=cost_stats(frame, fee, gain, stop),
                              features=feature_stats(raw, names, admitted)),
                  intraday_label_lags=label_lags(frame, y_admitted),
                  crossing_amplitude_comparison=comparison,
                  crossing_amplitude_label_stats=label_stats(frame, np.where(admitted, crossing_y, -1)),
                  feature_allowlist=names)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(output=str(output), sha256=sha(output), labels=result["totals"]["labels_feature_admitted"]["classes"],
                          flips=result["totals"]["labels_feature_admitted"]["event_flip_30s"],
                          holding=result["totals"]["execution"]["complete_holding_seconds"],
                          costs=result["totals"]["costs"])))


if __name__ == "__main__":
    main()
