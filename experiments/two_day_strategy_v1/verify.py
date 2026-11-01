"""Independent saved-run audit; does not call the selector or portfolio engine.

Native simulator equality is checked through its saved 20 receipts. This audit
rehashes their sources and independently reconstructs entry masks, pair ranks,
forward decisions, and scalar account transitions from saved consequences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NS = 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def same_number(actual, expected, message):
    require(np.isclose(float(actual), float(expected), rtol=0, atol=1e-8),
            f"{message}: {actual!r} != {expected!r}")


def independent_masks(frame, rules, threshold):
    """Explicit X whitelist; no label, status, PnL or next-day input accepted."""
    fields = ["past_known", "x_base_mbo_now_known",
              "x_mbo_mbo_now_trade_unknown_fraction",
              "x_mbo_mbo_now_trade_imbalance",
              "x_mbo_mbo_now_depth_1_toward_imbalance",
              "x_base_mbo_now_trade_total_log", "past_net30_r", "past_migration_r",
              "h_memory_return", "h_permeable_departure", "h_failed_escape",
              "h_place_migration"]
    x = frame[fields].copy()
    known = x.past_known.eq(True)
    flow = known & x.x_base_mbo_now_known.eq(1)
    flow &= x.x_mbo_mbo_now_trade_unknown_fraction.le(rules["max_unknown_trade_fraction"])
    net = x.past_net30_r
    imbalance = x.x_mbo_mbo_now_trade_imbalance
    depth = x.x_mbo_mbo_now_depth_1_toward_imbalance
    toward = net.ge(rules["progress_r"])
    away = net.le(-rules["progress_r"])
    masks = {"toward_all": np.ones(len(x), bool), "away_all": np.ones(len(x), bool),
             "WAIT": np.zeros(len(x), bool)}
    for name in ("memory_return", "permeable_departure", "failed_escape", "place_migration"):
        masks[name] = x["h_"+name].eq(True)
    masks["flow_return"] = flow & imbalance.ge(rules["flow_imbalance"]) & toward & depth.ge(0)
    masks["flow_away"] = flow & imbalance.le(-rules["flow_imbalance"]) & away & depth.le(0)
    masks["stalled_flow_return"] = (flow & imbalance.le(-rules["flow_imbalance"])
        & net.abs().le(rules["stalled_abs_progress_r"]) & depth.ge(0)
        & x.x_base_mbo_now_trade_total_log.ge(threshold))
    masks["aligned_return"] = known & toward & x.past_migration_r.ge(rules["aligned_migration_r"])
    return pd.DataFrame(masks, index=frame.index).fillna(False).astype(bool), fields


def independently_rank_pairs(scores, selections, config, catalog):
    dates = config["dates"]; minimum = config["selection"]["min_trades_each_day"]
    require(len(selections) == len(dates)-1, "Wrong number of pair selections")
    require(not scores.duplicated(["pair_start", "pair_end", "strategy"]).any(), "Duplicate pair score")
    require(not selections.duplicated(["pair_start", "pair_end"]).any(), "Duplicate pair selection")
    for i in range(1, len(dates)):
        start, end = dates[i-1:i+1]
        table = scores[scores.pair_start.eq(start) & scores.pair_end.eq(end)]
        require(set(table.strategy) == set(catalog), f"Missing strategy in pair {start}")
        selected = selections[selections.pair_start.eq(start) & selections.pair_end.eq(end)]
        require(len(selected) == 1, f"Missing pair selection {start}")
        selected = selected.iloc[0]
        cutoff = int((pd.Timestamp(end, tz="UTC")+pd.Timedelta(days=1)).value)
        require(int(selected.selection_ns) == cutoff, "Selection clock mismatch")
        expected_next = dates[i+1] if i+1 < len(dates) else ""
        require(selected.next_date == expected_next, "Selection next-day mismatch")
        admissible = []; raw = []
        for row in table.itertuples(index=False):
            support = row.base_day1_trades >= minimum and row.base_day2_trades >= minimum
            eligible = (bool(row.base_complete) and bool(row.stress_complete) and support
                and row.stress_day1_trades >= minimum and row.stress_day2_trades >= minimum
                and all(getattr(row, f"{cost}_day{day}_net_usd") > 0
                        for cost in ("base", "stress") for day in (1, 2)))
            score = min(row.stress_day1_net_usd, row.stress_day2_net_usd)
            require(bool(row.eligible) == eligible, "Saved eligibility differs from independent gate")
            same_number(row.score_usd, score, "Saved robust score")
            for cost in ("base", "stress"):
                require(getattr(row, cost+"_max_feedback_ns") < cutoff, "Unmatured feedback")
            if row.strategy != "WAIT":
                if eligible:
                    admissible.append((float(score), row.strategy))
                if row.base_complete and support:
                    raw.append((float(row.base_net_usd), row.strategy))
        admissible.sort(key=lambda item: (-item[0], item[1]))
        raw.sort(key=lambda item: (-item[0], item[1]))
        winner = admissible[0][1] if admissible else "WAIT"
        runner = admissible[1][1] if len(admissible) > 1 else "WAIT"
        gap = admissible[0][0]-(admissible[1][0] if len(admissible) > 1 else 0.) if admissible else 0.
        require(selected.selected_strategy == winner, "Saved robust winner mismatch")
        require(selected.raw_leader == (raw[0][1] if raw else "WAIT"), "Saved raw leader mismatch")
        require(selected.runner_up == runner and selected.eligible_count == len(admissible), "Rank support mismatch")
        same_number(selected.score_gap_usd, gap, "Saved selection margin")


def policy_strategy(policy, selection, first):
    reference = policy.removeprefix("reverse_")
    mapping = {"adaptive": selection.selected_strategy,
               "raw_leader": selection.raw_leader,
               "frozen_first_pair": first.selected_strategy,
               "frozen_first_raw_pair": first.raw_leader,
               "always_toward": "toward_all", "always_away": "away_all", "WAIT": "WAIT"}
    require(reference in mapping, f"Unknown account policy: {policy}")
    return mapping[reference]


def audit_account(policy, condition, candidates, outcomes, logs, saved, config):
    """Scalar transition audit from source rows, with no production imports."""
    cfg = config["account"]; initial = float(cfg["initial_capital_usd"])
    cash = peak = initial; worst_drawdown = 0.; busy = -1
    current_day = None; day_cash = initial; day_wait = False
    counts = dict(offered=len(candidates), overlap=0, risk_rejected=0, wait_rejected=0,
                  empty=0, ineligible=0, censored=0, unresolved=0, halted=False)
    expected = []
    for c in candidates.sort_values(["t_ns", "candidate_id"]).itertuples(index=False):
        if c.date != current_day:
            current_day = c.date; day_cash = cash; day_wait = False
        if c.t_ns <= busy:
            counts["overlap"] += 1; continue
        if (day_wait or day_cash-cash >= cfg["daily_loss_fraction"]*day_cash
                or worst_drawdown >= cfg["max_drawdown_fraction"]*initial):
            counts["wait_rejected"] += 1; continue
        side = int(c.execution_side)
        r = outcomes[(int(c.origin_index), side)]
        risk = float(c.risk_per_oz)*cfg["contract_oz"]
        same_number(r.risk_per_oz, c.risk_per_oz, "Decision risk provenance")
        if risk > max(0., cash)*cfg["max_planned_risk_fraction"]:
            counts["risk_rejected"] += 1; continue
        if r.status == "INELIGIBLE":
            counts["ineligible"] += 1; continue
        if r.status == "EMPTY":
            counts["empty"] += 1
            busy = int(c.t_ns + config["execution"]["pending_timeout_seconds"]*NS)
            continue
        if r.status != "COMPLETE":
            counts["censored"] += 1
            if r.entry_ns > 0:
                counts["unresolved"] += 1; counts["halted"] = True; break
            day_wait = True; continue
        amount = float(r.net_per_oz)*cfg["contract_oz"]
        worst_drawdown = max(worst_drawdown, peak-(cash+r.mae_per_oz*cfg["contract_oz"]),
                             r.max_drawdown_per_oz*cfg["contract_oz"])
        peak = max(peak, cash+max(0., r.mfe_per_oz)*cfg["contract_oz"])
        cash += amount; peak = max(peak, cash); busy = int(r.exit_ns)
        expected.append(dict(candidate_id=int(c.candidate_id), origin_index=int(c.origin_index),
            side=side, qty=1, date=c.date, signal_ns=int(c.t_ns), entry_ns=int(r.entry_ns),
            exit_ns=int(r.exit_ns), reason=r.reason, pnl_usd=amount,
            net_per_oz=float(r.net_per_oz), r_multiple=float(r.net_per_oz/r.risk_per_oz),
            equity_usd=cash, max_drawdown_usd=worst_drawdown, planned_risk_usd=risk,
            holding_seconds=(r.exit_ns-r.entry_ns)/NS,
            original_signal_side=int(c.side), selected_strategy=c.audit_strategy))
    actual = logs[logs.policy.eq(policy) & logs.condition.eq(condition)].reset_index(drop=True)
    require(len(actual) == len(expected) == int(saved["trades"]), "Account trade count mismatch")
    for i, expected_row in enumerate(expected):
        actual_row = actual.iloc[i]
        for field, value in expected_row.items():
            if isinstance(value, float):
                same_number(actual_row[field], value, f"{policy}/{condition} row{i} {field}")
            else:
                require(actual_row[field] == value, f"{policy}/{condition} row{i} {field} mismatch")
    if len(actual) > 1:
        require((actual.signal_ns.to_numpy()[1:] > actual.exit_ns.to_numpy()[:-1]).all(), "Position overlap")
    for key, value in counts.items():
        require(saved.get(key, False if key == "halted" else 0) == value, f"{policy} count {key}")
    same_number(saved["net_usd"], cash-initial, "Continuous net PnL")
    same_number(saved["terminal_cash_usd"], cash, "Continuous terminal equity")
    same_number(saved["max_drawdown_usd"], worst_drawdown, "Continuous path drawdown")
    same_number(saved["exposure_seconds"], sum(r["holding_seconds"] for r in expected), "Exposure")
    require(saved["contracts"] == len(expected), "Fixed one-contract sizing")
    require(saved["active_days"] == len({r["date"] for r in expected}), "Active day count")
    require(bool(saved["capital_curve_complete"]) == (counts["unresolved"] == 0), "Censoring completeness")
    return dict(policy=policy, condition=condition, offered=len(candidates), trades=len(expected),
                complete=counts["unresolved"] == 0, net_usd=cash-initial)


def verify(run, *, allow_code_drift=False):
    started = time.perf_counter(); run = Path(run)
    if not run.is_absolute():
        run = ROOT/run
    manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "COMPLETE", "Incomplete saved run")
    counts = {"artifacts": 0, "sources": 0, "implementation": 0}; code_drift = []
    for filename, expected in manifest["artifacts"].items():
        require(digest(run/filename) == expected, f"Artifact hash mismatch: {filename}")
        counts["artifacts"] += 1
    for filename, expected in manifest["source_hashes"].items():
        require(digest(filename) == expected, f"Source hash mismatch: {filename}")
        counts["sources"] += 1
    for filename, expected in manifest["implementation_sha256"].items():
        if digest(ROOT/filename) != expected:
            code_drift.append(filename)
        counts["implementation"] += 1
    require(allow_code_drift or not code_drift, f"Implementation changed: {code_drift}")
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    catalog = summary["strategy_catalog"]
    f = pd.read_parquet(run/"observations.parquet")
    masks = pd.read_parquet(run/"masks.parquet").set_index("candidate_id")
    pairs = pd.read_parquet(run/"pair_scores.parquet")
    selections = pd.read_parquet(run/"selections.parquet")
    decisions = pd.read_parquet(run/"decisions.parquet")
    logs = pd.read_parquet(run/"trades.parquet")
    require(not f.candidate_id.duplicated().any(), "Duplicate source candidates")
    require(len(f) == summary["opportunities"], "Opportunity count")
    require(sorted(f.date.unique()) == config["dates"], "Study date coverage")
    source = pd.read_parquet(ROOT/config["atlas"]/"observations.parquet")
    inherited = source[source.date.isin(config["dates"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(f, inherited, check_exact=True)
    init = source[source.date.between("2026-08-03", "2026-08-14")]
    require(len(init) == summary["initialization_rows"], "Initialization row count")
    require(init.date.max() < config["dates"][0], "Threshold prefix overlaps study")
    threshold = float(init.x_base_mbo_now_trade_total_log.median())
    same_number(summary["activity_log_threshold"], threshold, "Past activity threshold")
    rebuilt, whitelist = independent_masks(f, config["rules"], threshold)
    require(set(rebuilt.columns) == set(catalog) == set(masks.columns), "Mask/catalog mismatch")
    require(set(masks.index) == set(f.candidate_id), "Mask population mismatch")
    for name in rebuilt:
        require(np.array_equal(masks.loc[f.candidate_id, name].to_numpy(), rebuilt[name].to_numpy()),
                f"Mask mismatch: {name}")
    source_clock_columns = ["target_available_ns", "episode_available_ns", "memory_available_ns",
        "bar_context_available_ns", "past_max_available_ns", "mbometa_now_max_source_available_ns"]
    for field in source_clock_columns:
        require((f[field] <= f.t_ns).all(), f"Future X clock: {field}")
    independently_rank_pairs(pairs, selections, config, catalog)
    forward = selections[selections.next_date.ne("")]
    require(len(forward) == len(config["dates"])-2, "Forward day count")
    require(not decisions.candidate_id.duplicated().any(), "Duplicate deployment decision")
    population = f[f.date.isin(forward.next_date)]
    require(set(decisions.candidate_id) == set(population.candidate_id), "Decision population differs")
    by_id = f.set_index("candidate_id")
    for decision in decisions.itertuples(index=False):
        choice = forward[forward.next_date.eq(decision.date)]
        require(len(choice) == 1, "Missing next-day selection")
        choice = choice.iloc[0]; origin = by_id.loc[decision.candidate_id]
        require(decision.t_ns == origin.t_ns and decision.side == origin.side, "Decision identity mismatch")
        require(decision.selection_ns == choice.selection_ns < decision.t_ns, "Decision cutoff mismatch")
        require(decision.pair_start == choice.pair_start and decision.pair_end == choice.pair_end,
                "Decision history mismatch")
        require(decision.pair_end < decision.date, "Future pair used for decision")
        require(decision.selected_strategy == choice.selected_strategy, "Unfrozen deployed strategy")
        require(bool(decision.accepted) == bool(masks.loc[decision.candidate_id, choice.selected_strategy]),
                "Deployed acceptance mismatch")
        require(decision.chosen_side == origin.side*catalog[choice.selected_strategy]["direction"],
                "Deployed side mismatch")
    outcomes = {}; statuses = {}; outcome_rows = {}
    for condition in ("base", "stress"):
        table = pd.read_parquet(run/(condition+"_outcomes.parquet"))
        require(len(table) == 2*len(f) and not table.duplicated(["origin_index", "side"]).any(),
                "Missing or duplicate side outcomes")
        require(set(table.candidate_id) == set(f.candidate_id), "Outcome candidate set mismatch")
        require(table.plan.eq("zone_target_300s").all(), "Wrong outcome plan")
        outcomes[condition] = {(int(r.origin_index), int(r.side)): r for r in table.itertuples(index=False)}
        statuses[condition] = table.status.value_counts().to_dict()
        require(statuses[condition] == summary["source_outcome_status"][condition], "Status summary mismatch")
        outcome_rows[condition] = len(table)
    checks = []; first = selections.iloc[0]
    result_keys = [(r["policy"], r["condition"]) for r in summary["results"]]
    require(len(set(result_keys)) == len(result_keys), "Duplicate account summary")
    if len(logs):
        require(set(zip(logs.policy, logs.condition)).issubset(set(result_keys)), "Unreported trade log")
    for saved in summary["results"]:
        policy = saved["policy"]; parts = []
        for choice in forward.itertuples(index=False):
            strategy = policy_strategy(policy, choice, first)
            day = f[f.date.eq(choice.next_date)]
            selected_ids = day.candidate_id[masks.loc[day.candidate_id, strategy].to_numpy()]
            selected = day[day.candidate_id.isin(selected_ids)].copy()
            sign = catalog[strategy]["direction"] * (-1 if policy.startswith("reverse_") else 1)
            selected["execution_side"] = selected.side*sign
            selected["audit_strategy"] = strategy
            parts.append(selected)
        candidates = pd.concat(parts, ignore_index=True)
        checks.append(audit_account(policy, saved["condition"], candidates, outcomes[saved["condition"]],
                                    logs, saved, config))
    native = summary["native_rechecks"]
    require(len(native) == 2*len(config["dates"]), "Wrong native replay receipt count")
    expected_keys = {(date, condition) for date in config["dates"] for condition in ("base", "stress")}
    require({(r["date"], r["condition"]) for r in native} == expected_keys, "Native receipt date coverage")
    for row in native:
        require(row["exact"] and row["fields"] == 10, "Unsuccessful native replay receipt")
        require(row["rows"] == 2*int(f.date.eq(row["date"]).sum()), "Native replay population mismatch")
    receipt = dict(status="PASS_SAVED_RUN_AUDIT" if not code_drift else "PASS_SAVED_DATA_CODE_DRIFT",
        run=str(run), manifest_sha256=digest(run/"manifest.json"), verifier_sha256=digest(__file__),
        hash_counts=counts, implementation_drift=code_drift, opportunities=len(f), masks_checked=len(rebuilt.columns),
        mask_x_whitelist=whitelist, source_clock_columns=source_clock_columns, pairs_checked=len(selections),
        decisions_checked=len(decisions), outcome_rows=outcome_rows, native_receipts_checked=len(native),
        continuous_account_checks=checks, elapsed_seconds=time.perf_counter()-started,
        limits=["Native BBO simulator equality is evidenced by saved receipts and rehashed source payloads; this verifier does not rerun the native simulator.",
                "Inherited atlas features and flags are matched byte-for-byte in dataframe values; their underlying feature construction is not independently recomputed here.",
                "Continuous deterministic accounts are independently reconstructed; stochastic control summaries and visual presentation are outside this audit.",
                "Passing this audit establishes consistency and the checked chronology, not predictive or economic advantage."])
    (run/"verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--allow-code-drift", action="store_true",
                        help="Audit older saved data while explicitly reporting changed implementation files.")
    args = parser.parse_args()
    receipt = verify(args.run, allow_code_drift=args.allow_code_drift)
    print(json.dumps({k: receipt[k] for k in ("status", "run", "hash_counts", "pairs_checked", "decisions_checked")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
