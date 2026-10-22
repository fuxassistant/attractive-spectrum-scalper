"""Calibration-only risk selection and chronological one-position replay.

Joint crossing probabilities describe NONE/LONG/SHORT/BOTH. A side is selected
by its marginal P(side), with LONG winning an exact tie. Future label maturity
never gates replay. Paired risk comparisons are explicitly oracle diagnostics,
not capital curves: their common cohort conditions on observed future outcomes.
Money is USD per reference ounce under the simulator's taker BBO assumptions.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

NS = 1_000_000_000
STATUSES = ("INELIGIBLE", "EMPTY", "CENSORED", "COMPLETE")
EXIT_REASONS = ("TAKE", "STOP", "TIME")


def _hash_keys(date, t):
    h = hashlib.sha256()
    h.update(np.asarray(t, dtype="<i8").tobytes())
    h.update(json.dumps(list(map(str, date)), separators=(",", ":")).encode())
    return h.hexdigest()


def _stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, q10=None, q50=None, q90=None)
    q = np.quantile(a, [.1, .5, .9])
    return dict(n=int(len(a)), mean=float(a.mean()), q10=float(q[0]),
                q50=float(q[1]), q90=float(q[2]))


def _aligned(frame, executions, p, config):
    required = {"date", "t_ns", "feature_available", "y_crossing", "R_event"}
    if not required <= set(frame) or not len(frame):
        raise ValueError("Nonempty origin frame with causal availability and crossing labels required")
    if frame.feature_available.dtype != bool:
        raise ValueError("feature_available must be boolean")
    for table, cols in ((frame, ("t_ns", "y_crossing")),
                        (executions, ("t_ns", "side", "entry_ns", "exit_ns"))):
        for name in cols:
            if name not in table or not np.issubdtype(table[name].dtype, np.integer):
                raise ValueError("Integer column required: " + name)
    t = frame.t_ns.to_numpy(np.int64)
    date = frame.date.astype(str).to_numpy()
    y = frame.y_crossing.to_numpy()
    if np.any(np.diff(t) <= 0) or not np.isin(y, [-1, 0, 1, 2, 3]).all():
        raise ValueError("Unique chronological origins and labels -1..3 required")
    if np.any(date[1:] < date[:-1]):
        raise ValueError("Origin dates must be chronological")
    p = np.asarray(p, float)
    if (p.shape != (len(t), 4) or not np.isfinite(p).all() or np.any(p < 0)
            or np.any(p > 1) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Finite normalized nonnegative joint probabilities [N,4] required")
    multipliers = np.asarray(config["risk"]["cost_multipliers"], float)
    if (not len(multipliers) or not np.isfinite(multipliers).all()
            or np.any(multipliers <= 0) or len(np.unique(multipliers)) != len(multipliers)):
        raise ValueError("Unique positive finite risk cost multipliers required")
    plans = [f"cost{m:g}" for m in multipliers]
    required_exec = {"date", "plan", "status", "reason", "net", "mae", "mfe", "R_trade", "R_event", "G_event"}
    if not required_exec <= set(executions):
        raise ValueError("Missing execution columns: " + str(sorted(required_exec-set(executions))))
    n, k = len(t), len(plans)
    if len(executions) != k*n*2:
        raise ValueError("Every plan requires exactly both sides of every origin")
    et = executions.t_ns.to_numpy(np.int64)
    root = np.searchsorted(t, et)
    if np.any(root >= n) or not np.array_equal(t[np.minimum(root, n-1)], et):
        raise ValueError("Execution origin clocks differ from the root frame")
    if not np.array_equal(date[root], executions.date.astype(str).to_numpy()):
        raise ValueError("Execution origin dates differ from the root frame")
    if not np.isin(executions.side, [-1, 1]).all():
        raise ValueError("Execution sides must be +1 or -1")
    plan_index = pd.Categorical(executions.plan, categories=plans).codes.astype(np.int64)
    if np.any(plan_index < 0):
        raise ValueError("Unknown execution risk plan")
    side_index = (executions.side.to_numpy() == -1).astype(np.int64)
    key = (plan_index*n+root)*2+side_index
    order = np.argsort(key)
    if not np.array_equal(key[order], np.arange(k*n*2)):
        raise ValueError("Duplicate or missing execution root/side/plan")
    columns = ["status", "reason", "net", "mae", "mfe", "entry_ns", "exit_ns", "R_trade", "R_event", "G_event"]
    columns += [c for c in ("trigger_ns", "entry_ordinal", "exit_ordinal", "trigger_ordinal") if c in executions]
    a = {name: executions[name].to_numpy()[order].reshape(k, n, 2) for name in columns}
    if not np.isin(a["status"], STATUSES).all():
        raise ValueError("Unknown execution status")
    complete = a["status"] == "COMPLETE"
    if (not np.isin(a["reason"][complete], EXIT_REASONS).all()
            or np.any(a["entry_ns"][complete] <= np.broadcast_to(t[None, :, None], complete.shape)[complete])
            or np.any(a["exit_ns"][complete] <= a["entry_ns"][complete])):
        raise ValueError("Malformed COMPLETE execution reason or clocks")
    for name in ("net", "mae", "mfe", "R_trade", "R_event", "G_event"):
        if not np.isfinite(a[name][complete]).all():
            raise ValueError("Nonfinite COMPLETE execution " + name)
    if (np.any(a["R_trade"][complete] <= 0) or np.any(a["R_event"][complete] <= 0)
            or np.any(a["G_event"][complete] <= 0)
            or np.any(a["mae"][complete] > a["net"][complete]+1e-9)
            or np.any(a["mfe"][complete] < a["net"][complete]-1e-9)):
        raise ValueError("Malformed COMPLETE risk or excursion envelope")
    event_r = frame.R_event.to_numpy(float)
    event_g = (frame.G_event.to_numpy(float) if "G_event" in frame else
               event_r*float(config["event_query"]["gain_multiple"]))
    for name, expected in (("R_event", event_r), ("G_event", event_g)):
        if not np.allclose(a[name], expected[None, :, None], rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError("Risk plans changed the fixed event reference: " + name)
    for j, plan in enumerate(plans):
        if "R_"+plan in frame and not np.allclose(a["R_trade"][j], frame["R_"+plan].to_numpy()[:, None],
                                                  rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError("Execution risk does not match root risk plan")
    splits = config["splits"]
    split_dates = [str(d) for dates in splits.values() for d in dates]
    if len(split_dates) != len(set(split_dates)):
        raise ValueError("Chronological splits overlap")
    if not splits.get("calibration") or not splits.get("test"):
        raise ValueError("Nonempty calibration and test date blocks required")
    if max(splits["calibration"]) >= min(splits["test"]):
        raise ValueError("Calibration must precede test")
    if not set(split_dates) <= set(date):
        raise ValueError("Configured split dates are missing from origin frame")
    key_hash = _hash_keys(date, t)
    proof = dict(roots=n, plans=plans, sides=[1, -1], executions=len(executions),
                 root_key_sha256=key_hash,
                 plan_root_key_sha256={plan: key_hash for plan in plans},
                 aligned_complete_validation="all COMPLETE rows, including unused branches",
                 probabilities_sha256=hashlib.sha256(p.astype("<f8").tobytes()).hexdigest())
    marginal = np.column_stack((p[:, 1]+p[:, 3], p[:, 2]+p[:, 3]))
    side = (marginal[:, 1] > marginal[:, 0]).astype(np.int64)
    return dict(t=t, date=date, y=y, available=frame.feature_available.to_numpy(),
                arrays=a, plans=plans, multipliers=multipliers, score=marginal.max(1), side=side,
                proof=proof, event_r=event_r, event_g=event_g)


def _projection(data, plan_index, threshold, days, pending_seconds, keep_ledger=False):
    a, t, date = data["arrays"], data["t"], data["date"]
    allowed = data["available"] & np.isin(date, days)
    candidate = np.flatnonzero(allowed & (data["score"] >= threshold))
    ledger, daily = [], {str(d): dict(trades=0, net=0., ineligible=0, empty=0, censored=0) for d in days}
    cash = peak = max_dd = 0.
    busy_until = -1
    counts = dict(trades=0, ineligible=0, empty=0, censored=0, skipped_busy=0)
    reasons = {reason: 0 for reason in EXIT_REASONS}
    wins = 0
    censor = None
    processed = 0
    for i in candidate:
        processed += 1
        now = int(t[i])
        if now <= busy_until:  # An exit at the decision timestamp cannot fund a same-clock reentry.
            counts["skipped_busy"] += 1
            continue
        si = int(data["side"][i])
        key = plan_index, i, si
        status = str(a["status"][key])
        day = daily[str(date[i])]
        if status == "INELIGIBLE":
            counts["ineligible"] += 1
            day["ineligible"] += 1
            continue
        item = dict(date=str(date[i]), t_ns=str(now), side=1 if si == 0 else -1,
                    probability=float(data["score"][i]), status=status, reason=str(a["reason"][key]))
        if status == "EMPTY":
            busy_until = now+int(round(pending_seconds*NS))
            counts["empty"] += 1
            day["empty"] += 1
            item.update(busy_until_ns=str(busy_until), net=None, closed_cash=cash)
        elif status == "CENSORED":
            counts["censored"] += 1
            day["censored"] += 1
            censor = dict(date=str(date[i]), t_ns=str(now), side=item["side"], reason=item["reason"])
            item.update(net=None, closed_cash=cash)
            if keep_ledger:
                ledger.append(item)
            break
        else:
            net = float(a["net"][key])
            cash += net
            peak = max(peak, cash)
            max_dd = max(max_dd, peak-cash)
            busy_until = int(a["exit_ns"][key])
            counts["trades"] += 1
            wins += net > 0
            reasons[item["reason"]] += 1
            day["trades"] += 1
            day["net"] += net
            item.update(entry_ns=str(int(a["entry_ns"][key])), exit_ns=str(busy_until),
                        net=net, mae=float(a["mae"][key]), mfe=float(a["mfe"][key]),
                        R_trade=float(a["R_trade"][key]), R_event=float(a["R_event"][key]),
                        G_event=float(a["G_event"][key]), closed_cash=cash,
                        closed_drawdown=peak-cash)
        if keep_ledger:
            ledger.append(item)
    summary = dict(plan=data["plans"][plan_index], threshold=float(threshold), completed=censor is None,
                   net=cash, closed_max_drawdown=max_dd, wins=int(wins),
                   win_rate=float(wins/counts["trades"]) if counts["trades"] else None,
                   origin_candidates=len(candidate), causal_feature_origins=int(allowed.sum()),
                   unprocessed_after_censor=len(candidate)-processed, halt=censor,
                   exit_reasons=reasons, daily=daily, **counts)
    return summary, ledger


def _paired(frame, data, config):
    a, y = data["arrays"], data["y"]
    common = data["available"] & (y >= 0) & (a["status"] == "COMPLETE").all(axis=(0, 2))
    cost = (frame.decision_spread.to_numpy(float)+float(config["execution"]["fee_roundtrip"])
            if "decision_spread" in frame else np.full(len(frame), np.nan))

    def describe(j, roots):
        net, risk, gain = (a[name][j, roots] for name in ("net", "R_trade", "G_event"))
        n = int(roots.sum())
        reason = a["reason"][j, roots]
        c = cost[roots, None]
        result = dict(roots=n, side_trials=2*n, net=_stats(net), R_trade=_stats(risk),
            net_R_trade=_stats(net/risk), net_R_event=_stats(net/a["R_event"][j, roots]),
            holding_seconds=_stats((a["exit_ns"][j, roots]-a["entry_ns"][j, roots])/NS),
            exit_reasons={r: int((reason == r).sum()) for r in EXIT_REASONS},
            decision_cost_R_trade=_stats(c/risk), fixed_gain_R_trade=_stats(gain/risk),
            decision_proxy_take_distance=_stats(gain+c),
            decision_proxy_stop_room=_stats(risk-c), conditional_crossing={})
        for si, name in enumerate(("LONG", "SHORT")):
            favorable = roots & ((y & (1 << si)) != 0)
            values = a["net"][j, favorable, si]
            event_gain = a["G_event"][j, favorable, si]
            count = len(values)
            result["conditional_crossing"][name] = dict(roots=count, net=_stats(values),
                positive_fraction=float((values > 0).mean()) if count else None,
                fixed_gain_retention_fraction=float((values >= event_gain).mean()) if count else None,
                take_fraction=float((a["reason"][j, favorable, si] == "TAKE").mean()) if count else None,
                stop_fraction=float((a["reason"][j, favorable, si] == "STOP").mean()) if count else None)
        return result

    plans = {}
    for j, plan in enumerate(data["plans"]):
        daily = {}
        for date in np.unique(data["date"]):
            day = data["date"] == date
            status = a["status"][j, day]
            daily[str(date)] = dict(roots=int(day.sum()),
                feature_origins=int((day & data["available"]).sum()),
                known_crossing_origins=int((day & (y >= 0)).sum()),
                common_roots=int((day & common).sum()),
                status_side_trials={s: int((status == s).sum()) for s in STATUSES})
        plans[plan] = dict(common=describe(j, common),
            by_split={name: describe(j, common & np.isin(data["date"], dates))
                      for name, dates in config["splits"].items()}, daily=daily)
    return dict(kind="oracle_paired_risk_diagnostic_not_policy", common_roots=int(common.sum()),
        common_root_key_sha256=_hash_keys(data["date"][common], data["t"][common]),
        definition="feature_available AND known fixed crossing label AND both sides COMPLETE in every risk plan",
        cost_definition="decision spread + execution roundtrip fee; decision-time distance proxy, not actual entry spread",
        retention_definition="Among true fixed crossing LONG/SHORT labels on the common future-observed cohort: net > 0 or net >= fixed G_event after actual exit latency",
        plans=plans)


def evaluate_risk_grid(frame, executions_long, p, config):
    """Choose one plan/threshold on CAL, then replay only that choice on TEST.

    Inputs must preserve every root, including unknown future labels. An EMPTY
    attempt occupies the pending timeout. A CENSORED attempt halts the entire
    projection; its preceding closed cash is only a partial result. Equity is
    closed-trade cash, without an invented mark-to-market drawdown.
    """
    data = _aligned(frame, executions_long, p, config)
    policy = config["policy"]
    if not policy.get("one_position", True):
        raise ValueError("Only one-position replay is supported")
    thresholds = np.asarray(policy["threshold_candidates"], float)
    if (not len(thresholds) or not np.isfinite(thresholds).all() or np.any(thresholds < 0)
            or len(np.unique(thresholds)) != len(thresholds)):
        raise ValueError("Unique finite nonnegative thresholds required")
    minimum = policy["minimum_calibration_trades"]
    if not isinstance(minimum, (int, np.integer)) or minimum < 1:
        raise ValueError("Positive integer minimum calibration trades required")
    pending = float(config["execution"]["pending_timeout_seconds"])
    if not np.isfinite(pending) or pending <= 0:
        raise ValueError("Positive finite pending timeout required")
    grid = []
    for j, plan in enumerate(data["plans"]):
        for threshold in thresholds:
            result, _ = _projection(data, j, threshold, config["splits"]["calibration"], pending)
            grid.append(result)
    valid = [r for r in grid if r["completed"] and r["trades"] >= minimum and r["net"] > 0]
    if valid:
        by_plan = dict(zip(data["plans"], data["multipliers"]))
        chosen = max(valid, key=lambda r: (r["net"], r["threshold"], -by_plan[r["plan"]]))
        selected = dict(plan=chosen["plan"], threshold=chosen["threshold"], reason="positive_complete_calibration")
    else:
        selected = dict(plan=data["plans"][int(np.argmin(data["multipliers"]))], threshold=1.01,
                        reason="WAIT_no_positive_complete_calibration_with_minimum_trades")
    j = data["plans"].index(selected["plan"])
    projections, ledgers = {}, {}
    for name in ("calibration", "test"):
        projections[name], ledgers[name] = _projection(data, j, selected["threshold"], config["splits"][name], pending, True)
    return dict(summary=dict(selected=selected, **projections,
        policy="max marginal P(LONG)=p1+p3 or P(SHORT)=p2+p3; LONG wins tie; threshold inclusive",
        selection_scope="CAL only; model selection is external and must use validation",
        economic_units="USD per reference ounce; taker BBO, no depth/queue/impact claim",
        drawdown_definition="closed-trade cash peak minus closed cash; excludes intratrade mark-to-market",
        future_unknown_admission="no y_crossing or future maturity filter in replay",
        alignment=data["proof"]), ledgers=ledgers, calibration_grid=grid,
        paired_diagnostics=_paired(frame, data, config))
