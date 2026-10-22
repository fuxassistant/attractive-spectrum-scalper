import copy
import json

import numpy as np
import pandas as pd
import pytest

from spectrum.research_v2.evaluation import evaluate_risk_grid

NS = 10**9


def sample(ncal=4, ntest=4, spacing=10):
    dates = ["2026-05-04"]*ncal+["2026-05-11"]*ntest
    t = np.r_[pd.Timestamp("2026-05-04T08:00Z").value+np.arange(ncal)*spacing*NS,
              pd.Timestamp("2026-05-11T08:00Z").value+np.arange(ntest)*spacing*NS]
    frame = pd.DataFrame(dict(date=dates, t_ns=t, feature_available=True,
                              y_crossing=np.tile([1, 2], (len(t)+1)//2)[:len(t)],
                              R_event=1., G_event=2., decision_spread=.2))
    rows = []
    for m in (2, 4, 6):
        for date, now in zip(dates, t):
            for side in (1, -1):
                net = 1. if side == 1 else -.5
                rows.append(dict(date=date, t_ns=now, plan=f"cost{m}", side=side,
                    status="COMPLETE", reason="TAKE" if net > 0 else "STOP", net=net,
                    mae=-.5, mfe=1., entry_ns=now+200_000_000, exit_ns=now+NS,
                    R_trade=m*.5, R_event=1., G_event=2.))
    executions = pd.DataFrame(rows)
    p = np.tile([.1, .9, 0., 0.], (len(t), 1))
    config = dict(risk=dict(cost_multipliers=[2., 4., 6.]),
                  event_query=dict(gain_multiple=2.),
                  execution=dict(fee_roundtrip=.3, pending_timeout_seconds=2.),
                  splits=dict(calibration=["2026-05-04"], test=["2026-05-11"]),
                  policy=dict(threshold_candidates=[.35, .5, .65, .8, 1.01],
                              minimum_calibration_trades=1, one_position=True))
    return frame, executions, p, config


def change(executions, mask, **values):
    for name, value in values.items():
        executions.loc[mask, name] = value


def test_exact_alignment_shuffled_input_and_deterministic_ties_and_json():
    f, e, p, c = sample()
    a = evaluate_risk_grid(f, e, p, c)
    b = evaluate_risk_grid(f, e.sample(frac=1, random_state=1), p, c)
    assert a == b
    assert a["summary"]["selected"] == dict(plan="cost2", threshold=.8, reason="positive_complete_calibration")
    assert a["summary"]["test"]["net"] == 4.
    assert a["summary"]["test"]["trades"] == 4
    assert isinstance(a["ledgers"]["test"][0]["t_ns"], str)
    assert isinstance(a["ledgers"]["test"][0]["entry_ns"], str)
    json.dumps(a, allow_nan=False)


def test_unknown_future_label_does_not_hide_censor_or_continue_next_day():
    f, e, p, c = sample()
    first_test = 4
    f.loc[first_test, "y_crossing"] = -1
    mask = (e.t_ns == f.t_ns.iloc[first_test]) & (e.side == 1) & (e.plan == "cost2")
    change(e, mask, status="CENSORED", reason="SUPPORT", net=np.nan, mae=np.nan, mfe=np.nan, exit_ns=0)
    result = evaluate_risk_grid(f, e, p, c)
    test = result["summary"]["test"]
    assert not test["completed"] and test["censored"] == 1
    assert test["trades"] == 0 and test["unprocessed_after_censor"] == 3
    assert result["ledgers"]["test"][0]["status"] == "CENSORED"


def test_empty_busy_timeout_and_same_clock_exit_reentry_are_strict():
    f, e, p, c = sample(ncal=1, ntest=7, spacing=1)
    first_test = 1
    mask = (e.t_ns == f.t_ns.iloc[first_test]) & (e.side == 1)
    change(e, mask, status="EMPTY", reason="ENTRY_TIMEOUT", net=np.nan, mae=np.nan, mfe=np.nan,
           entry_ns=0, exit_ns=0)
    result = evaluate_risk_grid(f, e, p, c)
    test = result["summary"]["test"]
    assert test["empty"] == 1
    assert test["trades"] == 2  # t=3,5; t=1,2 pending, t=4,6 exactly exit clock.
    assert test["skipped_busy"] == 4


def test_position_is_not_reset_at_date_boundary():
    f, e, p, c = sample(ncal=1, ntest=2, spacing=1)
    c["splits"]["test"].append("2026-05-12")
    f.loc[2, "date"] = "2026-05-12"
    e.loc[e.t_ns == f.t_ns.iloc[2], "date"] = "2026-05-12"
    # Synthetic adjacent dates deliberately exercise account state, not session clocks.
    mask = (e.t_ns == f.t_ns.iloc[1]) & (e.side == 1)
    change(e, mask, exit_ns=int(f.t_ns.iloc[2]+NS))
    result = evaluate_risk_grid(f, e, p, c)
    assert result["summary"]["test"]["trades"] == 1
    assert result["summary"]["test"]["skipped_busy"] == 1


def test_calibration_choice_ignores_test_outcome_mutation():
    f, e, p, c = sample()
    mask = (e.date == "2026-05-04") & (e.plan == "cost4") & (e.side == 1)
    change(e, mask, net=2., mfe=2.)
    before = evaluate_risk_grid(f, e, p, c)
    mask = (e.date == "2026-05-11") & (e.plan == "cost6")
    change(e, mask, net=10000., mfe=10000.)
    p[4:] = [.05, .05, .9, 0.]
    after = evaluate_risk_grid(f, e, p, c)
    assert before["summary"]["selected"] == after["summary"]["selected"]
    assert before["calibration_grid"] == after["calibration_grid"]
    assert after["summary"]["selected"]["plan"] == "cost4"
    assert after["summary"]["test"]["net"] == -2.


def test_calibration_incomplete_or_insufficient_or_nonpositive_falls_back_to_wait():
    f, e, p, c = sample()
    c["policy"]["minimum_calibration_trades"] = 5
    result = evaluate_risk_grid(f, e, p, c)
    assert result["summary"]["selected"]["threshold"] == 1.01
    assert result["summary"]["test"]["trades"] == 0
    c["policy"]["minimum_calibration_trades"] = 1
    mask = (e.date == "2026-05-04") & (e.side == 1)
    change(e, mask, net=0., mfe=0.)
    assert evaluate_risk_grid(f, e, p, c)["summary"]["selected"]["threshold"] == 1.01
    change(e, mask, status="CENSORED", reason="SUPPORT", net=np.nan, mfe=np.nan, mae=np.nan)
    assert evaluate_risk_grid(f, e, p, c)["summary"]["selected"]["threshold"] == 1.01


def test_both_probability_contributes_to_both_sides_and_long_wins_tie():
    f, e, p, c = sample()
    p[:] = [.1, .1, .1, .7]
    c["policy"]["threshold_candidates"] = [.8, 1.01]
    # Decimal arithmetic may be one ULP under .8; use exactly representable values.
    p[:] = [0., .125, .125, .75]
    result = evaluate_risk_grid(f, e, p, c)
    assert result["summary"]["test"]["trades"] == 4
    assert {row["side"] for row in result["ledgers"]["test"]} == {1}


def test_paired_common_cohort_same_for_all_plans_and_retention_is_separate_from_event():
    f, e, p, c = sample()
    mask = (e.t_ns == f.t_ns.iloc[0]) & (e.plan == "cost6") & (e.side == -1)
    change(e, mask, status="CENSORED", reason="SUPPORT", net=np.nan, mfe=np.nan, mae=np.nan)
    f.loc[1, "feature_available"] = False
    f.loc[2, "y_crossing"] = -1
    result = evaluate_risk_grid(f, e, p, c)
    paired = result["paired_diagnostics"]
    assert paired["common_roots"] == 5
    assert {v["common"]["roots"] for v in paired["plans"].values()} == {5}
    side = paired["plans"]["cost2"]["common"]["conditional_crossing"]["LONG"]
    assert side["positive_fraction"] == 1. and side["fixed_gain_retention_fraction"] == 0.
    assert paired["plans"]["cost6"]["daily"]["2026-05-04"]["status_side_trials"]["CENSORED"] == 1
    assert paired["plans"]["cost2"]["common"]["decision_cost_R_trade"]["q50"] == .5
    assert paired["plans"]["cost6"]["common"]["decision_cost_R_trade"]["q50"] == pytest.approx(1/6)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_date", "wrong_side", "new_gain", "nonfinite_complete", "clock", "excursion", "probability"])
def test_malformed_alignment_complete_or_probabilities_fail_closed(mutation):
    f, e, p, c = sample()
    if mutation == "missing":
        e = e.iloc[:-1]
    elif mutation == "duplicate":
        e.iloc[1] = e.iloc[0]
    elif mutation == "wrong_date":
        e.loc[0, "date"] = "2026-01-01"
    elif mutation == "wrong_side":
        e.loc[0, "side"] = 0
    elif mutation == "new_gain":
        e.loc[0, "G_event"] = 3.
    elif mutation == "nonfinite_complete":
        e.loc[e.index[-1], "net"] = np.nan  # Unselected SHORT branch still rejected.
    elif mutation == "clock":
        e.loc[0, "exit_ns"] = e.entry_ns.iloc[0]
    elif mutation == "excursion":
        e.loc[0, "mae"] = 2.
    else:
        p[-1, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_risk_grid(f, e, p, c)


def test_missing_cost_proxy_and_empty_common_cohort_are_json_null_not_nan():
    f, e, p, c = sample()
    f = f.drop(columns="decision_spread")
    f["y_crossing"] = -1
    result = evaluate_risk_grid(f, e, p, c)
    assert result["paired_diagnostics"]["common_roots"] == 0
    assert result["summary"]["test"]["trades"] == 4
    json.dumps(result, allow_nan=False)
