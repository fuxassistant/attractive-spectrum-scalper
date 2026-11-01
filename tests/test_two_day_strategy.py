"""Causal selection and execution invariants for the two-day entry study."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.two_day_strategy_v1.engine import (
    acceptance_rates, build_masks, offered, pair_record, replay, select_pair,
    stable_uniform,
)


NS = 1_000_000_000


@pytest.fixture
def config():
    path = Path(__file__).parents[1] / "experiments/two_day_strategy_v1/config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def observations():
    """Three distinct entry mechanisms and one unsupported observation."""
    return pd.DataFrame({
        "candidate_id": [101, 102, 103, 104],
        "past_known": [True, True, True, False],
        "x_base_mbo_now_known": [1, 1, 1, 0],
        "x_mbo_mbo_now_trade_unknown_fraction": [.1, .1, .1, .1],
        "x_mbo_mbo_now_trade_imbalance": [.5, -.5, -.5, .5],
        "x_mbo_mbo_now_depth_1_toward_imbalance": [.5, -.5, .5, .5],
        "x_base_mbo_now_trade_total_log": [5., 5., 5., 5.],
        "past_net30_r": [.5, -.5, 0., .5],
        "past_migration_r": [.5, -.5, 0., .5],
        "h_memory_return": [True, False, False, False],
        "h_permeable_departure": [False, True, False, False],
        "h_failed_escape": [False, False, True, False],
        "h_place_migration": [False, True, False, False],
    }, index=[7, 11, 19, 23])


def record(strategy="toward_all", **overrides):
    row = {"strategy": strategy, "base_complete": True, "stress_complete": True,
           "base_net_usd": 300., "base_day1_net_usd": 150.,
           "base_day2_net_usd": 150., "stress_day1_net_usd": 100.,
           "stress_day2_net_usd": 100., "base_day1_trades": 4,
           "base_day2_trades": 4, "stress_day1_trades": 4,
           "stress_day2_trades": 4}
    row.update(overrides)
    return row


def account_fixture(times=(1, 301, 305, 86401), nets=(1., 999., -2., 3.)):
    """Known outcomes include an entry 300s later while the first exit is pending."""
    c = pd.DataFrame({
        "candidate_id": np.arange(len(times), dtype=np.int64) + 100,
        "origin_index": np.arange(len(times), dtype=np.int64),
        "t_ns": np.asarray(times, dtype=np.int64) * NS,
        "side": 1, "target_nano": 101_000_000_000,
        "deadline_ns": 777 * NS,
        "date": ["2026-08-17" if t < 86400 else "2026-08-18" for t in times],
        "regime": "test", "time_regime": "test",
    })
    rows = []
    for i, (t, net) in enumerate(zip(times, nets)):
        for side in (1, -1):
            value = net if side == 1 else -net
            rows.append(dict(origin_index=i, side=side, risk_per_oz=1.,
                status="COMPLETE", reason="TIME", entry_ns=(t+1)*NS,
                exit_ns=(t+302 if i == 0 else t+2)*NS,
                net_per_oz=value, mae_per_oz=min(-.1, value),
                mfe_per_oz=max(0., value), max_drawdown_per_oz=max(.1, -value)))
    return c, pd.DataFrame(rows)


def test_masks_have_no_dependency_on_future_labels_or_executable_outcomes(config):
    f = observations()
    before = build_masks(f, config, activity_threshold=4.)
    future = f.assign(state="RETURN_FIRST", label_known=True,
                      net_per_oz=1e9, exit_ns=999_999_999_999,
                      status="COMPLETE", next_day_net_usd=1e9)
    future.loc[:, ["state", "status"]] = ["ADVERSE_FIRST", "CENSORED"]
    future.loc[:, ["net_per_oz", "next_day_net_usd"]] *= -1
    pd.testing.assert_frame_equal(before, build_masks(future, config, 4.))
    assert before.flow_return.tolist() == [True, False, False, False]
    assert before.flow_away.tolist() == [False, True, False, False]
    assert before.stalled_flow_return.tolist() == [False, False, True, False]


@pytest.mark.parametrize("field,value", [
    ("x_base_mbo_now_known", 0),
    ("x_base_mbo_now_known", np.nan),
    ("x_mbo_mbo_now_trade_unknown_fraction", .21),
    ("x_mbo_mbo_now_trade_unknown_fraction", np.nan),
    ("past_known", False),
])
def test_unknown_flow_cannot_become_a_directional_entry(config, field, value):
    f = observations()
    f[field] = value
    masks = build_masks(f, config, 4.)
    assert not masks[["flow_return", "flow_away", "stalled_flow_return"]].any().any()
    # Missing flow does not remove the separately defined unconditional controls.
    assert masks.toward_all.all() and masks.away_all.all()


def test_activity_threshold_comes_from_caller_and_wait_opens_nothing(config):
    f = observations()
    low = build_masks(f, config, 4.)
    high = build_masks(f, config, 6.)
    assert low.loc[19, "stalled_flow_return"]
    assert not high.loc[19, "stalled_flow_return"]
    assert not low.WAIT.any()
    assert not high.WAIT.any()


def test_offered_keeps_original_target_identity_and_drops_parent_deadline():
    c, _ = account_fixture()
    untouched = c.copy(deep=True)
    masks = pd.DataFrame({"away_all": [True, False, True, False]}, index=c.index)
    choice = offered(c, masks, "away_all")
    assert choice.candidate_id.tolist() == [100, 102]
    assert choice.side.tolist() == [-1, -1]
    assert choice.original_signal_side.tolist() == [1, 1]
    assert choice.target_nano.eq(101_000_000_000).all()
    assert "deadline_ns" not in choice
    pd.testing.assert_frame_equal(c, untouched)


@pytest.mark.parametrize("failure", [
    {"base_day1_net_usd": -1.},
    {"base_day2_net_usd": 0.},
    {"stress_day1_net_usd": -1.},
    {"stress_day2_net_usd": 0.},
    {"base_day1_trades": 2},
    {"base_day2_trades": 2},
    {"stress_day1_trades": 2},
    {"stress_day2_trades": 2},
    {"base_complete": False},
    {"stress_complete": False},
])
def test_selector_waits_when_either_day_cost_support_or_censoring_fails(config, failure):
    choice, ranked = select_pair([record(**failure)], config)
    assert choice["selected_strategy"] == "WAIT"
    assert choice["eligible_count"] == 0
    assert not ranked.eligible.any()


def test_selector_uses_worst_stressed_day_not_largest_total_or_single_win(config):
    rows = [record("toward_all", base_net_usd=10000.,
                   stress_day1_net_usd=2000., stress_day2_net_usd=1.),
            record("away_all", base_net_usd=1000.,
                   stress_day1_net_usd=90., stress_day2_net_usd=80.)]
    choice, ranked = select_pair(rows, config)
    assert choice["raw_leader"] == "toward_all"
    assert choice["selected_strategy"] == "away_all"
    assert choice["score_gap_usd"] == pytest.approx(79.)
    assert ranked.set_index("strategy").score_usd.to_dict() == {
        "toward_all": 1., "away_all": 80.}


def test_ties_and_future_columns_cannot_change_past_pair_choice(config):
    rows = [record("toward_all"), record("away_all")]
    expected, _ = select_pair(rows, config)
    altered = deepcopy(rows[::-1])
    for i, row in enumerate(altered):
        row.update(next_day_net_usd=(i+1)*1e9, future_winner="toward_all")
    actual, _ = select_pair(altered, config)
    assert actual == expected
    assert actual["selected_strategy"] == "away_all"
    assert actual["score_gap_usd"] == 0.


def test_negative_raw_leader_is_descriptive_and_does_not_override_wait(config):
    choice, _ = select_pair([record(base_net_usd=-200.,
        base_day1_net_usd=-100., base_day2_net_usd=-100.)], config)
    assert choice["raw_leader"] == "toward_all"
    assert choice["selected_strategy"] == "WAIT"


def test_acceptance_rates_use_only_supplied_past_pair_and_wait_stays_zero():
    history = pd.DataFrame({"side": [1, 1, -1, -1],
                            "phase": ["a", "a", "b", "b"]})
    mask = pd.Series([True, False, False, False])
    rate, cells = acceptance_rates(history, mask, ["side", "phase"], prior=2)
    assert rate == .25
    assert cells[(1, "a")] == pytest.approx(.375)
    assert cells[(-1, "b")] == pytest.approx(.125)
    # Adding irrelevant future/outcome columns to the past rows cannot refit rates.
    changed = history.assign(next_day_count=10**6, pnl_usd=-10**6)
    assert acceptance_rates(changed, mask, ["side", "phase"], prior=2) == (rate, cells)
    zero, zero_cells = acceptance_rates(history, mask & False, ["side", "phase"], prior=2)
    assert zero == 0. and set(zero_cells.values()) == {0.}


def test_random_acceptance_draw_is_stable_for_identity_when_population_changes():
    ids = np.array([100, 200, 300, 400], dtype=np.uint64)
    full = stable_uniform(ids, 17)
    np.testing.assert_array_equal(stable_uniform(ids[[3, 1]], 17), full[[3, 1]])
    assert ((full >= 0) & (full < 1)).all()
    assert not np.array_equal(full, stable_uniform(ids, 18))


def test_replay_preserves_busy_exit_latency_and_capital_across_days(config):
    c, outcomes = account_fixture()
    masks = pd.DataFrame({"toward_all": True}, index=c.index)
    choice = offered(c, masks, "toward_all")
    summary, log = replay(choice, outcomes, config)
    assert summary["overlap"] == 1
    assert log.candidate_id.tolist() == [100, 102, 103]
    assert log.equity_usd.tolist() == pytest.approx([100100., 99900., 100200.])
    assert summary["net_usd"] == pytest.approx(200.)
    assert (log.signal_ns.to_numpy()[1:] > log.exit_ns.to_numpy()[:-1]).all()
    assert log.selected_strategy.eq("toward_all").all()


def test_replay_filled_censor_halts_future_days_without_imputed_return(config):
    c, outcomes = account_fixture(times=(1, 305, 86401), nets=(1., 100., 100.))
    outcomes.loc[outcomes.origin_index.eq(1), ["status", "exit_ns", "net_per_oz"]] = [
        "CENSORED", 0, np.nan]
    chosen = offered(c, pd.DataFrame({"toward_all": True}, index=c.index), "toward_all")
    summary, log = replay(chosen, outcomes, config)
    assert log.candidate_id.tolist() == [100]
    assert summary["halted"] and summary["unresolved"] == 1
    assert not summary["capital_curve_complete"]
    assert summary["net_usd"] == pytest.approx(100.)


def test_pair_record_uses_actual_costed_trade_days_and_exit_feedback(config):
    c, outcomes = account_fixture()
    chosen = offered(c, pd.DataFrame({"toward_all": True}, index=c.index), "toward_all")
    stress = outcomes.copy()
    stress["net_per_oz"] -= .2
    stress["mae_per_oz"] -= .2
    result = pair_record("toward_all", ["2026-08-17", "2026-08-18"],
                         chosen, {"base": outcomes, "stress": stress}, config)
    assert result["base_day1_trades"] == 2 and result["base_day2_trades"] == 1
    assert result["base_day1_net_usd"] == pytest.approx(-100.)
    assert result["stress_day1_net_usd"] == pytest.approx(-140.)
    assert result["base_max_feedback_ns"] == 86403 * NS
    assert result["base_without_best_trade_usd"] == pytest.approx(-100.)


def test_wait_replay_has_zero_exposure_and_complete_capital(config):
    c, outcomes = account_fixture()
    chosen = offered(c, pd.DataFrame({"WAIT": False}, index=c.index), "WAIT")
    summary, log = replay(chosen, outcomes, config)
    assert log.empty and summary["trades"] == 0
    assert summary["net_usd"] == 0. and summary["exposure_seconds"] == 0.
    assert summary["capital_curve_complete"]


def test_forward_outcome_mutation_cannot_change_choice_already_deployed(config):
    from experiments.two_day_strategy_v1.run import evaluate

    cfg = deepcopy(config)
    cfg["dates"] = ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20"]
    cfg["controls"]["random_seeds"] = 1
    rows = []; outcomes = []
    for date in cfg["dates"]:
        day = int(pd.Timestamp(date, tz="UTC").value)
        for j in range(4):
            identity = len(rows) + 100
            t = day + (7*3600+j*310)*NS
            row = observations().iloc[0].to_dict()
            row.update(candidate_id=identity, origin_index=identity, t_ns=t,
                       date=date, side=1, target_nano=101_000_000_000,
                       deadline_ns=t+900*NS, regime="test", time_regime="test",
                       distance_r=1., risk_per_oz=1., spread_per_oz=.1,
                       past_efficiency30=.5, match_distance_bin=1,
                       match_phase="morning", match_vol_bin=1)
            rows.append(row)
            for side in (1, -1):
                net = float(side)
                outcomes.append(dict(origin_index=identity, side=side,
                    risk_per_oz=1., status="COMPLETE", reason="TIME",
                    entry_ns=t+NS, exit_ns=t+2*NS, net_per_oz=net,
                    mae_per_oz=min(-.1, net), mfe_per_oz=max(0., net),
                    max_drawdown_per_oz=max(.1, -net)))
    f = pd.DataFrame(rows); table = pd.DataFrame(outcomes)
    original = evaluate(f, {"base": table, "stress": table.copy()}, cfg, 4.)
    # Reverse every consequence on the first deployment day; this is legitimate
    # feedback for the following decision, never for the one already frozen.
    changed = table.copy()
    ids = f.loc[f.date.eq(cfg["dates"][2]), "origin_index"]
    affected = changed.origin_index.isin(ids)
    changed.loc[affected, "net_per_oz"] *= -1
    changed.loc[affected, "mae_per_oz"] = np.minimum(-.1, changed.loc[affected, "net_per_oz"])
    changed.loc[affected, "mfe_per_oz"] = np.maximum(0., changed.loc[affected, "net_per_oz"])
    changed.loc[affected, "max_drawdown_per_oz"] = np.maximum(.1, -changed.loc[affected, "net_per_oz"])
    mutated = evaluate(f, {"base": changed, "stress": changed.copy()}, cfg, 4.)
    frozen_fields = ["pair_start", "pair_end", "selection_ns", "selected_strategy",
                     "raw_leader", "runner_up", "score_gap_usd", "eligible_count"]
    pd.testing.assert_series_equal(original["selections"].iloc[0][frozen_fields],
                                   mutated["selections"].iloc[0][frozen_fields])
    date = cfg["dates"][2]
    a = original["decisions"].query("date == @date").reset_index(drop=True)
    b = mutated["decisions"].query("date == @date").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert original["selections"].iloc[1].selected_strategy != "WAIT"
    assert mutated["selections"].iloc[1].selected_strategy == "WAIT"
