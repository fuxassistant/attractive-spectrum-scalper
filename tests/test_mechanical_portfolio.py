"""Sequential account invariants independent of market-specific profitability."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.mechanical_lab.portfolio import AccountConfig, account


NS = 1_000_000_000


def fixtures(times=(1, 5, 12, 25), nets=(-1.0, 99.0, -1.0, 1.0)):
    candidates = pd.DataFrame({
        "candidate_id": np.arange(len(times)) + 100,
        "origin_index": np.arange(len(times)),
        "t_ns": np.asarray(times, dtype=np.int64) * NS,
        "date": "2026-08-03", "regime": "range_normal",
        "time_regime": "07_12_utc", "side": 1,
    })
    rows = []
    for i, (t, net) in enumerate(zip(times, nets)):
        for side in (1, -1):
            rows.append(dict(origin_index=i, side=side, risk_per_oz=1.0,
                status="COMPLETE", reason="TIME", entry_ns=(t + 1) * NS,
                exit_ns=(t + 8) * NS, net_per_oz=net,
                mae_per_oz=min(-0.3, net), mfe_per_oz=max(0.0, net),
                max_drawdown_per_oz=max(0.3, -net)))
    return candidates, pd.DataFrame(rows)


def test_busy_position_excludes_overlapping_opportunity_and_its_sizing_feedback():
    c, p = fixtures()
    summary, trades = account(c, p, sizing="martingale_capped")
    assert trades.candidate_id.tolist() == [100, 102, 103]
    assert trades.qty.tolist() == [1, 2, 4]
    assert summary["overlap"] == 1
    assert summary["net_usd"] == pytest.approx(100.0)


def test_size_changes_only_after_realized_previous_trade():
    c, p = fixtures(times=(1, 12, 25), nets=(-1.0, -1.0, 1.0))
    _, original = account(c, p, sizing="martingale_capped")
    altered = p.copy()
    altered.loc[altered.origin_index.eq(1), ["net_per_oz", "mfe_per_oz"]] = 1.0
    _, alternative = account(c, altered, sizing="martingale_capped")
    assert original.qty.tolist() == [1, 2, 4]
    assert alternative.qty.tolist() == [1, 2, 1]
    pd.testing.assert_series_equal(original.iloc[0], alternative.iloc[0])


def test_filled_censor_halts_account_and_never_becomes_zero_profit():
    c, p = fixtures(times=(1, 12, 25), nets=(1.0, 100.0, 100.0))
    p.loc[p.origin_index.eq(1), "status"] = "CENSORED"
    p.loc[p.origin_index.eq(1), "net_per_oz"] = np.nan
    summary, trades = account(c, p)
    assert trades.candidate_id.tolist() == [100]
    assert summary["unresolved"] == 1
    assert summary["halted"] is True
    assert summary["capital_curve_complete"] is False
    assert summary["net_usd"] == 100.0


def test_unfilled_censor_stops_only_current_day():
    c, p = fixtures(times=(1, 12, 25), nets=(1.0, 100.0, 1.0))
    p.loc[p.origin_index.eq(0), ["status", "entry_ns"]] = ["CENSORED", 0]
    c.loc[c.origin_index.eq(2), "date"] = "2026-08-04"
    summary, trades = account(c, p)
    assert trades.candidate_id.tolist() == [102]
    assert summary["unresolved"] == 0
    assert summary["wait_rejected"] == 1


def test_mtm_drawdown_preserves_within_trade_order_and_previous_account_peak():
    c, p = fixtures(times=(1, 12), nets=(0.2, 1.0))
    # First path -0.3 -> 5 -> 0.2 has an internal drawdown4.8, NOT5.3.
    p.loc[p.origin_index.eq(0), ["mae_per_oz", "mfe_per_oz", "max_drawdown_per_oz"]] = [-0.3, 5.0, 4.8]
    # Next path -1 -> 1 draws down5.8 from the prior account high.
    p.loc[p.origin_index.eq(1), ["mae_per_oz", "mfe_per_oz", "max_drawdown_per_oz"]] = [-1.0, 1.0, 1.0]
    summary, trades = account(c, p)
    assert trades.max_drawdown_usd.tolist() == pytest.approx([480.0, 580.0])
    assert summary["max_drawdown_usd"] == pytest.approx(580.0)


def test_keyed_coin_sides_are_stable_when_candidates_are_filtered():
    c, p = fixtures(times=tuple(range(1, 241, 12)), nets=(0.0,) * 20)
    _, full = account(c, p, direction="coin", seed=472901)
    _, filtered = account(c.loc[c.candidate_id.mod(3).ne(0)], p, direction="coin", seed=472901)
    full_sides = full.set_index("candidate_id").side
    assert full_sides.loc[filtered.candidate_id].tolist() == filtered.side.tolist()
    assert set(full.side) == {-1, 1}


def test_loss_at_max_martingale_stake_waits_for_next_day():
    c, p = fixtures(times=(1, 12, 25, 40, 52), nets=(-1.0,) * 5)
    c.loc[c.origin_index.eq(4), "date"] = "2026-08-04"
    summary, trades = account(c, p, sizing="martingale_capped")
    assert trades.qty.tolist() == [1, 2, 4, 1]
    assert trades.candidate_id.tolist() == [100, 101, 102, 104]
    assert summary["wait_rejected"] == 1


@pytest.mark.parametrize("overrides", [
    {"initial_capital_usd": -1}, {"contract_oz": 0},
    {"max_planned_risk_fraction": np.nan}, {"daily_loss_fraction": -0.1},
    {"max_drawdown_fraction": 1.1}, {"max_contracts": 1.5},
])
def test_invalid_account_budget_is_rejected(overrides):
    with pytest.raises(ValueError):
        replace(AccountConfig(), **overrides)


def test_duplicate_counterfactual_key_is_rejected_instead_of_last_row_winning():
    c, p = fixtures(times=(1,), nets=(1.0,))
    duplicate = pd.concat([p, p.iloc[[0]].assign(net_per_oz=1000.0)], ignore_index=True)
    with pytest.raises(ValueError, match="[Dd]uplicate|[Uu]nique"):
        account(c, duplicate)
