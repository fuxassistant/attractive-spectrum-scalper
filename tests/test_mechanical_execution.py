"""Synthetic path contracts for the experiment's order and risk semantics."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.mechanical_lab.execution import (
    CONTRACT_OZ, EXIT_PLANS, ExecutionConfig, ExitPlan, QuoteTape, simulate_pairs,
)


def tape(times, bids, *, spread=.2, valid=None, epoch=None, coverage=30., available=None):
    ts = np.rint(np.asarray(times) * 1e9).astype(np.int64)
    n = len(ts)
    frame = pd.DataFrame({"t_ns": ts,
                          "available_ns": ts if available is None else np.rint(np.asarray(available) * 1e9).astype(np.int64),
                          "bid_nano": np.rint(np.asarray(bids) * 1e9).astype(np.int64),
                          "ask_nano": np.rint((np.asarray(bids) + spread) * 1e9).astype(np.int64),
                          "valid": np.ones(n, bool) if valid is None else valid,
                          "epoch": np.zeros(n, np.int64) if epoch is None else epoch})
    return QuoteTape.from_frame(frame, coverage_end_ns=int(coverage * 1e9), entry_start_ns=0)


def plan(**kwargs):
    return replace(ExitPlan(name="test", risk_mode="fixed", fixed_stop_ticks=10.,
                            min_risk_per_oz=.1, max_risk_per_oz=10., take_r=1.,
                            max_hold_seconds=2.), **kwargs)


def trials(q, p=None, c=None, origin=0, risk=1.):
    return simulate_pairs(q, np.asarray([origin], np.int64), np.asarray([risk]), p or plan(), c)


def test_strict_later_entry_and_exit_publication_and_pending_reversal():
    q = tape([0, .1, .11, .3, .4, .41], [100, 100, 100, 101.6, 101.6, 100.1])
    r = trials(q).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "TAKE"
    assert (r.entry_ns, r.trigger_ns, r.exit_ns) == (110_000_000, 300_000_000, 410_000_000)
    assert r.net_per_oz == pytest.approx(-.4)
    assert r.mfe_per_oz == pytest.approx(1.1)
    assert r.max_drawdown_per_oz == pytest.approx(1.5)
    assert r.net_per_oz * CONTRACT_OZ == pytest.approx(-40.)


def test_fee_and_bbo_side_and_slippage_charged_once_per_fill():
    q = tape([0, .11, 2.11, 2.21, 2.22], [100] * 5)
    c = ExecutionConfig(slippage_ticks_per_fill=2.)
    rows = trials(q, plan(take_r=100.), c)
    assert (rows.status == "COMPLETE").all()
    assert (rows.reason == "TIME").all()
    np.testing.assert_allclose(rows.net_per_oz, -.2 - .3 - .4)
    np.testing.assert_allclose(rows.max_drawdown_per_oz, .9)
    np.testing.assert_allclose(rows.entry_price, [100.4, 99.8])
    np.testing.assert_allclose(rows.exit_price, [99.8, 100.4])


def test_availability_clock_controls_fill_not_source_time():
    q = tape([0, .05, .06, .2, .3, .4], [100, 100, 100, 102, 102, 102],
             available=[0, .1, .11, .3, .4, .41])
    r = trials(q).iloc[0]
    assert r.entry_ns == 110_000_000 and r.exit_ns == 410_000_000


def test_timer_wins_against_quote_at_exact_deadline():
    q = tape([0, .11, 1.0, 1.11, 1.21, 1.22], [100, 100, 100, 102, 102, 102])
    r = trials(q, plan(max_hold_seconds=1.)).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "TIME"
    assert r.trigger_ns == 1_110_000_000 and r.trigger_ordinal == -1
    assert r.exit_ns == 1_220_000_000


@pytest.mark.parametrize("bad", ["valid", "epoch"])
def test_invalid_or_changed_epoch_cannot_be_skipped_during_entry(bad):
    args = {bad: [True, False, True, True]} if bad == "valid" else {bad: [0, 1, 0, 0]}
    q = tape([0, .05, .11, .2], [100] * 4, **args)
    r = trials(q)
    assert (r.status == "CENSORED").all() and (r.reason == "SUPPORT").all()
    assert r.net_per_oz.isna().all() and (r.entry_ordinal == -1).all()


def test_invalid_latest_origin_does_not_fall_back_to_older_valid_quote():
    q = tape([0, .05, .11, .2], [100] * 4, valid=[True, False, True, True])
    r = trials(q, origin=50_000_000)
    assert (r.status == "INELIGIBLE").all() and (r.reason == "ORIGIN").all()


def test_invalid_quote_during_pending_exit_censors_even_if_later_recovers():
    q = tape([0, .11, .3, .35, .41], [100, 100, 102, 100, 102],
             valid=[True, True, True, False, True])
    r = trials(q).iloc[0]
    assert r.status == "CENSORED" and r.reason == "SUPPORT"
    assert np.isnan(r.net_per_oz) and r.exit_ordinal == -1
    assert r.entry_ordinal == 1 and r.trigger_ns == 300_000_000


def test_quote_gap_is_not_executable_price_jump():
    q = tape([0, .11, 3., 3.2], [100, 100, 102, 102])
    r = trials(q, plan(max_hold_seconds=10.), ExecutionConfig(max_quote_gap_seconds=1.))
    assert (r.status == "CENSORED").all() and (r.reason == "SUPPORT").all()


def test_entry_timeout_is_empty_only_with_known_coverage():
    q = tape([0, .1, 3.], [100] * 3)
    r = trials(q)
    assert (r.status == "EMPTY").all() and (r.reason == "ENTRY_TIMEOUT").all()
    short = tape([0, .1], [100] * 2, coverage=1.)
    # Known session clock extends beyond the observed capture.
    short = replace(short, deadline_ns=30_000_000_000)
    r = trials(short)
    assert (r.status == "CENSORED").all() and (r.reason == "COVERAGE").all()


def test_exit_timeout_is_censored_not_zero_return():
    q = tape([0, .11, .3, .4, 3.], [100, 100, 102, 102, 102])
    r = trials(q).iloc[0]
    assert r.status == "CENSORED" and r.reason == "EXIT_TIMEOUT"
    assert np.isnan(r.net_per_oz) and r.exit_ordinal == -1


def test_breakeven_uses_net_after_fees_and_can_fill_below_zero():
    q = tape([0, .11, .3, .4, .5, .6, .61], [100, 100, 101.6, 100.7, 100.4, 100.4, 100.2])
    r = trials(q, plan(take_r=10., stop_mode="breakeven")).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "STOP"
    assert r.final_stop_net_per_oz == 0. and r.stop_updates == 1
    assert r.trigger_ns == 500_000_000 and r.exit_ns == 610_000_000
    assert r.net_per_oz == pytest.approx(-.3)


def test_trailing_ratchet_never_loosens_and_pending_close_cannot_retract():
    q = tape([0, .11, .3, .4, .5, .6, .7, .71],
             [100, 100, 101.6, 102.6, 103.6, 102.4, 104.6, 104.7])
    r = trials(q, plan(take_r=10., stop_mode="ratchet")).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "STOP"
    assert r.stop_updates == 3 and r.final_stop_net_per_oz == pytest.approx(2.)
    assert r.trigger_ns == 600_000_000 and r.exit_ns == 710_000_000
    assert r.net_per_oz == pytest.approx(4.2)
    # Pending favourable quotes do not cancel the request or move its stop.
    assert r.mfe_per_oz == pytest.approx(4.2)


def test_chandelier_distance_is_frozen_origin_risk_and_monotone():
    q = tape([0, .11, .3, .4, .5, .6, .61], [100, 100, 102.2, 103.3, 102.2, 102.2, 102.1])
    r = trials(q, plan(take_r=10., stop_mode="chandelier")).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "STOP"
    assert r.final_stop_net_per_oz == pytest.approx(1.8)
    assert r.stop_updates == 2 and r.trigger_ns == 500_000_000


def test_risk_clamps_are_causal_and_nan_atr_is_not_imputed():
    q = tape([0, .11, 2.11, 2.22], [100] * 4)
    p = plan(risk_mode="atr", min_risk_per_oz=.8, max_risk_per_oz=6.)
    assert trials(q, p, risk=.01).risk_per_oz.iloc[0] == .8
    assert trials(q, p, risk=100.).risk_per_oz.iloc[0] == 6.
    assert (trials(q, p, risk=np.nan).reason == "RISK").all()
    assert (trials(q, p, risk=-1.).reason == "RISK").all()
    assert (trials(q, plan(), risk=np.nan).status == "COMPLETE").all()


def test_plan_mapping_matches_named_plan_and_empty_candidates_work():
    q = tape([0, .11, 2.11, 2.22], [100] * 4, coverage=1000.)
    named = simulate_pairs(q, np.asarray([0], np.int64), np.ones(1), "quick_time")
    mapped = simulate_pairs(q, np.asarray([0], np.int64), np.ones(1), EXIT_PLANS["quick_time"].__dict__)
    pd.testing.assert_frame_equal(named, mapped)
    assert simulate_pairs(q, np.asarray([], np.int64), np.asarray([])).empty


def test_future_after_completed_fill_cannot_change_trial():
    first = tape([0, .11, .3, .4, .41], [100, 100, 102, 102, 102])
    second = tape([0, .11, .3, .4, .41, .5], [100, 100, 102, 102, 102, 1],
                  valid=[True, True, True, True, True, False])
    pd.testing.assert_frame_equal(trials(first), trials(second))


@pytest.mark.parametrize("mode", ["fixed", "breakeven", "ratchet", "chandelier"])
def test_long_short_price_reflection_preserves_dynamic_path_accounting(mode):
    q = tape([0, .11, .3, .4, .5, .6, .7, .71],
             [4000, 4000, 4001.6, 4002.6, 4003.6, 4002.4, 4004.6, 4004.7])
    mirror = replace(q, bid=8_000_000_000_000 - q.ask, ask=8_000_000_000_000 - q.bid)
    p = plan(take_r=2.5, stop_mode=mode)
    a, b = trials(q, p).iloc[0], trials(mirror, p).iloc[1]
    for key in ("status", "reason", "entry_ns", "trigger_ns", "exit_ns", "stop_updates"):
        assert a[key] == b[key]
    for key in ("net_per_oz", "mae_per_oz", "mfe_per_oz", "max_drawdown_per_oz", "final_stop_net_per_oz"):
        assert a[key] == pytest.approx(b[key])


def test_static_plan_agrees_with_independent_existing_scalar_oracle():
    from spectrum.execution.barriers import SimulationConfig, scalar_reference

    rng = np.random.default_rng(991)
    times = np.arange(800) * .05
    prices = 4000 + np.cumsum(rng.choice([-.3, -.1, .1, .3], len(times)))
    valid = np.ones(len(times), bool)
    valid[500] = False
    epochs = np.zeros(len(times), np.int64)
    epochs[350:] = 1
    q = tape(times, prices, valid=valid, epoch=epochs, coverage=50.)
    origins = np.arange(1, 35, dtype=np.int64) * 1_000_000_000
    risks = rng.uniform(.9, 2.9, len(origins))
    p = plan(risk_mode="atr", take_r=1.5)
    frame = simulate_pairs(q, origins, risks, p)
    reference_config = SimulationConfig(fee_roundtrip=.3, latency_ms=100.,
                                         pending_timeout_seconds=2., max_quote_gap_seconds=30.,
                                         gain_multiple=1.5, stop_multiple=1., deadline_seconds=2.)
    for _, row in frame.iterrows():
        expected = scalar_reference(q, int(row.signal_ns), risks[int(row.origin_index)],
                                    int(row.side), reference_config)
        for key in ("status", "reason", "entry_ordinal", "trigger_ordinal", "exit_ordinal", "trigger_ns"):
            assert row[key] == expected[key]
        if row.status == "COMPLETE":
            for actual, wanted in (("net_per_oz", "net"), ("mae_per_oz", "mae"), ("mfe_per_oz", "mfe")):
                assert row[actual] == pytest.approx(expected[wanted], abs=1e-12)


def test_exact_net_take_at_realistic_gold_price_is_not_missed():
    q = tape([0, .11, .3, .4, .41], [4000, 4000, 4001.7, 4001.7, 4001.7])
    # 4001.7 - 4000.2 - .3 is exactly the 12-tick net target.
    r = trials(q, plan(fixed_stop_ticks=12.)).iloc[0]
    assert r.status == "COMPLETE" and r.reason == "TAKE"
    assert r.trigger_ns == 300_000_000 and r.net_per_oz == pytest.approx(1.2)


def test_validation_rejects_float_time_and_unsafe_parameters():
    q = tape([0, .11], [100, 100])
    with pytest.raises(ValueError, match="integer"):
        simulate_pairs(q, np.asarray([0.]), np.ones(1))
    with pytest.raises(ValueError):
        ExecutionConfig(latency_ms=2000.)
    with pytest.raises(ValueError):
        plan(be_lock_r=1.)
    with pytest.raises(ValueError, match="availability"):
        tape([0, .11], [100, 100], available=[0, .10])
