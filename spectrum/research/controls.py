"""Causal synthetic positive control, labelled by the production simulator.

This verifies that a training stack can recover an intentionally strong
past-to-future signal. It is not evidence of signal or profitability in GC.
The latent dynamics are not passed to the model or used to construct y.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from spectrum.research.simulation import NS, SimulationConfig, simulate_tape


FEATURE_NAMES = (
    "past_return_over_005", "past_four_return_mean_over_005",
    "price_from_first_past_quote_over_05", "spread_over_01", "quote_valid",
)


def make_control_data(seed: int, count: int = 384) -> dict:
    """Return x[N,16,5] float32 and engine-derived y[N] int64.

    Public extra fields ``tape``, ``all_origins_ns``, ``all_risk`` and
    ``simulation_config`` permit independent simulator replay. ``net`` and
    execution timestamps correspond to retained x/y rows. Any trial pair
    whose outcomes are not both COMPLETE and finite is excluded as unknown.
    Distinct train/test seeds generate distinct past and future innovations.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    regime_seed, past_seed, future_seed = np.random.SeedSequence(seed).spawn(3)
    regime_rng = np.random.default_rng(regime_seed)
    past_rng = np.random.default_rng(past_seed)
    future_rng = np.random.default_rng(future_seed)
    # Round robin only balances the latent dynamics. Labels remain measured
    # simulator consequences, including costs, latency and the stopping rule.
    regimes = np.resize(np.array([-1., 0., 1.]), count)
    regime_rng.shuffle(regimes)
    past_steps, future_steps, step_ns = 16, 48, NS // 4
    episode_steps = past_steps + future_steps
    quote_t = np.empty(count * episode_steps, dtype=np.int64)
    mids = np.empty(count * episode_steps, dtype=np.float64)
    spreads = np.empty_like(mids)
    x = np.empty((count, past_steps, len(FEATURE_NAMES)), dtype=np.float32)
    origins = np.empty(count, dtype=np.int64)
    risk = np.full(count, .5, dtype=np.float64)
    for i, regime in enumerate(regimes):
        # A quarter-second drift of +/- .05, with independent past/future
        # noise. Flat dynamics stay deliberately below spread and fees.
        past_increments = regime * .05 + past_rng.normal(0., .006, past_steps - 1)
        past_price = 2000. + np.r_[0., np.cumsum(past_increments)]
        future_increments = regime * .05 + future_rng.normal(0., .008, future_steps)
        future_price = past_price[-1] + np.cumsum(future_increments)
        past_spread = np.clip(.10 + past_rng.normal(0., .003, past_steps), .08, .12)
        future_spread = np.clip(.10 + future_rng.normal(0., .003, future_steps), .08, .12)
        returns = np.r_[0., np.diff(past_price)]
        rolling = np.array([returns[max(0, j - 3):j + 1].mean() for j in range(past_steps)])
        x[i] = np.column_stack((returns / .05, rolling / .05,
                                (past_price - past_price[0]) / .5,
                                past_spread / .10, np.ones(past_steps)))
        sl = slice(i * episode_steps, (i + 1) * episode_steps)
        quote_t[sl] = i * 24 * NS + np.arange(episode_steps) * step_ns
        mids[sl] = np.r_[past_price, future_price]
        spreads[sl] = np.r_[past_spread, future_spread]
        origins[i] = quote_t[i * episode_steps + past_steps - 1]
    tape = SimpleNamespace(
        quote_t=quote_t,
        bid=np.rint((mids - spreads / 2) * 1e9).astype(np.int64),
        ask=np.rint((mids + spreads / 2) * 1e9).astype(np.int64),
        valid=np.ones(len(quote_t), dtype=bool),
        epoch=np.repeat(np.arange(count, dtype=np.int64), episode_steps),
        coverage_end_ns=int(quote_t[-1] + step_ns),
        entry_start_ns=int(quote_t[0]), deadline_ns=int(quote_t[-1] + NS),
    )
    config = SimulationConfig(fee_roundtrip=.05, latency_ms=100.,
        pending_timeout_seconds=2., max_quote_gap_seconds=60.,
        gain_multiple=2., stop_multiple=1., deadline_seconds=8.)
    trials = simulate_tape(tape, origins, risk, config)
    net = trials.net.to_numpy().reshape(count, 2)
    complete = trials.status.to_numpy().reshape(count, 2) == "COMPLETE"
    known = complete.all(axis=1) & np.isfinite(net).all(axis=1)
    # NO latent regime is used below: a side succeeds only if actual delayed
    # liquidation net is positive. Both sides are independent fixed plans.
    positive = net[known] > 0
    y = (positive[:, 0].astype(np.int64) + 2 * positive[:, 1].astype(np.int64))
    retained = np.flatnonzero(known)
    counts = np.bincount(y, minlength=4)
    receipts = dict(
        recipe="synthetic_observable_drift_v1",
        seed=seed, requested_count=count, retained_count=len(retained),
        excluded_unknown_count=int((~known).sum()),
        state_order=["NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH"],
        state_counts=[int(v) for v in counts], feature_names=list(FEATURE_NAMES),
        past_steps=past_steps, sample_interval_seconds=.25,
        latent_dynamic_counts={str(int(v)): int((regimes == v).sum()) for v in [-1., 0., 1.]},
        simulation_config=asdict(config),
        why_causal=("Each feature at each past step uses prices/spreads at or before that step. "
                    "Past and future innovations use separate SeedSequence children; they share "
                    "only the deliberately persistent latent drift. That latent drift is not in X. "
                    "Future quotes are used only by simulate_tape to determine delayed-exit outcomes."),
        label_rule="COMPLETE finite two-side trials; y=1*(net_LONG>0)+2*(net_SHORT>0)",
        timing=("Quotes publish every 250 ms. A 100 ms requested latency executes at the next "
                "strictly eligible publication, generally 250 ms later. Each episode has its own epoch."),
        intended_scope=("Strong synthetic positive control only. It establishes pipeline learnability, "
                        "not market predictability, calibrated market probabilities or economic admission."),
    )
    return dict(x=x[known], y=y, receipts=receipts, net=net[known],
        origins_ns=origins[known], retained_episode_indices=retained,
        entry_ns=trials.entry_ns.to_numpy().reshape(count, 2)[known],
        exit_ns=trials.exit_ns.to_numpy().reshape(count, 2)[known],
        tape=tape, all_origins_ns=origins, all_risk=risk, simulation_config=asdict(config))
