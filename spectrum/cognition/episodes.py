"""Causal flow-response episodes and executable fixed-plan opportunity.

Only the feature functions consume X. Targets and execution fields are separate.
The experiment contract is contracts/opportunity_episodes_v1.json. Source flow
bounds include unknown aggressor volume. These are second aggregates, not order-ID
absorption or a claim about participant intent.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

NS = 1_000_000_000
CLASSES = ("TAKE", "STOP", "TIME", "EMPTY")
COMPLETE, CENSORED, EMPTY, NOT_ELIGIBLE, EXCLUDED = 1, 2, 3, 4, 5
BASE = ["log_sigma", "sigma60_ratio", "sigma3600_ratio", "minutes_since_london", "spread_ticks", "side"]
EPISODE = ["episode_active", "episode_age", "episode_aggression", "episode_pressure", "episode_return", "episode_book"]
RESPONSE = ["ret5", "ret15", "ret60", "pressure5", "pressure30", "unknown30", "book5", "book30",
            "depth1", "depth4", "volume_rate", "pressure_change5", "pressure_change15", "book_change15",
            "depth_change5", "depth_change15", "reaction5", "reaction15"]
FAMILIES = {"baseline": BASE, "episode": BASE + EPISODE, "response": BASE + EPISODE + RESPONSE,
            "response_unweighted": BASE + EPISODE + RESPONSE, "shuffled": BASE + EPISODE + RESPONSE}
SIGNED = {"episode_aggression", "episode_pressure", "episode_return", "episode_book", "ret5", "ret15", "ret60",
          "pressure5", "pressure30", "book5", "book30", "depth1", "depth4", "pressure_change5",
          "pressure_change15", "book_change15", "depth_change5", "depth_change15"}


def contract():
    return json.loads((Path(__file__).resolve().parents[2] / "contracts/opportunity_episodes_v1.json").read_text(encoding="utf-8"))


def lag(t, x, seconds, segment):
    """Exact timestamp lag; gaps and segment boundaries are unknown."""
    t = np.asarray(t, np.int64); x = np.asarray(x, float); segment = np.asarray(segment)
    j = np.searchsorted(t, t - seconds * NS); jc = np.minimum(j, len(t) - 1)
    ok = (t[jc] == t - seconds * NS) & (segment[jc] == segment)
    return np.where(ok, x[jc], np.nan)


def measured(df, value, mask):
    return np.where(df[mask].to_numpy(bool), df[value].to_numpy(float), np.nan)


def features(df, segment):
    """All features as of each row, prefix-invariant including episode starts.

    segment must identify both execution continuity and absent second rows.
    A future mutation cannot change any earlier value or episode identity.
    """
    c = contract(); ec = c["episode"]
    t = df.t_ns.to_numpy(np.int64); n = len(t); segment = np.asarray(segment)
    # Missing seconds reset episode state even when the quote epoch is unchanged.
    reset = np.r_[True, (np.diff(t) != NS) | (segment[1:] != segment[:-1])]
    continuity = np.cumsum(reset)
    sig = df.sigma600.to_numpy(float)
    r15 = measured(df, "ret15", "ret15_known")
    prior15 = lag(t, r15 * sig * np.sqrt(15), 15, continuity)
    ret30 = (r15 * sig * np.sqrt(15) + prior15) / np.maximum(sig * np.sqrt(30), 1e-12)
    low = measured(df, "flow_pressure30_low", "flow_pressure30_known")
    high = measured(df, "flow_pressure30_high", "flow_pressure30_known")
    book = measured(df, "flow_book30_pressure", "flow_book30_known")
    rate = measured(df, "flow_volume_rate_ratio", "flow_volume_rate_known")
    direction = np.select([low >= ec["minimum_absolute_pressure_lower_bound"],
                           high <= -ec["minimum_absolute_pressure_lower_bound"]], [1, -1], 0)
    trigger = ((direction != 0) & np.isfinite(ret30) & (direction * ret30 <= ec["maximum_aligned_return30_sigma"])
               & (rate >= ec["minimum_volume_rate_ratio"]) & (-direction * book >= ec["minimum_replenishment_pressure"]))
    out = {"log_sigma": np.log(np.maximum(sig, 1e-12)),
           "sigma60_ratio": np.where(df.sigma60_ratio > 0, df.sigma60_ratio, np.nan),
           "sigma3600_ratio": np.where(df.sigma3600_ratio > 0, df.sigma3600_ratio, np.nan),
           "minutes_since_london": df.minutes_since_london.to_numpy(float),
           "spread_ticks": df.spread_ticks.to_numpy(float)}
    active = np.zeros(n); age = np.full(n, np.nan); ep_id = np.zeros(n, np.int64); starts = np.zeros(n, bool)
    fields = {k: np.full(n, np.nan) for k in EPISODE[2:]}
    last = -1; eid = 0; cooldown = ec["cooldown_seconds"] * NS
    for i in range(n):
        if reset[i]:
            last = -1
        if trigger[i] and (last < 0 or t[i] - t[last] >= cooldown):
            last = i; eid += 1; starts[i] = True
        if last >= 0 and t[i] - t[last] < cooldown:
            active[i] = 1; age[i] = (t[i] - t[last]) / NS; ep_id[i] = eid
            fields["episode_aggression"][i] = direction[last]
            fields["episode_pressure"][i] = low[last] if direction[last] > 0 else high[last]
            fields["episode_return"][i] = ret30[last]
            fields["episode_book"][i] = book[last]
    out.update(episode_active=active, episode_age=age / ec["cooldown_seconds"], **fields)
    for w in (5, 15, 60):
        out[f"ret{w}"] = measured(df, f"ret{w}", f"ret{w}_known")
    for w in (5, 30):
        out[f"pressure{w}"] = (measured(df, f"flow_pressure{w}_low", f"flow_pressure{w}_known")
                                + measured(df, f"flow_pressure{w}_high", f"flow_pressure{w}_known")) / 2
        out[f"book{w}"] = measured(df, f"flow_book{w}_pressure", f"flow_book{w}_known")
    out["unknown30"] = measured(df, "flow_unknown30_fraction", "flow_pressure30_known")
    for level in (1, 4):
        out[f"depth{level}"] = measured(df, f"flow_depth{level}_imbalance", f"flow_depth{level}_known")
    out["volume_rate"] = rate
    for seconds in (5, 15):
        out[f"pressure_change{seconds}"] = out["pressure5"] - lag(t, out["pressure5"], seconds, continuity)
        out[f"depth_change{seconds}"] = out["depth1"] - lag(t, out["depth1"], seconds, continuity)
    out["book_change15"] = out["book5"] - lag(t, out["book5"], 15, continuity)
    # These two fields describe price response relative to the frozen episode,
    # not relative to the proposed trade. Their windows must start after onset.
    for w in (5, 15):
        out[f"reaction{w}"] = np.where(age >= w, -fields["episode_aggression"] * out[f"ret{w}"], np.nan)
    return pd.DataFrame(out, index=df.index), dict(episode_id=ep_id, episode_start=starts, trigger=trigger,
                                                  continuity=continuity, ret30=ret30)


def orient(x, side):
    out = x.copy()
    for col in SIGNED:
        out[col] = out[col] * side
    out["side"] = side
    return out


def categories(status, net, take=24, stop=12, fee=.30):
    """Source fixed-executable plan: stop > take > time at the triggering quote.

    COMPLETE gross at a barrier necessarily triggered that barrier even if time
    expired at the same quote. Direct replay verifies this source assumption.
    Float32 net storage needs a sub-tick tolerance, not a one-tick tolerance.
    """
    status = np.asarray(status); gross = (np.asarray(net, float) + fee) / .1
    out = np.full(len(status), -1, np.int8); done = status == COMPLETE
    if np.any(done & ~np.isfinite(gross)):
        raise ValueError("Completed outcome has unknown net")
    out[done] = 2
    out[done & (gross >= take - 1e-5)] = 0
    out[done & (gross <= -stop + 1e-5)] = 1
    out[status == EMPTY] = 3
    return out


def episode_weights(frame):
    """One total fitting weight per date/side/episode; non-episode rows weight 1.

    Weights are fit-only. They are not streaming features or evaluation weights.
    """
    out = np.ones(len(frame)); active = frame.episode_id.to_numpy() > 0
    if active.any():
        counts = frame.loc[active].groupby(["date", "side", "episode_id"])["t_ns"].transform("size")
        out[active] = 1 / counts.to_numpy(float)
    return out


def sequential(frame, score="expected_net", threshold=0.):
    """One continuous account, with no outcome-dependent candidate filtering.

    Returns completed-prefix trades and a censor receipt. Daily reports after a
    censor are unknown. Session boundaries do not revive a censored account.
    """
    f = frame.sort_values(["t_ns", "side"], ascending=[True, False], kind="stable")
    choices = f.loc[f.groupby("t_ns", sort=False)[score].idxmax()].sort_values("t_ns")
    busy = -1; chosen = []; censor = None
    for row in choices.itertuples():
        if row.t_ns < busy or getattr(row, score) <= threshold:
            continue
        if row.status in (CENSORED, EXCLUDED, NOT_ELIGIBLE) or not np.isfinite(row.busy_s) or not np.isfinite(row.net):
            censor = dict(date=row.date, t_ns=str(row.t_ns), status=int(row.status), side=int(row.side))
            break
        chosen.append(row.Index)
        busy = row.t_ns + int(round(row.busy_s * NS))
    return frame.loc[chosen].copy(), censor


def block_interval(values, resamples=2000, block=5, seed=61022):
    """Moving-block bootstrap of the mean of daily totals, preserving zeros."""
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        return [None, None]
    b = min(block, len(a)); rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(a) - b + 1, (resamples, int(np.ceil(len(a) / b))))
    idx = (starts[..., None] + np.arange(b)).reshape(resamples, -1)[:, :len(a)]
    return np.quantile(a[idx].mean(1), [.025, .975]).tolist()
