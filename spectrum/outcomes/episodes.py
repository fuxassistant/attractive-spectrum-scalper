"""Reusable formulas extracted from spectrum/cognition/episodes.py; see docs/MIGRATION.json."""
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
