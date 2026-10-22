import numpy as np
import pytest

from spectrum.cognition import race as rc


def brute(mid, k, G, h):
    m0 = mid[k]; hi = lo = 0.
    for j in range(1, h + 1):
        if k + j >= len(mid) or not np.isfinite(mid[k + j]):
            return rc.CENSORED, 0, np.nan
        d = mid[k + j] - m0
        if d >= G:
            return 1, j, -lo
        if d <= -G:
            return -1, j, hi
        hi, lo = max(hi, d), min(lo, d)
    return 0, 0, np.nan


def test_race_matches_a_direct_scan_and_scores_follow_costs():
    rng = np.random.default_rng(0)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 20000)); mid[7000:7005] = np.nan
    seg = np.ones(len(mid), np.int64); start = np.sort(rng.choice(19000, 400, replace=False))
    sigma = rng.uniform(.4, 1.2, len(start))
    for G in rc.GRID_G:
        r = rc.race(mid, seg, start, sigma, G)
        for i, k in enumerate(start):
            w, t, m = brute(mid, k, G, r['H'][i])
            assert r['winner'][i] == w and (w not in (1, -1) or (r['t_win'][i] == t and r['mae'][i] == pytest.approx(m)))
        won = r['winner'] == 1
        assert np.allclose(r['long'][won], (G - rc.C_WIN) / G) and np.allclose(r['short'][won], -(G + rc.C_LOSS) / G)
        assert np.all(r['opp'][np.isfinite(r['opp'])] >= 0)
        assert np.all((r['H'] >= rc.H_MIN) & (r['H'] <= rc.H_MAX))


def test_sign_flipped_null_keeps_increment_sizes():
    rng = np.random.default_rng(1)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 5000)); seg = np.ones(5000, np.int64)
    nul = rc.sign_flipped(mid, seg, rng)
    assert np.allclose(np.abs(np.diff(nul)), np.abs(np.diff(mid)))
    assert rc.runs(np.array([1, 1, 0, -1, -1, -1, 1])).tolist() == [2, 3, 1]


def test_per_row_barrier_keeps_the_horizon_and_matches_a_direct_scan():
    rng = np.random.default_rng(5)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 20000)); seg = np.ones(len(mid), np.int64)
    start = np.sort(rng.choice(19000, 300, replace=False)); sigma = rng.uniform(.4, 1.2, len(start))
    bar = rng.uniform(9, 30, len(start))
    r = rc.race(mid, seg, start, sigma, 18, barrier=bar)
    assert np.array_equal(r['H'], rc.horizon(sigma, 18))
    for i, k in enumerate(start):
        w, t, m = brute(mid, k, bar[i], r['H'][i])
        assert r['winner'][i] == w
    won = r['winner'] == 1
    assert np.allclose(r['long'][won], (bar[won] - rc.C_WIN) / bar[won])


def test_excursion_before_stop_matches_a_direct_scan():
    rng = np.random.default_rng(7)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 20000)); seg = np.ones(len(mid), np.int64)
    start = np.sort(rng.choice(19000, 300, replace=False)); sigma = rng.uniform(.4, 1.2, len(start))
    e = rc.excursion(mid, seg, start, sigma, 18)
    for i, k in enumerate(start):
        path = mid[k + 1:k + e['H'][i] + 1] - mid[k]
        for side, R in ((1, e['R_long'][i]), (-1, e['R_short'][i])):
            p = side * path; hit = np.flatnonzero(p <= -18)
            upto = p[:hit[0] + 1] if len(hit) else p
            assert R == pytest.approx(max(0., upto.max()) / 18)


def test_tug_of_war_fades_and_opposition_cuts_faster():
    u = np.zeros(400); u[300] = 1.                                   # one long impulse at t = 300
    s = rc.tug_of_war(u, half_life=50)
    assert s[301] == 0 and s[300] > 0 and s[250] == pytest.approx(s[300] / 2, rel=1e-9)
    assert np.allclose(rc.tug_of_war(np.ones(3000), half_life=50)[:100], 1., atol=1e-6)
    v = u.copy(); v[200] = -.2                                       # a weak short impulse earlier in time
    a, b = rc.tug_of_war(v, 50), rc.tug_of_war(v, 50, opposite_ratio=1.)
    assert a[150] < b[150]                                           # opposition shrinks the long score faster
    assert rc.impulse(np.array([5., .5]), np.array([0., 3.])).tolist() == [1., -.5]


def test_strength_and_impulse_handle_censored_sides():
    side, R = rc.strength(np.array([3., .5, np.nan, 2.5]), np.array([np.nan, np.nan, 1., 1.]))
    assert side.tolist() == [1., 1., -1., 1.]
    assert R[0] == 3. and np.isnan(R[1]) and np.isnan(R[2]) and R[3] == 2.5     # open outcomes stay unknown
    u = rc.impulse(np.array([1.5, 2., 3., 4.5]), np.array([0., 0., 0., 0.]))
    assert u.tolist() == [0., 0., .5, 1.]                                          # nothing below a real 2G move
