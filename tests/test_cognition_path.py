import numpy as np
import pytest

from spectrum.cognition import path_scan as ps
from spectrum.cognition import scalper as sc
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.passive import Quotes

NS = 10**9
TICK = 100_000_000


def random_tape(seed, n=6000, deadline_s=None):
    """Half-tick mid walk with jumps, varying spreads, repeated timestamps, gaps, invalid quotes and an epoch change."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0, .01, .05, .2, 1., 3.], n, p=[.1, .3, .3, .2, .08, .02]); dt[0] = 0
    t = np.rint(np.cumsum(dt) * NS).astype(np.int64)
    spread = rng.choice([1, 2, 3, 4, 6], n, p=[.3, .3, .2, .15, .05])
    mid2 = 40000 * 2 + np.cumsum(rng.choice([-3, -1, 0, 1, 3], n, p=[.05, .3, .3, .3, .05]))
    mid2 = np.where((mid2 + spread) % 2 == 0, mid2, mid2 + 1)          # bid/ask on the tick grid
    bid = ((mid2 - spread) // 2 * TICK).astype(np.int64); ask = bid + spread * TICK
    valid = rng.random(n) > .0003
    epoch = np.where(np.arange(n) < n * 3 // 4, 1, 2).astype(np.int32)
    dl = int(t[-1] * .9) if deadline_s is None else deadline_s * NS
    return QuoteTape(t, bid, ask, valid, epoch, int(t[-1]) + NS, 0, int(t[-1]), dl)


def reference(q, start, side, entry, take, stop, hold, horizon):
    st = np.zeros(len(start), np.int8); ex = np.zeros(len(start), np.int64); net = np.full(len(start), np.nan)
    sc.scan_exits(q.t, q.bid, q.ask, q.ok, q.epoch, q.deadline, start, np.full(len(start), side, np.int64), entry,
                  np.ones(len(start), bool), take, stop, np.full(len(start), hold * NS, np.int64), horizon, st, ex, net)
    return st, ex, net


@pytest.mark.parametrize('seed', range(6))
def test_plans_from_the_ladder_equal_scan_exits(seed):
    q = Quotes(random_tape(seed)); rng = np.random.default_rng(100 + seed)
    start = np.sort(rng.choice(np.flatnonzero(q.ok), 400, replace=False)).astype(np.int64)
    sigma = rng.uniform(.3, 3., len(start))
    thr = ps.thresholds(sigma); lad = ps.ladder(q, start, thr)
    for side in (1, -1):
        entry = (q.ask if side > 0 else q.bid)[start].astype(np.int64)
        for name, trig, hold, take, stop in ps.ADAPTIVE:
            ct, cs = ps.PLAN_LEVELS.index((take, hold)), ps.PLAN_LEVELS.index((stop, hold))
            unit = sigma * np.sqrt(hold) * TICK
            horizon = q.t[start] - rng.integers(0, 2 * NS, len(start)) + (hold + 5) * NS
            st, ex, net = reference(q, start, side, entry, take * unit, stop * unit, hold, horizon)
            a = ps.assemble(q.t, q.bid, q.ask, start, side, entry, lad, ct, cs, hold, horizon)
            assert np.array_equal(np.where(a['done'], sc.COMPLETE, sc.CENSORED), st), name
            done = st == sc.COMPLETE
            assert np.array_equal(a['exit_ns'][done], ex[done])
            assert np.array_equal(a['net_ticks'][done], net[done])
            assert .2 < done.mean() < 1, name


def brute(q, j0, thr):
    mid0 = .5 * (q.bid[j0] + q.ask[j0]); up = dn = -1; lo = hi = 0.
    for j in range(j0, len(q.t)):
        if not q.ok[j] or q.epoch[j] != q.epoch[j0] or q.t[j] >= q.deadline:
            break
        mv = .5 * (q.bid[j] + q.ask[j]) - mid0; lo = min(lo, mv); hi = max(hi, mv)
        if up < 0 and mv >= thr:
            up, up_lo = j, lo
        if dn < 0 and mv <= -thr:
            dn, dn_hi = j, hi
        if q.t[j] - q.t[j0] >= ps.SPAN * NS:
            break
    return up, dn, (up_lo if up >= 0 else None), (dn_hi if dn >= 0 else None)


def test_ladder_matches_a_direct_search_and_is_ordered():
    q = Quotes(random_tape(11)); rng = np.random.default_rng(3)
    start = np.sort(rng.choice(np.flatnonzero(q.ok), 120, replace=False)).astype(np.int64)
    thr = ps.thresholds(rng.uniform(.3, 3., len(start))); lad = ps.ladder(q, start, thr)
    for k in range(len(start)):
        for i in range(thr.shape[1]):
            up, dn, up_lo, dn_hi = brute(q, start[k], thr[k, i])
            assert lad['up_j'][k, i] == up and lad['dn_j'][k, i] == dn
            if up >= 0:
                assert lad['up_lo'][k, i] == up_lo
            if dn >= 0:
                assert lad['dn_hi'][k, i] == dn_hi
        o = np.argsort(thr[k])
        for key in ('up_j', 'dn_j'):
            j = lad[key][k, o]; reached = j >= 0
            assert np.all(np.diff(j[reached]) >= 0) and np.all(np.diff(reached.astype(int)) <= 0)
        both = (lad['up_j'][k] >= 0) & (lad['dn_j'][k] >= 0)
        assert not np.any(lad['up_j'][k, both] == lad['dn_j'][k, both])


def test_excursions_and_exit_bins():
    q = Quotes(random_tape(21)); rng = np.random.default_rng(8)
    start = np.sort(rng.choice(np.flatnonzero(q.ok)[:4000], 200, replace=False)).astype(np.int64)
    sigma = rng.uniform(.5, 2., len(start)); lad = ps.ladder(q, start, ps.thresholds(sigma))
    name, trig, hold, take, stop = ps.ADAPTIVE[3]
    ct, cs = ps.PLAN_LEVELS.index((take, hold)), ps.PLAN_LEVELS.index((stop, hold))
    for side in (1, -1):
        entry = (q.ask if side > 0 else q.bid)[start].astype(np.int64)
        a = ps.assemble(q.t, q.bid, q.ask, start, side, entry, lad, ct, cs, hold, q.t[start] + (hold + 5) * NS)
        for k in np.flatnonzero(a['done'] & (a['cat'] != ps.DEADLINE)):
            j0, j1 = start[k], a['exit_j'][k]
            mv = side * (.5 * (q.bid[j0:j1 + 1] + q.ask[j0:j1 + 1]) - .5 * (q.bid[j0] + q.ask[j0])) / TICK
            assert a['mfe'][k] == pytest.approx(mv.max()) and a['mae'][k] == pytest.approx(-mv.min())
            assert a['mv'][k] == pytest.approx(mv[-1])
            if a['cat'][k] == ps.TAKE:
                assert mv[-1] * TICK >= take * sigma[k] * np.sqrt(hold) * TICK
            if a['cat'][k] == ps.STOP:
                assert mv[-1] * TICK <= -stop * sigma[k] * np.sqrt(hold) * TICK
    b = ps.exit_bin(np.array([.5, 5., 5.01, 29., 31., np.nan]), 30)
    assert b.tolist() == [0, 0, 1, 2, 2, -1]


def test_brownian_null_reproduces_the_finite_horizon_table():
    from spectrum.cognition import path_null as pn
    # g = r = 10 net, cost 5: barriers +15 / -5 around the mid, sigma 1.5 ticks/sqrt(s)
    for t, up, dn, none in ((10, .0016, .2918, .7066), (30, .0655, .5428, .3917), (60, .1653, .6645, .1702)):
        p = pn.two_barrier(np.array(15.), np.array(5.), 1.5, t)
        assert [round(float(v), 4) for v in p] == pytest.approx([up, dn, none], abs=1e-4)
    assert float(pn.two_barrier(np.array(15.), np.array(5.), 1.5, 3600)[0]) == pytest.approx(.25, abs=1e-6)
    assert float(pn.never_positive(5., 1.5, 60)) == pytest.approx(.333, abs=1e-3)


def test_images_and_fourier_agree_and_bins_add_up():
    from spectrum.cognition import path_null as pn
    x, L = np.full(200, 3.), np.full(200, 10.); t = np.linspace(.5, 40, 200)
    rho = t * 1.2 ** 2 / L ** 2
    a = pn._images(L - x, L, 1.2 * np.sqrt(t)); b = pn._fourier_upper(x, L, rho)
    ok = (rho > .05) & (rho < 2)
    assert np.max(np.abs(a[ok] - b[ok])) < 1e-9
    take, stop, sigma = np.array([4.1, 7.5]), np.array([4.1, 3.6]), np.array([1., 2.])
    alpha, beta, (ft, fs, fn) = pn.plan_bins(take, stop, sigma, (5, 15, 30))
    surv = np.cumprod(np.c_[np.ones(2), 1 - alpha[:, :-1]], 1)
    assert np.sum(surv * alpha * beta, 1) == pytest.approx(ft, abs=1e-6)
    assert np.sum(surv * alpha * (1 - beta), 1) == pytest.approx(fs, abs=1e-6)


def test_brownian_null_matches_a_lattice_walk():
    from spectrum.cognition import path_null as pn
    rng = np.random.default_rng(2)
    n, dt, sigma = 20000, .01, 1.0
    steps = rng.normal(0, sigma * np.sqrt(dt), (n, 1000))      # 10 s
    path = np.cumsum(steps, 1); up = np.argmax(path >= 4, 1); dn = np.argmax(path <= -2, 1)
    hit_up = (path >= 4).any(1) & (~(path <= -2).any(1) | (up < dn))
    p = pn.two_barrier(np.array(4.), np.array(2.), sigma, 10)
    assert hit_up.mean() == pytest.approx(float(p[0]), abs=.012)


def test_when_and_what_losses_add_up_to_the_outcome_likelihood():
    from spectrum.cognition import path_model as pm
    rng = np.random.default_rng(4); n, K = 50, 5
    alpha, beta = rng.uniform(.05, .6, (n, K)), rng.uniform(.2, .8, (n, K))
    cat = rng.choice([ps.TAKE, ps.STOP, ps.TIME], n); b = np.where(cat == ps.TIME, K - 1, rng.integers(0, K, n))
    meta = dict(st=np.full(n, sc.COMPLETE), cat=cat, bin=b)
    pt, pstop, ptime = pm.components(alpha, beta)
    assert pt + pstop + ptime == pytest.approx(np.ones(n))
    when, what = pm.losses(alpha, beta, meta)
    surv = np.cumprod(np.c_[np.ones(n), 1 - alpha], 1)
    for i in range(n):
        if cat[i] == ps.TIME:
            p = surv[i, K]
        else:
            p = surv[i, b[i]] * alpha[i, b[i]] * (beta[i, b[i]] if cat[i] == ps.TAKE else 1 - beta[i, b[i]])
        assert when[i] + np.nan_to_num(what[i]) == pytest.approx(-np.log(p))


def test_account_is_busy_until_the_exit_and_resets_each_day():
    import pandas as pd
    from spectrum.cognition import path_model as pm
    best = pd.DataFrame(dict(date=['d1'] * 4 + ['d2'], t_ns=np.array([0, 5, 10, 40, 0]) * NS, s=[1., 1., 1., 1., 1.],
                             busy=[12., 1., 1., np.nan, 3.], hold=[30] * 5, net=[1., 2., 3., np.nan, 5.], st=[1] * 5))
    tr = pm.account(best, 's', theta=0.)
    assert tr.t_ns.tolist() == [0, 40 * NS, 0]            # 5 and 10 fall inside the first trade
    assert pm.account(best, 's', theta=1.).empty


def test_hedge_regret_stays_under_its_bound_and_ledger_identity_holds():
    import pandas as pd
    from spectrum.cognition import path_model as pm
    rng = np.random.default_rng(9)
    daily = pd.DataFrame(rng.normal(0, 20, (60, 6)), columns=[f'e{i}' for i in range(6)])
    daily['e3'] += 4; daily['WAIT'] = 0.
    h, mixed = pm.hedge(daily)
    assert h['regret'] <= h['bound'] and h['net'] == pytest.approx(mixed.sum())
    t = pd.DataFrame(dict(st=[sc.COMPLETE, sc.EMPTY, sc.COMPLETE], cat=[ps.TAKE, 0, ps.TIME], net=[1.2, 0., -.4],
                          p_take_full=[.3, .2, .4], p_stop_full=[.5, .5, .1], p_time_full=[.2, .3, .5],
                          v_take_full=[1., 1.1, .9], v_stop_full=[-1., -.9, -1.2], v_time_full=[-.2, -.1, -.3],
                          fill_full=[1., .7, .6]))
    t['ef_full'] = t.p_take_full * t.v_take_full + t.p_stop_full * t.v_stop_full + t.p_time_full * t.v_time_full
    t['eg_full'] = t.fill_full * t.ef_full
    led = pm.ledger(t)
    assert np.abs(led.identity_gap).max() < 1e-12
