import numpy as np
import pytest

import spectrum.features.families.anchored_profile as hp

NS = 10**9


def fake_base(seed, n=7200, london=1800, date='2025-10-02'):
    rng = np.random.default_rng(seed)
    mid = 40000 + np.cumsum(rng.choice([-.5, 0, .5], n))
    vol = rng.poisson(3, n).astype(float)
    b = dict(t_ns=np.arange(n, dtype=np.int64) * NS, mid=mid, spread=np.full(n, 2.), seg=np.ones(n, np.int64),
             step=np.r_[False, np.ones(n - 1, bool)], sigma=np.full(n, 1.5), sigma_known=np.ones(n, bool),
             trade_buy=vol * .5, trade_sell=vol * .4, trade_unknown=vol * .1, add_bid=vol, add_ask=vol, remove_bid=vol * .5,
             remove_ask=vol * .6, depth1=rng.uniform(-1, 1, n), depth4=rng.uniform(-1, 1, n))
    return b, dict(date=date, start_ns=0, end_ns=n * NS, london=london, ny_close=n - 600, n=n)


def test_features_do_not_see_the_future():
    prev, today = fake_base(1), fake_base(2)
    f0 = hp.features_day(prev, today)
    cut = 5000
    b2 = {k: v.copy() for k, v in today[0].items()}
    b2['mid'][cut + 1:] += np.cumsum(np.random.default_rng(9).normal(0, 3, len(b2['mid']) - cut - 1))
    b2['trade_buy'][cut + 1:] *= 7
    f1 = hp.features_day(prev, (b2, today[1]))
    cols = hp.PATH + hp.FLOW + hp.LEVELS
    early = f0.k <= cut
    assert np.allclose(f0.loc[early, cols].to_numpy(), f1.loc[early, cols].to_numpy())


def test_profile_scan_matches_a_direct_histogram():
    rng = np.random.default_rng(3)
    pb = rng.integers(0, 40, 3000); v = rng.poisson(2, 3000).astype(float); sample = np.sort(rng.choice(3000, 50, replace=False))
    poc, val, vah, dens, tot, occ = hp.profile_scan(pb.astype(np.int64), v, 0, sample.astype(np.int64), 40, .7)
    for i, s in enumerate(sample):
        h = np.bincount(pb[:s + 1], weights=v[:s + 1], minlength=40)
        assert h[poc[i]] == h.max() and tot[i] == pytest.approx(h.sum())
        assert h[val[i]:vah[i] + 1].sum() >= .7 * h.sum() - 1e-9
        assert dens[i] == pytest.approx(h[max(0, pb[s] - 2):pb[s] + 3].sum())
