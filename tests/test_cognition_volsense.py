import numpy as np
import pandas as pd
import pytest

from spectrum.cognition import volsense as vs

NS = 10**9


def fake(seed, n=9000, london=2000, date='2025-11-17', start='2025-11-16 23:00'):
    rng = np.random.default_rng(seed)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], n, p=[.1, .2, .4, .2, .1])); mid[3000:3004] = np.nan
    ok = np.isfinite(mid); step = np.r_[False, ok[1:] & ok[:-1]]
    vol = rng.poisson(3, n).astype(float)
    b = dict(mid=mid, step=step, seg=np.ones(n, np.int64), sigma=np.full(n, 1.2), sigma_known=np.r_[np.zeros(600, bool), np.ones(n - 600, bool)],
             sigma60=np.full(n, 1.1), sigma3600=np.full(n, 1.3), sigma3600_known=np.r_[np.zeros(3600, bool), np.ones(n - 3600, bool)],
             trade_buy=vol * .5, trade_sell=vol * .4, trade_unknown=vol * .1, add_bid=vol, add_ask=vol * 1.1, remove_bid=vol * .6, remove_ask=vol * .5,
             spread=rng.choice([2., 3., 4.], n), depth1=rng.uniform(-1, 1, n), depth4=rng.uniform(-1, 1, n),
             depth1_total=rng.uniform(5, 50, n), depth4_total=rng.uniform(50, 200, n), depth12_total=rng.uniform(200, 900, n))
    m = dict(date=date, start_ns=pd.Timestamp(start, tz='UTC').value, london=london, ny_close=n - 500, n=n)
    return b, m


def test_features_labels_and_recent_use_only_the_past():
    b, m = fake(0); cut = 6500
    f0 = vs.features(b, m); lab0 = vs.labels(b); r0 = vs.recent(lab0, m['n'])
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    rng = np.random.default_rng(9)
    b2['mid'][cut + 1:] += np.cumsum(rng.normal(0, 4, m['n'] - cut - 1))
    for k in ('trade_buy', 'trade_sell', 'add_bid', 'remove_ask', 'spread', 'depth4_total', 'sigma'):
        b2[k][cut + 1:] *= 3
    f1 = vs.features(b2, m); lab1 = vs.labels(b2); r1 = vs.recent(lab1, m['n'])
    for c in vs.FEATURES:
        if c in f0:
            assert np.allclose(f0[c][:cut + 1], f1[c][:cut + 1], equal_nan=True), c
    for c, v in r0.items():
        assert np.allclose(v[:cut + 1], r1[c][:cut + 1], equal_nan=True), c
    for G in vs.GRID_G:                                         # a label resolved before the cut does not move
        H = lab0[G]['H']; j = np.flatnonzero(np.isfinite(H) & (np.arange(m['n']) + H <= cut))
        assert np.allclose(lab0[G]['R_long'][j], lab1[G]['R_long'][j], equal_nan=True)
    assert set(vs.FEATURES) - set(f0) <= set(vs.FAMILIES['recent'])


def test_event_clock_follows_daylight_saving():
    summer = vs.event_seconds('2026-07-01', pd.Timestamp('2026-06-30 22:00', tz='UTC').value)
    winter = vs.event_seconds('2025-11-17', pd.Timestamp('2025-11-16 23:00', tz='UTC').value)
    for ev in (summer, winter):                                 # 08:30 ET and 10:30 London are 14.5 h and 11.5 h after the file start
        assert ev['us_data'] == 14.5 * 3600 and ev['lbma_am'] == 11.5 * 3600 and ev['ny_open'] == 15.5 * 3600
