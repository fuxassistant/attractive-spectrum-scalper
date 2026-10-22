import numpy as np
import pytest

from spectrum.cognition import bands as bd


def test_bands_use_only_the_past_and_match_a_direct_weighted_vwap():
    rng = np.random.default_rng(0)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 5000)); vol = rng.poisson(3, 5000).astype(float)
    b = bd.vwap_bands(mid, vol, windows=(240,))[240]
    run = np.cumsum(vol) / np.arange(1, len(vol) + 1)
    for t in (300, 1200, 4999):
        w = vol[t - 239:t + 1] + bd.FLOOR * run[t - 239:t + 1]; p = mid[t - 239:t + 1]
        vw = np.sum(w * p) / w.sum(); sd = np.sqrt(np.sum(w * (p - vw) ** 2) / w.sum())
        assert b[0][t] == pytest.approx(vw, abs=1e-6) and b[1][t] == pytest.approx(max(bd.K * sd, bd.B_MIN), abs=1e-6)
    mid2, vol2 = mid.copy(), vol.copy(); mid2[3001:] += 50; vol2[3001:] *= 9   # changing the future leaves bands at t untouched
    b2 = bd.vwap_bands(mid2, vol2, windows=(240,))[240]
    assert np.allclose(b[0][:3001], b2[0][:3001], equal_nan=True) and np.allclose(b[1][:3001], b2[1][:3001], equal_nan=True)


def test_future_questions_and_the_occupation_identity():
    rng = np.random.default_rng(1)
    mid = 40000 + np.cumsum(rng.choice([-1, -.5, 0, .5, 1], 4000)); seg = np.ones(4000, np.int64); vol = np.ones(4000)
    b = bd.vwap_bands(mid, vol, windows=(240,))[240]; start = np.arange(300, 3500, 7)
    f = bd.future(mid, seg, start, b[0][start], b[1][start], 60)
    z = (mid[start] - b[0][start]) / b[1][start]
    assert np.allclose(f['O'], z + f['D'])                        # O = z + D exactly
    for i in range(0, len(start), 37):
        t = start[i]; path = mid[t + 1:t + 61]; up = path >= b[0][t] + b[1][t]; dn = path <= b[0][t] - b[1][t]
        first = np.flatnonzero(up | dn)
        if abs(mid[t] - b[0][t]) >= b[1][t]:
            assert np.isnan(f['E'][i])                          # no exit question from outside the band
        else:
            assert f['E'][i] == (0 if not len(first) else (1 if up[first[0]] else -1))
        assert f['V'][i] == pytest.approx(np.mean(np.abs(path - mid[t])) / b[1][t])
