"""VWAP bands at four time scales and band-relative questions about the future (BANDS_V1).

At every second t and scale j (window W_j, horizon H_j = W_j / 4) the bands are frozen from the past only:
VWAP_j(t) over (t - W_j, t] with weights = traded volume of the second + a small causal floor (quiet seconds
still count), sigma_vw = volume-weighted dispersion of the mid around that VWAP, half-width B = max(K * sigma_vw,
B_MIN ticks), bands VWAP +- B. Questions about the path p(t + tau), tau = 1..H_j (all outcome, Y):
  O  sleeve occupation: time-mean of (p(t + tau) - VWAP) / B  ("price delta x time" in band units);
     without clipping it splits exactly into O = z + D, z = (p(t) - VWAP) / B known at t;
  D  time-mean of (p(t + tau) - p(t)) / B: the future part only;
  V  time-mean of |p(t + tau) - p(t)| / B: volatility in band units;
  E  +1 / -1 if the path leaves the frozen band upwards / downwards first within H, 0 if it stays inside;
     asked only from starts inside the band (NaN otherwise).
NaN when the path loses support or changes segment within the horizon.
"""
import numpy as np
from numba import njit

WINDOWS = (240, 600, 1200, 2800)
K = 2.
B_MIN = 2.
FLOOR = .05


def vwap_bands(mid, vol, windows=WINDOWS, k=K):
    """Causal rolling VWAP and half-width per second for every window; mid in ticks, vol per second."""
    ok = np.isfinite(mid)
    ref = mid[ok][0] if ok.any() else 0.                       # numerical anchor only; cancels out
    x = np.where(ok, mid - ref, 0.)
    v = np.nan_to_num(np.asarray(vol, float))
    run_mean = np.cumsum(v) / np.arange(1, len(v) + 1)          # causal floor: mean volume so far
    w = np.where(ok, v + FLOOR * np.maximum(run_mean, 1e-3), 0.)
    c0, c1, c2 = (np.r_[0., np.cumsum(a)] for a in (w, w * x, w * x * x))
    n = len(mid); i = np.arange(n); out = {}
    for W in windows:
        lo = np.clip(i + 1 - W, 0, None)
        s0, s1, s2 = c0[i + 1] - c0[lo], c1[i + 1] - c1[lo], c2[i + 1] - c2[lo]
        good = (s0 > 0) & (i + 1 - W >= 0)
        vw = np.where(good, s1 / np.where(good, s0, 1.), np.nan)
        var = np.where(good, np.maximum(s2 / np.where(good, s0, 1.) - vw ** 2, 0.), np.nan)
        out[W] = (vw + ref, np.maximum(k * np.sqrt(var), B_MIN))
    return out


@njit(cache=True)
def _future(mid, seg, start, vwap, half, H, O, D, V, E):
    n = len(mid)
    for i in range(len(start)):
        t = start[i]; p0 = mid[t]; vw = vwap[i]; b = half[i]
        O[i] = np.nan; D[i] = np.nan; V[i] = np.nan; E[i] = np.nan
        if not (np.isfinite(p0) and np.isfinite(vw) and b > 0) or t + H >= n:
            continue
        so = 0.; sd = 0.; sv = 0.; e = 0.; bad = False
        for j in range(1, H + 1):
            p = mid[t + j]
            if not np.isfinite(p) or seg[t + j] != seg[t]:
                bad = True; break
            so += (p - vw) / b; sd += (p - p0) / b; sv += abs(p - p0) / b
            if e == 0.:
                if p >= vw + b:
                    e = 1.
                elif p <= vw - b:
                    e = -1.
        if bad:
            continue
        O[i] = so / H; D[i] = sd / H; V[i] = sv / H
        E[i] = e if abs(p0 - vw) < b else np.nan          # an exit is only asked from inside the band


def future(mid, seg, start, vwap, half, H):
    """Band questions about the next H seconds from each start (vwap / half are the values frozen at start)."""
    n = len(start); O, D, V, E = (np.full(n, np.nan) for _ in range(4))
    _future(np.asarray(mid, float), np.asarray(seg, np.int64), np.asarray(start, np.int64), np.asarray(vwap, float),
            np.asarray(half, float), int(H), O, D, V, E)
    return dict(O=O, D=D, V=V, E=E)
