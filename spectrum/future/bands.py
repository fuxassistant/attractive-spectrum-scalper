"""Band-relative questions about the future (BANDS_V1); bands come from crossings.vwap.vwap_bands.

At every second t and scale j (window W_j, horizon H_j = W_j / 4) the bands are frozen from the past only:
VWAP_j(t) over (t - W_j, t] with weights = traded volume of the second + a small causal floor (quiet seconds
still count), sigma_vw = volume-weighted dispersion of the mid around that VWAP, half-width B = max(K * sigma_vw,
B_MIN ticks), bands VWAP +- B. Questions about the path p(t + tau), tau = 1..H_j (all outcome, Y):
  O  sleeve occupation: time-mean of (p(t + tau) - VWAP) / B  ("price delta x time" in band units);
     without clipping it splits exactly into O = z + D, z = (p(t) - VWAP) / B known at t;
  D  time-mean of (p(t + tau) - p(t)) / B: the future part only;
  V  time-mean of |p(t + tau) - p(t)| / B: volatility in band units;
  E  +1 / -1 if the frozen band is left upwards / downwards first within H, 0 if it is not; by the canon
     crossing rule bid OR ask: up when ask reaches VWAP + B, down when bid reaches VWAP - B. Asked only
     when at t ask < VWAP + B and bid > VWAP - B; NaN if both lines are reached in the same second.
O, D, V use the mid; E uses bid and ask (same units as mid).
NaN when the path loses support or changes segment within the horizon.
"""
import numpy as np
from numba import njit
from spectrum.leakage import future_output

@njit(cache=True)
def _future(mid, bid, ask, seg, start, vwap, half, H, O, D, V, E):
    n = len(mid)
    for i in range(len(start)):
        t = start[i]; p0 = mid[t]; vw = vwap[i]; b = half[i]
        O[i] = np.nan; D[i] = np.nan; V[i] = np.nan; E[i] = np.nan
        if not (np.isfinite(p0) and np.isfinite(vw) and b > 0) or t + H >= n:
            continue
        inside = ask[t] < vw + b and bid[t] > vw - b
        so = 0.; sd = 0.; sv = 0.; e = 0.; bad = False
        for j in range(1, H + 1):
            p = mid[t + j]
            if not (np.isfinite(p) and np.isfinite(bid[t + j]) and np.isfinite(ask[t + j])) or seg[t + j] != seg[t]:
                bad = True; break
            so += (p - vw) / b; sd += (p - p0) / b; sv += abs(p - p0) / b
            if e == 0.:
                up = ask[t + j] >= vw + b; down = bid[t + j] <= vw - b
                e = np.nan if up and down else 1. if up else -1. if down else 0.
        if bad:
            continue
        O[i] = so / H; D[i] = sd / H; V[i] = sv / H
        E[i] = e if inside else np.nan


@future_output()
def future(mid, bid, ask, seg, start, vwap, half, H):
    """Band questions about the next H seconds from each start (vwap / half are the values frozen at start)."""
    n = len(start); O, D, V, E = (np.full(n, np.nan) for _ in range(4))
    _future(np.asarray(mid, float), np.asarray(bid, float), np.asarray(ask, float), np.asarray(seg, np.int64), np.asarray(start, np.int64), np.asarray(vwap, float),
            np.asarray(half, float), int(H), O, D, V, E)
    return dict(future_O=O, future_D=D, future_V=V, future_E=E)
