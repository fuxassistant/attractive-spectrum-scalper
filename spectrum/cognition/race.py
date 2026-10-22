"""Long/short equity race from every second (RACE_V1): which side's path after costs won, and how cleanly.

From each decision second t the mid path decides a race between the long and the short equity:
the long wins when the mid has moved +G ticks before it moved -G, the short the mirror; nobody wins
when neither happens within the horizon H(t) = clip(K * (G / sigma_t)^2, H_MIN, H_MAX) seconds, so the
price scale is tied to costs (ticks) and the time scale to volatility (causal sigma600, ticks/sqrt(s)).
Costs follow the passive scalper: queue entry, limit take-profit (C_WIN ticks incl. fee), taker stop
and taker time exit (C_LOSS ticks). Scores are each side's race result after costs divided by G:
    win  (G - C_WIN) / G,   loss  -(G + C_LOSS) / G,   unresolved  (signed move at H - C_LOSS) / G.
Opportunity = the better side's score when positive; a win is clean when the winner's adverse
excursion before winning stayed below CLEAN * G. Everything here is outcome (Y), never a feature.
The 1-second as-of mid grid is used: touches inside a second are not seen.
"""
import numpy as np
from numba import njit

GRID_G = (12, 18, 27, 40)
K, H_MIN, H_MAX = 2., 20, 900
C_WIN, C_LOSS = 2.2, 4.5
CLEAN = .5
CENSORED = -9


@njit(cache=True)
def race_scan(mid, seg, start, G, hmax, winner, t_win, mae, end_move):
    """winner +1 long, -1 short, 0 unresolved by hmax, CENSORED on support loss or day end. G per row (ticks)."""
    n = len(mid)
    for i in range(len(start)):
        k = start[i]; m0 = mid[k]; hi = 0.; lo = 0.; w = 0; d = 0.; g = G[i]
        if not np.isfinite(m0):
            winner[i] = CENSORED; continue
        for j in range(1, hmax[i] + 1):
            idx = k + j
            if idx >= n or not np.isfinite(mid[idx]) or seg[idx] != seg[k]:
                w = CENSORED; break
            d = mid[idx] - m0
            if d >= g:
                w = 1; t_win[i] = j; mae[i] = -lo; break
            if d <= -g:
                w = -1; t_win[i] = j; mae[i] = hi; break
            if d > hi:
                hi = d
            if d < lo:
                lo = d
        winner[i] = w; end_move[i] = d


def horizon(sigma, G):
    return np.clip(np.round(K * (G / np.maximum(sigma, 1e-6)) ** 2), H_MIN, H_MAX).astype(np.int64)


def race(mid, seg, start, sigma, G, barrier=None):
    """Race outcome and scores for decision seconds `start`. The horizon comes from the scale G (ticks);
    `barrier` (ticks, per row) replaces G as the race magnitude when given, the horizon staying the same."""
    n = len(start); h = horizon(sigma, G)
    G = np.full(n, float(G)) if barrier is None else np.asarray(barrier, float)
    w = np.zeros(n, np.int64); tw = np.zeros(n, np.int64); mae = np.full(n, np.nan); end = np.zeros(n)
    race_scan(np.asarray(mid, float), np.asarray(seg, np.int64), np.asarray(start, np.int64), G, h, w, tw, mae, end)
    win, loss = (G - C_WIN) / G, -(G + C_LOSS) / G
    sl = np.select([w == 1, w == -1, w == 0], [win, loss, (end - C_LOSS) / G], np.nan)
    ss = np.select([w == -1, w == 1, w == 0], [win, loss, (-end - C_LOSS) / G], np.nan)
    clean = np.isin(w, (1, -1)) & (mae < CLEAN * G)
    opp = np.where(np.isfinite(sl), np.maximum(np.maximum(sl, ss), 0.), np.nan)
    return dict(winner=w, t_win=tw, mae=mae, end=end, H=h, long=sl, short=ss, opp=opp, clean=clean,
                dominance=(sl - ss) / 2)


def sign_flipped(mid, seg, rng):
    """Null path: the same 1-second increments (sizes, clustering, tick grid) with random signs."""
    d = np.diff(mid, prepend=np.nan)
    ok = np.isfinite(d) & (seg == np.roll(seg, 1))
    d = np.where(ok, d * rng.choice([-1., 1.], len(d)), 0.)
    base = mid[np.isfinite(mid)][0] if np.isfinite(mid).any() else 0.
    return np.where(np.isfinite(mid), base + np.cumsum(d), np.nan)


def runs(w):
    """Lengths (in samples) of maximal runs of the same non-zero winner."""
    w = np.asarray(w); out = []; cur = 0; val = 0
    for x in w:
        if x == val and x != 0:
            cur += 1
        else:
            if val != 0 and cur:
                out.append(cur)
            val, cur = x, (1 if x in (1, -1) else 0)
    if val in (1, -1) and cur:
        out.append(cur)
    return np.asarray(out)


@njit(cache=True)
def excursion_scan(mid, seg, start, G, hmax, fav_long, fav_short):
    """Favourable excursion of each side before its own stop (-G for that side) or the horizon, in ticks.
    NaN when the path loses support or the day ends before the side is stopped or the horizon is reached."""
    n = len(mid)
    for i in range(len(start)):
        k = start[i]; m0 = mid[k]; hi = 0.; lo = 0.; long_open = True; short_open = True
        fav_long[i] = np.nan; fav_short[i] = np.nan
        if not np.isfinite(m0):
            continue
        cut = False
        for j in range(1, hmax[i] + 1):
            idx = k + j
            if idx >= n or not np.isfinite(mid[idx]) or seg[idx] != seg[k]:
                cut = True; break
            d = mid[idx] - m0
            if long_open and d <= -G[i]:
                fav_long[i] = hi; long_open = False
            if short_open and d >= G[i]:
                fav_short[i] = -lo; short_open = False
            if d > hi:
                hi = d
            if d < lo:
                lo = d
            if not long_open and not short_open:
                break
        if not cut:
            if long_open:
                fav_long[i] = hi
            if short_open:
                fav_short[i] = -lo


def excursion(mid, seg, start, sigma, G, barrier=None):
    """Strength of each side: favourable move before its -G stop within H, in units of G."""
    n = len(start); h = horizon(sigma, G)
    bar = np.full(n, float(G)) if barrier is None else np.asarray(barrier, float)
    fl = np.full(n, np.nan); fs = np.full(n, np.nan)
    excursion_scan(np.asarray(mid, float), np.asarray(seg, np.int64), np.asarray(start, np.int64), bar, h, fl, fs)
    return dict(R_long=fl / bar, R_short=fs / bar, H=h)


@njit(cache=True)
def _tug(u, lam_neutral, lam_same, lam_opposite, gain):
    """Backward recursion: S(t) = lam_t * S(t+1) + gain * u(t); lam_t depends on whether u(t) agrees with S(t+1)."""
    n = len(u); s = np.zeros(n); nxt = 0.
    for t in range(n - 1, -1, -1):
        x = u[t]
        if not np.isfinite(x):
            x = 0.
        if x == 0.:
            lam = lam_neutral
        elif x * nxt < 0:
            lam = lam_opposite
        else:
            lam = lam_same
        nxt = lam * nxt + gain * x
        s[t] = nxt
    return s


def tug_of_war(u, half_life, opposite_ratio=5.):
    """Fading tug-of-war score of one-sided opportunity impulses u(t) in [-1, 1], integrated from the future
    to the present: without impulses the score halves every `half_life` seconds; an impulse against the
    current score shrinks it `opposite_ratio` times faster. Steady impulses of 1 give a score of 1."""
    lam = 2. ** (-1. / half_life); lam_opp = 2. ** (-opposite_ratio / half_life)
    return _tug(np.asarray(u, float), lam, lam, lam_opp, 1. - lam)


def strength(R_long, R_short, rho=2.):
    """Best side and its strength; NaN when a censored side leaves the outcome open (the known side below rho)."""
    L, S = np.asarray(R_long, float), np.asarray(R_short, float)
    side = np.where(np.nan_to_num(L, nan=-np.inf) >= np.nan_to_num(S, nan=-np.inf), 1., -1.)
    R = np.fmax(L, S)
    open_ = (~np.isfinite(L) | ~np.isfinite(S)) & ~(R >= rho)
    return side, np.where(open_, np.nan, R)


def impulse(R_long, R_short, lo=2., hi=4.):
    """Signed impulse: 0 unless the best side moved lo * G before its stop (a real one-sided opportunity),
    rising linearly to 1 at hi * G."""
    side, R = strength(R_long, R_short, lo)
    return np.where(np.isfinite(R), side * np.clip((R - lo) / (hi - lo), 0., 1.), np.nan)
