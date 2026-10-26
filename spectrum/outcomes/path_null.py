"""Finite-horizon Brownian null for two-barrier plans (PATH_MEASUREMENT_V1, null 1).

Driftless Brownian mid with sigma ticks/sqrt(s), started at x inside [0, L]: probability that the
upper barrier L (resp. the lower barrier 0) is hit first and by time t. Short times use the method
of images, long times the eigenfunction (Fourier) series; both agree to < 1e-9 where they overlap.
Barrier distances are rounded up to the half-tick lattice of the mid, as the scanner measures them.
"""
import numpy as np
from scipy.special import erfc

SWITCH = .5        # sigma^2 t / L^2 below this: images
IMAGES = 8
MODES = 40


def _images(x, L, s):
    k = np.arange(-IMAGES, IMAGES + 1).reshape((-1,) + (1,) * np.ndim(x))
    a = x + 2 * k * L
    return np.sum(np.sign(a) * erfc(np.abs(a) / (np.sqrt(2.) * s)), 0)


def _fourier_upper(x, L, rho):
    n = np.arange(1, MODES + 1).reshape((-1,) + (1,) * np.ndim(x))
    terms = 2 * (-1.) ** (n + 1) / (n * np.pi) * np.sin(n * np.pi * x / L) * np.exp(-(n * np.pi) ** 2 * rho / 2)
    return x / L - np.sum(terms, 0)


def first_upper(x, L, sigma, t):
    """P(hit L before 0 and by t), BM from x with sigma per sqrt(s). Arrays broadcast."""
    x, L, sigma, t = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, L, sigma, t)))
    s = sigma * np.sqrt(np.maximum(t, 0.)); rho = (s / np.maximum(L, 1e-12)) ** 2
    out = np.zeros(x.shape)
    small = (rho < SWITCH) & (s > 0); big = rho >= SWITCH
    if small.any():
        out[small] = _images(L[small] - x[small], L[small], s[small])
    if big.any():
        out[big] = _fourier_upper(x[big], L[big], rho[big])
    return np.clip(out, 0., 1.)


def two_barrier(up, down, sigma, t):
    """(P favourable barrier first by t, P adverse barrier first by t, P neither) for distances up/down in ticks."""
    L = up + down
    p_up = first_upper(down, L, sigma, t); p_dn = first_upper(up, L, sigma, t)
    return p_up, p_dn, np.clip(1 - p_up - p_dn, 0., 1.)


def lattice(d):
    """Mid moves on a half-tick grid: a threshold d ticks is reached at the next half-tick at or above it."""
    return np.ceil(np.asarray(d, float) * 2 - 1e-9) / 2


def plan_bins(take, stop, sigma, edges):
    """Brownian hazards per exit bin: alpha_k = P(event in bin k | none before), beta_k = P(take | event in k).
    take/stop are mid distances in ticks (n,), edges the bin ends in seconds (K,). Returns (n, K) arrays and
    the cumulative take/stop/neither probabilities at the last edge."""
    g, r = lattice(take)[:, None], lattice(stop)[:, None]; e = np.asarray(edges, float)[None, :]
    ft, fs, _ = two_barrier(g, r, np.asarray(sigma, float)[:, None], e)
    ft = np.concatenate([np.zeros((len(g), 1)), ft], 1); fs = np.concatenate([np.zeros((len(g), 1)), fs], 1)
    dt, ds = np.diff(ft, axis=1), np.diff(fs, axis=1); surv = 1 - ft[:, :-1] - fs[:, :-1]
    alpha = np.clip((dt + ds) / np.maximum(surv, 1e-12), 1e-6, 1 - 1e-6)
    beta = np.clip(dt / np.maximum(dt + ds, 1e-15), 1e-6, 1 - 1e-6)
    return alpha, beta, (ft[:, -1], fs[:, -1], 1 - ft[:, -1] - fs[:, -1])


def never_positive(cost, sigma, t):
    """P(max of the mid move over [0, t] stays below cost), driftless."""
    from scipy.special import erf
    return erf(np.asarray(cost, float) / (np.sqrt(2.) * sigma * np.sqrt(t)))
