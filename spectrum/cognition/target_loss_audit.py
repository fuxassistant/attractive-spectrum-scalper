"""Independent algebra and masked temporal diagnostics for the frozen HGB lab."""
import numpy as np
from scipy.special import expit, logsumexp, softmax

NS = 1_000_000_000


def loss_gradient_hessian(y, raw):
    """Per-row CE, derivative w.r.t. logits and Hessian diagonal (not weights)."""
    y = np.asarray(y, int); z = np.asarray(raw, float)
    if z.ndim == 1 or z.shape[1] == 1:
        z = z.reshape(-1); p = expit(z)
        return np.logaddexp(0., z) - y*z, p-y, p*(1-p)
    p = softmax(z, axis=1); onehot = np.eye(z.shape[1])[y]
    return logsumexp(z, axis=1)-z[np.arange(len(y)), y], p-onehot, p*(1-p)


def head_targets(category, head):
    y = np.asarray(category, int)
    mask = y >= 0 if head == 'fill' else (y >= 0) & (y < 3)
    return mask, (y[mask] < 3).astype(int) if head == 'fill' else y[mask]


def leaf_ids(model, tree, x):
    """Traverse actual histogram bins, including learned missing routes."""
    bins = model._bin_mapper.transform(np.asarray(x, dtype=np.float64))
    nodes = tree.nodes
    if np.any(nodes['is_categorical']):
        raise ValueError('This audit expects the actual numerical-only v2 models')
    ids = np.zeros(len(x), np.int64)
    while True:
        active = ~nodes['is_leaf'][ids].astype(bool)
        if not active.any():
            return ids
        r = np.flatnonzero(active); n = nodes[ids[r]]
        values = bins[r, n['feature_idx']]
        left = np.where(values == model._bin_mapper.missing_values_bin_idx_,
                        n['missing_go_to_left'].astype(bool), values <= n['bin_threshold'])
        ids[r] = np.where(left, n['left'], n['right'])


def consecutive_pairs(t, valid, segment, step_seconds, lag_seconds):
    """Require an unbroken origin grid and support across the whole lag."""
    t = np.asarray(t, np.int64); valid = np.asarray(valid, bool); segment = np.asarray(segment)
    k = int(lag_seconds // step_seconds)
    if k < 1 or k*step_seconds != lag_seconds or len(t) <= k:
        return np.empty(0, int), np.empty(0, int)
    breaks = np.r_[True, (np.diff(t) != step_seconds*NS) | (segment[1:] != segment[:-1]) |
                         ~valid[1:] | ~valid[:-1]]
    block = np.cumsum(breaks)
    a = np.arange(len(t)-k); b = a+k
    ok = valid[a] & valid[b] & (block[a] == block[b])
    return a[ok], b[ok]


def lag_record(t, values, segment, support, step, lag):
    """Missing values exclude endpoints; support/grid failures break links."""
    a, b = consecutive_pairs(t, support, segment, step, lag)
    v = np.asarray(values, float)
    keep = np.isfinite(v[a]) & np.isfinite(v[b]); a, b = a[keep], b[keep]
    x, y = v[a], v[b]
    return dict(n=len(a), sum_x=float(x.sum()), sum_y=float(y.sum()),
        sum_x2=float(x@x), sum_y2=float(y@y), sum_xy=float(x@y),
        sum_abs_change=float(np.abs(x-y).sum()), sum_sq_change=float(((x-y)**2).sum()),
        changed=int(np.count_nonzero(x != y)))


def combine_lags(records):
    if not records:
        return dict(n=0)
    r = {k: sum(d[k] for d in records) for k in records[0]}; n = r['n']
    if not n:
        return dict(n=0)
    vx=r['sum_x2']-r['sum_x']**2/n; vy=r['sum_y2']-r['sum_y']**2/n
    r.update(mean_abs_change=r['sum_abs_change']/n, mean_sq_change=r['sum_sq_change']/n,
             change_rate=r['changed']/n,
             correlation=(r['sum_xy']-r['sum_x']*r['sum_y']/n)/np.sqrt(vx*vy) if vx>0 and vy>0 else None)
    return r


def run_lengths(t, values, segment, support, step):
    """Observed equal-label stretches; lengths are grid spans, not holding time."""
    t=np.asarray(t); v=np.asarray(values); valid=np.asarray(support)&np.isfinite(v)
    if not len(t): return []
    boundary=np.r_[True, (np.diff(t)!=step*NS)|(np.diff(segment)!=0)|~valid[:-1]|~valid[1:]|(v[:-1]!=v[1:])]
    starts=np.flatnonzero(boundary); ends=np.r_[starts[1:],len(t)]
    return [(int(v[a]),int(b-a)) for a,b in zip(starts,ends) if valid[a]]
