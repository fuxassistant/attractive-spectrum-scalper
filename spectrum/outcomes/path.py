"""V6: factorized first-line hit, net outcome, time and path descriptors.

Every output of build_path_targets is a FUTURE LABEL, never an input feature.
The reference equity is liquidation against pre-entry capital, reserving the
roundtrip fee once. Integrals use the exact right-continuous quote path.
"""
import numpy as np
import pandas as pd
from numba import njit

VERSION = 'CROSSING_PATH_TARGET_V6'
METRICS = ('net', 'mae', 'mfe', 'max_drawdown', 'area_positive', 'area_negative',
           'underwater_seconds', 'first_positive_seconds', 'variation',
           'mae_before_cross', 'first_risk_seconds', 'risk_before_cross')


@njit(cache=True)
def scan_paths(t, bid, ask, entries, crosses, risks, fee):
    """Money per price unit, seconds, and money*seconds; no interpolation.

Same-nanosecond publications retain ordinal extrema and ordering, but have
zero integration weight. Peak starts at pre-entry zero (includes entry cost).
Samples at fractions of the *future* event interval are labels, not a clock
available to a live model. Sample 0 is the actual entry ordinal, not a later
publication sharing its timestamp; sample 16 is the actual crossing ordinal.
"""
    n = len(entries)
    out = np.full((n, 2, len(METRICS)), np.nan)
    samples = np.full((n, 2, 17), np.nan)
    for a in range(n):
        e, x = entries[a], crosses[a]
        for side in range(2):
            low, high, peak, dd = 1e100, -1e100, 0., 0.
            ap, am, water, variation = 0., 0., 0., 0.
            first_positive, first_risk = np.nan, np.nan
            prefix_low, risk_before = 1e100, 0.
            previous = 0.
            sample_cursor = 1
            duration_ns = t[x] - t[e]
            for j in range(e, x+1):
                net = ((bid[j]-ask[e]) if side == 0 else (bid[e]-ask[j])) / 1e9 - fee
                if j == e:
                    samples[a, side, 0] = net
                # Interior sample at its latest publication, bounded by x.
                while sample_cursor < 16 and (j == x or
                        16*(t[j+1]-t[e]) > sample_cursor*duration_ns):
                    samples[a, side, sample_cursor] = net
                    sample_cursor += 1
                low, high = min(low, net), max(high, net)
                peak = max(peak, net)
                dd = max(dd, peak-net)
                if j > e:
                    variation += abs(net-previous)
                if net > 0 and np.isnan(first_positive):
                    first_positive = (t[j]-t[e])/1e9
                if net <= -risks[a] and np.isnan(first_risk):
                    first_risk = (t[j]-t[e])/1e9
                if j < x:
                    prefix_low = min(prefix_low, net)
                    if net <= -risks[a]:
                        risk_before = 1.
                    dt = (t[j+1]-t[j])/1e9
                    ap += max(net, 0.)*dt
                    am += max(-net, 0.)*dt
                    water += dt if net < 0 else 0.
                previous = net
            samples[a, side, 16] = net
            out[a, side, :] = np.array([net, low, high, dd, ap, am, water,
                first_positive, variation, prefix_low, first_risk, risk_before])
    return out, samples


def haar_details(samples):
    """Signed orthonormal Haar details of 16 future interval-end samples.

Coarse-to-fine (1+2+4+8 coefficients). Event-relative resolution, NOT Hz or
physical market periods; keep elapsed seconds separately. No estimated STFT
on paths too short to support a frequency estimate.
"""
    a = np.asarray(samples, float)[..., 1:]
    levels = []
    while a.shape[-1] > 1:
        levels.append((a[..., ::2]-a[..., 1::2])/np.sqrt(2))
        a = (a[..., ::2]+a[..., 1::2])/np.sqrt(2)
    return np.concatenate(levels[::-1], axis=-1)


def cyclic_clock(t_ns, period_seconds):
    """UTC phase; real session schedules/DST must be supplied separately."""
    if period_seconds <= 0:
        raise ValueError('positive physical period required')
    phase = 2*np.pi*(np.asarray(t_ns, np.int64) % int(period_seconds*1e9))/(period_seconds*1e9)
    return np.column_stack((np.sin(phase), np.cos(phase)))


def build_path_targets(q, line_map, fee=.3, horizon_seconds=300., time_unit_seconds=60.):
    if not np.isfinite([fee, horizon_seconds, time_unit_seconds]).all() or fee < 0 or horizon_seconds <= 0 or time_unit_seconds <= 0:
        raise ValueError('invalid units or fee')
    m = line_map.copy().reset_index(drop=True)
    e = m.entry_ordinal.to_numpy(np.int64)
    x = m.cross_ordinal.to_numpy(np.int64)
    risk = m.R_event.to_numpy(float)
    eligible = m.ready.to_numpy(bool) & (e >= 0) & np.isfinite(risk) & (risk > 0)
    due = m.entry_ns.to_numpy(np.int64)+int(horizon_seconds*1e9)
    hit = eligible & (x >= 0) & (m.cross_ns.to_numpy() <= due)
    known = eligible & (hit | (m.support_end_ns.to_numpy() > due))
    m['hit_known'] = known
    m['hit'] = np.where(known, hit.astype(np.int8), -1)
    m['profile_known'] = hit
    m['target_available_ns'] = np.where(hit, m.cross_ns, np.where(known, due, 0))
    m['followup_seconds'] = np.where(eligible,
        np.maximum(0., (np.minimum(m.support_end_ns.to_numpy(), due)-m.entry_ns.to_numpy())/1e9), np.nan)
    m['censored'] = eligible & ~known
    # Existing map can contain a crossing after H: do not expose it as a V6 label.
    m['v6_event_id'] = np.where(hit, m.event_id, '')
    m['elapsed_seconds'] = np.where(hit, (m.cross_ns-m.entry_ns)/1e9, np.nan)
    m['elapsed_T'] = m.elapsed_seconds/time_unit_seconds
    m['velocity_known'] = hit & (m.elapsed_seconds.to_numpy() > 0)
    m['shape_known'] = m.velocity_known
    # Multiple lines at one publication form one packet; timestamps alone
    # cannot order different publications in the same nanosecond.
    if 't_ns' in m:
        packets = m.loc[hit, ['t_ns', 'cross_ordinal', 'cross_ns']].drop_duplicates().sort_values(['t_ns', 'cross_ordinal'])
        packets['packet_rank'] = packets.groupby('t_ns').cumcount()+1
        packets['since_previous_packet_s'] = packets.groupby('t_ns').cross_ns.diff()/1e9
        m = m.merge(packets, on=['t_ns', 'cross_ordinal', 'cross_ns'], how='left', validate='many_to_one')
    idx = np.flatnonzero(hit)
    if (x[idx] <= e[idx]).any() or (q.quote_t[e[idx]] != m.entry_ns.to_numpy()[idx]).any() or (q.quote_t[x[idx]] != m.cross_ns.to_numpy()[idx]).any():
        raise ValueError('entry/cross ordinal and timestamp mismatch')
    if (m.elapsed_seconds.to_numpy()[idx] < 0).any():
        raise ValueError('negative elapsed time')
    if hasattr(q, 'valid') and hasattr(q, 'epoch'):
        bad = (~q.valid) | (q.epoch < 0) | (q.bid <= 0) | (q.ask <= q.bid)
        invalid = np.r_[0, np.cumsum(bad, dtype=np.int64)]
        breaks = np.r_[0, np.cumsum((np.diff(q.quote_t) > 60*10**9) | (np.diff(q.quote_t) < 0) |
                                    (q.epoch[1:] != q.epoch[:-1]), dtype=np.int64)]
        if ((invalid[x[idx]+1]-invalid[e[idx]])>0).any() or ((breaks[x[idx]]-breaks[e[idx]])>0).any():
            raise ValueError('unsupported quote path')
    values, samples = scan_paths(q.quote_t, q.bid, q.ask, e[idx], x[idx], risk[idx], fee)
    columns = {}
    def put(name, a):
        full = np.full(len(m), np.nan)
        full[idx] = a
        columns[name] = full
    dt = m.elapsed_seconds.to_numpy()[idx]
    positive_time = dt > 0
    def divide_time(a):
        return np.divide(a, dt, out=np.full_like(a, np.nan), where=positive_time)
    for si, side in enumerate(('L', 'S')):
        for j, name in enumerate(METRICS):
            put(f'{name}_{side}', values[:, si, j])
        v = {name: values[:, si, j] for j, name in enumerate(METRICS)}
        for name in ('net', 'mae', 'mfe', 'max_drawdown', 'mae_before_cross', 'variation'):
            put(f'{name}_R_{side}', v[name]/risk[idx])
        put(f'velocity_R_per_T_{side}', divide_time(v['net']/risk[idx])*time_unit_seconds)
        for polarity in ('positive', 'negative'):
            area = v[f'area_{polarity}']
            put(f'area_{polarity}_RT_{side}', area/(risk[idx]*time_unit_seconds))
            put(f'mean_{polarity}_R_{side}', divide_time(area)/risk[idx])
        put(f'underwater_fraction_{side}', divide_time(v['underwater_seconds']))
        change = samples[:, si, -1]-samples[:, si, 0]
        put(f'efficiency_{side}', np.divide(abs(change), v['variation'], out=np.zeros(len(idx)), where=v['variation']>0))
        norm_samples = samples[:, si]/risk[idx, None]
        for j in range(17):
            put(f'path_q{j:02d}_R_{side}', norm_samples[:, j])
        for j in range(4):
            increment = norm_samples[:, (j+1)*4]-norm_samples[:, j*4]
            put(f'quarter_velocity_{j}_R_per_T_{side}', divide_time(increment)*4*time_unit_seconds)
        details = haar_details(norm_samples)
        details[~positive_time] = np.nan
        for j in range(15):
            put(f'haar_{j:02d}_{side}', details[:, j])
        # Reconstruction/shape diagnostics only: intentionally redundant with samples.
        for start, end in ((0, 1), (1, 3), (3, 7), (7, 15)):
            put(f'haar_energy_{end-start}_{side}', (details[:, start:end]**2).sum(axis=1))
    return pd.concat((m, pd.DataFrame(columns)), axis=1)


def learning_view(frame):
    """Explicit allow-list: hit BCE + conditional factual regression targets.

Not a trained loss or a scalar utility. A future fit should compare each head
with controls, tune head scales on TRAIN, and retain the frozen V5 baseline.
Absent/censored crosses never become zero net outcomes. Negative outcomes
are just as valid as positive ones. No 9-way mutually exclusive softmax.
"""
    names = [f'{metric}_{side}' for side in ('L', 'S') for metric in
        ('net_R', 'mae_R', 'max_drawdown_R', 'mean_positive_R', 'mean_negative_R', 'underwater_fraction')]
    names += ['elapsed_T']
    targets = frame[names].to_numpy(float)
    masks = frame.profile_known.to_numpy()[:, None] & np.isfinite(targets)
    return dict(hit=frame.hit.to_numpy(), hit_mask=frame.hit_known.to_numpy(),
                profile=targets, profile_mask=masks, profile_names=names)
