"""Materialize V6 on immutable V5 inputs; fail closed on source/hash drift."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition import bb_crossing as bb
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.research_v6.targets import build_path_targets, cyclic_clock, METRICS, VERSION


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def geometry(q, bank, target):
    k = bb.reduce_quotes(q)
    t = q.quote_t[k]
    price = (q.bid[k].astype(float)+q.ask[k])/2e9
    blocks = []
    for scale in (1, 3, 6):
        middle, half, support = bb.native_geometry(t, price, q.valid[k], q.epoch[k], scale)
        row, role, direction, cause, dp, dl = bb.crossing_rows(t, price, middle, half, support)
        b = pd.DataFrame(dict(ordinal=k[row], role=role, direction=direction,
            cause=[bb.CAUSES[c] for c in cause], price_step=dp, line_step=dl))
        old = bank.loc[bank.scale == scale].sort_values(['ordinal', 'role']).reset_index(drop=True)
        b = b.sort_values(['ordinal', 'role']).reset_index(drop=True)
        for name in ('ordinal', 'role', 'direction', 'cause'):
            if not np.array_equal(b[name].to_numpy(), old[name].to_numpy()):
                raise ValueError('frozen geometry mismatch '+name)
        b['event_id'] = old.event_id
        blocks.append(b[['event_id', 'cause', 'price_step', 'line_step']])
        ix = np.flatnonzero((target.scale.to_numpy() == scale) & target.profile_known.to_numpy())
        e = target.entry_ordinal.to_numpy()[ix]
        x = target.cross_ordinal.to_numpy()[ix]
        ei, xi = np.searchsorted(k, e, side='right')-1, np.searchsorted(k, x)
        r = target.role.to_numpy()[ix]-1
        # Entry-to-cross geometry, distinct from the last-strict-side step.
        delta_p = (q.bid[x].astype(float)+q.ask[x]-q.bid[e]-q.ask[e])/2e9
        delta_b = middle[xi]+r*half[xi]-middle[ei]-r*half[ei]
        target.loc[ix, 'price_move_R'] = delta_p/target.R_event.to_numpy()[ix]
        target.loc[ix, 'band_move_R'] = delta_b/target.R_event.to_numpy()[ix]
        target.loc[ix, 'relative_move_R'] = (delta_p-delta_b)/target.R_event.to_numpy()[ix]
    return target.merge(pd.concat(blocks), left_on='v6_event_id', right_on='event_id',
                        how='left', validate='many_to_one', suffixes=('', '_geometry'))


def verify(q, target, fee):
    hit = target.profile_known.to_numpy()
    ix = np.flatnonzero(hit)
    # Full-population agreement with separately implemented frozen extrema.
    errors = {}
    for side in ('L', 'S'):
        for new, old in (('net', 'mark'), ('mae', 'mae'), ('mfe', 'mfe')):
            # mae/mfe names overlap old columns: original copy retained below.
            old_name = f'v5_{old}_{side}'
            err = float(np.max(abs(target.loc[hit, f'{new}_{side}']-target.loc[hit, old_name])))
            if err > 1e-8:
                raise ValueError('V5 profile disagreement')
            errors[f'{new}_{side}'] = err
    probe = np.unique(np.r_[ix[np.linspace(0, len(ix)-1, 12, dtype=int)],
        target.loc[hit].nsmallest(3, 'elapsed_seconds').index,
        target.loc[hit].nlargest(3, 'elapsed_seconds').index])
    maximum = 0.
    for i in probe:
        row = target.iloc[i]
        e, x = int(row.entry_ordinal), int(row.cross_ordinal)
        tt = q.quote_t[e:x+1]
        if not q.valid[e:x+1].all() or not (q.epoch[e:x+1] == q.epoch[e]).all() or (np.diff(tt)>60e9).any():
            raise ValueError('unsupported profile')
        dt = np.diff(tt)/1e9
        for side in ('L', 'S'):
            net = ((q.bid[e:x+1]-q.ask[e]) if side == 'L' else (q.bid[e]-q.ask[e:x+1]))/1e9-fee
            plus = float(np.dot(np.maximum(net[:-1], 0), dt))
            minus = float(np.dot(np.maximum(-net[:-1], 0), dt))
            water = float(dt[net[:-1]<0].sum())
            peak = np.maximum.accumulate(np.maximum(net, 0))
            expected = dict(area_positive=plus, area_negative=minus, underwater_seconds=water,
                max_drawdown=float((peak-net).max()), variation=float(abs(np.diff(net)).sum()),
                mae_before_cross=float(net[:-1].min()), risk_before_cross=float((net[:-1]<=-row.R_event).any()))
            pos = np.flatnonzero(net>0)
            bad = np.flatnonzero(net<=-row.R_event)
            expected['first_positive_seconds'] = (tt[pos[0]]-tt[0])/1e9 if len(pos) else np.nan
            expected['first_risk_seconds'] = (tt[bad[0]]-tt[0])/1e9 if len(bad) else np.nan
            for name, value in expected.items():
                actual = row[f'{name}_{side}']
                if not np.isclose(actual, value, atol=1e-7, rtol=1e-10, equal_nan=True):
                    raise ValueError(f'independent probe {i} {name}: {actual} != {value}')
                if np.isfinite(value):
                    maximum = max(maximum, abs(actual-value))
            # Integer-time right-continuous interior sampling, bounded by x.
            for j in range(17):
                at = 0 if j == 0 else (len(tt)-1 if j == 16 else np.searchsorted(tt, tt[0]+((tt[-1]-tt[0])*j)//16, side='right')-1)
                if abs(row[f'path_q{j:02d}_R_{side}']-net[at]/row.R_event)>1e-8:
                    raise ValueError('sample probe mismatch')
    spread_sum = ((q.ask[target.cross_ordinal.to_numpy()[ix]]-q.bid[target.cross_ordinal.to_numpy()[ix]])+
                  (q.ask[target.entry_ordinal.to_numpy()[ix]]-q.bid[target.entry_ordinal.to_numpy()[ix]]))/1e9
    side_error = float(np.max(abs(target.loc[hit, 'net_L'].to_numpy()+target.loc[hit, 'net_S'].to_numpy()+spread_sum+2*fee)))
    if side_error > 1e-8:
        raise ValueError('long/short cost identity')
    return dict(full_population_extrema_errors=errors, independent_paths=len(probe)*2,
                max_probe_error=maximum, long_short_identity_error=side_error)


def build_day(date, c, base, out):
    started = time.perf_counter()
    dataset = Path(base['data']['root'])
    receipt_path = dataset/f'{date}.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'COMPLETE':
        raise ValueError('incomplete input')
    hashes = {str(receipt_path): sha(receipt_path)}
    for suffix in ('roots', 'map', 'bank', 'scene'):
        path = dataset/f'{date}_{suffix}.parquet'
        actual = sha(path)
        if actual != receipt['files'][path.name]:
            raise ValueError('input hash mismatch '+str(path))
        hashes[str(path)] = actual
    day = MonthDay(Path(base['data']['market_root'])/date[:7]/date, verify=True)
    # Verify actual quote bytes, not just manifest claims.
    for part in day.meta['quote_parts']:
        path = Path(part['root'])/'quotes.parquet'
        actual = sha(path)
        if actual != part['quote_file']['sha256']:
            raise ValueError('raw quote hash mismatch')
        hashes[str(path)] = actual
    q = QuoteTape.from_day(day)
    line_map = pd.read_parquet(dataset/f'{date}_map.parquet')
    for side in ('L', 'S'):
        for name in ('mark', 'mae', 'mfe'):
            line_map[f'v5_{name}_{side}'] = line_map[f'{name}_{side}']
    # Avoid duplicate columns when replacing the old extrema by new exact scans.
    line_map = line_map.drop(columns=['mae_L', 'mae_S', 'mfe_L', 'mfe_S'])
    target = build_path_targets(q, line_map, c['fee'], c['horizon_seconds'], c['time_unit_seconds'])
    target = geometry(q, pd.read_parquet(dataset/f'{date}_bank.parquet'), target)
    checks = verify(q, target, c['fee'])
    root = pd.read_parquet(dataset/f'{date}_roots.parquet')
    target = target.merge(root[['t_ns', 'y_crossing']], on='t_ns', validate='many_to_one')
    split = next(name for name, dates in base['splits'].items() if date in dates)
    target['split'] = split
    target.to_parquet(out/f'{date}_target.parquet', index=False)
    clock = root[['date', 't_ns', 'R_event', 'decision_spread']].copy()
    # Clock phase is known at the decision clock, even if last BBO is older.
    clock['available_ns'] = clock.t_ns
    for seconds in (60, 180, 360, 86400):
        clock[[f'utc_sin_{seconds}', f'utc_cos_{seconds}']] = cyclic_clock(clock.t_ns, seconds)
    clock.to_parquet(out/f'{date}_causal_clock.parquet', index=False)
    result = dict(status='COMPLETE', version=VERSION, date=date, split=split,
        roots=len(root), cells=len(target), hit=int(target.profile_known.sum()),
        no_hit=int((target.hit == 0).sum()), censored=int(target.censored.sum()),
        not_ready=int((~target.ready).sum()), zero_time_hits=int((target.profile_known & ~target.velocity_known).sum()),
        checks=checks, inputs=hashes, outputs={p.name: sha(p) for p in
            (out/f'{date}_target.parquet', out/f'{date}_causal_clock.parquet')},
        seconds=time.perf_counter()-started)
    save(out/f'{date}.json', result)
    QUOTE_CACHE.clear()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/research/crossing_path_target_v6.json')
    parser.add_argument('--date', action='append')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    base = json.loads(Path(c['base_config']).read_text())
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=True)
    identity = {p: sha(p) for p in (args.config, c['base_config'],
        'spectrum/research_v6/targets.py', __file__, 'spectrum/cognition/bb_crossing.py',
        str(Path(base['data']['root'])/'manifest.json'))}
    dates = args.date or base['data']['dates']
    manifest = dict(status='RUNNING', identity=identity, config=c, dates=dates, days=[])
    save(out/'manifest.json', manifest)
    for date in dates:
        result = build_day(date, c, base, out)
        manifest['days'].append(result)
        save(out/'manifest.json', manifest)
        print(json.dumps({k: result[k] for k in ('date', 'cells', 'hit', 'censored', 'seconds')}), flush=True)
        gc.collect()
    for path, digest in identity.items():
        if sha(path) != digest:
            raise ValueError('source changed during build')
    manifest['status'] = 'COMPLETE'
    save(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
