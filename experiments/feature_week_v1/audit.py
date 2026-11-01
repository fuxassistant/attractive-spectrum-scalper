"""Independent arithmetic, availability, clock and partition audits on real data."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile import ROOT, OUTPUT, DATES, WEEK, raw_for, source_for, FAMILIES, NS, MINUTE
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.io import save_json, sha
from spectrum.market.calendar import MarketCalendar
from spectrum.canon import load_canon
from spectrum.features.pipeline import compute
from spectrum.features.stream import FeatureStream


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {} if not len(values) else {k: float(v) for k, v in zip(
        ('p50', 'p90', 'p95', 'p99', 'max'), np.quantile(values, [.5, .9, .95, .99, 1]))}


def parity(first, second):
    items = []
    for path in sorted(first.rglob('*.parquet')):
        other = second / path.relative_to(first)
        same_bytes = sha(path) == sha(other)
        equal = same_bytes or pq.read_table(path).equals(pq.read_table(other))
        items.append(dict(path=path.relative_to(first).as_posix(), bytes_equal=same_bytes, values_equal=equal))
        if not equal:
            raise ValueError(f'Parallel partition result differs: {path}')
    return dict(status='PASS', files=len(items), byte_identical=sum(x['bytes_equal'] for x in items), details=items)


def audit_day(run, date):
    base = run / 'days' / date
    f = pd.read_parquet(base / 'input_seconds.parquet')
    x = pd.read_parquet(base / 'features.parquet')
    fast = pd.read_parquet(base / 'fast.parquet')
    names = [c for c in x if c+'__known' in x]
    if len(names) != 409:
        raise ValueError('Unexpected feature profile')
    if not f.t_ns.is_monotonic_increasing or f.t_ns.duplicated().any() or (f.available_ns < f.t_ns).any():
        raise ValueError('Clock integrity failed')
    cadence = []
    adjacent = f.t_ns.diff().eq(NS).to_numpy() & f.epoch.eq(f.epoch.shift()).to_numpy()
    for name in names:
        a, known = x[name].to_numpy(), x[name+'__known'].to_numpy(bool)
        if not np.isfinite(a).all() or np.any(a[~known] != 0):
            raise ValueError('Unknown is not finite zero: '+name)
        paired = adjacent & known & np.r_[False, known[:-1]]
        changed = paired & ~np.isclose(a, np.r_[a[0],a[:-1]], rtol=1e-10, atol=1e-9)
        cadence.append(dict(name=name, rows=len(a), known=int(known.sum()), pairs=int(paired.sum()),
                            changes=int(changed.sum()), distinct=int(pd.Series(a[known]).nunique())))
    errors = {}
    def check(name, reference, atol=1e-7):
        mask = x[name+'__known'].to_numpy(bool)
        err = np.abs(x[name].to_numpy()[mask] - np.asarray(reference)[mask])
        maximum = float(err.max()) if len(err) else 0.
        errors[name] = dict(samples=int(mask.sum()), max_absolute_error=maximum)
        if maximum > atol:
            raise ValueError(f'Independent arithmetic: {name}: {maximum}')
    check('spread_ticks', (f.ask-f.bid)/.1)
    check('depth_balance_12', (f.depth_bid_12-f.depth_ask_12)/(f.depth_bid_12+f.depth_ask_12))
    for n in (1, 5, 15, 30, 60, 128):
        check(f'displacement_{n}s', (f.mid-f.mid.shift(n))/.1)
        check(f'path_{n}s', f.c_path_ticks-f.c_path_ticks.shift(n), atol=1e-5)
    for side in ('buy', 'sell', 'unknown'):
        check(f'trade_{side}_30s', f['c_trade_'+side]-f['c_trade_'+side].shift(30))
    trades = pd.read_parquet(raw_for(date) / 'trades.parquet')
    ti = trades.t_ns.to_numpy()
    price = trades.price_nano.to_numpy()/NS
    qty = trades.quantity.to_numpy(float)
    exact = []
    rng = np.random.default_rng(20261029)
    for n in (60, 300, 900, 3600):
        stem = f'vwap_exact_{n}s_'
        candidates = np.flatnonzero(x[stem+'center__known'])
        for i in rng.choice(candidates, min(25,len(candidates)), replace=False):
            t = int(f.t_ns.iloc[i])
            lo, hi = np.searchsorted(ti, [t-n*NS, t], side='left')
            p, q = price[lo:hi], qty[lo:hi]
            mean = np.average(p, weights=q)
            sigma = np.sqrt(np.average((p-mean)**2, weights=q))/.1
            em, es = abs(mean-x[stem+'center'].iloc[i]), abs(sigma-x[stem+'sigma_ticks'].iloc[i])
            if em > 1e-7 or es > 1e-5:
                raise ValueError(f'Exact T VWAP mismatch {date} {n} {em} {es}')
            exact.append(dict(scale=n, center_error=float(em), sigma_ticks_error=float(es)))
    # Direct windowed DFT, independent single-window calculation.
    fft_errors = []
    for n in (32, 64, 128):
        name = f'fft_mid_change_{n}s_power'
        candidates = np.flatnonzero(x[name+'__known'])
        for i in rng.choice(candidates, min(25,len(candidates)), replace=False):
            increments = np.diff(f.mid.to_numpy()[i-n:i+1])/.1
            window = .5-.5*np.cos(2*np.pi*np.arange(n)/n)
            transform = np.fft.rfft((increments-increments.mean())*window)
            power = abs(transform)**2/(window@window)
            power[1:-1] *= 2
            error = abs(float(power[1:].sum()/n)-x[name].iloc[i])
            if error > 1e-7:
                raise ValueError('DFT mismatch')
            fft_errors.append(float(error))
    market = Path(source_for(date)['market'])
    quotes = pd.read_parquet(market/'quotes.parquet', columns=['t_ns','available_ns','epoch','bid_nano','ask_nano','valid'])
    quote_times = quotes.t_ns.to_numpy()
    indices = np.searchsorted(quote_times, f.t_ns.to_numpy(), side='left')-1
    safe = indices >= 0
    ix = np.maximum(indices, 0)
    safe &= f.valid.to_numpy() & quotes.valid.to_numpy()[ix] & (f.epoch.to_numpy()==quotes.epoch.to_numpy()[ix])
    b, a = quotes.bid_nano.to_numpy()/NS, quotes.ask_nano.to_numpy()/NS
    qerr = max(float(np.max(np.abs(f.bid.to_numpy()[safe]-b[ix[safe]]))),
               float(np.max(np.abs(f.ask.to_numpy()[safe]-a[ix[safe]]))))
    if qerr > 1e-8:
        raise ValueError('Seconds are not a strictly earlier complete BBO')
    # Interval membership is identical to the canon schedule without formatting
    # millions of duplicate calendar dates into strings.
    intervals = load_canon('canon_v1').venues['gc_globex'].intervals(int(quote_times[0]),int(quote_times[-1])+1)
    opened = np.zeros(len(quotes),bool)
    for lo,hi in intervals:
        opened |= (quote_times>=lo)&(quote_times<hi)
    sample = np.linspace(0,len(quotes)-1,min(2000,len(quotes)),dtype=int)
    if not np.array_equal(opened[sample],~MarketCalendar().closed(quote_times[sample])):
        raise ValueError('Schedule interval/point disagreement')
    valid = quotes.valid.to_numpy() & opened
    changed = valid & np.r_[False,valid[:-1]] & ((b != np.r_[b[0],b[:-1]]) | (a != np.r_[a[0],a[:-1]]))
    qdt = np.diff(quote_times[valid])/NS
    pdt = np.diff(quote_times[changed])/NS
    unchanged_mid = fast.ret1_ticks.eq(0) & fast.ledger_known
    additional = unchanged_mid & ((fast.trade_quantity>0) | (fast.bid_near_add>0) |
                                  (fast.ask_near_add>0) | (fast.bid_near_remove>0) | (fast.ask_near_remove>0))
    cadence_frame = pd.DataFrame(cadence)
    cadence_frame['date'] = date
    known_matrix=x[[n+'__known' for n in names]].to_numpy(bool)
    proxy_difference={}
    for n in (60,300,900,3600):
        a,b=f'vwap_exact_{n}s_center',f'vwap_proxy_{n}s_center'
        good=x[a+'__known']&x[b+'__known']
        proxy_difference[str(n)]=quantiles(abs(x.loc[good,a]-x.loc[good,b])/.1)
    result = dict(date=date, rows=len(f), known_price=int(fast.known.sum()), ledger_known=int(fast.ledger_known.sum()),
        admitted=int(x.admissible.sum()), late_more_1s=int(((f.available_ns-f.t_ns)>NS).sum()),
        latency_ms=quantiles((f.available_ns-f.t_ns)/1e6), formula_checks=errors,
        open_latency_ms=quantiles(((f.available_ns-f.t_ns)/1e6)[fast.known]),
        exact_checks=len(exact), exact_center_max_error=max(e['center_error'] for e in exact),
        exact_sigma_ticks_max_error=max(e['sigma_ticks_error'] for e in exact),
        fft_checks=len(fft_errors), fft_max_error=max(fft_errors),
        bbo_asof_checks=int(safe.sum()), bbo_max_error=qerr,
        complete_publications=int(valid.sum()), bbo_changes=int(changed.sum()),
        publication_interval_seconds=quantiles(qdt[qdt<60]), bbo_change_interval_seconds=quantiles(pdt[pdt<60]),
        quote_interval_ge60s=int((qdt>=60).sum()),
        unchanged_mid_seconds=int(unchanged_mid.sum()), unchanged_mid_with_flow=int(additional.sum()),
        all_features_known=int(known_matrix.all(axis=1).sum()),
        any_feature_known=int(known_matrix.any(axis=1).sum()),
        known_fraction_per_row=quantiles(known_matrix.mean(axis=1)),
        exact_proxy_center_difference_ticks=proxy_difference,
        exact_trade_rows=len(trades), exact_trade_quantity=int(trades.quantity.sum()))
    return result, cadence_frame


def partition_audit(run):
    date = WEEK[0]
    f = pd.read_parquet(run/'days'/date/'input_seconds.parquet')
    stored = pd.read_parquet(run/'days'/date/'features.parquet')
    # Three hours in a supported part of the day; arbitrary split is deliberately
    # unrelated to a source reset or a venue break.
    start = 8*3600
    frame = f.iloc[start:start+3*3600].reset_index(drop=True)
    reference = compute(frame, FAMILIES)
    stream = FeatureStream(FAMILIES, source_id='week-audit')
    max_error, mask_mismatches = 0., 0
    for lo, hi in ((0,4500), (4500,7500), (7500,len(frame))):
        got = stream.feed(frame.iloc[lo:hi])
        for name in reference.values:
            same = np.array_equal(got['known'][name], reference.masks[name][lo:hi])
            mask_mismatches += 0 if same else 1
            a, b = got['values'][name], reference.values[name][lo:hi]
            error = float(np.max(np.abs(a-b)))
            max_error = max(max_error,error)
            if not np.allclose(a,b,rtol=1e-6,atol=2e-5):
                raise ValueError(f'Chunk/restart value drift {name}: {error}')
        stream = FeatureStream.restore(stream.checkpoint(), expected_source_id='week-audit')
    if mask_mismatches:
        raise ValueError('Chunk/restart mask drift')
    fresh = compute(frame.iloc[4500:].reset_index(drop=True), FAMILIES)
    lost = {n:int((reference.masks[n][4500:] & ~fresh.masks[n]).sum()) for n in reference.values}
    cut = 7200
    prefix = compute(frame.iloc[:cut], FAMILIES)
    prefix_equal = all(np.array_equal(prefix.masks[n],reference.masks[n][:cut]) and
                       np.allclose(prefix.values[n],reference.values[n][:cut],rtol=1e-10,atol=1e-9)
                       for n in reference.values)
    if not prefix_equal:
        raise ValueError('Appending future rows changed the prefix')
    return dict(status='PASS', rows=len(frame), channels=len(reference.values),
                chunk_restart_max_error=max_error, chunk_restart_masks_equal=True,
                future_prefix_equal=prefix_equal, split_without_state_lost_known_rows=lost)


def minute_audit(run):
    m = pd.read_parquet(run/'minutes.parquet')
    errors = []
    for scale in (1,3,6):
        for i in np.flatnonzero(m['bb'+str(scale)].notna())[::97]:
            row = m.iloc[i]
            native = m[(m.t_ns<=row.t_ns)&(m.t_ns%(scale*MINUTE)==0)&(m.segment==row.segment)].tail(20)
            if len(native)!=20:
                raise ValueError('BB history too short')
            mean, half = native.close.mean(),2*np.std(native.close.to_numpy(),ddof=0)
            error = max(abs(mean-row['bb'+str(scale)]),abs(half-row['bb'+str(scale)+'_half']))
            if error>1e-7:
                raise ValueError('Native BB mismatch')
            errors.append(float(error))
    core = pd.read_parquet(run/'core_m1.parquet')
    bars = pd.concat([pd.read_parquet(Path(source_for(d)['market'])/'bars.parquet') for d in DATES],ignore_index=True)
    core_errors = []
    for scale in (1,3,6,9):
        name=f'bb_m{scale}_width_scale'
        candidates=core[core[name+'_known'].eq(1) & core.t_ns.mod(scale*MINUTE).eq(0)]
        for row in candidates.iloc[::31].itertuples():
            history=bars[(bars.t_ns<=row.t_ns)&(bars.epoch==row.epoch)].tail(20*scale)
            if len(history)!=20*scale or not history.valid.all() or not history.t_ns.diff().iloc[1:].eq(MINUTE).all():
                raise ValueError('Core BB lacks contiguous completed native bars')
            closes=history.c.to_numpy()[scale-1::scale]
            half=2*np.std(closes,ddof=1)
            error=abs(half-getattr(row,name)*row.scale)
            if not np.isclose(half,getattr(row,name)*row.scale,rtol=1e-7,atol=1e-8):
                raise ValueError('Core sample-SD BB mismatch')
            core_errors.append(float(error))
    previous_week = [c for c in core if c.startswith('previous_week_')]
    if np.any(core[previous_week].to_numpy()!=0):
        raise ValueError('Incomplete prior week falsely admitted')
    return dict(bb_checks=len(errors), bb_max_error=max(errors),
                core_bb_checks=len(core_errors),core_bb_max_error=max(core_errors),
                geometry_versions=dict(representation='SMA20 +/- 2 population SD, ddof=0; native M1/M3/M6',
                                       core='SMA20 +/- 2 sample SD, ddof=1; native M1/M3/M6/M9',
                                       cross_schema_same_BB=False),
                previous_week_partial_masked=previous_week,
                minutes=len(m), known_minutes=int(m.known.sum()))


def coverage(run):
    f = pd.concat([pd.read_parquet(run/'days'/d/'fast.parquet',
         columns=['t_ns','available_ns','known','ledger_known','scheduled_open','epoch','mid']) for d in DATES])
    begin = int(pd.Timestamp('2026-08-23 22:00',tz='UTC').value)
    end = int(pd.Timestamp('2026-08-28 21:00',tz='UTC').value)
    grid = np.arange(begin+NS,end+NS,NS,dtype=np.int64)
    expected = ~MarketCalendar().closed(grid-1)
    indexed = f.set_index('t_ns').reindex(grid)
    present = indexed.epoch.notna().to_numpy()
    observed = indexed.known.eq(True).to_numpy()
    missing = grid[expected & ~present]
    return dict(start_ns=begin,end_ns=end,expected_open_seconds=int(expected.sum()),
                present_open_seconds=int((expected&present).sum()),known_open_seconds=int((expected&observed).sum()),
                missing_open_seconds=len(missing), missing_endpoints_utc=[str(pd.Timestamp(t,tz='UTC')) for t in missing[:30]],
                scheduled_closed_seconds=int((~expected).sum()),
                present_but_unknown_open=int((expected&present&~observed).sum()))


def main(first_name, second_name):
    start = time.perf_counter()
    first, second = OUTPUT/first_name, OUTPUT/second_name
    out = OUTPUT/'audit'
    out.mkdir(exist_ok=True)
    if (out/'audit.json').exists():
        raise ValueError('Completed audit is immutable; use a separate output')
    results, cadence = [], []
    for date in WEEK:
        r, c = audit_day(first,date)
        results.append(r); cadence.append(c)
        print(json.dumps(dict(audited=date, formula_checks=len(r['formula_checks']), exact_checks=r['exact_checks'])),flush=True)
    pd.concat(cadence).to_parquet(out/'cadence.parquet',index=False)
    result = dict(status='PASS', day_audits=results, partition=partition_audit(first),
                  minute=minute_audit(first), coverage=coverage(first), parity=parity(first,second),
                  elapsed_seconds=time.perf_counter()-start)
    save_json(out/'audit.json',result)
    print(json.dumps(dict(status='PASS',seconds=result['elapsed_seconds'],coverage=result['coverage'],
                          parity={k:v for k,v in result['parity'].items() if k!='details'})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',default='serial_v2');p.add_argument('--second',default='parallel2');a=p.parse_args()
    main(a.first,a.second)
