"""Materialize a bounded, source-bound multi-scale observation dataset."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
import numpy as np
from scripts.build_crossing_path_target_v6 import sha, save
from spectrum.market_repr.representation import make_fast, make_minutes, anchored_context, config, anchor, NS, MINUTE, FAST_VALUES, DAY_VALUES, MarketObservationStore, ledger_releases
from spectrum.market_repr.zones import zone_and_event_tables

ROOT = Path('data/derived/market_representation_v1')
REPORT = Path('reports/market_representation_v1')


def raw_data(date):
    root = ROOT / 'raw' / date
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE'
    for name, digest in manifest['outputs'].items(): assert sha(root / name) == digest
    for path, digest in manifest['implementation'].items(): assert sha(path) == digest
    full = json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    entry = next(r for r in full['captures'] if r['entry']['selection']['date'] == date and not r['entry']['selection']['context_only'])
    assert entry['status'] == 'COMPLETE'
    assert entry['entry']['sha256'] == manifest['source']['partition_sha256']
    sec_path = Path(entry['market']) / 'seconds.parquet'
    bar_path = Path(entry['market']) / 'bars.parquet'
    quote_path = Path(entry['market']) / 'quotes.parquet'
    assert sha(sec_path) == entry['market_payloads']['seconds.parquet']
    assert sha(bar_path) == entry['market_payloads']['bars.parquet']
    assert sha(quote_path) == entry['market_payloads']['quotes.parquet']
    seconds = pd.read_parquet(sec_path)
    quotes = pd.read_parquet(quote_path, columns=['t_ns', 'available_ns'])
    seconds['ledger_release_ns'] = ledger_releases(seconds, quotes)
    return (seconds, pd.read_parquet(root / 'levels_1s.parquet'),
            pd.read_parquet(root / 'trades.parquet'), manifest, sec_path, bar_path, quote_path)


def build(dates):
    cfg = config(); ROOT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    started = time.time(); out = ROOT / 'observations'; out.mkdir(exist_ok=True)
    save(ROOT / 'manifest.json', dict(status='RUNNING', dates=dates, started_unix=started))
    all_fast, all_trades, all_bars, sources, per_day = [], [], [], {}, []
    for date in dates:
        if date not in cfg['dates']: raise ValueError('Date outside reviewed calendar')
        seconds, levels, trades, receipt, sec_path, bar_path, quote_path = raw_data(date)
        f = make_fast(seconds, levels, trades)
        f['date'] = date
        f.to_parquet(out / (date + '_fast.parquet'), index=False)
        # Independent T total vs existing cumulative trade volume within stable epochs.
        canonical = seconds.groupby('epoch')[['c_trade_buy', 'c_trade_sell', 'c_trade_unknown']].max().sum().sum()
        per_day.append(dict(date=date, symbol=receipt['symbol'], seconds=len(f), supported=int(f.known.sum()),
            ledger_supported=int(f.ledger_known.sum()), trade_qty=int(trades.quantity.sum()),
            canonical_trade_qty=int(canonical), raw_stats=receipt['stats'],
            tail_trade_qty=int(trades.loc[trades.t_ns > seconds.t_ns.max(), 'quantity'].sum()),
            raw_minus_sampled_after_capture_tail=float(trades.loc[trades.t_ns <= seconds.t_ns.max(), 'quantity'].sum() - canonical),
            max_prefix_publication_delay_ns=int((f.available_ns - f.source_available_ns).max()),
            x_last_available_ns=int(f.available_ns.max())))
        sources[str(ROOT / 'raw' / date / 'manifest.json')] = sha(ROOT / 'raw' / date / 'manifest.json')
        sources[str(sec_path)] = sha(sec_path)
        sources[str(bar_path)] = sha(bar_path)
        sources[str(quote_path)] = sha(quote_path)
        all_bars.append(pd.read_parquet(bar_path))
        all_fast.append(f); all_trades.append(trades.assign(date=date))
        print(json.dumps(dict(stage='fast', date=date, rows=len(f))), flush=True)
        del levels
    fast = pd.concat(all_fast, ignore_index=True).sort_values('t_ns').reset_index(drop=True)
    trades = pd.concat(all_trades, ignore_index=True).sort_values(['t_ns', 'ordinal']).reset_index(drop=True)
    minutes = make_minutes(fast, trades, pd.concat(all_bars, ignore_index=True))
    minutes.to_parquet(out / 'minutes.parquet', index=False)
    inventory, outputs, contexts = [], {}, {}
    for date in dates:
        previous = cfg['previous_day'][date]
        if previous not in dates:
            inventory.append(dict(date=date, status='CONTEXT_ONLY_PREVIOUS_MISSING', required_previous=previous)); continue
        current_info = next(r for r in per_day if r['date'] == date)
        prev_info = next(r for r in per_day if r['date'] == previous)
        assert current_info['symbol'] == prev_info['symbol'], 'Unmapped contract roll'
        c = anchored_context(minutes, trades, date, previous)
        available_start = anchor(previous) + MINUTE
        assert c.t_ns.iloc[0] == available_start, 'Context missing its opening minute'
        lo = anchor(date); hi = lo + 17 * 3600 * NS
        local = fast[(fast.t_ns >= lo - 900 * NS) & (fast.t_ns < hi)].copy()
        ledger = pd.read_parquet(ROOT / 'raw' / date / 'levels_1s.parquet')
        zones, events = zone_and_event_tables(local, c, ledger, date, current_info['symbol'])
        c.to_parquet(out / (date + '_context.parquet'), index=False)
        zones.to_parquet(out / (date + '_zones.parquet'), index=False)
        events.to_parquet(out / (date + '_events.parquet'), index=False)
        contexts[date] = c
        store = MarketObservationStore(fast, c, zones, events, symbol=current_info['symbol'], previous_date=previous, date=date)
        # Receipts from the actual loader, not metadata column counts.
        checks = [store.observe(int(pd.Timestamp(date + ' ' + h, tz='UTC').value)).receipt for h in ('08:00', '12:00', '17:00')]
        actual = store.observe(int(pd.Timestamp(date + ' 12:00', tz='UTC').value))
        np.savez_compressed(out / (date + '_example_tensor.npz'), fast=actual.fast['values'], fast_known=actual.fast['known'],
             fast_t_ns=actual.fast['t_ns'], day=actual.day['values'], day_known=actual.day['known'], day_t_ns=actual.day['t_ns'],
             fast_normalized=actual.fast['normalized'], day_normalized=actual.day['normalized'])
        inventory.append(dict(date=date, status='MATERIALIZED', previous=previous, symbol=current_info['symbol'],
             context_rows=len(c), zones=zones.zone_id.nunique(), zone_versions=len(zones), events=len(events), loader_checks=checks))
        print(json.dumps(dict(stage='context', date=date, minute_rows=len(c), events=len(events))), flush=True)
    for path in sorted(out.iterdir()):
        if path.is_file(): outputs[str(path)] = sha(path)
    implementation_paths = ['spectrum/market_repr/representation.py', 'spectrum/market_repr/zones.py',
                             'scripts/build_market_representation_v1.py', 'configs/research/market_representation_v1.json']
    manifest = dict(version=cfg['version'], status='COMPLETE', dates=dates, elapsed_seconds=time.time() - started,
        config=cfg, sources=sources, outputs=outputs, implementation={p: sha(p) for p in implementation_paths},
        calendar=dict(scope=cfg['calendar_scope'], london='08:00 Europe/London', previous_day=cfg['previous_day'],
                      scheduled_break='16:00-17:00 America/Chicago', inferred_holidays=False),
        raw_audit=per_day, inventory=inventory, feature_names=dict(fast=FAST_VALUES, day=DAY_VALUES),
        readiness='MATERIALIZED_PENDING_CAUSAL_VISUAL_DISTINCTNESS_AUDIT',
        limitations=['Development dates already exposed; not untouched confirmation',
           'Observed interaction roles, no participant identities or intent',
           'Unlinked removal is a withdrawal proxy; unresolved F counted separately in raw receipts',
           'Zone remaining visible depth not materialized; explicitly unknown',
           'Offset zones are illustrative controls, not distance/exposure-matched causal controls',
           '1s local branch loses subsecond ordering; raw publication trade tape and per-price ledger retained',
           'No learned player, execution economics, or claim that a neural model uses this loader'])
    save(ROOT / 'manifest.json', manifest)
    save(REPORT / 'source_to_tensor_inventory.json', manifest)
    print(json.dumps(dict(status='COMPLETE', seconds=manifest['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dates', nargs='+'); a = p.parse_args()
    try:
        build(a.dates or config()['dates'])
    except Exception as exc:
        save(ROOT / 'manifest.json', dict(status='FAILED', error=repr(exc), dates=a.dates or config()['dates']))
        raise
