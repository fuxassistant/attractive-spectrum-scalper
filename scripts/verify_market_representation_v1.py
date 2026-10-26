"""Independent raw T parity, payload identities, causal prefixes and real loader."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from spectrum.source import DbnSource
from spectrum.market_repr.representation import (make_fast, make_minutes, anchored_context,
    MarketObservationStore, ledger_releases, anchor, NS)
from spectrum.market_repr.zones import zone_and_event_tables
from spectrum.market_repr.store import open_observation_store
from scripts.build_market_representation_v1 import ROOT, REPORT, raw_data
from scripts.build_crossing_path_target_v6 import sha, save


def raw_trade_parity(date):
    receipt = json.loads((ROOT / 'raw' / date / 'manifest.json').read_text())
    source = receipt['source']; route = receipt['route']
    reader = DbnSource(source['archive'], source['member']); offset = 0; parts = []
    for block in reader:
        ix = np.flatnonzero((block['publisher_id'] == route[0]) & (block['instrument_id'] == route[1]) &
                           (block['channel_id'] == route[2]) & (block['action'] == b'T'))
        ix = ix[(offset + ix >= source['first_ordinal']) & (offset + ix <= source['last_ordinal'])]
        if len(ix):
            b = block[ix]
            parts.append(pd.DataFrame(dict(ordinal=offset + ix, price_nano=b['price'], quantity=b['size'].astype(np.int64),
                side=np.where(b['side'] == b'B', 1, np.where(b['side'] == b'A', -1, 0)), raw_recv_ns=b['ts_recv'].astype(np.int64))))
        offset += len(block)
    assert reader.sha256 == source['member_sha256']
    reference = pd.concat(parts, ignore_index=True)
    actual = pd.read_parquet(ROOT / 'raw' / date / 'trades.parquet')
    columns = ['ordinal', 'price_nano', 'quantity', 'side']
    pd.testing.assert_frame_equal(reference[columns], actual[columns], check_dtype=False)
    assert np.all(actual.available_ns.to_numpy() >= reference.raw_recv_ns.to_numpy())
    return dict(date=date, rows=len(actual), contracts=int(actual.quantity.sum()),
                status='PASS_EXACT_ROW_PARITY_WITH_RAW_T', F_in_trade_volume=False,
                independent='Direct vectorized DBN T filter; does not call event_facts or book reconstruction')


def reconcile_sampled_volume(date):
    # Cumulative counters sampled at 1s can miss an epoch's final subsecond
    # tail before a reset. The new T tape must retain those genuine trades.
    sec, levels, tr, raw, sec_path, bar_path, quote_path = raw_data(date)
    del levels
    q = pd.read_parquet(quote_path, columns=['source_ordinal', 'epoch'])
    ix = np.searchsorted(q.source_ordinal.to_numpy(), tr.ordinal.to_numpy(), side='left')
    tr['canonical_epoch'] = q.epoch.to_numpy()[ix]
    last = sec.groupby('epoch').t_ns.max()
    cutoff = tr.canonical_epoch.map(last).fillna(-1).to_numpy(np.int64)
    beyond_last_epoch_sample = tr.t_ns.to_numpy() > cutoff
    expected = tr.loc[~beyond_last_epoch_sample].groupby('canonical_epoch').quantity.sum()
    actual = sec.groupby('epoch')[['c_trade_buy', 'c_trade_sell', 'c_trade_unknown']].max().sum(axis=1)
    index = expected.index.union(actual.index)
    np.testing.assert_array_equal(expected.reindex(index, fill_value=0), actual.reindex(index, fill_value=0))
    return dict(date=date, raw_contracts=int(tr.quantity.sum()), sampled_contracts=int(actual.sum()),
        recovered_epoch_tail_contracts=int(tr.loc[beyond_last_epoch_sample, 'quantity'].sum()),
        unexplained_contracts=0, status='PASS_PER_EPOCH_RECONCILIATION')


def compare_prefix(date, hour, all_fast, all_trades, all_bars):
    cut = int(pd.Timestamp(date + ' ' + hour, tz='UTC').value)
    sec, lev, tr, raw, sec_path, bar_path, quote_path = raw_data(date)
    q = pd.read_parquet(quote_path, columns=['t_ns', 'available_ns'])
    q = q[q.available_ns <= cut]
    sec = sec[sec.available_ns <= cut].copy()
    sec['ledger_release_ns'] = ledger_releases(sec, q)
    prefix = make_fast(sec, lev[lev.available_ns <= cut], tr[tr.available_ns <= cut])
    prefix = prefix[prefix.available_ns <= cut].reset_index(drop=True)
    full = all_fast[(all_fast.date == date) & (all_fast.available_ns <= cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, full[prefix.columns], check_dtype=True, check_exact=True)
    cfg = json.loads((ROOT / 'manifest.json').read_text())['config']; previous = cfg['previous_day'][date]
    m = make_minutes(all_fast[all_fast.available_ns <= cut], all_trades[all_trades.available_ns <= cut], all_bars[all_bars.available_ns <= cut])
    c = anchored_context(m, all_trades[all_trades.available_ns <= cut], date, previous)
    full_c = pd.read_parquet(ROOT / 'observations' / (date + '_context.parquet'))
    full_c = full_c[full_c.available_ns <= cut].reset_index(drop=True)
    c = c[c.available_ns <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(c, full_c, check_exact=True)
    z, e = zone_and_event_tables(all_fast[all_fast.available_ns <= cut], c, lev[lev.available_ns <= cut], date, raw['symbol'])
    full_z = pd.read_parquet(ROOT / 'observations' / (date + '_zones.parquet'))
    full_e = pd.read_parquet(ROOT / 'observations' / (date + '_events.parquet'))
    pd.testing.assert_frame_equal(z, full_z[full_z.available_ns <= cut].reset_index(drop=True), check_exact=True)
    pd.testing.assert_frame_equal(e, full_e[full_e.available_ns <= cut].reset_index(drop=True), check_exact=True)
    obs = MarketObservationStore(all_fast, full_c, z, e, symbol=raw['symbol'], previous_date=previous, date=date).observe(cut)
    persisted = open_observation_store(ROOT, date).observe(cut)
    assert obs.receipt['first_day_ns'] == anchor(previous) + 60 * NS
    assert not obs.receipt['day_truncated'] and not obs.receipt['future_outcomes_present']
    for tensor in (obs.fast, obs.day):
        assert np.isfinite(tensor['values']).all() and np.isfinite(tensor['normalized']).all()
        assert (tensor['available_ns'] <= cut).all()
    for branch in ('fast', 'day'):
        for field in ('values', 'normalized', 'known', 't_ns', 'available_ns'):
            np.testing.assert_array_equal(getattr(obs, branch)[field], getattr(persisted, branch)[field])
    return dict(date=date, cutoff_ns=cut, fast_rows=len(prefix), day_rows=len(c), zone_rows=len(z), event_rows=len(e),
                status='PASS_EXACT_APPEND_PREFIX', actual_tensor=obs.receipt)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text()); assert manifest['status'] == 'COMPLETE'
    checked = 0
    for group in ('sources', 'outputs', 'implementation'):
        for path, h in manifest[group].items(): assert sha(path) == h; checked += 1
    for r in manifest['raw_audit']:
        assert r['raw_stats']['max_stock_residual'] == 0
    direct = []
    for date in manifest['dates']:
        direct.append(raw_trade_parity(date)); print(json.dumps(direct[-1]), flush=True)
    sampled = []
    for date in manifest['dates']:
        sampled.append(reconcile_sampled_volume(date)); print(json.dumps(sampled[-1]), flush=True)
    fast = pd.concat([pd.read_parquet(ROOT / 'observations' / (d + '_fast.parquet')) for d in manifest['dates']], ignore_index=True)
    trades = pd.concat([pd.read_parquet(ROOT / 'raw' / d / 'trades.parquet') for d in manifest['dates']], ignore_index=True)
    bars = pd.concat([pd.read_parquet(p) for p in manifest['sources'] if p.endswith('bars.parquet')], ignore_index=True)
    prefix = []
    for date, hour in (('2026-08-25', '12:00:00'), ('2026-08-27', '14:12:30')):
        prefix.append(compare_prefix(date, hour, fast, trades, bars)); print(json.dumps(prefix[-1]), flush=True)
    minutes = pd.read_parquet(ROOT / 'observations' / 'minutes.parquet')
    expected_minutes = (minutes.t_ns.iloc[-1] - minutes.t_ns.iloc[0]) // (60 * NS) + 1
    assert len(minutes) == expected_minutes, 'Hidden physical-minute deletion'
    receipt = dict(status='PASS', checked_hashes=checked, raw_trade_parity=direct, sampled_volume_reconciliation=sampled, prefix_checks=prefix,
        reader_implementation={p: sha(p) for p in ('spectrum/market_repr/store.py', 'scripts/verify_market_representation_v1.py')},
        minute_clock_rows=len(minutes), expected_physical_minutes=int(expected_minutes),
        minutes_known=int(minutes.known.sum()), scheduled_closed=int((~minutes.scheduled_open).sum()),
        open_unknown=int((minutes.scheduled_open & ~minutes.known).sum()),
        conservation='Stock deltas reconciled to the same book kernel at every affected price; independent T parser separately verifies all trade rows. Stock check is not an independent matching-engine implementation.',
        limits=['Two real-data prefix cuts plus synthetic restart/gap tests, not exhaustive replay at every nanosecond',
                'Source stream audit inherited by hashes; calendar limited to declared August week',
                'No representation-to-neural-policy integration or economic admission'])
    save(REPORT / 'verification.json', receipt)


if __name__ == '__main__': main()
