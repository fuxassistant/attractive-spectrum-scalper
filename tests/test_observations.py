"""Semantic failures: double counting, future leakage, gaps and fake memory."""
import pickle
import numpy as np
import pandas as pd
import pytest
from spectrum.core.vendor.l3 import MboRecord
from spectrum.core.vendor.l3 import F_LAST
from spectrum.core.vendor.l3 import F_SNAPSHOT
from spectrum.core.vendor.l3_v2 import L3EventEngineV2
from spectrum.representation.raw import InteractionCollector
from spectrum.representation.raw import FIELDS
from spectrum.representation.representation import NS
from spectrum.representation.representation import MINUTE
from spectrum.representation.representation import anchor
from spectrum.representation.representation import scheduled_open
from spectrum.representation.representation import make_fast
from spectrum.representation.representation import make_minutes
from spectrum.representation.representation import anchored_context
from spectrum.representation.representation import MarketObservationStore
from spectrum.representation.representation import FAST_VALUES
from spectrum.representation.representation import DAY_VALUES
from spectrum.representation.representation import ledger_releases
from spectrum.representation.zones import zone_and_event_tables
from spectrum.representation.zones import MECHANISMS
from spectrum.future.endpoint import future_endpoint_move


def record(i, t, action, side='N', price=0, size=0, oid=0, flags=F_LAST):
    t = int(t * NS)
    return MboRecord(i, 1, 7, 22, max(0, t - 1), t, i, oid, round(price * NS), size, flags, action, side)


def raw_sequence():
    return [record(0, 0, 'R', flags=F_SNAPSHOT),
        record(1, 0, 'A', 'B', 100, 20, 1, F_SNAPSHOT),
        record(2, 0, 'A', 'A', 100.2, 20, 2, F_SNAPSHOT | F_LAST),
        record(3, .5, 'T', 'B', 100.2, 7, flags=0),
        record(4, .5, 'F', 'A', 100.2, 7, 2, 0),
        record(5, .5, 'C', 'A', 100.2, 7, 2),
        record(6, .7, 'A', 'A', 100.2, 3, 3),
        record(7, .8, 'A', 'A', 100.2, 8, 4),
        record(8, .9, 'C', 'B', 100, 4, 1),
        record(9, 1.1, 'N'), record(10, 2.1, 'N')]


def collect(rows, restart_at=None):
    levels, trades = [], []
    engine = L3EventEngineV2(publisher_id=1, instrument_id=7, channel_id=22)
    c = InteractionCollector(levels.extend, trades.append)
    for j, row in enumerate(rows):
        ev = engine.append(row)
        if ev is not None: c.feed(ev, engine)
        if j == restart_at:
            # Emulate process serialization at a complete-publication boundary.
            assert not engine.pending
            state = {k: v for k, v in c.__dict__.items() if not k.startswith('emit_')}
            c2 = InteractionCollector(levels.extend, trades.append)
            c2.__dict__.update(pickle.loads(pickle.dumps(state)))
            c = c2
            engine = L3EventEngineV2.restore(engine.checkpoint())
    return pd.DataFrame(levels), pd.DataFrame(trades), c


def test_trade_is_once_refill_is_capped_and_snapshot_age_unknown():
    levels, trades, c = collect(raw_sequence())
    assert trades.quantity.sum() == 7
    ask = levels[levels.side == -1].iloc[0]
    assert ask.remove == ask.linked_fill == 7 and ask.unlinked_remove == 0
    assert ask.refill == 7 and ask['add'] == 11
    assert ask.refill_delay_qseconds == pytest.approx(3 * .2 + 4 * .3)
    assert ask.removed_age_known_qty == 0
    bid = levels[levels.side == 1].iloc[0]
    assert bid.unlinked_remove == 4 and bid.linked_fill == 0
    assert c.stats['max_stock_residual'] == 0
    assert levels.available_ns.min() == int(1.1 * NS)


def test_raw_prefix_and_restart_are_identical():
    rows = raw_sequence()
    a, at, _ = collect(rows)
    b, bt, _ = collect(rows, restart_at=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(at, bt)
    p, pt, _ = collect(rows[:-1])
    pd.testing.assert_frame_equal(a.iloc[:len(p)].reset_index(drop=True), p)
    pd.testing.assert_frame_equal(at, pt)


def test_final_partial_bucket_is_never_backdated():
    levels, trades, c = collect(raw_sequence()[:9])
    assert len(levels) == 0 and len(trades) == 1
    c.finish()
    assert c.stats['partial_end_level_rows'] == 2


def synthetic_fast(n=400, start='2026-08-25 08:00'):
    origin = pd.Timestamp(start, tz='UTC').value
    t = origin + np.arange(1, n + 1) * NS
    mid = 4600 + np.sin(np.arange(n) / 18) * .2
    sec = pd.DataFrame(dict(t_ns=t, available_ns=t + 10, epoch=1, valid=True, book_valid=True,
        age_ns=10, bid=mid - .1, ask=mid + .1, mid=mid, bid_size=4, ask_size=5,
        depth_bid_1=4, depth_ask_1=5, depth_bid_4=12, depth_ask_4=13, depth_bid_12=40, depth_ask_12=42,
        c_path_ticks=np.cumsum(np.r_[0, np.abs(np.diff(mid)) / .1])))
    levels = pd.DataFrame([dict(t_ns=x, available_ns=x+10, side=1, price_nano=int(4600*NS), **{c: 0. for c in FIELDS}) for x in t])
    trades = pd.DataFrame(dict(t_ns=t - NS // 2, available_ns=t - NS // 2,
        ordinal=np.arange(n), price_nano=np.round((mid + .1)*NS).astype(np.int64), quantity=1, side=1, supported=True, epoch=1))
    return sec, levels, trades


def test_transform_prefix_and_gap_preserve_physical_grid():
    s, l, t = synthetic_fast()
    full = make_fast(s, l, t)
    prefix = make_fast(s.iloc[:241], l.iloc[:241], t.iloc[:241])
    pd.testing.assert_frame_equal(prefix, full.iloc[:241].reset_index(drop=True))
    s.loc[150:155, 'valid'] = False
    gapped = make_fast(s, l, t)
    assert len(gapped) == len(s)
    assert gapped.ret30_ticks.iloc[156:186].isna().all()
    assert not gapped.ledger_known.iloc[156]
    assert np.isfinite(gapped.ret30_ticks.iloc[187])


def test_quote_freshness_is_not_a_physical_history_reset():
    s, l, t = synthetic_fast()
    s.loc[140:143, 'age_ns'] = 3 * NS
    f = make_fast(s, l, t)
    assert f.known.iloc[140:144].all() and not f.fresh_2s.iloc[140:144].any()
    assert f.local_segment.iloc[130] == f.local_segment.iloc[160]
    assert f.ret300_ticks.iloc[350] == pytest.approx((s.mid.iloc[350] - s.mid.iloc[50]) / .1)


def test_join_preserves_nanoseconds_and_delays_a_late_prefix():
    s, l, t = synthetic_fast(90)
    l.loc[0, 'available_ns'] = int(s.t_ns.iloc[2]) + 113
    l = l.drop(index=1).reset_index(drop=True)
    f = make_fast(s, l, t)
    assert f.source_available_ns.iloc[0] == int(s.t_ns.iloc[2]) + 113
    assert f.available_ns.iloc[:3].eq(int(s.t_ns.iloc[2]) + 113).all()
    assert f.available_ns.is_monotonic_increasing


def test_empty_sparse_ledger_requires_a_complete_publication_watermark():
    s, l, t = synthetic_fast(10)
    q = pd.DataFrame(dict(t_ns=s.t_ns.iloc[:4] + 100, available_ns=s.t_ns.iloc[:4] + 100))
    s['ledger_release_ns'] = ledger_releases(s, q)
    f = make_fast(s, l.iloc[:4], t.iloc[:4])
    cutoff = int(q.available_ns.iloc[-1])
    assert (f.available_ns <= cutoff).sum() == 4
    assert f.available_ns.iloc[4] == np.iinfo(np.int64).max


def test_minute_trade_vwap_matches_exact_and_incomplete_minute_dropped():
    s, l, t = synthetic_fast(140)
    f = make_fast(s, l, t)
    m = make_minutes(f, t)
    assert len(m) == 2
    assert m.trade_vwap_minute.iloc[0] == pytest.approx(np.average(t.price_nano.iloc[:60]/NS, weights=t.quantity.iloc[:60]))
    assert m.volume.sum() == 120
    assert m.available_ns.iloc[0] > m.t_ns.iloc[0]
    fp = make_fast(s.iloc[:95], l.iloc[:95], t.iloc[:95])
    pd.testing.assert_frame_equal(make_minutes(fp, t.iloc[:95]), m.iloc[:1].reset_index(drop=True))


def test_all_event_high_is_preserved_and_internal_missing_minute_retained():
    s, l, t = synthetic_fast(240)
    f = make_fast(s, l, t)
    base = make_minutes(f, t)
    bars = base[['t_ns', 'available_ns', 'open', 'high', 'low', 'close', 'coverage', 'known']].rename(
        columns=dict(open='o', high='h', low='l', close='c', known='valid'))
    bars.loc[0, 'h'] = 4610.
    bars = bars.drop(index=1)
    f = f.drop(index=119).reset_index(drop=True)
    m = make_minutes(f, t, bars)
    assert len(m) == 4 and m.high.iloc[0] == 4610
    assert m.t_ns.iloc[1] == base.t_ns.iloc[1] and not m.known.iloc[1]
    assert m.volume.sum() == t.quantity.sum()


def test_calendar_anchor_dst_and_scheduled_break_are_separate():
    assert anchor('2026-10-23') == pd.Timestamp('2026-10-23 07:00', tz='UTC').value
    assert anchor('2026-10-26') == pd.Timestamp('2026-10-26 08:00', tz='UTC').value
    t = pd.to_datetime(['2026-08-25 20:59Z', '2026-08-25 21:30Z', '2026-08-25 22:01Z']).asi8
    assert scheduled_open(t).tolist() == [True, False, True]


def test_vwap_proxy_denominator_uses_its_actual_price_support():
    a, b = anchor('2026-08-24'), anchor('2026-08-25')
    stamp = np.array([a + MINUTE, a + 2 * MINUTE, b, b + MINUTE])
    prices = np.array([100., 101., 102., 103.]); volume = np.array([1., 10., 1., 1.])
    m = pd.DataFrame(dict(t_ns=stamp, available_ns=stamp + 1, known=[True, False, True, True],
        high=prices, low=prices, volume=volume, pv=(prices - 4000) * volume,
        p2v=(prices - 4000)**2 * volume, midpoint_proxy_minute=[100., np.nan, 102., 103.]))
    t = pd.DataFrame(dict(t_ns=stamp-1, ordinal=np.arange(4), price_nano=(prices*NS).astype(np.int64), quantity=volume, supported=True))
    c = anchored_context(m, t, '2026-08-25', '2026-08-24')
    assert c.combined_midpoint_proxy.iloc[2] == pytest.approx(101.)
    assert c.combined_trade_vwap_on_proxy_support.iloc[2] == pytest.approx(101.)
    assert c.combined_proxy_volume_share.iloc[2] == pytest.approx(2/12)
    assert c.combined_vwap.iloc[-1] == pytest.approx(np.average(prices, weights=volume))


def test_diagnostic_deadband_is_exact_on_half_tick_grid():
    from spectrum.leakage import export_future_values
    delta, y = future_endpoint_move([4700., 4700., 4700., 4700.], [4700.2, 4699.8, 4700.25, 4699.75])
    assert export_future_values(delta).tolist() == [2., -2., 2.5, -2.5]
    assert export_future_values(y).tolist() == [1, 1, 2, 0]


def test_loader_does_not_expose_future_zeros_or_truncate_previous_day():
    s, l, t = synthetic_fast(1000)
    f = make_fast(s, l, t)
    stamp = anchor('2026-08-24') + np.arange(1, 1801) * MINUTE
    context = pd.DataFrame({name: np.ones(len(stamp)) for name in DAY_VALUES})
    context['t_ns'] = stamp; context['available_ns'] = stamp + NS
    context['known'] = True; context['scheduled_open'] = True
    context.loc[10, 'known'] = False
    zones = pd.DataFrame(dict(zone_id=['a', 'a'], available_ns=[stamp[0], stamp[-1]], price=[1., 999.]))
    events = pd.DataFrame(dict(available_ns=stamp[-300:], kind='CONTACT'))
    store = MarketObservationStore(f, context, zones, events, symbol='GCZ6', previous_date='2026-08-24', date='2026-08-25')
    now = int(pd.Timestamp('2026-08-25 08:10', tz='UTC').value)
    obs = store.observe(now)
    assert obs.receipt['first_day_ns'] == anchor('2026-08-24') + MINUTE
    assert obs.receipt['day_rows'] > 1500
    assert obs.day['available_ns'].max() <= now
    assert not obs.day['known'][10].any() and np.all(obs.day['values'][10] == 0)
    assert obs.zones.price.tolist() == [1.]
    assert obs.fast['t_ns'].max() < now
    assert len(obs.fast['names']) == obs.fast['values'].shape[1]
    assert obs.fast['normalized'].shape == obs.fast['values'].shape
    assert not obs.fast['known'][0, FAST_VALUES.index('bid_add')]


def test_stft_and_bands_do_not_bridge_a_gap():
    s, l, t = synthetic_fast(8500)
    f = make_fast(s, l, t)
    a = make_minutes(f, t)
    assert a.stft_entropy.notna().any()
    f.loc[7800:7860, 'known'] = False
    b = make_minutes(f, t)
    assert b.stft_entropy.iloc[130:].isna().all()
    assert b.bb3.iloc[130:].isna().all()


def test_zone_persistence_requires_thirty_observed_seconds_outside():
    start = anchor('2026-08-25')
    stamp = start + np.arange(50) * NS
    f = pd.DataFrame(dict(t_ns=stamp, available_ns=stamp + 10, known=True,
                          mid=np.r_[np.full(5, 100.), np.full(45, 100.5)]))
    for name in MECHANISMS: f[name] = False
    c = pd.DataFrame([dict(previous_recognized_ns=start, context_start_ns=anchor('2026-08-24'),
                          previous_high=100., previous_low=80., previous_vwap=90., previous_poc=95.)])
    ledger = pd.DataFrame(columns=['t_ns', 'available_ns', 'price_nano', 'linked_fill', 'unlinked_remove', 'ambiguous_remove', 'refill', 'add'])
    z, e = zone_and_event_tables(f, c, ledger, '2026-08-25', 'GCZ6')
    p = e[(e.kind == 'PERSISTENCE_CONFIRMED') & e.zone_id.eq('GCZ6:2026-08-25:previous_high')]
    assert p.t_ns.tolist() == [start + 35 * NS]
    assert p.available_ns.iloc[0] == start + 35 * NS + 10
    f.loc[20, 'known'] = False
    _, e = zone_and_event_tables(f, c, ledger, '2026-08-25', 'GCZ6')
    assert not ((e.kind == 'PERSISTENCE_CONFIRMED') & e.zone_id.eq('GCZ6:2026-08-25:previous_high')).any()
