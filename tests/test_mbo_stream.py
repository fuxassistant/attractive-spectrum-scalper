"""Streaming acceptance: raw-group atomicity, history, clocks and recovery."""
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from spectrum.core.schema import StreamConfig, NS, SNAP_SCHEMA, BAR_SCHEMA
from spectrum.core.stream import MboStream
from spectrum.core.features import build_features
from spectrum.core.feature_stream import FeatureStream
from spectrum.core.feature_registry import Field, Family
from spectrum.core.feature_families import default_families
from spectrum.core.state import seal, unseal
from spectrum.core.vendor.l3 import F_LAST, F_SNAPSHOT
from test_mbo_adapter import bootstrap_rows, rec, quiet_until
from test_mbo_features import sample

CONFIG = StreamConfig(1, 7, 22)


class Output:
    def __init__(self):
        self.quotes, self.snapshots, self.bars, self.features = [], [], [], []

    def callbacks(self):
        return dict(on_quote=self.quotes.append, on_snapshot=self.snapshots.append,
                    on_bar=self.bars.append, on_features=self.features.append)


def observations():
    rows = bootstrap_rows()
    for second in range(1, 25*60+1):
        if second % 7 == 0:
            rows.extend([rec(len(rows), second, 'T', 'B', 100.2, 2, flags=0),
                         rec(len(rows)+1, second, 'N')])
        else:
            rows.append(rec(len(rows), second, 'N'))
    return rows


def assert_outputs(left, right):
    for name in ('quotes', 'snapshots', 'bars'):
        np.testing.assert_array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))
    assert len(left.features) == len(right.features)
    for a, b in zip(left.features, right.features):
        assert a.origin() == b.origin()
        np.testing.assert_array_equal(a.values, b.values)


def test_one_record_batches_and_mid_group_restart_are_identical():
    rows = observations()
    one, chunks, restored = Output(), Output(), Output()
    core = MboStream(CONFIG, **one.callbacks())
    for row in rows:
        core.feed(row)
    core.finish(25*60*NS)
    other = MboStream(CONFIG, **chunks.callbacks())
    for start in range(0, len(rows), 13):
        other.feed_many(rows[start:start+13])
    other.finish(25*60*NS)
    third = MboStream(CONFIG, **restored.callbacks())
    cut = next(i+1 for i, row in enumerate(rows) if row.ts_recv_ns > 20*60*NS and row.action == 'T')
    third.feed_many(rows[:cut])
    assert third.engine.pending
    checkpoint = json.loads(json.dumps(third.checkpoint(), allow_nan=False))
    third = MboStream.restore(checkpoint, expected_config=CONFIG, **restored.callbacks())
    third.feed_many(rows[cut:])
    third.finish(25*60*NS)
    assert_outputs(one, chunks)
    assert_outputs(one, restored)
    assert any(frame.price_ready for frame in one.features)


def test_core_and_dataframe_adapter_use_exact_same_feature_machine():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(observations())
    core.finish(25*60*NS)
    snapshots = pd.DataFrame(out.snapshots, columns=[name for name, _ in SNAP_SCHEMA])
    bars = pd.DataFrame(out.bars, columns=[name for name, _ in BAR_SCHEMA])
    origins, matrix, manifest = build_features(snapshots, bars)
    np.testing.assert_array_equal(matrix, np.stack([frame.values for frame in out.features]))
    pd.testing.assert_frame_equal(origins, pd.DataFrame([frame.origin() for frame in out.features]))
    assert manifest['feature_names'] == core.feature_stream.registry.names


def test_configurable_tick_and_window_are_preserved_by_checkpoint():
    out = Output()
    config = replace(CONFIG, tick_nano=250_000_000)
    core = MboStream(config, feature_windows=(6, 60), **out.callbacks())
    rows = bootstrap_rows()
    rows[-1] = replace(rows[-1], price_nano=100_500_000_000)
    core.feed_many(rows)
    core.feed(rec(3, 1, 'M', 'A', 100.75, 20, 2))
    core.advance(2*NS+1)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert snap['c_path_ticks'] == .5  # midpoint +0.125 / tick0.25
    restored = MboStream.restore(core.checkpoint(), expected_config=config)
    assert restored.feature_stream.windows == (6, 60)
    assert restored.feature_stream.registry.names == core.feature_stream.registry.names


def test_same_timestamp_groups_not_published_until_exclusive_watermark():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(bootstrap_rows())
    core.feed(rec(3, 2, 'T', 'B', 100.2, 2))
    core.advance(2*NS)
    assert not out.snapshots
    core.feed(rec(4, 2, 'T', 'A', 100., 3))
    core.advance(2*NS+1)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert snap['c_trade_buy'] == 2 and snap['c_trade_sell'] == 3
    assert snap['available_ns'] == 2*NS+1
    with pytest.raises(ValueError, match='watermark'):
        core.feed(rec(5, 2, 'N'))


def test_live_delivery_delay_is_explicit_without_rewriting_source_clock():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(bootstrap_rows(), received_at_ns=NS)
    core.feed(rec(3, 2, 'T', 'B', 100.2, 2), received_at_ns=3*NS)
    core.advance(2*NS+1, received_at_ns=4*NS)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert snap['t_ns'] == 2*NS and snap['age_ns'] == 0
    assert snap['available_ns'] == 4*NS and out.quotes[-1][-1] == 3*NS
    restored = MboStream.restore(core.checkpoint())
    with pytest.raises(ValueError, match='Delivery time regression'):
        restored.feed(rec(4, 3, 'N'), received_at_ns=3*NS)


def test_timer_during_partial_reset_reads_committed_book_and_restores():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(bootstrap_rows())
    core.feed(rec(3, 1, 'R', flags=F_SNAPSHOT))
    core.feed(rec(4, 1, 'A', 'B', 90, 4, 7, F_SNAPSHOT))
    core.advance(2*NS+1)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert snap['bid'] == 100 and snap['depth_bid_1'] == 20
    checkpoint = json.loads(json.dumps(core.checkpoint(), allow_nan=False))
    core = MboStream.restore(checkpoint, **out.callbacks())
    core.feed(rec(5, 3, 'A', 'A', 90.2, 5, 8, F_SNAPSHOT | F_LAST))
    core.advance(4*NS+1)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert snap['bid'] == 90 and snap['depth_bid_1'] == 4
    assert snap['epoch'] > out.snapshots[0][1]


def test_incomplete_eof_requires_continue_or_explicit_gap():
    core = MboStream(CONFIG)
    core.feed_many(bootstrap_rows())
    core.feed(rec(3, 1, 'T', 'B', 100.2, 2, flags=0))
    with pytest.raises(ValueError, match='EOF inside F_LAST'):
        core.finish(2*NS)
    assert core.checkpoint()
    core.feed(rec(4, 1, 'N'))
    core.finish(2*NS)


def test_default_eof_does_not_invent_an_unobserved_minute_suffix():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(bootstrap_rows())
    core.feed(rec(3, 45, 'N'))
    core.finish()
    assert not out.bars and not out.features
    assert all(row[0] <= 45*NS for row in out.snapshots)


def test_declared_transport_loss_needs_book_recovery_and_invalidates_windows():
    out = Output()
    core = MboStream(CONFIG, **out.callbacks())
    core.feed_many(bootstrap_rows())
    core.feed(rec(3, 1, 'T', 'B', 100.2, 9, flags=0))
    core.mark_gap(2*NS, 'MISSING_SOURCE_RECORDS')
    core.feed(rec(4, 3, 'N'))
    core.advance(4*NS+1)
    snap = dict(zip(core.snapshot_names, out.snapshots[-1]))
    assert not snap['valid'] and not snap['book_valid']
    assert snap['c_trade_buy'] == 0  # unpublished partial trade was discarded
    core.feed_many([replace(row, source_ordinal=row.source_ordinal+5) for row in bootstrap_rows(5)])
    core.advance(6*NS+1)
    assert dict(zip(core.snapshot_names, out.snapshots[-1]))['valid']


def test_checkpoint_drift_route_duplicate_and_pending_bound_fail_closed():
    core = MboStream(CONFIG)
    core.feed_many(bootstrap_rows())
    with pytest.raises(ValueError, match='duplicate'):
        core.feed(bootstrap_rows()[-1])
    with pytest.raises(ValueError, match='route'):
        core.feed(replace(rec(3, 1, 'N'), instrument_id=77))
    with pytest.raises(ValueError, match='configuration'):
        MboStream.restore(core.checkpoint(), expected_config=replace(CONFIG, tick_nano=250_000_000))
    bad = core.checkpoint()
    bad['state']['config']['instrument_id'] = 99
    with pytest.raises(ValueError, match='checksum'):
        MboStream.restore(bad)
    bounded = MboStream(replace(CONFIG, max_pending_records=2))
    bounded.feed_many(bootstrap_rows()[:2])
    with pytest.raises(ValueError, match='bound'):
        bounded.feed(bootstrap_rows()[2])
    with pytest.raises(RuntimeError, match='failed'):
        bounded.checkpoint()


class Notional:
    name, version, fields = 'notional', '1', ('c_traded_notional',)

    def __init__(self): self.total = 0.
    def reset(self): self.total = 0.
    def values(self, t_ns): return (self.total,)
    def state_dict(self): return {'total': self.total}
    def load_state_dict(self, state): self.total = state['total']
    def update(self, event, book):
        self.total += sum(d.record_size*d.record_price_nano/NS for d in event.deltas
                          if d.action == 'T' and d.accepted and not d.flags & F_SNAPSHOT)


def test_event_measurement_and_feature_family_extend_without_core_edits():
    family = Family('effort_extension', '1', (Field('last_6s_notional', 'USD/oz * contracts'),),
                    lambda context: {'last_6s_notional': context.window(6).delta('traded_notional')})
    families = (*default_families(), family)
    output = Output()
    core = MboStream(CONFIG, families=families, metrics=(Notional(),), **output.callbacks())
    rows = observations()
    core.feed_many(rows[:55])
    saved = json.loads(json.dumps(core.checkpoint(), allow_nan=False))
    with pytest.raises(ValueError, match='metric'):
        MboStream.restore(saved, families=families)
    with pytest.raises(ValueError, match='schema'):
        MboStream.restore(saved, metrics=(Notional(),))
    core = MboStream.restore(saved, families=families, metrics=(Notional(),), **output.callbacks())
    core.feed_many(rows[55:])
    core.finish(25*60*NS)
    names = core.feature_stream.registry.names
    assert len(names) == 334
    assert any(frame.values[names.index('last_6s_notional')] > 0 for frame in output.features)


def test_feature_restart_retains_native_history_and_bounded_storage():
    snapshots, bars = sample(700)
    expected = build_features(snapshots, bars)[1]
    stream = FeatureStream()
    iterator = iter(snapshots.to_dict('records'))
    point = next(iterator, None)
    actual = []
    for i, bar in enumerate(bars.to_dict('records')):
        while point is not None and point['t_ns'] <= bar['t_ns']:
            stream.push_snapshot(point)
            point = next(iterator, None)
        actual.append(stream.push_bar(bar).values)
        if i == 311:
            state = unseal(json.loads(json.dumps(seal(stream.state_dict()), allow_nan=False)))
            stream = FeatureStream()
            stream.load_state_dict(state)
    np.testing.assert_array_equal(actual, expected)
    assert len(stream.snapshots) <= 92 and len(stream.history.closes) <= 61
    assert all(len(b.closes) <= 60 for b in stream.history.native.values())


def test_feature_restart_preserves_previous_week_and_session_summaries():
    times = pd.date_range('2026-08-03 12:00', periods=18, freq='D', tz='UTC').asi8
    first, second = FeatureStream(), FeatureStream()
    for i, t in enumerate(times):
        price = 3000.+i
        snapshot = dict(t_ns=int(t), epoch=0, valid=True, age_ns=0,
                        bid=price-.1, ask=price+.1, mid=price)
        bar = dict(t_ns=int(t), epoch=0, valid=True, o=price, h=price+1, l=price-1,
                   c=price, volume=1., coverage=1.)
        first.push_snapshot(snapshot)
        second.push_snapshot(snapshot)
        np.testing.assert_array_equal(first.push_bar(bar).values, second.push_bar(bar).values)
        if i == 9:
            saved = unseal(json.loads(json.dumps(seal(second.state_dict()), allow_nan=False)))
            second = FeatureStream()
            second.load_state_dict(saved)
    assert first.history.previous_week == second.history.previous_week


def test_new_core_has_no_dataframe_file_training_or_target_dependency():
    code = "import sys; from spectrum.core.stream import MboStream; assert not set(('pandas','pyarrow','torch','catboost')) & set(sys.modules); assert not any(n.endswith('.targets') for n in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True)
