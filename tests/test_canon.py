import numpy as np
import pandas as pd
import pytest
from spectrum.canon import STATUSES, load_canon
from spectrum.market.finalize_market import flags_for_times

NS = 1_000_000_000
canon = load_canon()
clock = canon.regimes()


def utc(text):
    return int(pd.Timestamp(text, tz='UTC').value)


def spans(day, kind=None, name=None):
    """UTC hour strings of intervals that intersect one UTC day."""
    frame = clock.intervals(utc(day), utc(day) + 86400 * NS)
    if kind:
        frame = frame[frame.kind == kind]
    if name:
        frame = frame[frame.name == name]
    fmt = lambda t: pd.Timestamp(t, tz='UTC').strftime('%m-%d %H:%M')
    return [(fmt(a), fmt(b)) for a, b in zip(frame.open_ns, frame.close_ns)]


def test_every_constant_declares_unit_status_and_source():
    assert canon.version == 'SPECTRUM_CANON_V1' and len(canon.fingerprint) == 64
    for constant in canon.constants.values():
        assert constant.unit and constant.source and constant.status in STATUSES
    assert canon.constants['fee_roundtrip_usd_per_oz'].value == 0.30
    assert canon.constants['spread_default_usd_per_oz'].value == 0.20
    assert canon.constants['spread_default_usd_per_oz'].status == 'DEFAULT'
    assert canon.value('gc_tick_nano') == round(canon.value('gc_tick_usd') * canon.value('price_nano_per_usd'))
    with pytest.raises(TypeError):
        canon.constants['fee_roundtrip_usd_per_oz'] = None


def test_spread_default_is_used_only_where_unmeasured_and_stays_marked():
    values, defaulted = canon.spread_or_default([0.1, 0.0, 0.3], [True, False, True])
    assert values.tolist() == [0.1, 0.2, 0.3] and defaulted.tolist() == [False, True, False]


def test_screenshot_moment_monday_2026_10_26_1806_utc():
    t = [utc('2026-10-26 18:06')]
    row = clock.flags(t).iloc[0]
    assert row.new_york and not (row.london or row.tokyo or row.sydney)
    assert row.gc_globex_open and row.xauusd_ctrader_open
    assert clock.label(t).tolist() == ['NEW_YORK']


def test_screenshot_day_in_the_autumn_mismatch_week():
    # UK left BST on 2026-10-25, the US keeps EDT until 2026-11-01.
    assert spans('2026-10-26', 'session', 'sydney') == [('10-25 20:00', '10-26 05:00'), ('10-26 20:00', '10-27 05:00')]
    assert spans('2026-10-26', 'session', 'tokyo') == [('10-26 00:00', '10-26 09:00')]
    assert spans('2026-10-26', 'session', 'london') == [('10-26 08:00', '10-26 17:00')]
    assert spans('2026-10-26', 'session', 'new_york') == [('10-26 12:00', '10-26 21:00')]
    assert spans('2026-10-26', 'overlap', 'london_new_york') == [('10-26 12:00', '10-26 17:00')]


@pytest.mark.parametrize('day, london, new_york, overlap, gc_break', [
    ('2026-01-14', '08:00', '13:00', ('13:00', '17:00'), 22),   # both on standard time
    ('2026-03-18', '08:00', '12:00', ('12:00', '17:00'), 21),   # US DST from 03-08, UK BST from 03-29
    ('2026-07-15', '07:00', '12:00', ('12:00', '16:00'), 21),   # both on summer time
    ('2026-11-04', '08:00', '13:00', ('13:00', '17:00'), 22),   # back to standard time
])
def test_daylight_saving_moves_sessions_overlaps_and_the_gc_break(day, london, new_york, overlap, gc_break):
    assert spans(day, 'session', 'london')[0][0][-5:] == london
    assert spans(day, 'session', 'new_york')[0][0][-5:] == new_york
    assert tuple(x[-5:] for x in spans(day, 'overlap', 'london_new_york')[0]) == overlap
    hours = utc(day) + np.arange(24) * 3600 * NS
    closed = np.flatnonzero(~clock.flags(hours).gc_globex_open.to_numpy())
    assert closed.tolist() == [gc_break]


def test_weekend_break_follows_new_york_time_through_the_dst_change():
    times = [utc(x) for x in ('2026-10-30 20:59', '2026-10-30 21:00', '2026-10-31 12:00',
                              '2026-11-01 22:59', '2026-11-01 23:00')]
    assert clock.flags(times).gc_globex_open.tolist() == [True, False, False, False, True]
    weekly = spans('2026-11-01', 'venue', 'gc_globex')
    assert weekly == [('11-01 23:00', '11-02 22:00')]


def test_gc_hours_match_the_finalized_market_schedule_every_minute_for_a_year():
    t = np.arange(utc('2025-10-01'), utc('2026-10-01'), 60 * NS, dtype=np.int64)
    closed, _ = flags_for_times(t, np.empty(0, np.int64))
    assert np.array_equal(~clock.flags(t).gc_globex_open.to_numpy(), closed)


def test_anchors_and_activity_window_are_local_wall_clock():
    lbma = canon.anchors['lbma_gold_pm'].times(utc('2026-07-13'), utc('2026-07-18'))
    assert [pd.Timestamp(x, tz='UTC').strftime('%a %H:%M') for x in lbma] == [
        'Mon 14:00', 'Tue 14:00', 'Wed 14:00', 'Thu 14:00', 'Fri 14:00']
    times = [utc(x) for x in ('2026-01-14 07:59', '2026-01-14 08:00', '2026-01-14 21:59', '2026-01-14 22:00')]
    assert clock.flags(times).activity_window.tolist() == [False, True, True, False]


def test_all_library_calendars_use_the_canon_for_a_full_year():
    from spectrum.market.calendar import MarketCalendar
    from spectrum.representation.representation import scheduled_open
    t = np.arange(utc('2025-10-01'), utc('2026-10-01'), 60 * NS, dtype=np.int64)
    venue = clock.flags(t).gc_globex_open.to_numpy()
    assert np.array_equal(scheduled_open(t), venue)
    assert np.array_equal(~MarketCalendar().closed(t), venue)
    assert {note['where'].split(' ')[0] for note in canon.notes('closed_hours')} >= {
        'market.calendar.MarketCalendar', 'representation.representation.scheduled_open'}


@pytest.mark.parametrize('date', ['2026-01-14', '2026-03-08', '2026-07-15', '2026-11-01'])
def test_batch_stream_and_checkpoint_share_the_canon(date):
    from spectrum.features import compute_batch
    from spectrum.features.stream import FeatureStream
    from test_foundation import frame
    data = frame(240)
    data['t_ns'] = utc(date + ' 21:59') + np.arange(len(data)) * NS
    data['available_ns'] = data.t_ns + 100
    expected = compute_batch(data, ('motion',))
    stream = FeatureStream(('motion',), source_id='calendar-test')
    first = stream.feed_batch(data.iloc[:90])
    stream = FeatureStream.restore(stream.checkpoint(), expected_source_id='calendar-test')
    second = stream.feed_batch(data.iloc[90:])
    np.testing.assert_array_equal(np.vstack([first.known, second.known]), expected.known)
    np.testing.assert_allclose(np.vstack([first.values, second.values]), expected.values)
    data['scheduled_closed'] = ~clock.flags(data.t_ns).gc_globex_open.to_numpy()
    np.testing.assert_array_equal(compute_batch(data, ('motion',)).known, expected.known)
    data.loc[0, 'scheduled_closed'] = not data.loc[0, 'scheduled_closed']
    with pytest.raises(ValueError, match='scheduled_closed disagrees'):
        compute_batch(data, ('motion',))


@pytest.mark.parametrize('field', ['canon_fingerprint', 'canon_implementation_sha256', 'tzdata_version'])
def test_checkpoint_rejects_calendar_semantic_drift(field):
    from spectrum.features.stream import FeatureStream
    from spectrum.core.state import seal, unseal
    stream = FeatureStream(source_id='test')
    saved = unseal(stream.checkpoint())
    saved['calendar'][field] = 'different'
    with pytest.raises(ValueError, match='calendar drift'):
        FeatureStream.restore(seal(saved), expected_source_id='test')


def test_legacy_calendar_checkpoint_requires_explicit_rebuild():
    from spectrum.features.stream import FeatureStream
    from spectrum.core.state import seal, unseal
    saved = unseal(FeatureStream(source_id='test').checkpoint())
    saved['contract'] = 'FEATURE_STREAM_V2'
    with pytest.raises(ValueError, match='contract drift'):
        FeatureStream.restore(seal(saved), expected_source_id='test')
