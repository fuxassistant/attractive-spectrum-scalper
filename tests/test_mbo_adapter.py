"""Independent synthetic invariants for the all-event MBO enrichment adapter."""
from dataclasses import replace
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from spectrum.core.adapter import (BAR_SCHEMA, COUNTERS, DEPTH, NS,
    QUOTE_SCHEMA, SNAP_SCHEMA, MapAccumulator, build_day, depth_vector)
from spectrum.core.vendor.glbx_dbn import MBO_DTYPE
from spectrum.core.vendor.l3 import F_LAST, F_SNAPSHOT, MboRecord
from spectrum.core.vendor.l3_v2 import L3EventEngineV2


class MemorySink:
    def __init__(self, schema):
        self.schema = schema
        self.rows = []

    def append(self, row):
        self.rows.append(tuple(row))

    def frame(self):
        return pd.DataFrame(self.rows, columns=[name for name, _ in self.schema])


def rec(i, seconds, action, side="N", price=0., size=0, order=0, flags=F_LAST):
    t = int(round(seconds * NS))
    return MboRecord(i, 1, 7, 22, max(0, t - 1), t, i, order,
                     int(round(price * NS)), size, flags, action, side)


def bootstrap_rows(t=0.):
    return [rec(0, t, "R", flags=F_SNAPSHOT),
            rec(1, t, "A", "B", 100., 20, 1, F_SNAPSHOT),
            rec(2, t, "A", "A", 100.2, 20, 2, F_SNAPSHOT | F_LAST)]


def run(rows, end, period=2 * NS, max_age=2 * NS):
    e = L3EventEngineV2(publisher_id=1, instrument_id=7, channel_id=22)
    q, s, b = [MemorySink(x) for x in (QUOTE_SCHEMA, SNAP_SCHEMA, BAR_SCHEMA)]
    acc = MapAccumulator(q, s, b, period_ns=period, max_age_ns=max_age)
    for row in rows:
        old_depth = depth_vector(e, e._fragment_state(0)) if row.action == "R" and not e.pending else None
        event = e.append(row)
        if event is not None:
            acc.on_event(event, e, old_depth)
    acc.finish(int(end * NS), e)
    return q.frame(), s.frame(), b.frame(), acc, e


def quiet_until(rows, start, end):
    for t in range(start, end + 1):
        rows.append(rec(len(rows), t, "N"))


def test_all_event_excursion_changes_path_and_m1_not_snapshot_close():
    rows = bootstrap_rows()
    # Midpoint briefly jumps five ticks and returns between 2s snapshots.
    rows += [rec(3, .5, "M", "B", 100.5, 20, 1, 0),
             rec(4, .5, "M", "A", 100.7, 20, 2),
             rec(5, 1., "M", "B", 100., 20, 1, 0),
             rec(6, 1., "M", "A", 100.2, 20, 2)]
    quiet_until(rows, 2, 61)
    q, s, b, _, _ = run(rows, 62)
    np.testing.assert_allclose(s.mid.dropna(), 100.1)
    assert q.bid.max() == 100.5
    assert b.iloc[0].h == pytest.approx(100.6)
    assert b.iloc[0].l == pytest.approx(100.1)
    assert s.iloc[0].c_path_ticks == pytest.approx(10.)
    assert b.iloc[0].coverage == 1.
    # First minute contains the initial invalid->valid recovery and is censored.
    assert not b.iloc[0].valid


def test_price_path_and_m1_are_invariant_to_snapshot_frequency():
    rows = bootstrap_rows()
    for t in np.arange(.5, 121., .5):
        level = 100. + .1 * (int(t * 2) % 7)
        rows += [rec(len(rows), t, "M", "B", level, 20, 1, 0),
                 rec(len(rows) + 1, t, "M", "A", level + .2, 20, 2)]
    q1, s1, b1, _, _ = run(rows, 122, period=NS)
    q2, s2, b2, _, _ = run(rows, 122, period=2 * NS)
    pd.testing.assert_frame_equal(q1, q2)
    pd.testing.assert_frame_equal(b1, b2)
    common = s1[s1.t_ns.isin(s2.t_ns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(common, s2)


def test_exact_timestamp_snapshot_uses_all_completed_groups_at_boundary():
    rows = bootstrap_rows()
    rows += [rec(3, 2., "M", "B", 100.1, 25, 1, 0),
             rec(4, 2., "M", "A", 100.3, 26, 2),
             rec(5, 2., "M", "B", 100.1, 29, 1)]
    _, s, _, _, _ = run(rows, 4)
    at2 = s[s.t_ns == 2 * NS].iloc[0]
    assert at2.mid == pytest.approx(100.2)
    assert at2.depth_bid_1 == 29 and at2.depth_ask_1 == 26
    assert at2.c_event_count == 3  # snapshot bootstrap + two completed events


def test_incomplete_group_does_not_change_published_price_or_depth():
    rows = bootstrap_rows() + [rec(3, 1., "M", "A", 100.5, 40, 2, 0)]
    q, s, _, acc, e = run(rows, 2)
    assert len(q) == 1 and len(e.pending) == 1
    assert s.iloc[0].ask == pytest.approx(100.2)
    assert s.iloc[0].depth_ask_1 == 20
    assert acc.c[COUNTERS.index("add_ask")] == 0


def test_future_append_preserves_all_prior_published_grids():
    rows = bootstrap_rows()
    quiet_until(rows, 1, 60)
    _, prefix, bars, _, _ = run(rows, 60)
    extended = rows + [rec(len(rows), 60.5, "M", "A", 101., 35, 2)]
    quiet_until(extended, 61, 120)
    _, full, full_bars, _, _ = run(extended, 120)
    # The EOF watermark at60s+1ns and the next event at60.5s release the same
    # final bucket at different actual times; observations must match exactly.
    pd.testing.assert_frame_equal(prefix.drop(columns='available_ns'), full[full.t_ns <= 60 * NS].reset_index(drop=True).drop(columns='available_ns'))
    pd.testing.assert_frame_equal(bars.drop(columns='available_ns'), full_bars[full_bars.t_ns <= 60 * NS].reset_index(drop=True).drop(columns='available_ns'))
    np.testing.assert_array_equal(prefix.available_ns[:-1], full.available_ns[:len(prefix)-1])


def test_trade_fill_cancel_counts_volume_once_and_matches_executed_removal():
    rows = bootstrap_rows() + [
        rec(3, 1., "T", "B", 100.2, 7, 0, 0),
        rec(4, 1., "F", "A", 100.2, 7, 2, 0),
        rec(5, 1., "C", "A", 100.2, 7, 2),
        rec(6, 1.5, "T", "N", 100.1, 3)]
    _, s, _, _, _ = run(rows, 2)
    row = s.iloc[0]
    assert row.c_trade_buy == 7 and row.c_trade_sell == 0 and row.c_trade_unknown == 3
    assert row.c_trade_count == 2
    assert row.c_remove_ask == row.c_matched_fill_remove_ask == 7
    assert row.depth_ask_1 == 13
    assert row.c_add_bid == row.c_add_ask == 0  # snapshots excluded


def test_post_execution_add_requires_same_price_positive_delay_and_caps_size():
    rows = bootstrap_rows() + [
        rec(3, 1., "T", "B", 100.2, 7, 0, 0),
        rec(4, 1., "F", "A", 100.2, 7, 2, 0),
        rec(5, 1., "C", "A", 100.2, 7, 2),
        rec(6, 1., "A", "A", 100.2, 2, 3),
        rec(7, 1.2, "A", "A", 100.3, 10, 4),
        rec(8, 1.5, "A", "A", 100.2, 5, 5),
        rec(9, 1.6, "A", "A", 100.2, 10, 6),
        rec(10, 1.9, "A", "A", 100.2, 10, 7)]
    _, s, _, _, _ = run(rows, 2)
    assert s.iloc[0].c_add_ask == 37
    assert s.iloc[0].c_post_trade_add_ask == 7


def test_overlapping_hits_retain_each_unexpired_cap():
    # Each actual execution keeps its own expiry. A later hit must neither
    # discard the earlier remaining budget nor extend its expiry.
    rows = bootstrap_rows() + [
        rec(3, 1., "T", "B", 100.2, 7, 0, 0),
        rec(4, 1., "F", "A", 100.2, 7, 2, 0),
        rec(5, 1., "C", "A", 100.2, 7, 2),
        rec(6, 1.1, "T", "B", 100.2, 7, 0, 0),
        rec(7, 1.1, "F", "A", 100.2, 7, 2, 0),
        rec(8, 1.1, "C", "A", 100.2, 7, 2),
        rec(9, 1.5, "A", "A", 100.2, 14, 3)]
    _, snapshots, _, _, _ = run(rows, 2)
    final = snapshots.iloc[0]
    assert final.c_trade_buy == 14
    assert final.c_matched_fill_remove_ask == 14
    assert final.c_add_ask == 14
    assert final.c_post_trade_add_ask == 14


def test_zero_support_has_false_validity_and_no_fabricated_prices_or_depth():
    rows = [rec(0, 0., "R", flags=F_SNAPSHOT | F_LAST)]
    _, s, bars, _, _ = run(rows, 60)
    assert not s.valid.any() and not bars.valid.any()
    assert s[["bid", "ask", "mid", *DEPTH]].isna().all().all()
    assert bars.coverage.eq(0).all()


def test_stale_quote_does_not_become_supported_by_sampling():
    _, s, b, _, _ = run(bootstrap_rows(), 60, max_age=2 * NS)
    assert s[s.t_ns == 2 * NS].valid.item()
    assert not s[s.t_ns > 2 * NS].valid.any()
    assert s.book_valid.all()  # source book integrity survives a quiet interval
    assert b.iloc[0].coverage == pytest.approx(2 / 60)
    assert not b.iloc[0].valid


def test_thirty_second_support_preserves_ordinary_quiet_gap_continuity():
    rows = bootstrap_rows() + [rec(3, 10., "N"), rec(4, 20., "N"),
                              rec(5, 30., "N"), rec(6, 40., "N")]
    _, s, _, _, _ = run(rows, 60, max_age=30 * NS)
    assert s.valid.all() and s.book_valid.all()
    assert s.epoch.nunique() == 1
    assert s.age_ns.max() == 20 * NS
    assert s.iloc[-1].c_event_count == 5


def test_thirty_second_freshness_expiry_is_distinct_from_invalid_book():
    _, s, b, _, _ = run(bootstrap_rows(), 60, max_age=30 * NS)
    assert s[s.t_ns <= 30 * NS].valid.all()
    assert not s[s.t_ns > 30 * NS].valid.any()
    assert s.book_valid.all()
    assert s[s.t_ns > 30 * NS][["bid", "ask", "mid", *DEPTH]].isna().all().all()
    assert b.iloc[0].coverage == pytest.approx(.5)


def test_whole_book_activity_is_preserved_outside_near_depth_envelope():
    rows = bootstrap_rows() + [rec(3, 1., "A", "B", 90., 100, 3)]
    _, s, _, _, _ = run(rows, 2)
    assert s.iloc[0].c_add_bid == 100
    assert s.iloc[0].depth_bid_12 == 20


def test_zero_capacity_sweep_is_unknown_not_free_fill():
    rows = bootstrap_rows()
    rows[2] = replace(rows[2], size=2)
    _, s, _, _, _ = run(rows, 2)
    assert s.iloc[0].sweep_buy_1 == 0
    assert np.isnan(s.iloc[0].sweep_buy_5)
    assert np.isnan(s.iloc[0].sweep_buy_10)


def test_receive_time_regression_is_rejected():
    rows = bootstrap_rows() + [rec(3, 1., "N"), rec(4, .9, "N")]
    with pytest.raises(ValueError, match="Receive-time regression"):
        run(rows, 2)


def test_raw_decode_batch_boundaries_preserve_quote_snapshots_and_bars(tmp_path):
    rows = bootstrap_rows()
    for j in range(1100):
        t = (j + 1) * .05
        rows += [rec(len(rows), t, "T", "B", 100.2, 1, flags=0),
                 rec(len(rows) + 1, t, "N")]
    raw = np.zeros(len(rows), dtype=MBO_DTYPE)
    for i, r in enumerate(rows):
        raw[i] = (14, 160, r.publisher_id, r.instrument_id, r.ts_event_ns,
                  r.order_id, r.price_nano, r.size, r.flags, r.channel_id,
                  r.action.encode(), r.side.encode(), r.ts_recv_ns, 0, r.sequence)
    archive = tmp_path / "raw.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("day.dbn.zst", zstd.ZstdCompressor().compress(b"DBN\x03" + struct.pack("<I", 0) + raw.tobytes()))
    outputs = []
    for block in (1024, 2048):
        out = tmp_path / str(block)
        receipt = build_day(archive, "day.dbn.zst", out, route=(1, 7, 22), end_ns=60 * NS, block_records=block)
        assert receipt["pending_unpublished_records"] == 0
        outputs.append(out)
    for name in ("quotes.parquet", "snapshots.parquet", "bars.parquet"):
        pd.testing.assert_frame_equal(pq.read_table(outputs[0] / name).to_pandas(), pq.read_table(outputs[1] / name).to_pandas())
