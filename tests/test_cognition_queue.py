import numpy as np
import pytest

from spectrum.cognition.queue import probe, FILLED, EXPIRED, CENSORED, UNPLACED, F_MAYBE_BAD_BOOK

P = 10**9  # one price unit in nano


def tape(rows):
    """rows: (ts, action, side, price, size, order_id[, flags])"""
    rows = [r if len(r) == 7 else (*r, 0) for r in rows]
    return dict(ts=np.array([r[0] for r in rows], np.int64),
                action=np.array([ord(r[1]) for r in rows], np.uint8),
                side=np.array([ord(r[2]) for r in rows], np.uint8),
                price=np.array([r[3] * P for r in rows], np.int64),
                size=np.array([r[4] for r in rows], np.int64),
                order_id=np.array([r[5] for r in rows], np.int64),
                flags=np.array([r[6] for r in rows], np.uint8))


BOOK = [(1, 'R', 'N', 0, 0, 0), (2, 'A', 'B', 100, 3, 1), (3, 'A', 'B', 100, 2, 2), (4, 'A', 'A', 101, 5, 9)]


TAIL = [(5000, 'A', 'B', 50, 1, 999)]  # tape continues past every cancel time


def one(rows, side=1, price=100, place=10, end=1000, tail=True):
    s, f, k, a = probe(tape(rows + (TAIL if tail else [])), [place], [side], [price * P], [end])
    return int(s[0]), int(f[0]), int(k[0]), int(a[0])


def test_orders_ahead_filling_do_not_fill_the_probe():
    rows = BOOK + [(20, 'T', 'A', 100, 3, 77), (20, 'F', 'B', 100, 3, 1), (21, 'C', 'B', 100, 3, 1)]
    assert one(rows) == (EXPIRED, 0, 0, 5)  # five lots ahead at placement


def test_order_behind_filled_proves_fifo_reached_probe():
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (30, 'T', 'A', 100, 6, 77), (30, 'F', 'B', 100, 3, 1),
                   (30, 'F', 'B', 100, 2, 2), (30, 'F', 'B', 100, 1, 3)]
    assert one(rows)[:3] == (FILLED, 30, 1)


def test_trade_through_and_worse_level_fill():
    assert one(BOOK + [(40, 'T', 'A', 99, 1, 77)])[:3] == (FILLED, 40, 3)
    rows = BOOK + [(11, 'A', 'B', 99, 1, 4), (41, 'F', 'B', 99, 1, 4)]
    assert one(rows)[:3] == (FILLED, 41, 2)
    # a trade AT the price is not proof: no fill by touch
    assert one(BOOK + [(40, 'T', 'A', 100, 1, 77)])[0] == EXPIRED


def test_size_increase_loses_priority_decrease_keeps_it():
    up = BOOK + [(15, 'M', 'B', 100, 4, 1), (16, 'C', 'B', 100, 2, 2), (30, 'F', 'B', 100, 1, 1)]
    assert one(up)[:3] == (FILLED, 30, 1)  # order 1 fell behind; order 2 left the queue
    down = BOOK + [(15, 'M', 'B', 100, 2, 1), (30, 'F', 'B', 100, 1, 1)]
    assert one(down)[0] == EXPIRED
    moved = BOOK + [(15, 'M', 'B', 99, 3, 1), (16, 'M', 'B', 100, 3, 1), (17, 'C', 'B', 100, 2, 2),
                    (30, 'F', 'B', 100, 1, 1)]
    assert one(moved)[:3] == (FILLED, 30, 1)


def test_same_timestamp_add_is_behind_probe_and_expiry_is_inclusive():
    clear = [(20, 'C', 'B', 100, 3, 1), (21, 'C', 'B', 100, 2, 2)]
    rows = BOOK + [(10, 'A', 'B', 100, 1, 5)] + clear + [(1000, 'F', 'B', 100, 1, 5)]
    assert one(rows)[:2] == (FILLED, 1000)
    rows = BOOK + [(10, 'A', 'B', 100, 1, 5)] + clear + [(1001, 'F', 'B', 100, 1, 5)]
    assert one(rows)[0] == EXPIRED


def test_reset_and_bad_book_censor_working_probe():
    assert one(BOOK + [(50, 'R', 'N', 0, 0, 0), (60, 'T', 'A', 90, 1, 7)])[:2] == (CENSORED, 50)
    rows = BOOK + [(50, 'A', 'B', 98, 1, 6, F_MAYBE_BAD_BOOK), (60, 'T', 'A', 90, 1, 7)]
    assert one(rows)[:2] == (CENSORED, 50)


def test_sell_side_mirror_and_unplaced_and_tape_end():
    assert one(BOOK + [(40, 'T', 'B', 102, 1, 7)], side=-1, price=101)[:3] == (FILLED, 40, 3)
    assert one(BOOK + [(40, 'T', 'B', 101, 1, 7)], side=-1, price=101)[0] == EXPIRED
    assert one(BOOK, place=10**6)[0] == UNPLACED
    assert one(BOOK + [(40, 'A', 'B', 98, 1, 6)], end=10**6, tail=False)[0] == CENSORED  # tape ended while working


def test_probes_are_independent_and_order_free():
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (30, 'F', 'B', 100, 1, 3), (40, 'T', 'A', 99, 1, 7)]
    s, f, k, _ = probe(tape(rows + TAIL), [35, 10, 11], [1, 1, 1], [100 * P] * 3, [1000] * 3)
    # five lots stay ahead, so the behind-fill at 30 is a FIFO violation: only the trade-through proves fills
    assert s.tolist() == [FILLED] * 3 and f.tolist() == [40, 40, 40] and k.tolist() == [3, 3, 3]
    assert probe.violations.tolist() == [0, 1, 1]


def test_behind_fill_with_queue_ahead_is_a_violation_not_a_fill():
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (30, 'F', 'B', 100, 1, 3)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert s[0] == EXPIRED and probe.violations[0] == 1


def test_sweep_reports_fills_before_cancels_inside_one_group():
    # F1, F2 (ahead, 5 lots), then F3 (behind) before the C records arrive
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (30, 'T', 'A', 100, 6, 77), (30, 'F', 'B', 100, 3, 1),
                   (30, 'F', 'B', 100, 2, 2), (30, 'F', 'B', 100, 1, 3), (30, 'C', 'B', 100, 3, 1)]
    assert one(rows)[:3] == (FILLED, 30, 1)
    # only 4 of 5 ahead lots reported -> a behind fill is not proof
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (30, 'F', 'B', 100, 3, 1), (30, 'F', 'B', 100, 1, 2),
                   (30, 'F', 'B', 100, 1, 3)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert s[0] == EXPIRED and probe.violations[0] == 1


def test_cancel_ahead_moves_probe_forward():
    rows = BOOK + [(12, 'A', 'B', 100, 1, 3), (20, 'C', 'B', 100, 3, 1), (21, 'C', 'B', 100, 2, 2),
                   (30, 'F', 'B', 100, 1, 3)]
    assert one(rows)[:3] == (FILLED, 30, 1)


def test_queue_state_uses_only_records_strictly_before_decision():
    from spectrum.cognition.queue_state import window_flows, level_sizes
    NS = 10**9
    rows = [(1, 'R', 'N', 0, 0, 0), (2, 'A', 'B', 100, 3, 1), (3, 'A', 'A', 101, 4, 2), (4, 'A', 'B', 99, 7, 3),
            (6 * NS, 'A', 'B', 100, 2, 4), (7 * NS, 'C', 'B', 100, 1, 1), (8 * NS, 'T', 'A', 100, 1, 50),
            (8 * NS, 'F', 'B', 100, 1, 1), (10 * NS, 'A', 'B', 100, 9, 5), (11 * NS, 'A', 'B', 50, 1, 6)]
    tp = tape(rows); t = np.array([10 * NS], np.int64)
    bid, ask = np.array([100 * P]), np.array([101 * P])
    sizes, placed = level_sizes(tp, t, bid, ask)
    assert placed[0] and sizes['bid1'][0] == 3 + 2 - 1 and sizes['ask1'][0] == 4
    assert sizes['bid2'][0] == 0  # one tick ($0.10) below 100 is empty; the order at 99 is ten ticks away
    f = window_flows(tp, t, bid, ask)
    assert f['bid_add_5'][0] == 2 and f['bid_cancel_5'][0] == 1 and f['bid_fill_5'][0] == 1   # 9 lots at t excluded
    assert f['tsell_5'][0] == 1 and f['tbuy_5'][0] == 0
    assert f['ask_add_30'][0] == 4 and f['ask_add_5'][0] == 0  # added at 3 ns: inside 30 s, outside 5 s


def test_fill_reported_before_placement_counts_as_consumed():
    # order 2 (ahead) has an F reported before the probe rests; its C arrives later.
    # Real data case 2025-10-01 3886.6 ask: V2 fell to a negative consumed count.
    rows = BOOK + [(9, 'T', 'A', 100, 2, 70), (9, 'F', 'B', 100, 2, 2),
                   (20, 'C', 'B', 100, 3, 1), (21, 'C', 'B', 100, 2, 2),
                   (25, 'A', 'B', 100, 1, 3), (30, 'F', 'B', 100, 1, 3)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert (s[0], f[0], k[0]) == (FILLED, 30, 1) and probe.violations[0] == 0


ICE = [(1, 'R', 'N', 0, 0, 0), (2, 'A', 'B', 100, 1, 1), (3, 'A', 'A', 101, 5, 9)]


def test_iceberg_refill_goes_behind_the_probe():
    # order 1 (1 lot, ahead) fills, then its display is restored by M: new priority, now behind
    rows = ICE + [(20, 'F', 'B', 100, 1, 1), (21, 'M', 'B', 100, 1, 1), (30, 'F', 'B', 100, 1, 1)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert (s[0], f[0], k[0]) == (FILLED, 30, 1)


def test_hidden_excess_on_last_order_ahead_proves_fill():
    rows = ICE + [(20, 'T', 'A', 100, 3, 77), (20, 'F', 'B', 100, 3, 1)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert (s[0], f[0], k[0]) == (FILLED, 20, 4)


def test_hidden_excess_with_queue_still_ahead_is_a_violation():
    rows = ICE + [(4, 'A', 'B', 100, 2, 2), (20, 'F', 'B', 100, 3, 1)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert s[0] == EXPIRED and probe.violations[0] == 1


def test_execution_below_our_bid_proves_fill_even_if_book_level_is_stale():
    # order 1 still shows 100 in the book but executes at 99: price priority reached our bid first
    rows = ICE + [(4, 'A', 'B', 100, 2, 2), (20, 'F', 'B', 99, 1, 1)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert (s[0], f[0], k[0]) == (FILLED, 20, 2)
    # and a stale-level F at our own price is not queue evidence
    rows = ICE + [(4, 'A', 'B', 100, 2, 2), (5, 'A', 'B', 98, 1, 8), (20, 'F', 'B', 100, 1, 8)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [1], [100 * P], [1000])
    assert s[0] == EXPIRED and probe.violations[0] == 0


def test_stale_book_level_of_repriced_order_is_not_a_worse_fill():
    # a sell order rests deep at 105 (worse than our ask 101) and is repriced
    # aggressively: its F at 100 arrives before its M. No trade reached 101.
    rows = ICE + [(4, 'A', 'A', 105, 2, 7), (20, 'F', 'A', 100, 2, 7), (21, 'M', 'A', 100, 0, 7)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [-1], [101 * P], [1000])
    assert s[0] == EXPIRED
    # an execution genuinely worse than our ask (at 102) still proves the fill
    rows = ICE + [(4, 'A', 'A', 102, 2, 7), (20, 'F', 'A', 102, 1, 7)]
    s, f, k, a = probe(tape(rows + TAIL), [10], [-1], [101 * P], [1000])
    assert (s[0], f[0], k[0]) == (FILLED, 20, 2)


def test_fill_proof_invariant_rejects_fill_without_trade_at_or_through_price():
    from spectrum.cognition.queue import check_fill_proofs
    tp = tape(ICE + [(20, 'T', 'B', 101, 1, 70)] + TAIL)
    assert check_fill_proofs(tp, [10], [-1], [101 * P], [FILLED], [20])
    with pytest.raises(AssertionError):
        check_fill_proofs(tp, [10], [-1], [102 * P], [FILLED], [20])  # no trade reached 102
