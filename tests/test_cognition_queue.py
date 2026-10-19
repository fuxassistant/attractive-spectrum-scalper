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
