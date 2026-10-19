import numpy as np
import pytest

from spectrum.cognition import scalper as sc
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.passive import Quotes, exit_path

NS = 10**9
TICK = 100_000_000


def quotes(seconds, bids, asks, valid=None, epoch=None):
    t = np.rint(np.asarray(seconds, float) * NS).astype(np.int64)
    b = np.rint(np.asarray(bids, float) * TICK).astype(np.int64); a = np.rint(np.asarray(asks, float) * TICK).astype(np.int64)
    n = len(t)
    return (t, b, a, np.asarray(valid if valid is not None else [True] * n),
            np.asarray(epoch if epoch is not None else [1] * n, np.int32))


def scan(q, start, side, entry_ticks, mid_mode, take_ticks, stop_ticks, hold_s, horizon_s, deadline_s=10**6):
    t, b, a, ok, ep = q
    st = np.zeros(1, np.int8); ex = np.zeros(1, np.int64); net = np.full(1, np.nan)
    sc.scan_exits(t, b, a, ok, ep, deadline_s * NS, np.array([start]), np.array([side]), np.array([entry_ticks * TICK]),
                  np.array([mid_mode]), np.array([take_ticks * TICK], float), np.array([stop_ticks * TICK], float),
                  np.array([hold_s * NS]), np.array([horizon_s * NS]), st, ex, net)
    return int(st[0]), ex[0] / NS, float(net[0])


def test_spread_widening_alone_does_not_trip_a_mid_stop():
    # long entry at ask 101 (mid 100.5); spread widens symmetrically, mid unchanged; then mid falls 4
    q = quotes([0, 1, 2, 3], [100, 98, 98, 96], [101, 103, 103, 97])
    st, ex, net = scan(q, 0, 1, 101, True, 10, 4, 100, 200)
    assert st == sc.COMPLETE and ex == 3 and net == pytest.approx(96 - 101)
    # the same tape with executable-tick triggers stops at the widening quote
    st, ex, net = scan(q, 0, 1, 101, False, 10, 3, 100, 200)
    assert ex == 1 and net == pytest.approx(98 - 101)


def test_take_time_deadline_and_censoring():
    q = quotes([0, 1, 2, 3], [100, 101, 102, 103], [101, 102, 103, 104])
    assert scan(q, 0, 1, 101, True, 2, 5, 100, 200)[:2] == (sc.COMPLETE, 2)        # mid +2
    assert scan(q, 0, 1, 101, True, 9, 9, 2, 200)[:2] == (sc.COMPLETE, 2)          # time limit
    assert scan(q, 0, 1, 101, True, 9, 9, 100, 200, deadline_s=1)[:2] == (sc.COMPLETE, 1)
    assert scan(q, 0, 1, 101, True, 9, 9, 100, 2)[0] == sc.CENSORED                  # open at horizon
    q2 = quotes([0, 1, 2], [100, 100, 100], [101, 101, 101], epoch=[1, 2, 2])
    assert scan(q2, 0, 1, 101, True, 9, 9, 100, 200)[0] == sc.CENSORED              # segment change


def test_exec_mode_matches_the_validated_exit_path():
    rng = np.random.default_rng(5)
    secs = np.arange(0, 400, .5); mid = 20000 + np.cumsum(rng.choice([-1, 0, 1], len(secs)))
    spread = rng.choice([1, 1, 2, 3], len(secs))
    t, b, a, ok, ep = quotes(secs, mid - spread / 2, mid + spread / 2)
    tape = QuoteTape(t, b, a, ok, ep, int(401 * NS), 0, 10**12, 10**12)
    pq = Quotes(tape)
    for start in (3, 40, 101):
        for side in (1, -1):
            entry = int((a if side > 0 else b)[start])
            ref = exit_path(pq, side, entry, start, 1, horizon_ns=int(t[start]) + 305 * NS)  # plan 24/12/300
            st, ex, net = scan((t, b, a, ok, ep), start, side, entry / TICK, False, 24, 12, 300, t[start] / NS + 305)
            assert ref[0] == 'COMPLETE' and st == sc.COMPLETE
            assert ex * NS == pytest.approx(ref[1]) and net == pytest.approx(side * (ref[2] - entry) / TICK)


def test_taker_entry_is_strictly_after_latency_and_needs_eligibility():
    t, b, a, ok, ep = quotes([0, .1, .15, 3], [100] * 4, [101] * 4, valid=[True, True, True, True])
    idx = sc.taker_entries(t, ok, np.array([0], np.int64), sc.LATENCY, sc.TIMEOUT)
    assert idx[0] == 2                       # the quote exactly at +100 ms is not eligible
    t, b, a, ok, ep = quotes([0, .5], [100] * 2, [101] * 2, valid=[False, True])
    assert sc.taker_entries(t, ok, np.array([0], np.int64), sc.LATENCY, sc.TIMEOUT)[0] == -2
