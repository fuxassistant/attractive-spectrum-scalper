import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import spectrum.crossings.bank as bb


def test_native_math_and_prefix_causality():
    t = np.arange(90) * bb.MINUTE + 1
    p = 5000 + np.sin(np.arange(90)*.3) + np.arange(90)*.1
    good = np.ones(90, bool); epoch = np.zeros(90, np.int64)
    m, h, support = bb.native_geometry(t, p, good, epoch, 3)
    i = 75
    closes = p[np.arange(20, 75, 3)]  # buckets 6..24, 19 completed closes
    values = np.r_[closes, p[i]]
    assert len(values) == 20
    np.testing.assert_allclose(m[i], values.mean(), atol=1e-10)
    np.testing.assert_allclose(h[i], 2*values.std(ddof=1), atol=1e-10)
    m0, h0, _ = bb.native_geometry(t, p, good, epoch, 3, 0)
    np.testing.assert_allclose(h[i]/h0[i], np.sqrt(20/19))
    p2 = p.copy(); p2[i+1:] += 1000
    m2, h2, _ = bb.native_geometry(t, p2, good, epoch, 3)
    np.testing.assert_allclose(m[:i+1], m2[:i+1], equal_nan=True)
    np.testing.assert_allclose(h[:i+1], h2[:i+1], equal_nan=True)
    good[70] = False
    _, _, ss = bb.native_geometry(t, p, good, epoch, 3)
    assert np.all(ss[70:] == -1)


def test_bid_or_ask_crossing_touch_line_motion_and_reset():
    t = np.arange(8, dtype=np.int64)
    bid = np.array([9., 9.5, 9.5, 10.5, 9.5, 8.5, 10.5, 8.5]); ask = bid + .5
    m = np.full(8, 10.); h = np.ones(8)*5; s = np.array([0,0,0,0,0,0,1,1])
    row, role, direction = bb.crossing_rows(t, bid, ask, m, h, s)
    # ask reaches the line at 1, bid passes it at 3, bid back at 4, ask leaves at 5; reset at 6
    assert row.tolist() == [1,3,4,5,7]
    assert role.tolist() == [1,1,1,1,1]
    assert direction.tolist() == [1,1,-1,-1,-1]
    # a moving line through a still quote is a crossing too
    row, role, direction = bb.crossing_rows(t[:2], np.array([10.,10.]), np.array([10.2,10.2]),
        np.array([9.,11.]), h[:2], s[:2])
    assert row.tolist() == [1] and direction.tolist() == [-1]


def test_future_crossings_status_maturity_and_economics_apart():
    from spectrum.future import future_crossings
    from spectrum.economics import future_crossing_marks
    t = np.array([0,10,20,30,40,50], np.int64)*bb.NS
    support = np.array([0,0,0,0,-1,1])
    tape = dict(t=t, bid=np.full(6, 10.), ask=np.full(6, 10.2), support={s: support for s in bb.SCALES})
    bank = pd.DataFrame([dict(scale=1, role=2, t_ns=t[3], ordinal=3, direction=1, event_id='e',
        support=0, line=11., bid=11.1, ask=11.3)])
    f = future_crossings(bank, tape, np.array([5,25,41])*bb.NS, 20*bb.NS, 60*bb.NS)
    upper = f[(f.scale == 1) & (f.role == 'upper')].set_index('t_ns')
    assert upper.future_status.tolist() == ['HORIZON', 'CROSSED', 'NOT_READY']
    assert upper.future_maturity_ns.tolist() == [25*bb.NS, 30*bb.NS, 0]
    lower = f[(f.scale == 1) & (f.role == 'lower')]
    assert lower.future_status.tolist() == ['HORIZON', 'SUPPORT_END', 'NOT_READY']
    assert 'fee' not in ''.join(f.columns) and all(c.startswith('future_') for c in f.columns
        if c not in ('t_ns', 'scale', 'role', 'origin_bid', 'origin_ask'))
    m = future_crossing_marks(upper.reset_index(), fee_usd_per_oz=.3)
    assert m.future_long_usd.tolist()[1] == pytest.approx(11.1 - 10.2 - .3)
    assert m.future_short_usd.tolist()[1] == pytest.approx(10. - 11.3 - .3)
    assert np.isnan(m.future_long_usd.tolist()[0])


def test_compression_preserves_continuity_and_crossings():
    t = (np.arange(15000)*1.01*bb.NS).astype(np.int64)
    bid = (5000e9 + (np.arange(len(t))//100 % 4)*1e9).astype(np.int64)
    q = SimpleNamespace(quote_t=t,bid=bid,ask=bid+200_000_000,
        valid=np.ones(len(t),bool),epoch=np.zeros(len(t),np.int64))
    k = bb.reduce_quotes(q); p = (q.bid.astype(float)+q.ask)/2e9
    m,h,s = bb.native_geometry(t,p,q.valid,q.epoch,1)
    mr,hr,sr = bb.native_geometry(t[k],p[k],q.valid[k],q.epoch[k],1)
    np.testing.assert_allclose(m[k],mr,equal_nan=True)
    np.testing.assert_allclose(h[k],hr,equal_nan=True)
    b_usd, a_usd = q.bid/1e9, q.ask/1e9
    a = bb.crossing_rows(t,b_usd,a_usd,m,h,s); b = bb.crossing_rows(t[k],b_usd[k],a_usd[k],mr,hr,sr)
    np.testing.assert_array_equal(a[0],k[b[0]])


def test_source_identity_depends_on_quotes_and_geometry_only():
    meta=dict(date='2026-02-02',symbol='GCJ6',quote_parts=[dict(quote_file=dict(sha256='abc'),start_ns=1,end_ns=2)],
        files={'x.npy':dict(sha256='x')})
    a=bb.source_identity(meta)
    meta['files']['x.npy']['sha256']='new-feature-array';meta['origins']='new-stride'
    assert a==bb.source_identity(meta)
    assert a!=bb.source_identity(meta,0)
    meta['quote_parts'][0]['quote_file']['sha256']='changed-quotes'
    assert a!=bb.source_identity(meta)


def test_clusters_keep_flicker_and_mark_visits_and_cooldown():
    s = bb.NS
    # middle flickers, upper is crossed, middle again, then a long silence on middle
    t = np.array([0, 1, 2, 5, 6, 30, 31], np.int64) * s
    role = np.array([1, 1, 1, 2, 1, 1, 1])
    ordinal = np.arange(7)
    support = np.zeros(7, np.int64)
    silence, visit, opportunity, opener, rank = bb.crossing_clusters(t, role, ordinal, support, 8 * s)
    assert visit.tolist() == [True, False, False, True, True, False, False]
    assert silence.tolist() == [-1, s, s, -1, 4 * s, 24 * s, s]
    assert opportunity.tolist() == [True, False, False, True, True, True, False]
    assert opener.tolist() == [0, 0, 0, 3, 4, 5, 5]
    assert rank.tolist() == [0, 1, 2, 0, 0, 0, 1]
    # a new segment starts fresh; crossings at one publication are not 'between' each other
    t2 = np.array([0, 0, 1, 2], np.int64) * s
    silence, visit, opportunity, _, _ = bb.crossing_clusters(
        t2, np.array([1, 2, 1, 1]), np.array([0, 0, 1, 2]), np.array([0, 0, 0, 1]), 8 * s)
    assert visit.tolist() == [True, True, False, True]
    assert silence.tolist() == [-1, -1, s, -1]


def test_cluster_markers_are_causal():
    s = bb.NS
    t = np.array([0, 1, 2, 5, 6, 30, 31], np.int64) * s
    role = np.array([1, 1, 1, 2, 1, 1, 1]); ordinal = np.arange(7); support = np.zeros(7, np.int64)
    full = bb.crossing_clusters(t, role, ordinal, support, 8 * s)
    for cut in range(1, 7):
        part = bb.crossing_clusters(t[:cut], role[:cut], ordinal[:cut], support[:cut], 8 * s)
        for a, b in zip(full, part):
            assert a[:cut].tolist() == b.tolist()


def test_future_crossings_can_ask_for_the_next_opportunity():
    from spectrum.future import future_crossings
    t = np.array([0, 10, 11, 12, 30], np.int64) * bb.NS
    support = np.zeros(5, np.int64)
    tape = dict(t=t, bid=np.full(5, 10.), ask=np.full(5, 10.2), support={s: support for s in bb.SCALES})
    bank = pd.DataFrame([dict(scale=1, role=1, t_ns=t[i], ordinal=i, direction=1, event_id=str(i), support=0,
                              line=10.1, bid=10., ask=10.2, opportunity=i in (1, 4)) for i in (1, 2, 3, 4)])
    f = future_crossings(bank[bank.opportunity], tape, np.array([10]) * bb.NS + 1, 60 * bb.NS, 90 * bb.NS)
    middle = f[(f.scale == 1) & (f.role == 'middle')]
    assert middle.future_cross_ns.tolist() == [30 * bb.NS]
