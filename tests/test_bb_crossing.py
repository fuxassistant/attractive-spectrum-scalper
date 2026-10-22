import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from spectrum.cognition import bb_crossing as bb


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


def test_touch_plateau_motion_and_reset():
    t = np.arange(8, dtype=np.int64); p = np.array([9., 10., 10., 11., 10., 9., 11., 9.])
    m = np.full(8, 10.); h = np.ones(8)*5; s = np.array([0,0,0,0,0,0,1,1])
    row, role, direction, cause, _, _ = bb.crossing_rows(t, p, m, h, s)
    assert row.tolist() == [3,5,7]
    assert role.tolist() == [1,1,1]
    assert direction.tolist() == [1,-1,-1]
    assert cause.tolist() == [0,0,0]
    row, role, direction, cause, dp, dl = bb.crossing_rows(t[:2], np.array([10.,10.]),
        np.array([9.,11.]), h[:2], s[:2])
    assert row.tolist() == [1] and cause.tolist() == [1]
    assert dp[0] == 0 and dl[0] == 2


def test_query_ids_ties_censor_and_maturity():
    t = np.array([0,10,20,30,40,50], np.int64)*bb.NS
    q = dict(t=t, price=np.ones(6)*10, geometries={s:dict(middle=np.ones(6)*10,
        half=np.ones(6), support=np.array([0,0,0,0,-1,1])) for s in bb.SCALES})
    bank = pd.DataFrame([dict(ordinal=3,t_ns=t[3],reduced_row=3,support=0,
        packet_id='same',scale=s,role=r) for s,r in [(1,0),(1,1),(3,2),(6,1),(9,0)]])
    f = bb.make_targets(bank,q,np.array([5,15,30,41])*bb.NS,60*bb.NS,20)
    assert f.m1_y.tolist() == [4,3,-1,-1]  # 5+20: mature NONE; 15: tied lower/middle
    assert f.global_y.tolist() == [13,12,-1,-1]
    assert f.loc[1,'m1_mask'] == 3
    g = bb.make_targets(bank,q,np.array([14,16])*bb.NS,60*bb.NS,20)
    assert g.m1_id.tolist() == ['same/M1','same/M1']
    assert (g.m1_event_ns > g.t_ns).all()


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
    a = bb.crossing_rows(t,p,m,h,s); b = bb.crossing_rows(t[k],p[k],mr,hr,sr)
    np.testing.assert_array_equal(a[0],k[b[0]])


def test_event_weight_and_gradient():
    w = bb.event_weights(['a','a','a','b'])
    assert w[:3].sum() == pytest.approx(w[3])
    z = np.array([[.2,.8,-.1],[.5,-.2,.1]])
    y = np.array([1,0]); ww = np.array([.4,1.6])
    def loss(zz):
        p = np.exp(zz-zz.max(axis=1)[:,None]); p /= p.sum(axis=1)[:,None]
        return -np.sum(ww*np.log(p[np.arange(2),y]))
    p = np.exp(z-z.max(axis=1)[:,None]); p /= p.sum(axis=1)[:,None]
    g = bb.ce_gradient(p,y,ww)
    for i in range(2):
        for j in range(3):
            plus=z.copy(); minus=z.copy(); plus[i,j]+=1e-6; minus[i,j]-=1e-6
            assert g[i,j] == pytest.approx((loss(plus)-loss(minus))/2e-6,abs=1e-9)
    with pytest.raises(ValueError): bb.ce_gradient(p,np.array([-1,0]))


def test_event_source_identity_excludes_query_and_features():
    meta=dict(date='2026-02-02',symbol='GCJ6',quote_parts=[dict(quote_file=dict(sha256='abc'),start_ns=1,end_ns=2)],
        files={'execution_valid.npy':dict(sha256='v'),'execution_segment.npy':dict(sha256='e'),'x.npy':dict(sha256='x')})
    a=bb.source_identity(meta)
    meta['files']['x.npy']['sha256']='new-feature-array';meta['origins']='new-stride'
    assert a==bb.source_identity(meta)
    assert a!=bb.source_identity(meta,0)
    meta['files']['execution_segment.npy']['sha256']='changed-support'
    assert a!=bb.source_identity(meta)


def test_execution_marks_latency_support_and_extrema():
    t = np.arange(20,dtype=np.int64)*200_000_000
    bid = (1000e9 + np.arange(20)*100_000_000).astype(np.int64); ask=bid+200_000_000
    valid=np.ones(20,bool); ep=np.zeros(20,np.int64); k=np.arange(20,dtype=np.int64)
    breaks=bb.execution_breaks(t,valid,ep); blocks=bb.extremum_blocks(bid,ask,k)
    ints, vals=bb.trade_profile(t,bid,ask,valid,ep,breaks,k,blocks,np.array([t[0]]),
        np.array([t[4]]),np.array([4],np.int64),t[-1])
    assert ints[0,:4].tolist() == [1,5,4,1]
    assert vals[0,0] == pytest.approx((bid[4]-ask[1])/1e9-.3)
    assert vals[0,2] == pytest.approx((bid[5]-ask[1])/1e9-.3)
    assert vals[0,4] == pytest.approx(-.5)
    valid[3]=False; breaks=bb.execution_breaks(t,valid,ep)
    ints, vals=bb.trade_profile(t,bid,ask,valid,ep,breaks,k,blocks,np.array([t[0]]),
        np.array([t[4]]),np.array([4],np.int64),t[-1])
    assert ints[0,3] == -1 and np.isnan(vals[0,0]) and np.isnan(vals[0,2])
