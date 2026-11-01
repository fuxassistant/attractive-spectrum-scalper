import numpy as np
import pandas as pd
import pytest

from experiments.edge_diagnostics_v1.cost_audit import (
    NS, spread_histogram, histogram_summary, decompose_trades, barrier_probe,
)


def quotes(times, bids, spreads, valid=None):
    return pd.DataFrame(dict(available_ns=np.asarray(times,np.int64)*NS,
        bid_nano=np.rint(np.asarray(bids)*NS).astype(np.int64),
        ask_nano=np.rint((np.asarray(bids)+spreads)*NS).astype(np.int64),
        valid=np.ones(len(times),bool) if valid is None else valid,epoch=np.zeros(len(times),np.int64)))


def test_elapsed_time_weighting_and_hour_boundary_are_not_update_weighting():
    q=quotes([0,1,3],[100]*3,[.2,.8,.4])
    h=spread_histogram(q,start_ns=0,end_ns=10*NS,coverage_end_ns=20*NS)
    assert h.duration_ns.sum()==10*NS
    assert histogram_summary(h)['mean_spread_per_oz']==pytest.approx(.46)
    cut=spread_histogram(q,start_ns=2*NS,end_ns=4*NS,coverage_end_ns=20*NS)
    assert histogram_summary(cut)['mean_spread_per_oz']==pytest.approx(.6)


def test_invalid_and_stale_intervals_do_not_get_forward_filled():
    q=quotes([0,10,50],[100]*3,[.2,.8,.4],valid=[True,False,True])
    h=spread_histogram(q,start_ns=0,end_ns=100*NS,coverage_end_ns=100*NS)
    assert h.duration_ns.sum()==40*NS
    assert histogram_summary(h)['mean_spread_per_oz']==pytest.approx(.35)


def test_same_timestamp_quotes_have_zero_exposure_except_last_and_coverage_clips():
    q=quotes([0,0,2],[100]*3,[9.,.2,.4])
    h=spread_histogram(q,start_ns=0,end_ns=10*NS,coverage_end_ns=3*NS)
    assert h.spread_nano.tolist()==[200_000_000,400_000_000]
    assert h.duration_ns.tolist()==[2*NS,NS]


@pytest.mark.parametrize('side,slip',[(1,0.),(-1,0.),(1,1.),(-1,1.)])
def test_fixed_execution_midpoint_spread_fee_slippage_identity(side,slip):
    q=quotes([0,1],[4000,4001],[.2,.4]);entry=4000.2+slip*.1 if side==1 else 4000-slip*.1
    exit=4001-slip*.1 if side==1 else 4001.4+slip*.1
    net=side*(exit-entry)-.3
    t=pd.DataFrame(dict(entry_ordinal=[0],exit_ordinal=[1],entry_ns=[0],exit_ns=[NS],
        side=[side],entry_price=[entry],exit_price=[exit],net_per_oz=[net],qty=[1],risk_per_oz=[1.2]))
    row=decompose_trades(q,t,slippage_ticks_per_fill=slip).iloc[0]
    assert row.native_mid_move_per_oz==pytest.approx(side*1.1)
    assert row.spread_crossing_per_oz==pytest.approx(.3)
    assert row.fee_over_r==pytest.approx(.25)
    assert row.reconstructed_net_per_oz==pytest.approx(net)
    assert row.slippage_per_oz==pytest.approx(2*slip*.1)


def test_exact_line_probe_agrees_with_corrected_library_041():
    p=barrier_probe()
    assert p['current']['reason']=='TAKE' and p['current']['trigger_ns']==300_000_000
    assert p['legacy']['reason']=='TAKE' and p['legacy']['trigger_ns']==300_000_000
