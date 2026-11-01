import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.scalper_view_audit_v1.audit import shifted_episodes,recognition_table,gate_selection,NS


def tape(times,prices):
    p=np.rint(np.array(prices)*NS).astype(np.int64)
    return pd.DataFrame(dict(available_ns=np.array(times,np.int64)*NS,bid_nano=p-50_000_000,
        ask_nano=p+50_000_000,valid=True,epoch=np.ones(len(p),np.int64),source_ordinal=np.arange(len(p),dtype=np.int64)))


def variant(offset=0,hold=3):
    return dict(bin_ticks=10,departure_ticks=10,confirmation_seconds=hold,grid_offset_ticks=offset)


def test_half_grid_shift_moves_addresses_without_changing_source_clock():
    q=tape([0,1,2,6],[99.8,100.8,102.8,102.8]);before=q.copy()
    ev,ep=shifted_episodes(q,6*NS,variant(5))
    assert_frame_equal(q,before)
    assert (ev.zone_low_nano%(10**9)==500_000_000).all()
    assert ep.loc[~ep.has_previous_departure,'previous_departure_mid_nano'].eq(0).all()
    row=ep[ep.zone_low_nano==100_500_000_000].iloc[0]
    assert row.arrival_ns==NS and row.end_ns==5*NS and row.outcome=='PASS_UP'
    assert row.arrival_mid_nano==100_800_000_000


def test_recognition_uses_last_live_excursion_not_first_failed_attempt():
    q=tape([0,1,2,3,4,8],[99.5,100.5,102.2,100.5,102.2,102.2])
    ev,ep=shifted_episodes(q,8*NS,variant())
    f=recognition_table(ev,ep,0,9*NS)
    r=f[f.zone_low_nano==100*NS].iloc[0]
    assert r.last_gate_ns==4*NS and r.last_edge_ns==4*NS
    assert r.confirmation_ns==7*NS and r.last_gate_to_confirmation_s==3
    assert r.signed_move_after_last_edge_per_oz==0


def test_first_gate_selection_does_not_depend_on_its_future_confirmation():
    q=tape([0,1,2,6],[99.5,100.5,102.2,102.2])
    ev,_=shifted_episodes(q,6*NS,variant())
    full=gate_selection(ev,'x',0,10*NS,10)
    prefix=gate_selection(ev[ev.available_ns<=2*NS],'x',0,10*NS,10)
    assert_frame_equal(full.reset_index(drop=True),prefix.reset_index(drop=True))


def test_forming_event_is_available_before_completed_event_and_no_backdating():
    q=tape([0,1,2,6],[99.5,100.5,102.2,102.2])
    ev,_=shifted_episodes(q,6*NS,variant())
    z=ev[ev.zone_low_nano==100*NS]
    assert z[z.available_ns<=2*NS].event_type.tolist()==['ARRIVAL','EDGE_EXIT','GATE_START']
    assert z[z.event_type=='DEPARTURE_CONFIRMED'].available_ns.item()==5*NS


@pytest.mark.parametrize('offset',[True,.5])
def test_grid_offset_is_native_integer_ticks(offset):
    with pytest.raises(ValueError,match='Integer grid offset'):
        shifted_episodes(tape([0,1],[100.,100.]),NS,variant(offset))
