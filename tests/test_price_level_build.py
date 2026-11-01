import numpy as np
import pandas as pd
import pytest
from experiments.price_level_memory_v1.build import heat_cells
from experiments.price_level_memory_v1.memory import NS


def quotes(times,mid,valid=None):
    return pd.DataFrame(dict(available_ns=np.asarray(times,dtype=np.int64)*NS,
        bid_nano=np.rint((np.array(mid)-.05)*NS).astype(np.int64),
        ask_nano=np.rint((np.array(mid)+.05)*NS).astype(np.int64),
        valid=valid if valid is not None else True,epoch=1))


def trades(times=(),prices=(),qty=()):
    return pd.DataFrame(dict(available_ns=np.asarray(times,dtype=np.int64)*NS,
        price_nano=np.rint(np.array(prices)*NS).astype(np.int64),quantity=np.array(qty,dtype=np.int64),supported=True))


def test_native_dwell_split_at_cell_boundary_and_clipped_to_asof():
    q=quotes([55,65,70],[100.1,100.6,200.1])
    h=heat_cells(q,trades([59,60,67],[100.1,100.6,100.6],[2,3,5]),55*NS,67*NS)
    by={(r[0],r[1]):r[2:] for r in h}
    assert by[(0,100)]==[5,2]
    assert by[(60,100)]==[5,0]
    assert by[(60,100.5)]==[2,3]  # print at R excluded
    assert sum(r[2] for r in h)==12


def test_heat_does_not_invent_dwell_across_invalid_or_stale_gap():
    q=quotes([0,100,110,120],[100.1,101.1,102.1,103.1],[True,False,True,True])
    h=heat_cells(q,trades(),0,120*NS)
    assert sum(r[2] for r in h)==40
    assert not any(r[1]==101 for r in h)


def test_history_cell_never_includes_future_within_same_five_minute_bucket():
    q=quotes([290,300,310],[100.1,100.6,101.1])
    a=heat_cells(q,trades(),290*NS,305*NS,bin_seconds=300)
    b=heat_cells(q.iloc[:2],trades(),290*NS,305*NS,bin_seconds=300)
    assert a==b
    assert sum(r[2] for r in a)==15


def test_small_buckets_rejected_instead_of_wrong_multibucket_split():
    with pytest.raises(ValueError,match='at least'):
        heat_cells(quotes([0],[100.1]),trades(),0,100*NS,bin_seconds=10)
