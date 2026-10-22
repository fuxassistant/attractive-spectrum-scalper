import numpy as np
import pandas as pd
from scripts.bb_crossing_session import lookup,all_marks
from spectrum.cognition import bb_crossing as bb


def test_session_lookup_keeps_late_first_cross_but_never_bridges_support():
    p=pd.DataFrame(dict(t_ns=np.array([1900,4000])*bb.NS,label=[2,0],packet_id=['same/M9','later/M9'],
        ordinal=[10,20],line_mask=[4,1]))
    t=np.array([0,100,1900])*bb.NS;end=np.array([5000,1500,4000])*bb.NS
    f=lookup(p,t,end,np.ones(3,bool),'m9')
    assert f.m9_known.tolist()==[True,False,False]
    assert f.m9_delay_s.iloc[0]==1900 and f.m9_id.iloc[0]=='same/M9'


def test_session_batch_marks_match_brute_extrema_and_unknown_stays_nan():
    bid=np.array([100,102,98,106,104],np.int64)*1_000_000_000;ask=bid+200_000_000;k=np.arange(5,dtype=np.int64)
    blocks=bb.extremum_blocks(bid,ask,k);entries=np.array([0,1]);crosses=np.array([4,-1]);known=np.array([True,False]);breaks=np.full(5,5,np.int64)
    v=all_marks(bid,ask,k,blocks,entries,crosses,known,breaks)
    np.testing.assert_allclose(v[0],[(bid[4]-ask[0])/1e9-.3,(bid[0]-ask[4])/1e9-.3,
        (bid.min()-ask[0])/1e9-.3,(bid[0]-ask.max())/1e9-.3,
        (bid.max()-ask[0])/1e9-.3,(bid[0]-ask.min())/1e9-.3])
    assert np.isnan(v[1]).all()
