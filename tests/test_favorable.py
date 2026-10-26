import numpy as np
import pandas as pd
import pytest
from spectrum.outcomes.favorable import qualify
from spectrum.outcomes.favorable import origin_summary
from spectrum.outcomes.favorable import adjacent_changes


def sample():
    f=pd.DataFrame(dict(date=['x']*9,t_ns=[10**9]*9,split=['train']*9,
        scale=np.repeat([1,3,6],3),role=list(range(3))*3,profile_known=[True]*9,
        hit_known=[True]*9,cross_ordinal=list(range(1,10)),cross_ns=np.arange(1,10)*10**9,
        v6_event_id=[str(i) for i in range(9)],elapsed_seconds=np.arange(1.,10.)))
    for s in ('L','S'):
        f[f'net_R_{s}']=-.5;f[f'mfe_R_{s}']=3.;f[f'mae_R_{s}']=-1.
        f[f'risk_before_cross_{s}']=0.
    return f


def test_mfe_is_not_endpoint_and_exact_equality_passes():
    f=sample();f.loc[1,'net_R_L']=2.
    a=qualify(f)
    assert (a.mfe_qualified_L==1).all()
    assert a.endpoint_qualified_L.tolist()==[0,1,0,0,0,0,0,0,0]
    assert a.endpoint_margin_L.iloc[1]==0


def test_unknown_is_not_none_and_no_hit_has_no_margin():
    f=sample();f.loc[0,'profile_known']=False;f.loc[0,'hit_known']=False
    f.loc[1,'profile_known']=False
    a=qualify(f)
    assert a.endpoint_qualified_L.iloc[:2].tolist()==[-1,0]
    assert a.endpoint_margin_L.iloc[:2].isna().all()
    assert origin_summary(a).endpoint_state.iloc[0]==-1


def test_first_witness_uses_ordinal_not_best_net_or_timestamp():
    f=sample();f.loc[[1,2],'net_R_L']=[2.,3.]
    f.loc[2,'net_R_S']=2.;f.loc[1,'cross_ns']=f.loc[2,'cross_ns']
    a=origin_summary(qualify(f))
    assert a.endpoint_first_id_L.iloc[0]=='1'
    assert a.endpoint_first_id_S.iloc[0]=='2'
    assert a.endpoint_state.iloc[0]==3
    assert a.common_strength.iloc[0]==2.


def test_margin_threshold_monotone_and_future_columns_not_inputs():
    f=sample();f.loc[0,'net_R_L']=2.
    assert qualify(f,1.5).endpoint_qualified_L.sum()>=qualify(f,2).endpoint_qualified_L.sum()>=qualify(f,3).endpoint_qualified_L.sum()
    with pytest.raises(ValueError): qualify(f,0)


def test_switches_and_same_event_changes_are_distinct():
    f=pd.concat([sample().assign(t_ns=10**9+k*5*10**9) for k in range(3)],ignore_index=True)
    f.loc[f.t_ns==6*10**9,'net_R_L']=2.
    f.loc[f.t_ns==11*10**9,'v6_event_id']='new'
    labeled=qualify(f);before=labeled.copy(deep=True)
    a=adjacent_changes(labeled)
    pd.testing.assert_frame_equal(before,labeled)
    assert a[(a.side=='L')&(a.kind=='same_event')].flips.sum()==9
    assert a[(a.side=='L')&(a.kind=='event_switch')].flips.sum()==9
    assert a[(a.side=='S')&(a.kind=='same_event')].pairs.sum()==9


def test_numeric_tolerance_only_repairs_roundoff_at_exact_ratio():
    f=sample();f.loc[0,'net_R_L']=2.-1e-12;f.loc[1,'net_R_L']=2.-1e-6
    a=qualify(f)
    assert a.endpoint_qualified_L.iloc[:2].tolist()==[1,0]


def test_observed_nonfinite_is_rejected_but_masked_unknown_is_allowed():
    f=sample();f.loc[0,'mfe_R_L']=np.nan
    with pytest.raises(ValueError,match='nonfinite'):qualify(f)
    f.loc[0,'profile_known']=False;f.loc[0,'hit_known']=False
    assert qualify(f).mfe_qualified_L.iloc[0]==-1
