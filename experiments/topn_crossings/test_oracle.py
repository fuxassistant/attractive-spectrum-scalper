import numpy as np
import pandas as pd
import pytest

from spectrum.crossings.bank import SCALES
from spectrum.leakage import FutureFrame
from experiments.topn_crossings.oracle import make_windows, prepare, outcome_representatives, summarize_window, future_labels, write_labels, read_labels


def fixture():
    t=np.arange(1,9,dtype=np.int64)*1_000_000_000
    tape=dict(t=t,bid=np.array([10,9,11,13,8,12,15,14.]),ask=np.array([10,9,11,13,8,12,15,14.])+.2,
              support={s:np.zeros(8,np.int64) for s in SCALES})
    bank=pd.DataFrame(dict(t_ns=t[[3,6,7]],scale=1,role=0,support=0,event_id=['a','b','c'],
         cluster_event_id=['a','a','c'],ordinal=[3,6,7],bid=tape['bid'][[3,6,7]],ask=tape['ask'][[3,6,7]],
         line=[13.,15.,14.],opportunity=[True,False,True],cluster_rank=[0,1,0]))
    return tape,bank


def test_raw_first_crossing_and_all_origin_extrema():
    tape,bank=fixture()
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[9_000_000_000],partial=[False]))
    valid,ends=prepare(tape,9_000_000_000,calendar=False)
    rows,stats=outcome_representatives(bank,tape,windows,valid,ends)
    by_id=rows.set_index('event_id')
    assert by_id.loc['a','entry_ns']==2_000_000_000
    assert by_id.loc['b','entry_ns']==5_000_000_000
    assert by_id.loc['b','side']=='LONG'  # raw flicker is an eligible first exit
    assert by_id.loc['b','candidate_origins']==3
    assert by_id.loc['a','net']==pytest.approx(3.5)
    assert by_id.loc['b','net']==pytest.approx(6.5)
    assert by_id.loc['c','side']=='SHORT'
    assert set(rows['rank'])=={1,2,3}
    s=stats.iloc[0]
    assert s.queries==s.not_ready+s.crossed+s.window_censored+s.support_end
    assert (rows.maturity_ns==9_000_000_000).all()
    assert isinstance(future_labels(rows),FutureFrame)


def test_half_open_window_does_not_replace_first_event():
    tape,bank=fixture()
    windows=pd.DataFrame(dict(window=[0,1],start=[0,4_000_000_000],end=[4_000_000_000,9_000_000_000],partial=False))
    valid,ends=prepare(tape,9_000_000_000,calendar=False)
    rows,_=outcome_representatives(bank,tape,windows,valid,ends)
    assert 'a' not in set(rows.event_id)  # exit on right boundary excluded, origin at event cannot exit there
    assert set(rows.window)=={1}


def test_support_break_cannot_be_crossed():
    tape,bank=fixture();tape['support'][1][4:]=1
    bank.loc[bank.event_id.isin(['b','c']),'support']=1
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[9_000_000_000],partial=False))
    valid,ends=prepare(tape,9_000_000_000,calendar=False)
    rows,_=outcome_representatives(bank,tape,windows,valid,ends)
    assert rows.set_index('event_id').loc['b','candidate_origins']==2


def test_event_windows_do_not_split_simultaneous_line_aliases():
    tape,bank=fixture()
    aliases=pd.concat([bank]*3,ignore_index=True)
    windows=make_windows(tape['t'],aliases,9_000_000_000,'events_2')
    assert list(windows.end)==[8_000_000_000,9_000_000_000]
    assert list(windows.partial)==[False,True]


def test_controls_and_no_positive_are_not_capital_or_full_coverage():
    tape,bank=fixture();valid,ends=prepare(tape,9_000_000_000,calendar=False)
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[9_000_000_000],partial=False))
    rows,_=outcome_representatives(bank,tape,windows,valid,ends)
    result,curves=summarize_window(rows,4)
    assert result['positive_outcomes']==3
    curves=pd.DataFrame(curves)
    assert curves[(curves.control=='cluster')&(curves.k==4)].selected.item()==2
    empty=rows.iloc[:0]
    result,curves=summarize_window(empty)
    assert result['n80']==0 and result['top1_share'] is None
    assert all(r['coverage_of_event_mass'] is None for r in curves)


def test_no_outcome_cannot_generate_positive_labels():
    tape,bank=fixture();bank=bank.iloc[:0]
    valid,ends=prepare(tape,9_000_000_000,calendar=False)
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[9_000_000_000],partial=False))
    rows,stats=outcome_representatives(bank,tape,windows,valid,ends)
    assert rows.empty and stats.crossed.sum()==0


def test_explicit_serialization_restores_future_type(tmp_path):
    frame=pd.DataFrame(dict(net=[1.25],event_id=['e'],entry_ns=[123]))
    path=tmp_path/'y.parquet';write_labels(frame,path)
    saved=read_labels(path)
    assert isinstance(saved,FutureFrame)
    assert set(saved.columns)=={'future_net','future_event_id','future_entry_ns'}


def test_forward_horizon_keeps_entry_window_and_delays_rank_maturity():
    tape,bank=fixture();valid,ends=prepare(tape,9_000_000_000,calendar=False)
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[3_000_000_000],partial=False,
                             horizon_ns=[5_000_000_000],maturity_ns=[8_000_000_000]))
    rows,_=outcome_representatives(bank,tape,windows,valid,ends)
    assert set(rows.event_id)=={'a'}  # later b cannot replace first a
    assert rows.exit_ns.iloc[0]>windows.end.iloc[0]
    assert rows.maturity_ns.iloc[0]==8_000_000_000


def test_empty_aggregate_metrics_are_explicit_unknowns():
    from experiments.topn_crossings.run import finite_json
    summary,_=summarize_window(pd.DataFrame())
    assert summary['top10_flicker_fraction'] is None
    assert finite_json({'empty_mean':float('nan'),'n':0})=={'empty_mean':None,'n':0}


def test_top_n_weights_are_economic_mass_and_preserve_future():
    from experiments.topn_crossings.oracle import select_top_n
    from spectrum.leakage import export_future_values
    labels=future_labels(pd.DataFrame(dict(window=[0,0,0],rank=[1,2,3],net=[5.,3.,2.])))
    selected=select_top_n(labels,2)
    assert isinstance(selected,FutureFrame)
    assert export_future_values(selected.future_weight_within_top_n)==pytest.approx([5/8,3/8])
    assert select_top_n(labels,0).empty
