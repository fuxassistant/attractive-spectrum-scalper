import numpy as np
import pandas as pd
from scripts.bb_crossing_learn import account
from spectrum.cognition.bb_crossing import NS


def row(t, **changes):
    r=dict(date='2026-01-02',t_ns=t*NS,bb1_position=.2,pred_L=1.,pred_S=-1.,execution_status=1,
        entry_ns=t*NS+100_000_001,exit_ns=(t+45)*NS,termination='CROSS',net_L=2.,net_S=-3.,
        exit_mae_L=-1.,exit_mae_S=-3.,mark_L=2.,mark_S=-3.,cross_ns=(t+44)*NS,
        cross_id='event-'+str(t),entry_ordinal=1,exit_ordinal=10)
    r.update(changes);return r


def test_account_does_not_sum_overlapping_queries_or_skip_selected_censor():
    f=pd.DataFrame([row(0),row(30,net_L=1000.),row(60,execution_status=-1,termination='CENSORED',net_L=np.nan),row(90,net_L=1000.)])
    stats,trades=account(f,1,['2026-01-02'])
    assert stats['fills']==1
    assert stats['net'] is None and stats['observed_prefix_net']==2.
    assert stats['censor']['t_ns']==str(60*NS)
    assert trades.t_ns.tolist()==[0]


def test_account_wait_and_origin_geometry_are_causal_gates():
    f=pd.DataFrame([row(0,execution_status=-1,pred_L=0.,pred_S=-1.),
        row(30,bb1_position=np.nan,execution_status=-2),row(60,execution_status=0),row(90)])
    stats,trades=account(f,1,['2026-01-02','2026-01-05'])
    assert stats['censor'] is None and stats['fills']==1
    assert trades.t_ns.tolist()==[90*NS]
    assert stats['daily']=={'2026-01-02':2.,'2026-01-05':0.}


def test_no_extra_cooldown_after_actual_exit_and_short_uses_own_net():
    f=pd.DataFrame([row(0,exit_ns=30*NS),row(30,pred_L=-2.,pred_S=3.,net_S=4.)])
    stats,trades=account(f,1,['2026-01-02'])
    assert stats['fills']==2 and stats['net']==6.
    assert trades.side.tolist()==['L','S']
