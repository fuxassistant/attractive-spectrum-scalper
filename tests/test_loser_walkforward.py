"""Boundary tests for chronology, revoked permissions, and risk-only controls."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.loser_walkforward_v2.engine import contexts,generate,guarded,weekly_selector,score_probability
from experiments.loser_walkforward_v2.evaluate import fast_terminal
from experiments.loser_lab_v1.account import replay
from experiments.loser_walkforward_v2.online import correct
from experiments.loser_walkforward_v2.prepare import dump

CONFIG=json.loads((Path(__file__).parents[1]/'experiments/loser_walkforward_v2/config.json').read_text())
NS=10**9


def test_receipt_serializes_model_arrays_and_unknown_values(tmp_path):
    path=tmp_path/'receipt.json'
    dump(path,{'coef':np.array([[.1,np.nan]]),'n':np.int64(4),'known':np.bool_(True)})
    assert json.loads(path.read_text())=={'coef':[[.1,None]],'n':4,'known':True}
    assert b'\r\n' not in path.read_bytes()


def frame(days=16):
    rng=np.random.default_rng(19); rows=[]
    fields=['distance_r','risk_per_oz','spread_per_oz','past_net30_r','past_net120_r','past_migration_r',
            'past_efficiency30','past_range120_r','past_zone_share_old','past_zone_share_new',
            'x_mem_log_median_dwell_seconds','x_mem_recent_pass_fraction','x_base_mbo_now_trade_total_log',
            'x_mbo_mbo_now_trade_unknown_fraction','x_base_mbo_now_known','x_mbo_mbo_now_trade_imbalance',
            'x_mbo_mbo_now_depth_1_toward_imbalance','x_mbo_mbo_now_book_net_pressure',
            'x_base_trend_atr','x_base_bb_z','x_base_efficiency20','x_base_log_atr_ratio']
    for day in pd.bdate_range('2026-06-22',periods=days):
        for j in range(8):
            t=day.value+(8*3600+j*600)*NS; sign=1 if j%2 else -1
            r={k:float(rng.uniform(.3,.9)) for k in fields}
            r.update(candidate_id=len(rows),date=str(day.date()),symbol='GCQ6',source_population='additional',side=sign,
                     t_ns=t,release_ns=t+300*NS,known=True,past_known=True,target='test',class_index=1 if sign>0 else 3,
                     event_sign=sign,risk_per_oz=2,x_base_mbo_now_known=1)
            rows.append(r)
    return pd.DataFrame(rows)


def test_switch_can_lose_by_continuation_or_reversal():
    p=np.array([[0,.6,.1,.2,.1],[0,.1,.2,.6,.1]])
    score,sign=score_probability(p,'switch')
    assert np.allclose(score,[.4,.4]) and sign.tolist()==[1,-1]
    assert score_probability(p,'fade')[0][1]<0


def test_both_small_or_unknown_impulses_wait():
    f=frame(1); f.loc[0,'past_net30_r']=.1; f.loc[1,'past_known']=False
    z,_=contexts(f,CONFIG)
    assert not z.loc[:1,'impulse_known'].any()


def test_gap_and_contract_reset_context_support():
    f=frame(3); dates=sorted(f.date.unique()); f.loc[f.date.eq(dates[1]),'symbol']='GCZ6'
    f.loc[f.date.eq(dates[2]),'date']='2026-07-09'; z,_=contexts(f,CONFIG)
    assert z.groupby('date').segment.first().tolist()==[0,1,2]


def test_feature_whitelist_excludes_future_and_old_trained_flags():
    f=frame(); f['h_future']=1e9; f['equity_usd']=-1e8; f['pnl']=1e8
    _,sets=contexts(f,CONFIG)
    assert all(not any(s in c for s in ['equity','pnl','h_future','class_index','event_sign']) for cols in sets.values() for c in cols)
    assert len(sets['full'])==19 and len(sets['multiscale'])==23 and len(sets['coarse'])==6


def test_future_labels_do_not_change_first_forward_predictions():
    f=frame(12); a,_=generate(f,CONFIG); day=a.date.min(); altered=f.copy(); later=altered.date.ge(day)
    altered.loc[later,'class_index']=0; altered.loc[later,'event_sign']=0
    b,_=generate(altered,CONFIG)
    pd.testing.assert_frame_equal(a[a.date.eq(day)].reset_index(drop=True),b[b.date.eq(day)].reset_index(drop=True))


def licence_data():
    f=frame(12); rows=[]
    for base in ['fade5_g10','switch5_g10']:
        r=f[['candidate_id','t_ns','date','target','symbol','source_population']].copy()
        r['policy']=base; r['side']=-f.event_sign; r['accepted']=True; r['segment']=0; r['cell']='choppy_aligned'
        r['max_feedback_ns']=-1; r['reason']='ACCEPT'; rows.append(r)
    y=f[['candidate_id','known','event_sign','release_ns']]
    return pd.concat(rows,ignore_index=True),y


def test_licence_requires_prior_support_and_current_labels_cannot_help():
    p,y=licence_data(); a,receipts=guarded(p,y,CONFIG); dates=sorted(p.date.unique())
    assert not a[a.date.eq(dates[0])].accepted.any()
    assert a[a.date.eq(dates[4])].accepted.all()
    poison=y.copy(); ids=p.loc[p.date.ge(dates[4]),'candidate_id'].unique(); poison.loc[poison.candidate_id.isin(ids),'event_sign']*=-1
    b,_=guarded(p,poison,CONFIG)
    pd.testing.assert_frame_equal(a[a.date.le(dates[4])].reset_index(drop=True),b[b.date.le(dates[4])].reset_index(drop=True))
    assert all(r['max_release_ns']<r['cutoff_ns'] for r in receipts)


def test_licence_cannot_borrow_support_across_segment():
    p,y=licence_data(); last=p.date.max(); p.loc[p.date.eq(last),'segment']=1
    a,_=guarded(p,y,CONFIG)
    assert not a[a.date.eq(last)].accepted.any()


def test_weekly_selection_only_sees_preceding_prequential_days():
    f=frame(25); rows=[]
    for w in CONFIG['windows']:
        for gate in CONFIG['gates']:
            for mode in ['fade','switch']:
                r=f[['candidate_id','t_ns','date','target','symbol','source_population']].copy()
                r['policy']=f'{mode}{w}_g{round(gate*100):02d}'; r['side']=-f.event_sign
                r['accepted']=True; r['reason']='ACCEPT'; r['max_feedback_ns']=-1; rows.append(r)
    p=pd.concat(rows,ignore_index=True); y=f[['candidate_id','known','event_sign','release_ns']]
    a,receipt=weekly_selector(p,y,CONFIG); selected=a[a.accepted]; first=selected.date.min()
    assert len(selected)>0 and not a[a.date.lt(first)].accepted.any()
    poison=y.copy(); ids=f.loc[f.date.ge(first),'candidate_id']; poison.loc[poison.candidate_id.isin(ids),'event_sign']*=-1
    b,_=weekly_selector(p,poison,CONFIG)
    pd.testing.assert_frame_equal(a[a.date.eq(first)].reset_index(drop=True),b[b.date.eq(first)].reset_index(drop=True))
    assert all(r['max_feedback_ns']<r['cutoff_ns'] for r in receipt)


@pytest.mark.parametrize('style',['fixed','after_loss','after_gain'])
def test_fast_risk_null_matches_full_paper_account(style):
    rng=np.random.default_rng(7); unit=rng.choice([-2.1,-2,0.1,2,2.1],300); ids=np.arange(len(unit)); threshold=np.full(len(unit),2.)
    d=pd.DataFrame(dict(candidate_id=ids,t_ns=ids*600*NS,date='2026-07-06',accepted=True,side=1))
    y=pd.DataFrame(dict(candidate_id=ids,known=True,threshold_per_oz=threshold,bid_delta_per_oz=unit,event_sign=np.sign(unit),exit_ns=ids*600*NS+NS))
    actual,_=replay(d,y,None,CONFIG,style=style); fast=fast_terminal(unit,threshold,style,CONFIG)
    assert actual['terminal_usd']==pytest.approx(fast['terminal_usd']) and actual['trades']==fast['trades']
    assert actual['bankrupt']==fast['bankrupt'] and actual['depleted80']==fast['depleted80']


def test_intraday_correction_waits_for_common_maturity_and_resets_contract():
    f=frame(3); p=f[['candidate_id','t_ns','date','target','symbol','source_population']].copy()
    p['policy']='switch5_g10'; p['side']=1; p['score']=0.; p['accepted']=False; p['max_feedback_ns']=-1
    for i,v in enumerate([.2,.2,.2,.2,.2]): p['p'+str(i)]=v
    y=f[['candidate_id','t_ns','date','target','known','class_index','event_sign','release_ns']]
    a=correct(p,f,y,CONFIG)
    assert (a.max_feedback_ns<a.t_ns).all()
    first_id=f.candidate_id.iloc[0]; changed=y.copy(); changed.loc[changed.candidate_id.eq(first_id),'event_sign']*=-1
    b=correct(p,f,changed,CONFIG)
    pd.testing.assert_frame_equal(a[a.candidate_id.eq(first_id)].reset_index(drop=True),b[b.candidate_id.eq(first_id)].reset_index(drop=True))
    f.loc[f.date.eq(f.date.max()),'symbol']='GCZ6'; c=correct(p,f,y,CONFIG)
    first_new=f.loc[f.date.eq(f.date.max()),'candidate_id'].iloc[0]
    assert c[c.candidate_id.eq(first_new)].feedback_count.eq(0).all()
