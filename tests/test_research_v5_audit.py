"""Independent audit rejects semantic, gradient, selection and lineage errors."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.audit_learning_v5 import (
    COEFFICIENTS, HEADS, array_sha, expected_feature_names, filter_forward,
    independently_reconstruct_targets, prefix_check, restored_batch, sha,
    tcn_forward, verify_feature_schema, verify_filter_gradients, verify_hierarchy,
    verify_receipt, verify_scaler, verify_selection, candidate_diagnostics,clipping_diagnostics,risk_monotonicity,
)


def prediction():
    y = np.array([[0,-1,-1,-1,-1], [1,0,-1,-1,-1], [1,1,0,1,1], [-1]*5],float)
    z = np.arange(20).reshape(4,5)/10-1
    p = 1/(1+np.exp(-z)); g=(p-np.maximum(y,0))*(y>=0)*COEFFICIENTS
    frame = pd.DataFrame()
    for j,h in enumerate(HEADS):
        for pre,a in [('y',y),('z',z),('p',p),('g',g)]:
            frame[pre+'_'+h]=a[:,j]
    return frame


def test_saved_gradient_reconstruction_and_unknown_protection():
    f=prediction(); checks,*_=verify_filter_gradients(f)
    assert checks['logit_gradient_max_error'] < 1e-14
    f.loc[3,'g_event']=.1
    with pytest.raises(AssertionError,match='gradient'):
        verify_filter_gradients(f)


def test_saved_sigmoid_cannot_be_replaced_with_composed_score():
    f=prediction(); f['p_event']*=.5
    with pytest.raises(AssertionError,match='sigmoid'):
        verify_filter_gradients(f)


@pytest.mark.parametrize('y',[[[0,0,-1,-1,-1]], [[1,0,1,1,1]], [[-1,1,-1,-1,-1]]])
def test_independent_conditional_mask_guard(y):
    with pytest.raises(AssertionError):
        verify_hierarchy(y)


def selection_fixture():
    ids=np.arange(8); seed=17; rng=np.random.default_rng(seed)
    order=rng.permutation(ids); stream=hashlib.sha256()
    history=[dict(update=0,validation_nll=.8,examples_seen=0,sampled_root_stream_sha256=stream.hexdigest())]
    for update,nll in enumerate([.7,.69995,.68],1):
        stream.update(np.asarray(order[(update-1)*2:update*2],dtype='<i8').tobytes())
        history.append(dict(update=update,validation_nll=nll,examples_seen=update*2,sampled_root_stream_sha256=stream.hexdigest()))
    training=dict(best_update=3,best_validation_nll=.68,updates_completed=3,examples_seen=6,sampled_root_stream_sha256=stream.hexdigest())
    config=dict(filter=dict(patience_checks=5,min_delta=.0001,max_updates=3,validate_every=1,batch_size=2))
    return history,training,config,ids,seed


def test_selection_replays_min_delta_and_sampler_and_rejects_tamper():
    h,t,c,ids,seed=selection_fixture()
    assert verify_selection(h,t,c,ids,seed)['selected_update']==3
    t['best_update']=2
    with pytest.raises(AssertionError,match='checkpoint'):
        verify_selection(h,t,c,ids,seed)
    t['best_update']=3;h[1]['sampled_root_stream_sha256']='bad'
    with pytest.raises(AssertionError,match='stream'):
        verify_selection(h,t,c,ids,seed)


def test_min_delta_eligible_initial_beats_naive_argmin_assumption():
    h,t,c,ids,seed=selection_fixture()
    h[1]['validation_nll']=.79999; h[2]['validation_nll']=.79998; h[3]['validation_nll']=.79997
    t.update(best_update=0,best_validation_nll=.8)
    assert verify_selection(h,t,c,ids,seed)['selected_update']==0


def window_fixture():
    f=pd.DataFrame(dict(date=['2026-08-24']*8,t_ns=np.arange(8)*5_000_000_000,
        physical_support_available=[True]*6+[False,True],physical_segment_token=[0]*6+[-1,1],
        feature_available=[True,True,False,True,True,True,False,True],y_crossing=[0]*8))
    raw=np.arange(16,dtype=np.float32).reshape(8,2)
    scaler=dict(mean=[0.,0.],scale=[1.,1.],cap=10.)
    return f,raw,scaler


def test_restored_context_preserves_holes_and_support_zero_and_prefix():
    f,raw,s=window_fixture()
    b=restored_batch(f,raw,s,[0,1],[3,6,7],length=4)
    assert b[0,:, -1].tolist()==[1,1,1,1]
    assert b[0,2,:].tolist()==[0,0,0,0,1]
    assert np.all(b[1]==0)
    assert b[2,:, -1].tolist()==[0,0,0,1]
    check=prefix_check(f,raw,['a','b'],['a','b'],s)
    assert check['prefix_equal']


def test_independent_window_matches_frozen_store_on_gaps_and_unknown():
    from spectrum.research_v3.learning import FeatureStore
    f,raw,s=window_fixture();f['legacy_segment_token']=np.arange(len(f));s['feature_names']=['a','b']
    store=FeatureStore(f,raw,['a','b'],{'micro':['a','b']})
    expected=store.batch(np.arange(8),family='micro',length=7,cadence_seconds=5,scaler=s,context_mode='restored')
    np.testing.assert_array_equal(restored_batch(f,raw,s,[0,1],np.arange(8),length=7),expected)


def test_exact_feature_whitelist_rejects_future_metadata_channel():
    micro=['v'+str(i) for i in range(70)]
    schema=dict(names=expected_feature_names(micro),features=182,base_frozen=True,long_day_context=False)
    assert verify_feature_schema(schema,micro)['future_outcome_id_columns']==0
    schema['names'][-1]='first_qualifying_event_id_L'
    with pytest.raises(AssertionError,match='whitelist'):
        verify_feature_schema(schema,micro)


def test_independent_functional_tcn_replay_matches_public_forward():
    from spectrum.research.models import build_model
    torch.manual_seed(8)
    config=dict(architecture='tcn',hidden=4,layers=2,kernel_size=3,dropout=.05)
    model=build_model(5,config).eval()
    x=torch.randn(7,64,5)
    with torch.no_grad():
        h,z,p=tcn_forward(x,model.state_dict(),config)
        torch.testing.assert_close(z,model(x),rtol=0,atol=0)
        torch.testing.assert_close(p.sum(1),torch.ones(7),atol=1e-7,rtol=1e-6)
        assert h.shape==(7,4)


def test_independent_filter_forward_matches_saved_network():
    from spectrum.research_v5.learning import FilterMLP
    model=FilterMLP(3,4,2,.05).eval()
    x=np.arange(18,dtype=np.float32).reshape(6,3)
    s=dict(mean=[2.,4.,6.],std=[1.,2.,3.])
    with torch.no_grad():
        expected=model(torch.tensor(np.clip((x-np.array(s['mean']))/s['std'],-10,10),dtype=torch.float32))
        torch.testing.assert_close(filter_forward(x,s,model.state_dict(),2),expected,atol=0,rtol=0)


def test_train_only_scaler_reconstruction_rejects_heldout_or_known_only_stats():
    x=np.array([[0,1],[10,2],[20,3],[1000,1000]],np.float32);ids=np.array([0,1,2])
    values=x[ids].astype(float)
    s=dict(mean=values.mean(0).tolist(),std=values.std(0).tolist(),finite_count=[3,3],
           fitted_rows=3,fitted_root_indices_sha256=array_sha(ids))
    assert verify_scaler(x,ids,s)['train_rows']==3
    s['mean']=x.mean(0).tolist()
    with pytest.raises(AssertionError,match='mean'):
        verify_scaler(x,ids,s)


def test_receipt_requires_complete_and_unchanged_payload_and_base(tmp_path):
    run=tmp_path/'run';run.mkdir();base=tmp_path/'base';base.mkdir()
    source=tmp_path/'source.py';source.write_text('pass')
    (run/'summary.json').write_text('{}');(base/'checkpoint.pt').write_bytes(b'frozen')
    b=dict(root=str(base),files={'checkpoint.pt':sha(base/'checkpoint.pt')})
    receipt=dict(status='COMPLETE',sources={str(source):sha(source)},outputs={'summary.json':sha(run/'summary.json')},base=b)
    (run/'receipt.json').write_text(json.dumps(receipt))
    assert verify_receipt(run,b)['base_files']==1
    (base/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError,match='base changed'):
        verify_receipt(run,b)


def target_fixture():
    f=pd.DataFrame(dict(date=['2026-08-24'],t_ns=[1000000000],entry_ns=[1100000000],entry_ordinal=[2],
        y_crossing=[1],G_event=[2.],R_cost2=[1.],R_cost4=[2.],R_cost6=[3.],
        first_qualifying_ns_L=[2000000000],first_qualifying_ns_S=[0],
        first_qualifying_ordinal_L=[4],first_qualifying_ordinal_S=[-1],
        first_qualifying_event_id_L=['A'],first_qualifying_event_id_S=['']))
    rows=[]
    for scale in (1,3,6):
        for role in range(3):
            good=(scale,role)==(1,0)
            rows.append(dict(t_ns=1000000000,scale=scale,role=role,ready=True,cross_ordinal=4 if good else -1,
                cross_ns=2000000000 if good else 0,event_id='A' if good else '',
                mark_L=2.2 if good else np.nan,mark_S=-2.8 if good else np.nan,
                mae_L=-2. if good else np.nan,mae_S=-2.8 if good else np.nan))
    return f,pd.DataFrame(rows)


def test_independent_risk_labels_equality_and_wrong_side_and_deadline():
    f,m=target_fixture()
    y,_=independently_reconstruct_targets(f,m,np.array([1]))
    assert y.tolist()==[[1,1,0,0,1]]
    y,_=independently_reconstruct_targets(f,m,np.array([-1]))
    assert y.tolist()==[[1,0,-1,-1,-1]]
    due=int(f.entry_ns.iloc[0])+300_000_000_000
    f.loc[0,'first_qualifying_ns_L']=due;m.loc[m.event_id=='A','cross_ns']=due
    y,w=independently_reconstruct_targets(f,m,np.array([1]))
    assert y.tolist()==[[1,1,-1,-1,-1]] and w['at_deadline'].tolist()==[True]


def test_independent_map_rejects_forged_positive_teacher():
    f,m=target_fixture();m.loc[m.event_id=='A','mark_L']=1.
    with pytest.raises(AssertionError,match='crossing labels'):
        independently_reconstruct_targets(f,m,np.array([1]))


def test_candidate_metrics_keep_unknown_and_conditional_denominators_separate():
    y=np.array([[0,-1,-1,-1,-1],[1,0,-1,-1,-1],[1,1,1,1,1],[-1]*5],float)
    bp=np.array([[.8,.1,.1,0],[.5,.4,.1,0],[.5,.4,.1,0],[.1,.8,.1,0]])
    result=candidate_diagnostics(y,bp,np.full((4,5),.5),np.zeros((4,5)),{'test':np.arange(4)})['by_role']['test']
    assert result['candidate_roots']==3 and result['unknown_candidates']==1
    assert result['candidate_head_metrics']['heads'][HEADS[0]]['n']==2
    assert result['candidate_head_metrics']['heads'][HEADS[1]]['n']==2
    assert result['candidate_head_metrics']['heads'][HEADS[2]]['n']==1
    assert result['event_coverage']==1


def test_clipping_ignores_train_constant_channels_but_reports_their_drift():
    x=np.array([[0,1],[2,1],[20,30]],np.float32)
    split={'train':np.array([0,1]),'test':np.array([2])}
    base=dict(mean=[0,0],scale=[1,1],constant_columns=[1])
    fs=dict(mean=[1,1],std=[1,1])
    result=clipping_diagnostics(x,x,split,base,[0,1],fs,['variable','train_constant'])
    assert result['filter_constant_channels']==1
    assert result['by_role']['test']['filter_all']['clipped_cells']==2
    assert result['by_role']['test']['filter_nonconstant']['clipped_cells']==1
    assert result['by_role']['test']['base_observed_nonconstant']['observed_cells']==1


def test_risk_monotonicity_reports_actual_predictions_without_repair():
    q=np.array([[.5,.5,.7,.6,.8],[.5,.5,.2,.4,.3],[.5,.5,.2,.4,.8]])
    y=np.array([[1,1,0,0,1],[0,-1,-1,-1,-1],[1,1,1,1,1]])
    saved=q.copy();result=risk_monotonicity(q,y,{'test':np.arange(3)})
    whole=result['by_role']['test']['all_admitted'];known=result['by_role']['test']['risk_known_conditional']
    assert whole['any_strict_violation_fraction']==pytest.approx(2/3)
    assert known['roots']==2 and known['any_strict_violation_fraction']==.5
    assert whole['cost4_above_cost6']['maximum']==pytest.approx(.1)
    np.testing.assert_array_equal(q,saved)
