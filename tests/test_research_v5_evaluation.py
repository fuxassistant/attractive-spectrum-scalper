import copy
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_opportunity_v5 import (hash_scores, score_channels, retention_threshold,
    economic_probabilities, alarm_metrics, head_metrics)
from scripts.run_opportunity_v5 import PNAMES, QNAMES, validate_recipe, read


def sample():
    n=6
    f=pd.DataFrame(dict(t_ns=np.arange(n,dtype=np.int64)*10**9,date='2026-08-24',
        feature_available=True,y_crossing=[0,1,2,3,-1,1],
        first_qualifying_event_id_L=['','L','','B','','L'],
        first_qualifying_event_id_S=['','','S','BS','','']))
    p=np.tile([.1,.6,.2,.1],(n,1))
    for j,name in enumerate(PNAMES):
        f[name]=p[:,j]
    for j,name in enumerate(QNAMES):
        f[name]=.7+j*.05
    f['side']=1
    return f


def test_hash_control_is_prefix_invariant_deterministic_and_label_free():
    t=np.arange(1000,dtype=np.int64)+1780000000000000000
    a=hash_scores(t,17)
    assert np.array_equal(a[:10],hash_scores(t[:10],17))
    assert np.array_equal(a,hash_scores(t,17))
    assert np.all((a>0)&(a<1))
    assert not np.array_equal(a,hash_scores(t,29))


def test_conservative_calibration_ties_never_exceed_budget():
    s=np.array([.9,.8,.8,.8,.1])
    rule=retention_threshold(s,np.ones(5,bool),.5)
    assert rule['allowed']==2 and rule['actual']==1
    assert rule['threshold']>.8
    assert retention_threshold(s,np.ones(5,bool),1)['actual']==5
    assert retention_threshold(s,np.zeros(5,bool),.5)['threshold']==1.01


def test_test_scores_cannot_change_cal_threshold():
    a=np.array([.1,.2,.3,.4,.5,.6]); use=np.arange(6)<3
    original=retention_threshold(a,use,.5)
    a[3:]=[-99,4,89]
    assert retention_threshold(a,use,.5)==original


def test_frozen_side_and_filter_scores_ignore_all_future_columns():
    f=sample(); original,side=score_channels(f,17)
    f['y_crossing']=-1; f['first_qualifying_event_id_L']='future mutation'
    f['maturity_ns']=2**60
    changed,side2=score_channels(f,17)
    assert np.array_equal(side,side2)
    for plan in original:
        for arm in original[plan]:
            assert np.array_equal(original[plan][arm],changed[plan][arm])
    f['side']=-1
    with pytest.raises(ValueError,match='changed frozen'):
        score_channels(f,17)


def test_score_adapter_preserves_positive_score_and_chosen_side():
    from scripts.audit_economics_v4 import prepare_probabilities
    s=np.array([.2,.7,1.]); side=np.array([1,-1,-1])
    score,actual=prepare_probabilities(economic_probabilities(s,side),3)
    assert np.array_equal(score,s) and np.array_equal(actual,side)


def test_none_wrong_unknown_risk_are_separate_and_future_unknown_not_false():
    f=sample(); candidate=np.ones(6,bool); alert=candidate.copy()
    result=alarm_metrics(f,np.ones(6,int),np.array([-1,0,-1,1,-1,1]),candidate,alert,candidate)
    assert result['alerts']==5 and result['unknown_alerts']==1
    assert result['none_false_positives']==1 and result['wrong_side_alerts']==1
    assert result['true_positives']==3
    assert result['safe_correct_alerts']==2 and result['unsafe_correct_alerts']==1
    assert result['precision']==3/5 and result['safe_precision']==2/5
    # Unavailable root remains unavailable even if truth is favorable.
    f.loc[1,'feature_available']=False
    assert alarm_metrics(f,np.ones(6,int),np.ones(6),candidate,alert,candidate)['alerts']==4


def test_conditional_head_metrics_mask_unknown():
    y=np.array([1,0,-1]); p=np.array([.8,.3,.99])
    a=head_metrics(y,p); p[2]=.01
    assert head_metrics(y,p)==a
    assert a['count']==2 and a['positives']==1


def test_v5_requires_original_target_bce_and_uniform_frozen_bases():
    c=read('configs/research/opportunity_pipeline_v5.json')
    validate_recipe(c)
    changed=copy.deepcopy(c); changed['event_query']['gain_multiple']=3
    with pytest.raises(ValueError,match='Frozen base recipe'):
        validate_recipe(changed)
    changed=copy.deepcopy(c); changed['loss']['event_loss']='focal'
    with pytest.raises(ValueError,match='Frozen base recipe'):
        validate_recipe(changed)
    changed=copy.deepcopy(c); changed['filter']['base_updates']=1
    with pytest.raises(ValueError,match='frozen base checkpoints'):
        validate_recipe(changed)
