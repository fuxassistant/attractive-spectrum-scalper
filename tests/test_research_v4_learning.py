"""Bounded soft weights, held-out isolation and exact fresh uniform control."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from spectrum.research.losses import classification_loss
from spectrum.research_v3 import learning as v3
from spectrum.research_v4.learning import (
    FIXED_WEIGHT_PARAMETERS, FeatureStore, bounded_class_normalize,
    compute_train_weights, logit_gradients, train_run, weighted_classification_loss,
)


LOSS=dict(formulation="factorized", event_loss="bce")


def example_store():
    y=np.array([0,0,1,1,1,2,2,3,3,-1,1,2],np.int64)
    t=pd.Timestamp("2026-06-01T08:00:00Z").value+np.arange(len(y),dtype=np.int64)*5_000_000_000
    f=pd.DataFrame(dict(date=["2026-06-01"]*len(y),t_ns=t,feature_available=True,physical_support_available=True,
        legacy_segment_token=np.zeros(len(y),np.int64),physical_segment_token=np.zeros(len(y),np.int64),y_crossing=y,
        first_qualifying_event_id_L=["","","A","A","B","","","A","B","unknown","A",""],
        first_qualifying_event_id_S=["","","","","","A","C","C","D","unknown","","A"]))
    return FeatureStore(f,np.arange(len(y)*2,dtype=np.float32).reshape(-1,2),["one","two"],{"micro":["one","two"]})


def tiny_store():
    count=32
    y=np.resize(np.array([0,1,2,3],np.int64),count)
    dates=np.repeat([f"2026-06-0{d+1}" for d in range(4)],8)
    t=np.concatenate([pd.Timestamp(date+"T08:00:00Z").value+np.arange(8,dtype=np.int64)*5_000_000_000 for date in dates[::8]])
    f=pd.DataFrame(dict(date=dates,t_ns=t,feature_available=True,physical_support_available=True,
        legacy_segment_token=np.zeros(count,np.int64),physical_segment_token=np.zeros(count,np.int64),y_crossing=y,
        first_qualifying_event_id_L=np.where(np.isin(y,[1,3]),"L-common",""),
        first_qualifying_event_id_S=np.where(np.isin(y,[2,3]),"S-common","")))
    f.loc[1,"first_qualifying_event_id_L"]="L-rare"
    raw=np.arange(count*2,dtype=np.float32).reshape(count,2)/10
    return FeatureStore(f,raw,["one","two"],{"micro":["one","two"]})


def tiny_config(weighting="soft_crossing_id"):
    return dict(schema="scientific_opportunity_pipeline_v4",run_id="unit",seed=17,
        data=dict(family="micro",sequence_length=7,cadence_seconds=5,context_mode="restored"),
        splits={role:[f"2026-06-0{i+1}"] for i,role in enumerate(("train","validation","calibration","test"))},
        model=dict(architecture="tcn",hidden=4,layers=1,kernel_size=2,dropout=0.),loss=LOSS,
        training=dict(device="cpu",batch_size=4,learning_rate=.01,weight_decay=0.,gradient_clip=1.,
                      max_updates=4,validate_every=2,patience_checks=2,min_delta=0.,cpu_threads=1,
                      weighting=weighting,id_weighting=copy.deepcopy(FIXED_WEIGHT_PARAMETERS),
                      train_metric_probe=8,gradient_probe=8))


def test_bounded_equation_preserves_mass_without_post_clip_renormalization():
    raw=np.array([1e-8]+[.1]*8+[1.])
    w,a,residual=bounded_class_normalize(raw)
    np.testing.assert_allclose(w,np.clip(a*raw,.5,2.),rtol=0,atol=0)
    assert abs(w.sum()-len(raw))<1e-12 and abs(residual)<1e-12
    assert w.min()==.5 and w.max()==2.
    skew=np.array([.001]*9+[1.])
    naive=np.clip(skew/skew.mean(),.5,2.)
    naive*=len(naive)/naive.sum()
    assert naive.max()>2.  # The rejected clip-then-renormalize procedure.
    bounded,_,_=bounded_class_normalize(skew)
    assert bounded.min()>=.5 and bounded.max()<=2.
    assert bounded.sum()==pytest.approx(len(skew),abs=1e-12)


@pytest.mark.parametrize("raw",[np.array([.02]),np.ones(8)*.00001,np.geomspace(1e-6,1,101)])
def test_bounded_normalization_handles_degenerate_and_wide_classes(raw):
    w,_,_=bounded_class_normalize(raw)
    assert np.isfinite(w).all() and np.all((w>=.5)&(w<=2.))
    assert w.sum()==pytest.approx(len(raw),abs=1e-11)


def test_soft_both_formula_pools_memberships_and_preserves_every_class_mass():
    store=example_store()
    w,receipt=compute_train_weights(store,np.arange(9),"soft_crossing_id")
    # A appears four times across both sides; B,C twice and D once.
    raw=np.array([1,1,.5,.5,2**-.5,.5,2**-.5,(.5+2**-.5)/2,(2**-.5+1)/2])
    y=store.frame.y_crossing.to_numpy()[:9]
    for value,name in enumerate(v3.STATE_NAMES):
        mask=y==value
        a=receipt["class_normalizers"][name]
        np.testing.assert_allclose(w[:9][mask],np.clip(a*raw[mask],.5,2.))
        assert w[:9][mask].sum()==pytest.approx(mask.sum(),abs=1e-12)
        assert receipt["by_joint_class"][name]["raw"]["mean"]==pytest.approx(raw[mask].mean())
    assert receipt["parameters"]==FIXED_WEIGHT_PARAMETERS
    assert receipt["positive_side_memberships"]==9
    assert receipt["unique_positive_crossing_ids"]==4
    assert receipt["solver"]["post_clip_renormalization"] is False
    np.testing.assert_array_equal(w[:2],1)
    np.testing.assert_array_equal(w[9:],0)
    _,old=v3.compute_train_weights(store,np.arange(9),"crossing_id")
    assert receipt["crossing_id_count_table_sha256"]==old["crossing_id_count_table_sha256"]


def test_unknowns_and_heldout_id_multiplicity_do_not_change_train_weights():
    source=example_store()
    a,receipt_a=compute_train_weights(source,np.arange(9),"soft_crossing_id")
    changed=source.frame.copy()
    changed.loc[9:,"y_crossing"]=3
    changed.loc[9:,"first_qualifying_event_id_L"]="A"
    changed.loc[9:,"first_qualifying_event_id_S"]="A"
    other=FeatureStore(changed,source.raw.copy(),source.feature_names,source.groups)
    b,receipt_b=compute_train_weights(other,np.arange(9),"soft_crossing_id")
    np.testing.assert_array_equal(a,b)
    assert receipt_a==receipt_b
    with pytest.raises(ValueError,match="known feature-available"):
        compute_train_weights(source,np.arange(10),"soft_crossing_id")
    source.frame.loc[2,"first_qualifying_event_id_L"]=""
    with pytest.raises(ValueError,match="nonempty crossing ID"):
        compute_train_weights(source,np.arange(9),"soft_crossing_id")


def test_uniform_equals_original_loss_and_unknown_has_zero_gradient():
    z=torch.tensor([[.4,.1,-.3,.7],[-.2,.3,.1,-.4],[.7,-.2,.8,.4],[-.4,.2,.5,.1],[.2,.1,-.1,.3]],dtype=torch.float64)
    y=torch.tensor([0,1,2,3,-1])
    w=torch.tensor([1,1,1,1,0],dtype=torch.float64)
    actual=weighted_classification_loss(z,y,w,LOSS)
    torch.testing.assert_close(actual,classification_loss(z,y,LOSS),rtol=0,atol=0)
    gradient,error=logit_gradients(z.numpy(),y.numpy(),LOSS,w.numpy())
    assert error<1e-12
    np.testing.assert_array_equal(gradient[-1],0)
    np.testing.assert_array_equal(gradient[0,1:],0)
    weights,receipt=compute_train_weights(example_store(),np.arange(9),"uniform")
    np.testing.assert_array_equal(weights[:9],1)
    assert receipt["overall"]["kish_effective_sample_size"]==9


def test_soft_loss_has_original_known_root_denominator_and_actual_gradients():
    z=np.array([[.4,.1,-.3,.7],[-.2,.3,.1,-.4],[.7,-.2,.8,.4],[-.4,.2,.5,.1]],np.float64)
    y=np.array([0,1,2,3],np.int64)
    w=np.array([1,.5,1.25,2.])
    losses=classification_loss(torch.tensor(z),torch.tensor(y),LOSS,reduction="none").numpy()
    result=weighted_classification_loss(torch.tensor(z),y,w,LOSS)
    assert float(result)==pytest.approx(np.dot(w,losses)/4)
    assert float(result)!=pytest.approx(np.dot(w,losses)/w.sum())
    gradient,error=logit_gradients(z,y,LOSS,w)
    assert error<1e-12
    eps=1e-6
    for row,column in np.ndindex(z.shape):
        plus,minus=z.copy(),z.copy(); plus[row,column]+=eps; minus[row,column]-=eps
        numeric=(weighted_classification_loss(torch.tensor(plus),y,w,LOSS,"sum")-
                 weighted_classification_loss(torch.tensor(minus),y,w,LOSS,"sum"))/(2*eps)
        assert gradient[row,column]==pytest.approx(float(numeric),abs=2e-9)


def test_fresh_uniform_control_matches_frozen_v3_bit_for_bit(tmp_path):
    store=tiny_store(); config=tiny_config("uniform")
    old_config=copy.deepcopy(config); old_config["schema"]="scientific_opportunity_pipeline_v3"
    old_config["training"].pop("id_weighting")
    old=v3.train_run(store,old_config,tmp_path/"v3")
    fresh=train_run(store,config,tmp_path/"v4")
    assert old["history"]==fresh["history"]
    assert old["summary"]["cohorts"]==fresh["summary"]["cohorts"]
    assert old["summary"]["scaler_sha256"]==fresh["summary"]["scaler_sha256"]
    pd.testing.assert_frame_equal(old["predictions"],fresh["predictions"],check_exact=True)
    a=torch.load(tmp_path/"v3/checkpoint.pt",weights_only=False,map_location="cpu")
    b=torch.load(tmp_path/"v4/checkpoint.pt",weights_only=False,map_location="cpu")
    for name in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][name],b["state_dict"][name],atol=0,rtol=0)


def test_soft_checkpoint_is_independent_of_heldout_ids_and_cal_test_values(tmp_path):
    source=tiny_store(); config=tiny_config()
    a=train_run(source,config,tmp_path/"a")
    frame=source.frame.copy()
    frame.loc[8:,"first_qualifying_event_id_L"]="heldout-change"
    frame.loc[8:,"first_qualifying_event_id_S"]="heldout-change"
    frame.loc[16:,"y_crossing"]=np.resize([3,2,1,0],16)
    raw=source.raw.copy(); raw[16:]+=1e6
    other=FeatureStore(frame,raw,source.feature_names,source.groups)
    b=train_run(other,config,tmp_path/"b")
    assert a["history"]==b["history"]
    assert a["summary"]["weight_diagnostics"]==b["summary"]["weight_diagnostics"]
    assert a["summary"]["scaler_sha256"]==b["summary"]["scaler_sha256"]
    assert a["summary"]["training"]["best_update"]==b["summary"]["training"]["best_update"]
    selected=min(a["history"],key=lambda row:row["validation_nll"])
    assert a["summary"]["training"]["best_update"]==selected["update"]
    nontrain=a["predictions"]["split"]!="train"
    assert a["predictions"].loc[nontrain,"optimization_weight_train_only"].isna().all()
