"""Preregistered V4 recipe and matched initialization/sampling invariants."""
import copy
from pathlib import Path

import pytest

from scripts.run_opportunity_v4 import (
    experiment_configs, read, save, source_receipt, validate_recipe, verify_paired_invariants,
)


def config():
    return read("configs/research/opportunity_pipeline_v4.json")


def test_recipe_is_four_fresh_restored_arms_and_deep_copies_config():
    source=config()
    validate_recipe(source)
    arms=list(experiment_configs(source))
    assert [(a["seed"],a["training"]["weighting"]) for a in arms]==[
        (17,"uniform"),(17,"soft_crossing_id"),(29,"uniform"),(29,"soft_crossing_id")]
    assert len({a["run_id"] for a in arms})==4
    assert all(a["data"]["context_mode"]=="restored" for a in arms)
    assert all(a["data"]["root"]==source["data"]["root"] for a in arms)
    arms[0]["training"]["id_weighting"]["alpha"]=9.
    assert source["training"]["id_weighting"]["alpha"]==.5
    assert arms[1]["training"]["id_weighting"]["alpha"]==.5
    assert "previously inspected" in source["experiment"]["historical_exposure"]


@pytest.mark.parametrize("path,value",[
    (("training","id_weighting","alpha"),1.),
    (("training","id_weighting","min_weight"),.25),
    (("training","id_weighting","max_weight"),3.),
    (("training","learning_rate"),.003),
    (("training","max_updates"),100),
    (("training","weighting"),"crossing_id"),
    (("model","hidden"),64),
    (("loss","event_loss"),"focal"),
    (("execution","fee_roundtrip"),0.),
    (("event_query","gain_multiple"),3.),
    (("data","root"),"some-new-dataset"),
    (("data","context_mode"),"legacy"),
    (("splits","test"),["2026-08-03"]),
    (("experiment","seeds"),[17]),
    (("experiment","contexts"),["legacy","restored"]),
    (("experiment","weightings"),["uniform","crossing_id"]),
    (("experiment","false_signal_protocol","fixed_thresholds"),[.1,.5]),
    (("experiment","false_signal_protocol","calibration_none_fpr","budget"),.1),
    (("experiment","false_signal_protocol","calibration_none_fpr","score_dtype"),"float32"),
    (("experiment","false_signal_protocol","calibration_none_fpr","selection"),"optimize on TEST"),
    (("experiment","false_signal_protocol","side_tie_break"),"SHORT"),
    (("experiment","false_signal_protocol","fixed_test_execution_stress","thresholds"),[.1,.9]),
])
def test_changed_recipe_is_rejected_before_any_fit(path,value):
    changed=config(); cursor=changed
    for field in path[:-1]: cursor=cursor[field]
    cursor[path[-1]]=value
    with pytest.raises(ValueError): validate_recipe(changed)


def paired_fixture(tmp_path):
    summaries=[]
    for arm in experiment_configs(config()):
        seed=arm["seed"]
        out=tmp_path/arm["run_id"];out.mkdir()
        save(out/"history.json",[{"update":0,"sampled_root_stream_sha256":"empty"},
                                  {"update":150,"sampled_root_stream_sha256":f"prefix-{seed}"}])
        summaries.append(dict(run_id=arm["run_id"],scaler_sha256="same-scaler",cohorts={"train":{"root_indices_sha256":"same-roots"}},
            training={key:f"{key}-{seed}" for key in ("train_metric_probe_sha256","gradient_probe_sha256","initial_state_sha256")}))
    return summaries


def test_paired_guard_checks_cohorts_initial_states_probes_and_sampled_prefixes(tmp_path):
    summaries=paired_fixture(tmp_path)
    verify_paired_invariants(summaries,tmp_path)
    changed=copy.deepcopy(summaries);changed[0]["scaler_sha256"]="other"
    with pytest.raises(ValueError,match="cohorts or scalers"):
        verify_paired_invariants(changed,tmp_path)
    for key in ("initial_state_sha256","train_metric_probe_sha256","gradient_probe_sha256"):
        changed=copy.deepcopy(summaries);changed[0]["training"][key]="other"
        with pytest.raises(ValueError,match="seed-level"):
            verify_paired_invariants(changed,tmp_path)
    path=tmp_path/summaries[0]["run_id"]/"history.json"
    history=read(path);history[1]["sampled_root_stream_sha256"]="other";save(path,history)
    with pytest.raises(ValueError,match="sampled TRAIN-root prefix"):
        verify_paired_invariants(summaries,tmp_path)
    with pytest.raises(ValueError,match="Four distinct"):
        verify_paired_invariants(summaries[:3],tmp_path)


def test_source_receipt_binds_wrapper_and_frozen_dependencies():
    manifest=dict(identity=dict(sources={"spectrum/research_v2/targets.py":"unused-here",
                                        "scripts/prepare_opportunity_v3.py":"unused-here"}))
    sources=source_receipt("configs/research/opportunity_pipeline_v4.json",manifest)
    for name in ("scripts/run_opportunity_v4.py","scripts/run_opportunity_v2.py",
                 "spectrum/research_v4/learning.py","spectrum/research_v3/learning.py",
                 "spectrum/research_v2/learning.py","spectrum/research_v2/evaluation.py",
                 "spectrum/research_v2/targets.py","spectrum/research/losses.py",
                 "spectrum/research/models.py","scripts/prepare_opportunity_v3.py",
                 "configs/research/opportunity_pipeline_v3.json","configs/research/opportunity_pipeline_v4.json"):
        assert str(Path(name).resolve()) in sources
        assert len(sources[str(Path(name).resolve())])==64
