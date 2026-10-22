"""Hierarchical filter loss, causal preprocessing, selection and reproducibility."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from spectrum.research_v5.learning import (
    HEAD_NAMES, HEAD_WEIGHTS, FilterMLP, evaluate, fit_scaler, logit_gradients,
    masked_bce, predict, probability_metrics, train_filter, transform, validate_labels,
)


def example_labels(n=48):
    rows = np.array([[0,-1,-1,-1,-1], [1,0,-1,-1,-1], [1,1,0,1,1],
                     [1,1,0,0,1], [1,1,1,1,1], [-1,-1,-1,-1,-1]], np.float32)
    return np.tile(rows, (int(np.ceil(n / len(rows))), 1))[:n]


def tiny_config(**overrides):
    c = dict(hidden=4, layers=2, dropout=.05, batch_size=7, learning_rate=.01,
        weight_decay=.0001, gradient_clip=1., max_updates=6, validate_every=2,
        patience_checks=4, min_delta=0., cpu_threads=1, device="cpu",
        train_metric_probe=16, gradient_probe=16, head_names=list(HEAD_NAMES), base_updates=0)
    c.update(overrides)
    return dict(filter=c)


def example_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 7)).astype(np.float32)
    y = example_labels()
    return x, y, np.arange(24), np.arange(24, 36)


def test_masked_objective_uses_common_root_count_not_known_head_count():
    y = example_labels(6)
    z = torch.tensor([[.1,-.2,.3,-.4,.5]] * 6, dtype=torch.float64, requires_grad=True)
    yy = torch.tensor(y, dtype=z.dtype)
    pieces = F.binary_cross_entropy_with_logits(z, yy.clamp_min(0), reduction="none")
    expected = (pieces * (yy >= 0) * z.new_tensor(HEAD_WEIGHTS)).sum() / 5
    torch.testing.assert_close(masked_bce(z, yy), expected, rtol=0, atol=0)
    roots = masked_bce(z, yy, "none")
    assert roots[-1].item() == 0
    torch.testing.assert_close(roots.sum() / 5, expected, rtol=0, atol=0)
    # Three known risk heads collectively have unit loss weight, not weight 3.
    zz = torch.zeros((1, 5), dtype=torch.float64)
    assert masked_bce(zz, torch.ones_like(zz)).item() == pytest.approx(3 * np.log(2))


def test_actual_gradients_match_masked_sigmoid_and_unknown_zero():
    y = example_labels()
    z = np.random.default_rng(2).normal(0, 20, y.shape)
    actual, error = logit_gradients(z, y)
    assert error < 1e-14 and np.isfinite(actual).all()
    assert np.all(actual[y < 0] == 0)
    assert np.max(np.abs(actual[:, 2:])) <= 1 / 3
    zz = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    loss = masked_bce(zz, torch.tensor(y, dtype=torch.float64))
    loss.backward()
    np.testing.assert_allclose(zz.grad.numpy(), actual / (y[:, 0] >= 0).sum(), atol=1e-16, rtol=1e-14)


def test_partial_risk_mask_does_not_renormalize_remaining_risk_heads():
    z = np.zeros((1, 5))
    y = np.array([[1, 1, -1, 1, -1]], np.float32)
    g, _ = logit_gradients(z, y)
    np.testing.assert_allclose(g, [[-.5, -.5, 0, -1/6, 0]], atol=1e-16)


def test_all_unknown_batch_has_connected_exact_zero_gradient():
    z = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    loss = masked_bce(z, torch.full_like(z, -1))
    loss.backward()
    assert loss.item() == 0 and torch.equal(z.grad, torch.zeros_like(z))


@pytest.mark.parametrize("labels", [
    [[0,1,-1,-1,-1]], [[-1,0,-1,-1,-1]], [[1,0,1,-1,-1]],
    [[1,-1,-1,0,-1]], [[1,1,2,0,1]], [[1,1,np.nan,0,1]], [[0,-1]],
])
def test_invalid_hierarchy_rejected(labels):
    with pytest.raises(ValueError):
        validate_labels(np.array(labels))


def test_scaler_is_train_only_label_independent_and_missing_safe():
    x = np.array([[1,2,np.nan], [3,2,np.nan], [10000,-10000,4]], np.float32)
    scaler = fit_scaler(x, np.array([0,1]))
    np.testing.assert_allclose(scaler["mean"], [2,2,0])
    np.testing.assert_allclose(scaler["std"], [1,1,1])
    assert scaler["finite_count"] == [2,2,0]
    changed = x.copy()
    changed[-1] = [-7,8,-9]
    assert fit_scaler(changed, np.array([0,1])) == scaler
    z = transform(x, scaler)
    np.testing.assert_array_equal(z[:2], [[-1,0,0], [1,0,0]])
    assert np.isfinite(z).all() and np.max(np.abs(z)) <= 10


def test_metric_composite_is_same_proper_loss_and_empty_heads_are_unknown():
    y = example_labels()
    z = np.random.default_rng(2).normal(size=y.shape)
    p = np.exp(-np.logaddexp(0, -z))
    metric = probability_metrics(y, p, z)
    loss = masked_bce(torch.tensor(z), torch.tensor(y, dtype=torch.float64)).item()
    assert metric["composite_nll"] == pytest.approx(loss, abs=1e-14)
    empty = probability_metrics(np.array([[0,-1,-1,-1,-1]], np.float32), np.full((1,5),.5))
    assert empty["heads"][HEAD_NAMES[1]]["n"] == 0
    assert empty["heads"][HEAD_NAMES[1]]["nll"] is None
    saturated = probability_metrics(np.array([[0,-1,-1,-1,-1]], np.float32), np.ones((1,5)), np.full((1,5),100.))
    assert saturated["composite_nll"] == 100.  # Stable exact logit NLL, no probability clipping ceiling.


def test_heldout_features_and_labels_cannot_affect_fit(tmp_path):
    x, y, train, val = example_data()
    first = train_filter(x, y, train, val, tiny_config(), tmp_path / "a", 17)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[36:] = 5000
    changed_y[36:] = [1,1,0,1,1]
    second = train_filter(changed_x, changed_y, train, val, tiny_config(), tmp_path / "b", 17)
    a, b = first["summary"], second["summary"]
    assert a["scaler_sha256"] == b["scaler_sha256"]
    for key in ("initial_state_sha256", "selected_state_sha256", "sampled_root_stream_sha256",
                "train_metric_probe_sha256", "gradient_probe_sha256", "best_update"):
        assert a["training"][key] == b["training"][key]
    assert first["history"] == second["history"]
    np.testing.assert_array_equal(first["predictions"][:36], second["predictions"][:36])
    assert set(a["evaluation"]) == {"train", "validation"}
    checkpoint = torch.load(tmp_path / "a/checkpoint.pt", weights_only=False)
    model = FilterMLP(x.shape[1], 4, 2, .05)
    model.load_state_dict(checkpoint["state_dict"])
    scaler = json.loads((tmp_path / "a/scaler.json").read_text())
    # Different batching must not change prediction semantics (no batch statistics).
    np.testing.assert_allclose(predict(model, x, scaler, batch_size=3), first["predictions"], atol=1e-7, rtol=1e-6)
    saved = pd.read_parquet(tmp_path / "a/predictions.parquet")
    assert len(saved) == len(x)
    for j, head in enumerate(HEAD_NAMES):
        np.testing.assert_array_equal(saved["p_" + head], first["predictions"][:, j])
        assert np.all(saved.loc[y[:,j] < 0, "g_" + head] == 0)
    assert first["gradient"]["full_clock_actual_vs_analytical_max_abs_error"] < 1e-14
    assert first["summary"]["cohorts"]["train"]["admitted_roots"] == len(train)
    assert first["summary"]["cohorts"]["train"]["known_event_roots"] == 20


def test_validation_changes_selection_but_not_training_stream_or_scaler(tmp_path):
    x = np.zeros((32, 3), np.float32)
    y = np.full((32,5), -1, np.float32)
    y[:,0] = 0
    train, val = np.arange(16), np.arange(16,24)
    c = tiny_config(dropout=0., learning_rate=.05, max_updates=4, validate_every=1, patience_checks=10)
    good = train_filter(x, y, train, val, c, tmp_path / "match", 17)
    bad_y = y.copy()
    bad_y[val,0] = 1
    bad = train_filter(x, bad_y, train, val, c, tmp_path / "opposite", 17)
    assert good["summary"]["training"]["best_update"] > 0
    assert bad["summary"]["training"]["best_update"] == 0
    assert good["summary"]["scaler_sha256"] == bad["summary"]["scaler_sha256"]
    assert good["summary"]["training"]["sampled_root_stream_sha256"] == bad["summary"]["training"]["sampled_root_stream_sha256"]
    assert [r["optimization_loss"] for r in good["history"]] == [r["optimization_loss"] for r in bad["history"]]


def test_initial_candidate_and_saved_initial_gradient_are_real(tmp_path):
    x, y, train, val = example_data()
    result = train_filter(x, y, train, val, tiny_config(min_delta=100., patience_checks=1), tmp_path, 29)
    tr = result["summary"]["training"]
    assert tr["best_update"] == 0 and tr["updates_completed"] == 2
    assert tr["selected_checkpoint"] == "untrained_network"
    assert tr["initial_state_sha256"] == tr["selected_state_sha256"]
    assert result["gradient"]["before"] == result["gradient"]["selected"]
    with pytest.raises(ValueError, match="already exist"):
        train_filter(x, y, train, val, tiny_config(), tmp_path, 29)


def test_train_unknown_rows_still_fit_scaler_but_never_enter_optimization(tmp_path):
    x, y, train, val = example_data()
    x[5] = 100
    result = train_filter(x, y, train, val, tiny_config(max_updates=1), tmp_path, 17)
    scaler = json.loads((tmp_path / "scaler.json").read_text())
    np.testing.assert_allclose(scaler["mean"], x[train].astype(np.float64).mean(0), rtol=0, atol=1e-15)
    probe = result["summary"]["training"]["train_metric_probe_indices"]
    assert np.all(y[probe,0] >= 0)
    assert 5 not in result["gradient"]["before"]["root_indices"]


@pytest.mark.parametrize("train,val", [(np.arange(10),np.arange(9,20)), (np.arange(10,20),np.arange(10)),
                                       (np.array([0,0]),np.arange(10,20))])
def test_split_overlap_reversal_duplicate_rejected(tmp_path, train, val):
    x, y, _, _ = example_data()
    with pytest.raises(ValueError):
        train_filter(x, y, train, val, tiny_config(), tmp_path, 17)


def test_nonfinite_parameters_and_input_fail_closed(tmp_path):
    x, y, train, val = example_data()
    c = tiny_config(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        train_filter(x, y, train, val, c, tmp_path, 17)
    x[0,0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        train_filter(x, y, train, val, tiny_config(), tmp_path, 17)
