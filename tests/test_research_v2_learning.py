"""Lazy causal data, shared preprocessing, and held-out selection boundaries."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spectrum.research_v2.learning import (
    FeatureStore, fit_scaler, load_dataset, logit_gradients, split_indices,
    train_run, _prediction_frame,
)
from spectrum.research.losses import classification_loss


def store_for_tests(count=12):
    raw = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10
    start = pd.Timestamp("2026-04-01T08:00:00Z").value
    frame = pd.DataFrame(dict(date=["2026-04-01"] * count,
        t_ns=start + np.arange(count, dtype=np.int64) * 5_000_000_000,
        feature_available=np.ones(count, bool), segment_token=np.zeros(count, np.int64),
        y_crossing=np.resize(np.array([0, 1, 2, 3], np.int64), count)))
    return FeatureStore(frame, raw, ["geometry", "flow", "context"],
                        {"geometry": ["geometry"], "micro": ["geometry", "flow"],
                         "rich": ["geometry", "flow", "context"]})


def test_lazy_windows_reset_only_on_causal_boundaries_and_do_not_read_future():
    source = store_for_tests(9)
    f, x = source.frame.copy(), source.raw.copy()
    f.loc[1, "y_crossing"] = -1
    f.loc[3, "feature_available"] = False
    f.loc[5:, "segment_token"] = 1
    f.loc[6:, "t_ns"] += 5_000_000_000
    f.loc[7:, "date"] = "2026-04-02"
    store = FeatureStore(f, x, source.feature_names, source.groups)
    scaler = fit_scaler(store, np.arange(3))
    batch = store.batch(np.array([2, 3, 4, 5, 6, 7]), family="rich", length=4,
                        cadence_seconds=5, scaler=scaler)
    assert batch.shape == (6, 4, 7)
    # Unknown Y at row 1 does not break a feature history.
    np.testing.assert_array_equal(batch[:, :, -1].sum(1), [3, 0, 1, 1, 1, 1])
    changed = x.copy()
    changed[3:] = 1e8
    future_store = FeatureStore(f, changed, source.feature_names, source.groups)
    np.testing.assert_array_equal(batch[0], future_store.batch(np.array([2]), family="rich", length=4,
        cadence_seconds=5, scaler=scaler)[0])
    assert all(v.ndim == 1 for v in store._segments.values())
    assert store.raw.ndim == 2


def test_one_shared_train_only_scaler_and_family_projection():
    store = store_for_tests()
    store.raw[1, 1] = np.nan
    scaler = fit_scaler(store, np.arange(6), chunk_size=2)
    changed = FeatureStore(store.frame, store.raw.copy(), store.feature_names, store.groups)
    changed.raw[6:] = 1e9
    changed.frame.loc[:5, "y_crossing"] = -1  # Labels cannot change preprocessing.
    assert scaler == fit_scaler(changed, np.arange(6), chunk_size=2)
    np.testing.assert_allclose(scaler["mean"], np.nanmean(store.raw[:6], axis=0), atol=1e-7)
    np.testing.assert_allclose(scaler["scale"], np.nanstd(store.raw[:6].astype(float), axis=0), atol=1e-12)
    rich = store.batch(np.array([5]), family="rich", length=5, cadence_seconds=5, scaler=scaler)
    micro = store.batch(np.array([5]), family="micro", length=5, cadence_seconds=5, scaler=scaler)
    geometry = store.batch(np.array([5]), family="geometry", length=5, cadence_seconds=5, scaler=scaler)
    np.testing.assert_array_equal(micro[:, :, :2], rich[:, :, :2])
    np.testing.assert_array_equal(micro[:, :, 2:4], rich[:, :, 3:5])
    np.testing.assert_array_equal(geometry[:, :, 0], rich[:, :, 0])
    np.testing.assert_array_equal(geometry[:, :, 1], rich[:, :, 3])
    np.testing.assert_array_equal(geometry[:, :, 2], rich[:, :, -1])


def test_loader_checks_completed_manifest_and_payload_hashes(tmp_path):
    store = store_for_tests(6)
    roots_file, x_file = "2026-04-01_roots.parquet", "2026-04-01_x.npy"
    store.frame.to_parquet(tmp_path / roots_file)
    np.save(tmp_path / x_file, store.raw)
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in (roots_file, x_file)}
    manifest = dict(status="COMPLETE", feature_names=store.feature_names, groups=store.groups,
                    days=[dict(date="2026-04-01", roots_file=roots_file, x_file=x_file, files=files)])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_dataset(tmp_path)
    np.testing.assert_array_equal(loaded.raw, store.raw)
    assert loaded.source_receipt["manifest_sha256"]
    np.save(tmp_path / x_file, store.raw + 1)
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_dataset(tmp_path)
    manifest["status"] = "RUNNING"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not COMPLETE"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("loss", [dict(event_loss="bce"), dict(event_loss="focal", gamma=1.5),
                                  dict(event_loss="proper_confidence", epsilon=1/3)])
def test_saved_unreduced_gradients_are_actual_loss_derivatives(loss):
    z = np.array([[.3, -.5, .4, .2], [-.7, .2, .6, -.3], [.9, -.1, .8, .5],
                  [.1, .3, .2, .4], [-.2, .4, -.6, .8]], dtype=np.float64)
    y = np.array([0, 1, 2, 3, -1], np.int64)
    grad, analytic_error = logit_gradients(z, y, loss)
    eps = 1e-6
    numeric = np.empty_like(z)
    for row, col in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[row, col] += eps
        minus[row, col] -= eps
        numeric[row, col] = (classification_loss(torch.tensor(plus), torch.tensor(y), loss, reduction="sum") -
                             classification_loss(torch.tensor(minus), torch.tensor(y), loss, reduction="sum")) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, atol=1e-8)
    np.testing.assert_array_equal(grad[-1], 0)
    np.testing.assert_array_equal(grad[0, 1:], 0)
    if loss["event_loss"] == "bce":
        assert analytic_error < 1e-12
    else:
        assert analytic_error is None


def test_temporal_label_changes_use_indicators_not_ordinal_class_distance():
    store = store_for_tests(4)
    store.frame["y_crossing"] = [0, 3, 0, -1]
    logits = np.zeros((4, 4), np.float32)
    p = np.tile(np.array([.5, 1/6, 1/6, 1/6], np.float32), (4, 1))
    config = dict(data=dict(cadence_seconds=5), loss=dict(formulation="factorized", event_loss="bce"),
                  splits=dict(train=["2026-04-01"]))
    result, artifact = _prediction_frame(store, p, logits, config)
    np.testing.assert_allclose(result.delta_target_event.iloc[1:3], [1, -1])
    np.testing.assert_allclose(result.delta_target_long.iloc[1:3], [1, -1])
    assert np.isnan(result.delta_target_event.iloc[3]) and np.isnan(result.g_event.iloc[3])
    assert artifact["temporal"]["event"]["train"]["target_flip_fraction"] == 1
    assert artifact["temporal"]["event"]["train"]["adjacent_known_pairs"] == 2


def tiny_config():
    return dict(schema="scientific_opportunity_pipeline_v2", run_id="unit", seed=17,
        data=dict(family="rich", sequence_length=4, cadence_seconds=5),
        splits={role: [f"2026-04-0{i+1}"] for i, role in enumerate(("train", "validation", "calibration", "test"))},
        model=dict(architecture="mlp", hidden=4, layers=1, kernel_size=3, dropout=0.),
        loss=dict(formulation="factorized", event_loss="bce"),
        training=dict(device="cpu", batch_size=4, learning_rate=.01, weight_decay=0., gradient_clip=1.,
                      max_updates=4, validate_every=2, patience_checks=2, min_delta=0., cpu_threads=1,
                      weighting="uniform_roots", train_metric_probe=8, gradient_probe=8))


def test_checkpoint_and_scaler_do_not_depend_on_calibration_or_test(tmp_path):
    original = store_for_tests(32)
    frame = original.frame.copy()
    for d in range(4):
        sl = slice(8*d, 8*d+7)
        frame.loc[sl, "date"] = f"2026-04-0{d+1}"
        frame.loc[sl, "t_ns"] = pd.Timestamp(f"2026-04-0{d+1}T08:00:00Z").value + np.arange(8)*5_000_000_000
    store = FeatureStore(frame, original.raw.copy(), original.feature_names, original.groups)
    config = tiny_config()
    a = train_run(store, config, tmp_path / "a")
    other_frame = frame.copy()
    other_frame.loc[16:, "y_crossing"] = np.resize([3, 2, 1, 0], 16)
    other_raw = store.raw.copy()
    other_raw[16:] += 1e6
    other = FeatureStore(other_frame, other_raw, store.feature_names, store.groups)
    b = train_run(other, config, tmp_path / "b")
    sa = json.loads((tmp_path / "a/scaler.json").read_text())
    sb = json.loads((tmp_path / "b/scaler.json").read_text())
    assert sa == sb
    assert a["history"] == b["history"]
    assert a["summary"]["training"]["best_update"] == b["summary"]["training"]["best_update"]
    ca = torch.load(tmp_path / "a/checkpoint.pt", weights_only=False, map_location="cpu")
    cb = torch.load(tmp_path / "b/checkpoint.pt", weights_only=False, map_location="cpu")
    for name in ca["state_dict"]:
        torch.testing.assert_close(ca["state_dict"][name], cb["state_dict"][name], atol=0, rtol=0)
    # Architecture/feature experiments consume the exact same root cohort.
    cfg_geometry = copy.deepcopy(config)
    cfg_geometry["data"]["family"] = "geometry"
    for role, ids in split_indices(store, config).items():
        np.testing.assert_array_equal(ids, split_indices(store, cfg_geometry)[role])
    assert not a["summary"]["memory"]["materialized_full_sequence_tensor"]
    with pytest.raises(ValueError, match="immutable run"):
        train_run(store, config, tmp_path / "a")
