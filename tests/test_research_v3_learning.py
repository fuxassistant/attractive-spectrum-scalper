"""Causal restored windows and TRAIN-only identity reweighting boundaries."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spectrum.research.losses import classification_loss
from spectrum.research_v2.learning import FeatureStore as LegacyStore
from spectrum.research_v3.learning import (
    FeatureStore, compute_train_weights, context_diagnostic, fit_scaler, load_dataset,
    logit_gradients, split_indices, train_run, weighted_classification_loss,
)


LOSS = dict(formulation="factorized", event_loss="bce")


def make_store(count=16):
    y = np.resize(np.array([0, 1, 2, 3], np.int64), count)
    origin = pd.Timestamp("2026-06-01T08:00:00Z").value + np.arange(count, dtype=np.int64) * 5_000_000_000
    frame = pd.DataFrame(dict(date=["2026-06-01"] * count, t_ns=origin,
        feature_available=np.ones(count, bool), physical_support_available=np.ones(count, bool),
        legacy_segment_token=np.zeros(count, np.int64), physical_segment_token=np.zeros(count, np.int64),
        y_crossing=y, source_endpoint_ns=origin-1_000_000_000, source_available_ns=origin-500_000_000,
        source_age_seconds=np.ones(count),
        first_qualifying_event_id_L=np.where(np.isin(y, [1, 3]), "L-common", ""),
        first_qualifying_event_id_S=np.where(np.isin(y, [2, 3]), "S-common", "")))
    raw = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10
    return FeatureStore(frame, raw, ["geometry", "flow", "micro_lag"],
                        {"geometry": ["geometry"], "micro": ["geometry", "flow", "micro_lag"]})


def rebuilt(store, frame=None, raw=None):
    f = (store.frame if frame is None else frame).copy().drop(columns="segment_token", errors="ignore")
    return FeatureStore(f, store.raw.copy() if raw is None else raw, store.feature_names, store.groups)


def batch(store, scaler, ids, mode, length=7):
    return store.batch(np.asarray(ids), family="micro", length=length, cadence_seconds=5,
                       scaler=scaler, context_mode=mode)


def test_missing_x_keeps_its_clock_position_without_forward_fill_or_erasing_history():
    source = make_store(9)
    frame = source.frame.copy()
    frame.loc[3, "feature_available"] = False
    frame.loc[4:, "legacy_segment_token"] = 1
    frame.loc[2, "y_crossing"] = -1  # Future maturity must not define context.
    store = rebuilt(source, frame)
    scaler = fit_scaler(store, np.arange(7))
    restored = batch(store, scaler, [6], "restored")[0]
    legacy = batch(store, scaler, [6], "legacy")[0]
    np.testing.assert_array_equal(restored[:, -1], np.ones(7))
    np.testing.assert_array_equal(restored[3, :-1], np.zeros(6))
    assert restored[:, 3:6].sum() == 6 * 3
    assert np.any(restored[0, :3] != 0)  # Old admitted observation survives.
    np.testing.assert_array_equal(legacy[:, -1], [0, 0, 0, 0, 1, 1, 1])
    old = LegacyStore(store.frame, store.raw, store.feature_names, store.groups)
    expected = old.batch(np.array([6]), family="micro", length=7, cadence_seconds=5, scaler=scaler)
    np.testing.assert_array_equal(legacy, expected[0])
    # Both arms see exactly the same current token and feature schema.
    np.testing.assert_array_equal(legacy[-1], restored[-1])


def test_physical_break_date_and_clock_gap_cannot_be_bridged():
    source = make_store(11)
    frame = source.frame.copy()
    frame.loc[4, ["feature_available", "physical_support_available"]] = False
    frame.loc[4, "physical_segment_token"] = -1
    frame.loc[5:, "physical_segment_token"] = 1
    frame.loc[8:, "date"] = "2026-06-02"
    frame.loc[9:, "t_ns"] += 5_000_000_000
    store = rebuilt(source, frame)
    scaler = fit_scaler(store, np.arange(8))
    result = batch(store, scaler, [4, 5, 7, 8, 9], "restored")
    np.testing.assert_array_equal(result[:, :, -1].sum(1), [0, 1, 3, 1, 1])
    assert np.all(result[0] == 0)


@pytest.mark.parametrize("mode", ["legacy", "restored"])
def test_future_values_labels_identities_and_metadata_do_not_change_past_window(mode):
    source = make_store()
    scaler = fit_scaler(source, np.arange(6))
    before = batch(source, scaler, [5], mode)
    frame = source.frame.copy()
    frame["y_crossing"] = -1
    frame["first_qualifying_event_id_L"] = "future-outcome-change"
    frame["first_qualifying_event_id_S"] = "future-outcome-change"
    frame.loc[6:, "physical_segment_token"] = 900
    frame.loc[6:, "legacy_segment_token"] = 800
    raw = source.raw.copy()
    raw[6:] = 1e9
    changed = rebuilt(source, frame, raw)
    np.testing.assert_array_equal(before, batch(changed, scaler, [5], mode))
    assert scaler == fit_scaler(changed, np.arange(6))
    assert before.shape[-1] == 2 * len(source.groups["micro"]) + 1
    assert all(a.ndim == 1 for a in changed._segments.values())


def test_context_report_counts_supported_positions_and_observations_separately():
    source = make_store(7)
    frame = source.frame.copy()
    frame.loc[3, "feature_available"] = False
    store = rebuilt(source, frame)
    config = dict(data=dict(sequence_length=7, cadence_seconds=5, context_mode="restored"))
    result = context_diagnostic(store, config, {"test": np.array([6])}, 7)["by_split"]["test"]
    assert result["supported_clock_rows"]["q50"] == 7
    assert result["observed_feature_rows"]["q50"] == 6
    assert result["oldest_observed_feature_span_seconds"]["q50"] == 30
    assert result["full_supported_clock_receptive_field_fraction"] == 1
    assert result["fully_observed_receptive_field_fraction"] == 0


def test_admitted_future_publication_and_unsupported_x_are_rejected():
    source = make_store()
    frame = source.frame.copy()
    frame.loc[0, "source_available_ns"] = frame.loc[0, "t_ns"] + 1
    with pytest.raises(ValueError, match="publication"):
        rebuilt(source, frame)
    frame = source.frame.copy()
    frame.loc[0, "physical_support_available"] = False
    with pytest.raises(ValueError, match="physical support"):
        rebuilt(source, frame)


def test_identity_weights_pool_train_memberships_and_preserve_joint_class_mass():
    source = make_store(12)
    frame = source.frame.copy()
    frame["y_crossing"] = [0, 0, 1, 1, 1, 2, 2, 3, 3, -1, 1, 2]
    frame["first_qualifying_event_id_L"] = ["", "", "A", "A", "B", "", "", "A", "B", "unknown", "A", ""]
    frame["first_qualifying_event_id_S"] = ["", "", "", "", "", "A", "C", "C", "D", "unknown", "", "A"]
    store = rebuilt(source, frame)
    train_ids = np.arange(9)
    w, receipt = compute_train_weights(store, train_ids, "crossing_id")
    # Pool memberships across sides: A occurs L rows 2/3/7 and S row5 => 4.
    raw = np.array([1, 1, 1/4, 1/4, 1/2, 1/4, 1/2, (1/4+1/2)/2, (1/2+1)/2])
    expected = raw.copy()
    y = frame.y_crossing.to_numpy()[:9]
    for state in range(4):
        mask = y == state
        expected[mask] *= mask.sum() / raw[mask].sum()
        assert np.isclose(w[:9][mask].sum(), mask.sum())
    np.testing.assert_allclose(w[:9], expected)
    np.testing.assert_array_equal(w[9:], 0)
    assert receipt["unique_positive_crossing_ids"] == 4
    assert receipt["positive_side_memberships"] == 9
    assert receipt["overall"]["kish_effective_sample_size"] < 9
    changed = frame.copy()
    changed.loc[9:, "first_qualifying_event_id_L"] = "A"  # Never counted.
    changed.loc[9:, "first_qualifying_event_id_S"] = "A"
    changed.loc[9:, "y_crossing"] = 3
    other, other_receipt = compute_train_weights(rebuilt(source, changed), train_ids, "crossing_id")
    np.testing.assert_array_equal(w, other)
    assert receipt == other_receipt
    uniform, uniform_receipt = compute_train_weights(store, train_ids, "uniform")
    np.testing.assert_array_equal(uniform[:9], 1)
    assert uniform_receipt["overall"]["kish_effective_sample_size"] == 9


def test_missing_positive_train_id_is_error_not_silently_uniform():
    source = make_store()
    source.frame.loc[1, "first_qualifying_event_id_L"] = ""
    with pytest.raises(ValueError, match="nonempty crossing ID"):
        compute_train_weights(source, np.arange(8), "crossing_id")


def test_weighted_loss_uses_known_root_denominator_and_actual_finite_derivatives():
    z = np.array([[.3, -.5, .4, .2], [-.7, .2, .6, -.3], [.9, -.1, .8, .5],
                  [.1, .3, .2, .4], [-.2, .4, -.6, .8]], np.float64)
    y = np.array([0, 1, 2, 3, -1], np.int64)
    w = np.array([1., .25, 1.5, 3., 0.])
    per = classification_loss(torch.tensor(z), torch.tensor(y), LOSS, reduction="none").numpy()
    got = weighted_classification_loss(torch.tensor(z), torch.tensor(y), w, LOSS)
    assert float(got) == pytest.approx(np.dot(w, per) / 4)
    assert float(got) != pytest.approx(np.dot(w, per) / w.sum())
    gradient, error = logit_gradients(z, y, LOSS, w)
    plain, _ = logit_gradients(z, y, LOSS)
    np.testing.assert_allclose(gradient, plain * w[:, None], atol=1e-14)
    assert error < 1e-12
    epsilon = 1e-6
    for row, column in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        numeric = (weighted_classification_loss(torch.tensor(plus), y, w, LOSS, "sum") -
                   weighted_classification_loss(torch.tensor(minus), y, w, LOSS, "sum")) / (2 * epsilon)
        assert gradient[row, column] == pytest.approx(float(numeric), abs=2e-9)
    np.testing.assert_array_equal(gradient[-1], 0)
    np.testing.assert_array_equal(gradient[0, 1:], 0)


def test_loader_requires_context_metadata_and_verifies_hashes(tmp_path):
    source = make_store(6)
    roots_file, x_file = "2026-06-01_roots.parquet", "2026-06-01_x.npy"
    source.frame.to_parquet(tmp_path / roots_file)
    np.save(tmp_path / x_file, source.raw)
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in (roots_file, x_file)}
    manifest = dict(status="COMPLETE", feature_names=source.feature_names, groups=source.groups,
                    days=[dict(date="2026-06-01", roots_file=roots_file, x_file=x_file, files=files)])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_dataset(tmp_path)
    np.testing.assert_array_equal(loaded.raw, source.raw)
    np.save(tmp_path / x_file, source.raw + 1)
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_dataset(tmp_path)


def tiny_config():
    return dict(schema="scientific_opportunity_pipeline_v3", run_id="unit", seed=17,
        data=dict(family="micro", sequence_length=7, cadence_seconds=5, context_mode="restored"),
        splits={role: [f"2026-06-0{i+1}"] for i, role in enumerate(("train", "validation", "calibration", "test"))},
        model=dict(architecture="tcn", hidden=4, layers=1, kernel_size=2, dropout=0.), loss=LOSS,
        training=dict(device="cpu", batch_size=4, learning_rate=.01, weight_decay=0., gradient_clip=1.,
                      max_updates=4, validate_every=2, patience_checks=2, min_delta=0., cpu_threads=1,
                      weighting="crossing_id", train_metric_probe=8, gradient_probe=8))


def four_split_store():
    source = make_store(32)
    frame = source.frame.copy()
    for day in range(4):
        ids = np.arange(8*day, 8*day+8)
        date = f"2026-06-0{day+1}"
        origin = pd.Timestamp(date + "T08:00:00Z").value + np.arange(8)*5_000_000_000
        frame.loc[ids, "date"] = date
        frame.loc[ids, "t_ns"] = origin
        frame.loc[ids, "source_endpoint_ns"] = origin-1_000_000_000
        frame.loc[ids, "source_available_ns"] = origin-500_000_000
    frame.loc[1, "first_qualifying_event_id_L"] = "rare-L"
    return rebuilt(source, frame)


def test_checkpoint_weights_and_scaler_ignore_cal_test_and_all_heldout_ids(tmp_path):
    source, config = four_split_store(), tiny_config()
    original = train_run(source, config, tmp_path / "a")
    frame = source.frame.copy()
    frame.loc[8:, "first_qualifying_event_id_L"] = "changed-heldout-ID"
    frame.loc[8:, "first_qualifying_event_id_S"] = "changed-heldout-ID"
    frame.loc[16:, "y_crossing"] = np.resize([3, 2, 1, 0], 16)
    raw = source.raw.copy()
    raw[16:] += 1e6
    changed = train_run(rebuilt(source, frame, raw), config, tmp_path / "b")
    assert original["history"] == changed["history"]
    for filename in ("weighting.json", "scaler.json"):
        assert json.loads((tmp_path / "a" / filename).read_text()) == json.loads((tmp_path / "b" / filename).read_text())
    a = torch.load(tmp_path / "a/checkpoint.pt", weights_only=False, map_location="cpu")
    b = torch.load(tmp_path / "b/checkpoint.pt", weights_only=False, map_location="cpu")
    for name in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], atol=0, rtol=0)
    best_row = min(original["history"], key=lambda row: row["validation_nll"])
    assert a["best_update"] == best_row["update"]
    prediction = original["predictions"]
    assert prediction.loc[prediction["split"] != "train", "optimization_weight_train_only"].isna().all()
    assert prediction.loc[prediction["split"] != "train", "g_event_training_weighted"].isna().all()
    assert original["gradient_artifact"]["selected"]["weighted"]["analytic_bce_max_abs_error"] < 1e-6
    assert not original["summary"]["memory"]["materialized_full_sequence_tensor"]


def test_paired_arms_share_initial_state_probes_scaler_cohort_and_sample_order(tmp_path):
    store, config = four_split_store(), tiny_config()
    a = train_run(store, config, tmp_path / "weighted_restored")
    other = copy.deepcopy(config)
    other["data"]["context_mode"] = "legacy"
    other["training"]["weighting"] = "uniform"
    b = train_run(store, other, tmp_path / "uniform_legacy")
    for key in ("initial_state_sha256", "train_metric_probe_sha256", "gradient_probe_sha256"):
        assert a["summary"]["training"][key] == b["summary"]["training"][key]
    assert a["summary"]["cohorts"] == b["summary"]["cohorts"]
    assert a["summary"]["scaler_sha256"] == b["summary"]["scaler_sha256"]
    for left, right in zip(a["history"], b["history"]):
        assert left["sampled_root_stream_sha256"] == right["sampled_root_stream_sha256"]
    for role in config["splits"]:
        assert a["gradient_artifact"]["temporal"]["event"][role]["adjacent_known_pairs"] == b["gradient_artifact"]["temporal"]["event"][role]["adjacent_known_pairs"]
    with pytest.raises(ValueError, match="immutable run"):
        train_run(store, config, tmp_path / "weighted_restored")
