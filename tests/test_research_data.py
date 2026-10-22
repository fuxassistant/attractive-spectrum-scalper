import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from spectrum.research.data import (
    GEOMETRY_FEATURES, RICH_FEATURES, _join_features,
    build_sequence_indices, prepare_sequences,
    _source_paths, MAP_ROOT, BB_ROOT,
)


def roots():
    return pd.DataFrame(dict(date=["A"] * 5 + ["B"] * 3,
                             t_ns=np.array([0, 30, 60, 90, 150, 180, 210, 240]) * 10**9,
                             feature_available=[True, True, False, True, True, True, True, True],
                             y=[0, -1, 1, 2, 0, -1, 3, 1]))


def test_sequence_clock_boundaries_and_unknown_labels_do_not_remove_history():
    f = roots()
    expected = [[-1, -1, 0], [-1, 0, 1], [-1, -1, -1], [-1, -1, 3],
                [-1, -1, 4], [-1, -1, 5], [-1, 5, 6], [5, 6, 7]]
    observed = build_sequence_indices(f, 3)
    np.testing.assert_array_equal(observed, expected)
    f.y = 999
    np.testing.assert_array_equal(build_sequence_indices(f, 3), observed)
    f.loc[7, "t_ns"] = f.loc[6, "t_ns"]
    with pytest.raises(ValueError, match="chronological"):
        build_sequence_indices(f, 3)


def test_train_scaler_unique_rows_and_future_mutation_invariance():
    f = pd.DataFrame(dict(date=["A"] * 5, t_ns=np.arange(5) * 30 * 10**9, feature_available=True))
    idx = build_sequence_indices(f, 3)
    raw = np.array([[0., np.nan], [2., 4.], [4., 4.], [100., 8.], [200., 8.]])
    x, sc = prepare_sequences(raw, idx, np.array([1, 2]))
    # Overlapping train windows must not give row 1 twice the scaler weight.
    assert sc["train_feature_row_indices"] == [0, 1, 2]
    np.testing.assert_allclose(sc["mean"], [2., 4.])
    assert sc["finite_count"] == [3, 2]
    changed = raw.copy()
    changed[3:] = -1e9
    x2, sc2 = prepare_sequences(changed, idx, np.array([1, 2]))
    for key in ("mean", "scale", "finite_count", "train_feature_row_indices"):
        assert sc2[key] == sc[key]
    np.testing.assert_array_equal(x[:3], x2[:3])
    assert np.max(np.abs(x[:, :, :2])) <= 10
    assert x[0, -1, 3] == 0  # Missing feature has its own mask.
    assert x[0, -1, -1] == 1  # Missing feature is not left padding.
    assert np.all(x[0, :2] == 0)


def test_feature_allowlist_does_not_consume_future_metadata_or_drop_origins():
    base = pd.DataFrame(dict(date=["A"] * 3, t_ns=[1, 2, 3]))
    source = pd.DataFrame(dict(date=["A"] * 2, t_ns=[1, 3], bb1_position=[.1, .3],
                               m1_y=[2, 0], m1_event_ns=[10, 20], m1_support_until_ns=[99, 99]))
    frame, x = _join_features(base, source, ["bb1_position"])
    assert len(frame) == 3
    assert frame.feature_available.tolist() == [True, False, True]
    assert np.isnan(x[1, 0])
    source[["m1_y", "m1_event_ns", "m1_support_until_ns"]] = -123456
    _, x2 = _join_features(base, source, ["bb1_position"])
    np.testing.assert_array_equal(x, x2)
    assert len(RICH_FEATURES) == 65 and len(GEOMETRY_FEATURES) == 16
    assert not any("event" in n or "support_until" in n or "maturity" in n for n in RICH_FEATURES + GEOMETRY_FEATURES)


def test_future_sequence_index_rejected_and_all_missing_channel_is_masked():
    raw = np.array([[np.nan, 1.], [np.nan, 1.]])
    x, sc = prepare_sequences(raw, np.array([[-1, 0], [0, 1]]), np.array([0]))
    assert sc["all_missing_columns"] == [0]
    assert sc["constant_columns"] == [1]
    assert np.all(x[:, :, 0] == 0) and np.all(x[:, :, 2] == 0)
    with pytest.raises(ValueError, match="past-only"):
        prepare_sequences(raw, np.array([[0, 1], [0, 1]]), np.array([0]))


def test_explicit_source_path_switches_are_honored():
    assert _source_paths({}) == (MAP_ROOT, BB_ROOT)
    map_path = Path("alternative_crossings")
    features_path = Path("alternative_features")
    actual = _source_paths(dict(roots_dir=str(map_path), feature_roots_dir=features_path))
    assert actual == (map_path, features_path)
    assert _source_paths(dict(roots_dir="relative_map"))[0] == Path("relative_map")
