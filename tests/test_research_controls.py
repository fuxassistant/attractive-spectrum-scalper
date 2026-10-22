"""Synthetic positives must come from the same executable target engine."""

import numpy as np

from spectrum.research import controls
from spectrum.research.simulation import simulate_tape


def test_control_reproducibility_and_independent_seeds():
    a = controls.make_control_data(12, count=30)
    b = controls.make_control_data(12, count=30)
    other = controls.make_control_data(13, count=30)
    assert a["x"].shape == (30, 16, 5)
    assert a["x"].dtype == np.float32 and a["y"].dtype == np.int64
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["y"], b["y"])
    np.testing.assert_array_equal(a["net"], b["net"])
    assert a["receipts"] == b["receipts"]
    assert not np.array_equal(a["x"], other["x"])
    assert not np.array_equal(a["net"], other["net"])
    assert a["receipts"]["state_counts"] == [10, 10, 10, 0]


def test_labels_match_replayed_actual_delayed_execution():
    data = controls.make_control_data(101, count=36)
    trials = simulate_tape(data["tape"], data["all_origins_ns"], data["all_risk"],
                           data["simulation_config"])
    net = trials.net.to_numpy().reshape(-1, 2)[data["retained_episode_indices"]]
    np.testing.assert_array_equal(net, data["net"])
    expected = (net[:, 0] > 0).astype(np.int64) + 2 * (net[:, 1] > 0).astype(np.int64)
    np.testing.assert_array_equal(data["y"], expected)
    assert (data["entry_ns"] > data["origins_ns"][:, None]).all()
    assert (data["exit_ns"] > data["entry_ns"]).all()
    # Every X price is from the past portion of the very tape being simulated.
    for row, episode in enumerate(data["retained_episode_indices"]):
        start = episode * 64
        q = data["tape"]
        mid = (q.bid[start:start + 16] + q.ask[start:start + 16]) / 2e9
        np.testing.assert_allclose(data["x"][row, :, 2], (mid - mid[0]) / .5, atol=2e-7)


def test_labels_follow_engine_results_and_unknowns_are_excluded(monkeypatch):
    original = controls.simulate_tape

    def altered_engine(*args, **kwargs):
        trial = original(*args, **kwargs)
        # Engine-derived unknowns must be excluded, not mapped to NONE.
        trial.loc[0, "status"] = "CENSORED"
        trial.loc[0, "net"] = np.nan
        # A deliberate engine outcome perturbation changes the label even
        # though latent dynamics and past features have not changed.
        trial.loc[2:3, "net"] = [1., 2.]
        return trial

    baseline = controls.make_control_data(2, count=12)
    monkeypatch.setattr(controls, "simulate_tape", altered_engine)
    altered = controls.make_control_data(2, count=12)
    assert altered["receipts"]["excluded_unknown_count"] == 1
    assert len(altered["y"]) == 11
    assert altered["retained_episode_indices"][0] == 1
    assert altered["y"][0] == 3
    np.testing.assert_array_equal(altered["x"][0], baseline["x"][1])
