"""Reject misleading reevaluation inputs before consuming any trial outcomes."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.mechanical_lab import evaluate as evaluation
from experiments.mechanical_lab.data import digest, dump
from experiments.mechanical_lab.run import HERE, validate


def default_config():
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def toy_run(tmp_path, *, stress=False):
    root = tmp_path / "run"
    root.mkdir()
    config = default_config()
    config["splits"] = {
        "discovery": ["2026-08-03", "2026-08-03"],
        "validation": ["2026-08-17", "2026-08-17"],
        "diagnostic_test": ["2026-08-24", "2026-08-24"],
        "extension": ["2026-08-31", "2026-08-31"],
    }
    dump(root / "frozen_config.json", config)
    (root / "source_snapshot").mkdir()
    source = HERE / "portfolio.py"
    (root / "source_snapshot" / source.name).write_bytes(source.read_bytes())
    relative = str(source.relative_to(evaluation.ROOT))
    days = []
    for split, (date, _) in config["splits"].items():
        if stress and split not in ("diagnostic_test", "extension"):
            continue
        directory = root / "days" / date
        directory.mkdir(parents=True)
        hashes = {}
        for name in ("features", "candidates", "all_signals", "outcomes"):
            filename = name + ".parquet"
            pd.DataFrame({"date": [date], "split": [split]}).to_parquet(directory / filename, index=False)
            hashes[filename] = digest(directory / filename)
        receipt = {"status": "COMPLETE", "date": date, "artifact_sha256": hashes}
        dump(directory / "receipt.json", receipt)
        days.append(receipt)
    manifest = {"status": "COMPLETE", "config_sha256": digest(root / "frozen_config.json"),
                "code_identity": {relative: digest(source)}, "days": days}
    dump(root / "manifest.json", manifest)
    return root, manifest


def test_frozen_default_and_supported_recipe_validate():
    config = default_config()
    validate(config)
    config["recipe"] = {"entry": {"id": "momentum_second", "parameters": {"first_body_atr": 1.0}},
                        "veto": {"id": "trend_aligned"},
                        "exit": {"id": "breakeven", "parameters": {"take_r": 2.5}},
                        "sizing": "martingale_capped"}
    validate(config)


@pytest.mark.parametrize("change", ["hours", "gate", "entry_timeout", "stress_timeout", "coin_count"])
def test_fixed_experiment_contract_cannot_be_silently_reconfigured(change):
    config = deepcopy(default_config())
    if change == "hours":
        config["research_hours_utc"] = ["08:00", "20:30"]
    elif change == "gate":
        config["candidate_gates"]["min_trades_validation_and_test"] = 1
    elif change == "entry_timeout":
        config["execution"]["pending_timeout_seconds"] = 3.0
    elif change == "stress_timeout":
        config["stress"]["pending_timeout_seconds"] = 3.0
    else:
        config["coin_seeds"] = 10
    with pytest.raises(ValueError):
        validate(config)


def test_verified_run_loads_all_declared_days(tmp_path):
    root, _ = toy_run(tmp_path)
    candidates, outcomes = evaluation.load_run(root)
    assert len(candidates) == len(outcomes) == 4
    assert set(candidates.split) == {"discovery", "validation", "diagnostic_test", "extension"}


def test_running_manifest_cannot_be_treated_as_a_complete_run(tmp_path):
    root, manifest = toy_run(tmp_path)
    manifest["status"] = "RUNNING"
    dump(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="COMPLETE"):
        evaluation.load_run(root)


def test_missing_day_is_rejected_even_when_remaining_parquets_are_readable(tmp_path):
    root, manifest = toy_run(tmp_path)
    # Keep files intact, but remove one expected receipt from manifest identity.
    manifest["days"] = manifest["days"][:-1]
    dump(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dates"):
        evaluation.load_run(root)


def test_edited_frozen_config_is_rejected(tmp_path):
    root, _ = toy_run(tmp_path)
    with (root / "frozen_config.json").open("a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="config hash"):
        evaluation.load_run(root)


def test_archived_code_tamper_is_rejected(tmp_path):
    root, _ = toy_run(tmp_path)
    with (root / "source_snapshot" / "portfolio.py").open("a", encoding="utf-8") as f:
        f.write("\n# changed archived implementation\n")
    with pytest.raises(ValueError, match="implementation hash"):
        evaluation.load_run(root)


def test_live_code_must_match_frozen_execution_contract(tmp_path, monkeypatch):
    root, _ = toy_run(tmp_path)
    actual_digest = evaluation.digest
    source = (HERE / "portfolio.py").resolve()

    def changed_live_hash(path):
        return "0" * 64 if Path(path).resolve() == source else actual_digest(path)

    monkeypatch.setattr(evaluation, "digest", changed_live_hash)
    with pytest.raises(ValueError, match="[Ii]mplementation|[Cc]ode|[Ll]ive"):
        evaluation.load_run(root)


def test_modified_output_is_rejected_before_parquet_read(tmp_path):
    root, _ = toy_run(tmp_path)
    (root / "days" / "2026-08-24" / "outcomes.parquet").write_bytes(b"corrupted outcome")
    with pytest.raises(ValueError, match="artifact hash"):
        evaluation.load_run(root)


def test_day_receipt_must_match_root_manifest(tmp_path):
    root, _ = toy_run(tmp_path)
    path = root / "days" / "2026-08-24" / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["unreviewed_extra_field"] = True
    dump(path, receipt)
    with pytest.raises(ValueError, match="receipt mismatch"):
        evaluation.load_run(root)


def test_stress_run_has_separate_expected_day_contract(tmp_path):
    root, _ = toy_run(tmp_path, stress=True)
    candidates, _ = evaluation.load_run(root, stress=True)
    assert set(candidates.split) == {"diagnostic_test", "extension"}
    with pytest.raises(ValueError, match="dates"):
        evaluation.load_run(root, stress=False)


def test_evaluation_cannot_override_the_run_frozen_account_contract(tmp_path):
    root, _ = toy_run(tmp_path)
    config = json.loads((root / "frozen_config.json").read_text(encoding="utf-8"))
    config["account"]["initial_capital_usd"] *= 2
    with pytest.raises(ValueError, match="[Cc]onfig|[Cc]ontract"):
        evaluation.evaluate(root, config)


def test_other_recipe_stress_run_cannot_be_attached_to_this_run(tmp_path):
    root, _ = toy_run(tmp_path)
    config = json.loads((root / "frozen_config.json").read_text(encoding="utf-8"))
    stress = tmp_path / "other_stress"
    stress.mkdir()
    other = deepcopy(config)
    other["execution"]["fee_roundtrip_per_oz"] *= 2
    dump(stress / "frozen_config.json", other)
    with pytest.raises(ValueError, match="[Cc]onfig|[Cc]ontract"):
        evaluation.evaluate(root, config, stress)
