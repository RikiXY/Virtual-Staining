from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.snapshots import (
    save_environment_snapshot,
    save_stage_config_snapshots,
)


def _write_run_config(path: Path, *, run_name: str = "snapshots_run") -> RunConfig:
    path.write_text(
        f"""
dataset_root: {path.parent / "dataset"}
results_path: {path.parent / "results"}
run_name: {run_name}
image_size: [32, 32]
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return RunConfig.from_yaml(path)


def test_save_stage_config_snapshots_writes_input_resolved_and_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config = _write_run_config(config_path)

    input_dest = tmp_path / "artifacts" / "config" / "input.yaml"
    resolved_dest = tmp_path / "artifacts" / "config" / "resolved.yaml"
    hash_dest = tmp_path / "artifacts" / "metadata" / "config_hash.txt"

    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=input_dest,
        resolved_dest=resolved_dest,
        hash_dest=hash_dest,
    )

    assert input_dest.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    assert yaml.safe_load(resolved_dest.read_text(encoding="utf-8")) == config.to_yaml_dict()
    expected = f"sha256:{hashlib.sha256(resolved_dest.read_bytes()).hexdigest()}"
    assert config_hash == expected
    assert hash_dest.read_text(encoding="utf-8") == expected


def test_save_stage_config_snapshots_is_hash_stable_for_same_effective_config(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first = _write_run_config(first_path, run_name="stable_run")
    second = _write_run_config(second_path, run_name="stable_run")

    first_hash = save_stage_config_snapshots(
        first,
        first_path,
        input_dest=tmp_path / "first" / "config" / "input.yaml",
        resolved_dest=tmp_path / "first" / "config" / "resolved.yaml",
        hash_dest=tmp_path / "first" / "metadata" / "config_hash.txt",
    )
    second_hash = save_stage_config_snapshots(
        second,
        second_path,
        input_dest=tmp_path / "second" / "config" / "input.yaml",
        resolved_dest=tmp_path / "second" / "config" / "resolved.yaml",
        hash_dest=tmp_path / "second" / "metadata" / "config_hash.txt",
    )

    assert first_hash == second_hash


def test_save_environment_snapshot_writes_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "virtual_staining.experiment.snapshots.collect_environment",
        lambda: {"python_version": "3.11.0", "cuda_available": False},
    )

    dest = tmp_path / "metadata" / "environment.json"
    save_environment_snapshot(dest)

    assert json.loads(dest.read_text(encoding="utf-8")) == {
        "python_version": "3.11.0",
        "cuda_available": False,
    }
