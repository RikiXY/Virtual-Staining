from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tests.config_helpers import write_run_config
from tests.image_helpers import write_rgb_pair
from tests.manifest_helpers import make_manifest_record, write_manifest_csv
from virtual_staining.config.run import RunConfig
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runner import run_training


def _write_train_config(tmp_path: Path, dataset_root: Path) -> Path:
    return write_run_config(
        tmp_path,
        """\
        image_size: [32, 32]
        training:
          epochs: 1
          num_workers: 0
        """,
        filename="train.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="smoke_run",
    )


def _write_training_manifest(dataset_root: Path) -> None:
    records = (
        make_manifest_record("00000_00000", "train", ext=".png"),
        make_manifest_record("00256_00000", "val", ext=".png"),
    )
    write_manifest_csv(dataset_root, records)


def _write_val_only_manifest(dataset_root: Path) -> None:
    records = (make_manifest_record("00256_00000", "val", ext=".png"),)
    write_manifest_csv(dataset_root, records)


def _write_missing_file_manifest(dataset_root: Path) -> None:
    records = (
        make_manifest_record("00000_00000", "train", ext=".png"),
        make_manifest_record("00256_00000", "val", ext=".png"),
    )
    write_manifest_csv(dataset_root, records)


def test_run_training_raises_if_manifest_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))

    config_path = _write_train_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        run_training(config, config_path)


def test_run_training_raises_if_required_train_split_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    val_dir = dataset_root / "splits" / "val"
    val_dir.mkdir(parents=True)
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_val_only_manifest(dataset_root)

    config_path = _write_train_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(ValueError, match="train"):
        run_training(config, config_path)


def test_run_training_raises_if_manifest_input_file_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_missing_file_manifest(dataset_root)

    config_path = _write_train_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        run_training(config, config_path)


def test_run_training_writes_resolved_config_and_hash(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_training_manifest(dataset_root)

    config_path = _write_train_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    def _fake_train(self, seed: int, start_epoch: int = 0, reporter=None) -> TrainingResult:
        return TrainingResult(final_epoch=start_epoch, best_checkpoint_path=None)

    monkeypatch.setattr("virtual_staining.training.trainer.Trainer.train", _fake_train)

    run_training(config, config_path)

    run_root = tmp_path / "results" / "smoke_run"
    input_path = run_root / "config" / "input.yaml"
    resolved_path = run_root / "config" / "resolved.yaml"
    hash_path = run_root / "metadata" / "config_hash.txt"
    environment_path = run_root / "metadata" / "environment.json"

    assert input_path.exists()
    assert resolved_path.exists()
    assert hash_path.exists()
    assert environment_path.exists()
    assert input_path.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    assert yaml.safe_load(resolved_path.read_text(encoding="utf-8")) == config.to_yaml_dict()

    expected = f"sha256:{hashlib.sha256(resolved_path.read_bytes()).hexdigest()}"
    assert hash_path.read_text(encoding="utf-8") == expected

    run_metadata = json.loads((run_root / "metadata" / "run.json").read_text(encoding="utf-8"))
    manifest_path = dataset_root / "manifests" / "manifest.csv"
    manifest_hash = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"

    assert run_metadata["manifest_path"] == str(manifest_path)
    assert run_metadata["manifest_sha256"] == manifest_hash
    assert run_metadata["stages_present"] == ["train"]
    assert run_metadata["last_completed_stage"] == "train"

    stage_metadata = json.loads(
        (run_root / "metadata" / "stages" / "train.json").read_text(encoding="utf-8")
    )
    assert stage_metadata["stage"] == "train"
    assert stage_metadata["status"] == "completed"
    assert stage_metadata["config_hash"] == expected
    assert stage_metadata["manifest_sha256"] == manifest_hash

    events = [
        json.loads(line)
        for line in (run_root / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
    assert all(event["stage"] == "train" for event in events)


def test_run_training_writes_failed_stage_metadata_and_events(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_training_manifest(dataset_root)

    config_path = _write_train_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    def _fail_train(self, seed: int, start_epoch: int = 0, reporter=None) -> TrainingResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("virtual_staining.training.trainer.Trainer.train", _fail_train)

    with pytest.raises(RuntimeError, match="boom"):
        run_training(config, config_path)

    run_root = tmp_path / "results" / "smoke_run"
    stage_metadata = json.loads(
        (run_root / "metadata" / "stages" / "train.json").read_text(encoding="utf-8")
    )
    assert stage_metadata["status"] == "failed"
    assert stage_metadata["error"] == "boom"

    events = [
        json.loads(line)
        for line in (run_root / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_failed"]
