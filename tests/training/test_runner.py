from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tests.config_helpers import write_run_config
from tests.image_helpers import write_rgb_pair
from tests.manifest_helpers import make_manifest_record, write_manifest_csv
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runner import _resolve_resume_checkpoint_path, run_training


def _write_train_config(tmp_path: Path, dataset_root: Path) -> Path:
    return write_run_config(
        tmp_path,
        """\
        image_size: [32, 32]
        training:
          epochs: 1
          num_workers: 0
        losses:
          generator:
            - name: adversarial_bce
              weight: 1.0
            - name: l1
              weight: 25.0
          discriminator:
            - name: adversarial_bce
              weight: 1.0
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


def test_resolve_resume_checkpoint_path_rejects_missing_explicit_path(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run")
    paths.create_directories()
    expected_path = (paths.checkpoints_dir / "missing.pth").resolve()

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_resume_checkpoint_path("missing.pth", paths)

    assert str(expected_path) in str(exc_info.value)


def test_resolve_resume_checkpoint_path_rejects_wrong_suffix(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run")
    paths.create_directories()
    expected_path = (paths.checkpoints_dir / "checkpoint.txt").resolve()

    with pytest.raises(ValueError) as exc_info:
        _resolve_resume_checkpoint_path("checkpoint.txt", paths)

    assert ".pth" in str(exc_info.value)
    assert str(expected_path) in str(exc_info.value)


def test_resolve_resume_checkpoint_path_accepts_relative_checkpoint_name(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run")
    paths.create_directories()
    checkpoint_path = paths.checkpoints_dir / "ep000.pth"
    checkpoint_path.write_bytes(b"placeholder")

    resolved_path = _resolve_resume_checkpoint_path("ep000.pth", paths)

    assert resolved_path == checkpoint_path.resolve()


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


def test_run_training_resume_latest_raises_when_no_checkpoints_exist(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_training_manifest(dataset_root)
    config_path = write_run_config(
        tmp_path,
        """\
        image_size: [32, 32]
        training:
          epochs: 1
          num_workers: 0
          resume: latest
        losses:
          generator:
            - name: adversarial_bce
              weight: 1.0
            - name: l1
              weight: 25.0
          discriminator:
            - name: adversarial_bce
              weight: 1.0
        """,
        filename="train.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="smoke_run",
    )
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="resume='latest'.*no checkpoints found"):
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
    assert yaml.safe_load(resolved_path.read_text(encoding="utf-8")) == config.to_dict()

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
    assert stage_metadata["stopped_early"] is False

    events = [
        json.loads(line)
        for line in (run_root / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
    assert all(event["stage"] == "train" for event in events)


def test_run_training_writes_early_stopping_result_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        return TrainingResult(
            final_epoch=3,
            best_checkpoint_path=None,
            stopped_early=True,
            stop_epoch=3,
            stop_reason="early_stopping: val_ssim did not improve",
            early_stopping_monitor="val_ssim",
            early_stopping_mode="max",
            early_stopping_best_epoch=1,
            early_stopping_best_value=0.8,
        )

    monkeypatch.setattr("virtual_staining.training.trainer.Trainer.train", _fake_train)

    run_training(config, config_path)

    run_root = tmp_path / "results" / "smoke_run"
    stage_metadata = json.loads(
        (run_root / "metadata" / "stages" / "train.json").read_text(encoding="utf-8")
    )
    assert stage_metadata["stopped_early"] is True
    assert stage_metadata["stop_epoch"] == 3
    assert stage_metadata["stop_reason"] == "early_stopping: val_ssim did not improve"
    assert stage_metadata["early_stopping_monitor"] == "val_ssim"
    assert stage_metadata["early_stopping_mode"] == "max"
    assert stage_metadata["early_stopping_best_epoch"] == 1
    assert stage_metadata["early_stopping_best_value"] == pytest.approx(0.8)

    events = [
        json.loads(line)
        for line in (run_root / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["details"]["stopped_early"] is True
    assert events[-1]["details"]["early_stopping_monitor"] == "val_ssim"


def test_run_training_uses_separate_seeded_loader_generators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_training_manifest(dataset_root)

    config_path = write_run_config(
        tmp_path,
        """\
        image_size: [32, 32]
        training:
          epochs: 1
          seed: 123
          num_workers: 0
        losses:
          generator:
            - name: adversarial_bce
              weight: 1.0
            - name: l1
              weight: 25.0
          discriminator:
            - name: adversarial_bce
              weight: 1.0
        """,
        filename="train.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="smoke_run",
    )
    config = RunConfig.from_yaml(config_path)
    loader_kwargs: list[dict[str, object]] = []

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs) -> None:
            self.dataset = dataset
            loader_kwargs.append(kwargs)

    def _fake_train(self, seed: int, start_epoch: int = 0, reporter=None) -> TrainingResult:
        return TrainingResult(final_epoch=start_epoch, best_checkpoint_path=None)

    monkeypatch.setattr("virtual_staining.training.runner.DataLoader", _FakeDataLoader)
    monkeypatch.setattr("virtual_staining.training.trainer.Trainer.train", _fake_train)

    run_training(config, config_path)

    assert len(loader_kwargs) == 2
    train_generator = loader_kwargs[0]["generator"]
    val_generator = loader_kwargs[1]["generator"]
    assert isinstance(train_generator, torch.Generator)
    assert isinstance(val_generator, torch.Generator)
    assert train_generator is not val_generator
    assert train_generator.initial_seed() == 123
    assert val_generator.initial_seed() == 124
    assert loader_kwargs[0]["shuffle"] is True
    assert loader_kwargs[1]["shuffle"] is False


def test_run_training_wires_augmentation_virtual_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    write_rgb_pair(train_dir, size=(32, 32))
    write_rgb_pair(val_dir, "00256_00000", size=(32, 32))
    _write_training_manifest(dataset_root)

    config_path = write_run_config(
        tmp_path,
        """\
        image_size: [32, 32]
        training:
          epochs: 1
          seed: 123
          num_workers: 0
        augmentation:
          enabled: true
          expansion_factor: 3
          intensity: medium
        losses:
          generator:
            - name: adversarial_bce
              weight: 1.0
            - name: l1
              weight: 25.0
          discriminator:
            - name: adversarial_bce
              weight: 1.0
        """,
        filename="train.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="augmented_run",
    )
    config = RunConfig.from_yaml(config_path)
    loader_datasets: list[Any] = []

    class _FakeDataLoader:
        def __init__(self, dataset, **_kwargs) -> None:
            self.dataset = dataset
            loader_datasets.append(dataset)

    def _fake_train(self, seed: int, start_epoch: int = 0, reporter=None) -> TrainingResult:
        return TrainingResult(final_epoch=start_epoch, best_checkpoint_path=None)

    def _paired_transform(source, target, mask):
        return source, target, mask

    monkeypatch.setattr("virtual_staining.training.runner.DataLoader", _FakeDataLoader)
    monkeypatch.setattr(
        "virtual_staining.training.runner.build_training_paired_transform",
        lambda *_args, **_kwargs: _paired_transform,
    )
    monkeypatch.setattr("virtual_staining.training.trainer.Trainer.train", _fake_train)

    run_training(config, config_path)

    assert len(loader_datasets) == 2
    assert len(loader_datasets[0]) == 3
    assert len(loader_datasets[1]) == 1
    assert loader_datasets[0].paired_transform is _paired_transform

    stage_metadata = json.loads(
        (tmp_path / "results" / "augmented_run" / "metadata" / "stages" / "train.json").read_text(
            encoding="utf-8"
        )
    )
    assert stage_metadata["train_sample_count"] == 1
    assert stage_metadata["effective_train_sample_count"] == 3
    assert stage_metadata["augmentation_enabled"] is True
    assert stage_metadata["augmentation_intensity"] == "medium"
    assert stage_metadata["augmentation_expansion_factor"] == 3


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
