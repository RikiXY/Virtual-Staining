from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tests.manifest_helpers import write_manifest_csv
from virtual_staining.config.run import RunConfig
from virtual_staining.data.manifest import ManifestRecord
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runner import run_training


def _write_rgb_pair(directory: Path, prefix: str = "00000_00000") -> None:
    Image.new("RGB", (32, 32)).save(directory / f"{prefix}_source.png")
    Image.new("RGB", (32, 32)).save(directory / f"{prefix}_target.png")


def _write_training_manifest(dataset_root: Path) -> None:
    records = (
        ManifestRecord(
            sample_id="00000_00000",
            split="train",
            input_path=Path("dataset_train/00000_00000_source.png"),
            target_path=Path("dataset_train/00000_00000_target.png"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        ),
        ManifestRecord(
            sample_id="00256_00000",
            split="val",
            input_path=Path("dataset_val/00256_00000_source.png"),
            target_path=Path("dataset_val/00256_00000_target.png"),
            input_modality="label_free",
            target_modality="stained",
            x=256,
            y=0,
            width=256,
            height=256,
        ),
    )
    write_manifest_csv(dataset_root, records)


def _write_val_only_manifest(dataset_root: Path) -> None:
    records = (
        ManifestRecord(
            sample_id="00256_00000",
            split="val",
            input_path=Path("dataset_val/00256_00000_source.png"),
            target_path=Path("dataset_val/00256_00000_target.png"),
            input_modality="label_free",
            target_modality="stained",
            x=256,
            y=0,
            width=256,
            height=256,
        ),
    )
    write_manifest_csv(dataset_root, records)


def _write_missing_file_manifest(dataset_root: Path) -> None:
    records = (
        ManifestRecord(
            sample_id="00000_00000",
            split="train",
            input_path=Path("dataset_train/00000_00000_source.png"),
            target_path=Path("dataset_train/00000_00000_target.png"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        ),
        ManifestRecord(
            sample_id="00256_00000",
            split="val",
            input_path=Path("dataset_val/00256_00000_source.png"),
            target_path=Path("dataset_val/00256_00000_target.png"),
            input_modality="label_free",
            target_modality="stained",
            x=256,
            y=0,
            width=256,
            height=256,
        ),
    )
    write_manifest_csv(dataset_root, records)


def test_run_training_raises_if_manifest_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    _write_rgb_pair(train_dir)
    _write_rgb_pair(val_dir, prefix="00256_00000")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
dataset_root: {dataset_root}
results_path: {tmp_path / "results"}
run_name: smoke_run
image_size: [32, 32]
training:
  epochs: 1
  num_workers: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        run_training(config, config_path)


def test_run_training_raises_if_required_train_split_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    val_dir = dataset_root / "dataset_val"
    val_dir.mkdir(parents=True)
    _write_rgb_pair(val_dir, prefix="00256_00000")
    _write_val_only_manifest(dataset_root)

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
dataset_root: {dataset_root}
results_path: {tmp_path / "results"}
run_name: smoke_run
image_size: [32, 32]
training:
  epochs: 1
  num_workers: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(ValueError, match="train"):
        run_training(config, config_path)


def test_run_training_raises_if_manifest_input_file_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    _write_rgb_pair(val_dir, prefix="00256_00000")
    _write_missing_file_manifest(dataset_root)

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
dataset_root: {dataset_root}
results_path: {tmp_path / "results"}
run_name: smoke_run
image_size: [32, 32]
training:
  epochs: 1
  num_workers: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        run_training(config, config_path)


def test_run_training_writes_resolved_config_and_hash(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    _write_rgb_pair(train_dir)
    _write_rgb_pair(val_dir, prefix="00256_00000")
    _write_training_manifest(dataset_root)

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
dataset_root: {dataset_root}
results_path: {tmp_path / "results"}
run_name: smoke_run
image_size: [32, 32]
training:
  epochs: 1
  num_workers: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
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
