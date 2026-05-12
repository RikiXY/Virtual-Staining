from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from PIL import Image

from virtual_staining.config.run import RunConfig
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runner import run_training


def _write_rgb_pair(directory: Path, prefix: str = "00000_00000") -> None:
    Image.new("RGB", (32, 32)).save(directory / f"{prefix}_source.png")
    Image.new("RGB", (32, 32)).save(directory / f"{prefix}_target.png")


def test_run_training_writes_resolved_config_and_hash(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    _write_rgb_pair(train_dir)
    _write_rgb_pair(val_dir)

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
    resolved_path = run_root / "config" / "resolved.yaml"
    hash_path = run_root / "metadata" / "config_hash.txt"

    assert resolved_path.exists()
    assert hash_path.exists()
    assert yaml.safe_load(resolved_path.read_text(encoding="utf-8")) == config.to_yaml_dict()

    expected = f"sha256:{hashlib.sha256(resolved_path.read_bytes()).hexdigest()}"
    assert hash_path.read_text(encoding="utf-8") == expected
