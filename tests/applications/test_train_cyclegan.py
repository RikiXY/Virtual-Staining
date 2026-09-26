from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.config_helpers import cyclegan_config_data, write_config_data
from tests.image_helpers import write_rgb_image
from virtual_staining.applications.train import train
from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout


def _write_domains(dataset_root: Path) -> None:
    layouts = {
        "domains/label_free/{split}": {"train": 3, "val": 2},
        "prepared/{split}/stained": {"train": 5, "val": 1},
    }
    for pattern, counts in layouts.items():
        for split, count in counts.items():
            for index in range(count):
                write_rgb_image(
                    dataset_root / pattern.format(split=split) / f"{index}.png",
                    size=(40, 36),
                    color=(40 * index, 100, 200 - 30 * index),
                )


def _config(
    tmp_path: Path,
    name: str,
    *,
    epochs: int,
    resume: str | None,
    scheduler: dict[str, Any] | None = None,
) -> Path:
    data: dict[str, Any] = cyclegan_config_data(tmp_path)
    data["data"]["domains"] = {
        "label_free": "domains/label_free",
        "stained": "prepared/{split}/stained/*.png",
    }
    data["training"].update(
        epochs=epochs,
        resume=resume,
        # Plateau policy does not depend on the epoch budget, so extending epochs is a resume.
        scheduler=scheduler or {"name": "reduce_on_plateau", "patience": 0},
    )
    return write_config_data(tmp_path / f"{name}.yaml", data)


def _pool_sizes(path: Path) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["method"]["name"] == "cyclegan"
    pools = payload["state"]["replay_pools"]
    return [len(pools[name]["images"]) for name in ("fake_A", "fake_B")]


def test_cyclegan_trains_checkpoints_and_resumes_through_generic_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_domains(tmp_path / "dataset")
    first_path = _config(tmp_path, "first", epochs=1, resume=None)
    first = RunConfig.from_yaml(first_path)
    layout = RunLayout.from_project(first.project)

    first_result = train(first, first_path)

    assert first_result.final_epoch == 0
    assert first_result.best_checkpoint_path == layout.checkpoints_dir / "ep000.pth"
    # Unequal domains: len = max(3, 5) = 5 samples, so each pool holds 5 fakes.
    assert _pool_sizes(layout.checkpoints_dir / "ep000.pth") == [5, 5]
    assert [path.name for path in layout.output_val_dir.iterdir()] == ["epoch0_batch0_preview.tif"]

    second_path = _config(tmp_path, "second", epochs=2, resume="latest")
    second_result = train(RunConfig.from_yaml(second_path), second_path)

    assert second_result.final_epoch == 1
    # The fresh runtime resumed the replay pools rather than restarting them empty.
    assert _pool_sizes(layout.checkpoints_dir / "ep001.pth") == [10, 10]
    with layout.epochs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["0", "1"]
    for row in rows:
        assert float(row["loss_G_val"]) > 0
        assert float(row["loss_train_raw_generator_cycle_l1"]) > 0
        assert float(row["loss_val_raw_discriminator_adversarial_lsgan"]) > 0
        assert row["val_ssim"] == row["val_psnr"] == ""


def test_resume_rejects_a_changed_linear_decay_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_domains(tmp_path / "dataset")
    linear = {"name": "linear_decay", "decay_start_epoch": 0}
    first_path = _config(tmp_path, "first", epochs=1, resume=None, scheduler=linear)
    train(RunConfig.from_yaml(first_path), first_path)

    second_path = _config(tmp_path, "second", epochs=2, resume="latest", scheduler=linear)
    with pytest.raises(CheckpointCompatibilityError, match=r"scheduler\.epochs is 1 .* but 2"):
        train(RunConfig.from_yaml(second_path), second_path)
