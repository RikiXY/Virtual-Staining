from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from tests.config_helpers import cyclegan_config_data, write_config_data
from tests.manifest_helpers import write_aligned_test_manifest
from virtual_staining.applications.infer_images import infer_images
from virtual_staining.applications.pipeline import run_stage
from virtual_staining.checkpoint_contract import CHECKPOINT_FORMAT_VERSION
from virtual_staining.evaluation.unpaired import (
    UNPAIRED_FEATURE_COMPARISON_CSV,
    UNPAIRED_IMAGE_STATISTICS_CSV,
)
from virtual_staining.utils.artifacts import generated_filename

_SAMPLE_IDS = ["00000_00000", "00256_00000"]


def _write_noise(path: Path, rng: np.random.Generator, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 128, size=(32, 32, 3)) + offset
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def _write_dataset(dataset_root: Path) -> None:
    """Independent domain A/B collections plus a separate aligned held-out test manifest."""
    rng = np.random.default_rng(0)
    counts = {
        "label_free": {"train": 3, "val": 2, "test": 2},
        "stained": {"train": 4, "val": 2, "test": 3},
    }
    for offset, (domain, splits) in enumerate(counts.items()):
        for split, count in splits.items():
            for index in range(count):
                path = dataset_root / "domains" / domain / split / f"{domain}_{index}.png"
                _write_noise(path, rng, 64 * offset)
    write_aligned_test_manifest(dataset_root, _SAMPLE_IDS)
    for sample_id in _SAMPLE_IDS:
        _write_noise(dataset_root / "splits" / "test" / f"{sample_id}_source.png", rng, 0)
        _write_noise(dataset_root / "splits" / "test" / f"{sample_id}_target.png", rng, 64)


def _config(tmp_path: Path, name: str, **overrides: Any) -> Path:
    data = cyclegan_config_data(tmp_path)
    data["training"].update(epochs=1, augmentation={"enabled": False})
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return write_config_data(tmp_path / f"{name}.yaml", data)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cyclegan_train_resume_infer_evaluate_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_dataset(tmp_path / "dataset")
    run_root = tmp_path / "results" / "cyclegan_run"
    checkpoints = run_root / "checkpoints"

    # Train, then resume the latest checkpoint into a freshly constructed runtime.
    run_stage(_config(tmp_path, "train"), "train")
    first = torch.load(checkpoints / "ep000.pth", map_location="cpu", weights_only=True)
    assert first["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert first["method"]["name"] == "cyclegan"
    run_stage(_config(tmp_path, "resume", training={"epochs": 2, "resume": "latest"}), "train")
    assert (checkpoints / "ep001.pth").is_file()
    assert [row["epoch"] for row in _rows(run_root / "metrics" / "epochs.csv")] == ["0", "1"]

    # Both directions infer from the same checkpoint into the shared output root.
    a_to_b = _config(tmp_path, "a_to_b", inference={"direction": "A_to_B"})
    b_to_a = _config(
        tmp_path,
        "b_to_a",
        inference={"direction": "B_to_A"},
        evaluation={"protocol": "paired", "output_dir": str(run_root / "evaluation_paired")},
    )
    run_stage(a_to_b, "infer")
    assert _json(run_root / "metadata" / "stages" / "infer.json")["details"][
        "checkpoint_path"
    ] == str(checkpoints / "ep001.pth")
    run_stage(b_to_a, "infer")
    output_test = run_root / "artifacts" / "output_test"
    assert sorted(path.name for path in output_test.iterdir()) == sorted(
        generated_filename(sample_id, ".png", direction)
        for sample_id in _SAMPLE_IDS
        for direction in ("A_to_B", "B_to_A")
    )

    # Recursive directory inference keeps relative folders; equal basenames never collide.
    inputs = tmp_path / "inputs"
    rng = np.random.default_rng(1)
    for folder in ("slide1", "slide2/nested"):
        _write_noise(inputs / folder / "tile.png", rng, 0)
    generated = tmp_path / "generated"
    infer_images(a_to_b, (f"label_free={inputs}",), generated, recursive=True)
    infer_images(b_to_a, (f"stained={inputs}",), generated, recursive=True)
    assert sorted(path.relative_to(generated).as_posix() for path in generated.rglob("*.png")) == [
        "slide1/tile_A_to_B_generated.png",
        "slide1/tile_B_to_A_generated.png",
        "slide2/nested/tile_A_to_B_generated.png",
        "slide2/nested/tile_B_to_A_generated.png",
    ]

    # Default CycleGAN protocol: unpaired collection diagnostics for the active direction only.
    run_stage(a_to_b, "evaluate")
    unpaired_dir = run_root / "evaluation"
    statistics = _rows(unpaired_dir / UNPAIRED_IMAGE_STATISTICS_CSV)
    generated_rows = [row for row in statistics if row["collection"] == "generated"]
    assert len(generated_rows) == len(_SAMPLE_IDS)
    assert all("_A_to_B_generated" in row["path"] for row in generated_rows)
    assert sum(row["collection"] == "reference" for row in statistics) == 3
    assert (unpaired_dir / UNPAIRED_FEATURE_COMPARISON_CSV).is_file()
    assert not (unpaired_dir / "per_image_metrics.csv").exists()
    unpaired = _json(unpaired_dir / "evaluation_metadata.json")
    assert unpaired["evaluation_protocol"] == "unpaired"
    assert unpaired["pairwise_metrics_available"] is False
    assert unpaired["reference_domain"] == "stained"
    assert unpaired["limitations"]

    # Explicit paired protocol against the aligned manifest: generated A vs real A.
    run_stage(b_to_a, "evaluate")
    paired_dir = run_root / "evaluation_paired"
    metrics = _rows(paired_dir / "per_image_metrics.csv")
    assert [row["sample_id"] for row in metrics] == _SAMPLE_IDS
    assert all("_B_to_A_generated" in row["generated_path"] for row in metrics)
    assert all(row["target_path"].endswith("_source.png") for row in metrics)
    paired = _json(paired_dir / "evaluation_metadata.json")
    assert paired["evaluation_protocol"] == "paired"
    assert paired["training_pairing"] == "unpaired"
    assert paired["pairwise_metrics_available"] is True
    assert paired["inference_direction"] == "B_to_A"
    assert paired["reference_domain"] == "label_free"
    assert "limitations" not in paired

    # Shared run provenance: one run record, per-stage records, snapshots, environments.
    assert _json(run_root / "metadata" / "run.json")["stages_present"] == [
        "train",
        "infer",
        "evaluate",
    ]
    for stage in ("train", "infer", "evaluate"):
        record = _json(run_root / "metadata" / "stages" / f"{stage}.json")
        assert record["status"] == "completed"
        assert record["config"]["sha256"]
        assert (run_root / "config" / stage / "resolved.yaml").is_file()
        assert (run_root / "metadata" / "environments" / f"{stage}.json").is_file()
