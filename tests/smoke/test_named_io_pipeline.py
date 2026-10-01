"""CPU end-to-end runs of the N-input -> M-output contract on synthetic slides.

The synthetic targets are fixed colour transforms of the inputs; these runs prove the
software contract (naming, ordering, masks, provenance, checkpoints, evaluation) only and
say nothing about staining quality or any benefit of predicting several outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
import yaml

from tests.config_helpers import write_config_data
from virtual_staining.applications.inventory_authoring import (
    InventoryRequest,
    preview_inventory,
    write_inventory,
)
from virtual_staining.applications.pipeline import run_stage
from virtual_staining.checkpoint_contract import CHECKPOINT_FORMAT_VERSION
from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.runner import load_inference_generator
from virtual_staining.utils.artifacts import generated_path

_SIZE = 128
_OFFSETS = {"LF": (0, 0, 0), "AF": (30, 10, 0), "HE": (12, 6, 18), "PAS": (40, 0, 25)}


def _white_mask(img: np.ndarray, _params: object) -> np.ndarray:
    return np.full(img.shape[:2], 255, dtype=np.uint8)


def _write_raw_dataset(root: Path, inputs: tuple[str, ...], targets: tuple[str, ...]) -> None:
    """One slide set; every target has its own mask (PAS covers only the left half)."""
    y, x = np.indices((_SIZE, _SIZE), dtype=np.uint16)
    base = np.stack([(x * 3 + y * 5) % 180, (x + 2 * y) % 170, (x + y) % 160], axis=-1)
    for name in (*inputs, *targets):
        image = np.clip(base + np.array(_OFFSETS[name]), 0, 255).astype(np.uint8)
        (root / "raw" / name).mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(root / "raw" / name / "S1.tif"), image)
    for name in targets:
        mask = np.full((_SIZE, _SIZE), 255, dtype=np.uint8)
        if name == "PAS":
            mask[:, _SIZE // 2 :] = 0
        (root / "masks" / name).mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(root / "masks" / name / "S1.tif"), mask)
    # The synthetic assets share one grid: every non-reference asset is declared aligned.
    declared = [f"input__{name}_aligned" for name in inputs[1:]]
    declared += [f"target__{name}_aligned" for name in targets]
    (root / "meta.csv").write_text(
        "key,patient_id," + ",".join(declared) + "\n"
        "S1.tif,patient-1," + ",".join("true" for _ in declared) + "\n",
        encoding="utf-8",
    )
    request = InventoryRequest(
        dataset_root=root,
        inputs=tuple((name, f"raw/{name}") for name in inputs),
        targets=tuple((name, f"raw/{name}") for name in targets),
        reference=inputs[0],
        target_masks=tuple((name, f"masks/{name}") for name in targets),
        metadata=Path("meta.csv"),
    )
    preview = preview_inventory(request)
    assert preview.valid, preview.issues
    write_inventory(preview)


def _config(
    tmp_path: Path,
    inputs: tuple[str, ...],
    targets: tuple[str, ...],
    outputs: tuple[str, ...],
    **training: Any,
) -> Path:
    data: dict[str, Any] = {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "runs"),
        "run_name": "named_io",
        "image_size": [32, 32],
        # One patch-split slide set: no biological independence is claimed. Constant
        # synthetic mask patches repeat byte-for-byte, so only membership is hashed.
        "data": {"group_validation": "unavailable", "hash_policy": "membership"},
        "preprocessing": {
            "inputs": {
                "inventory": "inputs/slide_sets.csv",
                "modalities": list(inputs),
                "reference": inputs[0],
                "target_modalities": list(targets),
            },
            "patching": {"patch_size": [32, 32], "grid_movement": [32, 32], "margin": 0},
            "masks": {"save_patch_masks": True},
            "filtering": {
                "foreground": {"min_ratio": 0.0},
                "max_white_ratio": 1.0,
                "max_largest_white_component_ratio": 1.0,
            },
            "split": {"unit": "patch", "train": 0.6, "val": 0.2, "test": 0.2, "seed": 3},
            "io": {"tiled": False},
        },
        "model": {
            "inputs": list(inputs),
            "outputs": list(outputs),
            "generator": {"base_channels": 4},
            "discriminator": {"ndf": 4},
        },
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "seed": 11,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
            "log_rate": 1,
            "losses": {
                "generator": [
                    {"name": "adversarial_bce", "weight": 1.0},
                    # Target-specific masks: each output's L1 uses only its own mask.
                    {"name": "l1", "weight": 25.0, "params": {"mask": {"enabled": True}}},
                ],
                "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
            },
            **training,
        },
        "inference": {"checkpoint_policy": "latest"},
        "evaluation": {"bootstrap_iterations": 10},
    }
    return write_config_data(tmp_path / "run.yaml", data)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_and_train(tmp_path: Path, config_path: Path) -> RunLayout:
    with patch(
        "virtual_staining.data.slide_set_processor.calculate_mask_with_multiple_parameters",
        side_effect=_white_mask,
    ):
        run_stage(config_path, "prepare")
    run_stage(config_path, "train")
    return RunLayout.from_project(RunConfig.from_yaml(config_path).project)


@pytest.mark.parametrize(
    ("inputs", "targets", "outputs"),
    [(("LF",), ("HE",), ("HE",)), (("LF", "AF"), ("HE",), ("HE",))],
)
def test_one_output_runs_prepare_train_infer_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: tuple[str, ...],
    targets: tuple[str, ...],
    outputs: tuple[str, ...],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_raw_dataset(tmp_path / "dataset", inputs, targets)
    config_path = _config(tmp_path, inputs, targets, outputs)
    layout = _prepare_and_train(tmp_path, config_path)
    run_stage(config_path, "infer")
    run_stage(config_path, "evaluate")

    payload = torch.load(layout.checkpoints_dir / "ep000.pth", weights_only=True)
    assert (payload["method"]["inputs"], payload["method"]["outputs"]) == (list(inputs), ["HE"])
    generated = sorted(p.parent.name for p in layout.output_test_dir.rglob("*_generated.tif"))
    assert generated and set(generated) == {"HE"}
    rows = _rows(layout.evaluation_dir / "per_image_metrics.csv")
    assert rows and {row["output_name"] for row in rows} == {"HE"}
    history = _rows(layout.epochs_csv)[0]
    assert "val_ssim__HE" in history and "loss_train_raw_generator_l1__HE" in history


def test_two_inputs_two_outputs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    inputs, targets, outputs = ("LF", "AF"), ("HE", "PAS"), ("PAS", "HE")
    dataset_root = tmp_path / "dataset"
    _write_raw_dataset(dataset_root, inputs, targets)
    config_path = _config(tmp_path, inputs, targets, outputs)

    # Raw inventory -> prepare -> manifest v4.
    layout = _prepare_and_train(tmp_path, config_path)
    config = RunConfig.from_yaml(config_path)
    manifest = load_manifest_or_raise(config.project)
    assert manifest.metadata.to_dict() == {
        "schema_version": "4.0",
        "input_modalities": ["LF", "AF"],
        "target_modalities": ["HE", "PAS"],
        "reference_modality": "LF",
        "coordinate_space": "reference_level0_pixels",
        "pixel_center": "integer",
    }
    header = (dataset_root / "manifests" / "manifest.csv").read_text().splitlines()[0]
    assert header.split(",")[3:9] == [
        "input__LF",
        "input__AF",
        "target__HE",
        "target__PAS",
        "foreground_mask__HE",
        "foreground_mask__PAS",
    ]
    masks: dict[tuple[str, str], np.ndarray] = {}
    for record in manifest.records:
        for name, path in record.foreground_mask_paths.items():
            assert path is not None
            mask = cv2.imread(str(dataset_root / path), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            masks[record.sample_id, name] = mask
    assert len(masks) == 2 * len(manifest.records)
    # Each target keeps its own mask: PAS is empty wherever its raw mask was.
    assert any(
        not np.array_equal(masks[record.sample_id, "HE"], masks[record.sample_id, "PAS"])
        for record in manifest.records
    )
    assert (dataset_root / "manifests" / "slide_sets.csv").read_text().count("PAS__alignment") == 2

    # Dataset loading in the configured output order with target-specific masks.
    sample = PairedManifestDataset(
        manifest, input_names=inputs, target_names=outputs, include_foreground_mask=True
    )[0]
    assert tuple(sample["targets"]) == outputs
    assert tuple(sample["masks"]["foreground_mask"]) == outputs

    # One training/validation cycle produced per-output columns and a v4 checkpoint.
    history = _rows(layout.epochs_csv)
    assert [row["epoch"] for row in history] == ["0"]
    for output in outputs:
        assert history[0][f"val_ssim__{output}"]
        assert history[0][f"loss_val_raw_generator_l1__{output}"]
    assert "val_ssim" not in history[0]
    payload = torch.load(layout.checkpoints_dir / "ep000.pth", weights_only=True)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["method"]["outputs"] == ["PAS", "HE"]
    train_snapshot = _json(
        Path(_json(layout.stage_record("train"))["consumed_data"]["metadata_path"])
    )
    assert train_snapshot["selection"]["targets"] == ["PAS", "HE"]

    # Checkpoint validation/load and the resume path.
    generator, _ = load_inference_generator(config, layout, torch.device("cpu"))
    assert generator.output_names == outputs
    resume = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume["training"].update(epochs=2, resume="latest")
    resume_path = write_config_data(tmp_path / "resume.yaml", resume)
    run_stage(resume_path, "train")
    assert [row["epoch"] for row in _rows(layout.epochs_csv)] == ["0", "1"]

    # Inference publishes every (sample_id, output_name) artifact.
    run_stage(resume_path, "infer")
    test_records = manifest.filter_split("test").records
    for test_record in test_records:
        for output in outputs:
            assert generated_path(
                layout.output_test_dir, test_record.sample_id, output, ".tif"
            ).is_file()
    produced = _json(Path(_json(layout.stage_record("infer"))["produced_data"]["metadata_path"]))
    assert produced["selection"]["output_domains"] == ["PAS", "HE"]
    assert produced["row_count"] == 2 * len(test_records)
    assert _json(layout.stage_record("infer"))["details"]["checkpoint_path"].endswith("ep001.pth")

    # Paired evaluation per output, never pooled.
    run_stage(resume_path, "evaluate")
    rows = _rows(layout.evaluation_dir / "per_image_metrics.csv")
    assert [(row["sample_id"], row["output_name"]) for row in rows] == [
        (test_record.sample_id, output) for test_record in test_records for output in outputs
    ]
    summary = _rows(layout.evaluation_dir / "summary.csv")
    assert {row["output_name"] for row in summary} == {"PAS", "HE"}
    assert all(int(row["count"]) == len(test_records) for row in summary)
    metadata = _json(layout.evaluation_dir / "evaluation_metadata.json")
    assert metadata["reference_domains"] == ["PAS", "HE"]
    assert metadata["generated_producer"]["status"] == "linked"
