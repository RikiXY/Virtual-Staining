"""Tests verifying that inference output naming and shapes match evaluator expectations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from tests.config_helpers import write_run_config, yaml_section
from tests.image_helpers import make_rgb_image, write_rgb_image, write_rgb_pair
from tests.manifest_helpers import make_manifest_record, write_manifest_csv
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.evaluator import evaluate_pair
from virtual_staining.evaluation.io import collect_image_files, extract_single_sample_id
from virtual_staining.inference import InferenceResult
from virtual_staining.inference.runner import (
    InferenceResult as RunnerInferenceResult,
)
from virtual_staining.inference.runner import (
    run_inference as _run_inference_impl,
)
from virtual_staining.inference.single import (
    SingleInferenceResult,
    run_image_directory_inference,
    run_image_path_inference,
    run_single_image_inference,
)
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.utils.dimensions import to_torchvision_hw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_SIZE = (32, 32)
_SAMPLE_ID = "00512_09216"


def test_inference_result_remains_package_export() -> None:
    assert InferenceResult is RunnerInferenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pair(directory: Path, prefix: str = _SAMPLE_ID, ext: str = ".png") -> None:
    write_rgb_pair(directory, prefix, size=_IMAGE_SIZE, ext=ext)


def _save_checkpoint(path: Path, image_size: tuple[int, int] = _IMAGE_SIZE) -> None:
    G = UNetGenerator()
    D = PatchGANDiscriminator()
    torch.save(
        {
            "format_version": 2,
            "generator_state_dict": G.state_dict(),
            "image_size": image_size,
            "epoch": 0,
            "normalization_contract": {
                "input_range": "[-1, 1]",
                "output_range": "[-1, 1]",
            },
            "architecture": {
                "name": "pix2pix",
                "generator": {
                    "class": "UNetGenerator",
                    "in_channels": G.in_channels,
                    "out_channels": G.out_channels,
                    "base_channels": G.base_channels,
                    "norm": G.norm,
                    "dropout": G.dropout,
                    "bilinear": G.bilinear,
                    "output_activation": "tanh",
                },
                "discriminator": {
                    "class": "PatchGANDiscriminator",
                    "in_channels": D.in_channels,
                    "ndf": D.ndf,
                    "norm": D.norm,
                    "use_sigmoid": D.use_sigmoid,
                },
            },
        },
        path,
    )


def _write_test_manifest(dataset_root: Path, test_dir: Path) -> None:
    records = []
    for source_path in sorted(test_dir.glob("*_source.*")):
        sample_id = source_path.stem[: -len("_source")]
        target_path = test_dir / f"{sample_id}_target{source_path.suffix}"
        if not target_path.exists():
            continue
        records.append(
            make_manifest_record(
                sample_id,
                "test",
                ext=source_path.suffix,
                input_path=source_path.relative_to(dataset_root),
                target_path=target_path.relative_to(dataset_root),
            )
        )
    write_manifest_csv(dataset_root, records)


def _write_non_test_manifest(dataset_root: Path) -> None:
    records = (
        make_manifest_record(
            _SAMPLE_ID,
            "val",
            ext=".png",
            input_path=Path(f"test/{_SAMPLE_ID}_source.png"),
            target_path=Path(f"test/{_SAMPLE_ID}_target.png"),
        ),
    )
    write_manifest_csv(dataset_root, records)


def _run_inference(
    checkpoint_path: Path | None,
    test_folder: str,
    output_folder: str,
    image_size: tuple[int, int] = _IMAGE_SIZE,
    checkpoint_policy: str | None = None,
    checkpoint_metric: str | None = None,
    checkpoint_rank: int | None = None,
) -> RunConfig:
    dataset_root = Path(test_folder).parent
    _write_test_manifest(dataset_root, Path(test_folder))
    inference_lines = []
    if checkpoint_path is not None:
        inference_lines.append(f"checkpoint_path: {checkpoint_path}")
    if checkpoint_policy is not None:
        inference_lines.append(f"checkpoint_policy: {checkpoint_policy}")
    if checkpoint_metric is not None:
        inference_lines.append(f"checkpoint_metric: {checkpoint_metric}")
    if checkpoint_rank is not None:
        inference_lines.append(f"checkpoint_rank: {checkpoint_rank}")
    inference_lines.append(f"output_dir: {Path(output_folder)}")
    section = (
        f"image_size: [{image_size[0]}, {image_size[1]}]\n"
        f"{yaml_section('inference', chr(10).join(inference_lines))}"
    )
    config_path = write_run_config(
        Path(output_folder).parent,
        section,
        filename="infer.yaml",
        dataset_root=dataset_root,
        results_path=Path(output_folder).parent,
        run_name="test_run",
    )
    config = RunConfig.from_yaml(config_path)
    _run_inference_impl(config, config_path)
    return config


def _write_inference_config(
    tmp_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    dataset_root: Path | None = None,
) -> Path:
    dataset_root = tmp_path if dataset_root is None else dataset_root
    inference_yaml = f"checkpoint_path: {checkpoint}\noutput_dir: {output_dir}"
    section = (
        f"image_size: [{_IMAGE_SIZE[0]}, {_IMAGE_SIZE[1]}]\n"
        f"{yaml_section('inference', inference_yaml)}"
    )
    return write_run_config(
        tmp_path,
        section,
        filename="infer.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path,
        run_name="test_run",
    )


def test_run_inference_raises_if_manifest_missing(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)

    config_path = _write_inference_config(tmp_path, checkpoint, output_dir)
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        _run_inference_impl(config, config_path)


def test_run_inference_raises_if_required_test_split_missing(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _write_non_test_manifest(tmp_path)
    _save_checkpoint(checkpoint)

    config_path = _write_inference_config(tmp_path, checkpoint, output_dir)
    config = RunConfig.from_yaml(config_path)

    with pytest.raises(ValueError, match="test"):
        _run_inference_impl(config, config_path)


def test_single_image_inference_writes_default_output_without_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "single_source.png"
    output_dir = tmp_path / "single_output"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_path, size=(64, 64))
    _save_checkpoint(checkpoint)
    config_path = _write_inference_config(tmp_path, checkpoint, output_dir)
    config = RunConfig.from_yaml(config_path)

    result = run_single_image_inference(config, input_path)

    expected_output = output_dir / "single_target_generated.png"
    assert result.output_path == expected_output
    assert result.mode == "tile"
    assert expected_output.exists()
    with Image.open(expected_output) as generated:
        assert generated.size == (64, 64)


def test_single_image_inference_resize_mode_writes_configured_size(tmp_path: Path) -> None:
    input_path = tmp_path / "single_source.png"
    output_dir = tmp_path / "single_output"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_path, size=(64, 64))
    _save_checkpoint(checkpoint)
    config_path = _write_inference_config(tmp_path, checkpoint, output_dir)
    config = RunConfig.from_yaml(config_path)

    result = run_single_image_inference(config, input_path, mode="resize")

    assert result.mode == "resize"
    with Image.open(result.output_path) as generated:
        assert generated.size == _IMAGE_SIZE


def test_single_image_inference_writes_explicit_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "arbitrary.png"
    output_path = tmp_path / "custom" / "generated.png"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_path, size=(64, 64))
    _save_checkpoint(checkpoint)
    config_path = _write_inference_config(tmp_path, checkpoint, tmp_path / "unused")
    config = RunConfig.from_yaml(config_path)

    result = run_single_image_inference(config, input_path, output_path)

    assert result.output_path == output_path
    assert output_path.exists()


def test_image_path_inference_file_uses_images_default_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "single_source.png"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_path, size=_IMAGE_SIZE)
    _save_checkpoint(checkpoint)
    section = (
        f"image_size: [{_IMAGE_SIZE[0]}, {_IMAGE_SIZE[1]}]\n"
        "inference:\n"
        f"  checkpoint_path: {checkpoint}\n"
    )
    config_path = write_run_config(
        tmp_path,
        section,
        filename="infer.yaml",
        dataset_root=tmp_path,
        results_path=tmp_path / "results",
        run_name="test_run",
    )
    config = RunConfig.from_yaml(config_path)

    result = run_image_path_inference(config, input_path)

    assert isinstance(result, SingleInferenceResult)
    assert result.output_path == (
        tmp_path
        / "results"
        / "test_run"
        / "artifacts"
        / "output_images"
        / "single_target_generated.png"
    )


def test_directory_image_inference_handles_mixed_formats(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_dir / "alpha_source.png", size=(64, 64))
    write_rgb_image(input_dir / "beta.jpg", size=(64, 64))
    write_rgb_image(input_dir / "gamma.bmp", size=(64, 64))
    (input_dir / "notes.txt").write_text("skip me\n", encoding="utf-8")
    _save_checkpoint(checkpoint)
    config_path = _write_inference_config(tmp_path, checkpoint, tmp_path / "unused")
    config = RunConfig.from_yaml(config_path)

    result = run_image_directory_inference(config, input_dir, output_dir)

    expected_outputs = {
        output_dir / "alpha_target_generated.png",
        output_dir / "beta_target_generated.jpg",
        output_dir / "gamma_target_generated.bmp",
    }
    assert {item.output_path for item in result.results} == expected_outputs
    assert result.output_dir == output_dir
    assert len(result.results) == 3
    for output_path in expected_outputs:
        assert output_path.exists()
        with Image.open(output_path) as generated:
            assert generated.size == (64, 64)


def test_directory_image_inference_can_force_output_format(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_image(input_dir / "alpha_source.jpg", size=(64, 64))
    write_rgb_image(input_dir / "nested" / "beta.bmp", size=(64, 64))
    _save_checkpoint(checkpoint)
    config_path = _write_inference_config(tmp_path, checkpoint, tmp_path / "unused")
    config = RunConfig.from_yaml(config_path)

    result = run_image_directory_inference(
        config,
        input_dir,
        output_dir,
        recursive=True,
        output_format="png",
    )

    assert {
        output_dir / "alpha_target_generated.png",
        output_dir / "nested" / "beta_target_generated.png",
    } == {item.output_path for item in result.results}


# ---------------------------------------------------------------------------
# Naming tests: inference output filename conventions
# ---------------------------------------------------------------------------


def test_output_filename_ends_with_target_generated_suffix(tmp_path: Path) -> None:
    """Inference output stem must end with '_target_generated' for the evaluator to find it."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    outputs = list(output_dir.iterdir())
    assert len(outputs) == 1, f"Expected 1 output file, got {len(outputs)}"

    stem = outputs[0].stem
    assert stem.endswith("_target_generated"), (
        f"Output stem '{stem}' does not end with '_target_generated'; "
        "the evaluator will not discover this file."
    )


def test_inference_writes_stage_scoped_snapshot_files(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)

    config = _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    run_root = config.project.run_root
    assert (run_root / "config" / "inference.input.yaml").exists()
    assert (run_root / "config" / "inference.resolved.yaml").exists()
    assert (run_root / "metadata" / "inference_config_hash.txt").exists()
    assert (run_root / "metadata" / "inference_environment.json").exists()
    assert not (run_root / "config" / "input.yaml").exists()
    assert not (run_root / "config" / "resolved.yaml").exists()
    assert not (run_root / "metadata" / "config_hash.txt").exists()


def test_inference_resolves_generic_best_checkpoint_policy_from_selection(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    run_root = tmp_path / "test_run"
    checkpoints_dir = run_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    checkpoint = checkpoints_dir / "ep001.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)
    (checkpoints_dir / "best.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {
                    "val_ssim": {
                        "mode": "max",
                        "top_k": 1,
                        "best": {
                            "rank": 1,
                            "epoch": 1,
                            "checkpoint_path": "ep001.pth",
                            "metric_value": 0.9,
                        },
                        "records": [
                            {
                                "rank": 1,
                                "epoch": 1,
                                "checkpoint_path": "ep001.pth",
                                "metric_value": 0.9,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = _run_inference(
        checkpoint_path=None,
        checkpoint_policy="best",
        checkpoint_metric="val_ssim",
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    metadata = json.loads(
        (config.project.run_root / "metadata" / "stages" / "infer.json").read_text(encoding="utf-8")
    )
    assert metadata["checkpoint_path"] == str(checkpoint)


def test_inference_resolves_top_k_checkpoint_policy_from_selection(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    run_root = tmp_path / "test_run"
    checkpoints_dir = run_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    checkpoint_1 = checkpoints_dir / "ep001.pth"
    checkpoint_2 = checkpoints_dir / "ep002.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint_1)
    _save_checkpoint(checkpoint_2)
    (checkpoints_dir / "best.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {
                    "val_ssim": {
                        "mode": "max",
                        "top_k": 2,
                        "best": {
                            "rank": 1,
                            "epoch": 1,
                            "checkpoint_path": "ep001.pth",
                            "metric_value": 0.9,
                        },
                        "records": [
                            {
                                "rank": 1,
                                "epoch": 1,
                                "checkpoint_path": "ep001.pth",
                                "metric_value": 0.9,
                            },
                            {
                                "rank": 2,
                                "epoch": 2,
                                "checkpoint_path": "ep002.pth",
                                "metric_value": 0.8,
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = _run_inference(
        checkpoint_path=None,
        checkpoint_policy="top_k",
        checkpoint_metric="val_ssim",
        checkpoint_rank=2,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    metadata = json.loads(
        (config.project.run_root / "metadata" / "stages" / "infer.json").read_text(encoding="utf-8")
    )
    assert metadata["checkpoint_path"] == str(checkpoint_2)


def test_inference_best_raises_when_best_record_missing(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"

    _write_pair(test_dir)

    with pytest.raises(FileNotFoundError, match="best.json"):
        _run_inference(
            checkpoint_path=None,
            checkpoint_policy="best",
            checkpoint_metric="val_ssim",
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_best_raises_when_best_record_invalid(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    run_root = tmp_path / "test_run"
    checkpoints_dir = run_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    _write_pair(test_dir)
    (checkpoints_dir / "best.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        _run_inference(
            checkpoint_path=None,
            checkpoint_policy="best",
            checkpoint_metric="val_ssim",
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_writes_stage_metadata_json(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)

    config = _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    metadata_path = config.project.run_root / "metadata" / "stages" / "infer.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = config.project.manifest_path
    expected_manifest_hash = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"

    assert metadata["stage"] == "infer"
    assert metadata["status"] == "completed"
    assert metadata["completed_at"]
    assert metadata["started_at"]
    assert metadata["checkpoint_path"] == str(checkpoint)
    assert metadata["manifest_path"] == str(manifest_path)
    assert metadata["manifest_sha256"] == expected_manifest_hash
    assert metadata["output_dir"] == str(output_dir)
    assert metadata["test_sample_count"] == 1
    assert metadata["inferred_count"] == 1

    events = [
        json.loads(line)
        for line in (config.project.run_root / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
    assert all(event["stage"] == "infer" for event in events)


def test_inference_preserves_existing_training_snapshot_files(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)

    run_root = tmp_path / "test_run"
    config_dir = run_root / "config"
    metadata_dir = run_root / "metadata"
    config_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (config_dir / "input.yaml").write_text("train input\n", encoding="utf-8")
    (config_dir / "resolved.yaml").write_text("train resolved\n", encoding="utf-8")
    (metadata_dir / "config_hash.txt").write_text("sha256:train\n", encoding="utf-8")

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    assert (config_dir / "input.yaml").read_text(encoding="utf-8") == "train input\n"
    assert (config_dir / "resolved.yaml").read_text(encoding="utf-8") == "train resolved\n"
    assert (metadata_dir / "config_hash.txt").read_text(encoding="utf-8") == "sha256:train\n"


def test_output_sample_id_matches_target(tmp_path: Path) -> None:
    """extract_single_sample_id must succeed and return the correct ID for an inference output."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir, prefix=_SAMPLE_ID)
    _save_checkpoint(checkpoint)

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    generated_path = next(output_dir.iterdir())
    target_path = test_dir / f"{_SAMPLE_ID}_target.png"

    sample_id = extract_single_sample_id(target_path, generated_path)
    assert sample_id == _SAMPLE_ID


def test_collect_image_files_finds_inference_output(tmp_path: Path) -> None:
    """collect_image_files must index '_target_generated' outputs by sample ID."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir, prefix=_SAMPLE_ID)
    _save_checkpoint(checkpoint)

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    generated_files = collect_image_files(output_dir, "_target_generated", "Generated")
    assert _SAMPLE_ID in generated_files, (
        f"Sample ID '{_SAMPLE_ID}' not found among collected files: {sorted(generated_files)}"
    )


# ---------------------------------------------------------------------------
# Evaluator tests: shape validation
# ---------------------------------------------------------------------------


def test_evaluate_pair_raises_on_shape_mismatch(tmp_path: Path) -> None:
    """evaluate_pair must raise ValueError when target and generated have different shapes."""
    target_path = tmp_path / f"{_SAMPLE_ID}_target.png"
    generated_path = tmp_path / f"{_SAMPLE_ID}_target_generated.png"

    write_rgb_image(target_path, size=(32, 32))
    write_rgb_image(generated_path, size=(64, 64))

    with pytest.raises(ValueError, match="same shape"):
        evaluate_pair(target_path, generated_path)


def test_evaluate_pair_returns_all_metrics_for_matching_shapes(tmp_path: Path) -> None:
    """evaluate_pair must return a complete metrics dict when target and generated match."""
    target_path = tmp_path / f"{_SAMPLE_ID}_target.png"
    generated_path = tmp_path / f"{_SAMPLE_ID}_target_generated.png"

    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(target_path)
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(generated_path)

    metrics, shape = evaluate_pair(target_path, generated_path)

    expected_keys = {
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "pcc_gray",
        "pcc_r",
        "pcc_g",
        "pcc_b",
        "pcc_rgb_mean",
    }
    assert set(metrics.keys()) == expected_keys
    assert shape == (32, 32, 3)
    assert 0.0 <= metrics["mae"] <= 1.0
    assert metrics["mse"] >= 0.0
    assert metrics["psnr"] > 0.0


# ---------------------------------------------------------------------------
# End-to-end: inference output is directly consumable by the evaluator
# ---------------------------------------------------------------------------


def test_inference_output_is_evaluable_end_to_end(tmp_path: Path) -> None:
    """Full round-trip: inference output can be paired with its target and fully evaluated."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir, prefix=_SAMPLE_ID)
    _save_checkpoint(checkpoint)

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=_IMAGE_SIZE,
    )

    target_path = test_dir / f"{_SAMPLE_ID}_target.png"
    generated_path = output_dir / f"{_SAMPLE_ID}_target_generated.png"

    assert generated_path.exists(), (
        f"Inference did not produce the expected file at {generated_path}"
    )

    sample_id = extract_single_sample_id(target_path, generated_path)
    assert sample_id == _SAMPLE_ID

    metrics, shape = evaluate_pair(target_path, generated_path)
    assert shape[2] == 3
    assert set(metrics.keys()) >= {"mae", "mse", "ssim"}


# ---------------------------------------------------------------------------
# Dimension-order: to_torchvision_hw correctness
# ---------------------------------------------------------------------------


def test_to_torchvision_hw_produces_correct_tensor_shape() -> None:
    """Resize via to_torchvision_hw(width, height) must yield a tensor of shape (C, H, W)."""
    width, height = 48, 32  # deliberately non-square
    wh = (width, height)
    resize = transforms.Resize(to_torchvision_hw(wh))

    img = make_rgb_image(size=(64, 64))
    tensor = transforms.ToTensor()(resize(img))

    assert tensor.shape == (3, height, width), (
        f"Expected tensor shape (3, {height}, {width}), got {tuple(tensor.shape)}"
    )


def test_inference_non_square_produces_correct_output_shape(tmp_path: Path) -> None:
    """Inference with a non-square image_size must produce tensors of the intended (H, W)."""
    width, height = 48, 32
    image_size = (width, height)

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    write_rgb_pair(test_dir, _SAMPLE_ID, size=(64, 64))
    _save_checkpoint(checkpoint, image_size=image_size)

    _run_inference(
        checkpoint_path=checkpoint,
        test_folder=str(test_dir),
        output_folder=str(output_dir),
        image_size=image_size,
    )

    generated = next(output_dir.iterdir())
    out_img = Image.open(generated)
    assert out_img.size == (width, height), (
        f"Expected output image size (width={width}, height={height}), got PIL size {out_img.size}"
    )


# ---------------------------------------------------------------------------
# Architecture metadata: inference contract
# ---------------------------------------------------------------------------


def test_inference_raises_on_missing_architecture_metadata(tmp_path: Path) -> None:
    """test_inference must raise ValueError when the checkpoint has no architecture metadata."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt.pop("architecture")
    torch.save(ckpt, checkpoint)

    with pytest.raises(ValueError, match="architecture metadata"):
        _run_inference(
            checkpoint_path=checkpoint,
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_raises_on_architecture_mismatch(tmp_path: Path) -> None:
    """test_inference must raise ValueError when checkpoint generator arch doesn't match."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)

    G = UNetGenerator(base_channels=32)
    D = PatchGANDiscriminator()
    torch.save(
        {
            "format_version": 2,
            "generator_state_dict": G.state_dict(),
            "image_size": _IMAGE_SIZE,
            "epoch": 0,
            "normalization_contract": {
                "input_range": "[-1, 1]",
                "output_range": "[-1, 1]",
            },
            "architecture": {
                "name": "pix2pix",
                "generator": {
                    "class": "UNetGenerator",
                    "in_channels": G.in_channels,
                    "out_channels": G.out_channels,
                    "base_channels": G.base_channels,
                    "norm": G.norm,
                    "dropout": G.dropout,
                    "bilinear": G.bilinear,
                    "output_activation": "tanh",
                },
                "discriminator": {
                    "class": "PatchGANDiscriminator",
                    "in_channels": D.in_channels,
                    "ndf": D.ndf,
                    "norm": D.norm,
                    "use_sigmoid": D.use_sigmoid,
                },
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="base_channels"):
        _run_inference(
            checkpoint_path=checkpoint,
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_raises_on_checkpoint_format_version_mismatch(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["format_version"] = 1
    torch.save(ckpt, checkpoint)

    with pytest.raises(ValueError, match="format version"):
        _run_inference(
            checkpoint_path=checkpoint,
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_raises_on_output_activation_mismatch(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["architecture"]["generator"]["output_activation"] = "sigmoid"
    torch.save(ckpt, checkpoint)

    with pytest.raises(ValueError, match="output_activation"):
        _run_inference(
            checkpoint_path=checkpoint,
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


def test_inference_raises_on_normalization_contract_mismatch(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    _save_checkpoint(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["normalization_contract"] = {"input_range": "[0, 1]", "output_range": "[0, 1]"}
    torch.save(ckpt, checkpoint)

    with pytest.raises(ValueError, match="normalization_contract"):
        _run_inference(
            checkpoint_path=checkpoint,
            test_folder=str(test_dir),
            output_folder=str(output_dir),
            image_size=_IMAGE_SIZE,
        )


# ---------------------------------------------------------------------------
# Non-finite metric values: identical and constant images
# ---------------------------------------------------------------------------


def test_evaluate_pair_identical_images_returns_psnr_inf(tmp_path: Path) -> None:
    """evaluate_pair on identical images must return PSNR=inf without crashing."""
    arr = np.full((*_IMAGE_SIZE, 3), 128, dtype=np.uint8)
    target_path = tmp_path / f"{_SAMPLE_ID}_target.png"
    generated_path = tmp_path / f"{_SAMPLE_ID}_target_generated.png"
    Image.fromarray(arr).save(target_path)
    Image.fromarray(arr).save(generated_path)

    metrics, _ = evaluate_pair(target_path, generated_path)

    assert math.isinf(metrics["psnr"]), (
        f"Expected PSNR=inf for identical images, got {metrics['psnr']}"
    )


def test_evaluate_pair_constant_target_returns_pcc_nan(tmp_path: Path) -> None:
    """evaluate_pair with a constant-value target must return PCC=nan without crashing."""
    target = np.zeros((*_IMAGE_SIZE, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    generated = rng.integers(1, 256, (*_IMAGE_SIZE, 3), dtype=np.uint8)

    target_path = tmp_path / f"{_SAMPLE_ID}_target.png"
    generated_path = tmp_path / f"{_SAMPLE_ID}_target_generated.png"
    Image.fromarray(target).save(target_path)
    Image.fromarray(generated).save(generated_path)

    metrics, _ = evaluate_pair(target_path, generated_path)

    assert math.isnan(metrics["pcc_gray"]), (
        f"Expected PCC=nan for constant target, got {metrics['pcc_gray']}"
    )
