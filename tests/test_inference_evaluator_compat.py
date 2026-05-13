"""Tests verifying that inference output naming and shapes are compatible with the evaluator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.io import collect_image_files, extract_single_sample_id
from virtual_staining.evaluation.metrics import evaluate_pair
from virtual_staining.inference.runner import run_inference as _run_inference_impl
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.utils.dimensions import to_torchvision_hw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_SIZE = (32, 32)
_SAMPLE_ID = "00512_09216"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pair(directory: Path, prefix: str = _SAMPLE_ID, ext: str = ".png") -> None:
    arr = np.zeros((*_IMAGE_SIZE, 3), dtype=np.uint8)
    Image.fromarray(arr).save(directory / f"{prefix}_source{ext}")
    Image.fromarray(arr).save(directory / f"{prefix}_target{ext}")


def _save_checkpoint(path: Path, image_size: tuple[int, int] = _IMAGE_SIZE) -> None:
    G = UNetGenerator()
    D = PatchGANDiscriminator()
    torch.save(
        {
            "generator_state_dict": G.state_dict(),
            "image_size": image_size,
            "epoch": 0,
            "architecture": {
                "name": "pix2pix",
                "gan_loss": "bce",
                "generator": {
                    "class": "UNetGenerator",
                    "in_channels": G.in_channels,
                    "out_channels": G.out_channels,
                    "base_channels": G.base_channels,
                    "norm": G.norm,
                    "dropout": G.dropout,
                    "bilinear": G.bilinear,
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


def _run_inference(
    checkpoint_path: Path,
    test_folder: str,
    output_folder: str,
    image_size: tuple[int, int] = _IMAGE_SIZE,
) -> RunConfig:
    config_path = Path(output_folder).parent / "infer.yaml"
    config_path.write_text(
        f"""
dataset_root: {Path(test_folder).parent}
results_path: {Path(output_folder).parent}
run_name: test_run
image_size: [{image_size[0]}, {image_size[1]}]
inference:
  checkpoint_path: {checkpoint_path}
  test_dir: {Path(test_folder)}
  output_dir: {Path(output_folder)}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig.from_yaml(config_path)
    _run_inference_impl(config, config_path)
    return config


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

    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(target_path)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(generated_path)

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

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
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

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(test_dir / f"{_SAMPLE_ID}_source.png")
    Image.fromarray(arr).save(test_dir / f"{_SAMPLE_ID}_target.png")
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
# Architecture metadata: inference compatibility
# ---------------------------------------------------------------------------


def test_inference_raises_on_missing_architecture_metadata(tmp_path: Path) -> None:
    """test_inference must raise ValueError when the checkpoint has no architecture metadata."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "ep000.pth"

    _write_pair(test_dir)
    G = UNetGenerator()
    torch.save(
        {"generator_state_dict": G.state_dict(), "image_size": _IMAGE_SIZE, "epoch": 0},
        checkpoint,
    )

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
            "generator_state_dict": G.state_dict(),
            "image_size": _IMAGE_SIZE,
            "epoch": 0,
            "architecture": {
                "name": "pix2pix",
                "gan_loss": "bce",
                "generator": {
                    "class": "UNetGenerator",
                    "in_channels": G.in_channels,
                    "out_channels": G.out_channels,
                    "base_channels": G.base_channels,
                    "norm": G.norm,
                    "dropout": G.dropout,
                    "bilinear": G.bilinear,
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
