from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    ComparisonRequest,
    GeneratedSampleEvaluationRequest,
    InferenceRequest,
    RunEvaluationRequest,
    SingleSampleRequest,
)
from virtual_staining.checkpoint_contract import (
    CHECKPOINT_FORMAT_VERSION,
    NORMALIZATION_CONTRACT,
    make_arch_metadata,
)
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator


def _write_checkpoint(path: Path) -> None:
    generator = ConcatUNetGenerator(("label_free",), base_channels=4)
    discriminator = PatchGANDiscriminator(in_channels=6, ndf=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": make_arch_metadata(generator, discriminator, target_modality="stained"),
            "normalization_contract": NORMALIZATION_CONTRACT,
            "generator_state_dict": generator.state_dict(),
            "image_size": (32, 32),
        },
        path,
    )


def _service(tmp_path: Path) -> ApplicationService:
    _write_checkpoint(tmp_path / "checkpoints" / "model.pth")
    return ApplicationService(
        Path("checkpoints"),
        Path("outputs"),
        Path("results"),
        working_directory=tmp_path,
    )


def test_public_api_invokes_inference_without_nicegui(tmp_path: Path) -> None:
    service = _service(tmp_path)
    model = service.discover_models().models[0]

    result = service.run_inference(
        InferenceRequest(model.identifier, Image.new("RGB", (32, 32)), "source.png")
    )

    assert result.generated_image.size == (32, 32)
    assert result.provenance.model_identifier == "model.pth"


def test_single_sample_evaluation_uses_shared_api_and_writes_artifacts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    model = service.discover_models().models[0]

    result = service.evaluate_sample(
        SingleSampleRequest(
            model_identifier=model.identifier,
            source_image=Image.new("RGB", (32, 32), (40, 80, 120)),
            target_image=Image.new("RGB", (32, 32), (80, 40, 120)),
            source_filename="case 1.png",
            target_filename="case 1 target.png",
        )
    )

    assert set(("ssim", "psnr", "mae", "rmse", "mse", "pcc_rgb_mean")) <= set(result.metrics)
    assert result.difference_map.mode == "L"
    assert result.metrics_csv.is_file()
    assert result.generated_path.name == "case_1_target_generated.png"


def test_target_can_be_loaded_and_evaluated_after_inference(tmp_path: Path) -> None:
    service = _service(tmp_path)
    model = service.discover_models().models[0]
    inference = service.run_inference(
        InferenceRequest(
            model.identifier,
            Image.new("RGB", (32, 32), (40, 80, 120)),
            "late-target.png",
        )
    )

    result = service.evaluate_generated_sample(
        GeneratedSampleEvaluationRequest(
            inference=inference,
            target_image=Image.new("RGB", (32, 32), (80, 40, 120)),
            target_filename="target-loaded-afterwards.png",
        )
    )

    assert result.inference.generated_image is inference.generated_image
    assert result.metrics_csv.is_file()


def test_public_api_reports_invalid_inputs_as_controlled_errors(tmp_path: Path) -> None:
    service = _service(tmp_path)
    model = service.discover_models().models[0]

    with pytest.raises(ApplicationError, match="Expected input size"):
        service.run_inference(
            InferenceRequest(model.identifier, Image.new("RGB", (64, 32)), "wrong.png")
        )

    with pytest.raises(ApplicationError, match="exactly one"):
        service.evaluate_run(RunEvaluationRequest())


def _write_evaluated_run(root: Path, name: str, values: tuple[float, float]) -> Path:
    run = root / name
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    fieldnames = [
        "sample_id",
        "target_path",
        "generated_path",
        "ssim",
        "psnr",
        "mae",
        "rmse",
        "mse",
        "pcc_rgb_mean",
        "pcc_gray",
    ]
    for index, value in enumerate(values):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "target_path": "",
                "generated_path": "",
                "ssim": value,
                "psnr": 20 + value,
                "mae": 1 - value,
                "rmse": 1 - value,
                "mse": 1 - value,
                "pcc_rgb_mean": value,
                "pcc_gray": value,
            }
        )
    with (evaluation / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary_csv(rows, evaluation)
    return run


def test_run_loading_and_comparison_are_exposed_by_public_api(tmp_path: Path) -> None:
    service = _service(tmp_path)
    run_a = _write_evaluated_run(tmp_path / "results", "a", (0.5, 0.7))
    run_b = _write_evaluated_run(tmp_path / "results", "b", (0.6, 0.8))

    loaded = service.evaluate_run(
        RunEvaluationRequest(
            run_path=run_a,
            ensure_plots=False,
            build_representative_panels=False,
        )
    )
    compared = service.compare_runs(
        ComparisonRequest(run_a=run_a, run_b=run_b, metric="ssim", mode="paired")
    )

    assert loaded.summary["ssim"]["mean"] == pytest.approx(0.6)
    assert len(loaded.representatives["ssim"]) == 3
    assert compared.summary["paired_samples"] == 2
    assert compared.summary["favors"] == "b"
    assert compared.plot_paths


def test_comparison_can_switch_paired_unpaired_and_back(tmp_path: Path) -> None:
    service = _service(tmp_path)
    run_a = _write_evaluated_run(tmp_path / "results", "a", (0.5, 0.7))
    run_b = _write_evaluated_run(tmp_path / "results", "b", (0.6, 0.8))

    results = [
        service.compare_runs(ComparisonRequest(run_a=run_a, run_b=run_b, metric="ssim", mode=mode))
        for mode in ("paired", "unpaired", "paired")
    ]

    assert [result.mode for result in results] == ["paired", "unpaired", "paired"]
    assert all(result.plot_paths for result in results)
