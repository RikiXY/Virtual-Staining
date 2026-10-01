from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from tests.checkpoint_helpers import write_ui_checkpoint as _write_checkpoint
from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    ComparisonRequest,
    GeneratedSampleEvaluationRequest,
    InferenceRequest,
    RunEvaluationRequest,
    SingleSampleRequest,
)
from virtual_staining.evaluation.summaries import write_summary_csv


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
    assert result.generated_path.name == "case_1_generated.png"
    assert result.generated_path.parent.name == model.target_domain
    with result.metrics_csv.open() as stream:
        assert next(csv.DictReader(stream))["output_name"] == model.target_domain


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
    metrics = [key for key in rows[0] if key not in {"sample_id", "target_path", "generated_path"}]
    fieldnames += ["output_name", *[f"{name}_status" for name in metrics]]
    for row in rows:
        row["output_name"] = "HE"
        row.update({f"{name}_status": "finite" for name in metrics})
    with (evaluation / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary_csv(
        rows, ("ssim", "psnr", "mae", "rmse", "mse", "pcc_rgb_mean", "pcc_gray"), evaluation
    )
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

    assert loaded.summary["HE/ssim"]["finite_mean"] == pytest.approx(0.6)
    assert len(loaded.representatives["HE/ssim"]) == 3
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


def test_run_reports_keep_outputs_and_non_numeric_statuses_separate(tmp_path: Path) -> None:
    from virtual_staining.evaluation.summaries import read_per_image_metrics_csv

    service = _service(tmp_path)
    run = _write_evaluated_run(tmp_path / "results", "multi", (0.5, 0.7))
    csv_path = run / "evaluation" / "per_image_metrics.csv"
    rows = read_per_image_metrics_csv(csv_path)
    rows += [{**row, "output_name": "PAS", "ssim": "0.1"} for row in rows]
    rows[0].update(pcc_gray="", pcc_gray_status="undefined")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary_csv(rows, service.supported_metrics, csv_path.parent)

    result = service.evaluate_run(
        RunEvaluationRequest(
            run_path=run,
            ensure_plots=True,
            build_representative_panels=False,
        )
    )

    assert result.summary["HE/ssim"]["finite_mean"] == pytest.approx(0.6)
    assert result.summary["PAS/ssim"]["finite_mean"] == pytest.approx(0.1)
    assert result.summary["HE/pcc_gray"]["undefined_count"] == 1
    assert all(sample.value == 0.1 for sample in result.representatives["PAS/ssim"])
    assert {p.name for p in result.plot_paths} >= {
        "HE__ssim_histogram.png",
        "PAS__ssim_histogram.png",
    }
    other = _write_evaluated_run(tmp_path / "results", "other", (0.6, 0.8))
    with pytest.raises(ApplicationError, match="Choose an output"):
        service.compare_runs(ComparisonRequest(run, other))
    comparison = service.compare_runs(ComparisonRequest(run, other, output_name="HE"))
    assert comparison.summary["paired_samples"] == 2
    assert comparison.output_directory.name == "HE"


def test_single_sample_serializes_undefined_metrics_as_statuses(tmp_path: Path) -> None:
    import json

    service = _service(tmp_path)
    model = service.discover_models().models[0]
    result = service.evaluate_sample(
        SingleSampleRequest(
            model.identifier,
            Image.new("RGB", (32, 32)),
            Image.new("RGB", (32, 32)),
            "source.png",
            "target.png",
        )
    )
    payload = json.loads((result.output_directory / "sample.json").read_text())
    assert payload["metrics"]["pcc_gray"]["status"] == "undefined"
    assert payload["metrics"]["pcc_gray"]["value"] is None
