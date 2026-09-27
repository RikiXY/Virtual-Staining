from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from virtual_staining.applications import compare as compare_app
from virtual_staining.applications.pipeline import run_stage
from virtual_staining.cli._output import color_for_metric


def _metrics_csv(run_path: Path) -> Path:
    path = run_path / "evaluation" / "per_image_metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("sample_id,ssim\na,0.9\n", encoding="utf-8")
    return path


def test_compare_resolves_run_inputs_and_defaults(tmp_path: Path) -> None:
    run_a = tmp_path / "results" / "a"
    run_b = tmp_path / "results" / "b"
    _metrics_csv(run_a)
    _metrics_csv(run_b)

    resolved = compare_app._resolve_request(
        compare_app.CompareRequest(mode="paired", run_a=run_a, run_b=run_b)
    )

    assert resolved.csv_a == run_a / "evaluation" / "per_image_metrics.csv"
    assert resolved.csv_b == run_b / "evaluation" / "per_image_metrics.csv"
    assert resolved.output_dir == tmp_path / "results" / "comparisons" / "a_vs_b" / "paired_ssim"
    assert resolved.higher_is_better is True
    assert resolved.thresholds == (0.65, 0.75, 0.85)
    assert (resolved.min_value, resolved.max_value) == (0.0, 1.0)


def test_compare_resolves_explicit_canonical_csv_inputs_to_shared_results(
    tmp_path: Path,
) -> None:
    csv_a = _metrics_csv(tmp_path / "results" / "a")
    csv_b = _metrics_csv(tmp_path / "results" / "b")

    resolved = compare_app._resolve_request(
        compare_app.CompareRequest(mode="paired", csv_a=csv_a, csv_b=csv_b)
    )

    assert resolved.output_dir == tmp_path / "results" / "comparisons" / "a_vs_b" / "paired_ssim"


def test_compare_run_input_does_not_probe_historical_metrics_path(tmp_path: Path) -> None:
    run_a = tmp_path / "results" / "a"
    run_b = tmp_path / "results" / "b"
    historical = run_a / "metrics" / "per_image_metrics.csv"
    historical.parent.mkdir(parents=True)
    historical.write_text("sample_id,ssim\na,0.9\n", encoding="utf-8")
    _metrics_csv(run_b)

    with pytest.raises(
        FileNotFoundError,
        match=f"Expected: {run_a / 'evaluation' / 'per_image_metrics.csv'}",
    ):
        compare_app._resolve_request(
            compare_app.CompareRequest(mode="paired", run_a=run_a, run_b=run_b)
        )


def _custom_csv(run_path: Path, *, metadata: bool) -> Path:
    path = run_path / "evaluation" / "per_image_metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("sample_id,bias\na,2.0\nb,5.0\n", encoding="utf-8")
    if metadata:
        (path.parent / "evaluation_result.json").write_text(
            json.dumps(
                {"schema_version": 1, "metrics": [{"name": "bias", "higher_is_better": False}]}
            ),
            encoding="utf-8",
        )
    return path


def test_compare_custom_metric_uses_recorded_direction_and_data_range(tmp_path: Path) -> None:
    csv_a = _custom_csv(tmp_path / "a", metadata=True)
    csv_b = _custom_csv(tmp_path / "b", metadata=True)

    resolved = compare_app._resolve_request(
        compare_app.CompareRequest(mode="unpaired", csv_a=csv_a, csv_b=csv_b, column="bias")
    )

    assert resolved.higher_is_better is False
    assert resolved.thresholds == ()
    assert (resolved.min_value, resolved.max_value) == (2.0, 5.0)


def test_compare_unknown_metric_without_metadata_needs_explicit_direction(
    tmp_path: Path,
) -> None:
    csv_a = _custom_csv(tmp_path / "a", metadata=False)
    csv_b = _custom_csv(tmp_path / "b", metadata=False)
    request = compare_app.CompareRequest(mode="paired", csv_a=csv_a, csv_b=csv_b, column="bias")

    with pytest.raises(ValueError, match="Ranking direction of 'bias' is unknown"):
        compare_app._resolve_request(request)
    explicit = compare_app._resolve_request(replace(request, higher_is_better=True))
    assert explicit.higher_is_better is True


def test_metric_colors_follow_presentation_thresholds() -> None:
    assert color_for_metric("ssim", 0.9) == "green"
    assert color_for_metric("ssim", 0.8) == "yellow"
    assert color_for_metric("ssim", 0.7) == "orange"
    assert color_for_metric("ssim", 0.5) == "red"
    assert color_for_metric("unknown", 0.5) == "cyan"


def test_pipeline_rejects_unknown_stage_before_loading_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(tmp_path / "missing.yaml", "publish")
