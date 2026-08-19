from __future__ import annotations

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


def test_metric_colors_follow_domain_thresholds() -> None:
    assert color_for_metric("ssim", 0.9) == "green"
    assert color_for_metric("ssim", 0.8) == "yellow"
    assert color_for_metric("ssim", 0.7) == "orange"
    assert color_for_metric("ssim", 0.5) == "red"
    assert color_for_metric("unknown", 0.5) == "cyan"


def test_pipeline_rejects_unknown_stage_before_loading_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(tmp_path / "missing.yaml", "publish")
