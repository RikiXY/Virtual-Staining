from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.applications import compare as compare_app
from virtual_staining.applications.pipeline import run_stage


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


def test_pipeline_rejects_unknown_stage_before_loading_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(tmp_path / "missing.yaml", "publish")
