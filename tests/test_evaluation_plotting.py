from __future__ import annotations

from pathlib import Path

from virtual_staining.evaluation.plotting import METRIC_NAMES, save_dataset_plots


def _row(value: float) -> dict[str, object]:
    return {metric: value for metric in METRIC_NAMES}


def test_save_dataset_plots_creates_expected_files(tmp_path: Path) -> None:
    rows = [_row(0.5), _row(0.6), _row(0.7)]

    saved_paths = save_dataset_plots(rows, tmp_path)

    expected_names = {f"{metric}_histogram.png" for metric in METRIC_NAMES}
    expected_names.add("metrics_boxplot.png")

    assert {path.name for path in saved_paths} == expected_names
    assert all(path.is_file() for path in saved_paths)
