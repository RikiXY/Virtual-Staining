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


# ---------------------------------------------------------------------------
# Non-finite value handling in plots
# ---------------------------------------------------------------------------


def test_save_dataset_plots_skips_inf_psnr_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when PSNR is inf (identical images)."""
    row = dict(_row(0.5))
    row["psnr"] = float("inf")
    saved_paths = save_dataset_plots([row, _row(0.6)], tmp_path / "inf_psnr")
    assert all(p.is_file() for p in saved_paths)


def test_save_dataset_plots_skips_nan_pcc_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when PCC metrics are nan (constant images)."""
    row = dict(_row(0.5))
    row["pcc_gray"] = float("nan")
    row["pcc_rgb_mean"] = float("nan")
    saved_paths = save_dataset_plots([row, _row(0.6)], tmp_path / "nan_pcc")
    assert all(p.is_file() for p in saved_paths)


def test_save_dataset_plots_all_non_finite_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when every value for a metric is non-finite."""
    rows: list[dict[str, object]] = [{metric: float("inf") for metric in METRIC_NAMES}]
    saved_paths = save_dataset_plots(rows, tmp_path / "all_nonfinite")
    assert all(p.is_file() for p in saved_paths)
