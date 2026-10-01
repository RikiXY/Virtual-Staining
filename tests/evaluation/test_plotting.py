from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

from virtual_staining.evaluation.plotting import histogram_edges, save_dataset_plots
from virtual_staining.evaluation.reports import build_metric_row
from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    MetricDefinition,
    MetricResult,
    ResolvedMetric,
)

_METRICS = tuple(
    BUILTIN_METRIC_DEFINITIONS[name].resolve({}, name) for name in ("mae", "psnr", "pcc_gray")
)


def _row(value: float, **overrides: MetricResult) -> dict[str, object]:
    results = {metric.name: MetricResult.of(value) for metric in _METRICS} | overrides
    return build_metric_row("s", "HE", "t.png", "g.png", (8, 8, 3), results, set_id="S")


def test_save_dataset_plots_writes_one_histogram_per_requested_metric(tmp_path: Path) -> None:
    saved_paths = save_dataset_plots([_row(0.5), _row(0.6), _row(0.7)], _METRICS, tmp_path)

    assert {path.name for path in saved_paths} == {
        "HE__mae_histogram.png",
        "HE__psnr_histogram.png",
        "HE__pcc_gray_histogram.png",
        "metrics_boxplot.png",
    }
    assert all(path.is_file() for path in saved_paths)


def test_save_dataset_plots_never_mixes_outputs_in_one_histogram(tmp_path: Path) -> None:
    rows = [_row(0.5), {**_row(0.9), "output_name": "PAS"}]

    saved_paths = save_dataset_plots(rows, _METRICS[:1], tmp_path)

    assert sorted(path.name for path in saved_paths) == [
        "HE__mae_histogram.png",
        "PAS__mae_histogram.png",
        "metrics_boxplot.png",
    ]


def test_plotting_from_worker_thread_uses_non_interactive_backend(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=1) as executor:
            saved_paths = executor.submit(
                save_dataset_plots, [_row(0.5)], _METRICS, tmp_path
            ).result()

    assert matplotlib.get_backend().lower() == "agg"
    assert all(path.is_file() for path in saved_paths)
    assert not any("GUI outside of the main thread" in str(item.message) for item in caught)


# ---------------------------------------------------------------------------
# Non-finite value handling in plots
# ---------------------------------------------------------------------------


def test_save_dataset_plots_ignores_non_finite_results(tmp_path: Path) -> None:
    rows = [
        _row(0.5, psnr=MetricResult.of(math.inf), pcc_gray=MetricResult.undefined("constant")),
        _row(0.6, pcc_gray=MetricResult.undefined("constant")),
    ]
    saved_paths = save_dataset_plots(rows, _METRICS, tmp_path)
    assert all(p.is_file() for p in saved_paths)


def test_unknown_metric_histogram_uses_data_not_an_invented_range(tmp_path: Path) -> None:
    custom = MetricDefinition("custom", "1", "tests", lambda *_: {}, higher_is_better=None).resolve(
        {}, "custom"
    )
    assert isinstance(custom, ResolvedMetric)

    edges = histogram_edges([3.0, 7.0], custom.definition.plot_range)

    assert (edges[0], edges[-1]) == (3.0, 7.0)
    assert (histogram_edges([0.2], (0.0, 1.0))[0], histogram_edges([0.2], (0.0, 1.0))[-1]) == (
        0.0,
        1.0,
    )


def test_plotting_existing_report_preserves_recorded_metric_range(
    tmp_path: Path, monkeypatch
) -> None:
    import matplotlib.pyplot as plt

    from virtual_staining.evaluation.reports import MetricInfo

    limits = []
    original = plt.xlim

    def record_limits(*args, **kwargs):
        if args:
            limits.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(plt, "xlim", record_limits)
    paths = save_dataset_plots(
        [_row(0.5)],
        (MetricInfo("mae", False, plot_range=(0.2, 0.8)),),
        tmp_path,
    )
    assert (0.2, 0.8) in limits
    assert all(path.is_file() for path in paths)
