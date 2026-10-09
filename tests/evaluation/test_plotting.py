from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from tests.image_helpers import write_rgb_image
from virtual_staining.evaluation import comparison, diagnostics, panels
from virtual_staining.evaluation.plotting import (
    histogram_edges,
    save_dataset_plots,
    save_unpaired_feature_plot,
)
from virtual_staining.evaluation.reports import build_metric_row
from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    MetricDefinition,
    MetricResult,
    ResolvedMetric,
)

matplotlib.use("Agg")

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


@pytest.fixture
def caller_figure(monkeypatch: pytest.MonkeyPatch):
    caller, ax = plt.subplots()
    line = ax.plot([0, 1], [1, 0])[0]
    before = set(plt.get_fignums())
    created = []
    original_figure = plt.figure

    def record_figure(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        if fig is not caller and fig not in created:
            created.append(fig)
        return fig

    monkeypatch.setattr(plt, "figure", record_figure)

    def check():
        assert set(plt.get_fignums()) == before
        assert caller.axes == [ax]
        assert list(ax.lines) == [line]
        np.testing.assert_array_equal(line.get_ydata(), [1, 0])
        caller.canvas.draw()

    yield caller, check
    # Only clean up recorded test figures, after the registry assertions have run.
    for fig in created:
        plt.close(fig)
    plt.close(caller)


@pytest.fixture(
    params=[
        "dataset",
        "features",
        "error",
        "intensity",
        "channel_scatter",
        "panel",
        "stacked",
        "histogram",
        "ecdf",
        "delta",
        "scatter",
    ]
)
def plot_call(request: pytest.FixtureRequest, tmp_path: Path):
    image = write_rgb_image(tmp_path / "input.png", size=(200, 200))
    pixels = np.linspace(0, 1, 48).reshape(4, 4, 3)
    a, b = np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.6, 0.7])
    path = tmp_path / "plot.png"
    cases = {
        "dataset": (
            partial(save_dataset_plots, [_row(0.5)], _METRICS[:1], tmp_path),
            [tmp_path / "HE__mae_histogram.png", tmp_path / "metrics_boxplot.png"],
            "hist",
        ),
        "features": (
            partial(
                save_unpaired_feature_plot, {"mean_r": a.tolist()}, {"mean_r": b.tolist()}, path
            ),
            path,
            "hist",
        ),
        "error": (
            partial(diagnostics._make_error_histogram, pixels, pixels / 2, path),
            path,
            "hist",
        ),
        "intensity": (
            partial(diagnostics._make_intensity_overlay_histogram, pixels, pixels / 2, path),
            path,
            "hist",
        ),
        "channel_scatter": (
            partial(diagnostics._make_scatter_by_channel, pixels, pixels / 2, path),
            path,
            "scatter",
        ),
        "panel": (partial(panels.save_comparison_panel, image, image, image, path), path, "imshow"),
        "stacked": (partial(panels._save_stacked_image_panel, [image], path), path, "imshow"),
        "histogram": (
            partial(
                comparison.plot_distribution_histogram,
                a,
                b,
                np.linspace(0, 1, 5),
                "A",
                "B",
                "mae",
                tmp_path,
            ),
            None,
            "hist",
        ),
        "ecdf": (
            partial(comparison.plot_distribution_ecdf, a, b, "A", "B", "mae", tmp_path),
            None,
            "step",
        ),
        "delta": (
            partial(comparison.plot_paired_delta_histogram, b - a, "mae", tmp_path),
            None,
            "hist",
        ),
        "scatter": (
            partial(
                comparison.plot_paired_scatter,
                pd.DataFrame({"value_a": a, "value_b": b}),
                "A",
                "B",
                "mae",
                tmp_path,
            ),
            None,
            "scatter",
        ),
    }
    call, result, operation = cases[request.param]
    comparison_names = {
        "histogram": "histogram_comparison.png",
        "ecdf": "ecdf_comparison.png",
        "delta": "paired_delta_histogram.png",
        "scatter": "paired_scatter.png",
    }
    paths = (
        result
        if isinstance(result, list)
        else [result or tmp_path / comparison_names[request.param]]
    )
    return call, result, paths, operation


def test_plot_success_preserves_figure_registry(plot_call, caller_figure, tmp_path: Path) -> None:
    call, expected, paths, _ = plot_call
    _, check = caller_figure
    assert call() == expected
    check()
    assert set(tmp_path.glob("*.png")) == {tmp_path / "input.png", *paths}
    for path in paths:
        with Image.open(path) as image:
            image.verify()


@pytest.mark.parametrize("stage", ["plot", "layout", "save"])
def test_plot_failures_release_owned_figures(
    plot_call, caller_figure, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    call, _, _, operation = plot_call
    caller, check = caller_figure
    error = RuntimeError(f"injected {stage} failure")

    def fail(*args, **kwargs):
        # Cleanup must close the owned figure even when the caller's is current.
        plt.figure(caller.number)
        raise error

    owner, name = (
        (Axes, operation)
        if stage == "plot"
        else (Figure, {"layout": "tight_layout", "save": "savefig"}[stage])
    )
    monkeypatch.setattr(owner, name, fail)
    counts = []
    before = set(plt.get_fignums())
    for _ in range(3):
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is error
        counts.append(set(plt.get_fignums()))
    assert counts == [before] * 3
    check()


@pytest.mark.parametrize("kind", ["dataset", "features"])
def test_plot_failure_before_first_artist_closes_figure(
    tmp_path: Path, caller_figure, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _, check = caller_figure
    error = RuntimeError("injected data preparation failure")

    def fail(*args, **kwargs):
        raise error

    if kind == "dataset":
        monkeypatch.setattr(np, "ones", fail)
        call = partial(save_dataset_plots, [_row(0.5)], _METRICS[:1], tmp_path)
    else:
        monkeypatch.setattr(np, "histogram_bin_edges", fail)
        call = partial(
            save_unpaired_feature_plot, {"r": [0.5]}, {"r": [0.6]}, tmp_path / "features.png"
        )
    for _ in range(3):
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is error
        check()


def test_dataset_boxplot_failure_preserves_completed_histograms(
    tmp_path: Path, caller_figure, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, check = caller_figure

    def fail(*args, **kwargs):
        raise RuntimeError("injected boxplot failure")

    monkeypatch.setattr(Axes, "boxplot", fail)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="injected boxplot failure"):
            save_dataset_plots([_row(0.5)], _METRICS, tmp_path)
        check()
        assert {path.name for path in tmp_path.glob("*.png")} == {
            f"HE__{metric.name}_histogram.png" for metric in _METRICS
        }


@pytest.mark.parametrize("kind", ["dataset", "diagnostics", "summary"])
def test_later_save_failure_preserves_completed_output(
    tmp_path: Path, caller_figure, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _, check = caller_figure
    image = write_rgb_image(tmp_path / "HE" / "sample_generated.png", size=(200, 200))
    output = tmp_path / "output"
    if kind == "dataset":
        call = partial(save_dataset_plots, [_row(0.5)], _METRICS, output)
        first_name = "HE__mae_histogram.png"
    elif kind == "diagnostics":
        call = partial(diagnostics.save_diagnostic_plots, image, image, output)
        first_name = "sample__HE_error_histogram.png"
    else:
        entry: panels.DiagnosticEntry = {
            "kind": "best",
            "sample_id": "sample",
            "metric_value": 0.5,
            "comparison_path": image,
            "error_histogram_path": image,
            "intensity_overlay_histogram_path": image,
            "target_vs_generated_scatter_by_channel_path": image,
        }
        call = partial(panels.save_metric_diagnostics_summary, "mae", output, [entry])
        first_name = "mae_comparisons_best_median_worst.png"
    savefig = Figure.savefig
    saves = 0

    def fail_second_save(fig, path, **kwargs):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError("injected write failure")
        return savefig(fig, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_second_save)
    for _ in range(3):
        saves = 0
        with pytest.raises(OSError, match="injected write failure"):
            call()
        check()
        assert saves == 2
        assert [path.name for path in output.iterdir()] == [first_name]
        with Image.open(output / first_name) as saved:
            saved.verify()
