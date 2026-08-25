from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from virtual_staining.evaluation.diagnostics import save_diagnostic_plots
from virtual_staining.evaluation.panels import (
    DiagnosticEntry,
    build_metric_case_artifacts,
    save_comparison_panel,
    save_metric_diagnostics_summary,
)
from virtual_staining.evaluation.selection import (
    METRIC_SELECTION_ORDER,
    select_representative_rows,
    write_metric_selection_summary,
)
from virtual_staining.evaluation.summaries import read_per_image_metrics_csv, read_summary_csv
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.utils.artifacts import generated_sample_id


@dataclass(frozen=True)
class ComparePanelsRequest:
    mode: Literal["single", "from_metrics"]
    source_image: Path | None = None
    generated_image: Path | None = None
    target_image: Path | None = None
    save_path: Path | None = None
    with_diagnostics: bool = False
    run_path: Path | None = None


@dataclass
class SinglePanelResult:
    saved_path: Path
    diagnostic_paths: list[Path] = field(default_factory=list)


@dataclass
class FromMetricsResult:
    run_path: Path
    available_metrics: list[str]
    per_metric_representative_rows: dict[str, dict[str, dict[str, str]]]
    saved_aggregated_paths: list[Path]
    metrics_dir: Path
    hide_graphs_path: bool = False


def compare_panels(request: ComparePanelsRequest) -> SinglePanelResult | FromMetricsResult:
    """Run a single-pair panel comparison or metric-based representative comparisons."""
    if request.mode == "single":
        return _run_single(request)
    if request.mode == "from_metrics":
        return _run_from_metrics(request)
    raise ValueError(f"Unsupported compare_panels mode: {request.mode}")


def _infer_run_dir_from_generated_path(generated_path: str | Path) -> Path:
    try:
        return RunLayout.from_artifact_path(Path(generated_path)).root
    except ValueError as exc:
        raise ValueError(
            f"{exc} Generated paths must be inside a run's artifacts subtree."
        ) from None


def _infer_default_save_path(generated_image: str | Path) -> Path:
    generated_path = Path(generated_image)
    sample_id = generated_sample_id(generated_path)
    layout = RunLayout.from_artifact_path(generated_path)
    return layout.comparisons_dir / f"{sample_id}_comparison.png"


def _infer_diagnostics_dir(save_path: str | Path) -> Path:
    return Path(save_path).parent / "diagnostics"


def _infer_case_diagnostics_dir(save_path: str | Path, generated_image: str | Path) -> Path:
    diagnostics_dir = _infer_diagnostics_dir(save_path)
    sample_id = generated_sample_id(generated_image)
    return diagnostics_dir / sample_id


def _run_single(request: ComparePanelsRequest) -> SinglePanelResult:
    assert request.source_image is not None
    assert request.generated_image is not None
    assert request.target_image is not None

    save_path = (
        request.save_path
        if request.save_path is not None
        else _infer_default_save_path(request.generated_image)
    )

    saved_path = save_comparison_panel(
        source_path=request.source_image,
        generated_path=request.generated_image,
        target_path=request.target_image,
        save_path=save_path,
    )

    diagnostic_paths: list[Path] = []

    if request.with_diagnostics:
        diagnostics_dir = _infer_case_diagnostics_dir(saved_path, request.generated_image)
        diagnostic_paths = save_diagnostic_plots(
            generated_path=request.generated_image,
            target_path=request.target_image,
            save_dir=diagnostics_dir,
        )

    return SinglePanelResult(saved_path=saved_path, diagnostic_paths=diagnostic_paths)


def _run_from_metrics(request: ComparePanelsRequest) -> FromMetricsResult:
    assert request.run_path is not None
    layout = RunLayout(request.run_path.resolve())
    run_path = layout.root
    summary_csv = layout.summary_csv
    per_image_csv = layout.per_image_metrics
    summary_rows = read_summary_csv(summary_csv)
    per_image_rows = read_per_image_metrics_csv(per_image_csv)
    metrics_dir = layout.comparisons_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection_summary_rows: list[dict[str, object]] = []
    saved_aggregated_paths: list[Path] = []
    available_metrics = [metric for metric in METRIC_SELECTION_ORDER if metric in summary_rows]

    if not available_metrics:
        raise ValueError(
            f"No supported metrics found in {summary_csv}. "
            f"Expected one of: {', '.join(METRIC_SELECTION_ORDER)}"
        )

    per_metric_representative_rows: dict[str, dict[str, dict[str, str]]] = {}

    for metric_name in available_metrics:
        metric_summary = summary_rows[metric_name]
        metric_dir = metrics_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        representative_rows = select_representative_rows(
            metric_name,
            metric_summary,
            per_image_rows,
        )
        per_metric_representative_rows[metric_name] = representative_rows
        metric_selection_rows: list[dict[str, object]] = []
        metric_diagnostic_entries: list[DiagnosticEntry] = []

        for kind, row in representative_rows.items():
            selection_row, diagnostic_entry = build_metric_case_artifacts(
                metric_name=metric_name,
                kind=kind,
                row=row,
                metric_summary=metric_summary,
                metric_dir=metric_dir,
            )
            selection_summary_rows.append(selection_row)
            metric_selection_rows.append(selection_row)
            metric_diagnostic_entries.append(diagnostic_entry)

        write_metric_selection_summary(metric_selection_rows, metric_dir / "selection_summary.csv")
        kind_order = {"best": 0, "median": 1, "worst": 2}
        metric_diagnostic_entries.sort(key=lambda entry: kind_order[entry["kind"]])
        aggregated_paths = save_metric_diagnostics_summary(
            metric_name=metric_name,
            metric_dir=metric_dir,
            diagnostic_entries=metric_diagnostic_entries,
        )
        saved_aggregated_paths.extend(aggregated_paths)

    write_metric_selection_summary(
        selection_summary_rows,
        metrics_dir / "metrics_selection_summary.csv",
    )

    return FromMetricsResult(
        run_path=run_path,
        available_metrics=available_metrics,
        per_metric_representative_rows=per_metric_representative_rows,
        saved_aggregated_paths=saved_aggregated_paths,
        metrics_dir=metrics_dir,
    )
