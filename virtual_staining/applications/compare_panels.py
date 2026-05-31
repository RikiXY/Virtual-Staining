from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from virtual_staining.evaluation.panels import (
    METRIC_SELECTION_ORDER,
    DiagnosticEntry,
    build_metric_case_artifacts,
    extract_generated_sample_id,
    save_comparison_panel,
    save_diagnostic_plots,
    save_metric_diagnostics_summary,
    write_metric_selection_summary,
)
from virtual_staining.evaluation.selection import select_ranked_samples
from virtual_staining.evaluation.summaries import read_per_image_metrics_csv, read_summary_csv


@dataclass(frozen=True)
class ComparePanelsRequest:
    mode: Literal["single", "from_metrics"]
    source_image: Path | None = None
    generated_image: Path | None = None
    target_image: Path | None = None
    save_path: Path | None = None
    with_diagnostics: bool = False
    run_path: Path | None = None
    metrics: tuple[str, ...] | None = None
    kinds: tuple[str, ...] = ("best", "median", "worst")
    top_k: int = 1


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
    per_metric_ranked_rows: dict[str, dict[str, list[dict[str, str]]]] = field(default_factory=dict)


def compare_panels(request: ComparePanelsRequest) -> SinglePanelResult | FromMetricsResult:
    """Run a single-pair panel comparison or metric-based representative comparisons."""
    if request.mode == "single":
        return _run_single(request)
    if request.mode == "from_metrics":
        return _run_from_metrics(request)
    raise ValueError(f"Unsupported compare_panels mode: {request.mode}")


def _infer_run_dir_from_generated_path(generated_path: str | Path) -> Path:
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer run directory from generated path. Expected a path like "
            ".../results/NAME_RUN/artifacts/output_test/..."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/artifacts/output_test/..."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. "
            "Please provide --save-path explicitly."
        )

    return run_dir


def _infer_default_save_path(generated_image: str | Path) -> Path:
    generated_path = Path(generated_image)
    sample_id = extract_generated_sample_id(generated_path)
    run_dir = _infer_run_dir_from_generated_path(generated_path)
    return run_dir / "comparisons" / f"{sample_id}_comparison.png"


def _infer_diagnostics_dir(save_path: str | Path) -> Path:
    return Path(save_path).parent / "diagnostics"


def _infer_case_diagnostics_dir(save_path: str | Path, generated_image: str | Path) -> Path:
    diagnostics_dir = _infer_diagnostics_dir(save_path)
    sample_id = extract_generated_sample_id(generated_image)
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
            source_path=request.source_image,
            generated_path=request.generated_image,
            target_path=request.target_image,
            save_dir=diagnostics_dir,
        )

    return SinglePanelResult(saved_path=saved_path, diagnostic_paths=diagnostic_paths)


def _run_from_metrics(request: ComparePanelsRequest) -> FromMetricsResult:
    assert request.run_path is not None
    if request.top_k <= 0:
        raise ValueError("compare_panels top_k must be a positive integer.")
    run_path = request.run_path.resolve()
    evaluation_dir = run_path / "evaluation"
    summary_csv = evaluation_dir / "summary.csv"
    per_image_csv = evaluation_dir / "per_image_metrics.csv"
    summary_rows = read_summary_csv(summary_csv)
    per_image_rows = read_per_image_metrics_csv(per_image_csv)
    metrics_dir = run_path / "comparisons" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection_summary_rows: list[dict[str, object]] = []
    saved_aggregated_paths: list[Path] = []
    available_metrics = _resolve_panel_metrics(request.metrics, summary_rows)

    if not available_metrics:
        raise ValueError(
            f"No supported metrics found in {summary_csv}. "
            f"Expected one of: {', '.join(METRIC_SELECTION_ORDER)}"
        )

    per_metric_representative_rows: dict[str, dict[str, dict[str, str]]] = {}
    per_metric_ranked_rows: dict[str, dict[str, list[dict[str, str]]]] = {}

    for metric_name in available_metrics:
        metric_summary = summary_rows[metric_name]
        metric_dir = metrics_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        selected_samples = select_ranked_samples(
            per_image_rows,
            metric_name,
            top_k=request.top_k,
            kinds=request.kinds,
            median_value=metric_summary["median"],
        )
        ranked_rows = {
            kind: [cast(dict[str, str], sample.row) for sample in samples]
            for kind, samples in selected_samples.items()
        }
        per_metric_ranked_rows[metric_name] = ranked_rows
        representative_rows = {kind: rows[0] for kind, rows in ranked_rows.items() if rows}
        per_metric_representative_rows[metric_name] = representative_rows
        metric_selection_rows: list[dict[str, object]] = []
        metric_diagnostic_entries: list[DiagnosticEntry] = []

        for kind in request.kinds:
            for selected_sample in selected_samples.get(cast(Any, kind), []):
                row = cast(dict[str, str], selected_sample.row)
                selection_row, diagnostic_entry = build_metric_case_artifacts(
                    metric_name=metric_name,
                    kind=kind,
                    row=row,
                    metric_summary=metric_summary,
                    metric_dir=metric_dir,
                    rank=selected_sample.rank,
                    include_rank_in_filename=request.top_k > 1,
                )
                selection_summary_rows.append(selection_row)
                metric_selection_rows.append(selection_row)
                metric_diagnostic_entries.append(diagnostic_entry)

        write_metric_selection_summary(metric_selection_rows, metric_dir / "selection_summary.csv")
        kind_order = {"best": 0, "median": 1, "worst": 2}
        metric_diagnostic_entries.sort(
            key=lambda entry: (kind_order[entry["kind"]], entry.get("rank", 0))
        )
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
        per_metric_ranked_rows=per_metric_ranked_rows,
    )


def _resolve_panel_metrics(
    requested_metrics: tuple[str, ...] | None,
    summary_rows: dict[str, dict[str, float]],
) -> list[str]:
    if requested_metrics is None:
        return [metric for metric in METRIC_SELECTION_ORDER if metric in summary_rows]

    missing_metrics = [metric for metric in requested_metrics if metric not in summary_rows]
    if missing_metrics:
        raise ValueError(
            "Configured compare panel metrics were not found in summary.csv: "
            f"{', '.join(missing_metrics)}"
        )
    return list(requested_metrics)
