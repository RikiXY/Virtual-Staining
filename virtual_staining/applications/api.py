from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from virtual_staining.applications.compare import CompareRequest, compare
from virtual_staining.applications.compare_panels import (
    ComparePanelsRequest,
    FromMetricsResult,
    compare_panels,
)
from virtual_staining.applications.evaluate_single import evaluate_pair
from virtual_staining.applications.pipeline import run_stage
from virtual_staining.applications.ui_inference import (
    CatalogIssue,
    ModelCatalog,
    ModelDescriptor,
    ResultProvenance,
    SavedInferenceResult,
    UIInferenceError,
    UIInferenceResult,
    UIInferenceService,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.diagnostics import compute_absolute_difference_map
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.selection import select_representative_rows
from virtual_staining.evaluation.summaries import (
    read_per_image_metrics_csv,
    read_summary_csv,
)
from virtual_staining.metrics import DEFAULT_METRICS, MetricQuality, metric_quality

logger = logging.getLogger(__name__)

# Public name used by presentation layers. The old name remains available from
# ui_inference for compatibility with the first inference-only UI.
ApplicationError = UIInferenceError
InferenceResult = UIInferenceResult

__all__ = [
    "ApplicationError",
    "ApplicationService",
    "CatalogIssue",
    "ComparisonRequest",
    "ComparisonResult",
    "GeneratedSampleEvaluationRequest",
    "InferenceRequest",
    "InferenceResult",
    "ModelCatalog",
    "ModelDescriptor",
    "MetricQuality",
    "RepresentativeSample",
    "ResultProvenance",
    "RunDescriptor",
    "RunEvaluationRequest",
    "RunEvaluationResult",
    "SavedInferenceResult",
    "SingleSampleRequest",
    "SingleSampleResult",
    "metric_quality",
]


@dataclass(frozen=True)
class InferenceRequest:
    """Portable input for one strict, single-patch inference operation."""

    model_identifier: str
    source_image: Image.Image
    source_filename: str


@dataclass(frozen=True)
class SingleSampleRequest:
    """Generate and evaluate one source/target pair."""

    model_identifier: str
    source_image: Image.Image
    target_image: Image.Image
    source_filename: str
    target_filename: str
    output_directory: Path | None = None


@dataclass(frozen=True)
class GeneratedSampleEvaluationRequest:
    """Evaluate an existing inference result against a target loaded later."""

    inference: InferenceResult
    target_image: Image.Image
    target_filename: str
    output_directory: Path | None = None


@dataclass(frozen=True)
class SingleSampleResult:
    inference: InferenceResult
    target_image: Image.Image
    difference_map: Image.Image
    metrics: dict[str, float]
    output_directory: Path
    source_path: Path
    generated_path: Path
    target_path: Path
    metrics_csv: Path


@dataclass(frozen=True)
class RunDescriptor:
    identifier: str
    display_name: str
    path: Path
    has_evaluation: bool
    sample_count: int | None = None
    last_event_at: str | None = None


@dataclass(frozen=True)
class RepresentativeSample:
    kind: str
    sample_id: str
    metric: str
    value: float
    source_path: Path | None
    generated_path: Path | None
    target_path: Path | None
    comparison_path: Path | None = None


@dataclass(frozen=True)
class RunEvaluationRequest:
    """Execute evaluation from a config or load an already evaluated run."""

    config_path: Path | None = None
    run_path: Path | None = None
    ensure_plots: bool = True
    build_representative_panels: bool = True


@dataclass(frozen=True)
class RunEvaluationResult:
    run: RunDescriptor
    summary: dict[str, dict[str, float]]
    rows: tuple[dict[str, str], ...]
    plot_paths: tuple[Path, ...]
    representatives: dict[str, tuple[RepresentativeSample, ...]]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonRequest:
    run_a: Path
    run_b: Path
    metric: str = "ssim"
    mode: Literal["paired", "unpaired"] = "paired"
    output_directory: Path | None = None
    tolerance: float = 0.0


@dataclass(frozen=True)
class ComparisonResult:
    mode: Literal["paired", "unpaired"]
    metric: str
    higher_is_better: bool
    output_directory: Path
    label_a: str
    label_b: str
    summary: dict[str, float | int | str]
    plot_paths: tuple[Path, ...]
    representatives_a: tuple[RepresentativeSample, ...] = field(default_factory=tuple)
    representatives_b: tuple[RepresentativeSample, ...] = field(default_factory=tuple)


class ApplicationService:
    """Stable Python-facing boundary shared by user interfaces and other clients.

    This class deliberately owns all knowledge of checkpoint discovery, experiment
    directories, evaluation CSVs, and today's implementation modules.
    """

    def __init__(
        self,
        checkpoint_directory: Path,
        output_directory: Path,
        results_directory: Path,
        *,
        working_directory: Path | None = None,
    ) -> None:
        self.working_directory = (working_directory or Path.cwd()).resolve()
        self.output_directory = self._resolve(output_directory)
        self.results_directory = self._resolve(results_directory)
        self._inference = UIInferenceService(
            checkpoint_directory,
            output_directory,
            working_directory=self.working_directory,
        )

    @property
    def supported_metrics(self) -> tuple[str, ...]:
        return tuple(DEFAULT_METRICS)

    def discover_models(self) -> ModelCatalog:
        return self._inference.discover_models()

    def validate_inference_input(
        self, model_identifier: str, source_image: Image.Image
    ) -> ModelDescriptor:
        return self._inference.validate_input(model_identifier, source_image)

    def run_inference(self, request: InferenceRequest) -> InferenceResult:
        return self._inference.run_inference(
            request.model_identifier,
            request.source_image,
            request.source_filename,
        )

    def save_inference_result(
        self,
        result: InferenceResult,
        output_directory: str | Path | None = None,
    ) -> SavedInferenceResult:
        return self._inference.save_result(result, output_directory)

    def evaluate_sample(self, request: SingleSampleRequest) -> SingleSampleResult:
        """Convenience operation that generates and immediately evaluates a pair."""
        if request.target_image.mode != "RGB":
            raise ApplicationError(
                "The target must be an RGB image. "
                f"Received image mode: {request.target_image.mode}."
            )
        if request.target_image.size != request.source_image.size:
            raise ApplicationError(
                "The source and target images must have identical dimensions. "
                f"Source: {request.source_image.size[0]} × "
                f"{request.source_image.size[1]} px; target: "
                f"{request.target_image.size[0]} × {request.target_image.size[1]} px."
            )
        inference = self.run_inference(
            InferenceRequest(
                model_identifier=request.model_identifier,
                source_image=request.source_image,
                source_filename=request.source_filename,
            )
        )
        return self.evaluate_generated_sample(
            GeneratedSampleEvaluationRequest(
                inference=inference,
                target_image=request.target_image,
                target_filename=request.target_filename,
                output_directory=request.output_directory,
            )
        )

    def evaluate_generated_sample(
        self, request: GeneratedSampleEvaluationRequest
    ) -> SingleSampleResult:
        """Evaluate a previously generated image without rerunning its model."""
        if request.target_image.mode != "RGB":
            raise ApplicationError(
                "The target must be an RGB image. "
                f"Received image mode: {request.target_image.mode}."
            )
        if request.target_image.size != request.inference.generated_image.size:
            raise ApplicationError(
                "The generated and target images must have identical dimensions. "
                f"Generated: {request.inference.generated_image.size[0]} × "
                f"{request.inference.generated_image.size[1]} px; target: "
                f"{request.target_image.size[0]} × {request.target_image.size[1]} px."
            )
        source_filename = request.inference.provenance.source_filename
        root = self._single_sample_directory(
            request.output_directory,
            source_filename,
        )
        sample_id = _safe_sample_id(source_filename)
        source_path = root / f"{sample_id}_source.png"
        target_path = root / f"{sample_id}_target.png"
        generated_path = root / f"{sample_id}_target_generated.png"
        evaluation_dir = root / "evaluation"
        inference = replace(
            request.inference,
            provenance=replace(
                request.inference.provenance,
                generated_filename=generated_path.name,
            ),
        )
        try:
            root.mkdir(parents=True, exist_ok=False)
            inference.source_image.save(source_path, format="PNG")
            request.target_image.save(target_path, format="PNG")
            inference.generated_image.save(generated_path, format="PNG")
            (root / "provenance.json").write_text(
                json.dumps(inference.provenance.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            evaluation = evaluate_pair(target_path, generated_path, evaluation_dir)
            (root / "sample.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_filename": source_filename,
                        "target_filename": request.target_filename,
                        "artifacts": {
                            "source": source_path.name,
                            "generated": generated_path.name,
                            "target": target_path.name,
                            "metrics": str(evaluation.single_case_csv.relative_to(root)),
                        },
                        "metrics": evaluation.metrics,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except ApplicationError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Single-sample evaluation failed in %s", root)
            raise ApplicationError(f"Single-sample evaluation failed: {exc}") from exc

        difference = compute_absolute_difference_map(
            inference.generated_image, request.target_image
        )
        difference_image = _difference_image(difference)
        return SingleSampleResult(
            inference=inference,
            target_image=request.target_image.copy(),
            difference_map=difference_image,
            metrics=evaluation.metrics,
            output_directory=root,
            source_path=source_path,
            generated_path=generated_path,
            target_path=target_path,
            metrics_csv=evaluation.single_case_csv,
        )

    def discover_runs(self) -> tuple[RunDescriptor, ...]:
        root = self.results_directory
        if not root.is_dir():
            return ()
        runs: list[RunDescriptor] = []
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() or path.name == "comparisons":
                continue
            descriptor = self._describe_run(path)
            if descriptor.has_evaluation or (path / "metadata" / "run.json").is_file():
                runs.append(descriptor)
        return tuple(runs)

    def evaluate_run(self, request: RunEvaluationRequest) -> RunEvaluationResult:
        if (request.config_path is None) == (request.run_path is None):
            raise ApplicationError("Choose exactly one run directory or evaluation config.")

        if request.config_path is not None:
            config_path = self._resolve(request.config_path)
            if not config_path.is_file():
                raise ApplicationError(f"Evaluation config not found: {config_path}")
            try:
                config = RunConfig.from_yaml(config_path)
                run_stage(config_path, "evaluate")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.exception("Configured evaluation failed for %s", config_path)
                raise ApplicationError(f"Run evaluation failed: {exc}") from exc
            run_path = self._resolve(config.project.run_root)
        else:
            assert request.run_path is not None
            run_path = self._resolve(request.run_path)

        return self._load_run_evaluation(
            run_path,
            ensure_plots=request.ensure_plots,
            build_representative_panels=request.build_representative_panels,
        )

    def compare_runs(self, request: ComparisonRequest) -> ComparisonResult:
        run_a = self._resolve(request.run_a)
        run_b = self._resolve(request.run_b)
        if run_a == run_b:
            raise ApplicationError("Choose two different runs to compare.")
        if request.metric not in self.supported_metrics:
            raise ApplicationError(
                f"Unsupported metric '{request.metric}'. "
                f"Choose one of: {', '.join(self.supported_metrics)}."
            )
        output_dir = (
            self._resolve(request.output_directory)
            if request.output_directory is not None
            else self.results_directory
            / "comparisons"
            / f"{run_a.name}_vs_{run_b.name}"
            / f"{request.mode}_{request.metric}"
        )
        try:
            result = compare(
                CompareRequest(
                    mode=request.mode,
                    run_a=run_a,
                    run_b=run_b,
                    column=request.metric,
                    output_dir=output_dir,
                    tolerance=request.tolerance,
                )
            )
            representatives_a = self._representatives_for(run_a, request.metric)
            representatives_b = self._representatives_for(run_b, request.metric)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Run comparison failed for %s and %s", run_a, run_b)
            raise ApplicationError(f"Run comparison failed: {exc}") from exc

        if result.paired_summary is not None:
            paired = result.paired_summary
            summary: dict[str, float | int | str] = {
                "paired_samples": paired.n_pairs,
                "mean_signed_delta": paired.mean_signed_delta,
                "median_signed_delta": paired.median_signed_delta,
                "share_b_better": paired.share_b_better,
                "share_a_better": paired.share_a_better,
                "share_equal": paired.share_equal,
                "wilcoxon_pvalue": paired.wilcoxon_pvalue,
                "favors": paired.better_label,
            }
            label_a, label_b = paired.label_a, paired.label_b
        else:
            assert result.group_a is not None
            assert result.group_b is not None
            assert result.unpaired_comparison is not None
            group_a = result.group_a
            group_b = result.group_b
            comparison = result.unpaired_comparison
            direction = 1.0 if result.higher_is_better else -1.0
            summary = {
                "samples_a": group_a.n,
                "samples_b": group_b.n,
                "mean_a": group_a.mean,
                "mean_b": group_b.mean,
                "mean_improvement_b": direction * (group_b.mean - group_a.mean),
                "median_a": group_a.median,
                "median_b": group_b.median,
                "median_improvement_b": direction * (group_b.median - group_a.median),
                "wasserstein_distance": comparison.wasserstein_between_groups,
                "ks_pvalue": comparison.ks_pvalue,
                "mannwhitney_pvalue": comparison.mannwhitney_pvalue,
                "favors": comparison.better_label,
            }
            label_a, label_b = group_a.label, group_b.label

        return ComparisonResult(
            mode=request.mode,
            metric=request.metric,
            higher_is_better=result.higher_is_better,
            output_directory=result.output_dir,
            label_a=label_a,
            label_b=label_b,
            summary=summary,
            plot_paths=tuple(sorted(result.output_dir.glob("*.png"))),
            representatives_a=representatives_a,
            representatives_b=representatives_b,
        )

    def _load_run_evaluation(
        self,
        run_path: Path,
        *,
        ensure_plots: bool,
        build_representative_panels: bool,
    ) -> RunEvaluationResult:
        if not run_path.is_dir():
            raise ApplicationError(f"Run directory not found: {run_path}")
        evaluation_dir = run_path / "evaluation"
        summary_path = evaluation_dir / "summary.csv"
        metrics_path = evaluation_dir / "per_image_metrics.csv"
        try:
            summary = read_summary_csv(summary_path)
            rows = read_per_image_metrics_csv(metrics_path)
        except (OSError, ValueError) as exc:
            raise ApplicationError(
                f"Evaluation results are incomplete for run '{run_path.name}': {exc}"
            ) from exc
        if not rows:
            raise ApplicationError(f"Run '{run_path.name}' contains no evaluated samples.")

        warnings: list[str] = []
        if ensure_plots and not tuple(evaluation_dir.glob("*.png")):
            try:
                plot_rows = cast(list[dict[str, object]], list(rows))
                save_dataset_plots(plot_rows, evaluation_dir)
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.append(f"Metric plots could not be created: {exc}")

        representatives = {
            metric: self._representatives_from_rows(metric, metric_summary, rows)
            for metric, metric_summary in summary.items()
            if metric in self.supported_metrics
        }
        if build_representative_panels:
            try:
                panels = compare_panels(
                    ComparePanelsRequest(mode="from_metrics", run_path=run_path)
                )
                assert isinstance(panels, FromMetricsResult)
                representatives = self._merge_panel_paths(representatives, panels)
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.append(f"Representative panels could not be created: {exc}")

        return RunEvaluationResult(
            run=self._describe_run(run_path),
            summary=summary,
            rows=tuple(rows),
            plot_paths=tuple(sorted(evaluation_dir.glob("*.png"))),
            representatives=representatives,
            warnings=tuple(warnings),
        )

    def _representatives_for(self, run_path: Path, metric: str) -> tuple[RepresentativeSample, ...]:
        summary = read_summary_csv(run_path / "evaluation" / "summary.csv")
        rows = read_per_image_metrics_csv(run_path / "evaluation" / "per_image_metrics.csv")
        if metric not in summary:
            raise ValueError(f"Metric '{metric}' is not available in run '{run_path.name}'.")
        return self._representatives_from_rows(metric, summary[metric], rows)

    @staticmethod
    def _representatives_from_rows(
        metric: str,
        metric_summary: dict[str, float],
        rows: list[dict[str, str]],
    ) -> tuple[RepresentativeSample, ...]:
        selected = select_representative_rows(metric, metric_summary, rows)
        return tuple(
            RepresentativeSample(
                kind=kind,
                sample_id=row["sample_id"],
                metric=metric,
                value=float(row[metric]),
                source_path=_optional_existing_path(row.get("source_path")),
                generated_path=_optional_existing_path(row.get("generated_path")),
                target_path=_optional_existing_path(row.get("target_path")),
            )
            for kind, row in selected.items()
        )

    @staticmethod
    def _merge_panel_paths(
        representatives: dict[str, tuple[RepresentativeSample, ...]],
        panels: FromMetricsResult,
    ) -> dict[str, tuple[RepresentativeSample, ...]]:
        merged: dict[str, tuple[RepresentativeSample, ...]] = {}
        for metric, samples in representatives.items():
            panel_rows = panels.per_metric_representative_rows.get(metric, {})
            merged[metric] = tuple(
                RepresentativeSample(
                    **{
                        **sample.__dict__,
                        "comparison_path": _optional_existing_path(
                            panel_rows.get(sample.kind, {}).get("comparison_path")
                        ),
                    }
                )
                for sample in samples
            )
        return merged

    def _single_sample_directory(
        self,
        output_directory: Path | None,
        source_filename: str,
    ) -> Path:
        base = (
            self._resolve(output_directory)
            if output_directory is not None
            else self.output_directory / "experiments" / "single_samples"
        )
        stem = _safe_sample_id(source_filename)
        candidate = base / stem
        index = 2
        while candidate.exists():
            candidate = base / f"{stem}_{index}"
            index += 1
        return candidate

    def _describe_run(self, path: Path) -> RunDescriptor:
        metadata_path = path / "metadata" / "run.json"
        metadata: dict[str, object] = {}
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                logger.warning("Could not read run metadata: %s", metadata_path)
        metrics_path = path / "evaluation" / "per_image_metrics.csv"
        sample_count: int | None = None
        if metrics_path.is_file():
            with suppress(OSError):
                sample_count = len(read_per_image_metrics_csv(metrics_path))
        return RunDescriptor(
            identifier=path.name,
            display_name=str(metadata.get("run_name") or path.name),
            path=path,
            has_evaluation=(path / "evaluation" / "summary.csv").is_file()
            and metrics_path.is_file(),
            sample_count=sample_count,
            last_event_at=(
                str(metadata["last_event_at"]) if metadata.get("last_event_at") else None
            ),
        )

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.working_directory / candidate).resolve()


def _safe_sample_id(filename: str) -> str:
    stem = Path(filename.replace("\\", "/")).stem
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in stem
    )
    return safe.strip("_-") or "sample"


def _difference_image(difference: np.ndarray) -> Image.Image:
    finite = np.nan_to_num(difference, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(finite * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def _optional_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None
