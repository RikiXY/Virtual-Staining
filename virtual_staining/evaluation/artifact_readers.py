from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

from virtual_staining.evaluation.comparison import (
    DECISION_BREAKDOWN_COLUMNS,
    PAIRED_DELTA_SUMMARY_COLUMNS,
    UNPAIRED_QUANTILE_COLUMNS,
    UNPAIRED_THRESHOLD_COLUMNS,
)
from virtual_staining.evaluation.panels import (
    RESIDUAL_HEATMAP_FIELDNAMES,
    SELECTION_SUMMARY_FIELDNAMES,
)
from virtual_staining.evaluation.ranking import ORGANIZATION_SUMMARY_FIELDNAMES
from virtual_staining.evaluation.reports import METRIC_FIELDNAMES
from virtual_staining.evaluation.summaries import SUMMARY_FIELDNAMES, WEAK_TAIL_FIELDNAMES

ArtifactStatus = Literal["present", "missing", "empty", "malformed"]

_MANIFEST_PATH_TYPES = {"run_relative", "absolute"}
_COMPARISON_SUMMARY_REQUIRED_COLUMNS = (
    "mode",
    "metric",
    "direction",
    "label_a",
    "label_b",
)
_GROUP_STATISTICS_REQUIRED_COLUMNS = ("label", "n", "mean", "median", "iqr")
_PAIRED_SAMPLE_DELTAS_REQUIRED_COLUMNS = (
    "value_a",
    "value_b",
    "raw_delta_b_minus_a",
    "signed_delta",
    "winner",
)
_PAIRED_METRIC_DELTA_SUMMARY_REQUIRED_COLUMNS = (
    "metric",
    "direction",
    "label_a",
    "label_b",
    "total_common_count",
    "finite_pair_count",
    "improved_count",
    "worsened_count",
    "equal_count",
)


class ArtifactReaderError(ValueError):
    """Base error for malformed evaluation artifacts."""


class MalformedArtifactError(ArtifactReaderError):
    """Raised when a required artifact exists but cannot be parsed safely."""


@dataclass(frozen=True)
class CsvArtifact:
    """A read-only CSV artifact with explicit state for app display logic."""

    name: str
    path: Path
    status: ArtifactStatus
    required: bool
    rows: tuple[dict[str, str], ...] = ()
    fieldnames: tuple[str, ...] = ()
    message: str | None = None

    @property
    def is_present(self) -> bool:
        return self.status in {"present", "empty"}

    @property
    def is_missing(self) -> bool:
        return self.status == "missing"

    @property
    def is_empty(self) -> bool:
        return self.status == "empty"

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class ArtifactManifestRecord:
    """One resolved record from evaluation/artifacts.json."""

    stage: str
    artifact_type: str
    path: str
    path_type: str
    resolved_path: Path
    description: str
    metric: str | None = None
    sample_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.resolved_path.exists()


@dataclass(frozen=True)
class ArtifactManifest:
    """Parsed evaluation artifact manifest with missing/empty state."""

    path: Path
    status: ArtifactStatus
    records: tuple[ArtifactManifestRecord, ...] = ()
    schema_version: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    path_policy: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    @property
    def is_present(self) -> bool:
        return self.status in {"present", "empty"}


@dataclass(frozen=True)
class EvaluationRunArtifacts:
    """Known evaluation artifacts for one run root."""

    run_root: Path
    evaluation_dir: Path
    manifest: ArtifactManifest
    per_image_metrics: CsvArtifact
    summary: CsvArtifact
    weak_tail: CsvArtifact
    residual_heatmaps: CsvArtifact
    skipped: CsvArtifact
    metric_selection_summaries: tuple[CsvArtifact, ...]
    organization_summary: CsvArtifact

    @property
    def tables(self) -> tuple[CsvArtifact, ...]:
        return (
            self.per_image_metrics,
            self.summary,
            self.weak_tail,
            self.residual_heatmaps,
            self.skipped,
            *self.metric_selection_summaries,
            self.organization_summary,
        )


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Known CSV reports under one vs-compare output directory."""

    comparison_dir: Path
    comparison_summary: CsvArtifact
    group_statistics: CsvArtifact
    unpaired_decision_breakdown: CsvArtifact
    unpaired_quantile_comparison: CsvArtifact
    unpaired_threshold_shares: CsvArtifact
    paired_decision_breakdown: CsvArtifact
    paired_delta_summary: CsvArtifact
    paired_sample_deltas: CsvArtifact
    paired_sample_deltas_all_metrics: CsvArtifact
    paired_metric_delta_summary: CsvArtifact

    @property
    def tables(self) -> tuple[CsvArtifact, ...]:
        return (
            self.comparison_summary,
            self.group_statistics,
            self.unpaired_decision_breakdown,
            self.unpaired_quantile_comparison,
            self.unpaired_threshold_shares,
            self.paired_decision_breakdown,
            self.paired_delta_summary,
            self.paired_sample_deltas,
            self.paired_sample_deltas_all_metrics,
            self.paired_metric_delta_summary,
        )


@dataclass(frozen=True)
class PairedEvaluationArtifacts:
    """Two evaluation runs plus optional comparison reports."""

    run_a: EvaluationRunArtifacts
    run_b: EvaluationRunArtifacts
    comparison: ComparisonArtifacts | None = None


def read_evaluation_run_artifacts(
    run_root: str | Path,
    *,
    artifact_manifest_path: str | Path | None = None,
) -> EvaluationRunArtifacts:
    """Read known evaluation artifacts for a run without mutating the run directory."""
    normalized_root = _normalize_path(run_root)
    evaluation_dir = normalized_root / "evaluation"
    manifest = read_artifact_manifest(
        normalized_root,
        artifact_manifest_path=artifact_manifest_path,
    )

    per_image_metrics = read_csv_artifact(
        "per_image_metrics",
        _first_manifest_path(
            manifest,
            "per_image_metrics_csv",
            fallback=evaluation_dir / "per_image_metrics.csv",
        ),
        required_columns=METRIC_FIELDNAMES,
        required=True,
    )
    summary = read_summary_artifact(
        _first_manifest_path(
            manifest,
            "summary_csv",
            fallback=evaluation_dir / "summary.csv",
        ),
        required=False,
    )
    weak_tail = read_csv_artifact(
        "weak_tail",
        _first_manifest_path(
            manifest,
            "weak_tail_csv",
            fallback=evaluation_dir / "weak_tail.csv",
        ),
        required_columns=WEAK_TAIL_FIELDNAMES,
        required=False,
    )
    residual_heatmaps = read_csv_artifact(
        "residual_heatmaps",
        _first_manifest_path(
            manifest,
            "residual_heatmaps_csv",
            fallback=evaluation_dir / "residual_heatmaps.csv",
        ),
        required_columns=RESIDUAL_HEATMAP_FIELDNAMES,
        required=False,
    )
    skipped = read_csv_artifact(
        "skipped",
        _first_manifest_path(
            manifest,
            "skipped_csv",
            fallback=evaluation_dir / "skipped.csv",
        ),
        required_columns=("sample_id", "reason", "target_path", "generated_path"),
        required=False,
    )
    metric_selection_summaries = _read_metric_selection_summaries(
        normalized_root,
        manifest,
    )
    organization_summary = read_csv_artifact(
        "organization_summary",
        _first_manifest_path(
            manifest,
            "organization_summary",
            fallback=evaluation_dir / "sorted_by_metrics" / "organization_summary.csv",
        ),
        required_columns=ORGANIZATION_SUMMARY_FIELDNAMES,
        required=False,
    )

    return EvaluationRunArtifacts(
        run_root=normalized_root,
        evaluation_dir=evaluation_dir,
        manifest=manifest,
        per_image_metrics=per_image_metrics,
        summary=summary,
        weak_tail=weak_tail,
        residual_heatmaps=residual_heatmaps,
        skipped=skipped,
        metric_selection_summaries=metric_selection_summaries,
        organization_summary=organization_summary,
    )


def read_paired_evaluation_artifacts(
    run_a_root: str | Path,
    run_b_root: str | Path,
    *,
    comparison_dir: str | Path | None = None,
) -> PairedEvaluationArtifacts:
    """Read two run artifact bundles and optional precomputed comparison reports."""
    run_a = read_evaluation_run_artifacts(run_a_root)
    run_b = read_evaluation_run_artifacts(run_b_root)
    comparison = read_comparison_artifacts(comparison_dir) if comparison_dir is not None else None
    return PairedEvaluationArtifacts(run_a=run_a, run_b=run_b, comparison=comparison)


def read_comparison_artifacts(comparison_dir: str | Path) -> ComparisonArtifacts:
    """Read known vs-compare CSV reports from one comparison output directory."""
    normalized_dir = _normalize_path(comparison_dir)
    return ComparisonArtifacts(
        comparison_dir=normalized_dir,
        comparison_summary=read_csv_artifact(
            "comparison_summary",
            normalized_dir / "comparison_summary.csv",
            required_columns=_COMPARISON_SUMMARY_REQUIRED_COLUMNS,
            required=False,
        ),
        group_statistics=read_csv_artifact(
            "group_statistics",
            normalized_dir / "group_statistics.csv",
            required_columns=_GROUP_STATISTICS_REQUIRED_COLUMNS,
            required=False,
        ),
        unpaired_decision_breakdown=read_csv_artifact(
            "unpaired_decision_breakdown",
            normalized_dir / "unpaired_decision_breakdown.csv",
            required_columns=DECISION_BREAKDOWN_COLUMNS,
            required=False,
        ),
        unpaired_quantile_comparison=read_csv_artifact(
            "unpaired_quantile_comparison",
            normalized_dir / "unpaired_quantile_comparison.csv",
            required_columns=UNPAIRED_QUANTILE_COLUMNS,
            required=False,
        ),
        unpaired_threshold_shares=read_csv_artifact(
            "unpaired_threshold_shares",
            normalized_dir / "unpaired_threshold_shares.csv",
            required_columns=UNPAIRED_THRESHOLD_COLUMNS,
            required=False,
        ),
        paired_decision_breakdown=read_csv_artifact(
            "paired_decision_breakdown",
            normalized_dir / "paired_decision_breakdown.csv",
            required_columns=DECISION_BREAKDOWN_COLUMNS,
            required=False,
        ),
        paired_delta_summary=read_csv_artifact(
            "paired_delta_summary",
            normalized_dir / "paired_delta_summary.csv",
            required_columns=PAIRED_DELTA_SUMMARY_COLUMNS,
            required=False,
        ),
        paired_sample_deltas=read_csv_artifact(
            "paired_sample_deltas",
            normalized_dir / "paired_sample_deltas.csv",
            required_columns=_PAIRED_SAMPLE_DELTAS_REQUIRED_COLUMNS,
            required=False,
        ),
        paired_sample_deltas_all_metrics=read_csv_artifact(
            "paired_sample_deltas_all_metrics",
            normalized_dir / "paired_sample_deltas_all_metrics.csv",
            required=False,
        ),
        paired_metric_delta_summary=read_csv_artifact(
            "paired_metric_delta_summary",
            normalized_dir / "paired_metric_delta_summary.csv",
            required_columns=_PAIRED_METRIC_DELTA_SUMMARY_REQUIRED_COLUMNS,
            required=False,
        ),
    )


def read_artifact_manifest(
    run_root: str | Path,
    *,
    artifact_manifest_path: str | Path | None = None,
) -> ArtifactManifest:
    """Read evaluation/artifacts.json and resolve manifest paths against the run root."""
    normalized_root = _normalize_path(run_root)
    path = (
        _normalize_path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else normalized_root / "evaluation" / "artifacts.json"
    )

    if not path.is_file():
        return ArtifactManifest(
            path=path,
            status="missing",
            message=f"Artifact manifest not found: {path}",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise MalformedArtifactError(f"Malformed artifact manifest {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise MalformedArtifactError(f"Artifact manifest is not a JSON object: {path}")

    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise MalformedArtifactError(
            f"Unsupported artifact manifest schema_version {schema_version!r}: {path}"
        )

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise MalformedArtifactError(f"Artifact manifest has no artifacts list: {path}")

    records = tuple(
        _manifest_record_from_payload(record, normalized_root, path) for record in raw_artifacts
    )
    status: ArtifactStatus = "present" if records else "empty"
    return ArtifactManifest(
        path=path,
        status=status,
        records=records,
        schema_version=schema_version if isinstance(schema_version, int) else None,
        created_at=_optional_str(payload.get("created_at")),
        updated_at=_optional_str(payload.get("updated_at")),
        path_policy=_optional_str(payload.get("path_policy")),
        payload=payload,
    )


def read_summary_artifact(
    path: str | Path,
    *,
    required: bool = False,
    name: str = "summary",
) -> CsvArtifact:
    """Read summary.csv, including files with evaluation count preambles."""
    csv_path = Path(path)
    if not csv_path.is_file():
        return CsvArtifact(
            name=name,
            path=csv_path,
            status="missing",
            required=required,
            message=f"CSV artifact not found: {csv_path}",
        )

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            fieldnames: list[str] | None = None
            for raw_row in reader:
                if raw_row and raw_row[0] == "metric":
                    fieldnames = raw_row
                    break
            if fieldnames is None:
                return _malformed_csv_artifact(
                    name,
                    csv_path,
                    "CSV header row beginning with 'metric' was not found.",
                    required=required,
                )
            malformed = _validate_fieldnames(name, csv_path, fieldnames, SUMMARY_FIELDNAMES)
            if malformed is not None:
                return _malformed_csv_artifact(name, csv_path, malformed, required=required)

            rows = _read_dict_rows(csv.DictReader(handle, fieldnames=fieldnames))
    except csv.Error as exc:
        return _malformed_csv_artifact(name, csv_path, str(exc), required=required)

    status: ArtifactStatus = "present" if rows else "empty"
    return CsvArtifact(
        name=name,
        path=csv_path,
        status=status,
        required=required,
        rows=rows,
        fieldnames=tuple(fieldnames),
    )


def read_csv_artifact(
    name: str,
    path: str | Path,
    *,
    required_columns: Sequence[str] = (),
    required: bool = False,
) -> CsvArtifact:
    """Read a CSV artifact with optional schema validation and structured missing states."""
    csv_path = Path(path)
    if not csv_path.is_file():
        return CsvArtifact(
            name=name,
            path=csv_path,
            status="missing",
            required=required,
            message=f"CSV artifact not found: {csv_path}",
        )

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames:
                return _malformed_csv_artifact(
                    name,
                    csv_path,
                    "CSV header row is missing.",
                    required=required,
                )
            malformed = _validate_fieldnames(name, csv_path, fieldnames, required_columns)
            if malformed is not None:
                return _malformed_csv_artifact(name, csv_path, malformed, required=required)

            rows = _read_dict_rows(reader)
    except csv.Error as exc:
        return _malformed_csv_artifact(name, csv_path, str(exc), required=required)

    status: ArtifactStatus = "present" if rows else "empty"
    return CsvArtifact(
        name=name,
        path=csv_path,
        status=status,
        required=required,
        rows=rows,
        fieldnames=tuple(fieldnames),
    )


def resolve_manifest_artifact_path(
    run_root: str | Path,
    path: str,
    path_type: str,
) -> Path:
    """Resolve a manifest artifact path using the package path policy."""
    if path_type not in _MANIFEST_PATH_TYPES:
        raise MalformedArtifactError(f"Unsupported artifact path_type: {path_type!r}")

    raw_path = Path(path)
    if path_type == "absolute" or raw_path.is_absolute():
        return raw_path.expanduser().resolve()

    return (_normalize_path(run_root) / raw_path).resolve()


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _required_str(record: Mapping[str, Any], key: str, manifest_path: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise MalformedArtifactError(
            f"Artifact manifest record has invalid {key!r}: {manifest_path}"
        )
    return value


def _manifest_record_from_payload(
    record: object,
    run_root: Path,
    manifest_path: Path,
) -> ArtifactManifestRecord:
    if not isinstance(record, Mapping):
        raise MalformedArtifactError(f"Artifact manifest record is not an object: {manifest_path}")

    stage = _required_str(record, "stage", manifest_path)
    artifact_type = _required_str(record, "artifact_type", manifest_path)
    path = _required_str(record, "path", manifest_path)
    path_type = record.get("path_type")
    if path_type is None:
        path_type = "absolute" if Path(path).is_absolute() else "run_relative"
    if not isinstance(path_type, str) or path_type not in _MANIFEST_PATH_TYPES:
        raise MalformedArtifactError(
            f"Artifact manifest record has invalid 'path_type': {manifest_path}"
        )

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise MalformedArtifactError(
            f"Artifact manifest record metadata is not an object: {manifest_path}"
        )

    return ArtifactManifestRecord(
        stage=stage,
        artifact_type=artifact_type,
        path=path,
        path_type=path_type,
        resolved_path=resolve_manifest_artifact_path(run_root, path, path_type),
        metric=_optional_str(record.get("metric")),
        sample_id=_optional_str(record.get("sample_id")),
        description=_optional_str(record.get("description")) or "",
        metadata=dict(metadata),
    )


def _first_manifest_path(
    manifest: ArtifactManifest,
    artifact_type: str,
    *,
    fallback: Path,
) -> Path:
    for record in manifest.records:
        if record.artifact_type == artifact_type:
            return record.resolved_path
    return fallback


def _read_metric_selection_summaries(
    run_root: Path,
    manifest: ArtifactManifest,
) -> tuple[CsvArtifact, ...]:
    paths: list[Path] = [
        record.resolved_path
        for record in manifest.records
        if record.artifact_type == "selection_summary"
    ]
    metrics_dir = run_root / "comparisons" / "metrics"
    paths.append(metrics_dir / "metrics_selection_summary.csv")

    if metrics_dir.is_dir():
        paths.extend(
            candidate
            for candidate in sorted(metrics_dir.glob("*/selection_summary.csv"))
            if candidate.is_file()
        )

    return tuple(
        read_csv_artifact(
            _run_relative_artifact_name("selection_summary", path, run_root),
            path,
            required_columns=SELECTION_SUMMARY_FIELDNAMES,
            required=False,
        )
        for path in _unique_paths(paths)
    )


def _run_relative_artifact_name(prefix: str, path: Path, run_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        suffix = resolved_path.relative_to(run_root).as_posix()
    except ValueError:
        suffix = resolved_path.name
    return f"{prefix}:{suffix}"


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def _validate_fieldnames(
    name: str,
    path: Path,
    fieldnames: Sequence[str],
    required_columns: Sequence[str],
) -> str | None:
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        return f"CSV artifact {name!r} at {path} is missing required columns: {', '.join(missing)}"
    return None


def _read_dict_rows(reader: csv.DictReader[str]) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise csv.Error(f"row {row_number} has more fields than the header")
        rows.append({key: "" if value is None else value for key, value in row.items()})
    return tuple(rows)


def _malformed_csv_artifact(
    name: str,
    path: Path,
    message: str,
    *,
    required: bool,
) -> CsvArtifact:
    full_message = f"Malformed CSV artifact {name!r} at {path}: {message}"
    if required:
        raise MalformedArtifactError(full_message)
    return CsvArtifact(
        name=name,
        path=path,
        status="malformed",
        required=required,
        message=full_message,
    )
