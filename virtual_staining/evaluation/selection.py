from __future__ import annotations

import csv
from pathlib import Path

from virtual_staining.metrics import DEFAULT_METRICS, is_higher_better_metric
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

METRIC_SELECTION_ORDER = list(DEFAULT_METRICS)
SELECTION_SUMMARY_FIELDNAMES = [
    "metric",
    "kind",
    "sample_id",
    "metric_value",
    "target_value",
    "abs_distance_from_target",
    "source_path",
    "target_path",
    "generated_path",
    "comparison_path",
]


def extract_generated_sample_id(path: str | Path) -> str:
    stem = Path(path).stem
    suffix = "_target_generated"
    if not stem.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")
    return stem[: -len(suffix)]


def find_existing_image(base_dir: str | Path, sample_id: str, suffix: str) -> Path:
    directory = Path(base_dir)
    for ext in sorted(VALID_IMAGE_EXTENSIONS):
        candidate = directory / f"{sample_id}{suffix}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find image for sample '{sample_id}' with suffix '{suffix}' inside {directory}"
    )


def infer_source_path_from_row(row: dict[str, str]) -> Path:
    sample_id = row["sample_id"]
    if row.get("source_path"):
        candidate = Path(row["source_path"])
        if candidate.is_file():
            return candidate

    if row.get("target_path"):
        try:
            return find_existing_image(Path(row["target_path"]).parent, sample_id, "_source")
        except FileNotFoundError:
            pass

    if row.get("generated_path"):
        generated_path = Path(row["generated_path"])
        try:
            return find_existing_image(
                generated_path.parents[1] / "splits" / "test", sample_id, "_source"
            )
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not infer source path for sample '{sample_id}'.")


def select_representative_rows(
    metric_name: str,
    metric_summary: dict[str, float],
    per_image_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    if not per_image_rows:
        raise ValueError("No per-image rows available for representative selection.")

    def metric_value(row: dict[str, str]) -> float:
        return float(row[metric_name])

    higher_is_better = is_higher_better_metric(metric_name)
    return {
        "best": (max if higher_is_better else min)(per_image_rows, key=metric_value),
        "median": min(
            per_image_rows,
            key=lambda row: abs(metric_value(row) - metric_summary["median"]),
        ),
        "worst": (min if higher_is_better else max)(per_image_rows, key=metric_value),
    }


def build_selection_summary_row(
    metric_name: str,
    kind: str,
    sample_id: str,
    metric_value: float,
    target_value: float,
    source_path: Path,
    target_path: Path,
    generated_path: Path,
    comparison_path: Path,
) -> dict[str, object]:
    return {
        "metric": metric_name,
        "kind": kind,
        "sample_id": sample_id,
        "metric_value": metric_value,
        "target_value": target_value,
        "abs_distance_from_target": abs(metric_value - target_value),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "comparison_path": str(comparison_path),
    }


def write_metric_selection_summary(rows: list[dict[str, object]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SELECTION_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
