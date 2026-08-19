from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_staining.utils.metrics import DEFAULT_METRICS, is_higher_better_metric

logger = logging.getLogger(__name__)

IMAGE_COLUMNS = [
    "generated_path",
    "target_path",
    "source_path",
]


def ensure_parent(path: Path) -> None:
    """Creates the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    """Places a file by hardlink, symlink or copy."""
    if not src.exists():
        logger.warning("Missing file: %s", src)
        return

    ensure_parent(dst)

    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as exc:
            logger.warning("Hard link failed for %s: %s; copying instead", src, exc)
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError as exc:
            logger.warning("Symlink failed for %s: %s; copying instead", src, exc)
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_existing_image_columns(df: pd.DataFrame) -> list[str]:
    """Returns path columns available in the metrics CSV."""
    return [column for column in IMAGE_COLUMNS if column in df.columns]


def infer_role_from_column(column: str) -> str:
    """Infers the image role from a CSV column name."""
    if column == "generated_path":
        return "generated"
    if column == "target_path":
        return "target"
    if column == "source_path":
        return "source"

    return column.replace("_path", "")


def export_ranked_subset(
    df_subset: pd.DataFrame,
    destination_dir: Path,
    image_columns: list[str],
    mode: str,
    overwrite: bool,
) -> int:
    """Exports a ranked DataFrame subset."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    placed_files = 0

    for rank, (_, row) in enumerate(df_subset.iterrows(), start=1):
        sample_id = str(row.get("sample_id", f"sample_{rank:04d}"))

        for column in image_columns:
            if pd.isna(row[column]):
                continue

            src = Path(str(row[column]))
            role = infer_role_from_column(column)
            filename = f"{rank:04d}_{sample_id}_{role}{src.suffix}"
            dst = destination_dir / filename
            before_exists = dst.exists() or dst.is_symlink()
            place_file(src, dst, mode=mode, overwrite=overwrite)
            after_exists = dst.exists() or dst.is_symlink()

            if after_exists and (overwrite or not before_exists):
                placed_files += 1

    return placed_files


def organize_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    image_columns: list[str],
    top_k: int,
    mode: str,
    overwrite: bool,
    include_all_ranked: bool,
) -> dict[str, Any] | None:
    """Organizes files for a single metric."""
    if metric not in df.columns:
        logger.warning("Metric %r not found in CSV; skipping", metric)
        return None

    try:
        higher_is_better = is_higher_better_metric(metric)
    except ValueError:
        logger.warning("Unknown metric direction for %r; skipping", metric)
        return None

    metric_values = pd.to_numeric(df[metric], errors="coerce")
    valid_df = df.loc[metric_values.notna()].copy()
    valid_df[metric] = metric_values[metric_values.notna()]

    if valid_df.empty:
        logger.warning("No valid numeric values for metric %r; skipping", metric)
        return None

    best_df = valid_df.sort_values(metric, ascending=not higher_is_better).head(top_k)
    worst_df = valid_df.sort_values(metric, ascending=higher_is_better).head(top_k)
    metric_dir = output_dir / metric

    best_files = export_ranked_subset(
        best_df,
        destination_dir=metric_dir / "best",
        image_columns=image_columns,
        mode=mode,
        overwrite=overwrite,
    )
    worst_files = export_ranked_subset(
        worst_df,
        destination_dir=metric_dir / "worst",
        image_columns=image_columns,
        mode=mode,
        overwrite=overwrite,
    )

    all_ranked_files = 0
    if include_all_ranked:
        ranked_df = valid_df.sort_values(metric, ascending=not higher_is_better)
        all_ranked_files = export_ranked_subset(
            ranked_df,
            destination_dir=metric_dir / "all_ranked",
            image_columns=image_columns,
            mode=mode,
            overwrite=overwrite,
        )

    return {
        "metric": metric,
        "valid_samples": len(valid_df),
        "best_samples": len(best_df),
        "worst_samples": len(worst_df),
        "best_files": best_files,
        "worst_files": worst_files,
        "all_ranked_files": all_ranked_files,
    }


def write_organization_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Writes the organization summary CSV."""
    summary_path = output_dir / "organization_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def organize_by_metrics(
    csv_path: Path,
    output_dir: Path,
    top_n: int = 10,
    metrics: list[str] | None = None,
    *,
    mode: str = "hardlink",
    overwrite: bool = False,
    include_all_ranked: bool = False,
) -> tuple[list[dict[str, Any]], Path | None, tuple[str, ...]]:
    """
    Read a per_image_metrics.csv and copy/link the top-N and worst-N images
    per metric into <output_dir>/<metric>/best/ and <output_dir>/<metric>/worst/.
    """
    df = pd.read_csv(csv_path)
    image_columns = get_existing_image_columns(df)

    if not image_columns:
        raise ValueError(
            f"No image path columns found. Expected at least one of: {', '.join(IMAGE_COLUMNS)}"
        )

    if "sample_id" not in df.columns:
        logger.warning("Column 'sample_id' not found; ranking will use fallback names")

    selected_metrics = metrics if metrics is not None else list(DEFAULT_METRICS)

    summary_rows: list[dict[str, Any]] = []

    for metric in selected_metrics:
        result = organize_metric(
            df=df,
            metric=metric,
            output_dir=output_dir,
            image_columns=image_columns,
            top_k=top_n,
            mode=mode,
            overwrite=overwrite,
            include_all_ranked=include_all_ranked,
        )

        if result is None:
            continue

        summary_rows.append(result)
    summary_csv = write_organization_summary(summary_rows, output_dir) if summary_rows else None
    return summary_rows, summary_csv, tuple(image_columns)
