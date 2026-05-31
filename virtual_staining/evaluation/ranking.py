from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_staining.evaluation.selection import RankedSample, select_ranked_samples
from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import DEFAULT_METRICS

IMAGE_COLUMNS = [
    "generated_path",
    "target_path",
    "source_path",
]
ORGANIZATION_SUMMARY_FIELDNAMES = [
    "metric",
    "kind",
    "rank_count",
    "export_mode",
    "selected_file_roles",
    "output_dir",
    "files_exported",
]


@dataclass(frozen=True)
class OrganizationResult:
    output_dir: Path
    summary_csv: Path | None
    summary_rows: list[dict[str, Any]]
    metrics: tuple[str, ...]
    image_columns: tuple[str, ...]


def ensure_parent(path: Path) -> None:
    """Creates the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    """Places a file by hardlink, symlink or copy."""
    if not src.exists():
        print_info("Warning", style(f"Missing file: {src}", "yellow"))
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
            print_info("Warning", style(f"Hard link failed for {src}: {exc}", "yellow"))
            print_info("Fallback", "Copying file instead")
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError as exc:
            print_info("Warning", style(f"Symlink failed for {src}: {exc}", "yellow"))
            print_info("Fallback", "Copying file instead")
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


def _samples_to_dataframe(samples: list[RankedSample]) -> pd.DataFrame:
    return pd.DataFrame([sample.row for sample in samples])


def _selected_file_roles(image_columns: list[str]) -> str:
    return ",".join(infer_role_from_column(column) for column in image_columns)


def _export_summary_row(
    *,
    metric: str,
    kind: str,
    rank_count: int,
    export_mode: str,
    image_columns: list[str],
    output_dir: Path,
    files_exported: int,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "kind": kind,
        "rank_count": rank_count,
        "export_mode": export_mode,
        "selected_file_roles": _selected_file_roles(image_columns),
        "output_dir": str(output_dir),
        "files_exported": files_exported,
    }


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
    """Export ranked files for a single metric."""
    if metric not in df.columns:
        print_info("Warning", style(f"Metric '{metric}' not found in CSV. Skipping.", "yellow"))
        return None

    rows = df.to_dict("records")
    try:
        all_ranked_samples = select_ranked_samples(
            rows,
            metric,
            kinds=("best",),
            top_k=max(len(rows), 1),
        )["best"]
    except ValueError:
        print_info(
            "Warning",
            style(f"Unknown metric direction for '{metric}'. Skipping.", "yellow"),
        )
        return None

    if not all_ranked_samples:
        print_info(
            "Warning",
            style(f"No valid numeric values for metric '{metric}'. Skipping.", "yellow"),
        )
        return None

    ranked_samples = select_ranked_samples(
        rows,
        metric,
        kinds=("best", "worst"),
        top_k=top_k,
    )
    best_samples = ranked_samples["best"]
    worst_samples = ranked_samples["worst"]
    best_df = _samples_to_dataframe(best_samples)
    worst_df = _samples_to_dataframe(worst_samples)
    metric_dir = output_dir / metric
    best_dir = metric_dir / "best"
    worst_dir = metric_dir / "worst"

    best_files = export_ranked_subset(
        best_df,
        destination_dir=best_dir,
        image_columns=image_columns,
        mode=mode,
        overwrite=overwrite,
    )
    worst_files = export_ranked_subset(
        worst_df,
        destination_dir=worst_dir,
        image_columns=image_columns,
        mode=mode,
        overwrite=overwrite,
    )
    summary_rows = [
        _export_summary_row(
            metric=metric,
            kind="best",
            rank_count=len(best_samples),
            export_mode=mode,
            image_columns=image_columns,
            output_dir=best_dir,
            files_exported=best_files,
        ),
        _export_summary_row(
            metric=metric,
            kind="worst",
            rank_count=len(worst_samples),
            export_mode=mode,
            image_columns=image_columns,
            output_dir=worst_dir,
            files_exported=worst_files,
        ),
    ]

    all_ranked_files = 0
    if include_all_ranked:
        all_ranked_dir = metric_dir / "all_ranked"
        ranked_df = _samples_to_dataframe(all_ranked_samples)
        all_ranked_files = export_ranked_subset(
            ranked_df,
            destination_dir=all_ranked_dir,
            image_columns=image_columns,
            mode=mode,
            overwrite=overwrite,
        )
        summary_rows.append(
            _export_summary_row(
                metric=metric,
                kind="all_ranked",
                rank_count=len(all_ranked_samples),
                export_mode=mode,
                image_columns=image_columns,
                output_dir=all_ranked_dir,
                files_exported=all_ranked_files,
            )
        )

    return {
        "metric": metric,
        "valid_samples": len(all_ranked_samples),
        "best_samples": len(best_samples),
        "worst_samples": len(worst_samples),
        "best_files": best_files,
        "worst_files": worst_files,
        "all_ranked_files": all_ranked_files,
        "summary_rows": summary_rows,
    }


def write_organization_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write the ranked file export summary CSV."""
    summary_path = output_dir / "organization_summary.csv"
    pd.DataFrame(rows, columns=ORGANIZATION_SUMMARY_FIELDNAMES).to_csv(
        summary_path,
        index=False,
    )
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
) -> OrganizationResult:
    """
    Read per_image_metrics.csv and export ranked sample files by metric.
    """
    df = pd.read_csv(csv_path)
    image_columns = get_existing_image_columns(df)

    if not image_columns:
        raise ValueError(
            f"No image path columns found. Expected at least one of: {', '.join(IMAGE_COLUMNS)}"
        )

    if "sample_id" not in df.columns:
        print_info(
            "Warning",
            style(
                "Column 'sample_id' not found. Ranking will use fallback names.",
                "yellow",
            ),
        )

    selected_metrics = metrics if metrics is not None else list(DEFAULT_METRICS)

    print_section("Ranked sample export")
    print_info("Metrics CSV", str(csv_path))
    print_info("Output dir", style(str(output_dir), "bold", "magenta"))
    print_info("Mode", mode)
    print_info("Top K", str(top_n))
    print_info("Image columns", ", ".join(image_columns))

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

        summary_rows.extend(result["summary_rows"])
        print_info("Organized metric", style(metric, "green"))

    summary_csv: Path | None = None
    if summary_rows:
        summary_csv = write_organization_summary(summary_rows, output_dir)
        print_info("Summary CSV", str(summary_csv))

    print_section("Done")
    print_info("Output written to", style(str(output_dir), "bold", "magenta"))
    return OrganizationResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        summary_rows=summary_rows,
        metrics=tuple(selected_metrics),
        image_columns=tuple(image_columns),
    )
