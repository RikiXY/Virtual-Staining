from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from common.cli_style import print_info, print_section, style


HIGHER_IS_BETTER = {
    "ssim": True,
    "psnr": True,
    "mae": False,
    "rmse": False,
    "mse": False,
    "pcc_gray": True,
    "pcc_rgb_mean": True,
    "pcc_r": True,
    "pcc_g": True,
    "pcc_b": True,
}

DEFAULT_METRICS = [
    "ssim",
    "psnr",
    "mae",
    "rmse",
    "mse",
    "pcc_rgb_mean",
    "pcc_gray",
]

IMAGE_COLUMNS = [
    "generated_path",
    "target_path",
    "source_path",
]


def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser CLI."""
    parser = argparse.ArgumentParser(
        prog="python tools/organize_by_metric.py",
        description=(
            "Organize generated, target and source images by metric ranking. "
            "By default, the script reads RUN/evaluation/per_image_metrics.csv "
            "and writes to RUN/evaluation/sorted_by_metrics/."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/organize_by_metric.py \\\n"
            "      --run-path local_workspace/results/RUN_NAME \\\n"
            "      --top-k 50\n"
            "\n"
            "  python tools/organize_by_metric.py \\\n"
            "      --metrics-csv local_workspace/results/RUN_NAME/evaluation/per_image_metrics.csv \\\n"
            "      --output-dir local_workspace/results/RUN_NAME/evaluation/sorted_by_metrics \\\n"
            "      --top-k 50 \\\n"
            "      --mode hardlink\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--run-path",
        type=Path,
        default=None,
        help=(
            "Path to a run directory like local_workspace/results/RUN_NAME. "
            "The script will read RUN/evaluation/per_image_metrics.csv."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help=(
            "Path to per_image_metrics.csv. Advanced alternative to --run-path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where sorted metric folders will be created. "
            "If omitted, defaults to RUN/evaluation/sorted_by_metrics/."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to use for sorting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of best/worst samples to export for each metric.",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to place files in the output folders.",
    )
    parser.add_argument(
        "--include-all-ranked",
        action="store_true",
        help="Also create a full ranked folder for each metric.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing links/files if present.",
    )

    return parser


def resolve_run_path(run_path: str | Path) -> Path:
    """Valida e risolve la directory di un run."""
    path = Path(run_path).resolve()

    if not path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {path}")

    return path


def resolve_metrics_csv(args: argparse.Namespace) -> Path:
    """Risolve il CSV delle metriche da --run-path o --metrics-csv."""
    if args.metrics_csv is not None:
        metrics_csv = args.metrics_csv.resolve()

        if not metrics_csv.is_file():
            raise FileNotFoundError(f"CSV not found: {metrics_csv}")

        return metrics_csv

    if args.run_path is None:
        raise ValueError("You must provide either --run-path or --metrics-csv.")

    run_path = resolve_run_path(args.run_path)
    metrics_csv = run_path / "evaluation" / "per_image_metrics.csv"

    if not metrics_csv.is_file():
        raise FileNotFoundError(
            f"Could not find per_image_metrics.csv. Expected: {metrics_csv}"
        )

    return metrics_csv


def infer_run_path_from_metrics_csv(metrics_csv: str | Path) -> Path | None:
    """Prova a risalire alla cartella del run partendo dal CSV."""
    path = Path(metrics_csv).resolve()

    if path.name == "per_image_metrics.csv" and path.parent.name == "evaluation":
        return path.parent.parent

    return None


def resolve_output_dir(args: argparse.Namespace, metrics_csv: Path) -> Path:
    """Risolvi la directory di output."""
    if args.output_dir is not None:
        return args.output_dir.resolve()

    if args.run_path is not None:
        run_path = resolve_run_path(args.run_path)
        return run_path / "evaluation" / "sorted_by_metrics"

    inferred_run_path = infer_run_path_from_metrics_csv(metrics_csv)

    if inferred_run_path is not None:
        return inferred_run_path / "evaluation" / "sorted_by_metrics"

    raise ValueError(
        "Could not infer output directory. Please provide --output-dir explicitly."
    )


def ensure_parent(path: Path) -> None:
    """Crea la directory parent di un file, se necessario."""
    path.parent.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    """Posiziona un file tramite hardlink, symlink o copia."""
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


def safe_metric_value(value: float) -> str:
    """Formatta un valore di metrica in modo sicuro per il nome file."""
    return f"{value:.6f}".replace(".", "p")


def get_existing_image_columns(df: pd.DataFrame) -> list[str]:
    """Restituisce le colonne path disponibili nel CSV."""
    return [col for col in IMAGE_COLUMNS if col in df.columns]


def infer_role_from_column(column: str) -> str:
    """Inferisce il ruolo immagine dal nome della colonna."""
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
    """Esporta una porzione ordinata del DataFrame."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    placed_files = 0

    for rank, (_, row) in enumerate(df_subset.iterrows(), start=1):
        sample_id = str(row.get("sample_id", f"sample_{rank:04d}"))

        for column in image_columns:
            src = Path(row[column])
            role = infer_role_from_column(column)
            suffix = src.suffix
            filename = f"{rank:04d}_{sample_id}_{role}{suffix}"
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
    """Organizza i file per una singola metrica."""
    if metric not in df.columns:
        print_info("Warning", style(f"Metric '{metric}' not found in CSV. Skipping.", "yellow"))
        return None

    if metric not in HIGHER_IS_BETTER:
        print_info("Warning", style(f"Unknown metric direction for '{metric}'. Skipping.", "yellow"))
        return None

    higher_is_better = HIGHER_IS_BETTER[metric]
    metric_values = pd.to_numeric(df[metric], errors="coerce")
    valid_df = df.loc[metric_values.notna()].copy()
    valid_df[metric] = metric_values[metric_values.notna()]

    if valid_df.empty:
        print_info("Warning", style(f"No valid numeric values for metric '{metric}'. Skipping.", "yellow"))
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
    """Scrive un CSV riassuntivo dell'organizzazione."""
    summary_path = output_dir / "organization_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics_csv = resolve_metrics_csv(args)
    output_dir = resolve_output_dir(args, metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    image_columns = get_existing_image_columns(df)

    if not image_columns:
        raise ValueError(
            "No image path columns found. Expected at least one of: "
            f"{', '.join(IMAGE_COLUMNS)}"
        )

    if "sample_id" not in df.columns:
        print_info("Warning", style("Column 'sample_id' not found. Ranking will use fallback names.", "yellow"))

    print_section("Organize outputs by metric")
    print_info("Metrics CSV", str(metrics_csv))
    print_info("Output dir", style(str(output_dir), "bold", "magenta"))
    print_info("Mode", args.mode)
    print_info("Top K", str(args.top_k))
    print_info("Image columns", ", ".join(image_columns))

    summary_rows: list[dict[str, Any]] = []

    for metric in args.metrics:
        result = organize_metric(
            df=df,
            metric=metric,
            output_dir=output_dir,
            image_columns=image_columns,
            top_k=args.top_k,
            mode=args.mode,
            overwrite=args.overwrite,
            include_all_ranked=args.include_all_ranked,
        )

        if result is None:
            continue

        summary_rows.append(result)
        print_info("Organized metric", style(metric, "green"))

    if summary_rows:
        summary_csv = write_organization_summary(summary_rows, output_dir)
        print_info("Summary CSV", str(summary_csv))

    print_section("Done")
    print_info("Output written to", style(str(output_dir), "bold", "magenta"))


if __name__ == "__main__":
    main()