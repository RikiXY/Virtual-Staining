from __future__ import annotations

import csv
from pathlib import Path

from common.cli_style import (
    metric_color_name,
    print_info,
    print_section,
    style,
)

METRIC_SELECTION_ORDER = [
    "ssim",
    "psnr",
    "mae",
    "rmse",
    "mse",
    "pcc_rgb_mean",
    "pcc_gray",
]

HIGHER_IS_BETTER = {
    "ssim": True,
    "psnr": True,
    "mae": False,
    "rmse": False,
    "mse": False,
    "pcc_rgb_mean": True,
    "pcc_gray": True,
}

VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}

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
    """Estrae il sample id dal nome del generated file."""
    stem = Path(path).stem
    suffix = "_target_generated"

    if not stem.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")

    return stem[: -len(suffix)]


def infer_run_dir_from_generated_path(generated_path: str | Path) -> Path:
    """Prova a ricavare la cartella del run da un path generated dentro results/."""
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer run directory from generated path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. "
            "Please provide --save-path explicitly."
        )

    return run_dir


def infer_default_save_path(generated_image: str | Path) -> Path:
    """Costruisce il save path di default per un confronto singolo."""
    generated_path = Path(generated_image)
    sample_id = extract_generated_sample_id(generated_path)
    run_dir = infer_run_dir_from_generated_path(generated_path)
    return run_dir / "comparisons" / f"{sample_id}_comparison.png"


def infer_diagnostics_dir(save_path: str | Path) -> Path:
    """Ricava la cartella diagnostics a partire dal save path del pannello."""
    save_path = Path(save_path)
    return save_path.parent / "diagnostics"


def infer_case_diagnostics_dir(save_path: str | Path, generated_image: str | Path) -> Path:
    """Ricava la cartella diagnostics del singolo sample."""
    diagnostics_dir = infer_diagnostics_dir(save_path)
    sample_id = extract_generated_sample_id(generated_image)
    return diagnostics_dir / sample_id


def find_existing_image(base_dir: str | Path, sample_id: str, suffix: str) -> Path:
    """Cerca un file immagine esistente provando tutte le estensioni supportate."""
    directory = Path(base_dir)

    for ext in sorted(VALID_IMAGE_EXTENSIONS):
        candidate = directory / f"{sample_id}{suffix}{ext}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find image for sample '{sample_id}' with suffix '{suffix}' inside {directory}"
    )


def infer_source_path_from_row(row: dict[str, str]) -> Path:
    """Prova a ricostruire il path della source a partire da una riga CSV."""
    sample_id = row["sample_id"]

    if row.get("source_path"):
        candidate = Path(row["source_path"])
        if candidate.is_file():
            return candidate

    if row.get("target_path"):
        target_dir = Path(row["target_path"]).parent
        try:
            return find_existing_image(target_dir, sample_id, "_source")
        except FileNotFoundError:
            pass

    if row.get("generated_path"):
        generated_path = Path(row["generated_path"])
        dataset_test_dir = generated_path.parents[1] / "dataset_test"
        try:
            return find_existing_image(dataset_test_dir, sample_id, "_source")
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not infer source path for sample '{sample_id}'.")


def read_summary_csv(path: str | Path) -> dict[str, dict[str, float]]:
    """Legge summary.csv e restituisce le statistiche aggregate per metrica."""
    summary_path = Path(path)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    rows: dict[str, dict[str, float]] = {}

    with summary_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header_found = False

        for row in reader:
            if not row:
                continue

            if row[0] == "metric":
                header_found = True
                continue

            if not header_found:
                continue

            metric_name = row[0].strip().lower()
            rows[metric_name] = {
                "count": float(row[1]),
                "mean": float(row[2]),
                "median": float(row[3]),
                "std": float(row[4]),
                "min": float(row[5]),
                "max": float(row[6]),
            }

    return rows


def read_per_image_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    """Legge per_image_metrics.csv e restituisce tutte le righe come dizionari."""
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Per-image metrics CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_representative_rows(
    metric_name: str,
    metric_summary: dict[str, float],
    per_image_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Seleziona i sample best, median e worst per una metrica."""
    if not per_image_rows:
        raise ValueError("No per-image rows available for representative selection.")

    if metric_name not in HIGHER_IS_BETTER:
        raise ValueError(f"Unknown metric direction for metric: {metric_name}")

    higher_is_better = HIGHER_IS_BETTER[metric_name]

    if higher_is_better:
        best_row = max(per_image_rows, key=lambda row: float(row[metric_name]))
        worst_row = min(per_image_rows, key=lambda row: float(row[metric_name]))
    else:
        best_row = min(per_image_rows, key=lambda row: float(row[metric_name]))
        worst_row = max(per_image_rows, key=lambda row: float(row[metric_name]))

    median_row = min(
        per_image_rows,
        key=lambda row: abs(float(row[metric_name]) - metric_summary["median"]),
    )

    return {
        "best": best_row,
        "median": median_row,
        "worst": worst_row,
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
    """Costruisce una riga standard per i CSV di selezione."""
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
    """Scrive il CSV con i sample selezionati per ogni metrica."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SELECTION_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_single_summary(saved_path: Path, diagnostic_paths: list[Path]) -> None:
    """Stampa il riepilogo finale della modalità single."""
    print_section("Single comparison")
    print_info("Saved comparison image", style(str(saved_path), "green"))

    for diagnostic_path in diagnostic_paths:
        print_info("Saved diagnostic plot", style(str(diagnostic_path), "magenta"))


def print_metric_based_selection(
    metric_name: str,
    representative_rows: dict[str, dict[str, str]],
) -> None:
    """Stampa i sample rappresentativi scelti per una metrica."""
    print_section(f"Metric {metric_name.upper()}")

    for kind, row in representative_rows.items():
        metric_value = float(row[metric_name])
        sample_id = row["sample_id"]
        color = metric_color_name(metric_name, metric_value)
        print_info(
            f"{kind.upper()} sample",
            style(f"{sample_id} | value={metric_value:.6f}", color),
        )


def print_metric_run_header(run_path: Path, available_metrics: list[str]) -> None:
    """Stampa l'intestazione generale della modalità from-metrics."""
    print_section("Metric-based representative comparisons")
    print_info("Run path", str(run_path))
    print_info("Metrics found", ", ".join(available_metrics))


def print_metric_saved_files(metrics_dir: Path) -> None:
    """Stampa il riepilogo finale dei file salvati in modalità from-metrics."""
    print_section("Saved files")
    print_info("Metric-based comparisons", style(str(metrics_dir), "bold", "magenta"))