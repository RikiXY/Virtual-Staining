from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from common.cli_style import color_metric, print_info, print_section, style

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\n"
        "pip install scikit-image"
    ) from exc


METRIC_NAMES = [
    "ssim",
    "psnr",
    "mae",
    "rmse",
    "mse",
    "pcc_rgb_mean",
    "pcc_gray",
]

VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}

METRIC_FIELDNAMES = [
    "sample_id",
    "target_path",
    "generated_path",
    "width",
    "height",
    "channels",
    "mae",
    "mse",
    "rmse",
    "psnr",
    "ssim",
    "pcc_gray",
    "pcc_r",
    "pcc_g",
    "pcc_b",
    "pcc_rgb_mean",
]


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Carica un'immagine da disco e la restituisce come array RGB uint8."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not open image: {image_path}") from exc

    return np.array(image)


def validate_same_shape(target: np.ndarray, generated: np.ndarray) -> None:
    """Verifica che target e generated abbiano esattamente la stessa shape."""
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )


def to_float01(image: np.ndarray) -> np.ndarray:
    """Converte l'immagine da uint8 [0,255] a float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def extract_sample_id(path: str | Path, suffix: str, label: str = "File") -> str:
    """Estrae il sample id togliendo il suffisso atteso dal nome file."""
    name = Path(path).stem

    if not name.endswith(suffix):
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")

    return name[: -len(suffix)]


def extract_single_sample_id(target_path: str | Path, generated_path: str | Path) -> str:
    """Controlla che target e generated appartengano allo stesso sample."""
    target_id = extract_sample_id(target_path, "_target", "Target")
    generated_id = extract_sample_id(generated_path, "_target_generated", "Generated")

    if target_id != generated_id:
        raise ValueError(
            "Target and generated files refer to different sample ids. "
            f"Got '{target_id}' and '{generated_id}'."
        )

    return target_id


def collect_image_files(directory_path: str | Path, suffix: str, label: str) -> dict[str, Path]:
    """Raccoglie i file validi di una cartella indicizzandoli per sample id."""
    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {directory}")

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if not path.stem.endswith(suffix):
            continue

        sample_id = extract_sample_id(path, suffix, label)
        files[sample_id] = path

    return files


def infer_default_output_dir(generated_path: str | Path) -> Path:
    """Prova a ricavare results/NAME_RUN/evaluation da un path interno al run."""
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer output directory from generated path. Expected the generated "
            "data to be inside a path like .../results/NAME_RUN/... Please provide "
            "--output-dir explicitly."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.name == "results":
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. Please provide "
            "--output-dir explicitly."
        )

    return run_dir / "evaluation"


def resolve_output_dir(output_dir: str | None, generated_path: str | Path) -> Path:
    """Usa l'output esplicito se presente, altrimenti prova a inferirlo."""
    if output_dir is not None:
        return Path(output_dir)

    return infer_default_output_dir(generated_path)


def compute_pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Calcola il Pearson Correlation Coefficient tra due immagini."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)

    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return np.nan

    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def rgb_to_gray_float(img: np.ndarray) -> np.ndarray:
    """Converte RGB in grayscale usando pesi standard."""
    if img.ndim == 2:
        return img.astype(np.float64)

    if img.shape[2] < 3:
        return img[..., 0].astype(np.float64)

    img = img.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def compute_pcc_gray(generated: np.ndarray, target: np.ndarray) -> float:
    """Calcola il PCC sulle immagini convertite in scala di grigi."""
    generated_gray = rgb_to_gray_float(generated)
    target_gray = rgb_to_gray_float(target)
    return compute_pcc(generated_gray, target_gray)


def compute_pcc_rgb(generated: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    """Calcola PCC separato sui canali RGB e la media dei tre canali."""
    if generated.ndim != 3 or target.ndim != 3 or generated.shape[2] < 3 or target.shape[2] < 3:
        pcc = compute_pcc(generated, target)
        return np.nan, np.nan, np.nan, pcc

    pcc_r = compute_pcc(generated[..., 0], target[..., 0])
    pcc_g = compute_pcc(generated[..., 1], target[..., 1])
    pcc_b = compute_pcc(generated[..., 2], target[..., 2])

    values = np.array([pcc_r, pcc_g, pcc_b], dtype=np.float64)
    pcc_rgb_mean = float(np.nanmean(values))

    return pcc_r, pcc_g, pcc_b, pcc_rgb_mean


def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    """Calcola il Mean Absolute Error su immagini normalizzate."""
    return float(np.mean(np.abs(target - generated)))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    """Calcola il PSNR assumendo immagini normalizzate nell'intervallo [0,1]."""
    mse = float(np.mean((target - generated) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
    """Calcola l'SSIM su immagini RGB normalizzate."""
    try:
        return float(
            structural_similarity(
                target,
                generated,
                channel_axis=2,
                data_range=1.0,
            )
        )
    except TypeError:
        return float(
            structural_similarity(
                target,
                generated,
                multichannel=True,
                data_range=1.0,
            )
        )


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Calcola le metriche per una coppia target/generated."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)

    validate_same_shape(target, generated)
    shape = target.shape

    target_float = to_float01(target)
    generated_float = to_float01(generated)

    mse = float(np.mean((target_float - generated_float) ** 2))
    pcc_r, pcc_g, pcc_b, pcc_rgb_mean = compute_pcc_rgb(generated_float, target_float)

    metrics = {
        "mae": compute_mae(target_float, generated_float),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr": compute_psnr(target_float, generated_float),
        "ssim": compute_ssim(target_float, generated_float),
        "pcc_gray": compute_pcc_gray(generated_float, target_float),
        "pcc_r": pcc_r,
        "pcc_g": pcc_g,
        "pcc_b": pcc_b,
        "pcc_rgb_mean": pcc_rgb_mean,
    }

    return metrics, shape


def build_metric_row(
    sample_id: str,
    target_path: str | Path,
    generated_path: str | Path,
    shape: tuple[int, int, int],
    metrics: dict[str, float],
) -> dict[str, object]:
    """Costruisce una riga standard per per_image_metrics.csv."""
    height, width, channels = shape

    return {
        "sample_id": sample_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": width,
        "height": height,
        "channels": channels,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "pcc_gray": metrics["pcc_gray"],
        "pcc_r": metrics["pcc_r"],
        "pcc_g": metrics["pcc_g"],
        "pcc_b": metrics["pcc_b"],
        "pcc_rgb_mean": metrics["pcc_rgb_mean"],
    }


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Costruisce le righe aggregate del summary.csv."""
    summary_rows: list[dict[str, object]] = []

    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        summary_rows.append(
            {
                "metric": metric,
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    return summary_rows


def write_per_image_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Scrive il CSV con una riga per ogni coppia valutata."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    """Scrive il CSV dei sample saltati con la relativa motivazione."""
    fieldnames = ["sample_id", "reason", "target_path", "generated_path"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_case_csv(row: dict[str, object], output_path: str | Path) -> None:
    """Scrive il CSV prodotto dalla modalità single."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def write_summary_csv(
    summary_rows: list[dict[str, object]],
    output_path: str | Path,
    num_targets_found: int,
    num_generated_found: int,
    num_pairs_evaluated: int,
    num_skipped: int,
) -> None:
    """Scrive il CSV riassuntivo con conteggi globali e statistiche per metrica."""
    fieldnames = ["metric", "count", "mean", "median", "std", "min", "max"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["num_targets_found", num_targets_found])
        writer.writerow(["num_generated_found", num_generated_found])
        writer.writerow(["num_pairs_evaluated", num_pairs_evaluated])
        writer.writerow(["num_skipped", num_skipped])
        writer.writerow([])
        writer.writerow(fieldnames)

        dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
        dict_writer.writerows(summary_rows)


def print_single_result(
    target_path: str | Path,
    generated_path: str | Path,
    metrics: dict[str, float],
    shape: tuple[int, int, int],
) -> None:
    """Stampa a terminale il riepilogo di una singola valutazione."""
    height, width, channels = shape

    print_section("Single-pair evaluation")
    print_info("Target", str(target_path))
    print_info("Generated", str(generated_path))
    print_info("Shape", f"{width}x{height}x{channels}")
    print()
    print_info("MAE", color_metric("mae", metrics["mae"]))
    print_info("RMSE", color_metric("rmse", metrics["rmse"]))
    print_info("PSNR", color_metric("psnr", metrics["psnr"]))
    print_info("SSIM", color_metric("ssim", metrics["ssim"]))
    print_info("MSE", color_metric("mse", metrics["mse"]))
    print_info("PCC gray", color_metric("pcc_gray", metrics["pcc_gray"]))
    print_info("PCC RGB mean", color_metric("pcc_rgb_mean", metrics["pcc_rgb_mean"]))


def print_dataset_summary(
    target_files: dict[str, Path],
    generated_files: dict[str, Path],
    per_image_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, str]],
) -> None:
    """Stampa un riepilogo finale della modalità dataset."""
    print_section("Dataset evaluation")
    print_info("Targets found", str(len(target_files)))
    print_info("Generated found", str(len(generated_files)))

    pairs_color = "green" if per_image_rows else "red"
    skipped_color = "green" if not skipped_rows else "yellow"
    print_info("Pairs evaluated", style(str(len(per_image_rows)), pairs_color))
    print_info("Skipped", style(str(len(skipped_rows)), skipped_color))

    if per_image_rows:
        print_section("Metric summary")
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in per_image_rows]
            print_info(f"{metric.upper()} mean", color_metric(metric, float(np.mean(values))))
            print_info(f"{metric.upper()} median", color_metric(metric, float(np.median(values))))