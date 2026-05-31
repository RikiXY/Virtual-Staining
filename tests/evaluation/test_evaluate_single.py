from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.applications.evaluate_single import EvaluateSingleRequest, evaluate_single


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    Image.fromarray(image).save(path)


def test_evaluate_single_writes_individual_case_csv_only(tmp_path: Path) -> None:
    target_dir = tmp_path / "dataset" / "splits" / "test"
    generated_dir = tmp_path / "results" / "run" / "artifacts" / "output_test"
    output_dir = tmp_path / "results" / "run" / "evaluation"
    target_path = target_dir / "00512_09216_target.png"
    generated_path = generated_dir / "00512_09216_target_generated.png"
    _write_rgb(target_path)
    _write_rgb(generated_path)

    result = evaluate_single(
        EvaluateSingleRequest(
            target_dir=target_dir,
            generated_dir=generated_dir,
            output_dir=output_dir,
            sample_id="00512_09216",
        )
    )

    assert result.single_case_csv == output_dir / "individual_cases" / "00512_09216_evaluation.csv"
    assert result.single_case_csv.exists()
    assert not (output_dir / "per_image_metrics.csv").exists()
    assert not (output_dir / "summary.csv").exists()
    assert not (output_dir / "skipped.csv").exists()

    with result.single_case_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "00512_09216"
    assert rows[0]["target_path"] == str(target_path)
    assert rows[0]["generated_path"] == str(generated_path)
    assert rows[0]["width"] == "8"
    assert rows[0]["height"] == "8"
    assert rows[0]["channels"] == "3"
