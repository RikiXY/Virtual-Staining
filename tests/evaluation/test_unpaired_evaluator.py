from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.evaluation.unpaired import FEATURE_NAMES, evaluate_unpaired_distributions


def _write_image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
    return path


def test_unpaired_evaluation_writes_statistics_and_graphs(tmp_path: Path) -> None:
    generated = (
        _write_image(tmp_path / "generated" / "a.png", 80),
        _write_image(tmp_path / "generated" / "b.png", 120),
    )
    real_target = (
        _write_image(tmp_path / "real" / "c.png", 90),
        _write_image(tmp_path / "real" / "d.png", 130),
    )

    result = evaluate_unpaired_distributions(
        generated,
        real_target,
        tmp_path / "evaluation",
        method="cyclegan",
        direction="A_to_B",
        save_graphs=True,
    )

    assert result.generated_count == 2
    assert result.real_target_count == 2
    assert len(result.graph_paths) == 2
    assert all(path.is_file() for path in result.graph_paths)
    with result.statistics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert set(rows[0]) == {"group", "path", *FEATURE_NAMES}
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["protocol"] == "unpaired"
    assert metadata["paired_metrics_available"] is False
    assert set(metadata["features"]) == set(FEATURE_NAMES)


def test_unpaired_evaluation_can_skip_graphs(tmp_path: Path) -> None:
    generated = (_write_image(tmp_path / "generated.png", 80),)
    real_target = (_write_image(tmp_path / "real.png", 90),)

    result = evaluate_unpaired_distributions(
        generated,
        real_target,
        tmp_path / "evaluation",
        method="custom",
        direction=None,
        save_graphs=False,
    )

    assert result.graph_paths == ()
    assert not (tmp_path / "evaluation" / "unpaired_feature_distributions.png").exists()
