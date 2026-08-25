from __future__ import annotations

import csv
from pathlib import Path

import pytest

from virtual_staining.training.helpers import metrics_fieldnames
from virtual_staining.training.history import TrainingHistory
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.validation_metrics import VALIDATION_IMAGE_METRIC_NAMES


def _metrics(epoch: int, *, validation: bool = True) -> tuple[EpochMetrics, EpochMetrics | None]:
    train = EpochMetrics(
        1.0 + epoch,
        2.0 + epoch,
        raw={"generator_l1": 3.0 + epoch},
        weighted={"generator_l1": 4.0 + epoch},
        current_weight={"generator_l1": 1.0},
    )
    val = (
        EpochMetrics(5.0 + epoch, 6.0 + epoch, image={"val_ssim": 0.5 + epoch})
        if validation
        else None
    )
    return train, val


def test_history_writes_one_union_csv_and_flushes(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "epochs.csv"
    with TrainingHistory(path, ["generator_l1"], resume_at=0) as history:
        reported = history.write_epoch(0, *_metrics(0))
        assert path.read_text(encoding="utf-8").count("\n") == 2
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "0"
    assert rows[0]["loss_G_train"] == "1.000000"
    assert rows[0]["loss_G_val"] == "5.000000"
    assert reported["val_ssim"] == 0.5
    assert rows[0]["loss_D_val"] == "6.000000"
    assert not (tmp_path / "metrics" / "train.csv").exists()


def test_history_blanks_validation_columns_when_validation_does_not_run(tmp_path: Path) -> None:
    path = tmp_path / "epochs.csv"
    with TrainingHistory(path, [], resume_at=0) as history:
        history.write_epoch(0, *_metrics(0, validation=False))
    row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert row["loss_G_val"] == ""
    assert row["val_ssim"] == ""


def test_resume_reconciles_epoch_history(tmp_path: Path) -> None:
    path = tmp_path / "epochs.csv"
    with TrainingHistory(path, [], resume_at=0) as history:
        for epoch in range(3):
            history.write_epoch(epoch, *_metrics(epoch))
    original = path.read_text(encoding="utf-8").splitlines()
    with TrainingHistory(path, [], resume_at=2) as history:
        history.write_epoch(2, *_metrics(9))
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [int(row["epoch"]) for row in rows] == [0, 1, 2]
    assert rows[0]["loss_G_train"] == original[1].split(",")[1]
    assert rows[1]["loss_G_train"] == original[2].split(",")[1]
    assert rows[2]["loss_G_train"] == "10.000000"


def test_resume_rejects_missing_gapped_duplicate_and_mismatched_history(tmp_path: Path) -> None:
    path = tmp_path / "epochs.csv"
    with pytest.raises(FileNotFoundError), TrainingHistory(path, [], resume_at=1):
        pass
    path.write_text("epoch,wrong\n0,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"), TrainingHistory(path, [], resume_at=1):
        pass

    fields = metrics_fieldnames([]) + VALIDATION_IMAGE_METRIC_NAMES
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: ("0" if name == "epoch" else "") for name in fields})
        writer.writerow({name: ("2" if name == "epoch" else "") for name in fields})
    with pytest.raises(ValueError, match="0..1"), TrainingHistory(path, [], resume_at=2):
        pass
