from __future__ import annotations

import json
from pathlib import Path

import pytest

from virtual_staining.checkpoint_selection import (
    CHECKPOINT_SELECTION_SCHEMA_VERSION,
    latest_checkpoint_path,
    load_best_checkpoint_record,
    resolve_checkpoint_path,
    update_checkpoint_selection,
)


def _checkpoint(root: Path, epoch: int) -> Path:
    path = root / f"ep{epoch:03d}.pth"
    path.touch()
    return path


def _update(
    root: Path,
    checkpoint: Path,
    *,
    epoch: int,
    metric: str = "val_ssim",
    value: float,
    mode: str = "max",
    top_k: int = 3,
) -> None:
    update_checkpoint_selection(
        root,
        metrics={metric: value},
        modes={metric: mode},
        top_k=top_k,
        epoch=epoch,
        checkpoint_path=checkpoint,
    )


def _payload(root: Path) -> dict:
    return json.loads((root / "best.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mode", "values", "expected_epochs"),
    [
        ("max", (0.4, 0.9, 0.7), [2, 3, 1]),
        ("min", (0.4, 0.9, 0.2), [3, 1, 2]),
    ],
)
def test_ranking_respects_mode(
    tmp_path: Path, mode: str, values: tuple[float, ...], expected_epochs: list[int]
) -> None:
    for epoch, value in enumerate(values, start=1):
        _update(tmp_path, _checkpoint(tmp_path, epoch), epoch=epoch, value=value, mode=mode)

    records = _payload(tmp_path)["metrics"]["val_ssim"]["records"]
    assert [record["epoch"] for record in records] == expected_epochs
    assert [record["rank"] for record in records] == [1, 2, 3]


@pytest.mark.parametrize("mode", ["min", "max"])
def test_ties_prefer_earlier_epoch_deterministically(tmp_path: Path, mode: str) -> None:
    for epoch in (3, 1, 2):
        _update(tmp_path, _checkpoint(tmp_path, epoch), epoch=epoch, value=0.5, mode=mode)

    records = _payload(tmp_path)["metrics"]["val_ssim"]["records"]
    assert [record["epoch"] for record in records] == [1, 2, 3]


def test_same_epoch_update_replaces_existing_record(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 4)
    _update(tmp_path, checkpoint, epoch=4, value=0.2)
    _update(tmp_path, checkpoint, epoch=4, value=0.8)

    records = _payload(tmp_path)["metrics"]["val_ssim"]["records"]
    assert records == [{"epoch": 4, "checkpoint_path": "ep004.pth", "metric_value": 0.8, "rank": 1}]


def test_top_k_truncates_and_reassigns_contiguous_ranks(tmp_path: Path) -> None:
    for epoch, value in ((1, 0.1), (2, 0.4), (3, 0.3), (4, 0.2)):
        _update(
            tmp_path,
            _checkpoint(tmp_path, epoch),
            epoch=epoch,
            value=value,
            top_k=2,
        )

    records = _payload(tmp_path)["metrics"]["val_ssim"]["records"]
    assert [(record["epoch"], record["rank"]) for record in records] == [(2, 1), (3, 2)]


def test_update_prunes_stale_checkpoint_records(tmp_path: Path) -> None:
    stale = _checkpoint(tmp_path, 1)
    _update(tmp_path, stale, epoch=1, value=0.9)
    stale.unlink()

    current = _checkpoint(tmp_path, 2)
    _update(tmp_path, current, epoch=2, value=0.8)

    records = _payload(tmp_path)["metrics"]["val_ssim"]["records"]
    assert [record["checkpoint_path"] for record in records] == ["ep002.pth"]


def test_ranked_resolution_reports_stale_selected_file(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 1)
    _update(tmp_path, checkpoint, epoch=1, value=0.9)
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match="points to missing file"):
        resolve_checkpoint_path(tmp_path, policy="best", metric="val_ssim")


def test_latest_resolution_uses_numeric_epoch_and_ignores_malformed_names(tmp_path: Path) -> None:
    older = _checkpoint(tmp_path, 999)
    latest = _checkpoint(tmp_path, 1000)
    (tmp_path / "epfinal.pth").touch()
    (tmp_path / "ep2000.pth").mkdir()

    assert latest_checkpoint_path(tmp_path) == latest
    assert resolve_checkpoint_path(tmp_path, policy="latest") == latest
    assert latest != older


def test_best_and_top_k_resolution_select_expected_rank(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path, 1)
    second = _checkpoint(tmp_path, 2)
    _update(tmp_path, first, epoch=1, value=0.7)
    _update(tmp_path, second, epoch=2, value=0.9)

    assert resolve_checkpoint_path(tmp_path, policy="best", metric="val_ssim") == second
    assert resolve_checkpoint_path(tmp_path, policy="top_k", metric="val_ssim", rank=2) == first


def test_one_checkpoint_can_rank_independently_under_multiple_metrics(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path, 1)
    second = _checkpoint(tmp_path, 2)
    update_checkpoint_selection(
        tmp_path,
        metrics={"loss_G_val": 0.2, "val_ssim": 0.8},
        modes={"loss_G_val": "min", "val_ssim": "max"},
        top_k=2,
        epoch=1,
        checkpoint_path=first,
    )
    update_checkpoint_selection(
        tmp_path,
        metrics={"loss_G_val": 0.1, "val_ssim": 0.7},
        modes={"loss_G_val": "min", "val_ssim": "max"},
        top_k=2,
        epoch=2,
        checkpoint_path=second,
    )

    assert resolve_checkpoint_path(tmp_path, policy="best", metric="loss_G_val") == second
    assert resolve_checkpoint_path(tmp_path, policy="best", metric="val_ssim") == first
    assert set(_payload(tmp_path)["metrics"]) == {"loss_G_val", "val_ssim"}


def test_selection_schema_version_has_single_owner(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 1)
    _update(tmp_path, checkpoint, epoch=1, value=0.5)

    assert _payload(tmp_path)["schema_version"] == CHECKPOINT_SELECTION_SCHEMA_VERSION


def test_legacy_single_record_is_rejected_actionably(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    (tmp_path / "best.json").write_text(
        json.dumps(
            {
                "policy": "best_val_loss",
                "metric": "loss_G_val",
                "epoch": 3,
                "checkpoint_path": checkpoint.name,
                "metric_value": 0.4,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy/unversioned best.json records are not supported"):
        load_best_checkpoint_record(tmp_path, policy="best", metric="loss_G_val")


def test_unknown_future_schema_is_rejected_actionably(tmp_path: Path) -> None:
    (tmp_path / "best.json").write_text(
        json.dumps({"schema_version": CHECKPOINT_SELECTION_SCHEMA_VERSION + 1, "metrics": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint selection schema_version"):
        load_best_checkpoint_record(tmp_path, policy="best", metric="loss_G_val")


def test_unversioned_multi_metric_schema_is_rejected_actionably(tmp_path: Path) -> None:
    (tmp_path / "best.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing schema_version"):
        load_best_checkpoint_record(tmp_path, policy="best", metric="loss_G_val")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_is_rejected_before_writing(tmp_path: Path, value: float) -> None:
    checkpoint = _checkpoint(tmp_path, 1)

    with pytest.raises(ValueError, match="must be finite"):
        _update(tmp_path, checkpoint, epoch=1, value=value)
    assert not (tmp_path / "best.json").exists()
