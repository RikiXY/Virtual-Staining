from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.builder import DatasetBuildResult


def test_dataset_build_result_fields() -> None:
    result = DatasetBuildResult(
        train_count=800,
        val_count=50,
        test_count=150,
        skipped_count=30,
        output_root=Path("/data/experiment"),
    )
    assert result.train_count == 800
    assert result.val_count == 50
    assert result.test_count == 150
    assert result.skipped_count == 30
    assert result.output_root == Path("/data/experiment")
    assert result.reused is False


def test_dataset_build_result_output_root_is_path() -> None:
    result = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/some/path"),
    )
    assert isinstance(result.output_root, Path)
    assert isinstance(result.reused, bool)


def test_dataset_build_result_frozen() -> None:
    result = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/tmp"),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.train_count = 999  # type: ignore[misc]


def test_dataset_build_result_reused_flag_is_settable() -> None:
    result = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/tmp"),
        reused=True,
    )
    assert result.reused is True
