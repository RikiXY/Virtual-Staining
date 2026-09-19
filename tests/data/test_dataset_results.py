from __future__ import annotations

import json
from pathlib import Path

import pytest

from virtual_staining.data.builder import DatasetBuildResult
from virtual_staining.data.manifest import MANIFEST_SCHEMA_VERSION


def test_dataset_build_metadata_version_and_round_trip(tmp_path: Path) -> None:
    result = DatasetBuildResult(8, 1, 2, 3, tmp_path)
    path = tmp_path / "dataset_build.json"
    result.save(path, num_sets=2, num_sets_excluded=0)
    assert json.loads(path.read_text())["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert DatasetBuildResult.load(path, output_root=tmp_path) == result


@pytest.mark.parametrize("version", [None, "2.0", "unknown", 3.0])
def test_dataset_build_metadata_rejects_invalid_version(tmp_path: Path, version: object) -> None:
    path = tmp_path / "dataset_build.json"
    data: dict[str, object] = {"patches": {"train": 8, "val": 1, "test": 2, "discarded": 3}}
    if version is not None:
        data["schema_version"] = version
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid dataset build metadata"):
        DatasetBuildResult.load(path, output_root=tmp_path)


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
