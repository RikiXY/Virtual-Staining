from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.manifest import DatasetManifest, ManifestMetadata, ManifestRecord


def _manifest(tmp_path: Path, *, modalities: tuple[str, ...] = ("LF", "AF")) -> DatasetManifest:
    metadata = ManifestMetadata("3.0", modalities, modalities[0], "target")
    inputs = {name: Path(f"splits/train/a__input__{name}.png") for name in modalities}
    return DatasetManifest(
        (
            ManifestRecord(
                "a", "set-1", "train", inputs, Path("splits/train/a__target.png"), 0, 0, 16, 16
            ),
        ),
        tmp_path,
        metadata,
    )


def test_v3_dynamic_columns_round_trip(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path)
    assert manifest.fieldnames == (
        "sample_id",
        "set_id",
        "split",
        "input__LF",
        "input__AF",
        "target_path",
        "foreground_mask_path",
        "x",
        "y",
        "width",
        "height",
    )
    loaded = DatasetManifest.from_csv(path, tmp_path, manifest.metadata)
    assert loaded.records == manifest.records


def test_manifest_requires_metadata_and_exact_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,set_id,split,input_path,target_path\na,s,train,a.png,t.png\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="required|exact v3 columns"):
        DatasetManifest.from_csv(path, tmp_path)
    with pytest.raises(ValueError, match="exact v3 columns"):
        DatasetManifest.from_csv(path, tmp_path, ManifestMetadata("3.0", ("LF",), "LF", "target"))


@pytest.mark.parametrize("bad_path", ["/absolute.png", "../outside.png", ""])
def test_manifest_rejects_unsafe_or_blank_paths(tmp_path: Path, bad_path: str) -> None:
    if bad_path:
        with pytest.raises(ValueError, match="relative|non-traversing"):
            ManifestRecord(
                "a", "s", "train", {"LF": Path(bad_path)}, Path("target.png"), 0, 0, 1, 1
            )
        return
    metadata = ManifestMetadata("3.0", ("LF",), "LF", "target")
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,set_id,split,input__LF,target_path,foreground_mask_path,x,y,width,height\na,s,train,,target.png,,0,0,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not be empty"):
        DatasetManifest.from_csv(path, tmp_path, metadata)


def test_manifest_validates_files_and_required_splits(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    for path in manifest.records[0].input_paths.values():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"x")
    target = tmp_path / manifest.records[0].target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    manifest.validate(check_files_exist=True, require_splits={"train"})
    with pytest.raises(ValueError, match="no records"):
        manifest.validate(require_splits={"test"})


def test_manifest_metadata_is_strict() -> None:
    with pytest.raises(ValueError, match="exactly 3.0"):
        ManifestMetadata("2.0", ("LF",), "LF", "target")
    with pytest.raises(ValueError, match="differ"):
        ManifestMetadata("3.0", ("LF",), "LF", "LF")
