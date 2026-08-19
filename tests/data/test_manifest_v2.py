from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.manifest import DatasetManifest, ManifestRecord


def test_v2_round_trip_uses_explicit_pair_and_mask_path(tmp_path: Path) -> None:
    record = ManifestRecord(
        sample_id="P1__x00000001_y00000002",
        pair_id="P1",
        split="train",
        input_path=Path("splits/train/P1/source.png"),
        target_path=Path("splits/train/P1/target.png"),
        foreground_mask_path=Path("splits/train/P1/mask.png"),
        input_modality="AF",
        target_modality="H&E",
        x=1,
        y=2,
        width=16,
        height=16,
    )
    path = tmp_path / "manifest.csv"
    DatasetManifest((record,), tmp_path, schema_version="2.0").to_csv(path)
    loaded = DatasetManifest.from_csv(path, tmp_path)
    assert loaded.schema_version == "2.0"
    assert loaded.records == (record,)


def test_mixed_manifest_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,pair_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected columns"):
        DatasetManifest.from_csv(path, tmp_path)
