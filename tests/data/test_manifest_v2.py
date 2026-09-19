from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.manifest import (
    DatasetManifest,
    ManifestMetadata,
    load_manifest_or_raise,
)


def test_v2_columns_fail_under_v3_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n"
        "P1__x00000001_y00000002,train,splits/train/source.png,splits/train/target.png,AF,H&E,1,2,16,16\n",
        encoding="utf-8",
    )
    metadata = ManifestMetadata("3.0", ("AF",), "AF", "H&E")
    with pytest.raises(ValueError, match="exact v3 columns"):
        DatasetManifest.from_csv(path, tmp_path, metadata)


def test_metadata_is_required_for_csv_loading(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,set_id,split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ManifestMetadata is required"):
        DatasetManifest.from_csv(path, tmp_path)


def test_project_loader_adapts_legacy_single_input_manifest(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "manifest.csv").write_text(
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n"
        "sample,train,splits/train/source.tif,splits/train/target.tif,stained,label_free,"
        "0,0,256,256\n",
        encoding="utf-8",
    )
    (manifests / "manifest_metadata.json").write_text(
        '{"schema_version": "1.0", "record_count": 1}\n', encoding="utf-8"
    )
    project = ProjectConfig(tmp_path, tmp_path / "results", "legacy", (256, 256))

    manifest = load_manifest_or_raise(project)

    assert manifest.metadata.schema_version == "3.0"
    assert manifest.metadata.input_modalities == ("stained",)
    assert manifest.metadata.target_modality == "label_free"
    assert manifest.records[0].set_id == tmp_path.name
