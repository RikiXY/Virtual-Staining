from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
)

_METADATA = ManifestMetadata(
    schema_version=MANIFEST_SCHEMA_VERSION,
    input_modalities=("AF",),
    target_modalities=("HE",),
    reference_modality="AF",
)


@pytest.mark.parametrize(
    "header",
    [
        # v2
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height",
        # v3 singular target columns
        "sample_id,set_id,split,input__AF,target_path,foreground_mask_path,x,y,width,height",
    ],
)
def test_earlier_manifest_columns_fail_under_the_v4_contract(tmp_path: Path, header: str) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(f"{header}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact v4 columns"):
        DatasetManifest.from_csv(path, tmp_path, _METADATA)


def test_metadata_is_required_for_csv_loading(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,set_id,split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ManifestMetadata is required"):
        DatasetManifest.from_csv(path, tmp_path)
