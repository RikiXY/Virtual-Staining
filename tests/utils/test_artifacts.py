from __future__ import annotations

from virtual_staining.utils.artifacts import generated_filename


def test_generated_filename_normalizes_suffix() -> None:
    assert generated_filename("00512_09216", ".tif") == "00512_09216_target_generated.tif"
    assert generated_filename("patch_001", ".PNG") == "patch_001_target_generated.png"
