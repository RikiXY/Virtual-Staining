from __future__ import annotations

from pathlib import Path

from tests.manifest_helpers import make_manifest_record
from virtual_staining.inference.outputs import generated_path_for_record


def test_generated_path_for_record_uses_sample_id(tmp_path: Path) -> None:
    record = make_manifest_record("00512_09216", "test")

    result = generated_path_for_record(record, tmp_path)

    assert result == tmp_path / "00512_09216_target_generated.tif"
