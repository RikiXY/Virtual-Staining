from __future__ import annotations

from pathlib import Path

import pytest

from tests.manifest_helpers import make_manifest_record
from virtual_staining.inference.outputs import generated_path_for_record


def test_generated_path_for_record_uses_sample_id_output_and_domain_suffix(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        "00512_09216",
        "test",
        target_paths={"HE": Path("t/he.TIF"), "PAS": Path("t/pas.png")},
    )

    assert generated_path_for_record(record, tmp_path, "HE") == (
        tmp_path / "HE" / "00512_09216_generated.tif"
    )
    assert generated_path_for_record(record, tmp_path, "PAS") == (
        tmp_path / "PAS" / "00512_09216_generated.png"
    )
    # CycleGAN B_to_A predicts an input domain: the suffix comes from that input.
    assert generated_path_for_record(record, tmp_path, "label_free").parent.name == "label_free"


def test_generated_path_for_record_rejects_unknown_domain(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no domain"):
        generated_path_for_record(make_manifest_record(), tmp_path, "PAS")
