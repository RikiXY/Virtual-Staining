from __future__ import annotations

from pathlib import Path

from tests.image_helpers import write_rgb_image
from virtual_staining.utils.artifacts import collect_generated_artifacts, generated_filename


def test_generated_filename_normalizes_suffix() -> None:
    assert generated_filename("00512_09216", ".tif") == "00512_09216_target_generated.tif"
    assert generated_filename("patch_001", ".PNG") == "patch_001_target_generated.png"
    assert generated_filename("x", ".png", "B_to_A") == "x_B_to_A_generated.png"


def test_collect_generated_artifacts_is_recursive_sorted_and_direction_specific(
    tmp_path: Path,
) -> None:
    for name in (
        "case2/y_A_to_B_generated.png",
        "case1/x_A_to_B_generated.png",
        "case1/x_B_to_A_generated.png",
        "case1/x_target_generated.png",
        "case1/unrelated.png",
    ):
        write_rgb_image(tmp_path / name)
    (tmp_path / "case1" / "notes_A_to_B_generated.txt").write_text("not an image")

    assert collect_generated_artifacts(tmp_path, "A_to_B") == (
        tmp_path / "case1/x_A_to_B_generated.png",
        tmp_path / "case2/y_A_to_B_generated.png",
    )
    assert collect_generated_artifacts(tmp_path, "B_to_A") == (
        tmp_path / "case1/x_B_to_A_generated.png",
    )
    assert collect_generated_artifacts(tmp_path) == (tmp_path / "case1/x_target_generated.png",)
