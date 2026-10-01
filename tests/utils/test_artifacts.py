from __future__ import annotations

from pathlib import Path

import pytest

from tests.image_helpers import write_rgb_image
from virtual_staining.utils.artifacts import (
    collect_generated_artifacts,
    generated_identity,
    generated_path,
)


def test_generated_path_is_keyed_by_sample_and_output(tmp_path: Path) -> None:
    assert generated_path(tmp_path, "00512_09216", "HE", ".TIF") == (
        tmp_path / "HE" / "00512_09216_generated.tif"
    )
    assert generated_path(tmp_path, "patch_001", "PAS", ".png") == (
        tmp_path / "PAS" / "patch_001_generated.png"
    )


def test_generated_path_inverts_to_its_pair(tmp_path: Path) -> None:
    for sample_id, output in (("s__x1_y2", "HE"), ("a__b", "b__c"), ("x", "PAS-2")):
        path = generated_path(tmp_path, sample_id, output, ".png")
        assert generated_identity(path) == (sample_id, output)


def test_generated_path_is_collision_free_where_a_flat_name_is_not(tmp_path: Path) -> None:
    # A flat "<sample>__<output>" name would collide for these two pairs.
    first = generated_path(tmp_path, "s__A", "B", ".png")
    second = generated_path(tmp_path, "s", "A__B", ".png")

    assert first != second
    assert generated_identity(first) != generated_identity(second)


@pytest.mark.parametrize("sample_id", ["", "a/b", "a\\b"])
def test_generated_path_rejects_non_component_sample_ids(tmp_path: Path, sample_id: str) -> None:
    with pytest.raises(ValueError, match="sample_id"):
        generated_path(tmp_path, sample_id, "HE", ".png")


def test_collect_generated_artifacts_is_recursive_sorted_and_output_specific(
    tmp_path: Path,
) -> None:
    for name in (
        "case2/HE/y_generated.png",
        "case1/HE/x_generated.png",
        "case1/PAS/x_generated.png",
        "HE/z_generated.png",
        "HE/unrelated.png",
    ):
        write_rgb_image(tmp_path / name)
    (tmp_path / "HE" / "notes_generated.txt").write_text("not an image")

    assert collect_generated_artifacts(tmp_path, "HE") == (
        tmp_path / "HE/z_generated.png",
        tmp_path / "case1/HE/x_generated.png",
        tmp_path / "case2/HE/y_generated.png",
    )
    assert collect_generated_artifacts(tmp_path, "PAS") == (tmp_path / "case1/PAS/x_generated.png",)
    assert collect_generated_artifacts(tmp_path, "missing") == ()


_UNSAFE_OUTPUT_NAMES = [".", "..", "../HE", "HE/PAS", "HE\\PAS", "1HE", "H&E", ""]
_SAFE_OUTPUT_NAMES = ["HE", "PAS", "H_E", "H-E", "HE2"]


@pytest.mark.parametrize("output_name", _UNSAFE_OUTPUT_NAMES)
def test_generated_helpers_reject_unsafe_output_names(tmp_path: Path, output_name: str) -> None:
    with pytest.raises(ValueError, match="output_name must be an identifier"):
        generated_path(tmp_path, "s1", output_name, ".png")
    with pytest.raises(ValueError, match="output_name must be an identifier"):
        collect_generated_artifacts(tmp_path, output_name)
    assert list(tmp_path.iterdir()) == []


# pathlib drops a "." component, so it can never reach the inversion as a parent name.
@pytest.mark.parametrize("parent", ["..", "1HE", "H&E"])
def test_generated_identity_rejects_unsafe_output_directories(parent: str) -> None:
    with pytest.raises(ValueError, match="output_name must be an identifier"):
        generated_identity(Path("out") / parent / "s1_generated.png")


@pytest.mark.parametrize("output_name", _SAFE_OUTPUT_NAMES)
def test_safe_output_names_stay_one_directory_below_the_output_dir(
    tmp_path: Path, output_name: str
) -> None:
    path = generated_path(tmp_path, "s1", output_name, ".png")

    assert path.parent.parent == tmp_path and path.parent.name == output_name
    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert generated_identity(path) == ("s1", output_name)
