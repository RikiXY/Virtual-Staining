from __future__ import annotations

import pytest

from virtual_staining.utils.dimensions import (
    parse_wh_size,
    parse_wh_size_from_aliases,
    to_torchvision_hw,
)


def test_parse_wh_size_valid_list() -> None:
    assert parse_wh_size([320, 256], (256, 256)) == (320, 256)


def test_parse_wh_size_valid_tuple() -> None:
    assert parse_wh_size((128, 64), (256, 256)) == (128, 64)


def test_parse_wh_size_none_returns_default() -> None:
    assert parse_wh_size(None, (256, 256)) == (256, 256)


def test_parse_wh_size_non_square() -> None:
    assert parse_wh_size([400, 200], (256, 256)) == (400, 200)


def test_parse_wh_size_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        parse_wh_size([256], (256, 256))


def test_parse_wh_size_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_wh_size("256x256", (256, 256))


def test_parse_wh_size_three_elements_raises() -> None:
    with pytest.raises(ValueError):
        parse_wh_size([256, 256, 3], (256, 256))


def test_parse_wh_size_from_aliases_first_wins() -> None:
    data: dict[str, object] = {"model_image_size": [128, 64], "image_size": [256, 256]}
    aliases = ("model_image_size", "image_size")
    assert parse_wh_size_from_aliases(data, aliases, (32, 32)) == (128, 64)


def test_parse_wh_size_from_aliases_falls_back_to_second() -> None:
    data: dict[str, object] = {"image_size": [512, 256]}
    aliases = ("model_image_size", "image_size")
    assert parse_wh_size_from_aliases(data, aliases, (32, 32)) == (512, 256)


def test_parse_wh_size_from_aliases_no_match_returns_default() -> None:
    aliases = ("model_image_size", "image_size")
    assert parse_wh_size_from_aliases({}, aliases, (32, 32)) == (32, 32)


def test_to_torchvision_hw_swaps_wh() -> None:
    assert to_torchvision_hw((320, 256)) == (256, 320)


def test_to_torchvision_hw_square_unchanged() -> None:
    assert to_torchvision_hw((256, 256)) == (256, 256)
