from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.config import (
    load_yaml_mapping,
    parse_bool_strict,
    reject_unknown_keys,
    section_with_shared_fields,
)


def test_load_yaml_mapping_valid(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("key: value\nnumber: 42\n")
    assert load_yaml_mapping(f) == {"key": "value", "number": 42}


def test_load_yaml_mapping_non_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml_mapping(f)


def test_reject_unknown_keys_no_unknown() -> None:
    reject_unknown_keys({"a": 1, "b": 2}, frozenset({"a", "b", "c"}), "test")


def test_reject_unknown_keys_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        reject_unknown_keys({"a": 1, "z": 99}, frozenset({"a", "b"}), "test")


def test_parse_bool_strict_true() -> None:
    assert parse_bool_strict(True, "flag") is True


def test_parse_bool_strict_false() -> None:
    assert parse_bool_strict(False, "flag") is False


def test_parse_bool_strict_string_raises() -> None:
    with pytest.raises(TypeError, match="YAML boolean"):
        parse_bool_strict("false", "flag")


def test_parse_bool_strict_int_raises() -> None:
    with pytest.raises(TypeError):
        parse_bool_strict(0, "flag")


def test_section_with_shared_fields_no_section_returns_full_data() -> None:
    data = {"a": 1, "b": 2}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result == {"a": 1, "b": 2}


def test_section_with_shared_fields_merges_shared() -> None:
    data = {"a": 10, "training": {"b": 20}}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result["a"] == 10
    assert result["b"] == 20


def test_section_with_shared_fields_section_overrides_shared() -> None:
    data = {"a": 10, "training": {"a": 99, "b": 20}}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result["a"] == 99
