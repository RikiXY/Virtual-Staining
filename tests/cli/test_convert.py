from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_staining.applications import convert as convert_app
from virtual_staining.cli import convert as convert_cli


class _Reader:
    def __init__(self, path: Path) -> None:
        self.size = (20, 10)

    def close(self) -> None:
        pass


def test_convert_images_uses_lossless_pyramidal_tiff_and_validates_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")
    output_dir = tmp_path / "converted"
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> None:
        calls.append(command)
        Path(command[3]).write_bytes(b"converted")

    monkeypatch.setattr(convert_app.subprocess, "run", run)
    monkeypatch.setattr(
        convert_app, "PillowRegionImageReader", lambda path: SimpleNamespace(size=(20, 10))
    )
    monkeypatch.setattr(convert_app, "OpenSlideRegionImageReader", _Reader)

    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.convert"):
        result = convert_app.convert_images((source,), output_dir)

    destination = output_dir / source.name
    assert result == (destination,)
    assert destination.read_bytes() == b"converted"
    assert calls[0][:3] == ["vips", "tiffsave", str(source.resolve())]
    assert calls[0][4:] == [
        "--tile",
        "--pyramid",
        "--bigtiff",
        "--compression=lzw",
        "--tile-width=256",
        "--tile-height=256",
    ]
    assert caplog.messages == [
        f"[1/1] Converting {source.resolve()} -> {destination}",
        f"[1/1] Converted {destination}",
    ]


@pytest.mark.parametrize("failure", ["existing", "duplicate"])
def test_convert_images_refuses_unsafe_destinations(tmp_path: Path, failure: str) -> None:
    first = tmp_path / "a" / "image.tif"
    second = tmp_path / "b" / "image.tif"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    output = tmp_path / "output"
    inputs = (first, second)
    error = ValueError
    if failure == "existing":
        output.mkdir()
        (output / "image.tif").write_bytes(b"existing")
        inputs = (first,)
        error = FileExistsError

    with pytest.raises(error):
        convert_app.convert_images(inputs, output)


def test_convert_images_removes_temporary_output_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")
    output = tmp_path / "output"

    def run(command: list[str], **_: object) -> None:
        Path(command[3]).write_bytes(b"invalid")

    monkeypatch.setattr(convert_app.subprocess, "run", run)
    monkeypatch.setattr(
        convert_app, "PillowRegionImageReader", lambda path: SimpleNamespace(size=(20, 10))
    )
    monkeypatch.setattr(
        convert_app,
        "OpenSlideRegionImageReader",
        lambda path: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    with pytest.raises(ValueError, match="unsupported"):
        convert_app.convert_images((source,), output)

    assert list(output.iterdir()) == []


def test_convert_cli_resolves_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "output"
    captured: list[tuple[tuple[Path, ...], Path]] = []
    levels: list[str] = []
    monkeypatch.setattr(
        convert_cli,
        "convert_images",
        lambda inputs, output_dir: captured.append((inputs, output_dir)) or (),
    )
    monkeypatch.setattr(convert_cli, "configure_logging", levels.append)

    convert_cli.main([str(source), "--output-dir", str(output)])

    assert captured == [((source.resolve(),), output.resolve())]
    assert levels == ["INFO"]


def test_convert_images_reports_missing_vips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")
    monkeypatch.setattr(
        convert_app.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(RuntimeError, match="libvips"):
        convert_app.convert_images((source,), tmp_path / "output")


def test_convert_images_reports_vips_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")
    monkeypatch.setattr(
        convert_app.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "vips", stderr="bad TIFF")
        ),
    )

    with (
        caplog.at_level(logging.INFO, logger="virtual_staining.applications.convert"),
        pytest.raises(RuntimeError, match="bad TIFF"),
    ):
        convert_app.convert_images((source,), tmp_path / "output")

    assert any("Converting" in message for message in caplog.messages)
    assert not any("Converted" in message for message in caplog.messages)


@pytest.mark.parametrize("kind", ["missing", "non_tiff", "empty"])
def test_convert_images_rejects_invalid_inputs(tmp_path: Path, kind: str) -> None:
    source = tmp_path / ("image.png" if kind == "non_tiff" else "image.tif")
    inputs = () if kind == "empty" else (source,)
    if kind == "non_tiff":
        source.write_bytes(b"image")

    with pytest.raises((FileNotFoundError, ValueError)):
        convert_app.convert_images(inputs, tmp_path / "output")


def test_convert_images_rejects_changed_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")

    def run(command: list[str], **_: object) -> None:
        Path(command[3]).write_bytes(b"converted")

    monkeypatch.setattr(convert_app.subprocess, "run", run)
    monkeypatch.setattr(
        convert_app, "PillowRegionImageReader", lambda path: SimpleNamespace(size=(21, 10))
    )
    monkeypatch.setattr(convert_app, "OpenSlideRegionImageReader", _Reader)

    with pytest.raises(RuntimeError, match="dimensions differ"):
        convert_app.convert_images((source,), tmp_path / "output")


def test_directory_inputs_are_recursive_and_preserve_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "slides"
    (source / "nested").mkdir(parents=True)
    (source / "top.tif").write_bytes(b"top")
    (source / "nested" / "deep.TIFF").write_bytes(b"deep")
    (source / "nested" / "notes.txt").write_text("ignore", encoding="utf-8")
    output = source / "converted"
    output.mkdir()
    (output / "old.tif").write_bytes(b"ignore output subtree")

    conversions = convert_app._conversion_paths((source,), output.resolve())

    assert conversions == (
        ((source / "nested" / "deep.TIFF").resolve(), output / "nested" / "deep.TIFF"),
        ((source / "top.tif").resolve(), output / "top.tif"),
    )


def test_directory_inputs_reject_empty_selection_and_cross_root_collisions(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIFF"):
        convert_app.convert_images((empty,), tmp_path / "output")

    roots = (tmp_path / "one", tmp_path / "two")
    for root in roots:
        root.mkdir()
        (root / "same.tif").write_bytes(b"image")
    with pytest.raises(ValueError, match="duplicate destinations"):
        convert_app.convert_images(roots, tmp_path / "output")
