from __future__ import annotations

import errno
import logging
from pathlib import Path

import pytest
from PIL import Image

from virtual_staining.applications import convert as convert_app
from virtual_staining.cli import convert as convert_cli


class _Reader:
    def __init__(self, path: Path) -> None:
        self.size = (20, 10)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "suffix", [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".PNG", ".JpEg", ".TiF"]
)
def test_convert_images_delegates_pyramidal_writing_to_image_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    suffix: str,
) -> None:
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"input")
    output_dir = tmp_path / "converted"
    calls: list[tuple[Path, Path]] = []

    def convert(source_path: Path, output_path: Path) -> None:
        assert output_path.is_file()
        assert output_path.stat().st_size == 0
        calls.append((source_path, output_path))
        output_path.write_bytes(b"converted")

    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", convert)

    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.convert"):
        result = convert_app.convert_images((source,), output_dir)

    destination = output_dir / (
        "source.tif" if suffix.lower() in {".png", ".jpg", ".jpeg"} else source.name
    )
    assert result == (destination,)
    assert destination.read_bytes() == b"converted"
    assert list(output_dir.iterdir()) == [destination]
    assert len(calls) == 1
    assert calls[0][0] == source.resolve()
    assert calls[0][1].parent == output_dir
    assert calls[0][1].name.startswith(".source.")
    assert calls[0][1].name.endswith(".tmp.tif")
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


def test_convert_images_rejects_jpeg_extension_collisions(tmp_path: Path) -> None:
    source = tmp_path / "slides"
    source.mkdir()
    (source / "sample.jpg").write_bytes(b"jpg")
    (source / "sample.jpeg").write_bytes(b"jpeg")

    with pytest.raises(ValueError, match="duplicate destinations"):
        convert_app.convert_images((source,), tmp_path / "output")


def test_convert_images_removes_temporary_output_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")
    output = tmp_path / "output"

    def fail(source_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"invalid")
        raise ValueError("unsupported")

    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", fail)

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


def test_convert_images_propagates_writer_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"input")

    def fail(source_path: Path, output_path: Path) -> None:
        raise RuntimeError("Could not convert source: bad TIFF")

    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", fail)

    with (
        caplog.at_level(logging.INFO, logger="virtual_staining.applications.convert"),
        pytest.raises(RuntimeError, match="bad TIFF"),
    ):
        convert_app.convert_images((source,), tmp_path / "output")

    assert any("Converting" in message for message in caplog.messages)
    assert not any("Converted" in message for message in caplog.messages)


@pytest.mark.parametrize("kind", ["missing", "non_tiff", "empty"])
def test_convert_images_rejects_invalid_inputs(tmp_path: Path, kind: str) -> None:
    source = tmp_path / ("image.bmp" if kind == "non_tiff" else "image.tif")
    inputs = () if kind == "empty" else (source,)
    if kind == "non_tiff":
        source.write_bytes(b"image")

    with pytest.raises((FileNotFoundError, ValueError)):
        convert_app.convert_images(inputs, tmp_path / "output")


def test_directory_inputs_are_recursive_and_preserve_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "slides"
    (source / "nested").mkdir(parents=True)
    (source / "top.tif").write_bytes(b"top")
    (source / "nested" / "deep.TIFF").write_bytes(b"deep")
    (source / "nested" / "photo.JPEG").write_bytes(b"photo")
    (source / "nested" / "picture.PnG").write_bytes(b"png")
    (source / "nested" / "notes.txt").write_text("ignore", encoding="utf-8")
    output = source / "converted"
    output.mkdir()
    (output / "old.tif").write_bytes(b"ignore output subtree")

    conversions = convert_app._conversion_paths((source,), output.resolve())

    assert conversions == (
        ((source / "nested" / "deep.TIFF").resolve(), output / "nested" / "deep.TIFF"),
        ((source / "nested" / "photo.JPEG").resolve(), output / "nested" / "photo.tif"),
        ((source / "nested" / "picture.PnG").resolve(), output / "nested" / "picture.tif"),
        ((source / "top.tif").resolve(), output / "top.tif"),
    )


def test_directory_inputs_reject_empty_selection_and_cross_root_collisions(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIFF, PNG or JPEG"):
        convert_app.convert_images((empty,), tmp_path / "output")

    roots = (tmp_path / "one", tmp_path / "two")
    for root in roots:
        root.mkdir()
        (root / "same.tif").write_bytes(b"image")
    with pytest.raises(ValueError, match="duplicate destinations"):
        convert_app.convert_images(roots, tmp_path / "output")


@pytest.mark.parametrize("suffixes", [(".png", ".jpg"), (".png", ".tif"), (".jpg", ".jpeg")])
def test_mapped_collisions_are_preflighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffixes: tuple[str, str]
) -> None:
    first = tmp_path / "first.tif"
    first.write_bytes(b"first")
    sources = tuple(tmp_path / f"same{suffix}" for suffix in suffixes)
    for source in sources:
        source.write_bytes(b"source")
    calls = []
    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="duplicate destinations") as error:
        convert_app.convert_images((first, *sources), tmp_path / "output")
    assert all(str(source) in str(error.value) for source in sources)
    assert str(tmp_path / "output" / "same.tif") in str(error.value)
    assert not calls
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("selection", ["repeated", "file_directory", "directories", "hardlink"])
def test_overlapping_sources_are_preflighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    root = tmp_path / "input"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "image.png"
    source.write_bytes(b"source")
    inputs = (source, source)
    if selection == "file_directory":
        inputs = (source, root)
    elif selection == "directories":
        inputs = (root, nested)
    elif selection == "hardlink":
        alias = root / "alias.png"
        alias.hardlink_to(source)
        inputs = (source, alias)
    calls = []
    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        convert_app.convert_images(inputs, tmp_path / "output")
    assert not calls


@pytest.mark.parametrize(
    "kind", ["file", "directory", "dangling_symlink", "source", "parent_alias"]
)
def test_destination_aliases_and_overlap_are_preflighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "first.png").write_bytes(b"source")
    nested = source / "nested"
    nested.mkdir()
    last = nested / "last.png"
    last.write_bytes(b"last")
    output = tmp_path / "output"
    (output / "nested").mkdir(parents=True)
    destination = output / "nested" / "last.tif"
    if kind == "file":
        destination.write_bytes(b"existing")
    elif kind == "directory":
        destination.mkdir()
    elif kind == "dangling_symlink":
        destination.symlink_to(tmp_path / "absent")
    elif kind == "source":
        output = source
    else:
        (source / "alias").mkdir()
        (source / "alias" / "last.jpg").write_bytes(b"other")
        (output / "alias").symlink_to(output / "nested", target_is_directory=True)
    calls = []
    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", lambda *args: calls.append(args))
    with pytest.raises((ValueError, FileExistsError)):
        convert_app.convert_images((source,), output)
    assert not calls
    assert last.read_bytes() == b"last"


@pytest.mark.parametrize("suffix", [".tif", ".png", ".jpg", ".jpeg"])
def test_concurrent_destination_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, suffix: str
) -> None:
    source = tmp_path / f"image{suffix}"
    Image.new("RGB", (32, 16), (10, 20, 30)).save(source)
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "unrelated"
    unrelated.write_bytes(b"keep")
    destination = output / "image.tif"

    real_link = convert_app.os.link

    def compete(temporary: Path, target: Path) -> None:
        target.write_bytes(b"competing writer")
        real_link(temporary, target)

    monkeypatch.setattr(convert_app.os, "link", compete)
    results = []
    with (
        caplog.at_level(logging.INFO, logger=convert_app.__name__),
        pytest.raises(FileExistsError, match="Destination already exists"),
    ):
        results.append(convert_app.convert_images((source,), output))
    assert not results
    assert destination.read_bytes() == b"competing writer"
    assert unrelated.read_bytes() == b"keep"
    assert set(output.iterdir()) == {destination, unrelated}
    assert not any("Converted" in message for message in caplog.messages)


@pytest.mark.parametrize("error_number", [errno.EXDEV, errno.EOPNOTSUPP, errno.EACCES])
def test_publication_errors_do_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    source = tmp_path / "image.png"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    monkeypatch.setattr(
        convert_app,
        "convert_to_pyramidal_tiff",
        lambda source, temp: temp.write_bytes(b"converted"),
    )

    def fail(*args: object) -> None:
        raise OSError(error_number, "publication failed")

    monkeypatch.setattr(convert_app.os, "link", fail)
    with pytest.raises(OSError, match="requires hard links on the same filesystem") as error:
        convert_app.convert_images((source,), output)
    assert error.value.errno == error_number
    assert list(output.iterdir()) == []


def test_batch_retains_previous_success_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tuple(tmp_path / name for name in ("first.png", "second.jpg"))
    for source in sources:
        source.write_bytes(b"source")

    def convert(source: Path, temporary: Path) -> None:
        temporary.write_bytes(b"converted")
        if source == sources[1]:
            raise RuntimeError("bad source")

    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", convert)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="bad source"):
        convert_app.convert_images(sources, output)
    assert list(output.iterdir()) == [output / "first.tif"]
    assert (output / "first.tif").read_bytes() == b"converted"


def test_native_mixed_directory_and_multiple_explicit_sources(tmp_path: Path) -> None:
    root = tmp_path / "input"
    (root / "nested").mkdir(parents=True)
    sources = (root / "one.PnG", root / "nested" / "two.JpEg", root / "three.TIFF")
    for source in sources:
        Image.new("RGB", (32, 16), (20, 40, 60)).save(source)
    output = root / "output"
    output.mkdir()
    (output / "previous.png").write_bytes(b"excluded")
    results = convert_app.convert_images((root,), output)
    assert set(results) == {output / "one.tif", output / "nested/two.tif", output / "three.TIFF"}
    second_output = tmp_path / "explicit"
    assert convert_app.convert_images(sources, second_output) == (
        second_output / "one.tif",
        second_output / "two.tif",
        second_output / "three.TIFF",
    )
    assert (output / "previous.png").read_bytes() == b"excluded"


def test_temporary_creation_failure_preserves_caller_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "image.png"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    output.mkdir()
    caller_file = output / ".image.caller.tmp.tif"
    caller_file.write_bytes(b"caller owned")

    def fail(**kwargs: object) -> None:
        raise FileExistsError(str(caller_file))

    monkeypatch.setattr(convert_app.tempfile, "mkstemp", fail)
    with pytest.raises(FileExistsError):
        convert_app.convert_images((source,), output)
    assert list(output.iterdir()) == [caller_file]
    assert caller_file.read_bytes() == b"caller owned"


@pytest.mark.parametrize("existing_parent", [False, True])
def test_destination_file_directory_conflicts_are_preflighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_parent: bool
) -> None:
    root = tmp_path / "input"
    (root / "image.tif").mkdir(parents=True)
    (root / "image.png").write_bytes(b"source")
    (root / "image.tif" / "nested.jpg").write_bytes(b"source")
    output = tmp_path / "output"
    if existing_parent:
        output.mkdir()
        (output / "image.tif").write_bytes(b"existing")
        (root / "image.png").rename(root / "first.png")
    calls = []
    monkeypatch.setattr(convert_app, "convert_to_pyramidal_tiff", lambda *args: calls.append(args))
    with pytest.raises((ValueError, NotADirectoryError)):
        convert_app.convert_images((root,), output)
    assert not calls


def test_explicit_tiff_can_publish_safely_in_ancestor_directory(tmp_path: Path) -> None:
    source = tmp_path / "nested" / "source.tif"
    source.parent.mkdir()
    Image.new("RGB", (32, 16)).save(source)
    before = source.read_bytes()
    assert convert_app.convert_images((source,), tmp_path) == (tmp_path / "source.tif",)
    assert source.read_bytes() == before
