from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining import cli
from virtual_staining.applications.inventory_authoring import (
    InventoryRequest,
    preview_inventory,
    render_inventory_csv,
)


def _dataset(root: Path, *names: str) -> None:
    for modality in ("LF", "AF", "HE"):
        for name in names:
            path = root / "raw" / modality / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")


def _argv(root: Path, action: str, *extra: str) -> list[str]:
    return [
        "inventory",
        action,
        "--dataset-root",
        str(root),
        "--input",
        "LF=raw/LF",
        "--input",
        "AF=raw/AF/**/*.svs",
        "--target-modality",
        "HE",
        "--target",
        "raw/HE",
        "--reference",
        "LF",
        *extra,
    ]


def test_preview_lists_every_issue_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, "c/S001.svs")
    (tmp_path / "raw/LF/S009.svs").write_bytes(b"placeholder")
    (tmp_path / "raw/HE/S010.svs").write_bytes(b"placeholder")

    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path, "preview"))

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "inputs: LF AF\nreference: LF\ntarget: HE\n" in out
    assert "key_rule: relative-path" in out and "metadata: none" in out
    assert "matched_sets: 1" in out and "set: S001 key=c/S001.svs" in out
    assert "[incomplete] key 'S009.svs'" in out and "[incomplete] key 'S010.svs'" in out
    assert "limitation: " in out and out.endswith("valid: false\n")
    assert not (tmp_path / "inputs").exists()


def test_write_matches_the_api_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, "S001.svs", "c/S002.svs")
    request = InventoryRequest(
        dataset_root=tmp_path,
        inputs=(("LF", "raw/LF"), ("AF", "raw/AF/**/*.svs")),
        target_modality="HE",
        target="raw/HE",
        reference="LF",
    )
    expected = render_inventory_csv(preview_inventory(request))

    cli.main(_argv(tmp_path, "write"))

    output = tmp_path / "inputs/slide_sets.csv"
    assert output.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out.endswith(f"wrote: {output.resolve()}\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path, "write"))
    assert exc.value.code == 1 and "refusing to overwrite" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == expected

    cli.main(_argv(tmp_path, "write", "--output", "inputs/other.csv"))
    assert (tmp_path / "inputs/other.csv").read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(
    "extra",
    [
        ["--input", "LF=raw/other"],
        ["--input", "missing-separator"],
        ["--input-mask", "XR=masks"],
        ["--key", "fuzzy"],
    ],
)
def test_malformed_requests_are_rejected_before_scanning(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path / "never-scanned", "preview", *extra))
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", [("--reference", "AF2"), ("--target-modality", "AF")])
def test_reference_and_target_names_are_checked(tmp_path: Path, flag: tuple[str, str]) -> None:
    argv = _argv(tmp_path, "preview")
    argv[argv.index(flag[0]) + 1] = flag[1]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
