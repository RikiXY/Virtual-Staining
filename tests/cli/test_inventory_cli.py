from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining import cli
from virtual_staining.applications.inventory_authoring import (
    InventoryRequest,
    preview_inventory,
    render_inventory_csv,
)


def _dataset(root: Path, *names: str, targets: tuple[str, ...] = ("HE",)) -> None:
    for modality in ("LF", "AF", *targets):
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
        "--target",
        "HE=raw/HE",
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
    assert "inputs: LF AF\nreference: LF\ntargets: HE\n" in out
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
        targets=(("HE", "raw/HE"),),
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
        ["--target", "HE=raw/other"],
        ["--target", "LF=raw/other"],
        ["--target", "raw/PAS"],
        ["--target-mask", "PAS=masks"],
        ["--target-mask", "masks/HE"],
        ["--target-modality", "HE"],
        ["--key", "fuzzy"],
    ],
)
def test_malformed_requests_are_rejected_before_scanning(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path / "never-scanned", "preview", *extra))
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", [("--reference", "AF2"), ("--target", "AF=raw/HE")])
def test_reference_and_target_names_are_checked(tmp_path: Path, flag: tuple[str, str]) -> None:
    argv = _argv(tmp_path, "preview")
    argv[argv.index(flag[0]) + 1] = flag[1]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_repeatable_targets_and_target_masks_match_the_api(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, "S001.svs", targets=("HE", "PAS"))
    for name in ("HE", "PAS"):
        path = tmp_path / "masks" / name / "S001.svs"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mask")
    request = InventoryRequest(
        dataset_root=tmp_path,
        inputs=(("LF", "raw/LF"), ("AF", "raw/AF/**/*.svs")),
        targets=(("PAS", "raw/PAS"), ("HE", "raw/HE")),
        reference="LF",
        target_masks=(("HE", "masks/HE"), ("PAS", "masks/PAS")),
    )
    expected = render_inventory_csv(preview_inventory(request))
    argv = _argv(tmp_path, "write")
    argv[argv.index("HE=raw/HE")] = "PAS=raw/PAS"
    argv += ["--target", "HE=raw/HE", "--target-mask", "HE=masks/HE"]
    argv += ["--target-mask", "PAS=masks/PAS"]

    cli.main(argv)

    assert "targets: PAS HE\n" in capsys.readouterr().out
    written = (tmp_path / "inputs/slide_sets.csv").read_text(encoding="utf-8")
    assert written == expected
    assert "target__PAS_mask" in written.splitlines()[0]


def _unpaired_argv(root: Path, action: str, *extra: str) -> list[str]:
    return [
        "inventory",
        action,
        "--dataset-root",
        str(root),
        "--pairing",
        "unpaired",
        "--domain",
        "HE=raw/HE",
        "--domain",
        "LF=raw/LF/**/*.png",
        *extra,
    ]


def test_unpaired_cli_preview_write_and_python_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, "a.png")
    (tmp_path / "raw/LF/b.png").write_bytes(b"placeholder")
    request = InventoryRequest(
        dataset_root=tmp_path,
        pairing="unpaired",
        domains=(("HE", "raw/HE"), ("LF", "raw/LF/**/*.png")),
    )
    expected = render_inventory_csv(preview_inventory(request))
    cli.main(_unpaired_argv(tmp_path, "preview"))
    out = capsys.readouterr().out
    assert "pairing: unpaired\ndomains: HE LF" in out
    assert f"domain: HE spec={tmp_path}/raw/HE images=1" in out
    assert f"domain: LF spec={tmp_path}/raw/LF/**/*.png images=2" in out
    assert "source: LF raw/LF/b.png" in out
    assert out.endswith("valid: true\n")
    assert "independence" in out and "image-content integrity" in out
    assert "matched_sets" not in out
    assert not (tmp_path / "inputs").exists()
    cli.main(_unpaired_argv(tmp_path, "write"))
    output = tmp_path / "inputs/paths.csv"
    assert output.read_text() == expected
    with pytest.raises(SystemExit) as exc:
        cli.main(_unpaired_argv(tmp_path, "write"))
    assert exc.value.code == 1
    assert output.read_text() == expected
    cli.main(_unpaired_argv(tmp_path, "write", "--output", "inputs/custom.csv"))
    assert (tmp_path / "inputs/custom.csv").read_text() == expected


@pytest.mark.parametrize(
    "extra",
    [
        ["--input", "LF=raw/LF"],
        ["--target", "HE=raw/HE"],
        ["--reference", "LF"],
        ["--input-mask", "LF=masks"],
        ["--target-mask", "HE=masks"],
        ["--key", "relative-path"],
        ["--key", "relative-stem"],
        ["--unknown"],
        ["--domain", "third=raw/third"],
        ["--domain", "broken"],
        ["--pairing", "other"],
    ],
)
def test_unpaired_cli_rejects_inapplicable_or_malformed_options(tmp_path: Path, extra) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_unpaired_argv(tmp_path / "never-scanned", "preview", *extra))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "domains",
    [
        [],
        ["--domain", "LF=raw/LF"],
        ["--domain", "LF=raw/LF", "--domain", "LF=raw/HE"],
        ["--domain", "LF=raw/LF", "--domain", "bad name=raw/HE"],
    ],
)
def test_unpaired_cli_requires_exactly_two_named_domains(tmp_path: Path, domains) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "inventory",
                "preview",
                "--dataset-root",
                str(tmp_path),
                "--pairing",
                "unpaired",
                *domains,
            ]
        )
    assert exc.value.code == 2


def test_paired_cli_rejects_domains(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path, "preview", "--domain", "LF=raw/LF"))
    assert exc.value.code == 2
