from __future__ import annotations

from pathlib import Path

import pytest

from tests.config_helpers import pix2pix_config_data, prepare_config_data, write_config_data
from virtual_staining import cli
from virtual_staining.applications.config_authoring import inspect_run_yaml, preflight


def _config(tmp_path: Path) -> Path:
    data = pix2pix_config_data(tmp_path / "unavailable")
    data["inference"] = {"checkpoint_path": str(tmp_path / "unavailable" / "missing.pth")}
    data["evaluation"] = {"output_dir": str(tmp_path / "unavailable" / "evaluation")}
    return write_config_data(tmp_path / "run.yaml", data)


def test_resolve_prints_the_resolved_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path)

    cli.main(["config", "resolve", "--config", str(path)])

    assert capsys.readouterr().out == inspect_run_yaml(path).resolved_yaml
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.yaml"]


def test_resolve_output_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path)
    output = tmp_path / "resolved.yaml"

    cli.main(["config", "resolve", "--config", str(path), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == inspect_run_yaml(path).resolved_yaml
    assert capsys.readouterr().out == ""

    output.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", "resolve", "--config", str(path), "--output", str(output)])
    assert exc.value.code == 1
    assert output.read_text(encoding="utf-8") == "keep me\n"
    assert "refusing to overwrite" in capsys.readouterr().err
    assert sorted(item.name for item in tmp_path.iterdir()) == ["resolved.yaml", "run.yaml"]


def test_invalid_config_fails_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = pix2pix_config_data(tmp_path)
    data["surprise"] = True
    path = write_config_data(tmp_path / "run.yaml", data)
    output = tmp_path / "resolved.yaml"

    for argv in (["resolve", "--output", str(output)], ["check"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["config", *argv[:1], "--config", str(path), *argv[1:]])
        assert exc.value.code == 1
        assert "Unknown key(s) in top level: surprise" in capsys.readouterr().err
    assert not output.exists()


def test_config_check_passes_with_no_assets_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path)

    cli.main(["config", "check", "--config", str(path), "--stages", "train", "infer", "evaluate"])

    out = capsys.readouterr().out
    assert "depth: config" in out and "valid: true" in out
    assert "content_verified: false" in out
    inspection = inspect_run_yaml(path, stages=("train", "infer", "evaluate"))
    assert f"config_sha256: {inspection.resolved_sha256}" in out
    assert not (tmp_path / "unavailable").exists()


def test_asset_check_matches_the_api_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path)
    stages = ["infer", "train"]

    with pytest.raises(SystemExit) as exc:
        cli.main(["config", "check", "--config", str(path), "--stages", *stages, "--assets"])

    assert exc.value.code == 1
    out = capsys.readouterr().out
    report = preflight(inspect_run_yaml(path).config, stages, depth="assets")
    for check in report.checks:
        assert f"[{check.status}] {check.check_id}: {check.message}" in out
    assert out.index("infer.config") < out.index("train.config")
    assert "stages: infer train" in out and "valid: false" in out
    assert not (tmp_path / "unavailable").exists()


def test_config_requires_an_action_and_known_stages(tmp_path: Path) -> None:
    for argv in (
        ["config"],
        ["config", "check", "--config", str(_config(tmp_path)), "--stages", "x"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code != 0


@pytest.mark.parametrize(
    "stages",
    [
        ("prepare",),
        ("train",),
        ("infer",),
        ("evaluate",),
        ("train", "infer"),
        ("infer", "evaluate"),
    ],
)
def test_scoped_check_resolve_and_export_share_identity(
    tmp_path: Path, stages: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    raw = prepare_config_data(tmp_path)
    if stages != ("prepare",):
        raw = pix2pix_config_data(tmp_path, outputs=("HE", "IHC"))
        if "train" not in stages:
            raw.pop("training")
            raw["model"].pop("discriminator")
        if "infer" in stages:
            raw["inference"] = {"checkpoint_policy": "latest"}
        if stages == ("evaluate",):
            raw["model"].pop("generator")
    path = write_config_data(tmp_path / "input.yaml", raw)
    args = ["--config", str(path), "--stages", *stages]
    inspection = inspect_run_yaml(path, stages=stages)
    cli.main(["config", "check", *args])
    out = capsys.readouterr().out
    assert f"config_sha256: {inspection.resolved_sha256}" in out
    assert "scope: selected execution:" in out and "valid: true" in out
    cli.main(["config", "resolve", *args])
    captured = capsys.readouterr()
    assert captured.out == inspection.resolved_yaml
    assert "scope: selected execution:" in captured.err
    output = tmp_path / "resolved.yaml"
    cli.main(["config", "resolve", *args, "--output", str(output)])
    assert output.read_bytes() == inspection.resolved_yaml.encode()
    with pytest.raises(SystemExit):
        cli.main(["config", "resolve", *args, "--output", str(output)])
    assert output.read_bytes() == inspection.resolved_yaml.encode()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.yaml", "resolved.yaml"]


@pytest.mark.parametrize("action", ["check", "resolve"])
@pytest.mark.parametrize("later", ["train", "infer", "evaluate", "unpaired"])
def test_scoped_errors_precede_assets_and_export(
    tmp_path: Path,
    action: str,
    later: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = prepare_config_data(tmp_path)
    stages = ["prepare"]
    if later == "unpaired":
        raw["data"] = {"pairing": "unpaired"}
        message = "uses domains"
    else:
        raw.update(
            results_path=str(tmp_path / "results"),
            run_name="run",
            model={"inputs": ["LF"], "outputs": ["HE"]},
        )
        stages.append(later)
        if later == "evaluate":
            raw["evaluation"] = {"metrics": [{"name": "invalid_metric"}]}
        message = {"train": "training", "infer": "inference", "evaluate": "invalid_metric"}[later]
    path = write_config_data(tmp_path / "input.yaml", raw)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid configuration reached preflight")

    monkeypatch.setattr("virtual_staining.applications.config_authoring.preflight", forbidden)
    extras = ["--assets"] if action == "check" else ["--output", str(tmp_path / "output.yaml")]
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", action, "--config", str(path), "--stages", *stages, *extras])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("action", ["check", "resolve"])
def test_unscoped_output_and_help_explain_limits(
    tmp_path: Path, action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config_data(tmp_path / "input.yaml", {"dataset_root": str(tmp_path / "absent")})
    extras = ["--assets"] if action == "check" else []
    cli.main(["config", action, "--config", str(path), *extras])
    captured = capsys.readouterr()
    assert "unscoped inspection; no selected execution validated" in captured.out + captured.err
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", action, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--stages" in help_text and "Omission inspects" in help_text
