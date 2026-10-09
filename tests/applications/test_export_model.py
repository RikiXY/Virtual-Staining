"""Portable model-bundle export over a real tracked run of the tiny external method."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tests.config_helpers import pix2pix_config_data, write_config_data
from tests.external_method.test_tiny_reconstruction import _definitions, _mapping, _paired_dataset
from tests.external_method.tiny_reconstruction import TINY_RESIDUAL, TinyReconstruction
from tests.image_helpers import write_rgb_image
from tests.manifest_helpers import make_manifest_record, manifest_metadata
from virtual_staining.applications.export_model import (
    BUNDLE_SCHEMA_VERSION,
    ExportCheckpointSelection,
    ModelBundle,
    export_model_bundle,
    verify_model_bundle,
)
from virtual_staining.applications.train import train
from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.definitions import DefinitionNotAvailableError, Definitions
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.runner import load_inference_generator, predict_batch
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.utils.hashing import sha256_file

_CPU = torch.device("cpu")
_SOURCE = "tests.external_method.tiny_reconstruction"


def _explicit(path: str | Path) -> ExportCheckpointSelection:
    return ExportCheckpointSelection("explicit", checkpoint_path=Path(path))


def _best(metric: str = "val_abs_bias") -> ExportCheckpointSelection:
    return ExportCheckpointSelection("best", metric=metric)


def _top_k(rank: int, metric: str = "val_abs_bias") -> ExportCheckpointSelection:
    return ExportCheckpointSelection("top_k", metric=metric, rank=rank)


_LATEST = ExportCheckpointSelection("latest")


def _train_run(root: Path) -> Path:
    definitions, _ = _definitions()
    _paired_dataset(root / "dataset")
    config_path = root / "external.yaml"
    config_path.write_text(yaml.safe_dump(_mapping(root)), encoding="utf-8")
    config = RunConfig.from_yaml(config_path, definitions)
    train(config, config_path)
    return RunLayout.from_project(config.project).root


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _train_run(tmp_path_factory.mktemp("export_source"))


@pytest.fixture
def run(trained_run: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "run"
    shutil.copytree(trained_run, copy)
    return copy


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    change(payload)
    torch.save(payload, path)


def _export(run: Path, output: Path, *selections: ExportCheckpointSelection) -> ModelBundle:
    return export_model_bundle(run, output, selections, _definitions()[0])


def _assert_fails_cleanly(
    run: Path,
    output: Path,
    selections: list[ExportCheckpointSelection],
    error: type[Exception],
    match: str,
    definitions: Definitions | None = None,
) -> None:
    before = _snapshot(run)
    siblings = set(output.parent.iterdir()) if output.parent.exists() else set()
    with pytest.raises(error, match=match):
        export_model_bundle(
            run, output, selections, definitions if definitions is not None else _definitions()[0]
        )
    assert _snapshot(run) == before
    after = set(output.parent.iterdir()) if output.parent.exists() else set()
    assert after == siblings  # neither a destination nor a staging directory remains


# --- successful export ---------------------------------------------------------------


def test_export_bundles_each_physical_checkpoint_once_with_every_selection(
    run: Path, tmp_path: Path
) -> None:
    best_epoch = json.loads((run / "checkpoints" / "best.json").read_text())["metrics"][
        "val_abs_bias"
    ]["best"]["epoch"]
    selections = (
        _best(),
        _top_k(1),
        _explicit(f"ep{best_epoch:03d}.pth"),
        _explicit(run / "checkpoints" / f"ep{best_epoch:03d}.pth"),
        _LATEST,
        _top_k(1, "loss_recon_val"),
        _top_k(2),
    )
    output = tmp_path / "bundles" / "tiny"

    bundle = _export(run, output, *selections)

    assert bundle.root == output.resolve()
    index = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert index == bundle.index
    assert index["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert "redistribute" in index["notice"]
    assert sorted(p.name for p in (output / "checkpoints").iterdir()) == ["ep000.pth", "ep001.pth"]
    assert [entry["path"] for entry in index["checkpoints"]] == [
        "checkpoints/ep000.pth",
        "checkpoints/ep001.pth",
    ]
    assert len(index["selections"]) == len(selections)
    best = index["selections"][0]
    assert best["policy"] == "best" and best["metric"] == "val_abs_bias" and best["rank"] == 1
    assert best["mode"] == "min" and isinstance(best["metric_value"], float)
    assert best["checkpoint"] == f"checkpoints/ep{best_epoch:03d}.pth"
    assert {record["checkpoint"] for record in index["selections"][:4]} == {best["checkpoint"]}
    latest = index["selections"][4]
    assert latest == {
        "policy": "latest",
        "metric": None,
        "rank": None,
        "metric_value": None,
        "mode": None,
        "epoch": 1,
        "checkpoint": "checkpoints/ep001.pth",
    }
    assert index["selections"][6]["rank"] == 2
    assert index["selections"][6]["checkpoint"] != best["checkpoint"]

    # Exact tracked configs and environment, with hashes; checkpoints bound to the config.
    resolved_hash = sha256_file(run / "config" / "train" / "resolved.yaml")
    assert index["config"]["resolved"] == {
        "path": "config/resolved.yaml",
        "sha256": resolved_hash,
        "role": "training_resolved",
    }
    assert index["config"]["input"]["role"] == "training_input"
    for bundled, source in (
        ("config/input.yaml", "config/train/input.yaml"),
        ("config/resolved.yaml", "config/train/resolved.yaml"),
        ("metadata/training_environment.json", "metadata/environments/train.json"),
    ):
        assert (output / bundled).read_bytes() == (run / source).read_bytes()
    for entry in index["checkpoints"]:
        assert entry["config_hash"] == resolved_hash
        assert entry["sha256"] == sha256_file(output / entry["path"])
        assert entry["sha256"] == sha256_file(run / entry["path"])
        assert entry["format_version"] == 4 and entry["image_size"] == [32, 32]
        assert entry["method"]["name"] == "tiny_reconstruction"
        assert entry["method"]["implementation"] == {"version": "1", "source": _SOURCE}
        assert entry["method"]["components"]["network"] == {
            "name": "tiny_conv",
            "version": "1",
            "source": _SOURCE,
            "options": {"width": 8},
        }
        assert set(entry["method"]) >= {"pairing", "inputs", "outputs", "prediction_directions"}
        assert "state" not in entry
    assert index["requirements"] == {
        "methods": [{"name": "tiny_reconstruction", "source": _SOURCE, "version": "1"}],
        "components": [{"name": "tiny_conv", "source": _SOURCE, "version": "1"}],
    }
    # Only relative paths, no provider code, no staging leftovers.
    assert str(run) not in json.dumps({k: v for k, v in index.items() if k != "notice"})
    assert not list(output.rglob("*.py"))
    assert [path.name for path in output.parent.iterdir()] == ["tiny"]


def test_hard_link_aliases_are_bundled_once(run: Path, tmp_path: Path) -> None:
    os.link(run / "checkpoints" / "ep001.pth", run / "checkpoints" / "ep001_alias.pth")

    bundle = _export(run, tmp_path / "bundle", _explicit("ep001_alias.pth"), _LATEST)

    assert [p.name for p in (tmp_path / "bundle" / "checkpoints").iterdir()] == ["ep001_alias.pth"]
    assert {r["checkpoint"] for r in bundle.index["selections"]} == {"checkpoints/ep001_alias.pth"}


def test_moved_bundle_reconstructs_without_the_run_or_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = _train_run(source)
    exported = _export(run, tmp_path / "exported", _best())
    checkpoint = exported.index["selections"][0]["checkpoint"]

    moved = tmp_path / "elsewhere" / "bundle"
    moved.parent.mkdir()
    shutil.move(tmp_path / "exported", moved)
    (source / "dataset").rename(tmp_path / "dataset_gone")
    run.rename(tmp_path / "run_gone")

    definitions, tiny = _definitions()
    assert verify_model_bundle(moved, definitions).root == moved.resolve()
    config = RunConfig.from_yaml(moved / "config" / "resolved.yaml", definitions)
    model, path = load_inference_generator(config, RunLayout(moved), _CPU, moved / checkpoint)
    output = predict_batch(model, {"source": torch.rand(1, 3, 32, 32) * 2 - 1}, _CPU, ("target",))[
        "target"
    ]

    assert path == moved / checkpoint
    assert tiny.built == {"network": 1}
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()

    # Without the provider's explicit definitions nothing is reconstructed.
    with pytest.raises(DefinitionNotAvailableError, match="tiny_reconstruction"):
        verify_model_bundle(moved)
    with pytest.raises(DefinitionNotAvailableError, match="tiny_reconstruction"):
        RunConfig.from_yaml(moved / "config" / "resolved.yaml")


# --- selection failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("selections", "error", "match"),
    [
        ([], ValueError, "At least one checkpoint selection"),
        ([_best("val_nope")], ValueError, "no records for metric 'val_nope'"),
        ([_top_k(9)], ValueError, "no rank 9 for metric 'val_abs_bias'"),
        ([_explicit("ep099.pth")], FileNotFoundError, "does not exist"),
        ([_explicit("../config/train/resolved.yaml")], ValueError, "outside the run"),
        ([_explicit("/etc/hostname")], ValueError, "outside the run"),
        ([_explicit("nested/ep001.pth")], ValueError, "directly in"),
        ([_explicit(".")], ValueError, "directly in"),
    ],
)
def test_invalid_selections_fail_before_writing(
    run: Path,
    tmp_path: Path,
    selections: list[ExportCheckpointSelection],
    error: type[Exception],
    match: str,
) -> None:
    _assert_fails_cleanly(run, tmp_path / "out" / "bundle", selections, error, match)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"policy": "top_k", "metric": "val_abs_bias", "rank": 0}, "rank must be an integer"),
        ({"policy": "top_k", "metric": "val_abs_bias"}, "requires rank"),
        ({"policy": "best"}, "requires metric"),
        ({"policy": "best", "metric": "val_abs_bias", "rank": 2}, "does not take rank"),
        ({"policy": "latest", "metric": "val_abs_bias"}, "does not take metric"),
        ({"policy": "explicit"}, "requires checkpoint_path"),
        ({"policy": "nearest"}, "Unsupported export checkpoint policy"),
    ],
)
def test_selection_requests_are_typed_and_strict(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExportCheckpointSelection(**kwargs)


def test_ranked_selection_requires_the_catalog(run: Path, tmp_path: Path) -> None:
    (run / "checkpoints" / "best.json").unlink()
    _assert_fails_cleanly(
        run, tmp_path / "out" / "bundle", [_best()], FileNotFoundError, "best.json"
    )


@pytest.mark.parametrize("target", ["inside", "outside"])
def test_symlinked_checkpoints_are_rejected(run: Path, tmp_path: Path, target: str) -> None:
    real = run / "checkpoints" / "ep001.pth"
    if target == "outside":
        real = Path(shutil.copy(real, tmp_path / "outside.pth"))
    (run / "checkpoints" / "link.pth").symlink_to(real)
    _assert_fails_cleanly(
        run, tmp_path / "out" / "bundle", [_explicit("link.pth")], ValueError, "symlink"
    )


# --- checkpoint validation failures ----------------------------------------------------


def _set_config_hash(value: str | None) -> Callable[[dict[str, Any]], None]:
    return lambda payload: payload.__setitem__("config_hash", value)


def _set_width(payload: dict[str, Any]) -> None:
    payload["method"]["components"]["network"]["options"]["width"] = 4


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_set_config_hash(None), "has no config_hash"),
        (_set_config_hash("sha256:" + "0" * 64), "does not match the tracked training"),
        (_set_width, "network.options.width is 4"),
        (lambda payload: payload.__setitem__("format_version", 3), "unsupported format version"),
        (lambda payload: payload.pop("format_version"), "unversioned"),
    ],
)
def test_incompatible_checkpoints_are_rejected(
    run: Path, tmp_path: Path, change: Callable[[dict[str, Any]], None], match: str
) -> None:
    _rewrite(run / "checkpoints" / "ep001.pth", change)
    _assert_fails_cleanly(
        run, tmp_path / "out" / "bundle", [_LATEST], CheckpointCompatibilityError, match
    )


def test_corrupt_checkpoint_is_rejected(run: Path, tmp_path: Path) -> None:
    (run / "checkpoints" / "ep001.pth").write_bytes(b"not a checkpoint")
    _assert_fails_cleanly(
        run,
        tmp_path / "out" / "bundle",
        [_explicit("ep001.pth")],
        CheckpointCompatibilityError,
        "cannot be read",
    )


@pytest.mark.parametrize(
    ("definitions", "match"),
    [
        (builtin_definitions(), "method.name='tiny_reconstruction' is not a registered"),
        (
            builtin_definitions().extend(
                methods=[TinyReconstruction()], components=[TINY_RESIDUAL]
            ),
            "'tiny_conv' is not a registered component",
        ),
    ],
)
def test_unregistered_external_definitions_are_rejected(
    run: Path, tmp_path: Path, definitions: Definitions, match: str
) -> None:
    _assert_fails_cleanly(
        run, tmp_path / "out" / "bundle", [_LATEST], DefinitionNotAvailableError, match, definitions
    )


def test_missing_tracked_config_is_rejected(run: Path, tmp_path: Path) -> None:
    (run / "config" / "train" / "input.yaml").unlink()
    _assert_fails_cleanly(
        run, tmp_path / "out" / "bundle", [_LATEST], FileNotFoundError, "input.yaml"
    )


# --- destination safety ----------------------------------------------------------------


def test_existing_destination_is_never_touched(run: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "bundle"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    _assert_fails_cleanly(run, output, [_LATEST], FileExistsError, "already exists")
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    _assert_fails_cleanly(run, run, [_LATEST], FileExistsError, "already exists")


def test_destination_inside_or_aliasing_the_run_is_rejected(run: Path, tmp_path: Path) -> None:
    _assert_fails_cleanly(run, run / "bundle", [_LATEST], ValueError, "inside the source run")
    (tmp_path / "alias").symlink_to(run)
    _assert_fails_cleanly(
        run, tmp_path / "alias" / "bundle", [_LATEST], ValueError, "inside the source run"
    )


def test_failed_verification_removes_only_staging(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import virtual_staining.applications.export_model as export_model

    def fail(root: Path, definitions: Definitions | None = None) -> None:
        assert root.name.startswith(".bundle.") and (root / "bundle.json").is_file()
        raise ValueError("verification failed")

    monkeypatch.setattr(export_model, "verify_model_bundle", fail)
    (tmp_path / "out").mkdir()
    _assert_fails_cleanly(run, tmp_path / "out" / "bundle", [_LATEST], ValueError, "verification")


def test_destination_created_during_publication_is_never_replaced(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import virtual_staining.applications.export_model as export_model

    output = tmp_path / "out" / "bundle"

    def verify_then_race(root: Path, definitions: Definitions | None = None) -> ModelBundle:
        bundle = verify_model_bundle(root, definitions)
        output.mkdir()  # another process wins the name after the up-front check
        (output / "sentinel.txt").write_text("theirs", encoding="utf-8")
        return bundle

    monkeypatch.setattr(export_model, "verify_model_bundle", verify_then_race)
    before = _snapshot(run)
    with pytest.raises(FileExistsError):
        _export(run, output, _LATEST)
    assert _snapshot(run) == before
    assert list(output.iterdir()) == [output / "sentinel.txt"]
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "theirs"
    assert list(output.parent.iterdir()) == [output]  # no staging directory remains


@pytest.mark.parametrize("kind", ["empty directory", "file", "symlink"])
def test_publish_never_replaces_an_existing_destination(tmp_path: Path, kind: str) -> None:
    from virtual_staining.applications.export_model import _publish_directory_no_replace

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "bundle.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "bundle"
    if kind == "empty directory":
        destination.mkdir()  # plain POSIX rename() would silently replace this
    elif kind == "file":
        destination.write_text("theirs", encoding="utf-8")
    else:
        destination.symlink_to(tmp_path / "elsewhere")

    with pytest.raises(FileExistsError):
        _publish_directory_no_replace(staging, destination)
    assert (staging / "bundle.json").is_file()
    if kind == "empty directory":
        assert destination.is_dir() and not any(destination.iterdir())
    elif kind == "file":
        assert destination.read_text(encoding="utf-8") == "theirs"
    else:
        assert destination.readlink() == tmp_path / "elsewhere"

    _publish_directory_no_replace(staging, tmp_path / "fresh")
    assert (tmp_path / "fresh" / "bundle.json").is_file() and not staging.exists()


# --- bundle verification -----------------------------------------------------------------


def _edit_index(bundle: Path, change: Callable[[dict[str, Any]], None]) -> None:
    index = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    change(index)
    (bundle / "bundle.json").write_text(json.dumps(index), encoding="utf-8")


def _checkpoint_path(value: str) -> Callable[[dict[str, Any]], None]:
    return lambda index: index["checkpoints"][0].__setitem__("path", value)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_checkpoint_path("checkpoints/../../run/checkpoints/ep001.pth"), "escapes"),
        (_checkpoint_path("/tmp/ep001.pth"), "must be under checkpoints/"),
        (lambda index: index["config"]["input"].__setitem__("path", "/etc/hosts"), "escapes"),
        (lambda index: index.__setitem__("schema_version", 2), "schema_version 2"),
        (lambda index: index.__setitem__("extra", 1), "do not match schema"),
        (lambda index: index["selections"][0].__setitem__("checkpoint", "x"), "unlisted"),
        (lambda index: index["selections"][0].__setitem__("rank", None), "rank must be set"),
        (lambda index: index["checkpoints"][0].__setitem__("epoch", 7), "does not match"),
        (lambda index: index["requirements"]["methods"].clear(), "requirements"),
    ],
)
def test_verification_rejects_index_tampering(
    run: Path, tmp_path: Path, change: Callable[[dict[str, Any]], None], match: str
) -> None:
    bundle = tmp_path / "bundle"
    _export(run, bundle, _best())
    _edit_index(bundle, change)

    with pytest.raises(ValueError, match=match):
        verify_model_bundle(bundle, _definitions()[0])


@pytest.mark.parametrize(
    "relative", ["checkpoints/ep001.pth", "config/resolved.yaml", "config/input.yaml"]
)
def test_verification_detects_file_tampering(run: Path, tmp_path: Path, relative: str) -> None:
    bundle = tmp_path / "bundle"
    _export(run, bundle, _LATEST)
    with (bundle / relative).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="does not match its recorded SHA-256"):
        verify_model_bundle(bundle, _definitions()[0])


def test_verification_rejects_non_strict_json(run: Path, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _export(run, bundle, _LATEST)
    text = (bundle / "bundle.json").read_text(encoding="utf-8")
    (bundle / "bundle.json").write_text(text.replace("null", "NaN", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="not strict JSON"):
        verify_model_bundle(bundle, _definitions()[0])


def _two_output_pix2pix_run(root: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Train a tiny N=1/M=2 Pix2Pix run; return its root and one prediction per output."""
    records = []
    for index, split in enumerate(("train", "train", "val", "test")):
        sample_id = f"{index * 256:05}_00000"
        paths = {
            name: Path(f"splits/{split}/{sample_id}__{name}.png") for name in ("LF", "PAS", "HE")
        }
        for offset, path in enumerate(paths.values()):
            write_rgb_image(root / "dataset" / path, size=(32, 32), color=(50 * offset, 40, index))
        records.append(
            make_manifest_record(
                sample_id,
                split,
                set_id=f"S{index}",
                input_paths={"LF": paths["LF"]},
                target_paths={"PAS": paths["PAS"], "HE": paths["HE"]},
            )
        )
    metadata = manifest_metadata(("LF",), ("PAS", "HE"))
    layout = DatasetLayout(root / "dataset")
    DatasetManifest(tuple(records), root / "dataset", metadata).to_csv(layout.manifest_path)
    layout.manifest_metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    data = pix2pix_config_data(root, inputs=("LF",), outputs=("PAS", "HE"))
    data["dataset_root"] = str(root / "dataset")
    data["data"] = {"pairing": "paired", "group_validation": "unavailable"}
    data["training"]["epochs"] = 1
    data["inference"] = {"checkpoint_policy": "best", "checkpoint_metric": "val_ssim__HE"}
    config_path = write_config_data(root / "run.yaml", data)
    config = RunConfig.from_yaml(config_path)
    train(config, config_path)
    model, _ = load_inference_generator(config, RunLayout.from_project(config.project), _CPU)
    torch.manual_seed(3)
    probe = {"LF": torch.rand(1, 3, 32, 32) * 2 - 1}
    return RunLayout.from_project(config.project).root, {
        **predict_batch(model, probe, _CPU, ("PAS", "HE")),
        "probe": probe["LF"],
    }


def test_two_output_pix2pix_bundle_reconstructs_every_named_output_after_a_move(
    tmp_path: Path,
) -> None:
    run, expected = _two_output_pix2pix_run(tmp_path / "source")
    exported = export_model_bundle(
        run, tmp_path / "exported", [_best("val_ssim__HE")], definitions=builtin_definitions()
    )
    checkpoint = exported.index["selections"][0]["checkpoint"]
    moved = tmp_path / "elsewhere" / "bundle"
    moved.parent.mkdir()
    shutil.move(tmp_path / "exported", moved)
    shutil.rmtree(tmp_path / "source")

    assert verify_model_bundle(moved, builtin_definitions()).root == moved.resolve()
    config = RunConfig.from_yaml(moved / "config" / "resolved.yaml")
    assert config.model is not None
    assert config.model.outputs == ("PAS", "HE")
    payload = torch.load(moved / checkpoint, map_location="cpu", weights_only=True)
    assert (payload["format_version"], payload["method"]["outputs"]) == (4, ["PAS", "HE"])
    model, _ = load_inference_generator(config, RunLayout(moved), _CPU, moved / checkpoint)
    restored = predict_batch(model, {"LF": expected["probe"]}, _CPU, ("PAS", "HE"))

    assert list(restored) == ["PAS", "HE"]
    for name in ("PAS", "HE"):
        assert torch.equal(restored[name], expected[name])
