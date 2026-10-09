"""Exercise publication failures through tracked inference and its real session."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.utils import save_image

from tests.config_helpers import pix2pix_config_data, write_config_data
from tests.image_helpers import write_rgb_image
from tests.manifest_helpers import make_manifest_record, manifest_metadata, write_manifest_csv
from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import AssetRow, load_snapshot
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference import outputs
from virtual_staining.inference.runner import InferenceResult

infer_app = importlib.import_module("virtual_staining.applications.infer")


@pytest.fixture
def tracked_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunConfig, Path]:
    inputs, targets = ("AF", "LF"), ("PAS", "HE")
    data = pix2pix_config_data(tmp_path, inputs=inputs, outputs=targets)
    data["data"] = {"group_validation": "unavailable", "hash_policy": "content"}
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"publication test checkpoint")
    data["inference"] = {"checkpoint_path": str(checkpoint)}
    path = write_config_data(tmp_path / "run.yaml", data)
    config = RunConfig.from_yaml(path)
    root = config.project.dataset_root
    records = [
        make_manifest_record(
            f"{i:05}_00000",
            split,
            input_paths={name: Path(f"{i}_{name}.png") for name in inputs},
            target_paths={name: Path(f"{i}_{name}.png") for name in targets},
            set_id=f"P{i}",
            width=32,
            height=32,
        )
        for i, split in enumerate(("train", "test", "test"))
    ]
    for i, record in enumerate(records):
        for j, asset in enumerate((*record.input_paths.values(), *record.target_paths.values())):
            write_rgb_image(root / asset, size=(32, 32), color=(20 * i, 30 * j, 40))
    metadata = manifest_metadata(inputs, targets)
    write_manifest_csv(root, records, metadata=metadata)
    (root / "manifests" / "manifest_metadata.json").write_text(json.dumps(metadata.to_dict()))

    def predictor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert tuple(batch) == inputs
        return {"PAS": batch["AF"], "HE": -batch["LF"]}

    monkeypatch.setattr(infer_app, "resolve_inference_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        infer_app, "load_inference_generator", lambda *args: (predictor, checkpoint)
    )
    return config, path


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fail_index", [0, 3])
@pytest.mark.parametrize(
    "failure", ["before", "partial", "invalid_raise", "invalid_return", "truncated", "replace"]
)
def test_tracked_publication_failure_preserves_destinations_and_bookkeeping(
    tracked_run: tuple[RunConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    fail_index: int,
    failure: str,
) -> None:
    config, config_path = tracked_run
    layout = RunLayout.from_project(config.project)
    destinations = [
        layout.output_test_dir / name / f"{i:05}_00000_generated.png"
        for i in (1, 2)
        for name in config.model.outputs
    ]
    originals: dict[Path, bytes] = {}
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            write_rgb_image(destination, size=(32, 32), color=(7, 8, 9))
            originals[destination] = destination.read_bytes()
    unrelated = destinations[0].parent / ".caller.partial.png"
    unrelated.write_bytes(b"caller-owned artifact")
    source_bytes = {
        p: p.read_bytes() for p in config.project.dataset_root.rglob("*") if p.is_file()
    }
    result = InferenceResult(layout.output_test_dir)
    generated_rows: list[AssetRow] = []
    staged: list[Path] = []

    def row_spy(**kwargs: Any) -> AssetRow:
        row = AssetRow(**kwargs)
        if row.role == "generated":
            generated_rows.append(row)
        return row

    def encoder(output: torch.Tensor, path: Path) -> None:
        index = len(staged)
        destination = destinations[index]
        assert path != destination and path.parent == destination.parent
        assert path.suffix == destination.suffix and path.is_file()
        assert path.stat().st_size == 0 and path not in staged
        staged.append(path)
        if index != fail_index or failure == "replace":
            save_image(output, path)
            return
        if failure == "partial":
            path.write_bytes(b"partial PNG")
        elif failure.startswith("invalid"):
            path.write_bytes(b"not an image")
        elif failure == "truncated":
            save_image(output, path)
            path.write_bytes(path.read_bytes()[:-12])
        if failure in {"before", "partial", "invalid_raise"}:
            raise OSError("injected encoder failure")

    real_replace = outputs.os.replace

    def replace(source: Path, destination: Path) -> None:
        if destination == destinations[fail_index] and failure == "replace":
            with Image.open(source) as image:
                image.load()
            raise OSError("injected replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(infer_app, "InferenceResult", lambda **kwargs: result)
    monkeypatch.setattr(infer_app, "AssetRow", row_spy)
    monkeypatch.setattr(outputs, "save_image", encoder)
    monkeypatch.setattr(outputs.os, "replace", replace)
    with pytest.raises((OSError, SyntaxError)):
        infer_app.infer(config, config_path)

    assert len(staged) == fail_index + 1 and all(not p.exists() for p in staged)
    assert result.generated_paths == destinations[:fail_index]
    assert result.num_samples == fail_index // 2
    assert [row.locator for row in generated_rows] == [
        p.relative_to(layout.output_test_dir).as_posix() for p in destinations[:fail_index]
    ]
    for destination in destinations[:fail_index]:
        with Image.open(destination) as image:
            image.load()
        if existing:
            assert destination.read_bytes() != originals[destination]
    for destination in destinations[fail_index:]:
        if existing:
            assert destination.read_bytes() == originals[destination]
        else:
            assert not destination.exists()
    expected = set(destinations if existing else destinations[:fail_index]) | {unrelated}
    assert {p for p in layout.output_test_dir.rglob("*") if p.is_file()} == expected
    assert unrelated.read_bytes() == b"caller-owned artifact"
    assert all(p.read_bytes() == content for p, content in source_bytes.items())
    stage = json.loads(layout.stage_record("infer").read_text())
    assert stage["status"] == "failed"
    assert stage["details"]["inferred_count"] == fail_index // 2
    assert stage["produced_data"] is None
    assert not layout.produced_data("infer").metadata.exists()
    consumed = load_snapshot(layout.consumed_data("infer"))
    assert len(consumed.rows) == 4 and all(row.split == "test" for row in consumed.rows)


def test_tracked_publication_matches_direct_write_bytes_and_provenance_on_rerun(
    tracked_run: tuple[RunConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, config_path = tracked_run
    layout = RunLayout.from_project(config.project)

    def direct_write(output: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(output, path)

    with monkeypatch.context() as legacy:
        legacy.setattr(infer_app, "save_rgb", direct_write)
        baseline = infer_app.infer(config, config_path)
    expected_bytes = {p: p.read_bytes() for p in baseline.generated_paths}
    consumed = load_snapshot(layout.consumed_data("infer"))
    produced = load_snapshot(layout.produced_data("infer"))
    details = json.loads(layout.stage_record("infer").read_text())["details"]
    assert baseline.num_samples == 2 and len(baseline.generated_paths) == 4
    for _ in range(2):
        for path in baseline.generated_paths:
            write_rgb_image(path, size=(32, 32), color=(7, 8, 9))
        assert infer_app.infer(config, config_path) == baseline
        assert {p: p.read_bytes() for p in baseline.generated_paths} == expected_bytes
        for before, after in (
            (consumed, load_snapshot(layout.consumed_data("infer"))),
            (produced, load_snapshot(layout.produced_data("infer"))),
        ):
            assert after == before
            assert after.snapshot_id == before.snapshot_id
            assert after.membership_sha256 == before.membership_sha256
            assert all(row.sha256 for row in after.rows)
        assert json.loads(layout.stage_record("infer").read_text())["details"] == details
        assert {p for p in layout.output_test_dir.rglob("*") if p.is_file()} == set(expected_bytes)
