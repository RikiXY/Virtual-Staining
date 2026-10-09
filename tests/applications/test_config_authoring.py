"""Shared run-config authoring, inspection, and read-only preflight."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.config_helpers import cyclegan_config_data, pix2pix_config_data, write_config_data
from tests.external_method.tiny_reconstruction import TINY_CONV, TINY_RESIDUAL, TinyReconstruction
from tests.manifest_helpers import make_manifest_record, manifest_metadata
from virtual_staining.applications.config_authoring import (
    PreflightReport,
    field_origins,
    inspect_run_mapping,
    inspect_run_yaml,
    preflight,
    write_config_yaml,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.experiment.snapshots import save_stage_config_snapshots
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.utils.artifacts import generated_path
from virtual_staining.utils.hashing import sha256_file

_RUNS = Path(__file__).resolve().parents[2] / "config" / "runs"


def _statuses(report: PreflightReport) -> dict[str, str]:
    return {check.check_id: check.status for check in report.checks}


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


# --- fixtures ------------------------------------------------------------------------


def _paired_config(tmp_path: Path, **sections: Any) -> dict[str, Any]:
    data = pix2pix_config_data(tmp_path, inputs=("label_free",))
    data["inference"] = {"checkpoint_policy": "latest"}
    data["evaluation"] = {}
    data.update(sections)
    return data


def _write_prepared(root: Path, *, patients: tuple[str, str, str] = ("p0", "p1", "p2")) -> None:
    """A prepared paired dataset: one record per split; the files are placeholders only."""
    records = []
    for index, split in enumerate(("train", "val", "test")):
        sample_id = f"{index * 256:05}_00000"
        record = make_manifest_record(sample_id, split, set_id=f"S{index}")
        for path in (*record.input_paths.values(), *record.target_paths.values()):
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(b"placeholder, never decoded")
        records.append(record)
    layout = DatasetLayout(root)
    DatasetManifest(tuple(records), root, manifest_metadata()).to_csv(layout.manifest_path)
    layout.manifest_metadata_path.write_text(json.dumps(manifest_metadata().to_dict()))
    with layout.slide_sets_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set_id", "specimen_id", "patient_id"])
        writer.writerows(
            [f"S{index}", f"sp{index}", patient] for index, patient in enumerate(patients)
        )


def _write_checkpoint(config: RunConfig) -> Path:
    path = RunLayout.from_project(config.project).checkpoints_dir / "ep001.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a checkpoint: deserializing this would fail")
    return path


def _write_generated(
    config: RunConfig, sample_ids: list[str], output_name: str = "stained"
) -> None:
    output = RunLayout.from_project(config.project).output_test_dir
    for sample_id in sample_ids:
        path = generated_path(output, sample_id, output_name, ".tif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")


def _write_inventory(root: Path, *, mask: str = "") -> dict[str, Any]:
    (root / "raw").mkdir(parents=True)
    for name in ("lf1.tif", "he1.tif"):
        (root / "raw" / name).write_bytes(b"slide")
    columns = (
        "set_id,input__label_free_path,input__label_free_aligned,"
        "target__stained_path,target__stained_aligned"
    )
    row = "S1,raw/lf1.tif,true,raw/he1.tif,false"
    if mask:
        columns, row = f"{columns},input__label_free_mask", f"{row},{mask}"
    (root / "inputs").mkdir()
    (root / "inputs" / "slide_sets.csv").write_text(f"{columns}\n{row}\n", encoding="utf-8")
    return {
        "inputs": {
            "inventory": "inputs/slide_sets.csv",
            "modalities": ["label_free"],
            "reference": "label_free",
            "target_modalities": ["stained"],
        },
        "split": {"unit": "set", "train": 0.8, "val": 0.1, "test": 0.1, "seed": 0},
    }


def _write_domains(root: Path) -> None:
    for domain in ("label_free", "stained"):
        for split in ("train", "val", "test"):
            path = root / "domains" / domain / split / f"{domain}_{split}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image placeholder")


@pytest.fixture
def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every executing owner fail loudly if preflight ever reaches it."""

    def forbidden(name: str) -> Any:
        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"preflight invoked {name}")

        return fail

    targets = {
        "virtual_staining.experiment.session.ExperimentSession.__init__": None,
        "virtual_staining.applications.prepare.prepare": None,
        "virtual_staining.applications.train.train": None,
        "virtual_staining.applications.infer.infer": None,
        "virtual_staining.applications.evaluate.evaluate": None,
        "virtual_staining.applications.evaluate.evaluate_samples": None,
        "virtual_staining.applications.evaluate.evaluate_unpaired_collections": None,
        "virtual_staining.data.builder.DatasetBuilder.run_all": None,
        "virtual_staining.data.consumption.build_snapshot": None,
        "virtual_staining.data.consumption.write_snapshot": None,
        "virtual_staining.data.consumption.sha256_file_verified": None,
        "virtual_staining.checkpoint_contract.read_checkpoint": None,
        "virtual_staining.inference.runner.read_checkpoint": None,
        "virtual_staining.inference.runner.validate_checkpoint": None,
        "virtual_staining.inference.runner.load_inference_generator": None,
        "virtual_staining.inference.runner.resolve_inference_device": None,
        "torch.cuda.is_available": None,
        "torch.load": None,
        "torch.nn.Module.__init__": None,
        "PIL.Image.open": None,
    }
    for target in targets:
        monkeypatch.setattr(target, forbidden(target))


# --- shared serialization ---------------------------------------------------------------


def test_resolved_bytes_and_hash_match_the_tracked_snapshot(tmp_path: Path) -> None:
    path = write_config_data(tmp_path / "run.yaml", cyclegan_config_data(tmp_path))
    inspection = inspect_run_yaml(path)
    resolved = tmp_path / "tracked" / "resolved.yaml"

    tracked_hash = save_stage_config_snapshots(
        inspection.config,
        path,
        input_dest=tmp_path / "tracked" / "input.yaml",
        resolved_dest=resolved,
    )

    assert resolved.read_bytes() == inspection.resolved_yaml.encode("utf-8")
    assert inspection.resolved_sha256 == tracked_hash == sha256_file(resolved)
    # The serializer kept the pre-existing tracked options, so recorded hashes are unchanged.
    legacy = tmp_path / "legacy.yaml"
    with legacy.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            inspection.config.to_dict(),
            handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        )
    assert sha256_file(legacy) == tracked_hash


def test_mapping_and_yaml_inspection_agree(tmp_path: Path) -> None:
    raw = pix2pix_config_data(tmp_path)
    by_mapping = inspect_run_mapping(raw)
    by_yaml = inspect_run_yaml(write_config_data(tmp_path / "run.yaml", raw))

    assert by_mapping.resolved_yaml == by_yaml.resolved_yaml
    assert by_mapping.resolved_sha256 == by_yaml.resolved_sha256
    assert by_mapping.config == by_yaml.config
    assert by_mapping.resolved == by_mapping.config.to_dict()
    assert by_mapping.origins == by_yaml.origins


@pytest.mark.parametrize("method_name", ["pix2pix", "cyclegan"])
def test_minimal_and_full_references_inspect_identically(method_name: str) -> None:
    suffix = "" if method_name == "pix2pix" else "_cyclegan"
    minimal = inspect_run_yaml(_RUNS / f"minimal_{method_name}.yaml")
    full = inspect_run_yaml(_RUNS / f"example{suffix}.yaml")

    assert minimal.resolved_sha256 == full.resolved_sha256
    assert "defaulted" in minimal.origins.values()
    assert set(full.origins.values()) == {"supplied"}


# --- origins ------------------------------------------------------------------------------


def test_pix2pix_supplied_and_defaulted_origins(tmp_path: Path) -> None:
    origins = inspect_run_mapping(pix2pix_config_data(tmp_path)).origins

    assert origins["dataset_root"] == "supplied"
    assert origins["method.name"] == "supplied"
    assert origins["training.epochs"] == "supplied"
    assert origins["training.losses.generator[1].weight"] == "supplied"
    assert origins["model.generator.base_channels"] == "supplied"
    assert origins["data.pairing"] == "defaulted"
    assert origins["data.hash_policy"] == "defaulted"
    assert origins["training.lr_g"] == "defaulted"
    assert set(origins.values()) == {"supplied", "defaulted"}


def test_cyclegan_supplied_and_defaulted_origins(tmp_path: Path) -> None:
    origins = inspect_run_mapping(cyclegan_config_data(tmp_path)).origins

    assert origins["data.domains.stained"] == "supplied"
    assert origins["training.checkpoint_rate"] == "supplied"
    assert origins["training.losses.generator[2].weight"] == "supplied"
    assert origins["inference.checkpoint_policy"] == "supplied"
    assert origins["method.replay_buffer_size"] == "defaulted"
    assert origins["training.scheduler.name"] == "defaulted"


def test_normalized_scalar_counts_as_supplied() -> None:
    authored = {"a": 3, "b": {}}
    resolved = {"a": [3, 3], "b": {"c": 1}, "d": {"e": [{"f": 1}]}}

    assert field_origins(authored, resolved) == {
        "a": "supplied",
        "b.c": "defaulted",
        "d.e[0].f": "defaulted",
    }


# --- authored form ---------------------------------------------------------------------


def test_authored_yaml_round_trips_advanced_fields(tmp_path: Path) -> None:
    raw = yaml.safe_load((_RUNS / "example.yaml").read_text(encoding="utf-8"))
    raw["evaluation"]["metrics"] = [{"name": "mae"}, {"name": "ssim"}]
    inspection = inspect_run_mapping(raw)

    assert yaml.safe_load(inspection.authored_yaml) == raw
    assert inspection.authored == raw
    assert inspect_run_mapping(yaml.safe_load(inspection.authored_yaml)) == inspection
    # The authored form is the caller's mapping, not the resolved one.
    assert inspection.authored != inspection.resolved


def test_authored_form_is_a_copy_in_caller_order(tmp_path: Path) -> None:
    raw = cyclegan_config_data(tmp_path)
    raw["image_size"] = (32, 32)
    inspection = inspect_run_mapping(raw)
    raw["training"]["epochs"] = 99

    assert inspection.authored["training"]["epochs"] == 2
    assert inspection.authored["image_size"] == [32, 32]
    assert list(yaml.safe_load(inspection.authored_yaml)) == list(raw)


@pytest.mark.parametrize(
    ("update", "error", "match"),
    [
        ({"surprise": 1}, ValueError, "Unknown key"),
        ({"training": {"epochs": 2, "surprise": 1}}, ValueError, "Unknown key"),
        ({"method": "pix2pix"}, TypeError, "method must be a YAML mapping"),
        ({"method": {"name": "cyclegan"}}, ValueError, "method.name='cyclegan' requires"),
        (
            {"inference": {"checkpoint_policy": "latest", "direction": "B_to_A"}},
            ValueError,
            "inference.direction is not supported",
        ),
    ],
)
def test_current_owners_still_reject_invalid_configs(
    tmp_path: Path, update: dict[str, Any], error: type[Exception], match: str
) -> None:
    raw = {**pix2pix_config_data(tmp_path), **update}

    with pytest.raises(error, match=match):
        inspect_run_mapping(raw)


def test_external_definitions_keep_method_options(tmp_path: Path) -> None:
    definitions = builtin_definitions().extend(
        methods=[TinyReconstruction()], components=[TINY_CONV, TINY_RESIDUAL]
    )
    raw = {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "results"),
        "run_name": "external",
        "image_size": [32, 32],
        "method": {
            "name": "tiny_reconstruction",
            "options": {"architecture": "tiny_residual", "learning_rate": 0.002},
        },
        "data": {"pairing": "paired", "group_validation": "unavailable"},
        "model": {"inputs": ["source"], "outputs": ["target"]},
        "training": {"batch_size": 2, "epochs": 1, "seed": 1, "num_workers": 0},
    }
    by_mapping = inspect_run_mapping(raw, definitions)
    by_yaml = inspect_run_yaml(write_config_data(tmp_path / "run.yaml", raw), definitions)

    assert by_mapping.resolved_sha256 == by_yaml.resolved_sha256
    assert yaml.safe_load(by_mapping.authored_yaml)["method"] == raw["method"]
    assert by_mapping.resolved["method"]["options"]["architecture"] == "tiny_residual"
    assert by_mapping.origins["method.options.learning_rate"] == "supplied"
    with pytest.raises(ValueError):
        inspect_run_mapping(raw)
    # Extra definitions never change how a built-in config resolves.
    builtin = pix2pix_config_data(tmp_path)
    assert (
        inspect_run_mapping(builtin, definitions).resolved_yaml
        == inspect_run_mapping(builtin).resolved_yaml
    )


def test_write_config_yaml_never_overwrites(tmp_path: Path) -> None:
    text = inspect_run_mapping(pix2pix_config_data(tmp_path)).authored_yaml
    destination = write_config_yaml(text, tmp_path / "run.yaml")

    with pytest.raises(FileExistsError):
        write_config_yaml("other: 1\n", destination)
    assert destination.read_text(encoding="utf-8") == text
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run.yaml"]


# --- config-depth preflight -------------------------------------------------------------


@pytest.mark.usefixtures("forbid_execution")
def test_config_check_needs_no_assets(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path / "absent")).config
    report = preflight(config, ["train", "infer", "evaluate"])

    assert report.valid and report.depth == "config" and not report.content_verified
    assert _statuses(report)["train.assets"] == "unverified"
    assert "no dataset" in report.limitations[0]
    assert not (tmp_path / "absent").exists()


def test_config_check_reports_a_missing_stage_section(tmp_path: Path) -> None:
    config = inspect_run_mapping(pix2pix_config_data(tmp_path)).config
    report = preflight(config, ["prepare", "infer"])

    assert not report.valid
    assert _statuses(report)["prepare.config"] == "invalid"
    assert _statuses(report)["infer.config"] == "invalid"
    with pytest.raises(ValueError, match="Unknown stage"):
        preflight(config, ["publish"])


# --- asset-depth preflight ----------------------------------------------------------------


@pytest.mark.usefixtures("forbid_execution")
def test_full_paired_asset_preflight_is_read_only(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root)
    _write_checkpoint(config)
    _write_generated(config, ["00512_00000"])
    before = _tree(tmp_path)

    for stages in (["train"], ["infer"], ["evaluate"]):
        report = preflight(config, stages, depth="assets")
        assert set(_statuses(report).values()) <= {"valid", "not_applicable"}, report.checks
        assert "valid" in _statuses(report).values()
        assert not report.content_verified
        assert any("not a frozen input snapshot" in text for text in report.limitations)
    assert preflight(config, ["train", "infer", "evaluate"], depth="assets").valid
    assert _tree(tmp_path) == before


def test_prepare_inventory_success_and_failures(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    preprocessing = _write_inventory(root)
    config = inspect_run_mapping(_paired_config(tmp_path, preprocessing=preprocessing)).config
    assert _statuses(preflight(config, ["prepare"], depth="assets"))["prepare.inventory"] == "valid"

    (root / "raw" / "he1.tif").unlink()
    report = preflight(config, ["prepare"], depth="assets")
    assert not report.valid
    assert "target__stained_path not found" in report.checks[-1].message

    missing = inspect_run_mapping(
        _paired_config(tmp_path / "other", preprocessing=preprocessing)
    ).config
    report = preflight(missing, ["prepare"], depth="assets")
    assert "inventory not found" in report.checks[-1].message


def test_prepare_rejects_a_missing_mask_and_bad_columns(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    preprocessing = _write_inventory(root, mask="raw/missing_mask.png")
    config = inspect_run_mapping(_paired_config(tmp_path, preprocessing=preprocessing)).config
    assert (
        "input__label_free_mask not found"
        in preflight(config, ["prepare"], depth="assets").checks[-1].message
    )

    inventory = root / "inputs" / "slide_sets.csv"
    inventory.write_text("set_id,unexpected\nS1,x\n", encoding="utf-8")
    report = preflight(config, ["prepare"], depth="assets")
    assert _statuses(report)["prepare.inventory"] == "invalid"
    assert "missing required columns" in report.checks[-1].message


def test_paired_training_manifest_checks(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    root = config.project.dataset_root
    assert "Manifest not found" in preflight(config, ["train"], depth="assets").checks[-1].message

    _write_prepared(root)
    assert preflight(config, ["train"], depth="assets").valid

    wrong_target = inspect_run_mapping(
        _paired_config(tmp_path, model={"inputs": ["label_free"], "outputs": ["other"]})
    ).config
    assert (
        "not manifest target modalities"
        in preflight(wrong_target, ["train"], depth="assets").checks[-1].message
    )

    (root / "splits" / "val").rename(root / "splits" / "moved")
    assert (
        "Manifest file not found" in preflight(config, ["train"], depth="assets").checks[-1].message
    )


def test_paired_training_group_leakage_is_invalid(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root, patients=("p0", "p0", "p2"))

    check = preflight(config, ["train"], depth="assets").checks[-1]

    assert check.status == "invalid"
    assert "patient_id values appear in more than one split" in check.message


def _unpaired(tmp_path: Path, **data: Any) -> RunConfig:
    raw = cyclegan_config_data(tmp_path)
    raw["data"].update(data)
    raw["evaluation"] = {"protocol": "unpaired"}
    return inspect_run_mapping(raw).config


def test_unpaired_training_domains_and_groups(tmp_path: Path) -> None:
    config = _unpaired(tmp_path)
    assert "no 'train' split" in preflight(config, ["train"], depth="assets").checks[-1].message

    root = config.project.dataset_root
    _write_domains(root)
    report = preflight(config, ["train"], depth="assets")
    assert report.valid and "status=unavailable" in report.checks[-1].message

    auto = _unpaired(tmp_path, group_validation="auto")
    assert (
        "No biological group metadata"
        in preflight(auto, ["train"], depth="assets").checks[-1].message
    )

    sidecar = root / "groups.csv"
    rows = ["path,domain,split,set_id,specimen_id,patient_id"]
    rows += [
        f"domains/{domain}/{split}/{domain}_{split}.png,{domain},{split},s-{split},sp-{split},"
        + ("shared" if split != "test" else "p-test")
        for domain in ("label_free", "stained")
        for split in ("train", "val", "test")
    ]
    sidecar.write_text("\n".join(rows) + "\n", encoding="utf-8")
    leaked = _unpaired(tmp_path, group_validation="auto", group_metadata="groups.csv")
    check = preflight(leaked, ["train"], depth="assets").checks[-1]
    assert check.status == "invalid" and "patient_id" in check.message


@pytest.mark.usefixtures("forbid_execution")
def test_inference_checkpoint_is_selected_but_never_loaded(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root)
    assert "no checkpoints found" in preflight(config, ["infer"], depth="assets").checks[-1].message

    checkpoint = _write_checkpoint(config)
    check = preflight(config, ["infer"], depth="assets").checks[-1]
    assert check.status == "valid" and str(checkpoint) in check.message

    explicit = inspect_run_mapping(
        _paired_config(tmp_path, inference={"checkpoint_path": str(tmp_path / "none.pth")})
    ).config
    report = preflight(explicit, ["train", "infer"], depth="assets")
    assert _statuses(report)["infer.checkpoint"] == "invalid"


def test_cyclegan_inference_reports_direction_specific_inputs(tmp_path: Path) -> None:
    raw = cyclegan_config_data(tmp_path)
    raw["inference"]["direction"] = "B_to_A"
    config = inspect_run_mapping(raw).config
    _write_prepared(config.project.dataset_root)

    check = preflight(config, ["infer"], depth="assets").checks[-2]

    assert check.check_id == "infer.manifest" and check.status == "valid"
    assert "direction=B_to_A" in check.message and "inputs=['stained']" in check.message
    assert "outputs=['label_free']" in check.message


def test_paired_evaluation_expects_output_named_artifacts(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root)
    check = preflight(config, ["evaluate"], depth="assets").checks[-1]
    assert check.check_id == "evaluate.generated" and check.status == "invalid"
    assert "stained/00512_00000_generated.tif" in check.message

    _write_generated(config, ["00512_00000"])
    assert preflight(config, ["evaluate"], depth="assets").valid

    absent = inspect_run_mapping(_paired_config(tmp_path / "absent")).config
    assert _statuses(preflight(absent, ["evaluate"], depth="assets")) == {
        "config.resolve": "valid",
        "evaluate.config": "valid",
        "evaluate.manifest": "invalid",
        "evaluate.generated": "unverified",
    }


def test_unpaired_evaluation_collections(tmp_path: Path) -> None:
    config = _unpaired(tmp_path)
    _write_domains(config.project.dataset_root)
    report = preflight(config, ["evaluate"], depth="assets")
    assert _statuses(report)["evaluate.reference"] == "valid"
    assert _statuses(report)["evaluate.generated"] == "invalid"

    _write_generated(config, ["a"], output_name="label_free")
    assert not preflight(config, ["evaluate"], depth="assets").valid
    _write_generated(config, ["a"], output_name="stained")
    assert preflight(config, ["evaluate"], depth="assets").valid

    raw = cyclegan_config_data(tmp_path)
    raw["evaluation"] = {"protocol": "unpaired", "reference_collection": "elsewhere"}
    explicit = inspect_run_mapping(raw).config
    assert (
        _statuses(preflight(explicit, ["evaluate"], depth="assets"))["evaluate.reference"]
        == "invalid"
    )


# --- planned artifacts ------------------------------------------------------------------


def test_artifacts_from_earlier_selected_stages_are_planned(tmp_path: Path) -> None:
    preprocessing = _write_inventory(tmp_path / "dataset")
    config = inspect_run_mapping(_paired_config(tmp_path, preprocessing=preprocessing)).config

    report = preflight(config, ["prepare", "train", "infer", "evaluate"], depth="assets")

    assert report.valid
    assert _statuses(report) == {
        "config.resolve": "valid",
        "prepare.config": "valid",
        "prepare.inventory": "valid",
        "train.config": "valid",
        "train.manifest": "planned",
        "infer.config": "valid",
        "infer.manifest": "planned",
        "infer.checkpoint": "planned",
        "evaluate.config": "valid",
        "evaluate.manifest": "planned",
        "evaluate.generated": "planned",
    }
    assert any("planned checks were not verified" in text for text in report.limitations)


def test_wrong_stage_order_does_not_plan(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root)

    report = preflight(config, ["infer", "train"], depth="assets")

    assert not report.valid
    assert _statuses(report)["infer.checkpoint"] == "invalid"
    evaluate_first = preflight(config, ["evaluate", "infer"], depth="assets")
    assert _statuses(evaluate_first)["evaluate.generated"] == "invalid"


def test_generated_dir_outside_inference_output_is_not_planned(tmp_path: Path) -> None:
    config = inspect_run_mapping(
        _paired_config(tmp_path, evaluation={"generated_dir": str(tmp_path / "external")})
    ).config
    _write_prepared(config.project.dataset_root)
    _write_checkpoint(config)

    report = preflight(config, ["infer", "evaluate"], depth="assets")

    assert _statuses(report)["evaluate.generated"] == "invalid"


# --- staleness --------------------------------------------------------------------------


def test_a_successful_preflight_does_not_freeze_assets(tmp_path: Path) -> None:
    config = inspect_run_mapping(_paired_config(tmp_path)).config
    _write_prepared(config.project.dataset_root)
    report = preflight(config, ["train"], depth="assets")
    kept = copy.deepcopy(report)
    assert report.valid

    record = make_manifest_record("00256_00000", "val")
    target = config.project.dataset_root / record.target_paths["stained"]
    target.unlink()

    # The earlier report is a value, not a lock: it still says valid and protected nothing.
    assert report == kept and report.valid
    rerun = preflight(config, ["train"], depth="assets")
    assert not rerun.valid and "Manifest file not found" in rerun.checks[-1].message


def _write_two_target_prepared(root: Path) -> None:
    records = []
    for index, split in enumerate(("train", "val", "test")):
        sample_id = f"{index * 256:05}_00000"
        record = make_manifest_record(
            sample_id,
            split,
            set_id=f"S{index}",
            input_paths={"label_free": Path(f"splits/{split}/{sample_id}_lf.tif")},
            target_paths={
                name: Path(f"splits/{split}/{sample_id}_{name}.tif") for name in ("HE", "PAS")
            },
        )
        for path in (*record.input_paths.values(), *record.target_paths.values()):
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(b"placeholder, never decoded")
        records.append(record)
    layout = DatasetLayout(root)
    metadata = manifest_metadata(("label_free",), ("HE", "PAS"))
    DatasetManifest(tuple(records), root, metadata).to_csv(layout.manifest_path)
    layout.manifest_metadata_path.write_text(json.dumps(metadata.to_dict()))
    with layout.slide_sets_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set_id", "specimen_id", "patient_id"])
        writer.writerows([f"S{index}", f"sp{index}", f"p{index}"] for index in range(3))


def test_two_output_asset_preflight_checks_every_selected_output(tmp_path: Path) -> None:
    data = pix2pix_config_data(tmp_path, inputs=("label_free",), outputs=("PAS", "HE"))
    data["inference"] = {"checkpoint_policy": "latest"}
    data["evaluation"] = {}
    config = inspect_run_mapping(data).config
    _write_two_target_prepared(config.project.dataset_root)

    statuses = _statuses(preflight(config, ["train", "infer", "evaluate"], depth="assets"))
    assert statuses["train.manifest"] == "valid"
    assert statuses["evaluate.generated"] == "planned"
    report = preflight(config, ["evaluate"], depth="assets")
    assert "2 of 2 expected generated file(s) missing" in report.checks[-1].message

    _write_generated(config, ["00512_00000"], output_name="HE")
    report = preflight(config, ["evaluate"], depth="assets")
    assert not report.valid
    assert "PAS/00512_00000_generated.tif" in report.checks[-1].message
    _write_generated(config, ["00512_00000"], output_name="PAS")
    assert preflight(config, ["evaluate"], depth="assets").valid

    selected = copy.deepcopy(data)
    selected["model"]["outputs"] = ["IHC"]
    subset = inspect_run_mapping(selected).config
    check = preflight(subset, ["train"], depth="assets").checks[-1]
    assert check.status == "invalid" and "not manifest target modalities" in check.message


def test_config_check_resolves_plural_outputs_without_assets(tmp_path: Path) -> None:
    data = pix2pix_config_data(tmp_path, outputs=("PAS", "HE"))
    inspection = inspect_run_mapping(data)

    assert inspection.resolved["model"]["outputs"] == ["PAS", "HE"]
    assert "target" not in inspection.resolved["model"]
    assert preflight(inspection.config, ["train"]).valid
    singular = copy.deepcopy(data)
    singular["model"]["target"] = "HE"
    with pytest.raises(ValueError, match="model.target is not part of the current schema"):
        inspect_run_mapping(singular)
