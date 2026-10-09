"""Selected operations share one resolver; omission never excuses malformed sections."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.config_helpers import cyclegan_config_data, pix2pix_config_data, prepare_config_data
from tests.external_method.test_tiny_reconstruction import _definitions, _mapping
from virtual_staining.config.run import RunConfig
from virtual_staining.definitions import Definitions
from virtual_staining.utils.hashing import sha256_bytes


def test_minimal_prepare_needs_no_method_definitions_or_tracked_identity(tmp_path: Path) -> None:
    raw = prepare_config_data(tmp_path)
    config = RunConfig.from_mapping(raw, Definitions(), stages=("prepare",))
    assert config.method is None and config.model is None
    assert config.project.results_path is None and config.project.run_name is None
    assert config.project.image_size == (256, 256)
    assert config.preprocessing is not None
    assert config.preprocessing.patching.patch_size == (256, 256)
    assert set(config.to_dict()) == {"dataset_root", "image_size", "preprocessing", "data"}
    again = RunConfig.from_mapping(config.to_dict(), Definitions(), stages=config.stages)
    assert again == config
    assert again.resolved_yaml() == config.resolved_yaml()
    assert config.resolved_yaml().startswith("# Resolve with stages: prepare\n")
    assert sha256_bytes(again.resolved_yaml().encode()) == sha256_bytes(
        config.resolved_yaml().encode()
    )


@pytest.mark.parametrize("stage", ["train", "infer", "evaluate"])
def test_independent_pix2pix_requirements_and_named_outputs(tmp_path: Path, stage: str) -> None:
    raw = pix2pix_config_data(tmp_path, outputs=("HE", "IHC"))
    if stage != "train":
        raw.pop("training")
        raw["model"].pop("discriminator")
    if stage == "evaluate":
        raw["model"].pop("generator")
    if stage == "infer":
        raw["inference"] = {"checkpoint_path": "checkpoints/ep000.pth"}
    config = RunConfig.from_mapping(raw, stages=(stage,))
    assert config.model is not None and config.method is not None
    assert config.model.inputs == ("LF", "AF")
    assert config.model.outputs == ("HE", "IHC")
    assert config.preprocessing is None and config.evaluation is None
    assert (config.training is not None) == (stage == "train")
    assert (config.method.options.discriminator is not None) == (stage == "train")
    assert (config.method.options.generator is not None) == (stage != "evaluate")
    assert (
        RunConfig.from_mapping(config.to_dict(), stages=(stage,)).resolved_yaml()
        == config.resolved_yaml()
    )


@pytest.mark.parametrize("stage", ["infer", "evaluate"])
@pytest.mark.parametrize("direction", ["A_to_B", "B_to_A"])
def test_cyclegan_needs_only_consumed_domains(tmp_path: Path, stage: str, direction: str) -> None:
    raw = cyclegan_config_data(tmp_path)
    raw.pop("training")
    raw["data"].pop("domains")
    raw["model"].pop("discriminator")
    raw["inference"] = {"direction": direction}
    if stage == "infer":
        raw["inference"]["checkpoint_policy"] = "latest"
    else:
        raw["model"].pop("generator")
        raw["evaluation"] = {"reference_collection": "independent/{split}/*.png"}
    config = RunConfig.from_mapping(raw, stages=(stage,))
    assert config.method is not None
    expected = ("stained",) if direction == "A_to_B" else ("label_free",)
    assert config.method.definition.prediction_outputs(config, direction) == expected
    assert config.data.domains == {}
    if stage == "evaluate":
        # Only the active reference domain is needed when no explicit collection is given.
        raw.pop("evaluation")
        raw["data"]["domains"] = {expected[0]: "independent/{split}/*.png"}
        RunConfig.from_mapping(raw, stages=(stage,))
        raw["data"].pop("domains")
        with pytest.raises(ValueError, match="evaluation.reference_collection"):
            RunConfig.from_mapping(raw, stages=(stage,))


@pytest.mark.parametrize("stage", ["train", "infer", "evaluate"])
def test_explicit_external_definitions_share_selected_resolution(
    tmp_path: Path, stage: str
) -> None:
    definitions, method = _definitions()
    raw = _mapping(tmp_path)
    if stage != "train":
        raw.pop("training")
    if stage != "infer":
        raw.pop("inference", None)
    config = RunConfig.from_mapping(raw, definitions, stages=(stage,))
    assert config.method is not None and config.method.definition is method
    assert RunConfig.from_mapping(config.to_dict(), definitions, stages=(stage,)) == config
    with pytest.raises(ValueError, match="not a registered method"):
        RunConfig.from_mapping(raw, stages=(stage,))


@pytest.mark.parametrize(
    "section", ["preprocessing", "training", "inference", "evaluation", "method", "model", "data"]
)
@pytest.mark.parametrize("bad", [None, [], "invalid", 2])
def test_malformed_supplied_sections_are_not_omission(
    tmp_path: Path, section: str, bad: object
) -> None:
    raw = prepare_config_data(tmp_path)
    raw[section] = bad
    with pytest.raises(TypeError, match=section + " must be a YAML mapping"):
        RunConfig.from_mapping(raw, stages=("prepare",))


@pytest.mark.parametrize(
    ("section", "bad", "field"),
    [
        ("evaluation", {"unknown": True}, "evaluation"),
        ("evaluation", {"save_graphs": "false"}, "evaluation.save_graphs"),
        ("inference", {"unknown": True}, "inference"),
        ("training", {"unknown": True}, "training"),
        ("method", {"provider": "executable.module"}, "method"),
        ("model", {"target": "HE"}, "model.target"),
        (
            "model",
            {"inputs": ["LF"], "outputs": ["HE"], "discriminator": {"ndf": "invalid"}},
            "ndf",
        ),
    ],
)
def test_invalid_inactive_sections_still_fail(
    tmp_path: Path, section: str, bad: dict[str, Any], field: str
) -> None:
    raw = prepare_config_data(tmp_path)
    raw["model"] = {"inputs": ["LF"], "outputs": ["HE"]}
    raw[section] = bad
    with pytest.raises((TypeError, ValueError), match=field):
        RunConfig.from_mapping(raw, stages=("prepare",))


@pytest.mark.parametrize("field", ["results_path", "run_name", "dataset_root"])
@pytest.mark.parametrize("bad", [None, False, 7, [], ""])
def test_supplied_project_fields_are_strict(tmp_path: Path, field: str, bad: object) -> None:
    raw = prepare_config_data(tmp_path)
    raw[field] = bad
    with pytest.raises(TypeError, match=field):
        RunConfig.from_mapping(raw, stages=("prepare",))


def test_nested_null_and_inconsistent_modalities_fail(tmp_path: Path) -> None:
    raw = prepare_config_data(tmp_path)
    raw["preprocessing"]["patching"] = None
    with pytest.raises(TypeError, match="preprocessing.patching"):
        RunConfig.from_mapping(raw, stages=("prepare",))
    raw["preprocessing"].pop("patching")
    raw["model"] = {"inputs": ["missing"], "outputs": ["HE"]}
    with pytest.raises(ValueError, match="model.inputs.*preprocessing.inputs.modalities"):
        RunConfig.from_mapping(raw, stages=("prepare",))


def test_current_complete_schema_keeps_effective_defaults(tmp_path: Path) -> None:
    raw = pix2pix_config_data(tmp_path)
    raw["inference"] = {"checkpoint_policy": "latest"}
    raw["preprocessing"] = deepcopy(prepare_config_data(tmp_path)["preprocessing"])
    raw["preprocessing"]["inputs"]["target_modalities"] = ["stained"]
    old_context = RunConfig.from_mapping(raw)
    selected = RunConfig.from_mapping(raw, stages=("prepare", "train", "infer", "evaluate"))
    assert selected.to_dict() == old_context.to_dict()


def test_checkpoint_selection_is_required_only_for_infer(tmp_path: Path) -> None:
    raw = pix2pix_config_data(tmp_path)
    raw.pop("training")
    raw["inference"] = {"output_dir": str(tmp_path / "generated")}
    RunConfig.from_mapping(raw, stages=("evaluate",))
    with pytest.raises(
        ValueError, match="inference.checkpoint_path or inference.checkpoint_policy"
    ):
        RunConfig.from_mapping(raw, stages=("infer",))


def test_evaluation_can_resolve_without_any_network_definitions(tmp_path: Path) -> None:
    from virtual_staining.methods.builtin import builtin_definitions

    builtins = builtin_definitions()
    definitions = Definitions(methods=builtins.methods, metrics=builtins.metrics)
    raw = pix2pix_config_data(tmp_path)
    raw.pop("training")
    raw["model"] = {"inputs": ["LF"], "outputs": ["HE"]}
    config = RunConfig.from_mapping(raw, definitions, stages=("evaluate",))
    assert config.method is not None
    assert config.method.options.generator is None and config.method.options.discriminator is None


def test_cyclegan_domain_requirements_belong_to_selected_operations(tmp_path: Path) -> None:
    raw = cyclegan_config_data(tmp_path)
    raw["data"].pop("domains")
    raw["evaluation"] = {}
    # Supplied training/evaluation options are validated, but their data is not consumed.
    RunConfig.from_mapping(raw, stages=("infer",))
    with pytest.raises(ValueError, match="data.domains requires training domains"):
        RunConfig.from_mapping(raw, stages=("train",))
    with pytest.raises(ValueError, match="evaluation.reference_collection"):
        RunConfig.from_mapping(raw, stages=("evaluate",))


def test_evaluation_default_metrics_are_validated_before_execution(tmp_path: Path) -> None:
    from virtual_staining.methods.builtin import builtin_definitions

    definitions = Definitions(methods=builtin_definitions().methods)
    raw = pix2pix_config_data(tmp_path)
    raw.pop("training")
    raw["model"] = {"inputs": ["LF"], "outputs": ["HE"]}
    with pytest.raises(ValueError, match="evaluation.metrics"):
        RunConfig.from_mapping(raw, definitions, stages=("evaluate",))
