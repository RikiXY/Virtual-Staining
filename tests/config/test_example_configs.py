"""Committed YAML examples: parse, minimal/full equivalence, and option coverage.

The full references (config/runs/example.yaml, example_cyclegan.yaml) document every
public option; these tests catch drift between them and the parsers. Mutually exclusive
choices are exercised as variants of the committed references.
"""

from __future__ import annotations

import copy
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

from virtual_staining.applications import run_queue as queue_module
from virtual_staining.applications.run_queue import (
    QueueAblationError,
    _build_ablation_summary,
    _load_local_run_queue,
    _preflight_run_configs,
)
from virtual_staining.checkpoint_selection import SUPPORTED_CHECKPOINT_POLICIES
from virtual_staining.config import data as preprocessing_module
from virtual_staining.config import evaluation, experiment_data, inference, losses, method, model
from virtual_staining.config import run as run_module
from virtual_staining.config import scheduler as scheduler_module
from virtual_staining.config import training as training_module
from virtual_staining.config.run import RunConfig
from virtual_staining.config.stages import VALID_STAGES
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.definitions import ComponentContext
from virtual_staining.loss_definitions import (
    _LOSS_MASK_KEYS,
    _REDUCTIONS,
    LOSS_DEFINITIONS,
    LossMaskSource,
    SsimChannelMode,
)
from virtual_staining.methods.builtin import (
    PIX2PIX_IMAGE_METRICS,
    CycleGANDefinition,
    Pix2PixDefinition,
    builtin_definitions,
)
from virtual_staining.models import components
from virtual_staining.utils.image_io import SUPPORTED_IMAGE_BACKENDS

_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "config" / "runs"
_QUEUES = _ROOT / "config" / "queues"
_FULL = {"pix2pix": _RUNS / "example.yaml", "cyclegan": _RUNS / "example_cyclegan.yaml"}
_MINIMAL = {
    "pix2pix": _RUNS / "minimal_pix2pix.yaml",
    "cyclegan": _RUNS / "minimal_cyclegan.yaml",
}
# The ranked Pix2Pix checkpoint columns of the one-output examples (output HE).
_PIX2PIX_CHECKPOINT_METRICS = Pix2PixDefinition().checkpoint_modes(("HE",))
_REFERENCE_TEXT = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (*_FULL.values(), _QUEUES / "example.yaml", _QUEUES / "example_ablation.yaml")
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    node: Any = data
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    if isinstance(node, list):
        index = int(last)
        node[index : index + 1] = [value]  # index == len(node) appends
    else:
        node[last] = value


def _get(data: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _variant(method_name: str, updates: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(_load(_FULL[method_name]))
    for dotted, value in updates.items():
        _set(data, dotted, value)
    return data


def _parse(tmp_path: Path, data: dict[str, Any]) -> RunConfig:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return RunConfig.from_yaml(path)


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    """Dot paths of every leaf; lists of mappings are indexed, other lists are leaves."""
    if isinstance(value, dict) and value:
        return {
            path
            for key, item in value.items()
            for path in _leaf_paths(item, f"{prefix}.{key}" if prefix else str(key))
        }
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return {
            path
            for index, item in enumerate(value)
            for path in _leaf_paths(item, f"{prefix}.{index}")
        }
    return {prefix}


# --- 1. Committed examples parse ---------------------------------------------


@pytest.mark.parametrize("path", sorted(_RUNS.glob("*.yaml")), ids=lambda path: path.name)
def test_committed_run_examples_parse(path: Path) -> None:
    config = RunConfig.from_yaml(path)

    assert RunConfig.from_yaml(path).to_dict() == config.to_dict()


@pytest.mark.parametrize(
    ("method_name", "pairing", "architecture"),
    [("pix2pix", "paired", "concat_unet"), ("cyclegan", "unpaired", "resnet")],
)
def test_full_references_select_their_method(
    method_name: str, pairing: str, architecture: str
) -> None:
    config = RunConfig.from_yaml(_FULL[method_name])

    assert config.method is not None
    assert config.method.name == method_name
    assert config.data.pairing == pairing
    assert config.method.options.generator.name == architecture


# --- 2. Minimal starters are the full references with defaults omitted ------------


@pytest.mark.parametrize("method_name", sorted(_FULL))
def test_minimal_starter_resolves_to_full_reference(method_name: str) -> None:
    minimal = RunConfig.from_yaml(_MINIMAL[method_name]).to_dict()
    full = RunConfig.from_yaml(_FULL[method_name]).to_dict()

    assert minimal == full


@pytest.mark.parametrize("method_name", sorted(_FULL))
def test_minimal_starter_is_shorter_than_full_reference(method_name: str) -> None:
    assert len(_leaf_paths(_load(_MINIMAL[method_name]))) < len(
        _leaf_paths(_load(_FULL[method_name]))
    )


# --- 3. Full references write every effective field ---------------------------


@pytest.mark.parametrize("method_name", sorted(_FULL))
def test_full_reference_writes_every_effective_field(method_name: str) -> None:
    raw = _load(_FULL[method_name])
    resolved = RunConfig.from_yaml(_FULL[method_name]).to_dict()

    missing = sorted(_leaf_paths(resolved) - _leaf_paths(raw))

    assert missing == []


# --- 4. Reviewed public-option coverage map --------------------------------------
# Each entry pairs the parser's current key/choice set with the reviewed set the
# examples document. A new parser key or choice fails here until the reference YAMLs,
# this map, and the variant tests below are updated.


def _dataclass_keys(cls: type) -> frozenset[str]:
    return frozenset(field.name for field in fields(cls))


def _owned(section: str) -> frozenset[str]:
    """Keys the built-in definitions own in ``section``."""
    return frozenset(
        key
        for definition in builtin_definitions().methods.values()
        for key in definition.owned_keys.get(section, ())
    )


def _component_keys(definition: Any) -> frozenset[str]:
    """Option keys a built-in component resolves (its defaults fill every key)."""
    context = ComponentContext(field="model.component", image_size=(256, 256))
    return frozenset(definition.resolve({}, context).options)


_LOSS_PARAM_KEYS = {name: definition.param_keys for name, definition in LOSS_DEFINITIONS.items()}

_PUBLIC_OPTIONS: dict[str, tuple[frozenset[str], set[str]]] = {
    "top-level keys": (
        run_module._TOP_LEVEL_KEYS,
        {
            "dataset_root",
            "results_path",
            "run_name",
            "image_size",
            "manifest_path",
            "method",
            "data",
            "model",
            "preprocessing",
            "training",
            "inference",
            "evaluation",
        },
    ),
    "method keys": (method.METHOD_KEYS | _owned("method"), {"name", "replay_buffer_size"}),
    "method names": (frozenset(builtin_definitions().methods), {"pix2pix", "cyclegan"}),
    "data keys": (
        experiment_data._DATA_KEYS,
        {"pairing", "domains", "hash_policy", "group_validation", "group_metadata"},
    ),
    "data pairings": (frozenset(get_args(experiment_data.DataPairing)), {"paired", "unpaired"}),
    "hash policies": (frozenset(get_args(experiment_data.HashPolicy)), {"content", "membership"}),
    "group validation": (
        frozenset(get_args(experiment_data.GroupValidation)),
        {"auto", "patient", "specimen", "set", "unavailable"},
    ),
    "model keys": (
        model.MODEL_KEYS | _owned("model"),
        {"inputs", "outputs", "generator", "discriminator"},
    ),
    "generator architectures": (
        frozenset(
            {Pix2PixDefinition.generator_architecture, CycleGANDefinition.generator_architecture}
        ),
        {"concat_unet", "resnet"},
    ),
    "concat_unet keys": (
        _component_keys(components.CONCAT_UNET) | {"architecture"},
        {"architecture", "base_channels", "norm", "dropout", "bilinear"},
    ),
    "resnet keys": (
        _component_keys(components.RESNET) | {"architecture"},
        {"architecture", "base_channels", "norm", "blocks"},
    ),
    "norms": (frozenset(get_args(components.NormName)), {"batch", "instance"}),
    "discriminator keys": (_component_keys(components.PATCHGAN), {"ndf", "norm", "use_sigmoid"}),
    "preprocessing sections": (
        preprocessing_module._SECTION_KEYS,
        {"inputs", "patching", "masks", "alignment", "filtering", "split", "io"},
    ),
    "preprocessing.inputs keys": (
        _dataclass_keys(preprocessing_module.InputConfig),
        {"inventory", "modalities", "reference", "target_modalities", "hash_verification"},
    ),
    "preprocessing.patching keys": (
        _dataclass_keys(preprocessing_module.PatchingConfig),
        {"patch_size", "grid_movement", "margin", "save_discarded_patches"},
    ),
    "preprocessing.masks keys": (
        _dataclass_keys(preprocessing_module.MaskConfig),
        {
            "generation",
            "strategy",
            "scale",
            "lowres_filtering",
            "save_resolved_masks",
            "save_patch_masks",
        },
    ),
    "mask strategies": (
        frozenset(preprocessing_module.ALLOWED_MASK_STRATEGIES),
        {"connected_components", "hsv"},
    ),
    "preprocessing.alignment keys": (
        _dataclass_keys(preprocessing_module.AlignmentConfig),
        {"mode", "method", "validate_declared", "on_failure"},
    ),
    "preprocessing.filtering keys": (
        _dataclass_keys(preprocessing_module.FilteringConfig),
        {"foreground", "max_white_ratio", "white_threshold", "max_largest_white_component_ratio"},
    ),
    "preprocessing.filtering.foreground keys": (
        _dataclass_keys(preprocessing_module.ForegroundFilterConfig),
        {"enabled", "policy", "min_ratio"},
    ),
    "preprocessing.split keys": (
        _dataclass_keys(preprocessing_module.SplitConfig),
        {"unit", "train", "val", "test", "seed", "assignment_file"},
    ),
    "preprocessing.io keys": (
        _dataclass_keys(preprocessing_module.IOConfig),
        {"tiled", "backend", "max_memory_gb"},
    ),
    "image backends": (SUPPORTED_IMAGE_BACKENDS, {"auto", "pillow", "openslide"}),
    "training keys": (
        training_module.TRAINING_KEYS | _owned("training"),
        {
            "batch_size",
            "epochs",
            "lr_g",
            "lr_d",
            "beta1",
            "beta2",
            "seed",
            "num_workers",
            "validate_rate",
            "checkpoint_rate",
            "checkpoint_top_k",
            "log_rate",
            "resume",
            "scheduler",
            "early_stopping",
            "augmentation",
            "losses",
        },
    ),
    "scheduler keys": (
        scheduler_module._SCHEDULER_KEYS,
        {"name", "decay_start_epoch", "monitor", "mode", "factor", "patience", "min_lr"},
    ),
    "scheduler names": (
        frozenset(get_args(scheduler_module.LearningRateSchedulerName)),
        {"none", "linear_decay", "reduce_on_plateau"},
    ),
    "early stopping keys": (
        training_module._EARLY_STOPPING_KEYS,
        {"monitor", "mode", "patience", "min_delta"},
    ),
    "augmentation keys": (
        training_module._AUGMENTATION_KEYS,
        {"enabled", "expansion_factor", "intensity", "photometric_inputs"},
    ),
    "augmentation intensities": (
        frozenset(get_args(training_module.AugmentationIntensity)),
        {"light", "medium", "strong"},
    ),
    "checkpoint metrics": (
        frozenset({"loss_G_val", *(f"val_{name}__<output>" for name in PIX2PIX_IMAGE_METRICS)}),
        {
            "loss_G_val",
            "val_ssim__<output>",
            "val_psnr__<output>",
            "val_mae__<output>",
            "val_rmse__<output>",
            "val_pcc_gray__<output>",
            "val_pcc_rgb_mean__<output>",
        },
    ),
    "loss term keys": (losses._LOSS_TERM_KEYS, {"name", "weight", "enabled", "params", "schedule"}),
    "loss names": (
        frozenset(LOSS_DEFINITIONS),
        {"adversarial_bce", "l1", "ssim", "adversarial_lsgan", "cycle_l1", "identity_l1"},
    ),
    "l1 params": (_LOSS_PARAM_KEYS["l1"], {"reduction", "mask"}),
    "ssim params": (
        _LOSS_PARAM_KEYS["ssim"],
        {"data_range", "window_size", "sigma", "channel_mode", "reduction", "mask"},
    ),
    "param-free losses": (
        frozenset(name for name, keys in _LOSS_PARAM_KEYS.items() if not keys),
        {"adversarial_bce", "adversarial_lsgan", "cycle_l1", "identity_l1"},
    ),
    "loss reductions": (frozenset(_REDUCTIONS), {"mean", "sum", "none"}),
    "ssim channel modes": (frozenset(get_args(SsimChannelMode)), {"rgb", "gray"}),
    "loss mask keys": (
        _LOSS_MASK_KEYS,
        {"enabled", "source", "foreground_weight", "background_weight", "ignore_empty_mask"},
    ),
    "loss mask sources": (frozenset(get_args(LossMaskSource)), {"foreground_mask"}),
    "loss schedule keys": (
        losses._LOSS_SCHEDULE_KEYS,
        {"type", "start_epoch", "end_epoch", "epoch", "factor"},
    ),
    "loss schedule types": (
        frozenset(get_args(losses.LossScheduleType)),
        {
            "constant",
            "linear_warmup",
            "linear_decay",
            "step",
            "cosine",
            "turn_on_after_epoch",
            "turn_off_after_epoch",
        },
    ),
    "inference keys": (
        inference._INFERENCE_KEYS,
        {
            "checkpoint_path",
            "checkpoint_policy",
            "checkpoint_metric",
            "checkpoint_rank",
            "output_dir",
            "direction",
        },
    ),
    "checkpoint policies": (SUPPORTED_CHECKPOINT_POLICIES, {"latest", "best", "top_k"}),
    "inference directions": (
        frozenset(CycleGANDefinition.prediction_directions),
        {"A_to_B", "B_to_A"},
    ),
    "evaluation keys": (
        evaluation._EVALUATION_KEYS,
        {
            "save_graphs",
            "generated_dir",
            "output_dir",
            "bootstrap_iterations",
            "bootstrap_seed",
            "protocol",
            "metrics",
            "input_failures",
            "reference_collection",
        },
    ),
    "evaluation protocols": (
        frozenset(get_args(evaluation.EvaluationProtocol)),
        {"paired", "unpaired"},
    ),
    "evaluation input failures": (
        frozenset(get_args(evaluation.InputFailureMode)),
        {"strict", "permissive"},
    ),
    "evaluation metrics": (
        frozenset(builtin_definitions().metrics),
        {
            "mae",
            "mse",
            "rmse",
            "psnr",
            "ssim",
            "pcc_gray",
            "pcc_r",
            "pcc_g",
            "pcc_b",
            "pcc_rgb_mean",
        },
    ),
    "queue keys": (queue_module._QUEUE_KEYS, {"name", "continue_on_failure", "jobs", "ablation"}),
    "queue job keys": (queue_module._QUEUE_JOB_KEYS, {"config_path", "label", "notes", "stages"}),
    "ablation keys": (queue_module._ABLATION_KEYS, {"fixed_fields", "variable_fields"}),
    "stages": (frozenset(VALID_STAGES), {"prepare", "train", "infer", "evaluate"}),
}

# Choices validated inline (no importable constant); pinned by the variant tests below.
_INLINE_CHOICES = {
    "hash_verification": {"cached", "always"},
    "mask generation": {"never", "if_missing", "always"},
    "alignment modes": {"auto", "always", "never"},
    "alignment methods": {"affine_sift"},
    "alignment failure": {"error", "skip_set"},
    "foreground policies": {"reference", "target", "all", "intersection", "union"},
    "split units": {"patch", "set", "specimen", "patient"},
}


@pytest.mark.parametrize("family", sorted(_PUBLIC_OPTIONS))
def test_public_option_family_matches_reviewed_map(family: str) -> None:
    current, reviewed = _PUBLIC_OPTIONS[family]

    assert set(current) == reviewed, f"{family}: update the reference YAMLs and this map"


@pytest.mark.parametrize(
    "token",
    sorted(
        {token for _, reviewed in _PUBLIC_OPTIONS.values() for token in reviewed}
        | {token for choices in _INLINE_CHOICES.values() for token in choices}
    ),
)
def test_every_public_option_appears_in_a_reference(token: str) -> None:
    assert token in _REFERENCE_TEXT


# Families that apply to a CycleGAN run; its reference must explain them on its own.
_CYCLEGAN_FAMILIES = (
    "method keys",
    "data keys",
    "hash policies",
    "group validation",
    "model keys",
    "resnet keys",
    "norms",
    "discriminator keys",
    "training keys",
    "scheduler keys",
    "scheduler names",
    "early stopping keys",
    "augmentation keys",
    "augmentation intensities",
    "loss term keys",
    "loss schedule keys",
    "loss schedule types",
    "inference keys",
    "checkpoint policies",
    "inference directions",
    "evaluation keys",
    "evaluation protocols",
    "evaluation input failures",
)


@pytest.mark.parametrize(
    "token",
    sorted(
        {"dataset_root", "manifest_path", "results_path", "run_name", "image_size"}
        | {"adversarial_lsgan", "cycle_l1", "identity_l1", "loss_G_val"}
        | {token for family in _CYCLEGAN_FAMILIES for token in _PUBLIC_OPTIONS[family][1]}
    ),
)
def test_cyclegan_reference_documents_applicable_option_itself(token: str) -> None:
    assert token in _FULL["cyclegan"].read_text(encoding="utf-8")


_PLACEHOLDER = re.compile(
    r"same (?:as|rules as|keys and rules as)|as in example|documented (?:elsewhere|in full in)"
    r"|see (?:the )?docs|see (?:config/runs/)?example\.yaml",
    re.IGNORECASE,
)


@pytest.mark.parametrize("method_name", sorted(_FULL))
def test_full_reference_does_not_defer_to_another_file(method_name: str) -> None:
    text = _FULL[method_name].read_text(encoding="utf-8")

    assert _PLACEHOLDER.findall(text) == []


def test_manifest_path_is_a_consumer_override_not_a_preparation_output() -> None:
    root = Path("dataset")
    override = Path("elsewhere/manifest.csv")
    project = RunConfig.from_yaml(_FULL["pix2pix"]).project
    project = replace(project, dataset_root=root, manifest_path_override=override)

    # prepare builds its layout from dataset_root alone; consumers use the project layout.
    assert DatasetLayout(root).manifest_path == root / "manifests" / "manifest.csv"
    assert DatasetLayout.from_project(project).manifest_path == override
    # Only the CSV moves; its metadata stays under dataset_root.
    assert DatasetLayout.from_project(project).manifest_metadata_path == (
        root / "manifests" / "manifest_metadata.json"
    )
    for path in _FULL.values():
        text = path.read_text(encoding="utf-8")
        assert "not a preparation output" in text.lower()
        # Provenance/split companions are optional; never document them as required.
        for companion in ("split_assignment.csv", "dataset_fingerprint.json"):
            lines = [line for line in text.splitlines() if companion in line]
            assert lines and all("optional" in line for line in lines), companion


# --- 5. Valid choice variants ----------------------------------------------------

_SSIM_FULL_PARAMS = {
    "data_range": 1.0,
    "window_size": 7,
    "sigma": 1.0,
    "channel_mode": "gray",
    "reduction": "sum",
    "mask": {
        "enabled": True,
        "source": "foreground_mask",
        "foreground_weight": 1.0,
        "background_weight": 0.25,
        "ignore_empty_mask": False,
    },
}
_L1 = "training.losses.generator.1"
_VALID_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("pix2pix", {"data.hash_policy": "membership"}),
    *[
        ("pix2pix", {"data.group_validation": value})
        for value in _PUBLIC_OPTIONS["group validation"][1]
    ],
    ("pix2pix", {"manifest_path": "elsewhere/manifest.csv"}),
    ("pix2pix", {"image_size": [320, 256]}),
    ("pix2pix", {"preprocessing.inputs.hash_verification": "always"}),
    ("pix2pix", {"preprocessing.patching.grid_movement": [128, 128]}),
    ("pix2pix", {"preprocessing.patching.save_discarded_patches": True}),
    (
        "pix2pix",
        {
            "preprocessing.masks.generation": "never",
            "preprocessing.filtering.foreground.enabled": False,
        },
    ),
    ("pix2pix", {"preprocessing.masks.generation": "always"}),
    ("pix2pix", {"preprocessing.masks.strategy": "hsv", "preprocessing.masks.scale": 0.25}),
    ("pix2pix", {"preprocessing.masks.lowres_filtering": True}),
    ("pix2pix", {"preprocessing.masks.save_resolved_masks": True}),
    ("pix2pix", {"preprocessing.masks.save_patch_masks": True}),
    ("pix2pix", {"preprocessing.alignment.mode": "always"}),
    ("pix2pix", {"preprocessing.alignment.mode": "never"}),
    ("pix2pix", {"preprocessing.alignment.validate_declared": False}),
    ("pix2pix", {"preprocessing.alignment.on_failure": "skip_set"}),
    *[
        ("pix2pix", {"preprocessing.filtering.foreground.policy": value})
        for value in _INLINE_CHOICES["foreground policies"]
    ],
    ("pix2pix", {"preprocessing.filtering.white_threshold": 0}),
    ("pix2pix", {"preprocessing.split.unit": "patch", "data.group_validation": "unavailable"}),
    ("pix2pix", {"preprocessing.split.unit": "set"}),
    ("pix2pix", {"preprocessing.split.unit": "specimen"}),
    ("pix2pix", {"preprocessing.split.assignment_file": "splits/frozen.csv"}),
    (
        "pix2pix",
        {
            "preprocessing.split.train": 1.0,
            "preprocessing.split.val": 0.0,
            "preprocessing.split.test": 0.0,
        },
    ),
    *[("pix2pix", {"preprocessing.io.backend": value}) for value in SUPPORTED_IMAGE_BACKENDS],
    ("pix2pix", {"preprocessing.io.tiled": False}),
    ("pix2pix", {"preprocessing.io.max_memory_gb": 8.0}),
    ("pix2pix", {"model.generator.dropout": True, "model.generator.norm": "instance"}),
    ("pix2pix", {"model.discriminator.norm": "batch", "model.discriminator.ndf": 32}),
    ("pix2pix", {"training.seed": None}),
    ("pix2pix", {"training.num_workers": 0}),
    ("pix2pix", {"training.resume": "latest"}),
    ("pix2pix", {"training.resume": "ep049.pth"}),
    ("pix2pix", {"training.scheduler": {"name": "linear_decay", "decay_start_epoch": 50}}),
    *[
        (
            "pix2pix",
            {
                "training.scheduler": {
                    "name": "reduce_on_plateau",
                    "monitor": metric,
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 0.00002,
                }
            },
        )
        for metric in sorted(_PIX2PIX_CHECKPOINT_METRICS)
    ],
    ("pix2pix", {"training.early_stopping": {"monitor": "val_ssim__HE", "patience": 15}}),
    ("pix2pix", {"training.early_stopping": {"monitor": "loss_D_val", "min_delta": 0.01}}),
    ("pix2pix", {"training.early_stopping": {"monitor": "loss_val_weighted_generator_l1__HE"}}),
    *[
        (
            "pix2pix",
            {"training.augmentation": {"enabled": True, "expansion_factor": 2, "intensity": value}},
        )
        for value in ("light", "medium", "strong")
    ],
    (
        "pix2pix",
        {"training.augmentation": {"intensity": "medium", "photometric_inputs": ["AF", "LF"]}},
    ),
    ("pix2pix", {"training.augmentation": {"intensity": "strong", "photometric_inputs": []}}),
    (
        "pix2pix",
        {
            "preprocessing.inputs.target_modalities": ["HE", "PAS"],
            "model.outputs": ["PAS", "HE"],
            "inference.checkpoint_metric": "val_ssim__PAS",
            "training.early_stopping": {"monitor": "val_ssim__HE"},
        },
    ),
    (
        "pix2pix",
        {"preprocessing.inputs.target_modalities": ["HE", "PAS"], "model.inputs": ["LF"]},
    ),
    ("pix2pix", {f"{_L1}.enabled": False}),
    ("pix2pix", {f"{_L1}.weight": 0.0}),
    ("pix2pix", {f"{_L1}.params": {"reduction": "none"}}),
    ("pix2pix", {f"{_L1}.params": {"mask": {"enabled": True, "background_weight": 0.0}}}),
    (
        "pix2pix",
        {
            "training.losses.generator.2": {
                "name": "ssim",
                "weight": 1.0,
                "params": _SSIM_FULL_PARAMS,
            }
        },
    ),
    ("pix2pix", {"training.losses.generator.2": {"name": "ssim", "weight": 1.0}}),
    ("pix2pix", {f"{_L1}.schedule": {"type": "constant"}}),
    *[
        ("pix2pix", {f"{_L1}.schedule": {"type": kind, "start_epoch": 5, "end_epoch": 20}})
        for kind in ("linear_warmup", "linear_decay", "cosine")
    ],
    ("pix2pix", {f"{_L1}.schedule": {"type": "step", "epoch": 50, "factor": 0.5}}),
    *[
        ("pix2pix", {f"{_L1}.schedule": {"type": kind, "epoch": 20}})
        for kind in ("turn_on_after_epoch", "turn_off_after_epoch")
    ],
    ("pix2pix", {"inference": {"checkpoint_policy": "latest"}}),
    (
        "pix2pix",
        {
            "inference": {
                "checkpoint_policy": "top_k",
                "checkpoint_metric": "val_mae__HE",
                "checkpoint_rank": 2,
                "output_dir": "somewhere/output_test",
            }
        },
    ),
    ("pix2pix", {"inference": {"checkpoint_path": "checkpoints/ep099.pth"}}),
    (
        "pix2pix",
        {
            "evaluation": {
                "protocol": "paired",
                "save_graphs": True,
                "generated_dir": "gen",
                "output_dir": "eval",
                "bootstrap_iterations": 0,
                "bootstrap_seed": 7,
            }
        },
    ),
    ("cyclegan", {"method.replay_buffer_size": 0}),
    ("cyclegan", {"data.hash_policy": "membership"}),
    ("cyclegan", {"data.group_metadata": "domains/groups.csv", "data.group_validation": "auto"}),
    *[
        ("cyclegan", {"data.group_metadata": "domains/groups.csv", "data.group_validation": unit})
        for unit in ("patient", "specimen", "set")
    ],
    ("cyclegan", {"data.domains.label_free": "raw/{split}/label_free"}),
    ("cyclegan", {"image_size": [320, 256]}),
    ("cyclegan", {"model.generator.blocks": 6, "model.generator.base_channels": 32}),
    ("cyclegan", {"training.scheduler": {"name": "none"}}),
    (
        "cyclegan",
        {"training.scheduler": {"name": "reduce_on_plateau", "monitor": "loss_G_val"}},
    ),
    ("cyclegan", {"training.early_stopping": {"monitor": "loss_G_val"}}),
    ("cyclegan", {"training.early_stopping": {"monitor": "loss_val_total_generator"}}),
    ("cyclegan", {"training.losses.generator.2.enabled": False}),
    (
        "cyclegan",
        {"training.losses.generator.2.schedule": {"type": "turn_off_after_epoch", "epoch": 100}},
    ),
    ("cyclegan", {"inference.direction": "B_to_A"}),
    ("cyclegan", {"inference": {"checkpoint_policy": "latest", "direction": "A_to_B"}}),
    (
        "cyclegan",
        {
            "inference": {
                "checkpoint_policy": "top_k",
                "checkpoint_metric": "loss_G_val",
                "checkpoint_rank": 2,
            }
        },
    ),
    ("cyclegan", {"evaluation.protocol": "paired"}),
    (
        "cyclegan",
        {
            "evaluation.protocol": "paired",
            "evaluation.metrics": [{"name": "mae"}],
            "evaluation.input_failures": "permissive",
        },
    ),
    ("pix2pix", {"evaluation.metrics": [{"name": "ssim"}, {"name": "pcc_r"}, {"name": "mae"}]}),
    ("pix2pix", {"evaluation.input_failures": "permissive"}),
    (
        "pix2pix",
        {"evaluation.protocol": "unpaired", "evaluation.reference_collection": "real/{split}"},
    ),
    ("cyclegan", {"evaluation.reference_collection": "held_out/stained"}),
]


@pytest.mark.parametrize(("method_name", "updates"), _VALID_VARIANTS)
def test_valid_choice_variant_parses_and_resolves(
    tmp_path: Path, method_name: str, updates: dict[str, Any]
) -> None:
    resolved = _parse(tmp_path, _variant(method_name, updates)).to_dict()

    for dotted, value in updates.items():
        if value is None or not isinstance(value, str | int | float | bool | list):
            continue
        assert _get(resolved, dotted) == value, dotted


def test_optional_sections_resolve_documented_defaults(tmp_path: Path) -> None:
    config = _parse(
        tmp_path,
        _variant(
            "pix2pix",
            {
                "training.early_stopping": {"monitor": "val_mae__HE"},
                "training.scheduler": {"name": "reduce_on_plateau", "monitor": "val_psnr__HE"},
            },
        ),
    )
    assert config.training is not None
    assert config.training.early_stopping is not None
    assert config.training.early_stopping.mode == "min"
    assert config.method is not None
    assert config.method.options.training.scheduler.mode == "max"

    cyclegan = _load(_FULL["cyclegan"])
    del cyclegan["method"]["replay_buffer_size"]
    del cyclegan["inference"]["direction"]
    del cyclegan["evaluation"]["protocol"]
    resolved = _parse(tmp_path, cyclegan)
    assert resolved.method is not None
    assert resolved.method.options.replay_buffer_size == 50
    assert resolved.method.options.generator.options["blocks"] == 9
    assert resolved.inference is not None and resolved.inference.direction is None
    assert resolved.evaluation is not None and resolved.evaluation.protocol is None


# --- 6. Invalid combinations the examples teach --------------------------------

_OTHER_DOMAINS = {"label_free": "domains/label_free", "stained": "domains/stained"}
_INVALID_VARIANTS: list[tuple[str, dict[str, Any], str]] = [
    (
        "pix2pix",
        {"data.pairing": "unpaired", "data.domains": _OTHER_DOMAINS},
        "requires data.pairing='paired'",
    ),
    (
        "cyclegan",
        {"data.pairing": "paired", "data.domains": {}},
        "requires data.pairing='unpaired'",
    ),
    ("cyclegan", {"data.domains": {"label_free": "a", "he": "b"}}, "data.domains keys must be"),
    ("pix2pix", {"data.group_metadata": "groups.csv"}, "supported only with data.pairing"),
    ("pix2pix", {"data.hash_policy": "sha256"}, "data.hash_policy"),
    ("pix2pix", {"data.group_validation": "none"}, "data.group_validation"),
    ("pix2pix", {"method.replay_buffer_size": 0}, "Unknown key.*in method: replay_buffer_size"),
    ("cyclegan", {"method.replay_buffer_size": -1}, ">= 0"),
    ("cyclegan", {"model.generator": {"architecture": "concat_unet"}}, "architecture='resnet'"),
    ("pix2pix", {"model.generator": {"architecture": "resnet"}}, "architecture='concat_unet'"),
    ("pix2pix", {"model.generator.blocks": 9}, "Unknown key"),
    ("cyclegan", {"model.generator.dropout": False}, "Unknown key"),
    ("cyclegan", {"model.generator.norm": "batch"}, "must be 'instance'"),
    ("cyclegan", {"image_size": [250, 256]}, "multiples of 4"),
    ("pix2pix", {"model.generator.bilinear": True}, "bilinear=True is not supported"),
    ("pix2pix", {"model.discriminator.use_sigmoid": True}, "use_sigmoid=True"),
    ("pix2pix", {"model.target": "HE"}, "model.target is not part of the current schema"),
    (
        "pix2pix",
        {"preprocessing.inputs.target_modality": "HE"},
        "target_modality is not part of the current schema",
    ),
    (
        "pix2pix",
        {"model.outputs": ["PAS"], "inference.checkpoint_metric": "loss_G_val"},
        "not in preprocessing.inputs.target_modalities",
    ),
    ("pix2pix", {"model.inputs": ["AF", "XX"]}, "not in preprocessing.inputs.modalities"),
    ("pix2pix", {"model.outputs": []}, "at least one name"),
    ("pix2pix", {"model.outputs": "HE"}, "sequence of names"),
    ("pix2pix", {"model.outputs": ["HE", "HE"]}, "duplicate names"),
    ("pix2pix", {"model.outputs": ["H&E"]}, "invalid identifiers"),
    ("pix2pix", {"model.outputs": ["LF"]}, "must be disjoint"),
    ("pix2pix", {"preprocessing.inputs.target_modalities": ["H&E"]}, "invalid identifiers"),
    ("pix2pix", {"preprocessing.inputs.target_modalities": ["LF"]}, "must differ"),
    (
        "pix2pix",
        {
            "preprocessing.inputs.target_modalities": ["HE", "PAS"],
            "model.outputs": ["HE", "PAS"],
            "training.early_stopping": {"patience": 5},
        },
        "early_stopping.monitor is required",
    ),
    ("pix2pix", {"inference.checkpoint_metric": "val_ssim__PAS"}, "names output 'PAS'"),
    (
        "pix2pix",
        {
            "preprocessing.inputs.target_modalities": ["HE", "PAS"],
            "model.outputs": ["HE", "PAS"],
            "training.early_stopping": {"monitor": "loss_G_val"},
            "evaluation": {"protocol": "unpaired", "reference_collection": "real"},
        },
        "several simultaneous outputs",
    ),
    (
        "pix2pix",
        {"training.augmentation": {"intensity": "light", "photometric_inputs": ["LF"]}},
        "must be empty for intensity='light'",
    ),
    (
        "pix2pix",
        {"training.augmentation": {"intensity": "medium", "photometric_inputs": ["HE"]}},
        "not selected model.inputs",
    ),
    (
        "pix2pix",
        {"training.augmentation": {"intensity": "medium", "photometric_inputs": ["LF", "LF"]}},
        "duplicate names",
    ),
    (
        "pix2pix",
        {
            "training.augmentation": {
                "intensity": "medium",
                "photometric_inputs": ["foreground_mask"],
            }
        },
        "not selected model.inputs",
    ),
    (
        "pix2pix",
        {"image_size": [320, 256], "training.augmentation.enabled": True},
        "requires a square image_size",
    ),
    ("cyclegan", {"model.outputs": ["stained", "other"]}, "exactly one model.outputs entry"),
    ("cyclegan", {"training.augmentation.enabled": True}, "augmentation.enabled=false"),
    ("pix2pix", {"inference.direction": "A_to_B"}, "not supported by method.name='pix2pix'"),
    ("cyclegan", {"inference.direction": "sideways"}, r"direction must be one of \['A_to_B'"),
    ("pix2pix", {"evaluation.reference_collection": "real"}, "unpaired protocol only"),
    (
        "pix2pix",
        {"evaluation.protocol": "unpaired", "evaluation.reference_collection": ""},
        "non-empty path or pattern",
    ),
    (
        "pix2pix",
        {
            "evaluation.protocol": "unpaired",
            "evaluation.reference_collection": "real",
            "evaluation.metrics": [{"name": "mae"}],
        },
        "paired protocol only",
    ),
    (
        "pix2pix",
        {
            "evaluation.protocol": "unpaired",
            "evaluation.reference_collection": "real",
            "evaluation.input_failures": "permissive",
        },
        "paired protocol only",
    ),
    (
        "cyclegan",
        {"evaluation.protocol": "paired", "evaluation.reference_collection": "real"},
        "unpaired protocol only",
    ),
    ("pix2pix", {"evaluation.metrics": [{"name": "fid"}]}, "not a registered metric"),
    ("pix2pix", {"evaluation.metrics": [{"name": "mae"}, {"name": "mae"}]}, "more than once"),
    ("pix2pix", {"evaluation.metrics": [{"name": "ssim", "options": {"win": 3}}]}, "unknown"),
    ("pix2pix", {"evaluation.metrics": [{"name": "ssim", "weight": 1}]}, "unknown keys"),
    ("pix2pix", {"evaluation.metrics": [{"name": "ssim", "options": [3]}]}, "mapping"),
    ("pix2pix", {"evaluation.metrics": ["ssim"]}, "must be a mapping"),
    ("pix2pix", {"evaluation.metrics": "ssim"}, "list of mappings"),
    ("pix2pix", {"evaluation.metrics": []}, "at least one metric"),
    ("pix2pix", {"evaluation.input_failures": "skip"}, "input_failures"),
    ("cyclegan", {"evaluation.metrics": [{"name": "mae"}]}, "paired protocol only"),
    ("cyclegan", {"evaluation.input_failures": "permissive"}, "paired protocol only"),
    ("pix2pix", {"inference.checkpoint_metric": "val_loss"}, "not a checkpoint metric"),
    ("pix2pix", {"inference.checkpoint_metric": "val_ssim"}, "val_<metric>__<output>"),
    ("pix2pix", {"method.options": {}}, "Unknown key.*in method: options"),
    ("pix2pix", {"method.name": "stylegan"}, "not a registered method definition"),
    ("cyclegan", {"inference.checkpoint_metric": "val_ssim"}, "paired image-fidelity metric"),
    (
        "cyclegan",
        {"training.scheduler": {"name": "reduce_on_plateau", "monitor": "val_ssim"}},
        "paired image-fidelity metric",
    ),
    (
        "cyclegan",
        {"training.early_stopping": {"monitor": "val_psnr"}},
        "paired image-fidelity metric",
    ),
    ("pix2pix", {"training.early_stopping": {"monitor": "val_loss"}}, "not a checkpoint metric"),
    ("pix2pix", {"training.scheduler": {"name": "linear_decay"}}, "decay_start_epoch is required"),
    (
        "pix2pix",
        {"training.scheduler": {"name": "linear_decay", "decay_start_epoch": 100}},
        "less than epochs",
    ),
    ("pix2pix", {"preprocessing.split.test": 0.2}, "sum to 1"),
    ("pix2pix", {"preprocessing.split.unit": "slide"}, "split.unit"),
    ("pix2pix", {"preprocessing.masks.generation": "sometimes"}, "masks.generation"),
    ("pix2pix", {"preprocessing.masks.strategy": "otsu"}, "masks.strategy"),
    ("pix2pix", {"preprocessing.masks.scale": 0}, "masks.scale"),
    ("pix2pix", {"preprocessing.alignment.method": "elastic"}, "affine_sift"),
    ("pix2pix", {"preprocessing.alignment.on_failure": "warn"}, "on_failure"),
    ("pix2pix", {"preprocessing.filtering.foreground.policy": "any"}, "policy is invalid"),
    ("pix2pix", {"preprocessing.filtering.white_threshold": 256}, "white_threshold"),
    ("pix2pix", {"preprocessing.io.backend": "vips"}, "io.backend"),
    ("pix2pix", {"preprocessing.inputs.reference": "H&E"}, "inputs.reference"),
    ("pix2pix", {"preprocessing.inputs.hash_verification": "never"}, "hash_verification"),
    (
        "pix2pix",
        {"inference": {"checkpoint_policy": "latest", "checkpoint_rank": 2}},
        "checkpoint_rank is supported only",
    ),
    ("pix2pix", {"inference": {"checkpoint_policy": "best"}}, "checkpoint_metric is required"),
    ("pix2pix", {"inference.checkpoint_rank": 0}, "greater than 0"),
    ("pix2pix", {"inference.checkpoint_policy": "median"}, "Unknown checkpoint_policy"),
    ("pix2pix", {"training.losses.generator.2": {"name": "l1", "weight": 1.0}}, "Duplicate"),
    (
        "pix2pix",
        {"training.losses.discriminator.1": {"name": "l1", "weight": 1.0}},
        "supported only in losses.generator",
    ),
    (
        "pix2pix",
        {"training.losses.generator.2": {"name": "cycle_l1", "weight": 1.0}},
        "not supported by method.name='pix2pix'",
    ),
    ("cyclegan", {"training.losses.generator.1.weight": 0.0}, "requires an active"),
    ("cyclegan", {"training.losses.generator.1.params": {"reduction": "sum"}}, "Unknown key"),
    ("pix2pix", {f"{_L1}.params": {"window_size": 11}}, "Unknown key"),
    ("pix2pix", {f"{_L1}.params": {"mask": {"source": "tissue"}}}, "source"),
    (
        "pix2pix",
        {
            "training.losses.generator.2": {
                "name": "ssim",
                "weight": 1.0,
                "params": {"window_size": 4},
            }
        },
        "positive odd integer",
    ),
    ("pix2pix", {f"{_L1}.schedule": {"type": "linear_warmup"}}, "requires end_epoch"),
    ("pix2pix", {f"{_L1}.schedule": {"type": "step", "factor": 0.5}}, "requires epoch"),
]


@pytest.mark.parametrize(("method_name", "updates", "message"), _INVALID_VARIANTS)
def test_invalid_combination_is_rejected(
    tmp_path: Path, method_name: str, updates: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _parse(tmp_path, _variant(method_name, updates))


# --- Queue and ablation references ---------------------------------------------


def _write_queue(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _queue_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "variant",
        "jobs": [
            {"config_path": str(_MINIMAL["pix2pix"])},
            {"config_path": str(_RUNS / "example_ablation_ssim.yaml")},
        ],
    }
    data.update(overrides)
    return data


def test_example_queue_parses_and_preflights() -> None:
    queue = _load_local_run_queue(_QUEUES / "example.yaml")

    assert queue.continue_on_failure is False
    assert queue.ablation is None
    assert [job.config_path for job in queue.jobs] == [
        _FULL["pix2pix"].resolve(),
        _FULL["cyclegan"].resolve(),
    ]
    assert queue.jobs[0].stages is None
    assert queue.jobs[1].stages == ("train", "infer", "evaluate")
    assert all(job.label and job.notes for job in queue.jobs)
    assert len(_preflight_run_configs(queue)) == 2


def test_example_ablation_queue_preflights_declared_difference() -> None:
    queue = _load_local_run_queue(_QUEUES / "example_ablation.yaml")
    assert queue.ablation is not None

    summary = _build_ablation_summary(queue, _preflight_run_configs(queue))

    assert summary["variable_fields"] == ["run_name", "training.losses.generator"]
    assert [job["variable_values"]["run_name"] for job in summary["jobs"]] == [
        "example_run",
        "example_run_ssim",
    ]
    assert summary["fixed_values"]["training.seed"] == 42


@pytest.mark.parametrize("stages", [[stage] for stage in VALID_STAGES] + [list(VALID_STAGES)])
def test_queue_accepts_each_stage(tmp_path: Path, stages: list[str]) -> None:
    data = _queue_data(continue_on_failure=True)
    data["jobs"][0]["stages"] = stages

    queue = _load_local_run_queue(_write_queue(tmp_path, data))

    assert queue.continue_on_failure is True
    assert queue.jobs[0].stages == tuple(stages)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"jobs": [{"config_path": "a.yaml", "stages": []}]}, "non-empty list"),
        ({"jobs": [{"config_path": "a.yaml", "stages": "train"}]}, "non-empty list"),
        ({"jobs": [{"config_path": "a.yaml", "stages": ["deploy"]}]}, "unknown stage"),
        ({"jobs": []}, "non-empty 'jobs'"),
        ({"continue_on_failure": "yes"}, "YAML boolean"),
        (
            {"ablation": {"fixed_fields": ["run_name"], "variable_fields": ["run_name"]}},
            "both fixed and variable",
        ),
        ({"ablation": {"fixed_fields": ["model"], "variable_fields": []}}, "non-empty list"),
        ({"ablation": {"fixed_fields": ["model"]}}, "non-empty list"),
    ],
)
def test_queue_rejects_malformed_fields(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _load_local_run_queue(_write_queue(tmp_path, _queue_data(**overrides)))


@pytest.mark.parametrize(
    ("ablation", "message"),
    [
        ({"variable_fields": ["run_name"]}, "undeclared config difference"),
        (
            {"fixed_fields": ["training.losses"], "variable_fields": ["run_name"]},
            "fixed field",
        ),
    ],
)
def test_ablation_preflight_rejects_undeclared_differences(
    tmp_path: Path, ablation: dict[str, Any], message: str
) -> None:
    queue = _load_local_run_queue(_write_queue(tmp_path, _queue_data(ablation=ablation)))

    with pytest.raises(QueueAblationError, match=message):
        _build_ablation_summary(queue, _preflight_run_configs(queue))


@pytest.mark.parametrize("name", ["prepare", "train", "infer", "evaluate", "train_infer"])
def test_minimal_operation_examples_inspect_with_declared_stages(name: str) -> None:
    from virtual_staining.applications.config_authoring import inspect_run_yaml

    stages = tuple(name.split("_"))
    inspection = inspect_run_yaml(_RUNS / f"minimal_{name}.yaml", stages=stages)
    assert inspection.config.stages == stages
    assert inspection.resolved_yaml == inspection.config.resolved_yaml()
    if name == "prepare":
        assert not {"method", "model", "run_name", "results_path"} & inspection.resolved.keys()
    if name == "evaluate":
        assert set(inspection.resolved["model"]) == {"inputs", "outputs"}


@pytest.mark.parametrize("name", ["operations", "full"])
def test_operation_and_full_queue_examples_preflight(name: str) -> None:
    queue = _load_local_run_queue(_QUEUES / f"example_{name}.yaml")
    configs = _preflight_run_configs(queue)
    expected = [(stage,) for stage in VALID_STAGES] if name == "operations" else [VALID_STAGES] * 2
    assert [config.stages for config in configs] == expected
