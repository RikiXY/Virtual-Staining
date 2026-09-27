from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.config.training import TrainingConfig


class TrainingConfigError(ValueError):
    """A training configuration could not be validated or saved."""


@dataclass(frozen=True)
class TrainingConfigDraft:
    """Small, UI-facing input model for a training-only run configuration."""

    run_name: str
    dataset_root: str
    results_path: str
    input_modalities: tuple[str, ...]
    target_modality: str
    epochs: int
    generator_adversarial_weight: float = 1.0
    reconstruction_weight: float = 25.0
    discriminator_adversarial_weight: float = 1.0


@dataclass(frozen=True)
class TrainingConfigDocument:
    filename: str
    yaml_text: str


@dataclass(frozen=True)
class SavedTrainingConfig:
    path: Path
    document: TrainingConfigDocument


def build_training_config(draft: TrainingConfigDraft) -> TrainingConfigDocument:
    """Validate a draft and render the smallest useful training-only YAML document."""
    run_name = draft.run_name.strip()
    dataset_root = draft.dataset_root.strip()
    results_path = draft.results_path.strip()
    target_modality = draft.target_modality.strip()
    input_modalities = tuple(name.strip() for name in draft.input_modalities if name.strip())

    if not dataset_root:
        raise TrainingConfigError("Dataset root is required.")
    if not results_path:
        raise TrainingConfigError("Results path is required.")
    if not input_modalities:
        raise TrainingConfigError("Add at least one input modality.")
    if len(set(input_modalities)) != len(input_modalities):
        raise TrainingConfigError("Input modalities must be unique.")
    if not target_modality:
        raise TrainingConfigError("Target modality is required.")

    data: dict[str, Any] = {
        "dataset_root": dataset_root,
        "results_path": results_path,
        "run_name": run_name,
        "model": {
            "inputs": list(input_modalities),
            "target": target_modality,
        },
        "training": {
            "epochs": draft.epochs,
            "losses": {
                "generator": [
                    {
                        "name": "adversarial_bce",
                        "weight": draft.generator_adversarial_weight,
                    },
                    {"name": "l1", "weight": draft.reconstruction_weight},
                ],
                "discriminator": [
                    {
                        "name": "adversarial_bce",
                        "weight": draft.discriminator_adversarial_weight,
                    }
                ],
            },
        },
    }

    try:
        config = RunConfig(
            project=ProjectConfig.from_mapping(data),
            model=ModelConfig.from_mapping(data["model"]),
            training=TrainingConfig.from_mapping(data["training"]),
            preprocessing=None,
            inference=None,
            evaluation=None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigError(str(exc)) from exc

    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if config.training is None:  # pragma: no cover - guarded by construction above
        raise TrainingConfigError("Training configuration is missing.")
    return TrainingConfigDocument(filename=f"{_filename_stem(run_name)}.yaml", yaml_text=yaml_text)


def save_training_config(
    draft: TrainingConfigDraft,
    directory: Path,
) -> SavedTrainingConfig:
    """Save a validated document under ``directory`` without replacing an existing file."""
    document = build_training_config(draft)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        destination = _available_path(directory, document.filename)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(document.yaml_text)
    except OSError as exc:
        raise TrainingConfigError(f"Could not save the YAML file: {exc}") from exc
    return SavedTrainingConfig(path=destination, document=document)


def _filename_stem(run_name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name.strip()).strip("-_").lower()
    return stem or "training-run"


def _available_path(directory: Path, filename: str) -> Path:
    first = directory / filename
    if not first.exists():
        return first
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 2
    while (candidate := directory / f"{stem}_{index}{suffix}").exists():
        index += 1
    return candidate
