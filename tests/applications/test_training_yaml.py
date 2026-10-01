from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    TrainingConfigDraft,
)
from virtual_staining.config.run import RunConfig


def _draft(**changes: object) -> TrainingConfigDraft:
    draft = TrainingConfigDraft(
        run_name="LF to H&E baseline",
        dataset_root="local_workspace/datasets/sample",
        results_path="local_workspace/results",
        input_modalities=("autofluorescence", "label_free"),
        target_modality="HE",
        epochs=30,
        generator_adversarial_weight=1.0,
        reconstruction_weight=25.0,
        discriminator_adversarial_weight=1.0,
    )
    return replace(draft, **changes)


def _service(tmp_path: Path) -> ApplicationService:
    return ApplicationService(
        Path("checkpoints"),
        Path("outputs"),
        Path("results"),
        training_config_directory=Path("configs"),
        working_directory=tmp_path,
    )


def test_preview_builds_minimal_training_only_yaml(tmp_path: Path) -> None:
    document = _service(tmp_path).preview_training_config(_draft())
    data = yaml.safe_load(document.yaml_text)

    assert document.filename == "lf-to-h-e-baseline.yaml"
    assert set(data) == {"dataset_root", "results_path", "run_name", "model", "training"}
    assert data["model"] == {
        "inputs": ["autofluorescence", "label_free"],
        "outputs": ["HE"],
    }
    assert data["training"] == {
        "epochs": 30,
        "losses": {
            "generator": [
                {"name": "adversarial_bce", "weight": 1.0},
                {"name": "l1", "weight": 25.0},
            ],
            "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
        },
    }
    assert not {"preprocessing", "inference", "evaluation", "queue"} & set(data)


def test_preview_is_accepted_by_the_training_cli_schema(tmp_path: Path) -> None:
    document = _service(tmp_path).preview_training_config(_draft())
    config_path = tmp_path / document.filename
    config_path.write_text(document.yaml_text, encoding="utf-8")

    config = RunConfig.from_yaml(config_path)

    assert config.training is not None
    assert config.training.epochs == 30
    assert config.model.inputs == ("autofluorescence", "label_free")
    assert config.preprocessing is None
    assert config.inference is None
    assert config.evaluation is None


def test_save_never_overwrites_an_existing_config(tmp_path: Path) -> None:
    service = _service(tmp_path)

    first = service.save_training_config(_draft())
    second = service.save_training_config(_draft())

    assert first.path == tmp_path / "configs" / "lf-to-h-e-baseline.yaml"
    assert second.path == tmp_path / "configs" / "lf-to-h-e-baseline_2.yaml"
    assert first.path.read_text(encoding="utf-8") == first.document.yaml_text
    assert second.path.read_text(encoding="utf-8") == second.document.yaml_text


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_name": " "}, "run_name"),
        ({"dataset_root": " "}, "Dataset root"),
        ({"input_modalities": ()}, "input modality"),
        ({"input_modalities": ("label_free", "label_free")}, "unique"),
        ({"target_modality": " "}, "Target modality"),
        ({"epochs": 0}, "epochs"),
        ({"reconstruction_weight": -1.0}, "greater than or equal"),
    ],
)
def test_invalid_drafts_are_rejected(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ApplicationError, match=message):
        _service(tmp_path).preview_training_config(_draft(**changes))
