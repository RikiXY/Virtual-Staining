from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from nicegui.client import Client
from nicegui.page import page

from virtual_staining.applications.api import ApplicationService
from virtual_staining.ui.training_config_page import build_training_config_page


def test_training_config_page_starts_with_a_valid_minimal_preview(tmp_path: Path) -> None:
    service = ApplicationService(
        tmp_path / "checkpoints",
        tmp_path / "outputs",
        tmp_path / "results",
        training_config_directory=tmp_path / "configs",
    )

    with Client(page("/training-config-test")) as client:
        build_training_config_page(service)

    preview = next(
        element for element in client.elements.values() if "vs-yaml-preview" in element._classes
    )
    steps = [
        element for element in client.elements.values() if "vs-config-step" in element._classes
    ]
    buttons = {
        str(cast(Any, element).text): element
        for element in client.elements.values()
        if element.tag == "q-btn"
        and getattr(element, "text", None) in {"Download YAML", "Save on server"}
    }
    inputs = {
        element._props.get("label")
        for element in client.elements.values()
        if element._props.get("label") is not None
    }

    assert len(steps) == 3
    assert set(buttons) == {"Download YAML", "Save on server"}
    assert cast(Any, buttons["Download YAML"]).enabled is True
    assert cast(Any, buttons["Save on server"]).enabled is True
    assert {
        "Run name *",
        "Prepared dataset root *",
        "Results directory *",
        "Input modalities *",
        "Target modality *",
        "Epochs *",
        "Generator adversarial weight *",
        "L1 reconstruction weight *",
        "Discriminator weight *",
    } <= inputs
    preview_content = str(cast(Any, preview).content)
    assert "dataset_root:" in preview_content
    assert "model:" in preview_content
    assert "training:" in preview_content
    assert "losses:" in preview_content
    assert "preprocessing:" not in preview_content
    assert "inference:" not in preview_content
    assert "evaluation:" not in preview_content
    assert "queue:" not in preview_content
