from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.config.run import RunConfig

_EXAMPLES = Path(__file__).resolve().parents[2] / "config" / "runs"


@pytest.mark.parametrize(
    ("filename", "method", "pairing", "architecture"),
    [
        ("example.yaml", "pix2pix", "paired", "concat_unet"),
        ("example_cyclegan.yaml", "cyclegan", "unpaired", "resnet"),
    ],
)
def test_repository_example_configs_parse(
    filename: str, method: str, pairing: str, architecture: str
) -> None:
    config = RunConfig.from_yaml(_EXAMPLES / filename)

    assert config.method.name == method
    assert config.data.pairing == pairing
    assert config.model.generator.architecture == architecture
