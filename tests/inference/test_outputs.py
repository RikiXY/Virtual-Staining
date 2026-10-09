from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.utils import save_image

from tests.manifest_helpers import make_manifest_record
from virtual_staining.inference import outputs
from virtual_staining.inference.outputs import generated_path_for_record, save_rgb
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS


def test_generated_path_for_record_uses_sample_id_output_and_domain_suffix(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        "00512_09216",
        "test",
        target_paths={"HE": Path("t/he.TIF"), "PAS": Path("t/pas.png")},
    )

    assert generated_path_for_record(record, tmp_path, "HE") == (
        tmp_path / "HE" / "00512_09216_generated.tif"
    )
    assert generated_path_for_record(record, tmp_path, "PAS") == (
        tmp_path / "PAS" / "00512_09216_generated.png"
    )
    # CycleGAN B_to_A predicts an input domain: the suffix comes from that input.
    assert generated_path_for_record(record, tmp_path, "label_free").parent.name == "label_free"


def test_generated_path_for_record_rejects_unknown_domain(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no domain"):
        generated_path_for_record(make_manifest_record(), tmp_path, "PAS")


@pytest.mark.parametrize("suffix", sorted(VALID_IMAGE_EXTENSIONS))
def test_save_rgb_preserves_encoded_bytes_and_overwrites(tmp_path: Path, suffix: str) -> None:
    destination = tmp_path / "outputs" / f"image{suffix}"
    reference = tmp_path / f"reference{suffix}"
    image = torch.linspace(0, 1, 3 * 8 * 9).reshape(3, 8, 9)
    for output in (image, 1 - image, image):
        save_image(output, reference)
        save_rgb(output, destination)
        assert destination.read_bytes() == reference.read_bytes()
        assert list(destination.parent.iterdir()) == [destination]


def test_overlapping_publications_own_distinct_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "image.png"
    caller_owned = tmp_path / ".image.caller.partial.png"
    caller_owned.write_bytes(b"caller-owned scratch")
    pending: list[Path] = []

    def overlapping_encoder(output: torch.Tensor, path: Path) -> None:
        assert path.parent == destination.parent and path.suffix == destination.suffix
        assert path.is_file() and path.stat().st_size == 0
        assert path not in pending
        pending.append(path)
        save_image(output, path)
        if len(pending) == 1:
            original = path.read_bytes()
            save_rgb(1 - output, destination)
            assert path.read_bytes() == original
            assert not pending[1].exists()

    monkeypatch.setattr(outputs, "save_image", overlapping_encoder)
    save_rgb(torch.zeros(3, 8, 9), destination)
    with Image.open(destination) as image:
        assert image.getpixel((0, 0)) == (0, 0, 0)
    assert len(pending) == 2
    assert set(tmp_path.iterdir()) == {destination, caller_owned}
    assert caller_owned.read_bytes() == b"caller-owned scratch"


@pytest.mark.parametrize("existing", [False, True])
def test_save_rgb_decodes_pixels_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    destination = tmp_path / "image.bmp"
    if existing:
        save_image(torch.ones(3, 8, 9), destination)
    original = destination.read_bytes() if existing else None

    def truncated_encoder(output: torch.Tensor, path: Path) -> None:
        save_image(output, path)
        path.write_bytes(path.read_bytes()[:-8])
        # BMP's verify() accepts this header, but decoding must reject the missing pixels.
        with Image.open(path) as image:
            image.verify()

    monkeypatch.setattr(outputs, "save_image", truncated_encoder)
    with pytest.raises(OSError, match="truncated"):
        save_rgb(torch.zeros(3, 8, 9), destination)
    assert (destination.read_bytes() if existing else None) == original
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])
