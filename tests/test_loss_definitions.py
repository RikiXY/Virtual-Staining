from __future__ import annotations

import ast
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from virtual_staining.loss_definitions import (
    LOSS_DEFINITIONS,
    LossMaskConfig,
    LossParams,
    method_loss_names,
    ssim_loss_map,
)

_EXPECTED = {
    # name: (roles, methods, context, param keys)
    "adversarial_bce": ({"generator", "discriminator"}, {"pix2pix"}, "adversarial", set()),
    "l1": ({"generator"}, {"pix2pix"}, "reconstruction", {"reduction", "mask"}),
    "ssim": (
        {"generator"},
        {"pix2pix"},
        "reconstruction",
        {"data_range", "window_size", "sigma", "channel_mode", "reduction", "mask"},
    ),
    "adversarial_lsgan": ({"generator", "discriminator"}, {"cyclegan"}, "adversarial", set()),
    "cycle_l1": ({"generator"}, {"cyclegan"}, "reconstruction", set()),
    "identity_l1": ({"generator"}, {"cyclegan"}, "reconstruction", set()),
}


def _mask_params(**mask: object) -> LossParams:
    return LOSS_DEFINITIONS["l1"].parse_params({"mask": {"enabled": True, **mask}})


def test_exactly_one_definition_per_builtin_loss() -> None:
    assert list(LOSS_DEFINITIONS) == list(_EXPECTED)
    for name, definition in LOSS_DEFINITIONS.items():
        roles, methods, context, keys = _EXPECTED[name]
        assert definition.name == name
        assert (set(definition.roles), set(definition.methods)) == (roles, methods)
        assert (definition.context, set(definition.param_keys)) == (context, keys)
        assert definition.supports_mask == ("mask" in keys)


def test_method_compatibility_is_derived_from_definitions() -> None:
    assert method_loss_names("pix2pix") == {"adversarial_bce", "l1", "ssim"}
    assert method_loss_names("cyclegan") == {"adversarial_lsgan", "cycle_l1", "identity_l1"}


def _loss_name_tables(path: Path) -> list[str]:
    names = set(LOSS_DEFINITIONS)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literal_slices = {id(node.slice) for node in ast.walk(tree) if isinstance(node, ast.Subscript)}
    found = []
    for node in ast.walk(tree):
        if id(node) in literal_slices:
            continue  # typing aliases such as Literal[...]
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            elements = node.elts
        elif isinstance(node, ast.Dict):
            elements = [key for key in node.keys if key is not None]
        else:
            continue
        hits = [e.value for e in elements if isinstance(e, ast.Constant) and e.value in names]
        if len(hits) >= 2:
            found.append(f"{path}:{node.lineno}: {hits}")
    return found


def test_no_parallel_loss_tables_outside_canonical_definitions() -> None:
    package = Path("virtual_staining")
    tables = [
        table
        for path in sorted(package.glob("**/*.py"))
        if path != package / "loss_definitions.py"
        for table in _loss_name_tables(path)
    ]
    assert not tables, "Loss-name tables must derive from LOSS_DEFINITIONS:\n" + "\n".join(tables)
    source = "\n".join(path.read_text() for path in package.glob("**/*.py"))
    for removed in ("LOSS_REGISTRY", "_METHOD_LOSSES", "_GENERATOR_ONLY_LOSSES"):
        assert removed not in source


def test_config_loading_stays_torch_free() -> None:
    code = "import sys, virtual_staining.config.run; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("name", "params", "match"),
    [
        ("adversarial_bce", {"reduction": "mean"}, "Unknown key"),
        ("adversarial_lsgan", {"mask": {}}, "Unknown key"),
        ("cycle_l1", {"reduction": "sum"}, "Unknown key"),
        ("identity_l1", {"mask": {}}, "Unknown key"),
        ("l1", {"sigma": 1.0}, "Unknown key"),
        ("ssim", {"weight": 1.0}, "Unknown key"),
        ("l1", {"reduction": "median"}, "reduction must be one of"),
        ("ssim", {"channel_mode": "hsv"}, "channel_mode must be one of"),
        ("ssim", {"data_range": 0}, "data_range must be greater than 0"),
        ("ssim", {"data_range": math.nan}, "data_range must be a finite number"),
        ("ssim", {"data_range": math.inf}, "data_range must be a finite number"),
        ("ssim", {"sigma": -1.0}, "sigma must be greater than 0"),
        ("ssim", {"sigma": math.nan}, "sigma must be a finite number"),
        ("ssim", {"sigma": -math.inf}, "sigma must be a finite number"),
        ("ssim", {"window_size": 4}, "positive odd integer"),
        ("ssim", {"window_size": math.inf}, "window_size must be a finite number"),
        ("l1", {"mask": {"foreground_weight": math.nan}}, "foreground_weight must be a finite"),
        ("l1", {"mask": {"background_weight": math.inf}}, "background_weight must be a finite"),
        ("ssim", {"mask": {"foreground_weight": -1.0}}, "greater than or equal to 0"),
        ("l1", {"mask": {"source": "tissue"}}, "source must be one of"),
        ("l1", {"mask": {"threshold": 0.5}}, "Unknown key"),
    ],
)
def test_parse_params_rejects_invalid_values(
    name: str, params: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        LOSS_DEFINITIONS[name].parse_params(params)


def test_parse_params_resolves_defaults_without_normalizing() -> None:
    assert LOSS_DEFINITIONS["ssim"].parse_params({}) == LossParams()
    params = LOSS_DEFINITIONS["ssim"].parse_params(
        {"data_range": 2, "window_size": 7, "sigma": 0.5, "channel_mode": "gray"}
    )
    assert (params.data_range, params.window_size, params.sigma) == (2.0, 7, 0.5)
    assert params.channel_mode == "gray" and params.mask == LossMaskConfig()


def test_adversarial_primitives_match_reference_formulas() -> None:
    logits = torch.tensor([[-2.0, 0.0], [0.5, 3.0]])
    bce, lsgan = LOSS_DEFINITIONS["adversarial_bce"], LOSS_DEFINITIONS["adversarial_lsgan"]

    assert bce.adversarial_loss(logits, target_is_real=True).item() == pytest.approx(
        F.softplus(-logits).mean().item()
    )
    assert bce.adversarial_loss(logits, target_is_real=False).item() == pytest.approx(
        F.softplus(logits).mean().item()
    )
    # mean((x - 1)^2) and mean(x^2) for [-2, 0, 0.5, 3]
    assert lsgan.adversarial_loss(logits, target_is_real=True).item() == pytest.approx(
        (9 + 1 + 0.25 + 4) / 4
    )
    assert lsgan.adversarial_loss(logits, target_is_real=False).item() == pytest.approx(
        (4 + 0 + 0.25 + 9) / 4
    )
    with pytest.raises(ValueError, match="not a reconstruction loss"):
        bce.reconstruction_loss(logits, logits)
    with pytest.raises(ValueError, match="not an adversarial loss"):
        LOSS_DEFINITIONS["l1"].adversarial_loss(logits, target_is_real=True)


@pytest.mark.parametrize("name", ["l1", "cycle_l1", "identity_l1"])
def test_l1_primitives_share_mean_absolute_error(name: str) -> None:
    prediction = torch.tensor([[[[1.0, -1.0], [0.5, 0.0]]]])
    target = torch.zeros_like(prediction)

    loss = LOSS_DEFINITIONS[name].reconstruction_loss(prediction, target)

    assert loss.item() == pytest.approx(2.5 / 4)


def test_l1_reductions_keep_torch_semantics() -> None:
    prediction = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    l1 = LOSS_DEFINITIONS["l1"]
    for reduction in ("mean", "sum", "none"):
        params = l1.parse_params({"reduction": reduction})
        expected = F.l1_loss(prediction, target, reduction=reduction)
        assert torch.equal(l1.reconstruction_loss(prediction, target, params), expected)


def test_ssim_reductions_channel_mode_and_window_contract() -> None:
    generator = torch.Generator().manual_seed(0)
    prediction = torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1
    target = torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1
    ssim = LOSS_DEFINITIONS["ssim"]

    per_sample = ssim.reconstruction_loss(
        prediction, target, ssim.parse_params({"reduction": "none"})
    )
    assert per_sample.shape == (2,)
    assert ssim.reconstruction_loss(prediction, target).item() == pytest.approx(
        per_sample.mean().item()
    )
    summed = ssim.reconstruction_loss(prediction, target, ssim.parse_params({"reduction": "sum"}))
    assert summed.item() == pytest.approx(per_sample.sum().item())
    assert ssim.reconstruction_loss(target, target).item() == pytest.approx(0.0, abs=1e-6)

    gray = ssim_loss_map(prediction, target, ssim.parse_params({"channel_mode": "gray"}))
    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray_prediction = (prediction * weights).sum(1, keepdim=True) + (weights.sum() - 1)
    gray_target = (target * weights).sum(1, keepdim=True) + (weights.sum() - 1)
    assert gray.shape == (2, 1, 16, 16)
    assert torch.allclose(gray, ssim_loss_map(gray_prediction, gray_target), atol=1e-6)

    with pytest.raises(ValueError, match="at least window_size=11"):
        ssim.reconstruction_loss(prediction[..., :8, :8], target[..., :8, :8])


def test_reconstruction_mask_is_required_when_enabled() -> None:
    image = torch.zeros(1, 1, 2, 2)
    for name in ("l1", "ssim"):
        params = LOSS_DEFINITIONS[name].parse_params({"mask": {"enabled": True}})
        for masks in (None, {}):
            with pytest.raises(ValueError, match=f"loss '{name}' requires batch mask"):
                LOSS_DEFINITIONS[name].reconstruction_loss(image, image, params, masks=masks)


def test_masked_l1_weights_foreground_and_background() -> None:
    prediction = torch.tensor([[[[1.0, 3.0], [1.0, 3.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # NHW is accepted
    params = _mask_params(foreground_weight=3.0, background_weight=1.0)

    loss = LOSS_DEFINITIONS["l1"].reconstruction_loss(
        prediction, target, params, masks={"foreground_mask": mask}
    )

    assert loss.item() == pytest.approx((3 * 1 * 2 + 1 * 3 * 2) / (3 * 2 + 1 * 2))


def test_empty_masks_are_skipped_or_kept_as_configured() -> None:
    prediction = torch.tensor([[[[2.0, 2.0]]], [[[4.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    masks = {"foreground_mask": torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])}
    l1 = LOSS_DEFINITIONS["l1"]

    skipped = l1.reconstruction_loss(
        prediction, target, _mask_params(background_weight=0.5), masks=masks
    )
    kept = l1.reconstruction_loss(
        prediction,
        target,
        _mask_params(background_weight=0.5, ignore_empty_mask=False),
        masks=masks,
    )
    assert skipped.item() == pytest.approx(2.0)
    assert kept.item() == pytest.approx((2.0 + 4.0) / 2)

    all_empty = {"foreground_mask": torch.zeros(2, 1, 1, 2)}
    zero = l1.reconstruction_loss(prediction, target, _mask_params(), masks=all_empty)
    none_params = l1.parse_params({"reduction": "none", "mask": {"enabled": True}})
    empty = l1.reconstruction_loss(prediction, target, none_params, masks=all_empty)
    assert zero.item() == 0.0 and zero.ndim == 0
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (torch.ones(2, 2), "NCHW or NHW"),
        (torch.ones(2, 1, 4, 4), "batch dimension"),
        (torch.ones(1, 1, 4, 2), "spatial dimensions"),
        (torch.ones(1, 2, 4, 4), "expand"),
    ],
)
def test_masked_reduction_rejects_incompatible_masks(mask: torch.Tensor, match: str) -> None:
    image = torch.zeros(1, 3, 4, 4)
    with pytest.raises((ValueError, RuntimeError), match=match):
        LOSS_DEFINITIONS["l1"].reconstruction_loss(
            image, image, _mask_params(), masks={"foreground_mask": mask}
        )


def test_masked_ssim_and_l1_gradients_follow_mask_weights() -> None:
    generator = torch.Generator().manual_seed(1)
    target = torch.rand(1, 3, 16, 16, generator=generator) * 2 - 1
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8] = 1.0
    params = _mask_params(background_weight=0.0)
    for name in ("l1", "ssim"):
        definition = LOSS_DEFINITIONS[name]
        prediction = (torch.rand(1, 3, 16, 16, generator=generator) * 2 - 1).requires_grad_()
        resolved = definition.parse_params({"mask": params.mask.to_dict()})
        loss = definition.reconstruction_loss(
            prediction, target, resolved, masks={"foreground_mask": mask}
        )
        loss.backward()
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        assert prediction.grad[..., :8].abs().sum() > 0
        if name == "l1":  # no spatial support beyond the pixel itself
            assert prediction.grad[..., 8:].abs().sum() == 0

        full_map = ssim_loss_map(prediction, target) if name == "ssim" else None
        if full_map is not None:
            expected = full_map[..., :8].mean()
            assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
