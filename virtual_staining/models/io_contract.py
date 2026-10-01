from __future__ import annotations

import torch
from torchvision import transforms

RGB_MEAN = (0.5, 0.5, 0.5)
RGB_STD = (0.5, 0.5, 0.5)
MODEL_INPUT_RANGE = (-1, 1)
MODEL_OUTPUT_RANGE = (-1, 1)
GENERATOR_OUTPUT_ACTIVATION = "tanh"
NORMALIZATION_CONTRACT = {"input_range": "[-1, 1]", "output_range": "[-1, 1]"}


def build_model_input_transform(
    image_size: tuple[int, int] | None,
) -> transforms.Compose:
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize((image_size[1], image_size[0])))
    steps.extend((transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)))
    return transforms.Compose(steps)


def denormalize_model_output(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 0.5 + 0.5).clamp(0, 1)
