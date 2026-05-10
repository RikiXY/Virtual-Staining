from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast


class Predictor:
    """Runs the generator on a single batched input tensor."""

    def __init__(self, generator: nn.Module, device: torch.device, amp_enabled: bool) -> None:
        self.generator = generator
        self.device = device
        self.amp_enabled = amp_enabled
        self.generator.eval()

    @torch.no_grad()
    def predict_batch(self, source: torch.Tensor) -> torch.Tensor:
        """Return the generated tensor in [0, 1] range (clamped)."""
        source = source.to(self.device)
        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            output = self.generator(source)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        return output
