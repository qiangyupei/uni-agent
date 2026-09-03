"""Synthetic smoke task authored for this recipe; not benchmark data."""

import torch


class Model(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2.0
