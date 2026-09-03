"""Synthetic smoke task authored for this recipe; not benchmark data."""

import torch


class Model(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right
