"""AWGN channel helpers."""

from __future__ import annotations

import torch


def awgn(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


def power_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize so E_batch[||x||^2] is approximately 1."""
    mean_sq_norm = (x**2).sum(dim=1).mean()
    return x / torch.sqrt(mean_sq_norm + 1e-8)
