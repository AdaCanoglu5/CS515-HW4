"""Utilities for Part 2."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random, numpy, torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_power_normalize() -> None:
    """Assignment-required unit test for the channel power constraint."""
    from part2.channel import power_normalize

    x = torch.randn(1000, 4) * 3.7
    y = power_normalize(x)
    emp = (y**2).sum(dim=1).mean().item()
    assert abs(emp - 1.0) < 0.05, f"power constraint violated: {emp}"
