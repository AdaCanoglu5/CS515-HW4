"""Sanity checks for a trained Part 2 checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from part2.evaluate import build_from_checkpoint, checkpoint_path_for_tag
from part2.model import K, SIGMA2


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint_path_for_tag("main")
    model = build_from_checkpoint(ckpt, device)

    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    n_params_dec = sum(p.numel() for p in model.decoder.parameters())
    checkpoint_size_mb = ckpt.stat().st_size / (1024 * 1024)

    with torch.no_grad():
        m = torch.randint(0, model.m, (8192, K), device=device)
        x_hist, _ = model.encode_rounds(m, math.sqrt(SIGMA2))
        per_round = (x_hist**2).sum(dim=1).mean(dim=0)

    lines = [
        f"checkpoint = {ckpt}",
        f"mean_power_per_round = {per_round.tolist()}",
        f"total_param_count_encoder = {n_params_enc}",
        f"total_param_count_decoder = {n_params_dec}",
        f"checkpoint_size_mb = {checkpoint_size_mb:.3f}",
        f"max_round_power = {per_round.max().item():.6f}",
    ]
    for line in lines:
        print(line)
    assert per_round.max().item() <= 1.05, "power constraint violated"
    print("power constraint OK")

    Path("results").mkdir(exist_ok=True)
    with Path("results/sanity.txt").open("w") as f:
        for line in lines:
            f.write(f"{line}\n")
        f.write("power constraint OK\n")


if __name__ == "__main__":
    main()
