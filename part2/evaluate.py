"""Evaluation utilities for Part 2."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from part2.model import K, SIGMA2, FeedbackCodeSystem


def build_from_checkpoint(path: Path, device: torch.device) -> FeedbackCodeSystem:
    ckpt = torch.load(path, map_location=device)
    no_feedback = ckpt.get("no_feedback", ckpt.get("tag") == "no_feedback")
    print(f"loaded ckpt {path}: t_rounds={ckpt.get('t_rounds')} no_feedback={no_feedback}")
    model = FeedbackCodeSystem(
        t_rounds=ckpt.get("t_rounds", 4),
        no_feedback=no_feedback,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: FeedbackCodeSystem,
    sigma: float,
    n_eval: int = 200_000,
    batch: int = 8192,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or next(model.parameters()).device
    total = 0
    block_errors = 0.0
    symbol_errors = 0.0
    power_sum = 0.0
    power_count = 0

    while total < n_eval:
        current = min(batch, n_eval - total)
        m = torch.randint(0, model.m, (current, model.k), device=device)
        x_hist, y_hist = model.encode_rounds(m, sigma)
        logits = model.decoder(y_hist.transpose(1, 2))
        pred = logits.argmax(dim=-1)
        err = pred != m
        block_errors += err.any(dim=1).float().sum().item()
        symbol_errors += err.float().sum().item()
        per_round_power = (x_hist**2).sum(dim=1).mean(dim=0)
        power_sum += per_round_power.sum().item()
        power_count += model.t_rounds
        total += current

    return {
        "bler": block_errors / total,
        "ser": symbol_errors / (total * model.k),
        "mean_power": power_sum / power_count,
    }


def checkpoint_path_for_tag(tag: str) -> Path:
    canonical = Path("checkpoints") / f"part2_{tag}.pt"
    legacy = Path("checkpoints") / f"part2_{tag}_seed0.pt"
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    return canonical


def save_snr_plot(model: FeedbackCodeSystem, device: torch.device, n_eval: int, batch: int, tag: str) -> None:
    variances = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
    snr_db, bler = [], []
    for var in variances:
        metrics = evaluate(model, math.sqrt(var), n_eval=n_eval, batch=batch, device=device)
        snr_db.append(-10.0 * math.log10(var))
        bler.append(metrics["bler"])
        print(f"sigma2={var:.2f} snr_db={snr_db[-1]:.2f} bler={bler[-1]:.6f}")

    order = np.argsort(snr_db)
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(np.array(snr_db)[order], np.array(bler)[order], marker="o", label="learned feedback code")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/bler_vs_snr_{tag}.png", dpi=160)
    plt.close()

    with open(f"results/bler_vs_snr_{tag}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma2", "snr_db", "bler"])
        for var, snr, err in zip(variances, snr_db, bler):
            writer.writerow([var, snr, err])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    parser.add_argument("--n-eval", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--sigma2", type=float, default=SIGMA2)
    parser.add_argument("--plot-snr", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    device = torch.device(args.device)

    ckpt_path = checkpoint_path_for_tag(args.tag)
    model = build_from_checkpoint(ckpt_path, device)
    metrics = evaluate(model, math.sqrt(args.sigma2), n_eval=args.n_eval, batch=args.batch, device=device)
    print(metrics)

    out_path = Path("results") / f"part2_{args.tag}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})
            print(f"{key}: {value}")

    if args.plot_snr:
        save_snr_plot(model, device, args.n_eval, args.batch, args.tag)


if __name__ == "__main__":
    main()
