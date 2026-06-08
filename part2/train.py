"""Training entry point for Part 2."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from part2 import set_seed
from part2.model import K, M, SIGMA2, T_ROUNDS, FeedbackCodeSystem

DEFAULT_SEED = 0


def get_sigma(step: int, total_steps: int, var_start: float = 0.05, var_end: float = 0.25, frac: float = 0.30) -> float:
    if step >= frac * total_steps:
        return math.sqrt(var_end)
    f = step / (frac * total_steps)
    var = var_start + f * (var_end - var_start)
    return math.sqrt(var)


def lr_lambda(step: int, total_steps: int, warmup: int = 2000) -> float:
    if step < warmup:
        return max((step + 1) / warmup, 1e-8)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validation_bler(model: FeedbackCodeSystem, device: torch.device, batch: int = 8192) -> float:
    model.eval()
    m = torch.randint(0, model.m, (batch, model.k), device=device)
    logits = model(m, math.sqrt(SIGMA2))
    pred = logits.argmax(dim=-1)
    bler = (pred != m).any(dim=1).float().mean().item()
    model.train()
    return bler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        choices=["main", "main_30k", "no_feedback", "T1", "T2", "T3", "no_curriculum"],
        default="main",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    device = torch.device(args.device)
    default_steps = 100_000 if args.tag == "main" else 30_000
    total_steps = args.steps or default_steps
    no_feedback = args.tag == "no_feedback"
    t_rounds = {"T1": 1, "T2": 2, "T3": 3}.get(args.tag, T_ROUNDS)

    model = FeedbackCodeSystem(t_rounds=t_rounds, no_feedback=no_feedback).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = LambdaLR(optimizer, lambda step: lr_lambda(step, total_steps, args.warmup))
    log_path = Path("results") / f"part2_{args.tag}_train.csv"
    rows = []

    model.train()
    for step in range(total_steps):
        sigma = math.sqrt(SIGMA2) if args.tag == "no_curriculum" else get_sigma(step, total_steps)
        m = torch.randint(0, M, (args.batch_size, K), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(m, sigma)
        loss = F.cross_entropy(logits.reshape(-1, M), m.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == total_steps - 1:
            bler = validation_bler(model, device)
            lr = scheduler.get_last_lr()[0]
            row = {"step": step, "loss": loss.item(), "sigma": sigma, "lr": lr, "val_bler": bler}
            rows.append(row)
            print(
                f"step={step:06d} loss={loss.item():.5f} "
                f"sigma={sigma:.3f} lr={lr:.3e} val_BLER={bler:.4f}"
            )

    ckpt_path = Path("checkpoints") / f"part2_{args.tag}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "tag": args.tag,
            "t_rounds": t_rounds,
            "no_feedback": no_feedback,
            "steps": total_steps,
        },
        ckpt_path,
    )
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "sigma", "lr", "val_bler"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
