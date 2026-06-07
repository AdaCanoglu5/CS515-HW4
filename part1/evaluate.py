"""Evaluation for Part 1 models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from part1.data import DEFAULT_TICKERS, make_dataset, make_recent_return_baseline
from part1.train import TASKS, build_model


def _load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    ckpt_path = Path(args.checkpoint or f"checkpoints/part1_{args.task}_seed{args.seed}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(
        args.task,
        ckpt.get("hidden", args.hidden),
        ckpt.get("num_layers", args.num_layers),
        ckpt.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _collect(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, targets, xs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds.append(model(X).cpu().numpy())
            targets.append(y.numpy())
            xs.append(X.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(xs)


def evaluate_regression(pred: np.ndarray, y: np.ndarray, recent: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {"overall_mse": float(np.mean((pred - y) ** 2))}
    zero = np.zeros_like(y)
    metrics["zero_baseline_mse"] = float(np.mean((zero - y) ** 2))
    metrics["recent_return_baseline_mse"] = float(np.mean((recent - y) ** 2))
    metrics["directional_accuracy_d1"] = float(np.mean(np.sign(pred[:, 0]) == np.sign(y[:, 0])))
    for idx in range(y.shape[1]):
        metrics[f"mse_d{idx + 1}"] = float(np.mean((pred[:, idx] - y[:, idx]) ** 2))
        metrics[f"zero_mse_d{idx + 1}"] = float(np.mean((zero[:, idx] - y[:, idx]) ** 2))
        metrics[f"recent_mse_d{idx + 1}"] = float(np.mean((recent[:, idx] - y[:, idx]) ** 2))
    return metrics


def evaluate_turning(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(np.float32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "test_positive_fraction": float(np.mean(y)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    if len(np.unique(y)) == 2:
        metrics["pr_auc"] = float(average_precision_score(y, prob))
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS.keys(), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    mode, _, kind = TASKS[args.task]
    _, _, test_ds = make_dataset(args.tickers, mode=mode, cache_dir=args.cache_dir, gamma=args.gamma)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    model = _load_model(args, device)
    pred, y, X = _collect(model, loader, device)
    if kind == "turning":
        metrics = evaluate_turning(pred, y)
    else:
        recent = make_recent_return_baseline(args.tickers, mode=mode, cache_dir=args.cache_dir)
        if recent.shape != y.shape:
            raise ValueError(f"baseline shape {recent.shape} does not match targets {y.shape}")
        metrics = evaluate_regression(pred, y, recent)

    out_path = Path("results") / f"part1_{args.task}_seed{args.seed}_eval.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
