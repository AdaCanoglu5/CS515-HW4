"""Training entry point for Part 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from part1 import set_seed
from part1.data import DEFAULT_TICKERS, make_dataset
from part1.models import StockGRU, StockLSTM, TurningPointBiRNN


TASKS = {
    "lstm-reg": ("returns", "lstm", "regression"),
    "gru-reg": ("returns", "gru", "regression"),
    "lstm-rolling": ("rolling", "lstm", "regression"),
    "gru-rolling": ("rolling", "gru", "regression"),
    "bilstm-turning": ("turning", "bilstm", "turning"),
    "bigru-turning": ("turning", "bigru", "turning"),
}


def build_model(task: str, hidden: int, num_layers: int, dropout: float) -> nn.Module:
    _, model_name, kind = TASKS[task]
    if kind == "regression":
        if model_name == "lstm":
            return StockLSTM(hidden=hidden, num_layers=num_layers, dropout=dropout)
        return StockGRU(hidden=hidden, num_layers=num_layers, dropout=dropout)
    rnn_type = "lstm" if model_name == "bilstm" else "gru"
    return TurningPointBiRNN(hidden=hidden, num_layers=num_layers, dropout=dropout, rnn_type=rnn_type)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = criterion(pred, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.shape[0]
        total_n += X.shape[0]
    return total_loss / max(total_n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS.keys(), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    device = torch.device(args.device)
    mode, _, kind = TASKS[args.task]

    train_ds, val_ds, _ = make_dataset(
        tickers=args.tickers,
        mode=mode,
        cache_dir=args.cache_dir,
        gamma=args.gamma,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.task, args.hidden, args.num_layers, args.dropout).to(device)
    if kind == "turning":
        labels = train_ds.tensors[1]
        positives = labels.sum().item()
        negatives = labels.numel() - positives
        pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt = Path("checkpoints") / f"part1_{args.task}_seed{args.seed}.pt"
    log_path = Path("results") / f"part1_{args.task}_seed{args.seed}.csv"
    best_val = float("inf")
    stale = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.3e}")

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "task": args.task,
                    "seed": args.seed,
                    "gamma": args.gamma,
                    "hidden": args.hidden,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "tickers": args.tickers,
                },
                ckpt,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
