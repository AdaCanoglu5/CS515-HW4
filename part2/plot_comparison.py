"""Plot Part 2 BLER curves against the Shannon capacity boundary.

Capacity-boundary derivation:

Per-channel-use SNR = (1/d) / sigma^2 = 1/(4 sigma^2)
Capacity per use     = 0.5 * log2(1 + 1/(4 sigma^2)) bpcu
Total channel uses   = T * d = 16
Total capacity       = 16 * 0.5 * log2(1 + 1/(4 sigma^2))
                     = 8 log2(1 + 1/(4 sigma^2)) bits
Message size         = K * log2(M) = 4 * 3 = 12 bits
Solve C = 12:        sigma^2 = 1 / (4 * (2^1.5 - 1)) = 0.1367

On the plot's x-axis, SNR_dB = -10 * log10(sigma^2), so the boundary is
approximately 8.64 dB.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
SIGMA2_CAPACITY_BOUNDARY = 1.0 / (4.0 * (2.0**1.5 - 1.0))
SNR_DB_CAPACITY_BOUNDARY = -10.0 * math.log10(SIGMA2_CAPACITY_BOUNDARY)


def _tag_from_path(path: Path) -> str:
    return path.stem.removeprefix("bler_vs_snr_")


def plot_bler_comparison() -> None:
    paths = sorted(RESULTS.glob("bler_vs_snr_*.csv"))
    if not paths:
        raise FileNotFoundError("No results/bler_vs_snr_*.csv files found. Run scripts/sweep_all_checkpoints.sh first.")

    combined_rows = []
    plt.figure(figsize=(8, 5))
    for path in paths:
        tag = _tag_from_path(path)
        df = pd.read_csv(path).sort_values("snr_db")
        for row in df.to_dict("records"):
            combined_rows.append({"tag": tag, **row})
        plt.semilogy(df["snr_db"], df["bler"], marker="o", linewidth=2, label=tag)

    plt.axvline(SNR_DB_CAPACITY_BOUNDARY, color="black", linestyle="--", linewidth=1.5)
    plt.text(
        SNR_DB_CAPACITY_BOUNDARY + 0.15,
        0.35,
        "R = C (Shannon limit)",
        rotation=90,
        va="center",
        ha="left",
    )
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "comparison_bler_vs_snr.png", dpi=180)
    plt.close()

    with (RESULTS / "comparison_bler_vs_snr.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "sigma2", "snr_db", "bler"])
        writer.writeheader()
        writer.writerows(combined_rows)


def plot_rate_vs_capacity() -> None:
    snr_db = np.linspace(0, 14, 400)
    capacity_bits = 8.0 * np.log2(1.0 + (10.0 ** (snr_db / 10.0)) / 4.0)
    message_bits = 12.0

    plt.figure(figsize=(7, 4.5))
    plt.plot(snr_db, capacity_bits, color="#1f77b4", linewidth=2.5, label="capacity")
    plt.axhline(message_bits, color="black", linestyle="--", linewidth=1.5, label="message size = 12 bits")
    plt.fill_between(snr_db, 0, capacity_bits, color="#2ca02c", alpha=0.12)
    plt.fill_between(snr_db, capacity_bits, 18, color="#d62728", alpha=0.08)
    plt.axvline(SNR_DB_CAPACITY_BOUNDARY, color="black", linestyle=":", linewidth=1.2)
    plt.text(10.0, 8.0, "achievable", color="#2ca02c", fontsize=11)
    plt.text(1.0, 15.5, "not achievable", color="#d62728", fontsize=11)
    plt.xlabel("SNR (dB)")
    plt.ylabel("bits per message")
    plt.ylim(0, 18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "rate_vs_capacity.png", dpi=180)
    plt.close()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    plot_bler_comparison()
    plot_rate_vs_capacity()
    print("wrote results/comparison_bler_vs_snr.png")
    print("wrote results/comparison_bler_vs_snr.csv")
    print("wrote results/rate_vs_capacity.png")


if __name__ == "__main__":
    main()
