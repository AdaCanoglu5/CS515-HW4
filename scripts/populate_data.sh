#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
from part1.data import DEFAULT_END, DEFAULT_START, DEFAULT_TICKERS, download_data

frames = download_data(
    tickers=DEFAULT_TICKERS,
    start=DEFAULT_START,
    end=DEFAULT_END,
    cache_dir="data",
)

for ticker, df in frames.items():
    print(f"{ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
PY

echo "Data cache populated under data/*.csv"
