# CS515 HW4

Implementation for CS515 Deep Learning Homework 4.

- Part 1: financial forecasting with LSTM/GRU models.
- Part 2: learned feedback communication over an AWGN channel.

## Setup

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

## Quick Checks

```bash
bash scripts/verify.sh
```

## Data Cache

The `data/` directory starts empty by design. It is a cache for yfinance CSVs and is populated on demand with adjusted OHLC data for `AAPL`, `JPM`, and `XOM` over `2020-01-01` to `2026-01-01`.

```bash
bash scripts/populate_data.sh
```

## Part 1

```bash
python -m part1.train --task lstm-reg
python -m part1.evaluate --task lstm-reg
```

Tasks:

- `lstm-reg`
- `gru-reg`
- `lstm-rolling`
- `gru-rolling`
- `bilstm-turning`
- `bigru-turning`

For turning-point experiments, pass `--gamma 1.1` for the literal assignment interpretation and `--gamma 0.1` for the generous 10% interpretation.

Full Part 1 batch:

```bash
bash scripts/run_part1_all.sh
```

## Part 2

Main run at the ablation step budget:

```bash
bash scripts/run_part2_main_30k.sh
```

This writes `checkpoints/part2_main_30k.pt` and `results/part2_main_30k*` so it does not overwrite the full main run.

Full main run:

```bash
bash scripts/run_part2_full.sh
```

Full ablation suite:

```bash
bash scripts/run_part2_ablations.sh
```

Part 2 post-training plots and sanity checks:

```bash
bash scripts/train_no_curriculum_if_missing.sh
bash scripts/sweep_all_checkpoints.sh
python -m part2.plot_comparison
python -m part2.sanity
```
