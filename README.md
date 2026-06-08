# CS515 HW4

Run commands from the project root.

## Verify

```bash
bash scripts/verify.sh
```

## Populate Data

```bash
bash scripts/populate_data.sh
```

## Part 1

```bash
bash scripts/run_part1_all.sh
```

## Part 2

Train the full main model:

```bash
bash scripts/run_part2_full.sh
```

Train the 30k main comparison:

```bash
bash scripts/run_part2_main_30k.sh
```

Train ablations:

```bash
bash scripts/run_part2_ablations.sh
```

Generate comparison plots and sanity report:

```bash
bash scripts/sweep_all_checkpoints.sh
python -m part2.plot_comparison
python -m part2.sanity
```
