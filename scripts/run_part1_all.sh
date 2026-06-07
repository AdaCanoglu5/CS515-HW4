#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate

TASKS=(
  lstm-reg
  gru-reg
  lstm-rolling
  gru-rolling
)

for seed in 0; do
  for task in "${TASKS[@]}"; do
    python -m part1.train --task "$task" --seed "$seed"
    python -m part1.evaluate --task "$task" --seed "$seed"
  done

  for task in bilstm-turning bigru-turning; do
    for gamma in 1.1 0.1; do
      python -m part1.train --task "$task" --seed "$seed" --gamma "$gamma"
      python -m part1.evaluate --task "$task" --seed "$seed" --gamma "$gamma"
    done
  done
done
