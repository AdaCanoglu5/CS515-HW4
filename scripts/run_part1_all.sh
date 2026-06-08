#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TASKS=(
  lstm-reg
  gru-reg
  lstm-rolling
  gru-rolling
)

for task in "${TASKS[@]}"; do
  python -m part1.train --task "$task"
  python -m part1.evaluate --task "$task"
done

for task in bilstm-turning bigru-turning; do
  for gamma in 1.1 0.1; do
    python -m part1.train --task "$task" --gamma "$gamma"
    python -m part1.evaluate --task "$task" --gamma "$gamma"
  done
done
