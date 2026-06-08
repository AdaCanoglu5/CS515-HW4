#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f checkpoints/part2_no_curriculum.pt || -f checkpoints/part2_no_curriculum_seed0.pt ]]; then
  echo "no_curriculum checkpoint already exists"
else
  python -m part2.train --tag no_curriculum --steps 30000
fi
