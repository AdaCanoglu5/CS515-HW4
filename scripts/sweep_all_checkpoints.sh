#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate

shopt -s nullglob
declare -A seen_tags=()
for ckpt in checkpoints/part2_*.pt; do
  name="$(basename "$ckpt" .pt)"
  tag="${name#part2_}"
  tag="${tag%%_seed*}"
  if [[ -n "${seen_tags[$tag]:-}" ]]; then
    continue
  fi
  seen_tags[$tag]=1
  echo "=== sweeping $tag from $ckpt ==="
  python -m part2.evaluate --tag "$tag" --plot-snr --n-eval 200000 > "results/sweep_${tag}.log" 2>&1 || echo "  failed for $tag"
done
