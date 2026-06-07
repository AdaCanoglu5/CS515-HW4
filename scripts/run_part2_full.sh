#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate

python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"

python -m part2.train --tag main --seed 0

python -m part2.evaluate --tag main --seed 0 --plot-snr
