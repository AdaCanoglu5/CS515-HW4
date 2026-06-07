#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate

python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"
python -m part2.model
python -m part2.train --tag main --steps 5000 --batch-size 1024 --log-every 500
python -m part2.evaluate --tag main --n-eval 20000 --batch 2048 --plot-snr
