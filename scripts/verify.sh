#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
python -m compileall part1 part2 tests
python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"
python -m part2.model
