#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
python -m compileall part1 part2 tests
python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"
python -m part2.model
