#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"
python -m part2.model
python -m part2.train --tag main_30k --batch-size 4096 --log-every 500
python -m part2.evaluate --tag main_30k --n-eval 200000 --batch 8192 --plot-snr
