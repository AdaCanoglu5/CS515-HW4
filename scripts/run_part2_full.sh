#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "from part2 import test_power_normalize; test_power_normalize(); print('power_normalize test passed')"

python -m part2.train --tag main

python -m part2.evaluate --tag main --plot-snr
