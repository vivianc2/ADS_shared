#!/usr/bin/env bash
# Every CPU-runnable check for this experiment, in one command.
# Run INSIDE the skyrl container:  bash scripts/run_tests.sh
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== unit tests =="
PC_UV_WITH=pytest bash "$PKG_DIR/scripts/in_container.sh" \
  python -m pytest -q --no-header -p no:cacheprovider "$PKG_DIR/tests"

echo
echo "== smoke test =="
bash "$PKG_DIR/scripts/in_container.sh" python -m prompt_compare_rl.smoke_test
