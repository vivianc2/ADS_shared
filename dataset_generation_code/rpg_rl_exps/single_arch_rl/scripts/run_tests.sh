#!/usr/bin/env bash
# Every CPU-runnable check for this experiment, in one command.
# Run INSIDE the skyrl-pc container:  bash scripts/run_tests.sh
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== unit tests =="
SA_UV_WITH=pytest bash "$PKG_DIR/scripts/in_container.sh" \
  python -m pytest -q --no-header -p no:cacheprovider "$PKG_DIR/tests"

echo
echo "== launch pre-flight (full path, no Ray, no GPU) =="
for run in easy hard; do
  SA_PREFLIGHT_ONLY=1 bash "$PKG_DIR/scripts/run_one.sh" "$run"
done
