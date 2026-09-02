#!/usr/bin/env bash
# Run a python command from this package inside the SkyRL uv environment.
#
#   bash scripts/in_container.sh python -m prompt_compare_rl.build_dataset
#   bash scripts/in_container.sh python -m prompt_compare_rl.smoke_test
#   bash scripts/in_container.sh python -m pytest -q prompt_compare_rl/tests
#
# Nothing here is run-specific: it only pins the protocol, the import path, and the uv
# environment that owns pandas / datasets / transformers / skyrl.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_PARENT="$(dirname "$PKG_DIR")"
SKYRL_DIR="${PC_SKYRL_DIR:-/work/SkyRL}"

export RPG_SRC="${RPG_SRC:-/work/ADS_shared/dataset_generation_code}"
export RPG_PROTO="${RPG_PROTO:-rpg_v9}"
export PYTHONPATH="$PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/hf_cache}"
# Never let a stale override leak into a build/test process: the experiment's prompt
# selection is per-run and is set by run_one.sh only.
unset RPG_SYSTEM_PROMPT_FILE RPG_SYSTEM_PROMPT_SHA256 || true

cd "$SKYRL_DIR"
# PC_UV_WITH lets callers add packages that are not SkyRL dependencies (e.g. pytest).
UV_EXTRA=()
if [ -n "${PC_UV_WITH:-}" ]; then
  for pkg in $PC_UV_WITH; do UV_EXTRA+=(--with "$pkg"); done
fi
exec uv run --isolated --extra fsdp "${UV_EXTRA[@]}" "$@"
