#!/usr/bin/env bash
# Run a python command from this package inside the SkyRL environment.
#
#   bash scripts/in_container.sh python -m single_arch_rl.build_dataset
#   bash scripts/in_container.sh python -m pytest -q single_arch_rl/tests
#
# Nothing here is run-specific: it only pins the protocol, the import path, and the uv
# environment that owns pandas / datasets / transformers / skyrl.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_PARENT="$(dirname "$PKG_DIR")"
SKYRL_DIR="${SA_SKYRL_DIR:-/work/SkyRL}"

export RPG_SRC="${RPG_SRC:-/work/ADS_shared/dataset_generation_code}"
export RPG_PROTO="${RPG_PROTO:-rpg_v9}"
export PYTHONPATH="$PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/hf_cache}"

cd "$SKYRL_DIR"
# SA_UV_WITH lets callers add packages that are not SkyRL dependencies (e.g. pytest).
UV_EXTRA=()
if [ -n "${SA_UV_WITH:-}" ]; then
  for pkg in $SA_UV_WITH; do UV_EXTRA+=(--with "$pkg"); done
fi
exec uv run --isolated --extra fsdp "${UV_EXTRA[@]}" "$@"
