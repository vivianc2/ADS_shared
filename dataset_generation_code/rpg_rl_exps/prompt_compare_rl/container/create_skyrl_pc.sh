#!/usr/bin/env bash
# Create the `skyrl-pc` Docker container: the existing SkyRL environment plus a
# writable /data mount, so training can write 19 GB checkpoints to the 30 TB NFS
# volume instead of the 49 GB container filesystem.
#
# Run ON THE HOST:  bash container/create_skyrl_pc.sh
#
# This NEVER touches the existing `skyrl` container. Two deliberate differences
# from the original creation recipe:
#
#   1. It does not regenerate ~/rpg/passwd and ~/rpg/group. Those paths are bind-
#      mounted read-only into the RUNNING `skyrl` container, and ~/rpg/passwd is
#      now an empty root-owned directory (the real file was unlinked on
#      2026-08-23; the running container still holds the original inode). Reusing
#      that path would mount a directory over /etc/passwd and uid 1001 would stop
#      resolving. This script keeps its own copies under container/ instead --
#      byte-identical to what the running container has.
#
#   2. /data/rpg_rl_exps is mounted at the SAME path inside the container, so a
#      checkpoint path printed in a log or run_manifest.json is directly usable
#      from the host shell with no translation.
set -euo pipefail

NAME="${PC_CONTAINER:-skyrl-pc}"
IMG="${PC_IMAGE:-novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8}"
REPO_ROOT="${PC_HOST_REPO_ROOT:-$HOME/rpg}"
DATA_DIR="${PC_HOST_DATA_DIR:-/data/rpg_rl_exps}"
CDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$NAME" = "skyrl" ]; then
  echo "Refusing to use the name 'skyrl': that container must stay untouched." >&2
  exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Container '$NAME' already exists. Remove it first: docker rm -f $NAME" >&2
  exit 2
fi
[ -d "$DATA_DIR" ] && [ -w "$DATA_DIR" ] || {
  echo "$DATA_DIR is not a writable directory on the host." >&2
  exit 2
}

# /etc/passwd and /etc/group for the container, generated from the image the same
# way the original was, plus the `rpg` line for the invoking uid.
if [ ! -s "$CDIR/passwd" ] || [ ! -s "$CDIR/group" ]; then
  docker run --rm --entrypoint cat "$IMG" /etc/passwd > "$CDIR/passwd"
  docker run --rm --entrypoint cat "$IMG" /etc/group  > "$CDIR/group"
  echo "rpg:x:$(id -u):100:rpg:/work/home:/bin/bash" >> "$CDIR/passwd"
fi

docker run -d --name "$NAME" --runtime=nvidia --gpus all --shm-size=16g \
  --user "$(id -u)":100 \
  -e HOME=/work/home -e USER=rpg -e LOGNAME=rpg \
  -e HF_HOME=/work/hf_cache \
  -e XDG_CACHE_HOME=/work/home/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/work/home/.cache/torchinductor \
  -e TRITON_CACHE_DIR=/work/home/.cache/triton \
  -e RAY_TMPDIR=/work/tmp \
  -v "$REPO_ROOT":/work \
  -v "$CDIR/passwd":/etc/passwd:ro \
  -v "$CDIR/group":/etc/group:ro \
  -v "$DATA_DIR":"$DATA_DIR" \
  "$IMG" sleep infinity

echo "Created $NAME. Verifying:"
docker exec "$NAME" bash -lc "id; df -h $DATA_DIR | tail -1; \
  /work/SkyRL/.venv/bin/python -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"gpus\", torch.cuda.device_count())'"
echo
echo "Use it with:  PC_CONTAINER=$NAME bash scripts/launch_all.sh"
