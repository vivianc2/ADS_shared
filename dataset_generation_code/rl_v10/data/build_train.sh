#!/bin/bash
# RL v10 train split (2026-10-23, RL session; copied into git 2026-10-25 from ~/rpgrl/rl_v10/data/build_train.sh, then
# the release-format conversion added, as run on 10-23): v10 worlds, the 8 in-domain skins, 150 accepted worlds per
# structure, seed base 60M (unused: 10/11/20/21/40/41/43/50-53M are taken). Same recipe as the benchmark build.
# Check: regenerating the 270 test_in_domain worlds this way gives files byte-identical to rpg_benchmark/data/test.
#   O=~/rpgrl/rl_v10/data bash build_train.sh        (~70 min with W=16)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); RL=$(dirname "$HERE"); D=$(dirname "$RL")
export PYTHONPATH=$D/rpg_rl:$D/rpg_v9 RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 RPG_WORLD_VERSION=v10 OMP_NUM_THREADS=1
PY=${PY:-~/miniconda3/envs/ADS/bin/python}; O=${O:-~/rpgrl/rl_v10/data}; W=${W:-16}
echo "[$(date -u +%FT%TZ)] build"
cd $D && nice -n 5 $PY skyrl_rpg/rebuild_v9_sets.py balanced $O/train_v10_60M.parquet --per_arch 150 --seed0 60000000 \
  --skins train --label rl_train --workers $W 2>&1 | grep -v Warn | tail -3
echo "[$(date -u +%FT%TZ)] dump world files"
cd $D/rpg_v9 && $PY dump_worlds_from_parquet.py $O/train_v10_60M.parquet $O/worlds_train_v10_60M 2>&1 | grep -v Warn | tail -3
$PY $RL/to_release_format.py $O/worlds_train_v10_60M $O/worlds_train
echo "[$(date -u +%FT%TZ)] done"
