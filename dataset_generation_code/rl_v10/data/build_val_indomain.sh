#!/bin/bash
# In-domain validation split for RL v10 (2026-10-25): 15 worlds per structure (135) from the 8 training domains, seed
# base 62M (unused: 10/11/20/21/40/41/43/50-53M benchmark and earlier sets, 60M RL train; 61M only in a probe, never
# stored). Same recipe as build_train.sh, then its v3 (confounded) pair. Why: validation (rpg_benchmark/data/validation)
# holds only the 2 held-out domains, so checkpoints could not be chosen on the training domains.
#   O=~/rpgrl/rl_v10/data bash build_val_indomain.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); RL=$(dirname "$HERE"); D=$(dirname "$RL")
export PYTHONPATH=$D/rpg_rl:$D/rpg_v9 RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 RPG_WORLD_VERSION=v10 OMP_NUM_THREADS=1
PY=${PY:-~/miniconda3/envs/ADS/bin/python}; O=${O:-~/rpgrl/rl_v10/data}; W=${W:-16}
echo "[$(date -u +%FT%TZ)] build"
cd $D && nice -n 5 $PY skyrl_rpg/rebuild_v9_sets.py balanced $O/val_indomain_62M.parquet --per_arch 15 --seed0 62000000 \
  --skins train --label val_indomain --workers $W 2>&1 | grep -v Warn | tail -3
echo "[$(date -u +%FT%TZ)] dump + release format"
cd $D/rpg_v9 && $PY dump_worlds_from_parquet.py $O/val_indomain_62M.parquet $O/val_indomain_dump 2>&1 | grep -v Warn | tail -2
$PY $RL/to_release_format.py $O/val_indomain_dump $O/worlds_val_indomain
echo "[$(date -u +%FT%TZ)] v3 pair"
cd $RL && $PY build_cuefree.py $O/val_indomain_62M.parquet $O/val_indomain_v3_dump --design v3 --workers $W 2>&1 | grep -v Warn | tail -4
$PY to_release_format.py $O/val_indomain_v3_dump $O/worlds_val_indomain_confounded_v3
$PY cuefree_shortcut_probe.py $O/worlds_val_indomain_confounded_v3
echo "[$(date -u +%FT%TZ)] done"
