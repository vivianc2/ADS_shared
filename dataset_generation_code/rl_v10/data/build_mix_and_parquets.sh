#!/bin/bash
# After build_train.sh and build_val_indomain.sh: the v3 (confounded) validation pair, the standard + v3 training mix,
# and every SkyRL parquet (2026-10-25; the commands the RL and camera-ready sessions ran on 10-23/24, in one place).
# Parquet world_file paths are relative to the data root O (= RLV10_DATA_ROOT, /work/rl_v10/data in the container).
#   O=~/rpgrl/rl_v10/data bash build_mix_and_parquets.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); RL=$(dirname "$HERE"); D=$(dirname "$RL")
export PYTHONPATH=$D/rpg_rl:$D/rpg_v9 RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 RPG_WORLD_VERSION=v10 OMP_NUM_THREADS=1
PY=${PY:-~/miniconda3/envs/ADS/bin/python}; O=${O:-~/rpgrl/rl_v10/data}; W=${W:-16}
REL=$D/../rpg_benchmark/data
# the v10 split parquet behind rpg_benchmark/data/validation (same 270 world ids, checked 10-25)
VAL_PARQUET=${VAL_PARQUET:-$D/rpg_v9/experiment_datasets/v10_benchmark_2026-10-12/dev_heldout_domain.parquet}
cd $RL
# 1. v3 pair of the release validation split (268/270 pass their audits)
[ -d $O/worlds_validation_confounded_v3 ] || {
  $PY build_cuefree.py $VAL_PARQUET $O/v3_validation_dump --design v3 --workers $W
  $PY to_release_format.py $O/v3_validation_dump $O/worlds_validation_confounded_v3; }
# 2. training mix: half of each structure stays standard, half is rebuilt as v3 (make_train_mix.py docstring)
M=$O/train_mix_v3
[ -d $M/worlds_mix ] || {
  $PY make_train_mix.py split $O/train_v10_60M.parquet $M
  $PY build_cuefree.py $M/v3_rows.parquet $M/v3_dump --design v3 --workers $W
  $PY to_release_format.py $M/v3_dump $M/worlds_v3
  $PY make_train_mix.py link $O/worlds_train $M; }
# 3. eval world copies under the data root, then parquets (prompt = run_world's opening, checked at every reset)
for s in validation core54 core54_anon core108; do [ -d $O/worlds_$s ] || cp -r $REL/$s $O/worlds_$s; done
mk() { [ -f $O/$1.parquet ] || $PY build_dataset.py --root $O --worlds $2 --out $O/$1.parquet --split $1; }
mk train worlds_train
mk validation worlds_validation
mk core54 worlds_core54
mk core108 worlds_core108
mk validation_confounded_v3 worlds_validation_confounded_v3
mk val_indomain worlds_val_indomain
mk val_indomain_confounded_v3 worlds_val_indomain_confounded_v3
[ -f $M/train_mix_v3.parquet ] || $PY build_dataset.py --root $O --worlds train_mix_v3/worlds_mix \
  --out $M/train_mix_v3.parquet --split train
$PY $HERE/manifest.py write $O
