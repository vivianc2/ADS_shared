#!/usr/bin/env bash
# Rerun selected round4 agent cells of one model (2026-10-23, GPT-6 first-block wrapper), paper protocol as run_model.sh:
#   bash rerun_cells.sh NAME MODEL WORLDS_STRUCT WORLDS_DECISION AGENT...
# WORLDS_*: world dirs (data/core_struct, data/core_decision, or a probe subset). Output: out/NAME/round4/NAME_<agent>_core_<set>_round4/
# Shards 5 / 10 (shared machine; shard count only changes wall time). Token usage -> out/NAME/usage.jsonl.
set -uo pipefail
NAME=$1 MODEL=$2 WS=$3 WD=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$HERE/out/$NAME; PY=~/miniconda3/envs/ADS/bin/python
source ~/ADS_folder/key.txt
export BEDROCK_MAX_ATTEMPTS=10 BEDROCK_READ_TIMEOUT=900 ACED_USAGE_LOG=$OUT/usage.jsonl
PARSER=us.anthropic.claude-opus-4-6-v1
mkdir -p $OUT/logs
cell() {  # set worlds_dir preset nshard agent
  local set=$1 wd=$2 preset=$3 nsh=$4 agent=$5 run=${NAME}_${5}_core_${1}_round4
  local dir=$OUT/round4/$run i
  local common=(--run-name $run --output-dir $dir --agent-type $agent --limit-variant rounds --limit-preset $preset
                --max-turns 4 --max-new-tokens 128000 --world-model $PARSER --scientist-backend bedrock --scientist-model $MODEL)
  mkdir -p $dir; local ws=($wd/world_*.json) pids=()
  for ((i = 0; i < nsh; i++)); do
    local sd=$OUT/shards/${set}_$(basename $wd)/$i; mkdir -p $sd $OUT/tmp/${run}_$i
    local j; for ((j = i; j < ${#ws[@]}; j += nsh)); do ln -sf ${ws[$j]} $sd/; done
    ls $sd/world_*.json >/dev/null 2>&1 || continue
    TMPDIR=$OUT/tmp/${run}_$i $PY $HERE/code/run_agent_batch.py --worlds-dir $sd -o $dir/shard_$i.json "${common[@]}" > $OUT/logs/${run}_shard_$i.log 2>&1 &
    pids+=($!); sleep 1
  done
  wait "${pids[@]}"
  mkdir -p $OUT/tmp/${run}_merge
  TMPDIR=$OUT/tmp/${run}_merge $PY $HERE/code/run_agent_batch.py --worlds-dir $wd -o $dir/$run.json "${common[@]}" > $OUT/logs/${run}_merge.log 2>&1
  echo "[$(date +%H:%M)] done $run"
}
for agent in "$@"; do
  cell struct $WS 4_19_big 5 $agent & cell decision $WD adv_v3 10 $agent &
done
wait
