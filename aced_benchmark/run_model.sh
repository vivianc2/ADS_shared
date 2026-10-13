#!/usr/bin/env bash
# Run one model on ACED-Bench with the paper's main protocol, then score it.
#
#   bash run_model.sh NAME BACKEND MODEL
#
#   NAME     short tag used for folders and run names (e.g. opus55)
#   BACKEND  bedrock | openai | gemini
#   MODEL    model id for that backend
#
# What it runs (the cells behind the paper's main figures):
#   round4  4-turn budget, 3 agents (agent = reasoning, coder = active+code, coder_new = modular)
#           x 2 sets (core_struct: 10 worlds / 60 Q; core_decision: 60 worlds / 60 Q)
#   guess   guess-only, no data (guess_struct: 30 worlds / 180 Q; guess_decision: 48 worlds / 48 Q)
#   eval    score everything; results in out/NAME/eval/
#
# Environment:
#   BASE_URL, API_KEY  for BACKEND=openai (Nautilus, vLLM, Gemini's OpenAI endpoint, DeepSeek, ...)
#   MAX_TOKENS         scientist output cap, default 65536 (the paper used 8000 for Bedrock Llama, 16000 for GPT-4o)
#   GUESS_MAX_TOKENS   guess-only output cap, default = MAX_TOKENS (the paper used 4096; we never cap below the max)
#   STAGES             default "run eval"; "eval" alone re-scores finished runs
#   STRUCT_SHARDS / DECISION_SHARDS   parallel workers per cell, default 10 / 20 (as in the paper)
#   PYTHON             default: python
#
# Safe to stop and restart: finished questions are skipped (agent_logs/ in each run folder).
set -uo pipefail

if [ $# -lt 3 ]; then sed -n 2,26p "$0"; exit 1; fi
NAME=$1; BACKEND=$2; MODEL=$3
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/code
DATA=$HERE/data
OUT=$HERE/out/$NAME
PY=${PYTHON:-python}
MAX_TOKENS=${MAX_TOKENS:-65536}
GUESS_MAX_TOKENS=${GUESS_MAX_TOKENS:-$MAX_TOKENS}
STAGES=${STAGES:-"run eval"}
STRUCT_SHARDS=${STRUCT_SHARDS:-10}
DECISION_SHARDS=${DECISION_SHARDS:-20}
# World parser (turns the agent's requests into simulator queries) and answer extractor used for every paper
# number. The paper's eval files label it "opus-4-8-v1"; the raw run files show the real id below.
PARSER=us.anthropic.claude-opus-4-6-v1
export BEDROCK_READ_TIMEOUT=${BEDROCK_READ_TIMEOUT:-900}

SCI=(--scientist-backend "$BACKEND" --scientist-model "$MODEL")
GUESS=(--backend "$BACKEND" --model "$MODEL")
if [ -n "${BASE_URL:-}" ]; then SCI+=(--scientist-base-url "$BASE_URL"); GUESS+=(--base-url "$BASE_URL"); fi
if [ -n "${API_KEY:-}" ]; then SCI+=(--scientist-api-key "$API_KEY"); GUESS+=(--api-key "$API_KEY"); fi

mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/guess"

# One cell = one agent type on one world set. Shards run in parallel into the same run folder; the final pass over
# the full set only collects the finished questions (and redoes any a shard missed) into <run>.json.
run_cell() {
  local set=$1 agent=$2 nshard=$3 preset=$4
  local run=${NAME}_${agent}_${set}_round4
  local dir=$OUT/round4/$run
  local common=(--run-name "$run" --output-dir "$dir" --agent-type "$agent"
                --limit-variant rounds --limit-preset "$preset" --max-turns 4
                --max-new-tokens "$MAX_TOKENS" --world-model "$PARSER" "${SCI[@]}")
  mkdir -p "$dir"
  local pids=() i
  for ((i = 0; i < nshard; i++)); do
    local sd=$OUT/shards/$set/$i
    ls "$sd"/world_*.json >/dev/null 2>&1 || continue
    mkdir -p "$OUT/tmp/${run}_$i"
    TMPDIR=$OUT/tmp/${run}_$i $PY "$CODE/run_agent_batch.py" --worlds-dir "$sd" -o "$dir/shard_$i.json" \
      "${common[@]}" > "$OUT/logs/${run}_shard_$i.log" 2>&1 &
    pids+=($!)
    sleep 1
  done
  wait "${pids[@]}"
  mkdir -p "$OUT/tmp/${run}_merge"
  TMPDIR=$OUT/tmp/${run}_merge $PY "$CODE/run_agent_batch.py" --worlds-dir "$DATA/$set" -o "$dir/$run.json" \
    "${common[@]}" > "$OUT/logs/${run}_merge.log" 2>&1
  echo "[$(date +%H:%M)] done $run (exit $?)"
}

make_shards() {  # set nshard: world i goes to shard i mod nshard (symlinks); made once, before the cells start
  local set=$1 nshard=$2 worlds=("$DATA/$1"/world_*.json) i
  [ -d "$OUT/shards/$set" ] && return
  for ((i = 0; i < ${#worlds[@]}; i++)); do
    mkdir -p "$OUT/shards/$set/$((i % nshard))"
    ln -s "${worlds[$i]}" "$OUT/shards/$set/$((i % nshard))/"
  done
}

run_guess() {
  local set=$1 out=$OUT/guess/zero_shot_${1}.json
  if ls "$OUT/guess/zero_shot_${set}"_*.json >/dev/null 2>&1; then echo "skip guess $set (exists)"; return; fi
  $PY "$CODE/run_zero_shot.py" --worlds-dir "$DATA/$set" "${GUESS[@]}" --max-new-tokens "$GUESS_MAX_TOKENS" \
    -o "$out" > "$OUT/logs/guess_${set}.log" 2>&1
  echo "[$(date +%H:%M)] done guess $set (exit $?)"
}

score() {  # evaluator, output, inputs...
  local ev=$1 o=$2; shift 2
  for f in "$@"; do [ -f "$f" ] || { echo "MISSING $f" >&2; return 1; }; done
  $PY "$CODE/$ev" "$@" --details --extract-model "$PARSER" -o "$o" > "$o.log" 2>&1 \
    && echo "scored $(basename "$o")" || echo "FAILED $(basename "$o") (see $o.log)" >&2
}

if [[ " $STAGES " == *" run "* ]]; then
  echo "[$(date +%H:%M)] $NAME: $BACKEND $MODEL -> $OUT"
  make_shards core_struct "$STRUCT_SHARDS"
  make_shards core_decision "$DECISION_SHARDS"
  pids=()
  for agent in agent coder coder_new; do
    run_cell core_struct "$agent" "$STRUCT_SHARDS" 4_19_big & pids+=($!)
    run_cell core_decision "$agent" "$DECISION_SHARDS" adv_v3 & pids+=($!)
  done
  run_guess guess_struct & pids+=($!)
  run_guess guess_decision & pids+=($!)
  wait "${pids[@]}"
fi

if [[ " $STAGES " == *" eval "* ]]; then
  for agent in agent coder coder_new; do
    r=${NAME}_${agent}_core_struct_round4
    score evaluate_zero_shot.py "$OUT/eval/eval_$r.json" "$OUT/round4/$r/$r.json"
    r=${NAME}_${agent}_core_decision_round4
    score evaluate_advanced.py "$OUT/eval/eval_$r.json" "$OUT/round4/$r/$r.json"
  done
  score evaluate_zero_shot.py "$OUT/eval/eval_${NAME}_guess_struct.json" \
    "$OUT/guess/zero_shot_guess_struct_easy.json" "$OUT/guess/zero_shot_guess_struct_medium.json"
  score evaluate_advanced.py "$OUT/eval/eval_${NAME}_guess_decision.json" "$OUT/guess/zero_shot_guess_decision_hard.json"
  $PY "$HERE/summarize.py" "$OUT"
fi
