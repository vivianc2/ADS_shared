#!/usr/bin/env bash
# Camera-ready reruns (2026-10-20): one model on ACED-Bench with the paper's protocol, via run_model.sh.
#
#   bash launch_rerun.sh NAME MODEL [GUESS_FROM]
#
# NAME        run folder under out/ (e.g. opus55_r2)
# MODEL       Bedrock model id
# GUESS_FROM  optional folder with an earlier, still-valid guess/ run (zero_shot_guess_*.json): copied into out/NAME/guess
#             so run_model.sh skips guess-only and only rescores it (GPT-6: guess-only is single-shot and has no
#             draft/final double answers, so the wrapper fix does not change it)
#
# Fewer shards than the paper (5 / 10 instead of 10 / 20) because the machine is shared: the Modular agent's 45 s code
# timeout includes a spawn + pandas import that takes 3-9 s under load. Shard count only changes wall time.
# Token usage per call goes to out/NAME/usage.jsonl (bedrock_llm.py; GPT-6 on Bedrock reports inputTokens = 2).
set -uo pipefail
NAME=$1; MODEL=$2; GUESS_FROM=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/ADS_folder/key.txt
export PYTHON=~/miniconda3/envs/ADS/bin/python MAX_TOKENS=128000 STRUCT_SHARDS=5 DECISION_SHARDS=10
export BEDROCK_MAX_ATTEMPTS=10 BEDROCK_READ_TIMEOUT=900 ACED_USAGE_LOG=$HERE/out/$NAME/usage.jsonl
mkdir -p "$HERE/out/$NAME/guess"
if [ -n "$GUESS_FROM" ]; then cp -n "$GUESS_FROM"/guess/zero_shot_guess_*.json "$HERE/out/$NAME/guess/"; fi
cat > "$HERE/out/$NAME/run_info.txt" <<EOF
started: $(date -u +%FT%TZ) on $(hostname)
model:   $MODEL (bedrock)   max_tokens: $MAX_TOKENS   shards: $STRUCT_SHARDS / $DECISION_SHARDS
code:    aced_benchmark/code (bedrock_llm.py 2026-10-20: last text block; Claude 5.x filter adaptations)
guess:   ${GUESS_FROM:-run here}
EOF
bash "$HERE/run_model.sh" "$NAME" bedrock "$MODEL" > "$HERE/out/$NAME/run.log" 2>&1
echo "[finished $(date -u +%FT%TZ)]" >> "$HERE/out/$NAME/run_info.txt"
