#!/usr/bin/env bash
# Merge an RL v10 LoRA checkpoint and evaluate it with the same harness/protocol as the base-rate eval.
# In git since 2026-10-24 (was ~/rpgrl/rl_v10/evals/eval_ckpt.sh); THINKING (default off) is passed to the server.
#   eval_ckpt.sh RUN STEP "GPUS" PORT0 "SPLITS" [N]
#   e.g. eval_ckpt.sh rlv10_pilot1 10 "4 5 6 7" 8301 "validation validation_cuefree core54" 4
# BASE = the model the LoRA was trained on (default base Qwen3.5-9B). A split is looked up in rpg_benchmark/data
# (validation, test, core54, core108, ...), else as ~/rpgrl/rl_v10/data/worlds_<split> (val_indomain,
# val_indomain_confounded_v3, validation_confounded_v3), linked into ~/rpgrl/rl_v10/evals/_data.
# HISTORY / RESOLVER pass through to launch_eval.sh (paper protocol: HISTORY=stateless RESOLVER=llm).
# Choose checkpoints on validation / val_indomain (+ their _confounded_v3 pairs); read test / core108 once at the end.
set -u
RUN=$1; STEP=$2; GPUS=($3); P0=$4; SPLITS=$5; N=${6:-4}
W=$HOME/rpgrl; E=$W/rl_v10/evals; NAME=${RUN}_s${STEP}
SNAP=c202236235762e1c871ad0ccb60c8ee5ba337b9a
BASE=${BASE:-/work/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$SNAP}
CK=/work/rl_ckpt/rl_v10/$RUN/global_step_$STEP/policy/lora_adapter
OUT=/work/merged/rl_v10/$NAME
mkdir -p $W/logs $W/merged/rl_v10
docker exec -u root skyrl bash -lc "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /work/SkyRL/.venv/bin/python /work/pd_tooling/box2_eval_harness/merge_lora.py $BASE $CK $OUT" > $W/logs/merge_$NAME.log 2>&1
grep "MERGED" $W/logs/merge_$NAME.log; grep -q "missed: 0" $W/logs/merge_$NAME.log || { echo "ABORT: merge did not apply every LoRA weight (see $W/logs/merge_$NAME.log)"; exit 1; }
B=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/$SNAP
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja preprocessor_config.json video_preprocessor_config.json; do
  [ -f $W/merged/rl_v10/$NAME/$f ] || cp $B/$f $W/merged/rl_v10/$NAME/ 2>/dev/null
done
PORTS=""; for i in "${!GPUS[@]}"; do PORTS="$PORTS $((P0+i))"; done
# the harness picks the Qwen sampling preset (T 1.0, top_p 0.95, top_k 20) by "qwen3" in the model name; without it
# the client silently falls back to T 0.3 / top_p 0.9. So the served name must contain "qwen3".
SERVED=qwen3.5-9b-$NAME
HERE=$(cd "$(dirname "$0")" && pwd)
THINKING=${THINKING:-off}   # must match how the checkpoint was trained (run_rlv10.sh THINKING, default off)
MODEL=$W/merged/rl_v10/$NAME NAME=$SERVED THINKING=$THINKING bash $HERE/serve_base9b.sh "${GPUS[*]}" $P0
for p in $PORTS; do until curl -s -o /dev/null localhost:$p/v1/models; do sleep 15; done; done
echo "servers up $(date -u +%FT%TZ)"
for sp in $SPLITS; do
  D=$HOME/ADS_folder/ADS_shared/rpg_benchmark/data
  if [ ! -d $D/$sp ]; then
    D=$E/_data; mkdir -p $D
    [ -e $D/$sp ] || ln -s $W/rl_v10/data/worlds_$sp $D/$sp
    [ -d $D/$sp ] || { echo "no split $sp" >&2; exit 1; }
  fi
  TAG=""; [ "${HISTORY:-full}" = full ] || TAG="${TAG}_$HISTORY"; [ "${RESOLVER:-det}" = det ] || TAG="${TAG}_res$RESOLVER"
  DATA=$D NAME=${NAME}_$sp$TAG MODEL=$SERVED PORTS="$PORTS" SPLITS=$sp N=$N CONC=12 THINKING=$THINKING bash $HERE/launch_eval.sh
done
for p in $PORTS; do pid=$(pgrep -f "vllm serve .* --port $p" | head -1); [ -n "$pid" ] && kill $pid; done
echo "done $(date -u +%FT%TZ)"
