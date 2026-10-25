#!/usr/bin/env bash
# In git since 2026-10-24 (was ~/rpgrl/rl_v10/evals/launch_eval.sh). THINKING is recorded in run_info.txt; the
# server decides it (serve_base9b.sh).
# Claim-style eval of a vLLM-served model on the FROZEN release harness (rl_v10/bench = rpg_benchmark v10.2 copy),
# by default full history and the deterministic resolver (no LLM). n independent samples per world (sampling is the model preset:
# Qwen T 1.0, top_p 0.95, top_k 20); simulator seeds are fixed per world, so samples differ only by the policy.
#   NAME=base9b_full PORTS="8201 ... 8206" SPLITS="core54 validation" N=4 CONC=6 bash launch_eval.sh
# Layout: ~/rpgrl/rl_v10/evals/$NAME/{shards/<split>/<k>/, <split>_s<seed>/shard_<k>/result_*.json, logs/}
# Protocol (2026-10-25): HISTORY=full|stateless (default full = the RL protocol), RESOLVER=det|llm (default det = no
# LLM). The paper protocol is HISTORY=stateless RESOLVER=llm (Opus 4.8 resolver, as the leaderboard; the paper's 9B row
# also ran thinking on with max_new_tokens 32768, out/gate_v10.2/qwen35_9b/run_info.txt). RESOLVER=llm calls Bedrock:
# source ~/ADS_folder/key.txt in the calling shell first (this script never reads keys).
set -u
B=~/ADS_folder/ADS_shared/dataset_generation_code/rl_v10/bench; DATA=${DATA:-~/ADS_folder/ADS_shared/rpg_benchmark/data}
PY=~/miniconda3/envs/ADS/bin/python; O=~/rpgrl/rl_v10/evals/$NAME; PORTS=($PORTS); NS=${#PORTS[@]}
MODEL=${MODEL:-qwen3.5-9b}; MAXTOK=${MAXTOK:-32768}
HISTORY=${HISTORY:-full}; RESOLVER=${RESOLVER:-det}
case "$HISTORY" in full|stateless) ;; *) echo "HISTORY must be full or stateless" >&2; exit 1 ;; esac
case "$RESOLVER" in det) RES_ARGS=(--no-resolver-llm) ;; llm) RES_ARGS=() ;; *) echo "RESOLVER must be det or llm" >&2; exit 1 ;; esac
if [ "$RESOLVER" = llm ] && [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then echo "RESOLVER=llm needs Bedrock credentials in the environment" >&2; exit 1; fi
export RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20
mkdir -p $O/logs
for sp in $SPLITS; do
  [ -d $O/shards/$sp ] && continue
  $PY - "$DATA/$sp" "$O/shards/$sp" "$NS" <<'PY'
import sys,glob,os
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
ws=sorted(glob.glob(os.path.join(src,"world_*.json.gz")))
for i,w in enumerate(ws):
    d=os.path.join(dst,f"{i%n:02d}"); os.makedirs(d,exist_ok=True); os.symlink(w,os.path.join(d,os.path.basename(w)))
print(f"{src}: {len(ws)} worlds in {n} shards")
PY
done
cat > $O/run_info.txt <<INFO
started:  $(date -u +%FT%TZ) on $(hostname)
harness:  frozen copy $B ($(cd $B && $PY -c "import agent_loop;print(agent_loop.HARNESS_VERSION)")); md5s in ../BENCH_FROZEN_MD5.txt
model:    $MODEL via vLLM ports ${PORTS[*]}   max_new_tokens $MAXTOK
protocol: --history $HISTORY, resolver $RESOLVER, budget 15, max turns 32, default v10.2 prompt, thinking ${THINKING:-unset (set by the server)}
splits:   $SPLITS   samples per world: $N   concurrency per process: ${CONC}
INFO
cd $B
for sp in $SPLITS; do for s in $(seq 1 $N); do for k in $(seq 0 $((NS-1))); do
  kk=$(printf %02d $k); out=$O/${sp}_s$s/shard_$kk
  (
    for try in $(seq 0 10); do
      [ $try -gt 0 ] && echo "[restart $try, $(date -u +%FT%TZ)]" >> $O/logs/${sp}_s${s}_$kk.log
      VLLM_BASE_URL=http://localhost:${PORTS[$k]}/v1 $PY run_benchmark.py --worlds-dir $O/shards/$sp/$kk --backend vllm \
        --model $MODEL --max-new-tokens $MAXTOK --history $HISTORY "${RES_ARGS[@]}" --concurrency $CONC \
        --outdir $out --resume -v >> $O/logs/${sp}_s${s}_$kk.log 2>&1 && break
      sleep 30
    done
    echo "[exit $?, $(date -u +%FT%TZ)]" >> $O/logs/${sp}_s${s}_$kk.log
  ) &
done; done; done
wait
echo "ALL DONE $(date -u +%FT%TZ)" >> $O/run_info.txt
