#!/usr/bin/env bash
# GRPO + LoRA on RPG v10 through the release benchmark loop (rl_v10). Run INSIDE the skyrl container from /work/SkyRL:
#   RUN=<name> NUM_GPUS=4 STEPS=2 bash examples/train/rpg_v10/run_rlv10.sh [extra overrides]
# Protocol: full history, deterministic resolver, v10.2 prompt/parser/budget, Qwen preset sampling (T 1.0, top_p 0.95,
# top_k 20), 32k output per turn, context up to the model maximum (262144). Training on the exact prompts the benchmark
# shows: generator.step_wise_trajectories=true (see personal_docs/rl_v10/LOG_2026-10-23.md).
# THINKING=off (default since 2026-10-24, Vivian: the 9B RL line has always run thinking off, for stability) passes
# enable_thinking=false to the chat template; THINKING=on restores the 10-23 smoke-run setting. Evaluate a trained
# checkpoint with the same setting (rl_v10/eval/eval_ckpt.sh, THINKING passed through) and compare it with the base
# model under the same setting (thinking-off base: ~/rpgrl/rl_v10/evals/base9b_nothink).
# 2026-10-25 defaults:
#   VAL   = validation (was core54, a subset of the paper's test set: never select checkpoints on it);
#   TRAIN = the standard + v3 mix (was train.parquet, standard only);
#   BATCH/MINI = 8 worlds per update (was 4, the smoke-run size);
#   RLV10_MAX_TRAIN_SEQ = 65536: longer turn-sequences are dropped from the loss (skyrl/trainer_rlv10.py; smoke 7 OOM).
# Design notes: rl_v10/README.md.
set -euo pipefail
ROOT=/work
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/wandb_key.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_KEY_FILE" ]]; then set -a; . "$WANDB_KEY_FILE"; set +a; fi
set -x
export RLV10_SRC=$ROOT/ADS_shared/dataset_generation_code/rl_v10
export RLV10_DATA_ROOT="${RLV10_DATA_ROOT:-$ROOT/rl_v10/data}"
export RLV10_SANDBOX_PY="${RLV10_SANDBOX_PY:-$ROOT/envs/rpg_sandbox/bin/python}"
export RLV10_EPISODE_ROOT="${RLV10_EPISODE_ROOT:-/tmp/rlv10_episodes}"; mkdir -p "$RLV10_EPISODE_ROOT"
RUN="${RUN:?set RUN}"
export RLV10_TRACE_DIR="${RLV10_TRACE_DIR:-$ROOT/rl_v10/traces/$RUN}"
export RLV10_W_ACC="${RLV10_W_ACC:-1.0}" RLV10_W_A="${RLV10_W_A:-0.0}" RLV10_W_BENEFIT="${RLV10_W_BENEFIT:-0.5}" RLV10_W_B="${RLV10_W_B:-0.0}"
export RLV10_LOGITS_TO_KEEP="${RLV10_LOGITS_TO_KEEP:-1}"   # SkyRL local patches, personal_docs/rl_v10/skyrl_logits_to_keep_and_chunked.diff
export RLV10_CHUNKED_LOGPROBS="${RLV10_CHUNKED_LOGPROBS:-1}"
export RLV10_MAX_TRAIN_SEQ="${RLV10_MAX_TRAIN_SEQ:-65536}"   # 0 = keep every turn-sequence (trainer_rlv10.py)
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
NUM_GPUS="${NUM_GPUS:-4}"; MODEL="${MODEL:-Qwen/Qwen3.5-9B}"; STEPS="${STEPS:-2}"; LOGGER="${LOGGER:-wandb}"
TRAIN="${TRAIN:-$RLV10_DATA_ROOT/train_mix_v3/train_mix_v3.parquet}"; VAL="${VAL:-$RLV10_DATA_ROOT/validation.parquet}"
MAXCTX=262144; GEN=32768; PROMPT=$((MAXCTX - GEN))
THINKING="${THINKING:-off}"
case "$THINKING" in
  off) THINK_ARGS=(generator.chat_template_kwargs.enable_thinking=false) ;;
  on)  THINK_ARGS=() ;;
  *) echo "THINKING must be on or off" >&2; exit 1 ;;
esac
export RLV10_THINKING="$THINKING"   # stepenv.split_thinking: thinking off = no reasoning parser in eval, reply is all content
mkdir -p $ROOT/logs
uv run --no-sync --extra fsdp -m examples.train.rpg_v10.main_rpg_v10 \
  data.train_data="['$TRAIN']" data.val_data="['$VAL']" \
  environment.env_class=rpg_v10 environment.skyrl_gym.max_env_workers=256 \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model.lora.rank=16 trainer.policy.model.lora.alpha=32 trainer.policy.model.lora.target_modules=all-linear \
  trainer.strategy=fsdp trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.gpu_memory_utilization="${GPU_UTIL:-0.7}" \
  generator.inference_engine.max_num_batched_tokens=65536 generator.inference_engine.max_num_seqs=256 \
  generator.batched=false generator.use_conversation_multi_turn=true generator.step_wise_trajectories=true \
  generator.max_turns=32 "${THINK_ARGS[@]}" \
  generator.n_samples_per_prompt="${N_SAMPLES:-8}" \
  generator.max_input_length=$PROMPT trainer.max_prompt_length=$PROMPT \
  generator.sampling_params.max_generate_length=$GEN generator.eval_sampling_params.max_generate_length=$GEN \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=0.95 generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  generator.eval_sampling_params.temperature=1.0 generator.eval_sampling_params.top_p=0.95 \
  generator.eval_sampling_params.top_k=20 generator.eval_sampling_params.min_p=0.0 \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.advantage_batch_normalize=false \
  trainer.algorithm.use_kl_loss="${USE_KL:-false}" trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.optimizer_config.lr="${LR:-5.0e-6}" trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.train_batch_size="${BATCH:-8}" trainer.policy_mini_batch_size="${MINI:-8}" trainer.update_epochs_per_batch=1 \
  trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch="${TOK_MB:-32768}" trainer.remove_microbatch_padding=false \
  trainer.epochs=10 trainer.max_training_steps="$STEPS" \
  trainer.eval_before_train="${EVAL_BEFORE:-false}" trainer.eval_interval="${EVAL_INTERVAL:-1000}" \
  trainer.ckpt_interval="${CKPT_INTERVAL:-1000}" trainer.hf_save_interval=999 trainer.resume_mode=latest \
  trainer.logger="$LOGGER" trainer.project_name=rpg_rl_v10 trainer.run_name="$RUN" \
  trainer.ckpt_path="$ROOT/rl_ckpt/rl_v10/$RUN" trainer.export_path="$ROOT/rl_ckpt/rl_v10/${RUN}_hf" \
  "$@" 2>&1 | tee "$ROOT/logs/$RUN.log"
