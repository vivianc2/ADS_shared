#!/usr/bin/env bash
# GRPO + LoRA training on dose-window RPG worlds with the original combined reward.
# Run inside the SkyRL container from /work/SkyRL.
set -euo pipefail

WANDB_KEY_FILE="${WANDB_KEY_FILE:-/work/wandb_key.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_KEY_FILE" ]]; then
  # The credentials file is sourced before xtrace so its value is never printed.
  source "$WANDB_KEY_FILE"
fi

set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export HF_HOME="${HF_HOME:-/work/hf_cache}"
export RPG_PROTO="${RPG_PROTO:-rpg_v9}"
export PYTHONPATH="/work/ADS_shared/dataset_generation_code/rpg_rl_exps${PYTHONPATH:+:$PYTHONPATH}"
WORK_DIR="${WORK_DIR:-/work/ADS_shared/dataset_generation_code/rpg_rl_exps/long_rl_single_arch5e-7}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/long_rl_single_arch5e-7_cache}"
export WANDB_DIR="${WANDB_DIR:-$WORK_DIR/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WORK_DIR/wandb-data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/long_rl_single_arch5e-7_wandb_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/long_rl_single_arch5e-7_triton_cache}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/long_rl_single_arch5e-7_tilelang_cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/long_rl_single_arch5e-7_flashinfer}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/long_rl_single_arch5e-7_project_uv_cache}"
export VLLM_NO_USAGE_STATS=1

TRAIN_DATA="${TRAIN_DATA:-$WORK_DIR/dose_window_training.parquet}"
VAL_DATA="${VAL_DATA:-/work/ADS_shared/dataset_generation_code/rpg_v9/data_v9_deleaked/validation.parquet}"
NUM_GPUS="${NUM_GPUS:-2}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
LOGGER="${LOGGER:-wandb}"
CKPT_DIR="${CKPT_DIR:-/data/long_rl_single_arch5e-7/checkpoints}"
EXPORT_DIR="${EXPORT_DIR:-$WORK_DIR/exports}"

mkdir -p "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" \
  "$UV_CACHE_DIR" "$CKPT_DIR" "$EXPORT_DIR"

uv run --isolated --extra fsdp -m long_rl_single_arch5e-7.main_rpg_long \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.batched=false \
  generator.n_samples_per_prompt=8 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.algorithm.use_kl_loss=false \
  trainer.epochs=1 \
  trainer.max_training_steps=300 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.max_prompt_length=4096 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.eval_before_train=true \
  trainer.eval_interval=15 \
  trainer.ckpt_interval=25 \
  environment.env_class=rpg \
  trainer.logger="$LOGGER" \
  trainer.project_name="rpg_long_rl_single_arch5e-7" \
  trainer.run_name="rpg_qwen3.5_9b_grpo_lora_dose_window_original_reward_lr5e-7" \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.remove_microbatch_padding=false \
  "$@"
