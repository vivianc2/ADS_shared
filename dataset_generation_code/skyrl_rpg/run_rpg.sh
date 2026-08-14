#!/usr/bin/env bash
# GRPO + LoRA training on the RPG causal-discovery env (Qwen3-8B; 14B = change model path).
# Run INSIDE the SkyRL container, from the SkyRL repo root, after:
#   1) this package symlinked in:  ln -s /work/ADS_shared/dataset_generation_code/skyrl_rpg examples/train/rpg
#   2) science dep present:        uv pip install scipy        # oracle uses scipy.stats
#   3) dataset built:              uv run --isolated python -m examples.train.rpg.rpg_dataset \
#                                     --output_dir /work/data/rpg --train_size 512 --val_size 64 \
#                                     --archetypes confounded_chain
#   4) GPUs: avoid GPU0 (host eval server) -> export CUDA_VISIBLE_DEVICES=1,2,3,4
# Then: bash examples/train/rpg/run_rpg.sh
set -x

DATA_DIR="${DATA_DIR:-/work/data/rpg}"
NUM_GPUS="${NUM_GPUS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"      # final version: Qwen/Qwen3-14B (may need more GPUs / lower mem-util)
LOGGER="${LOGGER:-console}"

uv run --isolated --extra fsdp -m examples.train.rpg.main_rpg \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
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
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.max_prompt_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=20 \
  environment.env_class=rpg \
  trainer.logger="$LOGGER" \
  trainer.project_name="rpg_v7" \
  trainer.run_name="rpg_qwen3_8b_grpo_lora" \
  trainer.ckpt_path="/work/rl_ckpt/rpg_skyrl" \
  $@

# --- refinements to add once the base run works (confirm exact flag names in this SkyRL
#     build; our design doc §11/§13):
#   Dr.GRPO (mean-subtract, no std-normalization) — disable GRPO std-norm
#   DAPO dynamic sampling — drop zero-variance groups (dynamic_sampling=filter)
#   These are config flags; leaving defaults (std-norm on, no dynamic sampling) is a valid
#   plain-GRPO first run.
