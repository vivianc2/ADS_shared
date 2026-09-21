#!/usr/bin/env bash
# HARD-ARCHETYPE GRPO with the LEVER-IDENTIFICATION reward gate (2026-09-21).
#
# Trains Qwen3.5-9B (LoRA) from base on competing_causes / synergy_pair / hidden_subtype x the 8
# train skins; SkyRL-internal eval on the same archetypes x the 2 held-out skins (clinical,
# fermentation) = domain transfer within the trained archetypes. Dataset: build_hard_arch_ds.py.
#
# Reward (rpg_rl/reward.py): REWARD_MODE selects
#   gate_any   (default) RPG_LEVER_GATE=1  RPG_LEVER_MODE=any  -> 0 unless a causal lever is named;
#                        on top, 0.5*benefit + 0.5*battery, so both co-causes beat one (gradient
#                        toward completeness) while one-of-two still gives the group variance.
#   gate_full            RPG_LEVER_GATE=1  RPG_LEVER_MODE=full -> both co-causes / subtype treatment
#                        required. WARNING: base 9B is ~0% on these -> mostly all-zero groups.
#   lever_only           RPG_LEVER_ONLY=1  RPG_LEVER_MODE=any  -> binary "found a causal lever".
#   r1                   the unchanged 0.5A+0.5B reward (control).
#
# Trainer config = box2's collapse-fixed RL300-KL recipe (personal_docs/box_sync/box2_repro_2026-09-17/
# run_rl300_kl.sh): KL 0.001 to base, cosine LR 5e-6, rep-penalty 1.05, gen 4096, prompt 18432,
# Dr.GRPO (no std-norm), n=8, from BASE. Lengths are at the model max per CLAUDE.md rule #1.
#
# Run INSIDE the SkyRL container from the SkyRL repo root (examples/train/rpg -> skyrl_rpg symlink):
#   REWARD_MODE=gate_any bash examples/train/rpg/run_rpg_hard_lever.sh
# Requires: /work/wandb_key.txt (WANDB_API_KEY=... ; sourced before xtrace, never printed).
set -euo pipefail

WANDB_KEY_FILE="${WANDB_KEY_FILE:-/work/wandb_key.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_KEY_FILE" ]]; then
  set -a; . "$WANDB_KEY_FILE"; set +a
fi

REWARD_MODE="${REWARD_MODE:-gate_any}"
case "$REWARD_MODE" in
  gate_any)   export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any ;;
  gate_full)  export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=full ;;
  lever_only) export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=1 RPG_LEVER_MODE=any ;;
  r1)         export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any ;;
  *) echo "unknown REWARD_MODE=$REWARD_MODE" >&2; exit 2 ;;
esac

set -x
export RPG_SRC="${RPG_SRC:-/work/ADS_shared/dataset_generation_code}"
export RPG_PROTO="${RPG_PROTO:-rpg_v9}" RPG_SYNERGY_SOFT="${RPG_SYNERGY_SOFT:-20}"
export RPG_W_A="${RPG_W_A:-0.5}" RPG_W_B="${RPG_W_B:-0.5}" RPG_LEVER_BONUS="${RPG_LEVER_BONUS:-0.0}"
export DATA_DIR="${DATA_DIR:-$RPG_SRC/rpg_v9/experiment_datasets/rl_train/rl_hard_lever_ds}"
export NUM_GPUS="${NUM_GPUS:-8}" LOGGER="${LOGGER:-wandb}" MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
export HF_HOME="${HF_HOME:-/work/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="${RUN_NAME:-hard3_${REWARD_MODE}_9b}"
CKPT_ROOT="${CKPT_ROOT:-/work/rl_ckpt}"
LOG_DIR="${LOG_DIR:-/work/logs}"; mkdir -p "$LOG_DIR"
STEPS="${STEPS:-150}"

bash examples/train/rpg/run_rpg.sh \
  trainer.policy.model.lora.target_modules=all-linear \
  generator.n_samples_per_prompt=8 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.grpo_norm_by_std=false trainer.algorithm.advantage_batch_normalize=false \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.optimizer_config.lr=5.0e-6 trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=2 trainer.policy.optimizer_config.scheduler=cosine \
  trainer.train_batch_size=8 trainer.policy_mini_batch_size=2 trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=18432 generator.max_input_length=18432 \
  generator.sampling_params.max_generate_length=4096 generator.eval_sampling_params.max_generate_length=4096 \
  generator.sampling_params.repetition_penalty=1.05 generator.eval_sampling_params.repetition_penalty=1.05 \
  generator.eval_sampling_params.temperature=1.0 generator.inference_engine.max_num_batched_tokens=18432 \
  generator.inference_engine.gpu_memory_utilization=0.7 generator.inference_engine.max_num_seqs=512 \
  trainer.max_tokens_per_microbatch=4096 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.remove_microbatch_padding=false \
  trainer.epochs=10 trainer.eval_before_train=true trainer.max_training_steps="$STEPS" \
  trainer.eval_interval=15 trainer.ckpt_interval=5 trainer.resume_mode=latest \
  trainer.project_name=rpg_rl trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$CKPT_ROOT/$RUN_NAME" trainer.export_path="$CKPT_ROOT/${RUN_NAME}_hf" \
  trainer.hf_save_interval=999 "$@" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
