#!/usr/bin/env bash
# HARD-ARCHETYPE GRPO with the LEVER-IDENTIFICATION reward gate (2026-09-21).
#
# Trains Qwen3.5-9B (LoRA) from base on competing_causes / synergy_pair / hidden_subtype x the 8
# train skins; SkyRL-internal eval on the same archetypes x the 2 held-out skins (clinical,
# fermentation) = domain transfer within the trained archetypes. Dataset: build_hard_arch_ds.py.
#
# Reward (rpg_rl/reward.py): REWARD_MODE selects
# !! 2026-09-22: the ORIGINAL gate_any/gate_full/lever_only modes are SHOTGUN-HACKABLE. An answer
# !! that names every actuator passes all three (nothing caps |chosen|, and a superset always
# !! contains `must`), scoring 0.92/0.92/1.00 vs gold 1.00 with ZERO causal knowledge, and
# !! E[reward | k knobs named] rises monotonically in k. A `lever_ok` curve from those modes is
# !! NOT evidence of identification. Use the *_px1 modes below, which add
# !! RPG_LEVER_PRECISION=1 RPG_LEVER_MAX_EXTRA=1 so reward PEAKS at |chosen| = |causal| and a
# !! shotgun scores 0. Measurements: personal_docs/results/lever_gate_preflight/,
# !! personal_docs/rl_logs/reward_lever_gate_2026-09-21.md §9.
#
# !! P0 base-rate measurement (2026-09-22, base 9B, 120 held-out hard worlds x n=8) further shows
# !! the BINARY any-gate is nearly a no-op here: base any-mode lever_ok is already 57.7%
# !! (72.5 competing / 58.7 subtype / 42.4 synergy) while full-mode is 0.9%. The model names 0.82
# !! levers against a causal set of 2.33 -- the deficit on these archetypes is COMPLETENESS, not
# !! picking a non-causal knob. So prefer the recall-scaled mode, which grades the axis with the
# !! headroom (base mean 0.27, 59% of episodes strictly between 0 and 1):
# !! AND the multiplicative recall form (gate_any_rx1) turned out to CRUSH the GRPO signal --
# !! replayed on the 960 real base rollouts its mean within-group std is 0.076 with 18% degenerate
# !! groups, vs 0.171/0% for r1. Use the ADDITIVE form instead:
#   gate_any_add (RECOMMENDED) gate + at most 1 spurious lever + identification as its OWN
#                              additive term: 0.5*(recall*precision) + 0.3*part_a + 0.2*part_b.
#                              Replay on the base rollouts: mean 0.223, within-group std 0.162,
#                              0% degenerate groups, gold = 1.00.
#   gate_any_rx1 gate_any + recall*precision MULTIPLIER + at most 1 spurious lever. Correct
#                ordering (gold 1.000 / one-of-two 0.282 / shotgun 0.000) but sparse -- kept for
#                the ablation, not recommended as the main run.
#   gate_any_px1 gate_any + precision scaling + at most 1 spurious lever.
#   gate_full_px1              gate_full + the same anti-shotgun terms.
#   lever_only_px1             lever_only + the same (binary id, precision-scaled).
#
#   gate_any   (LEGACY/hackable) RPG_LEVER_GATE=1  RPG_LEVER_MODE=any  -> 0 unless a causal lever is named;
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

# RPG_ROOT: repo parent. /work inside the box1/box2 skyrl container; on a bare-metal
# box (e.g. the H200 box, glibc 2.35, no docker) set it to the checkout parent.
RPG_ROOT="${RPG_ROOT:-/work}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$RPG_ROOT/wandb_key.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_KEY_FILE" ]]; then
  set -a; . "$WANDB_KEY_FILE"; set +a
fi

REWARD_MODE="${REWARD_MODE:-gate_any_add}"
# anti-shotgun terms; the *_px1 / *_rx1 modes switch these on (see the banner above)
# remember any caller-supplied overrides BEFORE we reset the defaults, so a mode's own values
# don't get clobbered by the reset (a bare `:-` sees the reset value, not "unset").
_U_W_ID="${RPG_W_ID:-}" _U_W_A="${RPG_W_A:-}" _U_W_B="${RPG_W_B:-}"
export RPG_LEVER_PRECISION=0 RPG_LEVER_MAX_EXTRA=-1 RPG_LEVER_EXACT=0 RPG_LEVER_SCALE=none RPG_W_ID=0
case "$REWARD_MODE" in
  gate_any_add)   export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any \
                         RPG_LEVER_SCALE=precision RPG_LEVER_MAX_EXTRA=1 \
                         RPG_W_ID="${_U_W_ID:-0.5}" RPG_W_A="${_U_W_A:-0.3}" RPG_W_B="${_U_W_B:-0.2}" ;;
  gate_any_rx1)   export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any \
                         RPG_LEVER_SCALE=recall RPG_LEVER_MAX_EXTRA=1 ;;
  lever_only_rx1) export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=1 RPG_LEVER_MODE=any \
                         RPG_LEVER_SCALE=recall RPG_LEVER_MAX_EXTRA=1 ;;
  gate_any_px1)   export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any \
                         RPG_LEVER_PRECISION=1 RPG_LEVER_MAX_EXTRA=1 ;;
  gate_full_px1)  export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=full \
                         RPG_LEVER_PRECISION=1 RPG_LEVER_MAX_EXTRA=1 ;;
  lever_only_px1) export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=1 RPG_LEVER_MODE=any \
                         RPG_LEVER_PRECISION=1 RPG_LEVER_MAX_EXTRA=1 ;;
  gate_any)   export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any ;;   # legacy, hackable
  gate_full)  export RPG_LEVER_GATE=1 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=full ;;  # legacy, hackable
  lever_only) export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=1 RPG_LEVER_MODE=any ;;   # legacy, hackable
  r1)         export RPG_LEVER_GATE=0 RPG_LEVER_ONLY=0 RPG_LEVER_MODE=any ;;
  *) echo "unknown REWARD_MODE=$REWARD_MODE" >&2; exit 2 ;;
esac

set -x
export RPG_SRC="${RPG_SRC:-$RPG_ROOT/ADS_shared/dataset_generation_code}"
export RPG_PROTO="${RPG_PROTO:-rpg_v9}" RPG_SYNERGY_SOFT="${RPG_SYNERGY_SOFT:-20}"
export RPG_W_A="${RPG_W_A:-0.5}" RPG_W_B="${RPG_W_B:-0.5}" RPG_LEVER_BONUS="${RPG_LEVER_BONUS:-0.0}"
# (the case block above already pinned RPG_W_A/RPG_W_B for gate_any_add, so :- keeps them)
export DATA_DIR="${DATA_DIR:-$RPG_SRC/rpg_v9/experiment_datasets/rl_train/rl_hard_lever_ds}"
export NUM_GPUS="${NUM_GPUS:-8}" LOGGER="${LOGGER:-wandb}" MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
export HF_HOME="${HF_HOME:-$RPG_ROOT/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="${RUN_NAME:-hard3_${REWARD_MODE}_9b}"
CKPT_ROOT="${CKPT_ROOT:-$RPG_ROOT/rl_ckpt}"
LOG_DIR="${LOG_DIR:-$RPG_ROOT/logs}"; mkdir -p "$LOG_DIR"
STEPS="${STEPS:-150}"

bash examples/train/rpg/run_rpg.sh \
  trainer.policy.model.lora.target_modules=all-linear \
  generator.n_samples_per_prompt=8 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.grpo_norm_by_std=false trainer.algorithm.advantage_batch_normalize=false \
  trainer.algorithm.dynamic_sampling.type="${DYN_SAMPLING:-filter}" \
  trainer.algorithm.dynamic_sampling.max_sample_batches="${DYN_MAX_BATCHES:-30}" \
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
