# SkyRL local patches used by RL v10 (copied into git 2026-10-24)

Training runs on SkyRL at commit `7eac0e83c96660e0820cb187f959ebe456c3d4ee` (`~/rpgrl/SkyRL` on ygg, mounted in the
`skyrl` container as `/work/SkyRL`) with these local changes. Before 10-24 they lived only in that working tree.

| file | what |
|---|---|
| `skyrl_local_patches_7eac0e83.diff` | `git diff` of the whole working tree against the commit (every local change; see below) |
| `rlv10_chunked_logprobs.py` | new file: `skyrl/backends/skyrl_train/workers/rlv10_chunked_logprobs.py` (untracked in SkyRL, so not in the diff) |
| `skyrl_logits_to_keep_and_chunked.diff` | the model_wrapper.py part alone, as tested on 10-23; identical to `personal_docs/rl_v10/skyrl_logits_to_keep_and_chunked.diff` and to the model_wrapper.py hunk of the full diff (checked 10-24) |

**What the full diff contains:**
- **`model_wrapper.py`** (10-23, RL v10): `RLV10_LOGITS_TO_KEEP=1` computes logits only for response positions;
  `RLV10_CHUNKED_LOGPROBS=1` computes log-probs through the frozen lm_head in chunks.
  - Both are opt-in, and `run_rlv10.sh` turns both on.
  - Exactness: CPU tests in `personal_docs/rl_v10/LOG_2026-10-23.md` (log-probs differ by 0.0; all 496 LoRA
    gradients are identical).
  - Smoke 6 (10-24, FSDP2, 4 GPUs) passed.
- **`skyrl_gym_generator.py`** (10-08): rollout log-probs padded to the output length when a turn generated 0 tokens.
- **`generators/utils.py`** (09-22): `slice_generator_output` slices dict-valued fields (`trajectory_time_splits`).
- **`trainer_utils.py`** (10-08): dynamic sampling resamples when every group in a batch has constant reward,
  instead of crashing.
- **`pyproject.toml` / `uv.lock`**: scipy added (the RPG env needs it).
- **Untracked** `examples/train/rpg` and `examples/train/rpg_v10`: symlinks to `dataset_generation_code/skyrl_rpg`
  and `dataset_generation_code/rl_v10/skyrl`.

**Apply to a clean checkout:**
```
git clone <SkyRL> && cd SkyRL && git checkout 7eac0e83c96660e0820cb187f959ebe456c3d4ee
git apply <this folder>/skyrl_local_patches_7eac0e83.diff
cp <this folder>/rlv10_chunked_logprobs.py skyrl/backends/skyrl_train/workers/
ln -s <ADS_shared>/dataset_generation_code/rl_v10/skyrl examples/train/rpg_v10
```

**Sandbox interpreter** (`RLV10_SANDBOX_PY`, `~/rpgrl/envs/rpg_sandbox`): pinned packages in `../../sandbox_env_freeze.txt`.
