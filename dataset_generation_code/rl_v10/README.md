# RL on RPG v10 (`rl_v10/`)

GRPO + LoRA training of Qwen3.5-9B on the ACED-Repair (RPG v10.2) benchmark, with SkyRL. Written 2026-10-25. Logs and
results: `personal_docs/rl_v10/`. Start with `RL_V10_PRIORITIES.md`, `RL_FIXES_2026-10-24.md`,
`RL_DESIGN_REVIEW_2026-10-25.md` and `RL_START_DECISION_2026-10-25.md`.
**Known issues and open fixes (benchmark and RL): `personal_docs/KNOWN_ISSUES.md`. Read it before training.**

## 1. What an episode is

The RL environment is the benchmark loop itself:
- `stepenv.BenchStepEnv` runs the release harness's `agent_loop.run_world`, from the frozen copy in `bench/`, one
  turn at a time.
- The policy is plugged in as a stand-in LLM.
- So the prompt, the reply parser, the budget rules, the directives, the resolver, the code sandbox and the grader
  are the benchmark's own code. Nothing is re-implemented (the v9 RL env was, and the protocol gaps cost results).

`skyrl_env.RPGv10Env` wraps it for SkyRL. Its `init()` checks that the parquet's stored prompt equals a fresh
`reset()` (drift guard).

`bench/` = `rpg_benchmark` v10.2:
- frozen 2026-10-23;
- re-frozen 2026-10-25 with the hardened release sandbox (`BENCH_FROZEN_FROM.txt`, md5s in `BENCH_FROZEN_MD5.txt`).

Training also installs `sandbox_hardened.py` on top of it, and runs agent code on a pinned interpreter
(`RLV10_SANDBOX_PY`, package pins in `sandbox_env_freeze.txt`).

## 2. Design choices

| choice | RL | benchmark (paper) | why / evidence |
|---|---|---|---|
| interface | free text, the report lists every signal and control by name | same | deterministic resolver rejects 2.5% (thinking on) / 3.3% (off) of the base 9B's requested items, so an id menu would buy little and make RL a different task again |
| resolver | deterministic (`--no-resolver-llm`) | Opus 4.8 | no API cost, deterministic; grades identical on 1134/1134 stored episodes, shown text 98.6% |
| history | full conversation | stateless (last result + memory) | stateless 9B accepted 0/54; RL numbers are not comparable to the leaderboard row (see §5) |
| thinking (9B) | `THINKING=off` default (Vivian 10-24); `on` supported | on in the paper's 9B row | base part A, full history: core54 0.083 on / 0.037 off, validation 0.069 / 0.036. Compare a checkpoint only with the base under the same setting. Pilot both (RL_START_DECISION) |
| reward | accepted(core) + 0.5 · clip(benefit, 0, 1) · min(1, 4 / n_controls) | accepted(core) is the paper's metric | dense term for learning; the 4-control cap closes "every control at mid-range" (0.242 → 0.039). Harm is not penalised (benefit clipped at 0) |
| worlds | half standard, half v3 confounded (`cuefree.make_confounded`) | standard | on standard worlds a zero-experiment answer from passive correlations earns 0.222, above the base 9B (0.147); on v3 0.115 |
| domains | 8 training domains; clinical and fermentation never trained on | 10 | report test as seen (270) vs new (270) domains |
| structures | all 9 in training | 9 | each needs its own skill and the base is near zero; a leave-structures-out run is a follow-up |
| sampling | T 1.0, top_p 0.95, top_k 20, 32k output per turn, context 262,144 | same preset | the harness picks the Qwen preset only if the served name contains "qwen3" |
| per-world seeds | simulator seeds fixed per world | same | the 8 rollouts of a GRPO group differ only by the policy |
| training samples | step-wise (each turn trained on the exact prompt the benchmark showed) | n/a | token parity 457/457 (thinking on), 999/999 (off) |
| overlong turns | turn-sequences > 65,536 tokens dropped from the loss (`skyrl/trainer_rlv10.py`), counted per step | n/a | smoke 7 OOM at a 97,672-token history from one looping answer. Eval max is 56,020 (2,372 episodes) |

## 3. Data

All worlds are release-format world files (`to_release_format.py`). The data root (`RLV10_DATA_ROOT`) is
`~/rpgrl/rl_v10/data` on ygg and `/work/rl_v10/data` in the `skyrl` container; parquet `world_file` paths are relative
to it.

| set | folder under the data root | worlds | domains | use |
|---|---|---|---|---|
| train | `worlds_train` (seed 60M) | 1,350 | 8 training domains | source of the mix |
| train mix | `train_mix_v3/worlds_mix` | 1,341 (675 standard + 666 v3) | 8 | **training** (`train_mix_v3/train_mix_v3.parquet`) |
| validation | `worlds_validation` (= `rpg_benchmark/data/validation`) | 270 | clinical, fermentation | checkpoint selection |
| validation v3 | `worlds_validation_confounded_v3` | 268 | clinical, fermentation | checkpoint selection (cue-free) |
| in-domain validation | `worlds_val_indomain` (seed 62M) | 135 | 8 training domains | checkpoint selection |
| in-domain validation v3 | `worlds_val_indomain_confounded_v3` | 132 (3 failed the decoy audit) | 8 | checkpoint selection (cue-free) |
| test, core108, core54 | `rpg_benchmark/data/{test,core108,core54}` | 540 / 108 / 54 | all 10 | **read once** at the end, never for selection |

Build (offline, no API; `data/`):
1. `build_train.sh`;
2. `build_val_indomain.sh`;
3. `build_mix_and_parquets.sh`;
4. check with `python data/manifest.py check ~/rpgrl/rl_v10/data`.

Train-split checks:
- the same pipeline regenerates the 270 in-domain test worlds byte-identically;
- 0 world ids, SCMs or stories are shared with test/validation (`personal_docs/rl_v10/LOG_2026-10-23.md`).

The small evaluation splits are also in git (`data/worlds/`), so evals are reproducible without the data root.

Seed bases in use: 10/11/20/21/40/41/43/50–53M (benchmark and earlier sets), 60M (train), 62M (in-domain
validation). 61M was used only by an unstored probe.

`to_release_format.py` imports `personal_docs/scripts/v10/make_release.py` (the release record builder).

## 4. Training

Run inside the `skyrl` container from `/work/SkyRL` (SkyRL commit `7eac0e83` + `skyrl/patches/`, see its README):

```
RUN=<name> NUM_GPUS=4 STEPS=<n> THINKING=off bash examples/train/rpg_v10/run_rlv10.sh
```

Defaults (2026-10-25):
- `TRAIN` = the mix; `VAL` = validation;
- 8 worlds × 8 rollouts per update; LoRA r16 / α32;
- lr 5e-6; no KL;
- GRPO without std normalisation; `token_mean` loss;
- `RLV10_MAX_TRAIN_SEQ=65536`.

Per-step metrics to watch:
- `rlv10/overlong_dropped`, `rlv10/max_seq_tokens`, `rlv10/length_stop_frac`;
- per episode: `cap_cut_turns`, `thinking_cut_turns`, `suspicious_code_turns`, `sandbox_refusals`.

Traces of every episode go to `RLV10_TRACE_DIR`.

Tests:
- `test_reward_v10.py` (reward == D1 rescoring);
- `test_sandbox_isolation.py` (red team);
- `replay_parity.py` (env == benchmark);
- `skyrl/test_trainer_rlv10.py` (overlong drop inside SkyRL's advantage path; run in the container).

## 5. Evaluation

`eval/eval_ckpt.sh RUN STEP "GPUS" PORT0 "SPLITS" [N]` merges the LoRA and serves it. It then runs the frozen harness
through `eval/launch_eval.sh`, n = 4 at T 1.0 per world, and `eval/eval_stats.py` summarises.

- **RL protocol** (default): `HISTORY=full RESOLVER=det`, `THINKING` as trained.
- **Paper protocol** (to sit next to the leaderboard row): `HISTORY=stateless RESOLVER=llm`, on core108. It calls
  Bedrock for the resolver, so source the key file in the calling shell first. The paper's 9B row ran thinking on.
- **Base rates** to compare with, same protocol: `~/rpgrl/rl_v10/evals/base9b_full` (thinking on),
  `base9b_nothink` (off), `base9b_nothink_v3` (off, validation v3).

Claims: paired over the same worlds, world bootstrap CIs, checkpoint chosen on validation / in-domain validation, test
read once.
