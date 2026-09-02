#!/usr/bin/env python3
"""Pre-launch smoke test (requirement 9).

Three things are checked, all on CPU, in a few seconds, against the artifacts that the
runs will actually consume:

1. **Prompt injection** -- each built dataset really carries its own prompt, the three
   datasets are otherwise identical (same rows, same order, same worlds), the prompt
   files match, and the runtime assertion in ``sky_env`` accepts the right prompt and
   rejects the wrong one.
2. **Exact step count** -- the resolved SkyRL config gives exactly 8 optimizer steps
   with evaluations at 0, 4 and 8, and the training set is large enough to supply them.
3. **Output-directory isolation** -- every run-scoped path (checkpoints, exports, logs,
   Ray temp, LoRA sync, episode scratch, W&B, compiler caches) is pairwise disjoint
   across p1/p2/p3, and the three W&B run ids differ but are deterministic.

Run it with:  bash scripts/in_container.sh python -m prompt_compare_rl.smoke_test
Exit code 0 means the experiment is safe to launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from prompt_compare_rl import prompts as prompt_lib  # noqa: E402
from prompt_compare_rl.config import (  # noqa: E402
    EVAL_INTERVAL,
    EVAL_STEPS,
    MAX_TRAINING_STEPS,
    PROMPT_IDS,
    ExperimentConfig,
    run_env,
    run_overrides,
)


class SmokeFailure(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


# --------------------------------------------------------------------------------------
# 1. prompt injection
# --------------------------------------------------------------------------------------


def check_prompt_injection(cfg: ExperimentConfig) -> Dict[str, Any]:
    import pandas as pd

    texts = prompt_lib.read_materialized(cfg.prompt_dir)
    digests = {pid: prompt_lib.sha256_text(texts[pid]) for pid in PROMPT_IDS}
    _check(len(set(digests.values())) == 3, f"prompts are not distinct: {digests}")

    details: Dict[str, Any] = {"prompt_sha256": digests, "splits": {}}

    for split, filename in (("train", "train.parquet"), ("validation", "validation_small.parquet")):
        frames = {}
        for pid in PROMPT_IDS:
            path = cfg.dataset_dir(pid) / filename
            _check(path.exists(), f"missing dataset {path} -- run build_dataset.py")
            frames[pid] = pd.read_parquet(path)

        reference = frames[PROMPT_IDS[0]]
        n_rows = len(reference)
        _check(n_rows > 0, f"{split} dataset is empty")

        for pid, frame in frames.items():
            _check(len(frame) == n_rows, f"{split}/{pid} has {len(frame)} rows, expected {n_rows}")
            # the selected prompt, and only it, is in the system slot of every row
            row_digests = {
                hashlib.sha256(row[0]["content"].encode("utf-8")).hexdigest() for row in frame["prompt"]
            }
            _check(
                row_digests == {digests[pid]},
                f"{split}/{pid}: system prompt digests {row_digests} != {{{digests[pid]}}}",
            )
            _check(
                all(row[0]["role"] == "system" and row[1]["role"] == "user" for row in frame["prompt"]),
                f"{split}/{pid}: expected [system, user] message roles",
            )

        # everything except the system message must be identical across the three copies
        for pid in PROMPT_IDS[1:]:
            frame = frames[pid]
            for index in range(n_rows):
                a, b = reference.iloc[index], frame.iloc[index]
                _check(
                    a["prompt"][1]["content"] == b["prompt"][1]["content"],
                    f"{split} row {index}: observation differs between p1 and {pid}",
                )
                _check(
                    dict(a["extra_info"]) == dict(b["extra_info"]),
                    f"{split} row {index}: extra_info differs between p1 and {pid}",
                )
                _check(
                    a["data_source"] == b["data_source"] and a["env_class"] == b["env_class"],
                    f"{split} row {index}: data_source/env_class differ between p1 and {pid}",
                )

        details["splits"][split] = {
            "rows": n_rows,
            "worlds_sha256": hashlib.sha256(
                json.dumps(
                    [
                        [dict(r)["seed"], dict(r)["skin"], dict(r)["archetype"]]
                        for r in reference["extra_info"]
                    ],
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }

    return details


def check_prompt_length_parity(cfg: ExperimentConfig) -> Dict[str, Any]:
    """No prompt may be filtered out by ``trainer.max_prompt_length``.

    SkyRL drops rows whose templated prompt exceeds ``max_prompt_length``. p1 is the
    longest of the three prompts, so if that filter ever bit, the three runs would train
    on DIFFERENT world sets and the comparison would be invalid.
    """
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    limit = cfg.max_prompt_length
    report: Dict[str, Any] = {"max_prompt_length": limit, "longest": {}}
    for split, filename in (("train", "train.parquet"), ("validation", "validation_small.parquet")):
        for pid in PROMPT_IDS:
            frame = pd.read_parquet(cfg.dataset_dir(pid) / filename)
            longest = 0
            for row in frame["prompt"]:
                tokens = tokenizer.apply_chat_template(
                    [dict(m) for m in row], add_generation_prompt=True, tokenize=True, return_dict=False
                )
                longest = max(longest, len(tokens))
            report["longest"][f"{split}/{pid}"] = longest
            _check(
                longest <= limit,
                f"{split}/{pid}: longest prompt {longest} tokens exceeds max_prompt_length {limit}; "
                "rows would be filtered and the runs would no longer share worlds",
            )
    return report


def check_runtime_prompt_assertion(cfg: ExperimentConfig) -> Dict[str, Any]:
    """The env must accept the configured prompt and reject any other one."""
    os.environ["RPG_SRC"] = cfg.rpg_src
    os.environ.setdefault("RPG_PROTO", "rpg_v9")
    from prompt_compare_rl import sky_env

    texts = prompt_lib.read_materialized(cfg.prompt_dir)
    previous = os.environ.get("PC_PROMPT_SHA256")
    results = {}
    try:
        for pid in PROMPT_IDS:
            os.environ["PC_PROMPT_ID"] = pid
            os.environ["PC_PROMPT_SHA256"] = prompt_lib.sha256_text(texts[pid])
            good = [{"role": "system", "content": texts[pid]}, {"role": "user", "content": "obs"}]
            sky_env._verify_system_prompt(good)  # must not raise

            other = next(o for o in PROMPT_IDS if o != pid)
            bad = [{"role": "system", "content": texts[other]}, {"role": "user", "content": "obs"}]
            try:
                sky_env._verify_system_prompt(bad)
            except RuntimeError:
                results[pid] = "accepts self, rejects others"
            else:
                raise SmokeFailure(f"{pid}: the env accepted {other}'s prompt")
    finally:
        if previous is None:
            os.environ.pop("PC_PROMPT_SHA256", None)
            os.environ.pop("PC_PROMPT_ID", None)
        else:
            os.environ["PC_PROMPT_SHA256"] = previous
    return results


# --------------------------------------------------------------------------------------
# 2. exact step count
# --------------------------------------------------------------------------------------


def check_step_count(cfg: ExperimentConfig) -> Dict[str, Any]:
    import pandas as pd
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    details: Dict[str, Any] = {}
    # `validate_cfg` insists on a W&B key because the runs log to wandb. The smoke test
    # must not need real credentials, so a placeholder stands in for this process only;
    # run_one.sh sources the real key before launching.
    injected_wandb_key = "WANDB_API_KEY" not in os.environ
    if injected_wandb_key:
        os.environ["WANDB_API_KEY"] = "smoke-test-placeholder"
    try:
        return _check_step_count(cfg, details, SkyRLTrainConfig, validate_cfg)
    finally:
        if injected_wandb_key:
            os.environ.pop("WANDB_API_KEY", None)


def _check_step_count(cfg, details, SkyRLTrainConfig, validate_cfg) -> Dict[str, Any]:
    import pandas as pd

    for pid in PROMPT_IDS:
        resolved = SkyRLTrainConfig.from_cli_overrides(run_overrides(cfg, pid))
        validate_cfg(resolved)

        _check(resolved.trainer.max_training_steps == MAX_TRAINING_STEPS,
               f"{pid}: max_training_steps={resolved.trainer.max_training_steps}, want {MAX_TRAINING_STEPS}")
        _check(resolved.trainer.epochs == 1, f"{pid}: epochs must be 1")
        _check(resolved.trainer.update_epochs_per_batch == 1, f"{pid}: update_epochs_per_batch must be 1")

        optimizer_steps_per_global_step = (
            resolved.trainer.train_batch_size
            // resolved.trainer.policy_mini_batch_size
            * resolved.trainer.update_epochs_per_batch
        )
        _check(
            optimizer_steps_per_global_step == 1,
            f"{pid}: {optimizer_steps_per_global_step} optimizer steps per global step; "
            "set train_batch_size == policy_mini_batch_size so '8 steps' is unambiguous",
        )

        rows = len(pd.read_parquet(cfg.train_parquet(pid)))
        batches_available = rows // resolved.trainer.train_batch_size
        _check(
            batches_available >= MAX_TRAINING_STEPS,
            f"{pid}: only {batches_available} batches of {resolved.trainer.train_batch_size} "
            f"available from {rows} rows; need {MAX_TRAINING_STEPS}",
        )

        _check(resolved.trainer.eval_before_train, f"{pid}: eval_before_train must be true (step 0)")
        _check(resolved.trainer.eval_interval == EVAL_INTERVAL, f"{pid}: eval_interval must be {EVAL_INTERVAL}")
        planned = [0] + [s for s in range(1, MAX_TRAINING_STEPS + 1) if s % EVAL_INTERVAL == 0]
        if MAX_TRAINING_STEPS not in planned:
            planned.append(MAX_TRAINING_STEPS)
        _check(tuple(planned) == EVAL_STEPS, f"{pid}: eval schedule {planned} != {list(EVAL_STEPS)}")

        details[pid] = {
            "total_optimizer_steps": MAX_TRAINING_STEPS * optimizer_steps_per_global_step,
            "batches_available": batches_available,
            "eval_steps": planned,
            "seed": resolved.trainer.seed,
        }

    seeds = {d["seed"] for d in details.values()}
    _check(len(seeds) == 1, f"runs do not share a seed: {seeds}")
    return details


def check_required_settings(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Requirement (8) plus the 'identical except the prompt' invariant."""
    from skyrl.train.config import SkyRLTrainConfig

    per_prompt = {pid: run_overrides(cfg, pid) for pid in PROMPT_IDS}
    run_scoped_prefixes = (
        "data.train_data", "data.val_data", "trainer.ckpt_path", "trainer.export_path",
        "trainer.log_path", "trainer.run_name", "trainer.policy.model.lora.lora_sync_path",
    )
    shared = {
        pid: [o for o in overrides if not o.startswith(run_scoped_prefixes)]
        for pid, overrides in per_prompt.items()
    }
    reference = shared[PROMPT_IDS[0]]
    for pid in PROMPT_IDS[1:]:
        _check(
            shared[pid] == reference,
            f"{pid}: hyper-parameters differ from p1: "
            f"{[o for o in shared[pid] if o not in reference]}",
        )

    resolved = SkyRLTrainConfig.from_cli_overrides(per_prompt["p1"])
    wrap = (resolved.trainer.policy.fsdp_config.wrap_policy or {}).get("transformer_layer_cls_to_wrap")
    _check(wrap == "Qwen3_5DecoderLayer", f"FSDP wrap class is {wrap!r}, want 'Qwen3_5DecoderLayer'")
    _check(resolved.generator.inference_engine.language_model_only is True,
           "generator.inference_engine.language_model_only must be true (text-only)")
    _check(resolved.generator.chat_template_kwargs.get("enable_thinking") is True,
           "generator.chat_template_kwargs.enable_thinking must be explicitly true")
    _check(resolved.trainer.policy.model.lora.exclude_modules == ".*visual.*",
           "LoRA must exclude the vision tower for a text-only run")
    _check(resolved.generator.eval_sampling_params.temperature == 1.0,
           "eval sampling temperature must be stated explicitly (it does not inherit)")
    _check(resolved.trainer.algorithm.advantage_estimator == "grpo", "advantage estimator must be grpo")

    _check(resolved.trainer.max_ckpts_to_keep == -1,
           "max_ckpts_to_keep must be -1 so the step-4 checkpoint survives the step-8 save")

    env = run_env(cfg, "p1")
    _check(env["RPG_PROTO"] == "rpg_v9", "RPG_PROTO must be rpg_v9")
    return {
        "fsdp_layer_cls": wrap,
        "language_model_only": resolved.generator.inference_engine.language_model_only,
        "enable_thinking": resolved.generator.chat_template_kwargs.get("enable_thinking"),
        "lora": f"r{resolved.trainer.policy.model.lora.rank}"
                f"/a{resolved.trainer.policy.model.lora.alpha} exclude={resolved.trainer.policy.model.lora.exclude_modules}",
        "rpg_proto": env["RPG_PROTO"],
        "ckpt_interval": resolved.trainer.ckpt_interval,
        "ckpt_path": resolved.trainer.ckpt_path,
    }


# --------------------------------------------------------------------------------------
# 3. output-directory isolation
# --------------------------------------------------------------------------------------


def check_isolation(cfg: ExperimentConfig) -> Dict[str, Any]:
    seen: Dict[str, Tuple[str, str]] = {}
    per_prompt_paths = {pid: cfg.run_paths(pid) for pid in PROMPT_IDS}

    for pid, paths in per_prompt_paths.items():
        for key, value in paths.items():
            resolved = str(Path(value).resolve())
            if resolved in seen:
                other_pid, other_key = seen[resolved]
                raise SmokeFailure(
                    f"path collision: {pid}.{key} and {other_pid}.{other_key} both use {resolved}"
                )
            seen[resolved] = (pid, key)

    # No run directory may contain another's.
    roots = {pid: Path(paths["run_dir"]).resolve() for pid, paths in per_prompt_paths.items()}
    for pid_a, root_a in roots.items():
        for pid_b, root_b in roots.items():
            if pid_a >= pid_b:
                continue
            _check(
                not (root_a == root_b or root_a in root_b.parents or root_b in root_a.parents),
                f"run directories for {pid_a} and {pid_b} are nested: {root_a} / {root_b}",
            )

    envs = {pid: run_env(cfg, pid) for pid in PROMPT_IDS}
    isolated_env_keys = [
        "RAY_TMPDIR", "CUDA_VISIBLE_DEVICES", "WANDB_RUN_ID", "WANDB_DIR", "PC_RUN_DIR",
        "RPG_DATA_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT",
        "RPG_SYSTEM_PROMPT_FILE", "RPG_SYSTEM_PROMPT_SHA256",
    ]
    for key in isolated_env_keys:
        values = [envs[pid][key] for pid in PROMPT_IDS]
        _check(len(set(values)) == len(PROMPT_IDS), f"env var {key} is not unique per run: {values}")

    for pid in PROMPT_IDS:
        _check(envs[pid]["RAY_ADDRESS"] == "local",
               f"{pid}: RAY_ADDRESS must be 'local' so each job starts its own Ray cluster")
        _check(int(envs[pid]["CUDA_VISIBLE_DEVICES"]) == cfg.gpu_for(pid),
               f"{pid}: expected GPU {cfg.gpu_for(pid)}")
        _check(envs[pid]["WANDB_RESUME"] == "allow",
               f"{pid}: WANDB_RESUME must be 'allow' so a re-launch reuses the same run id")
        _check(envs[pid]["WANDB_RUN_ID"] == cfg.wandb_run_id(pid),
               f"{pid}: W&B run id is not the deterministic one")

    assigned = [cfg.gpu_for(pid) for pid in PROMPT_IDS]
    _check(len(set(assigned)) == len(PROMPT_IDS),
           f"the three runs must not share a GPU, got {assigned}")

    # The checkpoint save lock is the ONE thing the runs must share: it serializes the
    # ~18 GiB state-dict materialization that would otherwise happen three times at once.
    locks = {envs[pid]["PC_CKPT_LOCK"] for pid in PROMPT_IDS}
    _check(len(locks) == 1, f"PC_CKPT_LOCK must be shared by all runs, got {locks}")

    # Checkpoints are per-run, large, and must not sit on the small container filesystem.
    if cfg.ckpt_interval > 0:
        ckpt_root = Path(cfg.ckpt_root)
        _check(
            ckpt_root.is_dir() and os.access(ckpt_root, os.W_OK),
            f"checkpoint root {ckpt_root} is not writable; create the container with the "
            "/data mount (container/create_skyrl_pc.sh) or set PC_CKPT_INTERVAL=0",
        )
        free_gb = shutil.disk_usage(ckpt_root).free / 2**30
        needed_gb = 19 * len(PROMPT_IDS) * (MAX_TRAINING_STEPS // cfg.ckpt_interval)
        _check(free_gb >= needed_gb,
               f"checkpoint root has {free_gb:.0f} GB free, needs about {needed_gb} GB")
        for pid in PROMPT_IDS:
            ckpt = Path(cfg.run_paths(pid)["ckpt_path"]).resolve()
            _check(Path(cfg.out_root).resolve() not in ckpt.parents,
                   f"{pid}: checkpoints must not be written under out_root ({ckpt})")

    # LoRA sync dirs are the most dangerous shared path: the trainer writes the adapter
    # there and vLLM reads it back, so a shared dir silently swaps adapters between runs.
    lora_dirs = [str(Path(cfg.run_paths(pid)["lora_sync_path"]).resolve()) for pid in PROMPT_IDS]
    _check(len(set(lora_dirs)) == 3, f"LoRA sync directories are not unique: {lora_dirs}")
    for pid in PROMPT_IDS:
        overrides = run_overrides(cfg, pid)
        _check(
            any(o.startswith("trainer.policy.model.lora.lora_sync_path=") for o in overrides),
            f"{pid}: lora_sync_path is not overridden (the SkyRL default is shared /tmp)",
        )

    return {
        "ckpt_root": cfg.ckpt_root,
        "ckpt_interval": cfg.ckpt_interval,
        "shared_save_lock": next(iter(locks)),
        "run_dirs": {pid: str(roots[pid]) for pid in PROMPT_IDS},
        "gpus": {pid: envs[pid]["CUDA_VISIBLE_DEVICES"] for pid in PROMPT_IDS},
        "wandb_run_ids": {pid: envs[pid]["WANDB_RUN_ID"] for pid in PROMPT_IDS},
        "distinct_paths": len(seen),
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

Check = Tuple[str, Callable[[ExperimentConfig], Any], bool]

CHECKS: List[Check] = [
    ("required-settings", check_required_settings, False),
    ("output-isolation", check_isolation, False),
    ("prompt-injection", check_prompt_injection, True),
    ("prompt-runtime-assertion", check_runtime_prompt_assertion, True),
    ("step-count", check_step_count, True),
    ("prompt-length-parity", check_prompt_length_parity, True),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-dataset-checks", action="store_true",
                        help="only run the checks that do not need the built parquet files")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig()
    results: Dict[str, Any] = {"exp_tag": cfg.exp_tag, "checks": {}}
    failures: List[str] = []

    for name, fn, needs_dataset in CHECKS:
        if needs_dataset and args.skip_dataset_checks:
            results["checks"][name] = {"status": "skipped"}
            continue
        try:
            results["checks"][name] = {"status": "pass", "details": fn(cfg)}
        except Exception as exc:  # noqa: BLE001 - the report is the product
            results["checks"][name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(name)

    if args.as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"smoke test for experiment {cfg.exp_tag}")
        for name, outcome in results["checks"].items():
            marker = {"pass": "PASS", "FAIL": "FAIL", "skipped": "SKIP"}[outcome["status"]]
            print(f"  [{marker}] {name}")
            if outcome["status"] == "FAIL":
                print(f"         {outcome['error']}")
            elif outcome["status"] == "pass":
                print("         " + json.dumps(outcome["details"], default=str)[:600])
        print()
        print("RESULT: " + ("FAILED (" + ", ".join(failures) + ")" if failures else "all checks passed"))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
