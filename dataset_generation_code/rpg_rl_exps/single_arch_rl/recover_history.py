#!/usr/bin/env python3
"""Rebuild the run's metric history as CSV, from the artifacts on disk.

W&B holds almost nothing for ``sarl_v1-easy``. The run id is a pure function of the
config (``<exp_tag>-<run_id>``) with ``WANDB_RESUME=allow``, so the eight launches that
crashed before step 1 all attached to the *same* W&B run; each logged one
``error/tracebacks`` table, and each advanced the server-side ``_step`` by one, to 7.
SkyRL then calls ``wandb.log(metrics, step=global_step)``, so when the ninth launch
finally trained, steps 0-6 were rejected:

    wandb: WARNING Tried to log to step 0 that is less than the current step 7.
    Steps must be monotonically increasing, so this data will be ignored.

Only global steps 7 and 8 landed. Everything else has to come from the three artifacts
that survived independently of W&B:

===============================  ==========================================================
source                           what it yields
===============================  ==========================================================
``logs/train_<stamp>.log``       every training step: the group statistics ``main.py``
                                 prints, SkyRL's ``reward/avg_raw_reward`` line, the policy
                                 metrics dict, and the ``Started:``/``Finished:`` timers
                                 that back the whole ``timing/*`` namespace
``exports/dumped_evals/``        every evaluation, complete -- the same files
                                 ``report_eval.py`` reads, plus per-episode records
``checkpoints/*/trainer_state``  the resolved config and the dataloader position (no
                                 metrics: a checkpoint is ``{global_step, config}``)
``wandb/.../run-*.wandb``        the *complete* metric dict, for steps 7 and 8 only
===============================  ==========================================================

Nothing here is recomputed from model weights and nothing is invented: every row carries
the source it came from, and ``coverage.csv`` states, per metric key, which steps are
recoverable and which are gone for good.

    bash scripts/in_container.sh python -m single_arch_rl.recover_history
    bash scripts/in_container.sh python -m single_arch_rl.recover_history --out /tmp/csv

Reading ``trainer_state.pt`` needs ``torch`` and reading the W&B transaction log needs
``wandb``; both are optional -- without them the script emits everything else and says
which files it skipped. Run it inside the SkyRL container to get all of it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from single_arch_rl.config import RUN_IDS, ExperimentConfig  # noqa: E402

# --- log grammar -----------------------------------------------------------------------
# Ray prefixes every forwarded line with "(skyrl_entrypoint pid=N) " and the tqdm bars
# leave ANSI and cursor-movement escapes behind; strip both before matching.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RAY_PREFIX = re.compile(r"^\(\w+ pid=\d+\)\s?")
_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"

_STARTED = re.compile(_TS + r".*Started: '([a-z_]+)'")
_FINISHED = re.compile(_TS + r".*Finished: '([a-z_]+)', time cost: ([\d.]+)s")
#: main.py's per-step line -- the only place group_reward_var exists for steps 1-6.
_GROUP = re.compile(
    r"\[single_arch_rl\] step (\d+): group_reward_mean=([\d.eE+-]+) "
    r"group_reward_var=([\d.eE+-]+) nondegenerate=(\d+)% \((\d+) groups\)"
)
#: SkyRL's own line from postprocess_generator_output.
_REWARD = re.compile(
    r"reward/avg_pass_at_(\d+): ([\d.eE+-]+), reward/avg_raw_reward: ([\d.eE+-]+), "
    r"reward/mean_positive_reward: ([\d.eE+-]+)"
)
#: The policy metrics dict, printed as a python repr.
_POLICY = re.compile(r"trainer:train:\d+ - (\{'policy_entropy'.*?\})")
_ADAPTER = re.compile(r"\[single_arch_rl\] exported LoRA adapter -> (\S*global_step_(\d+))")

#: Timers that SkyRL also publishes as ``timing/<name>``.
TIMING_KEYS = (
    "step",
    "generate",
    "postprocess_generator_output",
    "log_train_results",
    "convert_to_training_input",
    "fwd_logprobs_values_reward",
    "compute_advantages_and_returns",
    "train_critic_and_policy",
    "policy_train",
    "sync_weights",
    "eval",
    "log_eval_results",
    "dump_eval_results",
    "save_checkpoints",
    "cleanup_old_checkpoints",
    "load_checkpoints",
    "init_weight_sync_state",
)

#: Decimal places a value survives with, when its only copy is a formatted log line.
#: ``main.py`` prints the group statistics with ``:.4f`` and loguru prints ``time cost:
#: %.2fs``, so those keys are recovered *rounded* -- exact enough to plot, not exact
#: enough to diff against W&B without saying so. Everything else is a full ``repr``.
LOG_PRECISION_DP: Dict[str, int] = {
    "reward/group_reward_mean": 4,
    "reward/group_reward_var": 4,
    "reward/frac_nondegenerate_groups": 2,
}
TIMING_DP = 2
FULL_PRECISION = -1  # a full repr; no rounding was applied

#: Phases that only run on some steps (eval_interval / ckpt_interval), so their absence
#: from an odd step is the schedule, not a gap in what was recovered.
PERIODIC_TIMERS = frozenset(
    {
        "timing/eval",
        "timing/eval_generate",
        "timing/log_eval_results",
        "timing/dump_eval_results",
        "timing/save_checkpoints",
        "timing/cleanup_old_checkpoints",
        "timing/load_checkpoints",
        "timing/init_weight_sync_state",
    }
)
#: W&B's own bookkeeping, not run metrics -- excluded from the recovery accounting.
WANDB_BOOKKEEPING = frozenset({"_step", "_runtime", "_timestamp"})


def log_precision(key: str) -> int:
    if key in LOG_PRECISION_DP:
        return LOG_PRECISION_DP[key]
    if key.startswith("timing/"):
        return TIMING_DP
    return FULL_PRECISION


#: Metric keys whose *only* surviving copy is the W&B transaction log (steps 7-8).
WANDB_ONLY_NAMESPACES = ("environment/", "generate/", "loss/")
WANDB_ONLY_KEYS = (
    "reward/group_reward_var_between",
    "reward/group_reward_var_max",
    "reward/group_reward_var_min",
    "reward/group_size_max",
    "reward/group_size_min",
    "policy/rollout_train_logprobs_abs_diff_max",
    "policy/rollout_train_logprobs_abs_diff_mean",
    "policy/rollout_train_logprobs_abs_diff_min",
    "policy/rollout_train_logprobs_abs_diff_std",
    "trainer/tokens_per_second_per_gpu",
)


def _clean(line: str) -> str:
    return _RAY_PREFIX.sub("", _ANSI.sub("", line)).rstrip("\n")


def _ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


# --- the training log ------------------------------------------------------------------


def pick_train_log(log_dir: Path) -> Optional[Path]:
    """The training log that actually has steps in it.

    Eight of the nine launches died before step 1; their logs are the same size as the
    real one, so pick by content -- the newest log containing a per-step line -- rather
    than by mtime.
    """
    best: Tuple[int, float, Optional[Path]] = (0, 0.0, None)
    for path in sorted(log_dir.glob("train_*.log")):
        try:
            steps = len(_GROUP.findall(path.read_text(errors="replace")))
        except OSError:
            continue
        if steps and (steps, path.stat().st_mtime) > best[:2]:
            best = (steps, path.stat().st_mtime, path)
    return best[2]


def parse_train_log(path: Path) -> Dict[str, Any]:
    """Per-step metrics and timings, keyed by SkyRL's own global_step.

    **Two streams share this file and only one of them is in order.** Ray forwards the
    driver's loguru output line by line, but buffers the plain ``print()`` calls from the
    actor and flushes them in a batch -- in this run, ``[single_arch_rl] step 1..7`` all
    arrive together, thousands of lines after the steps they describe. So:

    * loguru lines (the timers, ``reward/avg_raw_reward``, the policy dict) are attributed
      **positionally**: a step opens at ``Started: 'step'`` and owns everything until the
      next one. The eval and checkpoint save that follow a step land on that step, which
      is where SkyRL logs them too, and the pre-training eval runs before any step opens
      so it lands on step 0 -- matching ``global_step_0_evals``.
    * the ``print()`` lines are attributed by the **step number they carry themselves**
      (``main.py`` prints ``self.global_step``; the adapter path ends in
      ``global_step_<N>``). Their position in the file means nothing.

    The two are then cross-checked against each other: ``group_reward_mean`` must equal
    ``reward/avg_raw_reward`` whenever every group is the same size, which is exactly the
    invariant ``metrics.py`` documents. A mismatch would mean the streams disagree about
    which step a number belongs to, so it is reported rather than smoothed over.
    """
    steps: Dict[int, Dict[str, Any]] = {}
    timings: List[Dict[str, Any]] = []
    adapters: List[Dict[str, Any]] = []
    open_timers: Dict[str, float] = {}
    cur = 0
    row = steps.setdefault(0, {})

    for raw in path.read_text(errors="replace").splitlines():
        line = _clean(raw)

        started = _STARTED.search(line)
        if started and started.group(2) == "step":
            cur += 1
            row = steps.setdefault(cur, {})
        if started:
            open_timers[started.group(2)] = _ts(started.group(1))

        finished = _FINISHED.search(line)
        if finished:
            stamp, name, cost = finished.group(1), finished.group(2), float(finished.group(3))
            start = open_timers.pop(name, None)
            timings.append(
                {
                    "global_step": cur,
                    "phase": name,
                    "seconds": cost,
                    "started_at": datetime.fromtimestamp(start).isoformat(sep=" ") if start else "",
                    "finished_at": stamp,
                }
            )
            row.setdefault(f"timing/{name}", cost)

        # -- self-labelled (buffered) stream: trust the number in the line ---------------
        group = _GROUP.search(line)
        if group:
            steps.setdefault(int(group.group(1)), {}).update(
                {
                    "reward/group_reward_mean": float(group.group(2)),
                    "reward/group_reward_var": float(group.group(3)),
                    "reward/frac_nondegenerate_groups": int(group.group(4)) / 100.0,
                    "reward/num_groups": float(group.group(5)),
                }
            )

        adapter = _ADAPTER.search(line)
        if adapter:
            adapters.append(
                {"global_step": int(adapter.group(2)), "adapter_dir": adapter.group(1)}
            )

        # -- ordered (loguru) stream: attribute to the open step -------------------------
        reward = _REWARD.search(line)
        if reward:
            row[f"reward/avg_pass_at_{reward.group(1)}"] = float(reward.group(2))
            row["reward/avg_raw_reward"] = float(reward.group(3))
            row["reward/mean_positive_reward"] = float(reward.group(4))

        policy = _POLICY.search(line)
        if policy:
            # A python repr, not JSON: only quote style differs for these flat float dicts.
            for key, value in json.loads(policy.group(1).replace("'", '"')).items():
                row[f"policy/{key}"] = value

    checks = cross_check(steps)
    return {
        "steps": steps,
        "timings": timings,
        "adapters": sorted(adapters, key=lambda a: a["global_step"]),
        "checks": checks,
        "log": path,
    }


def cross_check(steps: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Do the buffered and the ordered stream agree about each step?

    ``metrics.py``: ``group_reward_mean`` is the mean over groups of each group's mean,
    which equals SkyRL's trajectory-level ``avg_raw_reward`` exactly when every group has
    the same size. ``num_groups`` and ``group_size_min/max`` say they always do here, so
    any disagreement is a step-attribution error, not a real difference.
    """
    rows: List[Dict[str, Any]] = []
    for step in sorted(steps):
        mean = steps[step].get("reward/group_reward_mean")
        raw = steps[step].get("reward/avg_raw_reward")
        if mean is None or raw is None:
            continue
        rows.append(
            {
                "global_step": step,
                "group_reward_mean": mean,
                "avg_raw_reward": raw,
                # main.py prints 4 decimals, so agreement is only checkable to that.
                "agrees_to_4dp": abs(mean - round(raw, 4)) < 5e-5,
            }
        )
    bad = [r["global_step"] for r in rows if not r["agrees_to_4dp"]]
    if bad:
        print(f"  ! streams disagree at steps {bad} -- see stream_crosscheck.csv", file=sys.stderr)
    return rows


# --- the evaluation dumps --------------------------------------------------------------


def parse_eval_dumps(export_path: Path) -> Dict[int, Dict[str, Any]]:
    """SkyRL's own eval dumps: aggregated metrics plus the per-episode records."""
    out: Dict[int, Dict[str, Any]] = {}
    root = export_path / "dumped_evals"
    if not root.is_dir():
        return out
    for directory in sorted(root.glob("global_step_*_evals")):
        try:
            step = int(directory.name.split("_")[2])
        except (IndexError, ValueError):
            continue
        agg_file = directory / "aggregated_results.jsonl"
        aggregated: Dict[str, Any] = {}
        if agg_file.exists():
            for line in agg_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    aggregated = json.loads(line)
                    break
        episodes: List[Dict[str, Any]] = []
        for jsonl in sorted(directory.glob("*.jsonl")):
            if jsonl.name == "aggregated_results.jsonl":
                continue
            for index, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                record = json.loads(line)
                extras = record.get("env_extras") or {}
                info = extras.get("extra_info", {}) or {}
                score = record.get("score")
                # Token-level rewards: the trajectory reward is their sum, which is how
                # SkyRL and metrics.py both score a trajectory.
                total = float(sum(score)) if isinstance(score, list) else float(score or 0.0)
                episodes.append(
                    {
                        "global_step": step,
                        "archetype": info.get("archetype", ""),
                        "data_source": record.get("data_source", ""),
                        "episode_index": index,
                        "seed": info.get("seed", ""),
                        "skin": info.get("skin", ""),
                        "split": info.get("split", ""),
                        "budget": info.get("budget", ""),
                        "max_turns": info.get("max_turns", ""),
                        "score": total,
                        "stop_reason": record.get("stop_reason", ""),
                        "n_score_tokens": len(score) if isinstance(score, list) else "",
                        "response_chars": len(record.get("output_response") or ""),
                        "prompt_chars": len(record.get("input_prompt") or ""),
                    }
                )
        out[step] = {"aggregated": aggregated, "episodes": episodes}
    return out


# --- the checkpoints -------------------------------------------------------------------


def parse_checkpoints(ckpt_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Inventory + the resolved config. A checkpoint carries no metric history.

    ``trainer_state.pt`` is ``{global_step, config}`` and ``data.pt`` is the dataloader
    snapshot, so the only run-level facts here are the config (which is where
    ``train_batch_size`` is authoritative) and how far the dataloader had walked.
    """
    rows: List[Dict[str, Any]] = []
    config: Dict[str, Any] = {}
    if not ckpt_path.is_dir():
        return rows, config
    try:
        import torch
    except ImportError:
        print("  ! torch unavailable: skipping checkpoint_*.csv", file=sys.stderr)
        return rows, config

    for directory in sorted(
        ckpt_path.glob("global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    ):
        if not directory.is_dir():
            continue
        state_file = directory / "trainer_state.pt"
        if not state_file.exists():
            continue
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        config = config or state.get("config", {})
        snapshot = {}
        data_file = directory / "data.pt"
        if data_file.exists():
            data = torch.load(data_file, map_location="cpu", weights_only=False)
            snap = (data.get("_snapshot") or {}).get("_main_snapshot", {})
            sampler = (snap.get("_sampler_iter_state") or {})
            snapshot = {
                "dataloader_snapshot_step": (data.get("_snapshot") or {}).get("_snapshot_step", ""),
                "dataloader_samples_yielded": sampler.get("samples_yielded", ""),
                "dataloader_steps_since_snapshot": data.get("_steps_since_snapshot", ""),
                "dataloader_iterator_finished": data.get("_iterator_finished", ""),
            }
        trainer = state.get("config", {}).get("trainer", {})
        generator = state.get("config", {}).get("generator", {})
        rows.append(
            {
                "global_step": state.get("global_step", ""),
                "checkpoint_dir": str(directory),
                "train_batch_size": trainer.get("train_batch_size", ""),
                "policy_mini_batch_size": trainer.get("policy_mini_batch_size", ""),
                "micro_train_batch_size_per_gpu": trainer.get("micro_train_batch_size_per_gpu", ""),
                "micro_forward_batch_size_per_gpu": trainer.get(
                    "micro_forward_batch_size_per_gpu", ""
                ),
                "max_tokens_per_microbatch": trainer.get("max_tokens_per_microbatch", ""),
                "update_epochs_per_batch": trainer.get("update_epochs_per_batch", ""),
                "n_samples_per_prompt": generator.get("n_samples_per_prompt", ""),
                "episodes_per_step": (
                    trainer.get("train_batch_size", 0) * generator.get("n_samples_per_prompt", 0)
                ),
                "eval_batch_size": trainer.get("eval_batch_size", ""),
                "eval_n_samples_per_prompt": generator.get("eval_n_samples_per_prompt", ""),
                "bytes_on_disk": sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()),
                **snapshot,
            }
        )
    return rows, config


def flatten(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f"{prefix}{key}.")
    elif isinstance(obj, (list, tuple)):
        yield prefix.rstrip("."), json.dumps(list(obj))
    else:
        yield prefix.rstrip("."), "" if obj is None else str(obj)


# --- the W&B transaction log -----------------------------------------------------------


def parse_wandb(wandb_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """The complete metric dict for whichever steps W&B actually accepted.

    The local ``.wandb`` transaction log is written before the server is consulted, so it
    is readable offline and is the only surviving copy of the ``environment/*``,
    ``generate/*`` and ``loss/*`` namespaces. Records without ``trainer/global_step`` are
    the crash-traceback tables from the eight aborted launches, not training data.
    """
    history: Dict[int, Dict[str, Any]] = {}
    dropped: List[Dict[str, Any]] = []
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal import datastore
    except ImportError:
        print("  ! wandb unavailable: skipping wandb_*.csv", file=sys.stderr)
        return history, dropped

    for path in sorted(wandb_dir.glob("wandb/run-*/run-*.wandb")):
        store = datastore.DataStore()
        try:
            store.open_for_scan(str(path))
        except Exception as exc:  # noqa: BLE001 - a truncated log must not stop the rest
            print(f"  ! unreadable {path.name}: {exc!r}", file=sys.stderr)
            continue
        while True:
            try:
                blob = store.scan_data()
            except Exception:  # noqa: BLE001 - trailing partial record after a crash
                break
            if blob is None:
                break
            record = pb.Record()
            record.ParseFromString(blob)
            if record.WhichOneof("record_type") != "history":
                continue
            items = {
                (item.key or "/".join(item.nested_key)): item.value_json
                for item in record.history.item
            }
            if "trainer/global_step" not in items:
                dropped.append(
                    {
                        "wandb_run_dir": path.parent.name,
                        "wandb_internal_step": record.history.step.num,
                        "n_keys": len(items),
                        "content": "error/tracebacks table from an aborted launch",
                    }
                )
                continue
            step = int(json.loads(items["trainer/global_step"]))
            parsed: Dict[str, Any] = {}
            for key, value in items.items():
                try:
                    decoded = json.loads(value)
                except (TypeError, ValueError):
                    continue
                if isinstance(decoded, (int, float)) and not isinstance(decoded, bool):
                    parsed[key] = decoded
            history[step] = parsed
    return history, dropped


# --- writing ---------------------------------------------------------------------------


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name:<34} {len(rows):>5} rows")
    return len(rows)


ARCHETYPE_COLUMNS = ("score", "part_a", "part_b", "truncated", "turns", "episodes")


def emit(cfg: ExperimentConfig, run_id: str, out_root: Path) -> None:
    paths = cfg.run_paths(run_id)
    out = out_root / run_id
    print(f"\n{run_id}  ({cfg.exp_tag})")

    log_file = pick_train_log(paths["log_path"]) if paths["log_path"].is_dir() else None
    if log_file is None:
        print("  no training log with completed steps -- run never trained; nothing to recover")
        return
    print(f"  training log: {log_file.name}")

    train = parse_train_log(log_file)
    evals = parse_eval_dumps(paths["export_path"])
    ckpt_rows, config = parse_checkpoints(paths["ckpt_path"])
    wandb_history, wandb_dropped = parse_wandb(paths["wandb_dir"])

    train_steps = {s: r for s, r in train["steps"].items() if s >= 1 and r}
    long_rows: List[Dict[str, Any]] = []

    # -- 1. training metrics, every step, from the log ------------------------------------
    keys = sorted({k for row in train_steps.values() for k in row})
    rows = []
    for step in sorted(train_steps):
        row: Dict[str, Any] = {"global_step": step}
        row.update({k: train_steps[step].get(k, "") for k in keys})
        num_groups = train_steps[step].get("reward/num_groups")
        row["episodes"] = (
            int(num_groups) * cfg.n_samples_per_prompt if num_groups else ""
        )
        rows.append(row)
        for key, value in train_steps[step].items():
            long_rows.append(
                {
                    "run": run_id,
                    "split": "train",
                    "global_step": step,
                    "key": key,
                    "value": value,
                    "source": "train_log",
                }
            )
    write_csv(out / "train_by_step.csv", ["global_step", "episodes", *keys], rows)

    # -- 2. per-phase timings, every step -------------------------------------------------
    write_csv(
        out / "step_timings.csv",
        ["global_step", "phase", "seconds", "started_at", "finished_at"],
        train["timings"],
    )
    write_csv(
        out / "stream_crosscheck.csv",
        ["global_step", "group_reward_mean", "avg_raw_reward", "agrees_to_4dp"],
        train["checks"],
    )

    # -- 3. the complete metric dict, for the steps W&B accepted --------------------------
    if wandb_history:
        wandb_keys = sorted({k for row in wandb_history.values() for k in row})
        write_csv(
            out / "train_full_from_wandb.csv",
            ["global_step", *wandb_keys],
            [
                {"global_step": step, **{k: wandb_history[step].get(k, "") for k in wandb_keys}}
                for step in sorted(wandb_history)
            ],
        )
        for step, row in wandb_history.items():
            for key, value in row.items():
                long_rows.append(
                    {
                        "run": run_id,
                        "split": "eval" if key.startswith("eval/") else "train",
                        "global_step": step,
                        "key": key,
                        "value": value,
                        "source": "wandb_txlog",
                    }
                )
        # Steps 7-8 exist in both places, which makes them a free audit of the whole
        # log-parsing path: if the two agree there, the same parser is trustworthy for
        # steps 1-6, where W&B has nothing to check against.
        audit = []
        for step in sorted(set(wandb_history) & set(train_steps)):
            for key, value in sorted(train_steps[step].items()):
                reference = wandb_history[step].get(key)
                if reference is None:
                    continue
                places = log_precision(key)
                if places == FULL_PRECISION:
                    tolerance = max(1e-9, abs(reference) * 1e-9)
                else:
                    # A rounded value is only good to half its last digit ...
                    tolerance = 0.5 * 10 ** (-places) + 1e-12
                    if key.startswith("timing/"):
                        # ... and a timer is worse than that: the number loguru prints and
                        # the number that reaches the metrics dict are two separate reads
                        # of the same span, so they disagree by a few ms on top of the
                        # rounding. Measured worst case in this run: 5.1 ms on
                        # timing/generate. Treat 10 ms as the floor of what a recovered
                        # timing is worth.
                        tolerance += 0.01
                audit.append(
                    {
                        "global_step": step,
                        "key": key,
                        "from_log": value,
                        "from_wandb": reference,
                        "abs_diff": abs(value - reference),
                        "log_precision_dp": "full" if places == FULL_PRECISION else places,
                        "agrees": abs(value - reference) <= tolerance,
                    }
                )
        write_csv(
            out / "validation_vs_wandb.csv",
            [
                "global_step",
                "key",
                "from_log",
                "from_wandb",
                "abs_diff",
                "log_precision_dp",
                "agrees",
            ],
            audit,
        )
        failed = [a for a in audit if not a["agrees"]]
        print(
            f"  audit: {len(audit) - len(failed)}/{len(audit)} log-derived values match W&B "
            f"at the log's own precision"
            + (f" -- {len(failed)} DISAGREE" if failed else "")
        )

    if wandb_dropped:
        write_csv(
            out / "wandb_dropped_records.csv",
            ["wandb_run_dir", "wandb_internal_step", "n_keys", "content"],
            wandb_dropped,
        )

    # -- 4. evaluations: pooled, per archetype, per episode -------------------------------
    if evals:
        pooled_keys = sorted(
            {
                k
                for e in evals.values()
                for k in e["aggregated"]
                if k.startswith("eval/all/") and "/archetype/" not in k
            }
        )
        write_csv(
            out / "eval_overall_by_step.csv",
            ["global_step", *pooled_keys],
            [
                {
                    "global_step": step,
                    **{k: evals[step]["aggregated"].get(k, "") for k in pooled_keys},
                }
                for step in sorted(evals)
            ],
        )

        arch_rows = []
        for step in sorted(evals):
            aggregated = evals[step]["aggregated"]
            episodes = evals[step]["episodes"]
            archetypes = sorted(
                {
                    k.split("/")[1].replace("rpg_v9_eval_", "")
                    for k in aggregated
                    if k.startswith("eval/rpg_v9_eval_") and k.endswith("/avg_score")
                }
            )
            for archetype in archetypes:
                scores = [e["score"] for e in episodes if e["archetype"] == archetype]
                reported = aggregated.get(f"eval/rpg_v9_eval_{archetype}/avg_score")
                recomputed = sum(scores) / len(scores) if scores else None
                row = {
                    "global_step": step,
                    "archetype": archetype,
                    "n_episodes": len(scores),
                    "avg_score": reported,
                    "pass_at_2": aggregated.get(f"eval/rpg_v9_eval_{archetype}/pass_at_2"),
                    "mean_positive_reward": aggregated.get(
                        f"eval/rpg_v9_eval_{archetype}/mean_positive_reward"
                    ),
                    "recomputed_avg_score": recomputed,
                    "recompute_matches": (
                        ""
                        if reported is None or recomputed is None
                        else abs(reported - recomputed) < 1e-6
                    ),
                }
                for column in ARCHETYPE_COLUMNS:
                    row[column] = aggregated.get(
                        f"eval/all/environment/archetype/{archetype}/{column}", ""
                    )
                arch_rows.append(row)
                for key, value in row.items():
                    if key in ("global_step", "archetype") or value in ("", None):
                        continue
                    long_rows.append(
                        {
                            "run": run_id,
                            "split": "eval",
                            "global_step": step,
                            "key": f"eval/archetype/{archetype}/{key}",
                            "value": value,
                            "source": "eval_dump",
                        }
                    )
        write_csv(
            out / "eval_by_archetype.csv",
            [
                "global_step",
                "archetype",
                "n_episodes",
                "avg_score",
                "recomputed_avg_score",
                "recompute_matches",
                "pass_at_2",
                "mean_positive_reward",
                *ARCHETYPE_COLUMNS,
            ],
            arch_rows,
        )

        episode_rows = [e for step in sorted(evals) for e in evals[step]["episodes"]]
        write_csv(
            out / "eval_episodes.csv",
            [
                "global_step",
                "archetype",
                "data_source",
                "episode_index",
                "seed",
                "skin",
                "split",
                "budget",
                "max_turns",
                "score",
                "stop_reason",
                "n_score_tokens",
                "response_chars",
                "prompt_chars",
            ],
            episode_rows,
        )
        for step in sorted(evals):
            for key, value in evals[step]["aggregated"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    long_rows.append(
                        {
                            "run": run_id,
                            "split": "eval",
                            "global_step": step,
                            "key": key,
                            "value": value,
                            "source": "eval_dump",
                        }
                    )

    # -- 5. what the checkpoints say ------------------------------------------------------
    if ckpt_rows:
        write_csv(out / "checkpoint_inventory.csv", list(ckpt_rows[0]), ckpt_rows)
    if config:
        write_csv(
            out / "resolved_config.csv",
            ["key", "value"],
            [{"key": k, "value": v} for k, v in sorted(flatten(config))],
        )
    if train["adapters"]:
        write_csv(
            out / "adapter_exports.csv", ["global_step", "adapter_dir"], train["adapters"]
        )

    # -- 6. one long table, and an honest statement of coverage ---------------------------
    write_csv(
        out / "metrics_long.csv",
        ["run", "split", "global_step", "key", "value", "source"],
        sorted(long_rows, key=lambda r: (r["split"], r["global_step"], r["key"])),
    )

    trained = sorted(train_steps)
    evaluated = sorted(evals)
    coverage: List[Dict[str, Any]] = []
    for key in keys:
        have = [s for s in trained if train_steps[s].get(key) not in (None, "")]
        places = log_precision(key)
        periodic = key in PERIODIC_TIMERS
        notes = []
        if periodic:
            notes.append("phase runs only on eval/checkpoint steps; absence is the schedule")
        if places != FULL_PRECISION:
            notes.append(f"recovered from a formatted log line; rounded to {places} dp")
        coverage.append(
            {
                "key": key,
                "namespace": key.split("/")[0],
                "source": "train_log",
                "steps_available": " ".join(map(str, have)),
                # "complete" means: recovered everywhere the value should exist.
                "complete": bool(have) if periodic else have == trained,
                "log_precision_dp": "full" if places == FULL_PRECISION else places,
                "note": "; ".join(notes),
            }
        )
    for key in sorted(
        {k for row in wandb_history.values() for k in row}
        if wandb_history
        else set(WANDB_ONLY_KEYS)
    ):
        if (
            key.startswith("eval/")
            or key in WANDB_BOOKKEEPING
            or any(c["key"] == key for c in coverage)
        ):
            continue
        have = sorted(s for s, row in wandb_history.items() if key in row)
        only = key.startswith(WANDB_ONLY_NAMESPACES) or key in WANDB_ONLY_KEYS
        coverage.append(
            {
                "key": key,
                "namespace": key.split("/")[0],
                "source": "wandb_txlog",
                "steps_available": " ".join(map(str, have)),
                "complete": have == trained,
                "log_precision_dp": "full",
                "note": (
                    "W&B rejected steps 0-6 (non-monotonic after 8 crashed launches); "
                    "no other copy on disk"
                    if only
                    else ""
                ),
            }
        )
    if evals:
        eval_keys = sorted({k for e in evals.values() for k in e["aggregated"]})
        for key in eval_keys:
            have = [s for s in evaluated if key in evals[s]["aggregated"]]
            coverage.append(
                {
                    "key": key,
                    "namespace": "eval",
                    "source": "eval_dump",
                    "steps_available": " ".join(map(str, have)),
                    "complete": have == evaluated,
                    "log_precision_dp": "full",
                    "note": "",
                }
            )
    write_csv(
        out / "coverage.csv",
        [
            "key",
            "namespace",
            "source",
            "steps_available",
            "complete",
            "log_precision_dp",
            "note",
        ],
        coverage,
    )

    recovered = sum(1 for c in coverage if c["complete"])
    print(
        f"  trained steps {trained[0]}-{trained[-1]}, evaluated at "
        f"{' '.join(map(str, evaluated))}; {recovered}/{len(coverage)} keys complete"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=",".join(RUN_IDS), help="comma-separated run ids")
    parser.add_argument(
        "--out",
        default=str(_HERE / "recovered"),
        help="output directory (default: single_arch_rl/recovered)",
    )
    args = parser.parse_args()

    # ExperimentConfig reads every SA_* knob in its field defaults, so the bare
    # constructor reproduces exactly the paths the run used.
    cfg = ExperimentConfig()
    out_root = Path(args.out)
    print(f"writing CSVs under {out_root}")
    for run_id in [r.strip() for r in args.runs.split(",") if r.strip()]:
        if run_id not in RUN_IDS:
            raise SystemExit(f"unknown run id {run_id!r}; expected one of {list(RUN_IDS)}")
        emit(cfg, run_id, out_root)


if __name__ == "__main__":
    main()
