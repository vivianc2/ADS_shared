#!/usr/bin/env python3
"""Run the current RPG probability-belief prompt on the prompt-compare dataset.

The full run is deliberately a single, matched inference condition: it reuses the
90 persisted prompt_compare_v3 worlds, the eight p2 rollout indices per world, and
the p2-derived request seeds.  Only the system prompt changes.  Turn-level raw model
responses are saved as atomic JSONL files and the final report compares the new
prompt with the stored p2 result, per archetype and overall.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


# Scientific imports come from read-only sibling directories.  Keep Python from
# creating/updating bytecode caches there so all task-owned writes stay local.
sys.dont_write_bytecode = True


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_CODE_DIR = EXPERIMENT_DIR.parents[1]
RPG_RL_DIR = DATASET_CODE_DIR / "rpg_rl"
RPG_V9_DIR = DATASET_CODE_DIR / "rpg_v9"
PROMPT_COMPARE_DIR = EXPERIMENT_DIR.parent / "prompt_compare"
DEFAULT_SOURCE_RUN = PROMPT_COMPARE_DIR / "runs" / "prompt_compare_v3"
DEFAULT_RUNS_ROOT = EXPERIMENT_DIR / "runs"

SCHEMA_VERSION = "prob_belief_v1"
PROMPT_ID = "prob_belief"
SEED_PAIR_PROMPT_ID = "p2"
BASELINE_CONFIG_ID = "p2_r1"
ROLLOUTS_PER_WORLD = 8
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CONTEXT_LIMIT_ACTION = '<action type="give_up">{}</action>'
EXPECTED_ARCHETYPES = (
    "confounded_chain",
    "collider_selection",
    "hidden_subtype",
    "surrogate_trap",
    "instrument_only",
    "competing_causes",
    "synergy_pair",
    "dose_window",
    "confounded_reversal",
)


class IncompleteRunError(RuntimeError):
    """Raised when final statistics would otherwise describe a partial run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: Any):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
    )


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            for record in records
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not a JSON object")
            records.append(value)
    return records


def completed_rollout_records(path: Path) -> list[dict[str, Any]] | None:
    try:
        records = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not records or records[-1].get("record_type") != "terminal":
        return None
    return records if records[-1].get("complete") is True else None


def _literal_string_assignment(path: Path, name: str) -> str:
    """Read a literal module constant without importing its scientific stack."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value_node = node.value
        if value_node is not None:
            value = ast.literal_eval(value_node)
            if not isinstance(value, str):
                raise TypeError(f"{path}:{name} is not a string literal")
            return value
    raise ValueError(f"could not find literal assignment {name} in {path}")


def current_system_prompt() -> str:
    return _literal_string_assignment(RPG_RL_DIR / "env.py", "SYSTEM_PROMPT")


def parse_csv_strings(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("expected at least one comma-separated value")
    return parts


def parse_ports(value: str) -> tuple[int, ...]:
    try:
        ports = tuple(int(part) for part in parse_csv_strings(value))
    except ValueError as exc:
        raise ValueError(f"invalid ports {value!r}") from exc
    if any(port < 1 or port > 65535 for port in ports):
        raise ValueError(f"ports must be between 1 and 65535: {ports}")
    return ports


def run_directory(runs_root: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "run-id must start with an alphanumeric and contain only letters, digits, ._-"
        )
    root = runs_root.resolve()
    try:
        root.relative_to(EXPERIMENT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"runs-root must stay inside the working directory {EXPERIMENT_DIR}"
        ) from exc
    return root / run_id


def derive_rollout_seed(master_seed: int, world_id: str, rollout_index: int) -> int:
    """Reuse the exact p2 seed derivation for a paired prompt comparison."""
    encoded = canonical_json({
        "master_seed": int(master_seed),
        "prompt_id": SEED_PAIR_PROMPT_ID,
        "rollout_index": int(rollout_index),
        "world_id": world_id,
    }).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % 2_147_483_647


def _source_snapshot(source_run: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    manifest_path = source_run / "manifest.json"
    stats_path = source_run / "stats.json"
    if not manifest_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"source prompt-compare manifest/stats are missing under {source_run}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    science = manifest.get("science", {})
    expected = science.get("expected", {})
    if manifest.get("schema_version") != "prompt_compare_v3":
        raise ValueError("source run is not prompt_compare_v3")
    if science.get("rpg_proto") != "rpg_v9" or science.get("rpg_synergy_soft") != 20:
        raise ValueError("source run does not use the required rpg_v9 protocol")
    if tuple(science.get("archetypes", ())) != EXPECTED_ARCHETYPES:
        raise ValueError("source run archetypes/order differ from the comparison contract")
    if (
        expected.get("worlds") != 90
        or expected.get("worlds_per_archetype") != 10
        or expected.get("rollouts_per_group") != ROLLOUTS_PER_WORLD
    ):
        raise ValueError("source run does not contain the full 90-world, G=8 dataset")
    if not stats.get("completeness", {}).get("complete"):
        raise ValueError("source prompt-compare result is incomplete")

    prompts = {item["prompt_id"]: item for item in science.get("prompts", [])}
    if SEED_PAIR_PROMPT_ID not in prompts:
        raise ValueError("source manifest has no p2 prompt")
    p2_prompt = prompts[SEED_PAIR_PROMPT_ID]
    p2_text = p2_prompt.get("system_prompt")
    if not isinstance(p2_text, str) or p2_prompt.get("prompt_sha256") != sha256_bytes(
        p2_text.encode("utf-8")
    ):
        raise ValueError("source p2 prompt/hash is invalid")

    worlds = manifest.get("worlds", [])
    counts = Counter(item.get("archetype") for item in worlds)
    if len(worlds) != 90 or counts != Counter({arch: 10 for arch in EXPECTED_ARCHETYPES}):
        raise ValueError("source manifest world cardinality is invalid")
    seen = set()
    checked_worlds = []
    for item in worlds:
        world_id = item.get("world_id")
        if not isinstance(world_id, str) or world_id in seen:
            raise ValueError("source manifest contains an invalid/duplicate world id")
        seen.add(world_id)
        world_path = source_run / item["file"]
        if not world_path.is_file() or sha256_file(world_path) != item.get("sha256"):
            raise ValueError(f"source world is missing or changed: {world_path}")
        record = json.loads(world_path.read_text(encoding="utf-8"))
        audits = record.get("oracle", {}).get("audits", {})
        if record.get("world_id") != world_id or not audits or not all(
            result.get("passed") for result in audits.values()
        ):
            raise ValueError(f"source world failed identity/audit validation: {world_path}")
        checked_worlds.append(dict(item))

    baseline_arch = [
        dict(row)
        for row in stats.get("per_archetype", [])
        if row.get("config_id") == BASELINE_CONFIG_ID
    ]
    if (
        len(baseline_arch) != len(EXPECTED_ARCHETYPES)
        or {row.get("archetype") for row in baseline_arch} != set(EXPECTED_ARCHETYPES)
        or any(row.get("n_rollouts") != 80 for row in baseline_arch)
    ):
        raise ValueError("source stats do not contain one complete p2_r1 archetype result")
    baseline_overall = [
        dict(row)
        for row in stats.get("overall", [])
        if row.get("config_id") == BASELINE_CONFIG_ID
    ]
    if len(baseline_overall) != 1 or baseline_overall[0].get("n_rollouts") != 720:
        raise ValueError("source stats do not contain the complete p2_r1 overall result")

    return {
        "source_run": str(source_run),
        "manifest_sha256": sha256_file(manifest_path),
        "stats_sha256": sha256_file(stats_path),
        "science_fingerprint": manifest.get("science_fingerprint"),
        "master_seed": int(science["master_seed"]),
        "model": science["model"],
        "sampling": dict(science["sampling"]),
        "inference": dict(science["inference"]),
        "environment": dict(science["environment"]),
        "p2_prompt_sha256": p2_prompt["prompt_sha256"],
        "worlds": checked_worlds,
        "p2_per_archetype": baseline_arch,
        "p2_overall": baseline_overall[0],
    }


@dataclass(frozen=True)
class SamplingConfig:
    max_input_tokens: int = 18432
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    enable_thinking: bool = True
    request_timeout_s: int = 900
    transport_retries: int = 3

    def science_record(self) -> dict[str, Any]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }


def _science_payload(args, source: dict[str, Any], sampling: SamplingConfig) -> dict[str, Any]:
    prompt = current_system_prompt()
    prompt_hash = sha256_bytes(prompt.encode("utf-8"))
    if prompt_hash == source["p2_prompt_sha256"]:
        raise ValueError("current env.py prompt is identical to p2; no new condition to evaluate")
    if args.model != source["model"]:
        raise ValueError(
            f"model must match the p2 baseline ({source['model']!r}), got {args.model!r}"
        )
    if sampling.science_record() != source["sampling"]:
        raise ValueError(
            "sampling settings must exactly match p2; do not change prompt and sampling together"
        )
    if args.budget != source["environment"]["budget"] or args.max_turns != source[
        "environment"
    ]["max_turns"]:
        raise ValueError("budget and max-turns must exactly match p2")
    if (
        args.dtype != source["inference"]["dtype"]
        or args.max_model_len != source["inference"]["max_model_len"]
        or args.disable_multimodal != source["inference"]["disable_multimodal"]
    ):
        raise ValueError("inference settings must exactly match p2")
    return {
        "schema_version": SCHEMA_VERSION,
        "condition": PROMPT_ID,
        "system_prompt": prompt,
        "system_prompt_sha256": prompt_hash,
        "system_prompt_source": str(RPG_RL_DIR / "env.py") + ":SYSTEM_PROMPT",
        "env_source_sha256": sha256_file(RPG_RL_DIR / "env.py"),
        "reward_source_sha256": sha256_file(RPG_RL_DIR / "reward.py"),
        "server_client_source_sha256": sha256_file(PROMPT_COMPARE_DIR / "servers.py"),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
        "rpg_proto": "rpg_v9",
        "rpg_synergy_soft": 20,
        "sampling": sampling.science_record(),
        "inference": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "disable_multimodal": args.disable_multimodal,
        },
        "environment": {
            "budget": args.budget,
            "max_turns": args.max_turns,
            "catalog_seed": "source world generation seed",
            "budget_semantics": "hard cap matching prompt_compare_v3",
        },
        "conversation": {
            "history_mode": "full_episode_openai_messages",
            "assistant_message_source": "environment_parser_input",
            "input_length_policy": "client_chat_template_stop_before_request",
            "context_overflow": "recorded_zero_reward_synthetic_give_up",
        },
        "seed_pairing": {
            "baseline_prompt_id": SEED_PAIR_PROMPT_ID,
            "derivation": "sha256(master_seed, world_id, p2, rollout_index)",
            "master_seed": source["master_seed"],
        },
        "expected": {
            "worlds": 90,
            "worlds_per_archetype": 10,
            "rollouts_per_world": ROLLOUTS_PER_WORLD,
            "episodes": 90 * ROLLOUTS_PER_WORLD,
        },
    }


def _science_fingerprint(science: dict[str, Any], source: dict[str, Any]) -> str:
    bound = {"science": science, "source": source}
    return sha256_bytes(canonical_json(bound).encode("utf-8"))


def prepare_run(args, sampling: SamplingConfig) -> tuple[Path, dict[str, Any]]:
    source = _source_snapshot(args.source_run)
    science = _science_payload(args, source, sampling)
    fingerprint = _science_fingerprint(science, source)
    run_dir = run_directory(args.runs_root, args.run_id)
    manifest_path = run_dir / "manifest.json"
    replace_empty = bool(getattr(args, "replace_empty_run", False))
    if manifest_path.exists():
        if replace_empty:
            material_paths = []
            for child_name in ("outputs", "logs", "summaries"):
                child = run_dir / child_name
                if child.exists():
                    material_paths.extend(path for path in child.rglob("*") if path.is_file())
            for child_name in ("stats.json",):
                child = run_dir / child_name
                if child.exists():
                    material_paths.append(child)
            if material_paths:
                preview = ", ".join(str(path.relative_to(run_dir)) for path in material_paths[:5])
                raise RuntimeError(
                    "--replace-empty-run refused because inference/summary artifacts exist: "
                    f"{preview}"
                )
        elif not args.resume:
            raise RuntimeError(
                f"run already has a manifest; pass --resume to reuse it: {run_dir}"
            )
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("science_fingerprint") != fingerprint:
                raise RuntimeError("resume settings/current prompt do not match the run manifest")
            validate_manifest(manifest, source)
            return run_dir, manifest
    if run_dir.exists() and any(run_dir.iterdir()) and not replace_empty:
        raise RuntimeError(f"refusing to write into non-empty run directory: {run_dir}")
    for child in ("outputs", "logs", "logs/episode_data", "summaries"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at": utc_now(),
        "science": science,
        "science_fingerprint": fingerprint,
        "runtime": {
            "gpus": list(args.gpus_resolved),
            "ports": list(args.ports_resolved),
            "host": args.host,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "request_timeout_s": sampling.request_timeout_s,
            "transport_retries": sampling.transport_retries,
        },
        "source": source,
    }
    atomic_write_json(manifest_path, manifest)
    validate_manifest(manifest, source)
    return run_dir, manifest


def validate_manifest(manifest: dict[str, Any], source: dict[str, Any] | None = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest schema version is incorrect")
    science = manifest.get("science", {})
    if manifest.get("science_fingerprint") != _science_fingerprint(
        science, manifest["source"]
    ):
        raise ValueError("manifest scientific fingerprint does not match its contents")
    if sha256_bytes(science["system_prompt"].encode("utf-8")) != science.get(
        "system_prompt_sha256"
    ):
        raise ValueError("manifest system-prompt hash is invalid")
    if science.get("condition") != PROMPT_ID:
        raise ValueError("manifest condition is not probability belief")
    if len(manifest.get("source", {}).get("worlds", [])) != 90:
        raise ValueError("manifest does not bind exactly 90 source worlds")
    current_sources = {
        "env_source_sha256": RPG_RL_DIR / "env.py",
        "reward_source_sha256": RPG_RL_DIR / "reward.py",
        "server_client_source_sha256": PROMPT_COMPARE_DIR / "servers.py",
        "runner_source_sha256": Path(__file__).resolve(),
    }
    for key, path in current_sources.items():
        if science.get(key) != sha256_file(path):
            raise ValueError(f"scientific source changed since manifest creation: {path}")
    if source is not None:
        if canonical_json(manifest["source"]) != canonical_json(source):
            raise ValueError("source dataset or p2 result provenance changed")


def output_path(run_dir: Path, world: dict[str, Any], rollout_index: int) -> Path:
    return (
        run_dir
        / "outputs"
        / world["archetype"]
        / world["world_id"]
        / PROMPT_ID
        / f"rollout_{rollout_index:02d}.jsonl"
    )


def _request_messages_hash(messages: list[dict[str, str]]) -> str:
    return sha256_bytes(canonical_json(messages).encode("utf-8"))


def _transcript_hash(turns: Iterable[dict[str, Any]]) -> str:
    fields = [
        {
            "turn_index": turn["turn_index"],
            "observation": turn["observation"],
            "raw_model_response": turn["raw_model_response"],
            "reasoning_content": turn.get("reasoning_content"),
            "parsed_action_type": turn.get("parsed_action_type"),
            "finish_reason": turn.get("finish_reason"),
        }
        for turn in turns
    ]
    return sha256_bytes(canonical_json(fields).encode("utf-8"))


def _assistant_history_content(record: dict[str, Any]) -> str:
    if record.get("synthetic_action"):
        return CONTEXT_LIMIT_ACTION
    raw = record["raw_model_response"]
    if not record.get("parser_input_synthesized"):
        return raw
    reasoning = record.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        raise ValueError("synthesized parser input has no reasoning content")
    return f"<reasoning>{reasoning}</reasoning>\n{raw}"


def _configure_imports() -> None:
    os.environ["RPG_PROTO"] = "rpg_v9"
    os.environ["RPG_SYNERGY_SOFT"] = "20"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    wanted = [str(RPG_RL_DIR), str(RPG_V9_DIR), str(PROMPT_COMPARE_DIR)]
    sys.path[:] = [entry for entry in sys.path if entry not in wanted]
    sys.path[:0] = wanted


def load_runtime():
    """Import optional scientific/model dependencies only for actual rollouts."""
    _configure_imports()
    env_module = importlib.import_module("env")
    reward_module = importlib.import_module("reward")
    agent_module = importlib.import_module("run_agent_v6")
    servers_module = importlib.import_module("servers")
    for module, expected in (
        (env_module, RPG_RL_DIR),
        (reward_module, RPG_RL_DIR),
        (agent_module, RPG_V9_DIR),
        (servers_module, PROMPT_COMPARE_DIR),
    ):
        if Path(module.__file__).resolve().parent != expected.resolve():
            raise RuntimeError(
                f"unsafe import resolution: {module.__name__} came from {module.__file__}"
            )
    for name in ("sampler", "engine", "oracle_v6"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != RPG_V9_DIR.resolve():
            raise RuntimeError(f"unsafe rpg_v9 import resolution for {name}")

    class BaselineCompatibleEnv(env_module.RPGEnv):
        """Restore prompt_compare_v3's documented hard experiment-budget cap.

        The current shared environment supplies the new prompt but no longer has the
        per-instance comparison hooks.  This local subclass changes no shared file and
        retains the p2 run's environment semantics, keeping prompt text as the only
        scientific intervention.
        """

        def step(self, model_text):
            action_type, _ = env_module._parse_action(model_text)
            if action_type not in {"measure", "intervene"} or self._used < self.budget:
                return super().step(model_text)
            if self._done:
                raise RuntimeError("step() called on a finished episode")
            self._turn += 1
            self._memory = env_module._tag(model_text, "memory") or self._memory
            self._latest = (
                "(experiment budget exhausted — no experiment run; use existing "
                "evidence and submit an answer)"
            )
            record = {
                "turn": self._turn,
                "action_type": action_type,
                "raw": model_text,
                "error": "experiment budget exhausted",
            }
            self.turns.append(record)
            if self._turn >= self.max_turns:
                return self._terminal(None, record, forced=True)
            return self._observation(), 0.0, False, {"turn_type": "budget_exhausted"}

    return SimpleNamespace(
        Env=BaselineCompatibleEnv,
        SYSTEM_PROMPT=env_module.SYSTEM_PROMPT,
        RewardConfig=reward_module.RewardConfig,
        compute_reward=reward_module.compute_reward,
        load_world_file=agent_module.load_world_file,
        servers=servers_module,
    )


def _evaluate_terminal(runtime, answer, world, cat, gold, battery, interventions):
    cfg = runtime.RewardConfig(
        w_a=0.5,
        w_b=0.5,
        c_invalid=0.0,
        strict_part_b=True,
        require_evidence=True,
        c_no_evidence=0.0,
    )
    result = runtime.compute_reward(
        answer,
        world,
        cat,
        gold,
        battery,
        cfg=cfg,
        n_interventions=interventions,
    )
    part_a = float(result["part_a"])
    part_b = float(result["part_b"])
    return {
        "score": 0.5 * part_a + 0.5 * part_b,
        "part_a": part_a,
        "part_b": part_b,
        "accepted": bool(result["accepted"]),
        "evaluation_error": bool(result.get("reward_error", False)),
    }


def _drive_episode(
    destination: Path,
    manifest: dict[str, Any],
    world_meta: dict[str, Any],
    rollout_index: int,
    env,
    client,
    evaluate,
) -> None:
    science = manifest["science"]
    seed = derive_rollout_seed(
        manifest["source"]["master_seed"], world_meta["world_id"], rollout_index
    )
    observation = env.reset()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": science["system_prompt"]}
    ]
    turns = []
    done = False
    terminal_reward = 0.0
    info: dict[str, Any] = {}
    context_error = None
    length_source = None
    while not done:
        messages.append({"role": "user", "content": observation})
        request_messages = [dict(message) for message in messages]
        try:
            generation = client.generate(request_messages, seed)
        except client.input_length_error as exc:
            context_error = str(exc)
            length_source = "client_input"
            generation = SimpleNamespace(
                raw_text="",
                action_text=CONTEXT_LIMIT_ACTION,
                reasoning_content=None,
                finish_reason="length",
                usage={},
                attempts=0,
                latency_s=0.0,
                prompt_tokens=exc.prompt_tokens,
            )
        except client.context_length_error as exc:
            context_error = str(exc)
            length_source = "server_context"
            generation = SimpleNamespace(
                raw_text="",
                action_text=CONTEXT_LIMIT_ACTION,
                reasoning_content=None,
                finish_reason="context_length",
                usage={},
                attempts=1,
                latency_s=0.0,
                prompt_tokens=exc.prompt_tokens,
            )
        next_observation, terminal_reward, done, info = env.step(generation.action_text)
        action_type = env.turns[-1].get("action_type") if env.turns else None
        turns.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": "turn",
            "run_id": manifest["run_id"],
            "world_id": world_meta["world_id"],
            "archetype": world_meta["archetype"],
            "prompt_id": PROMPT_ID,
            "baseline_prompt_id": SEED_PAIR_PROMPT_ID,
            "rollout_index": rollout_index,
            "turn_index": len(turns),
            "request_seed": seed,
            "request_message_count": len(request_messages),
            "request_messages_sha256": _request_messages_hash(request_messages),
            "request_prompt_tokens": getattr(generation, "prompt_tokens", None),
            "observation": observation,
            "raw_model_response": generation.raw_text,
            "reasoning_content": generation.reasoning_content,
            "parser_input_synthesized": generation.action_text != generation.raw_text,
            "synthetic_action": length_source is not None,
            "request_error": context_error,
            "parsed_action_type": action_type,
            "finish_reason": generation.finish_reason,
            "timing": {"latency_s": generation.latency_s},
            "transport_attempts": generation.attempts,
            "usage": generation.usage,
        })
        messages.append({"role": "assistant", "content": generation.action_text})
        observation = next_observation

    terminal_turn = env.turns[-1] if env.turns else {}
    answer = terminal_turn.get("answer_struct", {})
    evaluation = evaluate(answer, int(info.get("n_interventions", 0)))
    last_action = turns[-1].get("parsed_action_type") if turns else None
    if length_source == "client_input":
        termination_reason = "input_length"
    elif length_source == "server_context":
        termination_reason = "context_limit"
    elif info.get("forced"):
        termination_reason = "turn_cap"
    elif last_action == "give_up":
        termination_reason = "give_up"
    elif last_action == "answer":
        termination_reason = "answer"
    else:
        termination_reason = "terminal"
    reward = float(terminal_reward)
    if not math.isclose(reward, float(info.get("reward", reward)), abs_tol=1e-12):
        raise RuntimeError("terminal environment reward disagrees with terminal info")
    for metric in ("part_a", "part_b"):
        if not math.isclose(
            float(evaluation[metric]),
            float(info.get(metric, evaluation[metric])),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"strict evaluation {metric} disagrees with environment reward")
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "terminal",
        "run_id": manifest["run_id"],
        "world_id": world_meta["world_id"],
        "archetype": world_meta["archetype"],
        "prompt_id": PROMPT_ID,
        "baseline_prompt_id": SEED_PAIR_PROMPT_ID,
        "rollout_index": rollout_index,
        "request_seed": seed,
        "sampling": science["sampling"],
        "reward": reward,
        "invalid_id_fraction": float(info.get("invalid_id_fraction", 0.0)),
        "candidate_accepted": bool(info.get("accepted", False)),
        "candidate_reward_error": bool(info.get("reward_error", False)),
        "score": float(evaluation["score"]),
        "part_a": float(evaluation["part_a"]),
        "part_b": float(evaluation["part_b"]),
        "evaluation_accepted": bool(evaluation["accepted"]),
        "evaluation_error": bool(evaluation["evaluation_error"]),
        "termination_reason": termination_reason,
        "context_limit_error": context_error,
        "intervention_count": int(info.get("n_interventions", 0)),
        "experiment_count": int(info.get("queries_used", 0)),
        "turn_count": int(info.get("turns", len(turns))),
        "transcript_sha256": _transcript_hash(turns),
        "complete": True,
    }
    atomic_write_jsonl(destination, [*turns, terminal])


class _ClientAdapter:
    def __init__(self, client, servers):
        self._client = client
        self.input_length_error = servers.InputLengthExceededError
        self.context_length_error = servers.ContextLengthExceededError

    def generate(self, messages, seed):
        return self._client.generate(messages, seed)


def run_episode(
    run_dir: Path,
    manifest: dict[str, Any],
    world_meta: dict[str, Any],
    rollout_index: int,
    client,
    runtime,
) -> str:
    destination = output_path(run_dir, world_meta, rollout_index)
    completed = completed_rollout_records(destination)
    if completed is not None:
        try:
            validate_rollout_records(completed, manifest, world_meta, rollout_index)
        except ValueError:
            pass
        else:
            return "skipped"
    source_path = Path(manifest["source"]["source_run"]) / world_meta["file"]
    world, precomputed = runtime.load_world_file(str(source_path))
    env_cfg = manifest["science"]["environment"]
    data_dir = (
        run_dir
        / "logs"
        / "episode_data"
        / world_meta["world_id"]
        / f"rollout_{rollout_index:02d}"
    )
    env = runtime.Env(
        world=world,
        gold=precomputed["gold"],
        battery=precomputed["battery"],
        max_turns=env_cfg["max_turns"],
        budget=env_cfg["budget"],
        catalog_seed=world_meta["seed"],
        data_dir=str(data_dir),
    )

    def evaluate(answer, interventions):
        return _evaluate_terminal(
            runtime,
            answer,
            world,
            env.cat,
            precomputed["gold"],
            precomputed["battery"],
            interventions,
        )

    _drive_episode(
        destination,
        manifest,
        world_meta,
        rollout_index,
        env,
        _ClientAdapter(client, runtime.servers),
        evaluate,
    )
    return "completed"


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def validate_rollout_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    world: dict[str, Any],
    rollout_index: int,
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("rollout JSONL must contain turns plus a terminal")
    turns, terminal = records[:-1], records[-1]
    seed = derive_rollout_seed(
        manifest["source"]["master_seed"], world["world_id"], rollout_index
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "world_id": world["world_id"],
        "archetype": world["archetype"],
        "prompt_id": PROMPT_ID,
        "baseline_prompt_id": SEED_PAIR_PROMPT_ID,
        "rollout_index": rollout_index,
        "request_seed": seed,
    }
    if terminal.get("record_type") != "terminal" or terminal.get("complete") is not True:
        raise ValueError("final JSONL record is not a complete terminal")
    if any(terminal.get(key) != value for key, value in expected.items()):
        raise ValueError("terminal identifiers do not match manifest/path")
    if terminal.get("sampling") != manifest["science"]["sampling"]:
        raise ValueError("terminal sampling settings differ from manifest")
    if terminal.get("turn_count") != len(turns):
        raise ValueError("terminal turn count does not match turn records")
    if terminal.get("transcript_sha256") != _transcript_hash(turns):
        raise ValueError("terminal transcript hash is invalid")
    for metric in ("score", "part_a", "part_b"):
        number = _require_finite(terminal.get(metric), f"terminal.{metric}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"terminal.{metric} is outside [0,1]")
    reward = _require_finite(terminal.get("reward"), "terminal.reward")
    if not -0.25 <= reward <= 1.0:
        raise ValueError("terminal.reward is outside [-0.25,1]")
    for count in ("intervention_count", "experiment_count", "turn_count"):
        value = terminal.get(count)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"terminal.{count} is not a nonnegative integer")
    if terminal["intervention_count"] > terminal["experiment_count"]:
        raise ValueError("interventions exceed experiments")
    if terminal["experiment_count"] > manifest["science"]["environment"]["budget"]:
        raise ValueError("experiments exceed the hard budget")
    if not 1 <= terminal["turn_count"] <= manifest["science"]["environment"]["max_turns"]:
        raise ValueError("turn count is outside the configured range")

    history = [{"role": "system", "content": manifest["science"]["system_prompt"]}]
    max_input = manifest["science"]["sampling"]["max_input_tokens"]
    malformed = 0
    synthetic = 0
    for index, turn in enumerate(turns):
        if turn.get("record_type") != "turn" or turn.get("turn_index") != index:
            raise ValueError(f"turn {index} has invalid type/index")
        if any(turn.get(key) != value for key, value in expected.items()):
            raise ValueError(f"turn {index} identifiers do not match")
        if not isinstance(turn.get("observation"), str) or not isinstance(
            turn.get("raw_model_response"), str
        ):
            raise ValueError(f"turn {index} is missing text fields")
        reasoning = turn.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"turn {index} reasoning content is not text")
        for flag in ("parser_input_synthesized", "synthetic_action"):
            if not isinstance(turn.get(flag), bool):
                raise ValueError(f"turn {index} {flag} is not boolean")
        history.append({"role": "user", "content": turn["observation"]})
        if turn.get("request_message_count") != len(history):
            raise ValueError(f"turn {index} message count is invalid")
        if turn.get("request_messages_sha256") != _request_messages_hash(history):
            raise ValueError(f"turn {index} request history hash is invalid")
        prompt_tokens = turn.get("request_prompt_tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            raise ValueError(f"turn {index} prompt-token count is invalid")
        attempts = turn.get("transport_attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"turn {index} transport-attempt count is invalid")
        if turn.get("synthetic_action"):
            synthetic += 1
            if index != len(turns) - 1 or not turn.get("request_error"):
                raise ValueError("synthetic length fallback must be the documented final turn")
            if turn.get("parsed_action_type") != "give_up":
                raise ValueError("synthetic length fallback did not parse as give_up")
            if turn.get("finish_reason") == "length":
                if prompt_tokens <= max_input or attempts != 0:
                    raise ValueError("client input-length fallback metadata is invalid")
            elif (
                turn.get("finish_reason") != "context_length"
                or attempts != 1
                or prompt_tokens > max_input
            ):
                raise ValueError("server context fallback metadata is invalid")
        else:
            if turn.get("request_error") is not None or prompt_tokens > max_input or attempts < 1:
                raise ValueError(f"turn {index} normal request metadata is invalid")
        latency = _require_finite(
            turn.get("timing", {}).get("latency_s"), f"turn {index} latency"
        )
        if latency < 0:
            raise ValueError(f"turn {index} latency is negative")
        if turn.get("parsed_action_type") is None:
            malformed += 1
        history.append({"role": "assistant", "content": _assistant_history_content(turn)})
    is_length_terminal = terminal.get("termination_reason") in {
        "input_length",
        "context_limit",
    }
    if is_length_terminal != (synthetic == 1):
        raise ValueError("synthetic turn and terminal reason disagree")
    error = terminal.get("context_limit_error")
    if is_length_terminal != (isinstance(error, str) and bool(error)):
        raise ValueError("context-limit error detail is inconsistent")
    return {"terminal": terminal, "malformed_actions": malformed}


def run_rollouts(run_dir, manifest, base_urls, sampling, api_key, runtime):
    if len(base_urls) != 3:
        raise ValueError("exactly three vLLM worker URLs are required")
    server_sampling = runtime.servers.SamplingSettings(
        max_input_tokens=sampling.max_input_tokens,
        max_tokens=sampling.max_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        enable_thinking=sampling.enable_thinking,
        request_timeout_s=sampling.request_timeout_s,
        transport_retries=sampling.transport_retries,
    )
    counter = runtime.servers.ChatTemplateTokenCounter(
        manifest["science"]["model"], enable_thinking=sampling.enable_thinking
    )
    clients = [
        runtime.servers.VLLMClient(
            url,
            manifest["science"]["served_model_name"],
            server_sampling,
            api_key=api_key,
            token_counter=counter,
        )
        for url in base_urls
    ]
    worlds = manifest["source"]["worlds"]
    buckets = [worlds[index::3] for index in range(3)]
    lock = threading.Lock()
    progress = {"groups": 0, "completed": 0, "skipped": 0}
    prior_path = run_dir / "logs" / "rollout_errors.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if not isinstance(prior, list):
            prior = []
    except (OSError, json.JSONDecodeError):
        prior = []

    def worker(worker_index):
        errors = []
        for world in buckets[worker_index]:
            with ThreadPoolExecutor(max_workers=ROLLOUTS_PER_WORLD) as pool:
                futures = {
                    pool.submit(
                        run_episode,
                        run_dir,
                        manifest,
                        world,
                        rollout_index,
                        clients[worker_index],
                        runtime,
                    ): rollout_index
                    for rollout_index in range(ROLLOUTS_PER_WORLD)
                }
                for future in as_completed(futures):
                    rollout_index = futures[future]
                    try:
                        status = future.result()
                        with lock:
                            progress[status] += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append({
                            "at": utc_now(),
                            "worker_index": worker_index,
                            "base_url": base_urls[worker_index],
                            "world_id": world["world_id"],
                            "archetype": world["archetype"],
                            "prompt_id": PROMPT_ID,
                            "rollout_index": rollout_index,
                            "request_seed": derive_rollout_seed(
                                manifest["source"]["master_seed"],
                                world["world_id"],
                                rollout_index,
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
            with lock:
                progress["groups"] += 1
                if progress["groups"] % 10 == 0 or progress["groups"] == len(worlds):
                    print(
                        f"world groups {progress['groups']}/{len(worlds)}; "
                        f"episodes completed={progress['completed']} "
                        f"skipped={progress['skipped']}",
                        flush=True,
                    )
        return errors

    errors = []
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(worker, index) for index in range(3)]
        for future in as_completed(futures):
            errors.extend(future.result())
    atomic_write_json(prior_path, [*prior, *errors])
    return {**progress, "errors": len(errors)}


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(value["reward"]) for value in values]
    scores = [float(value["score"]) for value in values]
    terminations = Counter(value["termination_reason"] for value in values)
    return {
        "n_rollouts": len(values),
        "reward_mean": _mean(rewards),
        "reward_variance": float(statistics.pvariance(rewards)),
        "avg_score": _mean(scores),
        "best_score": max(scores),
        "avg_part_a": _mean(float(value["part_a"]) for value in values),
        "avg_part_b": _mean(float(value["part_b"]) for value in values),
        "accepted_rate": _mean(float(bool(value["evaluation_accepted"])) for value in values),
        "avg_interventions": _mean(float(value["intervention_count"]) for value in values),
        "avg_experiments": _mean(float(value["experiment_count"]) for value in values),
        "avg_turns": _mean(float(value["turn_count"]) for value in values),
        "termination_counts": dict(sorted(terminations.items())),
    }


def build_stats(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    actual_source = _source_snapshot(Path(manifest["source"]["source_run"]))
    validate_manifest(manifest, actual_source)
    terminals: dict[tuple[str, int], dict[str, Any]] = {}
    missing = []
    invalid = []
    malformed = 0
    for world in manifest["source"]["worlds"]:
        for rollout_index in range(ROLLOUTS_PER_WORLD):
            path = output_path(run_dir, world, rollout_index)
            relative = path.relative_to(run_dir).as_posix()
            records = completed_rollout_records(path)
            if records is None:
                missing.append(relative)
                continue
            try:
                result = validate_rollout_records(records, manifest, world, rollout_index)
            except ValueError as exc:
                invalid.append({"path": relative, "error": str(exc)})
                missing.append(relative)
                continue
            terminals[(world["world_id"], rollout_index)] = result["terminal"]
            malformed += result["malformed_actions"]
    expected = manifest["science"]["expected"]["episodes"]
    complete = not missing and len(terminals) == expected
    stats = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "run_metadata": {
            "run_id": manifest["run_id"],
            "model": manifest["science"]["model"],
            "condition": PROMPT_ID,
            "system_prompt_sha256": manifest["science"]["system_prompt_sha256"],
            "source_science_fingerprint": manifest["source"]["science_fingerprint"],
            "sampling": manifest["science"]["sampling"],
            "environment": manifest["science"]["environment"],
        },
        "baseline": {
            "prompt_id": SEED_PAIR_PROMPT_ID,
            "config_id": BASELINE_CONFIG_ID,
            "per_archetype": manifest["source"]["p2_per_archetype"],
            "overall": manifest["source"]["p2_overall"],
        },
        "per_world": [],
        "per_archetype": [],
        "overall": {},
        "errors": {
            "malformed_model_actions": malformed,
            "candidate_reward_errors": sum(
                bool(value.get("candidate_reward_error")) for value in terminals.values()
            ),
            "evaluation_errors": sum(
                bool(value.get("evaluation_error")) for value in terminals.values()
            ),
            "context_length_terminations": sum(
                value.get("termination_reason") in {"input_length", "context_limit"}
                for value in terminals.values()
            ),
        },
        "completeness": {
            "complete": complete,
            "expected_worlds": 90,
            "actual_worlds": len(manifest["source"]["worlds"]),
            "expected_rollouts_per_world": ROLLOUTS_PER_WORLD,
            "expected_episodes": expected,
            "completed_episodes": len(terminals),
            "missing_count": len(missing),
            "missing_paths": missing,
            "invalid_files": invalid,
        },
    }
    if not complete:
        return stats

    per_world = []
    for world in manifest["source"]["worlds"]:
        values = [
            terminals[(world["world_id"], index)]
            for index in range(ROLLOUTS_PER_WORLD)
        ]
        summary = _summarize(values)
        summary["best_of_8_score"] = summary.pop("best_score")
        per_world.append({
            "world_id": world["world_id"],
            "archetype": world["archetype"],
            **summary,
        })
    stats["per_world"] = per_world

    baseline_by_arch = {
        row["archetype"]: row for row in manifest["source"]["p2_per_archetype"]
    }
    per_archetype = []
    for archetype in EXPECTED_ARCHETYPES:
        world_ids = {
            world["world_id"]
            for world in manifest["source"]["worlds"]
            if world["archetype"] == archetype
        }
        values = [
            value
            for (world_id, _), value in terminals.items()
            if world_id in world_ids
        ]
        summary = _summarize(values)
        summary.pop("best_score")
        baseline = baseline_by_arch[archetype]
        baseline_metrics = {
            key: baseline[key] for key in ("n_rollouts", "avg_score", "avg_part_a", "avg_part_b")
        }
        deltas = {
            key: summary[key] - baseline[key]
            for key in ("avg_score", "avg_part_a", "avg_part_b")
        }
        per_archetype.append({
            "archetype": archetype,
            **summary,
            "p2_baseline": baseline_metrics,
            "delta_vs_p2": deltas,
        })
    stats["per_archetype"] = per_archetype

    values = list(terminals.values())
    overall = _summarize(values)
    overall.pop("best_score")
    overall["best_of_8_score"] = _mean(
        float(row["best_of_8_score"]) for row in per_world
    )
    overall["within_group_reward_variance"] = _mean(
        float(row["reward_variance"]) for row in per_world
    )
    baseline = manifest["source"]["p2_overall"]
    overall["p2_baseline"] = {
        key: baseline[key]
        for key in (
            "n_rollouts",
            "reward_mean",
            "within_group_reward_variance",
            "avg_score",
            "best_of_8_score",
            "avg_part_a",
            "avg_part_b",
        )
    }
    overall["delta_vs_p2"] = {
        key: overall[key] - baseline[key]
        for key in ("reward_mean", "avg_score", "best_of_8_score", "avg_part_a", "avg_part_b")
    }
    stats["overall"] = overall
    return stats


def _write_archetype_csv(run_dir: Path, stats: dict[str, Any]) -> Path:
    path = run_dir / "summaries" / "per_archetype.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "archetype",
        "n_rollouts",
        "avg_score",
        "p2_avg_score",
        "delta_avg_score",
        "avg_part_a",
        "p2_avg_part_a",
        "delta_avg_part_a",
        "avg_part_b",
        "p2_avg_part_b",
        "delta_avg_part_b",
        "reward_mean",
        "accepted_rate",
        "avg_interventions",
        "avg_experiments",
        "avg_turns",
        "termination_counts",
    ]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in stats["per_archetype"]:
                writer.writerow({
                    "archetype": row["archetype"],
                    "n_rollouts": row["n_rollouts"],
                    "avg_score": row["avg_score"],
                    "p2_avg_score": row["p2_baseline"]["avg_score"],
                    "delta_avg_score": row["delta_vs_p2"]["avg_score"],
                    "avg_part_a": row["avg_part_a"],
                    "p2_avg_part_a": row["p2_baseline"]["avg_part_a"],
                    "delta_avg_part_a": row["delta_vs_p2"]["avg_part_a"],
                    "avg_part_b": row["avg_part_b"],
                    "p2_avg_part_b": row["p2_baseline"]["avg_part_b"],
                    "delta_avg_part_b": row["delta_vs_p2"]["avg_part_b"],
                    "reward_mean": row["reward_mean"],
                    "accepted_rate": row["accepted_rate"],
                    "avg_interventions": row["avg_interventions"],
                    "avg_experiments": row["avg_experiments"],
                    "avg_turns": row["avg_turns"],
                    "termination_counts": canonical_json(row["termination_counts"]),
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def aggregate_run(run_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    stats = build_stats(run_dir)
    atomic_write_json(run_dir / "stats.json", stats)
    if require_complete and not stats["completeness"]["complete"]:
        missing = stats["completeness"]["missing_paths"]
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise IncompleteRunError(
            f"refusing partial statistics: {len(missing)} rollout files are missing/invalid\n"
            f"{preview}{suffix}"
        )
    if stats["completeness"]["complete"]:
        _write_archetype_csv(run_dir, stats)
    return stats


def validate_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    stats_path = run_dir / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"statistics file is missing: {stats_path}")
    saved = json.loads(stats_path.read_text(encoding="utf-8"))
    if not saved.get("completeness", {}).get("complete"):
        raise IncompleteRunError("stats.json says the run is incomplete")
    regenerated = build_stats(run_dir)
    for key in (
        "run_metadata",
        "baseline",
        "per_world",
        "per_archetype",
        "overall",
        "errors",
        "completeness",
    ):
        if canonical_json(saved.get(key)) != canonical_json(regenerated.get(key)):
            raise ValueError(f"stats.json cannot be regenerated exactly: {key}")
    if len(saved["per_world"]) != 90 or len(saved["per_archetype"]) != 9:
        raise ValueError("summary cardinality is invalid")
    csv_path = _write_archetype_csv(run_dir, saved)
    return {
        "complete": True,
        "worlds": 90,
        "episodes": 720,
        "per_archetype_records": 9,
        "raw_jsonl_files": 720,
        "summary_csv": str(csv_path),
    }


def _sampling(args) -> SamplingConfig:
    return SamplingConfig(
        max_input_tokens=args.max_input_tokens,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        enable_thinking=args.thinking,
        request_timeout_s=args.request_timeout,
        transport_retries=args.transport_retries,
    )


def _validate_all_args(args) -> None:
    args.gpus_resolved = parse_csv_strings(args.gpus)
    args.ports_resolved = parse_ports(args.ports)
    if len(args.gpus_resolved) != 3 or len(set(args.gpus_resolved)) != 3:
        raise ValueError("--gpus must contain exactly three unique GPU ids")
    if len(args.ports_resolved) != 3 or len(set(args.ports_resolved)) != 3:
        raise ValueError("--ports must contain exactly three unique ports")
    for name in ("budget", "max_turns", "max_input_tokens", "max_new_tokens", "max_model_len"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_input_tokens + args.max_new_tokens > args.max_model_len:
        raise ValueError("max-input-tokens + max-new-tokens exceeds max-model-len")
    if not 0.0 < args.top_p <= 1.0 or not 0.0 <= args.min_p <= 1.0:
        raise ValueError("top-p/min-p are outside their supported ranges")
    if args.top_k != -1 and args.top_k < 1:
        raise ValueError("top-k must be -1 or positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu-memory-utilization must be in (0,1]")
    if args.transport_retries < 0:
        raise ValueError("transport-retries cannot be negative")
    if args.resume and args.replace_empty_run:
        raise ValueError("--resume and --replace-empty-run are mutually exclusive")


def command_all(args) -> int:
    _validate_all_args(args)
    sampling = _sampling(args)
    run_dir, manifest = prepare_run(args, sampling)
    print(f"run directory: {run_dir}", flush=True)
    print(
        "matched dataset: 90 prompt_compare_v3 worlds; "
        "expected probability-belief episodes: 720",
        flush=True,
    )
    runtime = load_runtime()
    if runtime.SYSTEM_PROMPT != manifest["science"]["system_prompt"]:
        raise RuntimeError("imported env.SYSTEM_PROMPT differs from the manifest snapshot")
    inventory = runtime.servers.inspect_gpus(args.gpus_resolved)
    launch = {
        "started_at": utc_now(),
        "gpus": inventory,
        "ports": list(args.ports_resolved),
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
    }
    atomic_write_json(
        run_dir / "logs" / f"launch_{utc_now().replace(':', '').replace('+', '_')}.json",
        launch,
    )
    settings = runtime.servers.ServerSettings(
        model=args.model,
        served_model_name=args.served_model_name or args.model,
        host=args.host,
        ports=args.ports_resolved,
        gpus=args.gpus_resolved,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        health_timeout_s=args.health_timeout,
        executable=args.vllm_executable,
        disable_multimodal=args.disable_multimodal,
    )
    manager = runtime.servers.ServerManager(settings, run_dir / "logs")
    try:
        print("starting one-GPU fit preflight, then three vLLM workers...", flush=True)
        base_urls = manager.start()
        summary = run_rollouts(
            run_dir, manifest, base_urls, sampling, args.api_key, runtime
        )
        print(f"rollout execution: {json.dumps(summary, sort_keys=True)}", flush=True)
    finally:
        manager.stop()
    stats = aggregate_run(run_dir, require_complete=True)
    validated = validate_run(run_dir)
    print(f"final completeness: {json.dumps(validated, sort_keys=True)}", flush=True)
    print(
        f"overall new score={stats['overall']['avg_score']:.6f}; "
        f"p2={stats['overall']['p2_baseline']['avg_score']:.6f}; "
        f"delta={stats['overall']['delta_vs_p2']['avg_score']:+.6f}",
        flush=True,
    )
    for row in stats["per_archetype"]:
        print(
            f"{row['archetype']}: new={row['avg_score']:.6f} "
            f"p2={row['p2_baseline']['avg_score']:.6f} "
            f"delta={row['delta_vs_p2']['avg_score']:+.6f}",
            flush=True,
        )
    print(f"per-archetype summary: {run_dir / 'summaries' / 'per_archetype.csv'}")
    return 0


def command_inspect(args) -> int:
    source = _source_snapshot(args.source_run)
    prompt = current_system_prompt()
    report = {
        "source_run": source["source_run"],
        "source_worlds": len(source["worlds"]),
        "archetype_counts": dict(Counter(w["archetype"] for w in source["worlds"])),
        "p2_prompt_sha256": source["p2_prompt_sha256"],
        "current_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "current_prompt_has_probability_belief_instruction": (
            "maintain a probability belief" in prompt
        ),
        "p2_overall": source["p2_overall"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_aggregate(args) -> int:
    stats = aggregate_run(args.run_dir.resolve(), require_complete=not args.allow_incomplete)
    print(json.dumps(stats["completeness"], indent=2, sort_keys=True))
    return 0 if stats["completeness"]["complete"] else 2


def command_validate(args) -> int:
    print(json.dumps(validate_run(args.run_dir.resolve()), indent=2, sort_keys=True))
    return 0


def _add_all_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--gpus", required=True, help="three unique physical GPU ids")
    parser.add_argument("--ports", default="18005,18006,18007")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replace-empty-run",
        action="store_true",
        help="replace a pre-inference manifest only when no output/log/summary files exist",
    )
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=18432)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument(
        "--disable-multimodal", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--api-key", default="EMPTY", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    all_parser = commands.add_parser(
        "all", help="validate source, serve, run 720 episodes, aggregate, and validate"
    )
    _add_all_arguments(all_parser)
    all_parser.set_defaults(handler=command_all)
    inspect_parser = commands.add_parser(
        "inspect", help="validate the source dataset/p2 baseline and current prompt"
    )
    inspect_parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    inspect_parser.set_defaults(handler=command_inspect)
    aggregate_parser = commands.add_parser("aggregate", help="regenerate stats and CSV")
    aggregate_parser.add_argument("--run-dir", type=Path, required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")
    aggregate_parser.set_defaults(handler=command_aggregate)
    validate_parser = commands.add_parser("validate", help="validate a complete run")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
