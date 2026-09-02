"""World generation, rollout execution, aggregation, and validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bootstrap import EXPERIMENT_DIR, assert_v9_imports, configure_imports

configure_imports()
RESOLVED_IMPORTS = assert_v9_imports()

from candidates import (  # noqa: E402
    PROMPTS,
    REWARDS,
    configuration_definitions,
    evaluate_candidate_rewards,
    evaluate_terminal,
    prompt_definitions,
)
from env import RPGEnv  # noqa: E402
from generate_v7 import audit, to_record  # noqa: E402
from run_agent_v6 import load_world_file  # noqa: E402
from sampler import ARCHETYPES as SAMPLER_ARCHETYPES, sample_world  # noqa: E402
from skins import skin_names  # noqa: E402

from servers import (  # noqa: E402
    ChatTemplateTokenCounter,
    ContextLengthExceededError,
    Generation,
    InputLengthExceededError,
    SamplingSettings,
    VLLMClient,
)
from storage import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    completed_rollout_records,
    require_finite,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "prompt_compare_v3"
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
if tuple(SAMPLER_ARCHETYPES) != EXPECTED_ARCHETYPES:
    raise RuntimeError(
        "rpg_v9 archetype list changed; refusing to silently change the experiment: "
        f"{SAMPLER_ARCHETYPES!r}"
    )

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_FIGURES = (
    "prompt_score_summary.png",
    "prompt_reward_summary.png",
    "archetype_avg_score_heatmap.png",
    "archetype_avg_part_a_heatmap.png",
    "archetype_avg_part_b_heatmap.png",
)
REQUIRED_SHEETS = (
    "prompt_score_summary.csv",
    "prompt_reward_summary.csv",
)
CONTEXT_LIMIT_ACTION = '<action type="give_up">{}</action>'


class IncompleteRunError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return runs_root.resolve() / run_id


def derive_rollout_seed(master_seed: int, world_id: str, prompt_id: str,
                        rollout_index: int) -> int:
    """Stable seed for one prompt rollout, independent of post-hoc reward choice."""
    encoded = canonical_json({
        "master_seed": int(master_seed),
        "world_id": world_id,
        "prompt_id": prompt_id,
        "rollout_index": int(rollout_index),
    }).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % 2_147_483_647


def _science_payload(args, sampling: SamplingSettings) -> dict[str, Any]:
    worlds_per_archetype = 1 if args.smoke_test else 10
    # Smoke mode reduces the worlds from 90 to 9 but deliberately retains G=8,
    # so it exercises the same batching, variance, and best-of-eight paths.
    rollouts_per_group = 8
    configurations = configuration_definitions()
    prompts = prompt_definitions()
    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": int(args.seed),
        "rpg_proto": "rpg_v9",
        "rpg_synergy_soft": 20,
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
        "chat_template": "model default via vLLM OpenAI chat API",
        "conversation": {
            "history_mode": "full_episode_openai_messages",
            "assistant_message_source": "environment_parser_input",
            "input_length_policy": "client_chat_template_stop_before_request",
            "context_overflow": "recorded_zero_reward_server_fallback_no_retry",
        },
        "inference": {
            "dtype": args.dtype,
            "max_model_len": int(args.max_model_len),
            "disable_multimodal": bool(args.disable_multimodal),
        },
        "sampling": sampling.request_record(),
        "environment": {
            "budget": int(args.budget),
            "max_turns": int(args.max_turns),
            "catalog_seed": "world generation seed",
        },
        "expected": {
            "smoke_test": bool(args.smoke_test),
            "worlds_per_archetype": worlds_per_archetype,
            "worlds": len(EXPECTED_ARCHETYPES) * worlds_per_archetype,
            "configurations": len(configurations),
            "prompt_configurations": len(prompts),
            "reward_configurations": len(REWARDS),
            "rollouts_per_group": rollouts_per_group,
            "episodes": (
                len(EXPECTED_ARCHETYPES)
                * worlds_per_archetype
                * len(prompts)
                * rollouts_per_group
            ),
            "reward_evaluations": (
                len(EXPECTED_ARCHETYPES)
                * worlds_per_archetype
                * len(prompts)
                * rollouts_per_group
                * len(REWARDS)
            ),
        },
        "archetypes": list(EXPECTED_ARCHETYPES),
        "prompts": prompts,
        "configurations": configurations,
    }


def _science_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _validate_world_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    science = manifest.get("science", {})
    if manifest.get("science_fingerprint") != _science_fingerprint(science):
        raise RuntimeError("manifest scientific fingerprint does not match its contents")
    config_ids = [item.get("config_id") for item in science.get("configurations", [])]
    expected_config_ids = [f"{prompt}_{reward}" for prompt in PROMPTS for reward in REWARDS]
    if config_ids != expected_config_ids:
        raise RuntimeError(
            f"manifest configuration matrix is {config_ids}, expected {expected_config_ids}"
        )
    prompt_ids = [item.get("prompt_id") for item in science.get("prompts", [])]
    if prompt_ids != list(PROMPTS):
        raise RuntimeError(
            f"manifest prompt inference matrix is {prompt_ids}, expected {list(PROMPTS)}"
        )
    for prompt in science["prompts"]:
        system_prompt = prompt.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise RuntimeError(f"manifest prompt {prompt['prompt_id']} has no text")
        if prompt.get("prompt_sha256") != sha256_bytes(system_prompt.encode("utf-8")):
            raise RuntimeError(f"manifest prompt hash mismatch for {prompt['prompt_id']}")
    expected = science["expected"]
    entries = manifest.get("worlds", [])
    if len(entries) != expected["worlds"]:
        raise RuntimeError(
            f"manifest has {len(entries)} worlds, expected {expected['worlds']}"
        )
    world_ids = [entry["world_id"] for entry in entries]
    if len(set(world_ids)) != len(world_ids):
        raise RuntimeError("manifest contains duplicate world ids")
    counts = Counter(entry["archetype"] for entry in entries)
    wanted = {arch: expected["worlds_per_archetype"] for arch in EXPECTED_ARCHETYPES}
    if dict(counts) != wanted:
        raise RuntimeError(f"world archetype counts are {dict(counts)}, expected {wanted}")
    for entry in entries:
        path = run_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"manifest world is missing: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"manifest world hash mismatch: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("world_id") != entry["world_id"]:
            raise RuntimeError(f"world id mismatch in {path}")
        audits = record.get("oracle", {}).get("audits", {})
        if not audits or not all(result.get("passed") for result in audits.values()):
            raise RuntimeError(f"world does not contain a complete passed audit: {path}")


def _generate_worlds(run_dir: Path, master_seed: int, count_per_archetype: int,
                     max_attempts_per_archetype: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rejected: Counter[str] = Counter()
    skins = tuple(skin_names())
    if not skins:
        raise RuntimeError("rpg_v9 exposes no skins")

    for archetype_index, archetype in enumerate(EXPECTED_ARCHETYPES):
        accepted = 0
        attempts = 0
        while accepted < count_per_archetype and attempts < max_attempts_per_archetype:
            attempts += 1
            seed = master_seed + archetype_index * 1_000_000 + attempts
            skin = skins[(attempts - 1) % len(skins)]
            try:
                world = sample_world(seed, skin=skin, archetype=archetype)
                world["ground_truth"]["_seed"] = seed
            except Exception as exc:  # noqa: BLE001
                rejected[f"sample_exc:{type(exc).__name__}"] += 1
                continue
            world_id = world["world_id"]
            if world_id in seen_ids:
                rejected["duplicate_world_id"] += 1
                continue
            try:
                result = audit(world)
            except Exception as exc:  # noqa: BLE001
                rejected[f"audit_exc:{type(exc).__name__}"] += 1
                continue
            if not result["ok"]:
                for failure in result["fails"]:
                    rejected[f"audit:{failure}"] += 1
                continue
            record = to_record(world, result)
            relative = Path("worlds") / archetype / f"world_{world_id}.json"
            destination = run_dir / relative
            atomic_write_json(destination, record)
            entry = {
                "world_id": world_id,
                "archetype": archetype,
                "skin": world["domain"],
                "seed": seed,
                "file": relative.as_posix(),
                "sha256": sha256_file(destination),
                "audit_passed": True,
            }
            entries.append(entry)
            seen_ids.add(world_id)
            accepted += 1
        if accepted != count_per_archetype:
            raise RuntimeError(
                f"generated only {accepted}/{count_per_archetype} accepted worlds for "
                f"{archetype} after {attempts} attempts; rejection counts={dict(rejected)}"
            )
    return entries, dict(sorted(rejected.items()))


def prepare_run(args, sampling: SamplingSettings) -> tuple[Path, dict[str, Any]]:
    run_dir = run_directory(Path(args.runs_root), args.run_id)
    manifest_path = run_dir / "manifest.json"
    science = _science_payload(args, sampling)
    fingerprint = _science_fingerprint(science)

    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"run directory already has a manifest; pass --resume to reuse it: {run_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("science_fingerprint") != fingerprint:
            raise RuntimeError(
                "resume settings do not match the existing run's scientific fingerprint"
            )
        _validate_world_artifacts(run_dir, manifest)
        return run_dir, manifest

    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"refusing to write into non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("worlds", "outputs", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    worlds, rejected = _generate_worlds(
        run_dir,
        int(args.seed),
        science["expected"]["worlds_per_archetype"],
        int(args.world_max_attempts),
    )
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
            "dtype": args.dtype,
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "request_timeout_s": sampling.request_timeout_s,
            "transport_retries": sampling.transport_retries,
        },
        "resolved_imports": RESOLVED_IMPORTS,
        "world_generation": {
            "sampler": "rpg_v9.sample_world",
            "audit": "rpg_v9.generate_v7.audit",
            "serialization": "rpg_v9.generate_v7.to_record",
            "rejected_by_gate": rejected,
        },
        "worlds": worlds,
    }
    atomic_write_json(manifest_path, manifest)
    _validate_world_artifacts(run_dir, manifest)
    return run_dir, manifest


def output_path(run_dir: Path, world: dict[str, Any], prompt_id: str,
                rollout_index: int) -> Path:
    return (
        run_dir
        / "outputs"
        / world["archetype"]
        / world["world_id"]
        / prompt_id
        / f"rollout_{rollout_index:02d}.jsonl"
    )


def _transcript_hash(turn_records: Iterable[dict[str, Any]]) -> str:
    paired_fields = [
        {
            "turn_index": record["turn_index"],
            "observation": record["observation"],
            "raw_model_response": record["raw_model_response"],
            "reasoning_content": record.get("reasoning_content"),
            "parsed_action_type": record.get("parsed_action_type"),
            "finish_reason": record.get("finish_reason"),
        }
        for record in turn_records
    ]
    return sha256_bytes(canonical_json(paired_fields).encode("utf-8"))


def _request_messages_hash(messages: list[dict[str, str]]) -> str:
    return sha256_bytes(canonical_json(messages).encode("utf-8"))


def _assistant_history_content(record: dict[str, Any]) -> str:
    if record.get("synthetic_action"):
        if record.get("finish_reason") not in {"length", "context_length"}:
            raise ValueError("synthetic action does not identify a length limit")
        return CONTEXT_LIMIT_ACTION
    raw = record["raw_model_response"]
    if not record.get("parser_input_synthesized"):
        return raw
    reasoning = record.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        raise ValueError("synthesized parser input has no reasoning content")
    return f"<reasoning>{reasoning}</reasoning>\n{raw}"


def run_episode(run_dir: Path, manifest: dict[str, Any], world_meta: dict[str, Any],
                prompt: dict[str, Any], rollout_index: int, client: VLLMClient) -> str:
    prompt_id = prompt["prompt_id"]
    destination = output_path(run_dir, world_meta, prompt_id, rollout_index)
    completed_records = completed_rollout_records(destination)
    if completed_records is not None:
        try:
            validate_rollout_records(
                completed_records,
                manifest,
                world_meta,
                prompt,
                rollout_index,
            )
        except ValueError:
            pass
        else:
            return "skipped"

    world, precomputed = load_world_file(str(run_dir / world_meta["file"]))
    science = manifest["science"]
    env_cfg = science["environment"]
    seed = derive_rollout_seed(
        science["master_seed"], world_meta["world_id"], prompt_id,
        rollout_index,
    )
    data_dir = (
        run_dir / "logs" / "episode_data" / prompt_id
        / world_meta["world_id"] / f"rollout_{rollout_index:02d}"
    )
    env = RPGEnv(
        world=world,
        gold=precomputed["gold"],
        battery=precomputed["battery"],
        max_turns=env_cfg["max_turns"],
        budget=env_cfg["budget"],
        catalog_seed=world_meta["seed"],
        data_dir=str(data_dir),
        system_prompt=prompt["system_prompt"],
        reward_fn=REWARDS["r1"],
    )

    observation = env.reset()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": env.system_prompt},
    ]
    turn_records: list[dict[str, Any]] = []
    done = False
    terminal_reward = 0.0
    info: dict[str, Any] = {}
    context_limit_error: str | None = None
    length_limit_source: str | None = None
    while not done:
        messages.append({"role": "user", "content": observation})
        request_messages = [dict(message) for message in messages]
        try:
            generation = client.generate(request_messages, seed)
        except InputLengthExceededError as exc:
            context_limit_error = str(exc)
            length_limit_source = "client_input"
            generation = Generation(
                raw_text="",
                action_text=CONTEXT_LIMIT_ACTION,
                reasoning_content=None,
                finish_reason="length",
                usage={},
                attempts=0,
                latency_s=0.0,
                prompt_tokens=exc.prompt_tokens,
            )
        except ContextLengthExceededError as exc:
            context_limit_error = str(exc)
            length_limit_source = "server_context"
            generation = Generation(
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
        turn_records.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": "turn",
            "run_id": manifest["run_id"],
            "world_id": world_meta["world_id"],
            "archetype": world_meta["archetype"],
            "prompt_id": prompt_id,
            "rollout_index": rollout_index,
            "turn_index": len(turn_records),
            "request_seed": seed,
            "request_message_count": len(request_messages),
            "request_messages_sha256": _request_messages_hash(request_messages),
            "request_prompt_tokens": getattr(generation, "prompt_tokens", None),
            "observation": observation,
            "raw_model_response": generation.raw_text,
            "reasoning_content": generation.reasoning_content,
            "parser_input_synthesized": generation.action_text != generation.raw_text,
            "synthetic_action": length_limit_source is not None,
            "request_error": context_limit_error,
            "parsed_action_type": action_type,
            "finish_reason": generation.finish_reason,
            "timing": {"latency_s": generation.latency_s},
            "transport_attempts": generation.attempts,
            "usage": generation.usage,
        })
        messages.append({"role": "assistant", "content": generation.action_text})
        observation = next_observation

    terminal_turn = env.turns[-1] if env.turns else {}
    answer_struct = terminal_turn.get("answer_struct", {})
    evaluation = evaluate_terminal(
        answer_struct,
        world,
        env.cat,
        precomputed["gold"],
        precomputed["battery"],
        n_interventions=int(info.get("n_interventions", 0)),
    )
    candidate_rewards = evaluate_candidate_rewards(
        answer_struct,
        world,
        env.cat,
        precomputed["gold"],
        precomputed["battery"],
        n_interventions=int(info.get("n_interventions", 0)),
    )
    if not math.isclose(
        candidate_rewards["r1"], float(terminal_reward), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("terminal r1 reward disagrees with the environment return value")
    last_action = turn_records[-1].get("parsed_action_type") if turn_records else None
    if length_limit_source == "client_input":
        termination_reason = "input_length"
    elif length_limit_source == "server_context":
        termination_reason = "context_limit"
    elif info.get("forced"):
        termination_reason = "turn_cap"
    elif last_action == "give_up":
        termination_reason = "give_up"
    elif last_action == "answer":
        termination_reason = "answer"
    else:
        termination_reason = "terminal"

    terminal = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "terminal",
        "run_id": manifest["run_id"],
        "world_id": world_meta["world_id"],
        "archetype": world_meta["archetype"],
        "prompt_id": prompt_id,
        "rollout_index": rollout_index,
        "request_seed": seed,
        "sampling": science["sampling"],
        "candidate_rewards": candidate_rewards,
        "candidate_part_a": float(info.get("part_a", 0.0)),
        "candidate_part_b": float(info.get("part_b", 0.0)),
        "invalid_id_fraction": float(info.get("invalid_id_fraction", 0.0)),
        "candidate_accepted": bool(info.get("accepted", False)),
        "candidate_reward_error": bool(info.get("reward_error", False)),
        "score": evaluation["score"],
        "part_a": evaluation["part_a"],
        "part_b": evaluation["part_b"],
        "evaluation_accepted": evaluation["accepted"],
        "evaluation_error": evaluation["evaluation_error"],
        "termination_reason": termination_reason,
        "context_limit_error": context_limit_error,
        "intervention_count": int(info.get("n_interventions", 0)),
        "experiment_count": int(info.get("queries_used", 0)),
        "turn_count": int(info.get("turns", len(turn_records))),
        "transcript_sha256": _transcript_hash(turn_records),
        "complete": True,
    }
    atomic_write_jsonl(destination, [*turn_records, terminal])
    return "completed"


def run_rollouts(run_dir: Path, manifest: dict[str, Any], base_urls: tuple[str, ...],
                 sampling: SamplingSettings, api_key: str = "EMPTY") -> dict[str, Any]:
    if len(base_urls) != 3:
        raise ValueError("exactly three worker base URLs are required")
    science = manifest["science"]
    served_name = science["served_model_name"]
    token_counter = ChatTemplateTokenCounter(
        science["model"],
        enable_thinking=sampling.enable_thinking,
    )
    clients = [
        VLLMClient(
            url,
            served_name,
            sampling,
            api_key=api_key,
            token_counter=token_counter,
        )
        for url in base_urls
    ]
    groups = [
        (prompt, world)
        for prompt in science["prompts"]
        for world in manifest["worlds"]
    ]
    buckets = [groups[index::3] for index in range(3)]
    rollouts_per_group = science["expected"]["rollouts_per_group"]
    progress_lock = threading.Lock()
    progress = {"groups": 0, "completed": 0, "skipped": 0}

    prior_errors_path = run_dir / "logs" / "rollout_errors.json"
    if prior_errors_path.exists():
        prior_errors = json.loads(prior_errors_path.read_text(encoding="utf-8"))
        if not isinstance(prior_errors, list):
            prior_errors = []
    else:
        prior_errors = []

    def worker(worker_index: int) -> list[dict[str, Any]]:
        errors = []
        client = clients[worker_index]
        for prompt, world in buckets[worker_index]:
            with ThreadPoolExecutor(max_workers=rollouts_per_group) as group_pool:
                futures = {
                    group_pool.submit(
                        run_episode, run_dir, manifest, world, prompt, rollout_index, client
                    ): rollout_index
                    for rollout_index in range(rollouts_per_group)
                }
                for future in as_completed(futures):
                    rollout_index = futures[future]
                    try:
                        status = future.result()
                        with progress_lock:
                            progress[status] += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append({
                            "at": utc_now(),
                            "worker_index": worker_index,
                            "base_url": base_urls[worker_index],
                            "world_id": world["world_id"],
                            "archetype": world["archetype"],
                            "prompt_id": prompt["prompt_id"],
                            "rollout_index": rollout_index,
                            "request_seed": derive_rollout_seed(
                                science["master_seed"], world["world_id"],
                                prompt["prompt_id"], rollout_index,
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
            with progress_lock:
                progress["groups"] += 1
                if progress["groups"] % 10 == 0 or progress["groups"] == len(groups):
                    print(
                        f"rollout groups {progress['groups']}/{len(groups)}; "
                        f"episodes completed={progress['completed']} skipped={progress['skipped']}",
                        flush=True,
                    )
        return errors

    current_errors = []
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(worker, index) for index in range(3)]
        for future in as_completed(futures):
            current_errors.extend(future.result())
    atomic_write_json(prior_errors_path, [*prior_errors, *current_errors])
    return {**progress, "errors": len(current_errors)}


TERMINAL_REQUIRED = (
    "schema_version",
    "world_id",
    "run_id",
    "archetype",
    "prompt_id",
    "rollout_index",
    "request_seed",
    "sampling",
    "candidate_rewards",
    "candidate_part_a",
    "candidate_part_b",
    "invalid_id_fraction",
    "candidate_accepted",
    "candidate_reward_error",
    "score",
    "part_a",
    "part_b",
    "evaluation_accepted",
    "evaluation_error",
    "termination_reason",
    "context_limit_error",
    "intervention_count",
    "experiment_count",
    "turn_count",
    "transcript_sha256",
    "complete",
)


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _population_variance(values: list[float]) -> float:
    return float(statistics.pvariance(values))


def summarize_world_rollouts(values: list[dict[str, Any]],
                             reward_id: str) -> dict[str, float | int]:
    """Apply the exact per-world definitions to one completed rollout group."""
    if not values:
        raise ValueError("cannot summarize an empty rollout group")
    if reward_id not in REWARDS:
        raise ValueError(f"unknown reward id: {reward_id}")
    rewards = [float(value["candidate_rewards"][reward_id]) for value in values]
    scores = [float(value["score"]) for value in values]
    return {
        "n_rollouts": len(values),
        "reward_mean": _mean(rewards),
        "reward_variance": _population_variance(rewards),
        "avg_score": _mean(scores),
        "best_of_8_score": max(scores),
        "avg_part_a": _mean(float(value["part_a"]) for value in values),
        "avg_part_b": _mean(float(value["part_b"]) for value in values),
    }


def validate_terminal_schema(final: dict[str, Any]) -> None:
    absent = [key for key in TERMINAL_REQUIRED if key not in final]
    if absent:
        raise ValueError(f"terminal record missing keys {absent}")
    if final.get("record_type") != "terminal" or final.get("complete") is not True:
        raise ValueError("final record is not a completed terminal summary")
    context_error = final["context_limit_error"]
    if final.get("termination_reason") in {"input_length", "context_limit"}:
        if not isinstance(context_error, str) or not context_error:
            raise ValueError("length-limit terminal has no error detail")
    elif context_error is not None:
        raise ValueError("non-length terminal unexpectedly has a context-limit error")
    for metric in ("score", "part_a", "part_b"):
        require_finite(final[metric], metric)
    rewards = final["candidate_rewards"]
    if not isinstance(rewards, dict) or set(rewards) != set(REWARDS):
        raise ValueError("candidate_rewards must contain exactly r1, r2, and r3")
    for reward_id, value in rewards.items():
        number = require_finite(value, f"candidate_rewards.{reward_id}")
        if number < -0.25 or number > 1.0:
            raise ValueError(
                f"candidate_rewards.{reward_id}={number} is outside [-0.25, 1]"
            )
    for metric in (
        "candidate_part_a", "candidate_part_b", "invalid_id_fraction",
        "score", "part_a", "part_b",
    ):
        number = require_finite(final[metric], metric)
        if number < 0.0 or number > 1.0:
            raise ValueError(f"{metric}={number} is outside [0, 1]")
    candidate_part_a = float(final["candidate_part_a"])
    candidate_part_b = float(final["candidate_part_b"])
    invalid_fraction = float(final["invalid_id_fraction"])
    expected_rewards = {
        "r1": 0.5 * candidate_part_a + 0.5 * candidate_part_b - 0.25 * invalid_fraction,
        "r2": candidate_part_a - 0.25 * invalid_fraction,
        "r3": candidate_part_b - 0.25 * invalid_fraction,
    }
    for reward_id, expected_reward in expected_rewards.items():
        if not math.isclose(
            float(rewards[reward_id]), expected_reward, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"candidate_rewards.{reward_id} disagrees with its stored components"
            )


def _validate_rollout_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    world: dict[str, Any],
    prompt: dict[str, Any],
    rollout_index: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("rollout JSONL has no records")
    final = records[-1]
    turns = records[:-1]
    validate_terminal_schema(final)

    science = manifest["science"]
    prompt_id = prompt["prompt_id"]
    wanted_seed = derive_rollout_seed(
        science["master_seed"],
        world["world_id"],
        prompt_id,
        rollout_index,
    )
    expected_terminal_fields = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "world_id": world["world_id"],
        "archetype": world["archetype"],
        "prompt_id": prompt_id,
        "rollout_index": rollout_index,
        "request_seed": wanted_seed,
        "sampling": science["sampling"],
    }
    if any(final.get(key) != value for key, value in expected_terminal_fields.items()):
        raise ValueError("terminal identifiers or settings do not match the manifest/path")
    if final["turn_count"] != len(turns):
        raise ValueError("terminal turn_count does not match JSONL turns")
    if final["transcript_sha256"] != _transcript_hash(turns):
        raise ValueError("terminal transcript hash does not match turn records")

    for count_name in ("intervention_count", "experiment_count", "turn_count"):
        count = final[count_name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"terminal {count_name} is not a nonnegative integer")
    if final["intervention_count"] > final["experiment_count"]:
        raise ValueError("intervention_count exceeds experiment_count")
    if final["experiment_count"] > science["environment"]["budget"]:
        raise ValueError("experiment_count exceeds environment budget")
    if not 1 <= final["turn_count"] <= science["environment"]["max_turns"]:
        raise ValueError("turn_count is outside the environment turn range")

    request_history: list[dict[str, str]] = [
        {"role": "system", "content": prompt["system_prompt"]},
    ]
    malformed_actions = 0
    length_finishes = 0
    successful_transport_retries = 0
    synthetic_turns = 0
    max_input_tokens = science["sampling"]["max_input_tokens"]
    for turn_index, record in enumerate(turns):
        if record.get("record_type") != "turn":
            raise ValueError(f"record {turn_index} is not a turn record")
        expected_turn_fields = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "world_id": world["world_id"],
            "archetype": world["archetype"],
            "prompt_id": prompt_id,
            "rollout_index": rollout_index,
            "turn_index": turn_index,
            "request_seed": wanted_seed,
        }
        for key, wanted in expected_turn_fields.items():
            if record.get(key) != wanted:
                raise ValueError(f"turn {turn_index} has incorrect {key}")
        if not isinstance(record.get("observation"), str):
            raise ValueError(f"turn {turn_index} observation is not text")
        if not isinstance(record.get("raw_model_response"), str):
            raise ValueError(f"turn {turn_index} raw response is not text")
        reasoning = record.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"turn {turn_index} reasoning content is not text")
        if not isinstance(record.get("parser_input_synthesized"), bool):
            raise ValueError(f"turn {turn_index} parser synthesis flag is not boolean")
        if not isinstance(record.get("synthetic_action"), bool):
            raise ValueError(f"turn {turn_index} synthetic-action flag is not boolean")

        request_error = record.get("request_error")
        if record["synthetic_action"]:
            synthetic_turns += 1
            if turn_index != len(turns) - 1:
                raise ValueError("only the final turn may be a synthetic length fallback")
            if not isinstance(request_error, str) or not request_error:
                raise ValueError(f"turn {turn_index} synthetic action has no request error")
            if record.get("parsed_action_type") != "give_up":
                raise ValueError(f"turn {turn_index} length fallback is not give_up")
            finish_reason = record.get("finish_reason")
            if finish_reason not in {"length", "context_length"}:
                raise ValueError(f"turn {turn_index} synthetic action has invalid finish reason")
            expected_reason = (
                "input_length" if finish_reason == "length" else "context_limit"
            )
            if final["termination_reason"] != expected_reason:
                raise ValueError(
                    f"turn {turn_index} length fallback disagrees with terminal"
                )
        elif request_error is not None:
            raise ValueError(f"turn {turn_index} has an unexpected request error")

        request_history.append({"role": "user", "content": record["observation"]})
        if record.get("request_message_count") != len(request_history):
            raise ValueError(f"turn {turn_index} request message count is incorrect")
        if record.get("request_messages_sha256") != _request_messages_hash(request_history):
            raise ValueError(f"turn {turn_index} request history hash is incorrect")

        prompt_tokens = record.get("request_prompt_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 1
        ):
            raise ValueError(f"turn {turn_index} request prompt-token count is invalid")
        attempts = record.get("transport_attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"turn {turn_index} transport-attempt count is invalid")
        if record["synthetic_action"] and record["finish_reason"] == "length":
            if prompt_tokens <= max_input_tokens:
                raise ValueError(f"turn {turn_index} input stop did not exceed its limit")
            if attempts != 0:
                raise ValueError(f"turn {turn_index} client input stop reached transport")
        else:
            if prompt_tokens > max_input_tokens:
                raise ValueError(f"turn {turn_index} oversized prompt was sent to the server")
            if attempts < 1:
                raise ValueError(f"turn {turn_index} server request has no transport attempt")

        request_history.append({
            "role": "assistant",
            "content": _assistant_history_content(record),
        })
        latency = require_finite(
            record.get("timing", {}).get("latency_s"),
            f"turn {turn_index} latency",
        )
        if latency < 0:
            raise ValueError(f"turn {turn_index} latency is negative")
        if record.get("parsed_action_type") is None:
            malformed_actions += 1
        if record.get("finish_reason") == "length":
            length_finishes += 1
        successful_transport_retries += max(0, attempts - 1)

    length_terminal = final["termination_reason"] in {"input_length", "context_limit"}
    if length_terminal != (synthetic_turns == 1):
        raise ValueError("synthetic length action and terminal reason are inconsistent")
    return {
        "final": final,
        "malformed_actions": malformed_actions,
        "length_finishes": length_finishes,
        "successful_transport_retries": successful_transport_retries,
    }


def validate_rollout_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    world: dict[str, Any],
    prompt: dict[str, Any],
    rollout_index: int,
) -> dict[str, Any]:
    """Validate one complete rollout identically for resume and aggregation."""
    try:
        return _validate_rollout_records(
            records,
            manifest,
            world,
            prompt,
            rollout_index,
        )
    except ValueError:
        raise
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed rollout record: {exc}") from exc


def _load_error_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "logs" / "rollout_errors.json"
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [{"error_type": "InvalidErrorLog", "error": str(path)}]
    return value if isinstance(value, list) else []


def build_stats(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _validate_world_artifacts(run_dir, manifest)
    science = manifest["science"]
    configs = science["configurations"]
    prompts = science["prompts"]
    rollouts_per_group = science["expected"]["rollouts_per_group"]
    terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing_paths: list[str] = []
    invalid_files: list[dict[str, str]] = []
    malformed_actions = 0
    length_finishes = 0
    successful_transport_retries = 0

    for prompt in prompts:
        for world in manifest["worlds"]:
            for rollout_index in range(rollouts_per_group):
                path = output_path(run_dir, world, prompt["prompt_id"], rollout_index)
                relative = path.relative_to(run_dir).as_posix()
                records = completed_rollout_records(path)
                if records is None:
                    missing_paths.append(relative)
                    continue
                try:
                    validated = validate_rollout_records(
                        records,
                        manifest,
                        world,
                        prompt,
                        rollout_index,
                    )
                except ValueError as exc:
                    invalid_files.append({"path": relative, "error": str(exc)})
                    missing_paths.append(relative)
                    continue
                final = validated["final"]
                malformed_actions += validated["malformed_actions"]
                length_finishes += validated["length_finishes"]
                successful_transport_retries += validated["successful_transport_retries"]
                terminals[(prompt["prompt_id"], world["world_id"], rollout_index)] = final

    expected_episodes = science["expected"]["episodes"]
    complete = not missing_paths and len(terminals) == expected_episodes
    error_log = _load_error_log(run_dir)
    completeness = {
        "complete": complete,
        "expected_worlds": science["expected"]["worlds"],
        "actual_worlds": len(manifest["worlds"]),
        "expected_configurations": science["expected"]["configurations"],
        "actual_configurations": len(configs),
        "expected_prompt_configurations": science["expected"]["prompt_configurations"],
        "actual_prompt_configurations": len(prompts),
        "expected_rollouts_per_group": rollouts_per_group,
        "expected_episodes": expected_episodes,
        "completed_episodes": len(terminals),
        "expected_reward_evaluations": science["expected"]["reward_evaluations"],
        "completed_reward_evaluations": len(terminals) * len(REWARDS),
        "missing_count": len(missing_paths),
        "missing_paths": missing_paths,
        "invalid_files": invalid_files,
    }

    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "run_metadata": {
            "run_id": manifest["run_id"],
            "master_seed": science["master_seed"],
            "model": science["model"],
            "rpg_proto": science["rpg_proto"],
            "rpg_synergy_soft": science["rpg_synergy_soft"],
            "sampling": science["sampling"],
            "inference": science["inference"],
            "conversation": science["conversation"],
            "environment": science["environment"],
            "smoke_test": science["expected"]["smoke_test"],
            "science_fingerprint": manifest["science_fingerprint"],
        },
        "configurations": configs,
        "per_world": [],
        "per_archetype": [],
        "overall": [],
        "errors": {
            "rollout_failures_logged": len(error_log),
            "rollout_failures_by_type": dict(sorted(Counter(
                item.get("error_type", "unknown") for item in error_log
            ).items())),
            "malformed_model_actions": malformed_actions,
            "length_finish_reasons": length_finishes,
            "successful_transport_retries": successful_transport_retries,
            "candidate_reward_errors": sum(
                bool(item.get("candidate_reward_error")) for item in terminals.values()
            ),
            "evaluation_errors": sum(
                bool(item.get("evaluation_error")) for item in terminals.values()
            ),
            "context_length_terminations": sum(
                item.get("termination_reason") in {"input_length", "context_limit"}
                for item in terminals.values()
            ),
            "input_length_terminations": sum(
                item.get("termination_reason") == "input_length"
                for item in terminals.values()
            ),
            "server_context_length_terminations": sum(
                item.get("termination_reason") == "context_limit"
                for item in terminals.values()
            ),
        },
        "reward_application": {
            "mode": "post_hoc_shared_rollout",
            "rewards_per_terminal": len(REWARDS),
        },
        "completeness": completeness,
    }
    if not complete:
        return stats

    per_world = []
    for config in configs:
        config_id = config["config_id"]
        for world in manifest["worlds"]:
            values = [
                terminals[(config["prompt_id"], world["world_id"], index)]
                for index in range(rollouts_per_group)
            ]
            summary = summarize_world_rollouts(values, config["reward_id"])
            per_world.append({
                "config_id": config_id,
                "prompt_id": config["prompt_id"],
                "reward_id": config["reward_id"],
                "world_id": world["world_id"],
                "archetype": world["archetype"],
                **summary,
            })
    stats["per_world"] = per_world

    per_archetype = []
    for config in configs:
        config_id = config["config_id"]
        for archetype in EXPECTED_ARCHETYPES:
            values = [
                terminal
                for (candidate_prompt, world_id, _), terminal in terminals.items()
                if candidate_prompt == config["prompt_id"]
                and next(
                    world["archetype"] for world in manifest["worlds"]
                    if world["world_id"] == world_id
                ) == archetype
            ]
            per_archetype.append({
                "config_id": config_id,
                "prompt_id": config["prompt_id"],
                "reward_id": config["reward_id"],
                "archetype": archetype,
                "n_rollouts": len(values),
                "avg_score": _mean(float(value["score"]) for value in values),
                "avg_part_a": _mean(float(value["part_a"]) for value in values),
                "avg_part_b": _mean(float(value["part_b"]) for value in values),
            })
    stats["per_archetype"] = per_archetype

    overall = []
    for config in configs:
        config_id = config["config_id"]
        values = [
            terminal for (candidate_prompt, _, _), terminal in terminals.items()
            if candidate_prompt == config["prompt_id"]
        ]
        world_rows = [row for row in per_world if row["config_id"] == config_id]
        overall.append({
            "config_id": config_id,
            "prompt_id": config["prompt_id"],
            "reward_id": config["reward_id"],
            "n_rollouts": len(values),
            "reward_mean": _mean(
                float(value["candidate_rewards"][config["reward_id"]])
                for value in values
            ),
            "within_group_reward_variance": _mean(
                float(row["reward_variance"]) for row in world_rows
            ),
            "avg_score": _mean(float(value["score"]) for value in values),
            "best_of_8_score": _mean(float(row["best_of_8_score"]) for row in world_rows),
            "avg_part_a": _mean(float(value["part_a"]) for value in values),
            "avg_part_b": _mean(float(value["part_b"]) for value in values),
        })
    stats["overall"] = overall

    return stats


def aggregate_run(run_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    stats = build_stats(run_dir)
    atomic_write_json(run_dir / "stats.json", stats)
    if require_complete and not stats["completeness"]["complete"]:
        missing = stats["completeness"]["missing_paths"]
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise IncompleteRunError(
            f"aggregation refused partial statistics: {len(missing)} rollout files are "
            f"missing or incomplete\n{preview}{suffix}"
        )
    return stats


def _in_unit_interval(value: Any, label: str) -> None:
    number = require_finite(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label}={number} is outside [0, 1]")


def validate_run(run_dir: Path, *, require_figures: bool = True) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _validate_world_artifacts(run_dir, manifest)
    stats_path = run_dir / "stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"canonical statistics file is missing: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not stats.get("completeness", {}).get("complete"):
        raise IncompleteRunError("stats.json says the run is incomplete")

    expected = manifest["science"]["expected"]
    expected_config_ids = {
        config["config_id"] for config in manifest["science"]["configurations"]
    }
    if {row["config_id"] for row in stats["overall"]} != expected_config_ids:
        raise ValueError("overall stats do not contain exactly the nine configurations")
    if len(stats["per_world"]) != expected["configurations"] * expected["worlds"]:
        raise ValueError("per_world record count is incorrect")
    if len(stats["per_archetype"]) != expected["configurations"] * len(EXPECTED_ARCHETYPES):
        raise ValueError("per_archetype record count is incorrect")
    if len(stats["overall"]) != expected["configurations"]:
        raise ValueError("overall record count is incorrect")
    completeness = stats["completeness"]
    if completeness["completed_episodes"] != expected["episodes"]:
        raise ValueError("completed episode count is incorrect")
    if completeness["completed_reward_evaluations"] != expected["reward_evaluations"]:
        raise ValueError("completed reward-evaluation count is incorrect")
    if stats.get("reward_application") != {
        "mode": "post_hoc_shared_rollout",
        "rewards_per_terminal": expected["reward_configurations"],
    }:
        raise ValueError("reward application metadata is incorrect")

    for section in ("per_world", "per_archetype", "overall"):
        for index, row in enumerate(stats[section]):
            for metric in ("avg_score", "avg_part_a", "avg_part_b"):
                _in_unit_interval(row[metric], f"{section}[{index}].{metric}")
            if "best_of_8_score" in row:
                _in_unit_interval(
                    row["best_of_8_score"], f"{section}[{index}].best_of_8_score"
                )
            for metric in ("reward_mean", "reward_variance", "within_group_reward_variance"):
                if metric in row:
                    number = require_finite(row[metric], f"{section}[{index}].{metric}")
                    if metric == "reward_mean" and not -0.25 <= number <= 1.0:
                        raise ValueError(f"{section}[{index}].{metric} is outside [-0.25, 1]")
                    if "variance" in metric and not 0.0 <= number <= 0.390625 + 1e-12:
                        raise ValueError(
                            f"{section}[{index}].{metric} is outside [0, 0.390625]"
                        )

    for row in stats["per_world"]:
        if row["n_rollouts"] != expected["rollouts_per_group"]:
            raise ValueError("a per_world row has the wrong rollout count")
    expected_archetype_rollouts = (
        expected["worlds_per_archetype"] * expected["rollouts_per_group"]
    )
    for row in stats["per_archetype"]:
        if row["n_rollouts"] != expected_archetype_rollouts:
            raise ValueError("a per_archetype row has the wrong rollout count")
    expected_config_rollouts = expected["worlds"] * expected["rollouts_per_group"]
    for row in stats["overall"]:
        if row["n_rollouts"] != expected_config_rollouts:
            raise ValueError("an overall row has the wrong rollout count")

    regenerated = build_stats(run_dir)
    comparable_keys = (
        "run_metadata",
        "configurations",
        "per_world",
        "per_archetype",
        "overall",
        "errors",
        "reward_application",
        "completeness",
    )
    for key in comparable_keys:
        if canonical_json(stats[key]) != canonical_json(regenerated[key]):
            raise ValueError(f"stats.json cannot be reproduced from rollout files: {key}")

    missing_figures = []
    if require_figures:
        missing_figures = [
            name
            for name in (*REQUIRED_FIGURES, *REQUIRED_SHEETS)
            if not (run_dir / "figures" / name).is_file()
            or (run_dir / "figures" / name).stat().st_size == 0
        ]
        if missing_figures:
            raise ValueError(
                f"required visualization artifacts are missing: {missing_figures}"
            )
    return {
        "complete": True,
        "worlds": expected["worlds"],
        "configurations": expected["configurations"],
        "prompt_configurations": expected["prompt_configurations"],
        "episodes": expected["episodes"],
        "reward_evaluations": expected["reward_evaluations"],
        "per_world_records": len(stats["per_world"]),
        "per_archetype_records": len(stats["per_archetype"]),
        "overall_records": len(stats["overall"]),
        "figures": len(REQUIRED_FIGURES) if require_figures else 0,
    }


def default_runs_root() -> Path:
    return EXPERIMENT_DIR / "runs"
