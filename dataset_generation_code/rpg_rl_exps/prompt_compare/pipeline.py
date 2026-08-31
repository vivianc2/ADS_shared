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
    evaluate_terminal,
)
from env import RPGEnv  # noqa: E402
from generate_v7 import audit, to_record  # noqa: E402
from run_agent_v6 import load_world_file  # noqa: E402
from sampler import ARCHETYPES as SAMPLER_ARCHETYPES, sample_world  # noqa: E402
from skins import skin_names  # noqa: E402

from servers import SamplingSettings, VLLMClient  # noqa: E402
from storage import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    completed_final_record,
    read_jsonl,
    require_finite,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "prompt_compare_v1"
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
    "overall_avg_score_heatmap.png",
    "overall_best_of_8_heatmap.png",
    "reward_summary.png",
    "archetype_avg_score_heatmap.png",
    "archetype_avg_part_a_heatmap.png",
    "archetype_avg_part_b_heatmap.png",
)


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
    """Stable seed deliberately independent of reward_id."""
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
    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": int(args.seed),
        "rpg_proto": "rpg_v9",
        "rpg_synergy_soft": 20,
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
        "chat_template": "model default via vLLM OpenAI chat API",
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
            "rollouts_per_group": rollouts_per_group,
            "episodes": (
                len(EXPECTED_ARCHETYPES)
                * worlds_per_archetype
                * len(configurations)
                * rollouts_per_group
            ),
        },
        "archetypes": list(EXPECTED_ARCHETYPES),
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


def output_path(run_dir: Path, world: dict[str, Any], config_id: str,
                rollout_index: int) -> Path:
    return (
        run_dir
        / "outputs"
        / world["archetype"]
        / world["world_id"]
        / config_id
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


def run_episode(run_dir: Path, manifest: dict[str, Any], world_meta: dict[str, Any],
                config: dict[str, Any], rollout_index: int, client: VLLMClient) -> str:
    destination = output_path(run_dir, world_meta, config["config_id"], rollout_index)
    completed = completed_final_record(destination)
    if completed is not None:
        return "skipped"

    world, precomputed = load_world_file(str(run_dir / world_meta["file"]))
    science = manifest["science"]
    env_cfg = science["environment"]
    seed = derive_rollout_seed(
        science["master_seed"], world_meta["world_id"], config["prompt_id"],
        rollout_index,
    )
    data_dir = (
        run_dir / "logs" / "episode_data" / config["config_id"]
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
        system_prompt=PROMPTS[config["prompt_id"]],
        reward_fn=REWARDS[config["reward_id"]],
    )

    observation = env.reset()
    turn_records: list[dict[str, Any]] = []
    done = False
    candidate_reward = 0.0
    info: dict[str, Any] = {}
    while not done:
        generation = client.generate(env.system_prompt, observation, seed)
        next_observation, candidate_reward, done, info = env.step(generation.action_text)
        action_type = env.turns[-1].get("action_type") if env.turns else None
        turn_records.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": "turn",
            "run_id": manifest["run_id"],
            "world_id": world_meta["world_id"],
            "archetype": world_meta["archetype"],
            "config_id": config["config_id"],
            "prompt_id": config["prompt_id"],
            "reward_id": config["reward_id"],
            "rollout_index": rollout_index,
            "turn_index": len(turn_records),
            "request_seed": seed,
            "observation": observation,
            "raw_model_response": generation.raw_text,
            "reasoning_content": generation.reasoning_content,
            "parser_input_synthesized": generation.action_text != generation.raw_text,
            "parsed_action_type": action_type,
            "finish_reason": generation.finish_reason,
            "timing": {"latency_s": generation.latency_s},
            "transport_attempts": generation.attempts,
            "usage": generation.usage,
        })
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
    last_action = turn_records[-1].get("parsed_action_type") if turn_records else None
    if info.get("forced"):
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
        "config_id": config["config_id"],
        "prompt_id": config["prompt_id"],
        "reward_id": config["reward_id"],
        "rollout_index": rollout_index,
        "request_seed": seed,
        "sampling": science["sampling"],
        "candidate_reward": float(candidate_reward),
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
    clients = [VLLMClient(url, served_name, sampling, api_key=api_key) for url in base_urls]
    groups = [
        (config, world)
        for config in science["configurations"]
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
        for config, world in buckets[worker_index]:
            with ThreadPoolExecutor(max_workers=rollouts_per_group) as group_pool:
                futures = {
                    group_pool.submit(
                        run_episode, run_dir, manifest, world, config, rollout_index, client
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
                            "config_id": config["config_id"],
                            "rollout_index": rollout_index,
                            "request_seed": derive_rollout_seed(
                                science["master_seed"], world["world_id"],
                                config["prompt_id"], rollout_index,
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
    "config_id",
    "prompt_id",
    "reward_id",
    "rollout_index",
    "request_seed",
    "sampling",
    "candidate_reward",
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


def summarize_world_rollouts(values: list[dict[str, Any]]) -> dict[str, float | int]:
    """Apply the exact per-world definitions to one completed rollout group."""
    if not values:
        raise ValueError("cannot summarize an empty rollout group")
    rewards = [float(value["candidate_reward"]) for value in values]
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
    for metric in ("candidate_reward", "score", "part_a", "part_b"):
        require_finite(final[metric], metric)
    candidate_reward = float(final["candidate_reward"])
    if candidate_reward < -0.25 or candidate_reward > 1.0:
        raise ValueError(f"candidate_reward={candidate_reward} is outside [-0.25, 1]")
    for metric in (
        "candidate_part_a", "candidate_part_b", "invalid_id_fraction",
        "score", "part_a", "part_b",
    ):
        number = require_finite(final[metric], metric)
        if number < 0.0 or number > 1.0:
            raise ValueError(f"{metric}={number} is outside [0, 1]")


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
    rollouts_per_group = science["expected"]["rollouts_per_group"]
    terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing_paths: list[str] = []
    invalid_files: list[dict[str, str]] = []
    malformed_actions = 0
    length_finishes = 0
    successful_transport_retries = 0

    for config in configs:
        for world in manifest["worlds"]:
            for rollout_index in range(rollouts_per_group):
                path = output_path(run_dir, world, config["config_id"], rollout_index)
                relative = path.relative_to(run_dir).as_posix()
                final = completed_final_record(path)
                if final is None:
                    missing_paths.append(relative)
                    continue
                try:
                    records = read_jsonl(path)
                    validate_terminal_schema(final)
                    if (
                        final["world_id"] != world["world_id"]
                        or final["config_id"] != config["config_id"]
                        or final["rollout_index"] != rollout_index
                    ):
                        raise ValueError("terminal identifiers do not match output path")
                    if final["schema_version"] != SCHEMA_VERSION or final["run_id"] != manifest["run_id"]:
                        raise ValueError("terminal schema/run identifiers do not match manifest")
                    wanted_seed = derive_rollout_seed(
                        science["master_seed"], world["world_id"], config["prompt_id"],
                        rollout_index,
                    )
                    if final["request_seed"] != wanted_seed:
                        raise ValueError("stored request seed does not match deterministic seed")
                    if final["prompt_id"] != config["prompt_id"] or final["reward_id"] != config["reward_id"]:
                        raise ValueError("terminal prompt/reward ids do not match configuration")
                    if final["archetype"] != world["archetype"]:
                        raise ValueError("terminal archetype does not match world manifest")
                    if final["sampling"] != science["sampling"]:
                        raise ValueError("terminal sampling settings do not match manifest")
                    if final["turn_count"] != len(records) - 1:
                        raise ValueError("terminal turn_count does not match JSONL turns")
                    if final["transcript_sha256"] != _transcript_hash(records[:-1]):
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
                    for turn_index, record in enumerate(records[:-1]):
                        if record.get("record_type") != "turn":
                            raise ValueError(f"record {turn_index} is not a turn record")
                        for key, wanted in (
                            ("schema_version", SCHEMA_VERSION),
                            ("run_id", manifest["run_id"]),
                            ("world_id", world["world_id"]),
                            ("archetype", world["archetype"]),
                            ("config_id", config["config_id"]),
                            ("prompt_id", config["prompt_id"]),
                            ("reward_id", config["reward_id"]),
                            ("rollout_index", rollout_index),
                            ("turn_index", turn_index),
                            ("request_seed", wanted_seed),
                        ):
                            if record.get(key) != wanted:
                                raise ValueError(f"turn {turn_index} has incorrect {key}")
                        if not isinstance(record.get("observation"), str):
                            raise ValueError(f"turn {turn_index} observation is not text")
                        if not isinstance(record.get("raw_model_response"), str):
                            raise ValueError(f"turn {turn_index} raw response is not text")
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
                        successful_transport_retries += max(
                            0, int(record.get("transport_attempts", 1)) - 1
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    invalid_files.append({"path": relative, "error": str(exc)})
                    missing_paths.append(relative)
                    continue
                terminals[(config["config_id"], world["world_id"], rollout_index)] = final

    expected_episodes = science["expected"]["episodes"]
    complete = not missing_paths and len(terminals) == expected_episodes
    error_log = _load_error_log(run_dir)
    completeness = {
        "complete": complete,
        "expected_worlds": science["expected"]["worlds"],
        "actual_worlds": len(manifest["worlds"]),
        "expected_configurations": science["expected"]["configurations"],
        "actual_configurations": len(configs),
        "expected_rollouts_per_group": rollouts_per_group,
        "expected_episodes": expected_episodes,
        "completed_episodes": len(terminals),
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
        },
        "pairing": {"warning_count": 0, "warnings": []},
        "completeness": completeness,
    }
    if not complete:
        return stats

    per_world = []
    for config in configs:
        config_id = config["config_id"]
        for world in manifest["worlds"]:
            values = [
                terminals[(config_id, world["world_id"], index)]
                for index in range(rollouts_per_group)
            ]
            summary = summarize_world_rollouts(values)
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
                for (candidate_config, world_id, _), terminal in terminals.items()
                if candidate_config == config_id
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
            terminal for (candidate_config, _, _), terminal in terminals.items()
            if candidate_config == config_id
        ]
        world_rows = [row for row in per_world if row["config_id"] == config_id]
        overall.append({
            "config_id": config_id,
            "prompt_id": config["prompt_id"],
            "reward_id": config["reward_id"],
            "n_rollouts": len(values),
            "reward_mean": _mean(float(value["candidate_reward"]) for value in values),
            "within_group_reward_variance": _mean(
                float(row["reward_variance"]) for row in world_rows
            ),
            "avg_score": _mean(float(value["score"]) for value in values),
            "best_of_8_score": _mean(float(row["best_of_8_score"]) for row in world_rows),
            "avg_part_a": _mean(float(value["part_a"]) for value in values),
            "avg_part_b": _mean(float(value["part_b"]) for value in values),
        })
    stats["overall"] = overall

    pairing_warnings = []
    for prompt_id in PROMPTS:
        for world in manifest["worlds"]:
            for rollout_index in range(rollouts_per_group):
                hashes = {
                    reward_id: terminals[(f"{prompt_id}_{reward_id}", world["world_id"], rollout_index)][
                        "transcript_sha256"
                    ]
                    for reward_id in REWARDS
                }
                if len(set(hashes.values())) != 1:
                    pairing_warnings.append({
                        "prompt_id": prompt_id,
                        "world_id": world["world_id"],
                        "archetype": world["archetype"],
                        "rollout_index": rollout_index,
                        "request_seed": derive_rollout_seed(
                            science["master_seed"], world["world_id"], prompt_id,
                            rollout_index,
                        ),
                        "transcript_hashes": hashes,
                    })
    stats["pairing"] = {
        "warning_count": len(pairing_warnings),
        "warnings": pairing_warnings,
    }
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
        "pairing",
        "completeness",
    )
    for key in comparable_keys:
        if canonical_json(stats[key]) != canonical_json(regenerated[key]):
            raise ValueError(f"stats.json cannot be reproduced from rollout files: {key}")

    missing_figures = []
    if require_figures:
        missing_figures = [
            name
            for name in REQUIRED_FIGURES
            if not (run_dir / "figures" / name).is_file()
            or (run_dir / "figures" / name).stat().st_size == 0
        ]
        if missing_figures:
            raise ValueError(f"required figures are missing: {missing_figures}")
    return {
        "complete": True,
        "worlds": expected["worlds"],
        "configurations": expected["configurations"],
        "episodes": expected["episodes"],
        "per_world_records": len(stats["per_world"]),
        "per_archetype_records": len(stats["per_archetype"]),
        "overall_records": len(stats["overall"]),
        "pairing_warnings": stats["pairing"]["warning_count"],
        "figures": len(REQUIRED_FIGURES) if require_figures else 0,
    }


def default_runs_root() -> Path:
    return EXPERIMENT_DIR / "runs"
