#!/usr/bin/env python3
"""Expand SkyRL eval dumps into editable, per-world rollout artifacts.

Each recovered rollout keeps three complementary representations:

* ``raw.json``: the original JSONL record, without changing its fields;
* ``response.txt``: the exact decoded ``output_response``;
* ``transcript.txt``: ``input_prompt`` followed by ``output_response``.

Worlds are identified by ``(archetype, skin, seed)``.  Repeated records for a world
are numbered in their original dump order, which corresponds to SkyRL's independent
eval samples for that world.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _steps(value: str) -> List[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one step is required")
    return result


def _total_reward(value: Any) -> float:
    if isinstance(value, list):
        return float(sum(item for item in value if isinstance(item, (int, float))))
    return float(value or 0.0)


def _records(path: Path) -> Iterable[tuple[int, str, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, line, json.loads(line)


def recover(run_dir: Path, steps: List[int], output_name: str) -> Path:
    export_dir = run_dir / "exports"
    dump_root = export_dir / "dumped_evals"
    destination = export_dir / output_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if not dump_root.is_dir():
        raise FileNotFoundError(f"eval dump directory not found: {dump_root}")

    export_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.tmp-", dir=export_dir))
    index: List[Dict[str, Any]] = []
    step_summaries: Dict[str, Any] = {}

    try:
        for step in steps:
            source_dir = dump_root / f"global_step_{step}_evals"
            if not source_dir.is_dir():
                raise FileNotFoundError(f"eval dump not found for step {step}: {source_dir}")

            output_step = temporary / f"global_step_{step}"
            output_step.mkdir(parents=True)
            aggregate = source_dir / "aggregated_results.jsonl"
            if aggregate.exists():
                shutil.copy2(aggregate, output_step / aggregate.name)

            rollout_counters: Counter[tuple[str, str, int]] = Counter()
            archetype_counts: Counter[str] = Counter()
            world_rollouts: defaultdict[tuple[str, str, int], int] = defaultdict(int)

            sources = sorted(source_dir.glob("rpg_v9_eval_*.jsonl"))
            if not sources:
                raise FileNotFoundError(f"no per-archetype rollout dumps in {source_dir}")

            for source in sources:
                for source_line, raw_line, record in _records(source):
                    extras = record.get("env_extras") or {}
                    info = extras.get("extra_info", extras) or {}
                    missing = [key for key in ("archetype", "skin", "seed") if key not in info]
                    if missing:
                        raise ValueError(f"{source}:{source_line} lacks world identity: {missing}")

                    archetype = str(info["archetype"])
                    skin = str(info["skin"])
                    seed = int(info["seed"])
                    identity = (archetype, skin, seed)
                    rollout_index = rollout_counters[identity]
                    rollout_counters[identity] += 1
                    archetype_counts[archetype] += 1
                    world_rollouts[identity] += 1

                    relative_dir = Path(
                        f"global_step_{step}",
                        _safe_component(archetype),
                        f"{_safe_component(skin)}__seed_{seed}",
                    )
                    output_dir = temporary / relative_dir
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"rollout_{rollout_index}"
                    raw_path = output_dir / f"{stem}.raw.json"
                    response_path = output_dir / f"{stem}.response.txt"
                    transcript_path = output_dir / f"{stem}.transcript.txt"

                    raw_path.write_text(
                        raw_line if raw_line.endswith("\n") else raw_line + "\n",
                        encoding="utf-8",
                    )
                    response = str(record.get("output_response") or "")
                    prompt = str(record.get("input_prompt") or "")
                    response_path.write_text(response, encoding="utf-8")
                    transcript_path.write_text(prompt + response, encoding="utf-8")

                    index.append(
                        {
                            "global_step": step,
                            "archetype": archetype,
                            "skin": skin,
                            "seed": seed,
                            "rollout_index": rollout_index,
                            "data_source": record.get("data_source"),
                            "stop_reason": record.get("stop_reason"),
                            "total_reward": _total_reward(record.get("score")),
                            "source_file": str(source.relative_to(run_dir)),
                            "source_line": source_line,
                            "raw_json": str((relative_dir / f"{stem}.raw.json").as_posix()),
                            "response": str((relative_dir / f"{stem}.response.txt").as_posix()),
                            "transcript": str((relative_dir / f"{stem}.transcript.txt").as_posix()),
                        }
                    )

            step_summaries[str(step)] = {
                "source": str(source_dir.relative_to(run_dir)),
                "rollouts": sum(archetype_counts.values()),
                "worlds": len(world_rollouts),
                "rollouts_per_world": dict(sorted(Counter(world_rollouts.values()).items())),
                "rollouts_by_archetype": dict(sorted(archetype_counts.items())),
            }

        with (temporary / "index.jsonl").open("w", encoding="utf-8") as handle:
            for item in index:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

        manifest = {
            "run_dir": str(run_dir),
            "source": str(dump_root.relative_to(run_dir)),
            "steps": steps,
            "total_rollouts": len(index),
            "step_summaries": step_summaries,
            "representations": {
                "raw_json": "Original JSONL record with fields unchanged.",
                "response": "Exact decoded output_response text.",
                "transcript": "Exact input_prompt + output_response text.",
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# Recovered evaluations\n\n"
            "Recovered from `exports/dumped_evals` without modifying the source dumps.\n\n"
            "Each world directory is named `<skin>__seed_<seed>`. `rollout_0` and "
            "`rollout_1` are the two eval samples in their original dump order.\n\n"
            "- `*.raw.json`: original dump record\n"
            "- `*.response.txt`: raw decoded model response\n"
            "- `*.transcript.txt`: initial prompt plus the complete multi-turn response\n"
            "- `index.jsonl`: searchable metadata and relative paths for every rollout\n"
            "- `manifest.json`: counts and integrity-oriented summary\n\n"
            "A world is identified by `(archetype, skin, seed)`.\n",
            encoding="utf-8",
        )

        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=_steps, default=_steps("0,8"))
    parser.add_argument("--output-name", default="recover_evals")
    args = parser.parse_args()
    destination = recover(args.run_dir.resolve(), args.steps, args.output_name)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
