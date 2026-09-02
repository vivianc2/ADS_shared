#!/usr/bin/env python3
"""Compare the step-0 / step-4 / step-8 evaluations of the three prompt runs.

Reads what SkyRL already dumps -- ``<export_path>/dumped_evals/global_step_<N>_evals/``
-- so nothing has to be recomputed and the report cannot disagree with W&B.

For each run and each evaluated step it reports the five required numbers:

    avg score       mean trajectory reward over the validation episodes
    part_a          mean "found the fix" credit
    part_b          mean "understood the mechanism" credit
    truncation      fraction of episodes that never reached a terminal answer
    mean turns      mean number of environment turns per episode

``avg score`` and ``truncation`` are cross-checked against the per-episode records
(``rpg_v7.jsonl``: ``score`` and ``stop_reason``) so a mismatch between the aggregated
metrics and the raw episodes is surfaced rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from prompt_compare_rl.config import EVAL_STEPS, PROMPT_IDS, ExperimentConfig  # noqa: E402

METRIC_COLUMNS = [
    ("avg_score", "eval/all/avg_score"),
    ("part_a", "eval/all/environment/part_a"),
    ("part_b", "eval/all/environment/part_b"),
    ("truncation", "eval/all/environment/truncated"),
    ("mean_turns", "eval/all/environment/turns"),
]


def _eval_dir(cfg: ExperimentConfig, prompt_id: str, step: int) -> Path:
    return cfg.run_paths(prompt_id)["export_path"] / "dumped_evals" / f"global_step_{step}_evals"


def _read_aggregated(path: Path) -> Optional[Dict[str, Any]]:
    target = path / "aggregated_results.jsonl"
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _read_episodes(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for jsonl in sorted(path.glob("*.jsonl")):
        if jsonl.name == "aggregated_results.jsonl":
            continue
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def collect(cfg: ExperimentConfig, prompt_ids: List[str], steps: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"exp_tag": cfg.exp_tag, "steps": steps, "runs": {}}
    for prompt_id in prompt_ids:
        run: Dict[str, Any] = {
            "wandb_run_id": cfg.wandb_run_id(prompt_id),
            "export_path": str(cfg.run_paths(prompt_id)["export_path"]),
            "by_step": {},
        }
        for step in steps:
            directory = _eval_dir(cfg, prompt_id, step)
            aggregated = _read_aggregated(directory)
            if aggregated is None:
                run["by_step"][step] = {"status": "missing", "path": str(directory)}
                continue
            episodes = _read_episodes(directory)
            row: Dict[str, Any] = {"status": "ok", "path": str(directory), "episodes": len(episodes)}
            for name, key in METRIC_COLUMNS:
                value = aggregated.get(key)
                row[name] = float(value) if isinstance(value, (int, float)) else None
            if episodes:
                row["avg_score_from_episodes"] = _mean(
                    [float(e["score"]) for e in episodes if isinstance(e.get("score"), (int, float))]
                )
                row["truncation_from_stop_reason"] = _mean(
                    [0.0 if e.get("stop_reason") == "stop" else 1.0 for e in episodes]
                )
            row["per_archetype"] = {
                key.split("/")[-2]: aggregated[key]
                for key in aggregated
                if key.startswith("eval/all/environment/archetype/") and key.endswith("/score")
            }
            run["by_step"][step] = row
        out["runs"][prompt_id] = run
    return out


def _fmt(value: Optional[float], width: int = 10, digits: int = 4) -> str:
    return "".rjust(width, " ") if value is None else f"{value:.{digits}f}".rjust(width)


def render_table(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    header = (
        f"{'prompt':<8}{'step':>6}{'episodes':>10}"
        + "".join(name.rjust(12) for name, _ in METRIC_COLUMNS)
    )
    lines.append(f"experiment: {report['exp_tag']}")
    lines.append(header)
    lines.append("-" * len(header))
    for prompt_id, run in report["runs"].items():
        for step in report["steps"]:
            row = run["by_step"].get(step, {"status": "missing"})
            if row.get("status") != "ok":
                lines.append(f"{prompt_id:<8}{step:>6}{'--':>10}   (no eval dump yet: {row.get('path','?')})")
                continue
            cells = "".join(_fmt(row.get(name), 12) for name, _ in METRIC_COLUMNS)
            lines.append(f"{prompt_id:<8}{step:>6}{row['episodes']:>10}{cells}")
        lines.append("")
    lines.append("avg_score / part_a / part_b: higher is better. truncation: lower is better.")
    lines.append("Episodes that never reached a terminal answer contribute 0 to score/part_a/part_b,")
    lines.append("matching the reward the policy actually received for them.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-ids", default=",".join(PROMPT_IDS))
    parser.add_argument("--steps", default=",".join(str(s) for s in EVAL_STEPS))
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit the raw report as JSON")
    parser.add_argument("--out", default=None, help="also write the JSON report to this path")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig()
    prompt_ids = [p.strip() for p in args.prompt_ids.split(",") if p.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    report = collect(cfg, prompt_ids, steps)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2) if args.as_json else render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
