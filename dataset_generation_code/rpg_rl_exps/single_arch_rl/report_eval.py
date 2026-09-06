#!/usr/bin/env python3
"""Per-archetype evaluation table for the two single-archetype runs.

Reads what SkyRL already dumps -- ``<export_path>/dumped_evals/global_step_<N>_evals/`` --
so nothing is recomputed and the report cannot disagree with W&B.

Log requirement (2): for every run and every evaluated step, the **avg score, part_a and
part_b per archetype** on the 45-world held-out validation set, plus the pooled row.

    run   step  archetype              n   avg_score   part_a   part_b  trunc  turns
    easy     0  ALL                   90      0.1234   0.2000   0.0500  0.311    9.4
    easy     0  collider_selection    10      0.0900   0.1800   0.0000  0.400    8.1
    ...

* **avg_score** -- mean trajectory reward (``0.5*part_a + 0.5*part_b - 0.25*invalid_ids``).
* **part_a** -- "found the fix" credit. **part_b** -- "understood the mechanism" credit.
* **trunc** -- fraction of episodes that never reached a terminal answer.

part_a / part_b / trunc / turns come from the aggregated metrics that ``sky_env.py``
contributes (``eval/all/environment/archetype/<a>/<key>``). avg_score is cross-checked
against the per-episode records in the archetype's own ``rpg_v9_eval_<a>.jsonl`` dump, so
a disagreement is surfaced rather than hidden.

    bash scripts/in_container.sh python -m single_arch_rl.report_eval
    bash scripts/in_container.sh python -m single_arch_rl.report_eval --json
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

from single_arch_rl.config import RUN_IDS, ExperimentConfig  # noqa: E402

#: Metric suffixes reported per archetype, in column order.
COLUMNS = ("avg_score", "part_a", "part_b", "truncated", "turns")


def _eval_dirs(cfg: ExperimentConfig, run_id: str) -> Dict[int, Path]:
    root = cfg.run_paths(run_id)["export_path"] / "dumped_evals"
    found: Dict[int, Path] = {}
    if not root.is_dir():
        return found
    for path in root.glob("global_step_*_evals"):
        try:
            found[int(path.name.split("_")[2])] = path
        except (IndexError, ValueError):
            continue
    return dict(sorted(found.items()))


def _aggregated(path: Path) -> Optional[Dict[str, Any]]:
    target = path / "aggregated_results.jsonl"
    if not target.exists():
        return None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return None


def _episode_scores(path: Path) -> Dict[str, List[float]]:
    """Per-archetype trajectory scores, read back from the per-episode dumps."""
    scores: Dict[str, List[float]] = {}
    for jsonl in sorted(path.glob("*.jsonl")):
        if jsonl.name == "aggregated_results.jsonl":
            continue
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            extras = record.get("env_extras") or {}
            info = extras.get("extra_info", extras) or {}
            archetype = str(info.get("archetype", "unknown"))
            value = record.get("score")
            if isinstance(value, list):
                value = sum(value)
            scores.setdefault(archetype, []).append(float(value or 0.0))
    return scores


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def collect(cfg: ExperimentConfig, run_ids: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"exp_tag": cfg.exp_tag, "runs": {}}
    for run_id in run_ids:
        run: Dict[str, Any] = {
            "archetype": cfg.archetype_for(run_id),
            "wandb_run_id": cfg.wandb_run_id(run_id),
            "steps": {},
        }
        for step, directory in _eval_dirs(cfg, run_id).items():
            aggregated = _aggregated(directory) or {}
            episodes = _episode_scores(directory)
            rows: Dict[str, Dict[str, Any]] = {}
            archetypes = sorted(
                {key.split("/")[3] for key in aggregated
                 if key.startswith("eval/all/environment/archetype/")}
                | set(episodes)
            )
            for archetype in archetypes:
                prefix = f"eval/all/environment/archetype/{archetype}/"
                rows[archetype] = {
                    "episodes": int(aggregated.get(prefix + "episodes") or len(episodes.get(archetype, []))),
                    "avg_score": aggregated.get(prefix + "score"),
                    "part_a": aggregated.get(prefix + "part_a"),
                    "part_b": aggregated.get(prefix + "part_b"),
                    "truncated": aggregated.get(prefix + "truncated"),
                    "turns": aggregated.get(prefix + "turns"),
                    # Independent recomputation from the raw per-episode dump.
                    "avg_score_from_episodes": _mean(episodes.get(archetype, [])),
                }
            all_scores = [s for values in episodes.values() for s in values]
            rows["ALL"] = {
                "episodes": len(all_scores),
                "avg_score": aggregated.get("eval/all/avg_score"),
                "part_a": aggregated.get("eval/all/environment/part_a"),
                "part_b": aggregated.get("eval/all/environment/part_b"),
                "truncated": aggregated.get("eval/all/environment/truncated"),
                "turns": aggregated.get("eval/all/environment/turns"),
                "avg_score_from_episodes": _mean(all_scores),
            }
            run["steps"][step] = rows
        out["runs"][run_id] = run
    return out


def _fmt(value: Optional[float], width: int = 9, places: int = 4) -> str:
    return " " * (width - 1) + "-" if value is None else f"{value:{width}.{places}f}"


def render(report: Dict[str, Any]) -> str:
    lines = [
        f"{'run':<6}{'step':>5}  {'archetype':<20}{'n':>5}{'avg_score':>10}{'part_a':>9}"
        f"{'part_b':>9}{'trunc':>8}{'turns':>8}  {'check':>8}"
    ]
    for run_id, run in report["runs"].items():
        if not run["steps"]:
            lines.append(f"{run_id:<6}    -  (no evaluations dumped yet)")
            continue
        for step, rows in run["steps"].items():
            for name in ["ALL"] + [k for k in sorted(rows) if k != "ALL"]:
                row = rows[name]
                recomputed = row.get("avg_score_from_episodes")
                reported = row.get("avg_score")
                mismatch = (
                    "" if recomputed is None or reported is None
                    else ("" if abs(recomputed - reported) < 1e-6 else f"{recomputed:8.4f}")
                )
                lines.append(
                    f"{run_id:<6}{step:>5}  {name:<20}{row['episodes']:>5}"
                    f"{_fmt(row['avg_score'], 10)}{_fmt(row['part_a'])}{_fmt(row['part_b'])}"
                    f"{_fmt(row['truncated'], 8, 3)}{_fmt(row['turns'], 8, 2)}  {mismatch:>8}"
                )
            lines.append("")
    lines.append(
        "'check' is blank when avg_score recomputed from the per-episode dumps matches the "
        "aggregated value; a number there is the recomputed value."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", default=",".join(RUN_IDS))
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    args = parser.parse_args(argv)

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    unknown = [r for r in run_ids if r not in RUN_IDS]
    if unknown:
        raise SystemExit(f"unknown run ids: {unknown}")

    report = collect(ExperimentConfig(), run_ids)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
