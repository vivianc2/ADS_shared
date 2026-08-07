#!/usr/bin/env python3
"""Aggregate RPG v6 batch results into a statistically-informative report.

Reads one or more batch output dirs (each containing result_*.json + summary.json)
and produces a per-(model × topology) table with accept rate, part-A rate, mean
part-B, and Wilson 95% confidence intervals, plus the dominant failure mode per
cell. Supports multiple models for side-by-side comparison.

Usage:
    python analyze_results.py \
        --run opus4.8=results_v6/batch_opus \
        --run qwen3.6=results_v6/batch_qwen \
        --out results_v6/report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOPOLOGY_RE = re.compile(r"(bioreactor_titer_loss|datacenter_throughput|greenhouse_yield|clinic_readmission)")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def topology_of(world_id: str) -> str:
    m = TOPOLOGY_RE.search(world_id)
    return m.group(1) if m else "unknown"


def load_run(run_dir: str) -> List[Dict[str, Any]]:
    """Load per-world result rows from a batch dir."""
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "result_*.json"))):
        r = json.load(open(p))
        g = r.get("grade") or {}
        art = {}
        for t in r.get("turns", []):
            if t.get("artifact_check"):
                art = t["artifact_check"]
        rows.append({
            "world_id": r.get("world_id", ""),
            "topology": topology_of(r.get("world_id", "")),
            "model": r.get("model", "?"),
            "accepted": bool(g.get("accepted")),
            "partA": bool(g.get("part_a_utility_ok")),
            "partB": float(g.get("battery_fraction") or 0.0),
            "gap": float(g.get("utility_gap") or 0.0),
            "artifact_suspect": bool(art.get("suspect")),
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    k = sum(r["accepted"] for r in rows)
    ka = sum(r["partA"] for r in rows)
    p, lo, hi = wilson(k, n)
    return {"n": n, "accepted": k, "acc": round(p, 3), "acc_lo": round(lo, 3),
            "acc_hi": round(hi, 3), "partA": ka, "partA_rate": round(ka / n, 3) if n else 0,
            "mean_partB": round(sum(r["partB"] for r in rows) / n, 3) if n else 0,
            "artifact_suspects": sum(r["artifact_suspect"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=dir, e.g. opus4.8=results_v6/batch_opus (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs: Dict[str, List[Dict[str, Any]]] = {}
    for spec in args.run:
        label, d = spec.split("=", 1)
        runs[label] = load_run(d)

    topologies = ["bioreactor_titer_loss", "datacenter_throughput",
                  "greenhouse_yield", "clinic_readmission"]

    lines: List[str] = ["# RPG v6 — Results\n"]

    # ---- overall per model ----
    lines.append("## Overall (all topologies pooled)\n")
    lines.append("| Model | n | Accepted | Accuracy [95% CI] | Part-A rate | Mean Part-B | Artifact-flagged |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, rows in runs.items():
        s = summarize(rows)
        lines.append(f"| {label} | {s['n']} | {s['accepted']} | {s['acc']:.2f} "
                     f"[{s['acc_lo']:.2f}, {s['acc_hi']:.2f}] | {s['partA_rate']:.2f} | "
                     f"{s['mean_partB']:.2f} | {s['artifact_suspects']} |")
    lines.append("")

    # ---- per topology × model ----
    lines.append("## By topology (accept rate [95% CI], n)\n")
    header = "| Topology | " + " | ".join(runs.keys()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(runs) + 1))
    for topo in topologies:
        cells = []
        for label, rows in runs.items():
            sub = [r for r in rows if r["topology"] == topo]
            s = summarize(sub)
            cells.append(f"{s['accepted']}/{s['n']} ({s['acc']:.2f}) [{s['acc_lo']:.2f},{s['acc_hi']:.2f}]" if s["n"] else "—")
        lines.append(f"| {topo} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- part-A vs part-B decomposition per topology (per model) ----
    lines.append("## Decomposition: part-A (found the fix) vs part-B (understood structure)\n")
    for label, rows in runs.items():
        lines.append(f"### {label}\n")
        lines.append("| Topology | n | Accept | Part-A rate | Mean Part-B |")
        lines.append("|---|---|---|---|---|")
        for topo in topologies:
            sub = [r for r in rows if r["topology"] == topo]
            s = summarize(sub)
            if s["n"]:
                lines.append(f"| {topo} | {s['n']} | {s['accepted']}/{s['n']} | "
                             f"{s['partA_rate']:.2f} | {s['mean_partB']:.2f} |")
        lines.append("")

    # ---- artifact caveat ----
    total_art = sum(summarize(rows)["artifact_suspects"] for rows in runs.values())
    lines.append("## Notes\n")
    lines.append(f"- Accept = part-A (utility within tolerance of computed optimum) AND "
                 f"part-B (≥0.8 of structure items correct).")
    lines.append(f"- CIs are Wilson 95% intervals; wide at small n.")
    lines.append(f"- Artifact-flagged results across all runs: **{total_art}** — these "
                 f"were auto-flagged as possible harness/resolution issues and require "
                 f"inspection before being read as reasoning failures.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
