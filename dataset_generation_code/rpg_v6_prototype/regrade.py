#!/usr/bin/env python3
"""Re-grade existing result_*.json files with the current grader.

Grading logic evolved (fraction-of-benefit part A; parenthetical-stripping proxy
resolution). Results produced by an older grader can be re-scored offline WITHOUT
re-running the agent: we re-resolve the stored final answer and recompute the
grade from the world's stored oracle. Writes the new grade back into each result
file (preserving the full turn log), and prints a before/after summary.

Usage:
    python regrade.py --results-dir results_v6/batch20_opus --worlds-dir out_v6_batch20
    # add --write to persist; default is dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict

from run_agent_v6 import load_world_file, _translate_structured, _resolve_answer_intervention
from sim_v6 import SimV6
from oracle_v6 import grade as grade_fn
from oracle_v6 import optimal_intervention, counterfactual_battery


def find_world_file(worlds_dir: str, world_id: str) -> str:
    for p in glob.glob(os.path.join(worlds_dir, "world_*.json")):
        if world_id in p:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--write", action="store_true", help="persist re-graded results (default dry-run)")
    ap.add_argument("--fresh-oracle", action="store_true",
                    help="recompute gold + counterfactual battery from the world "
                         "(picks up oracle/battery fixes; slower). Default reuses stored.")
    args = ap.parse_args()

    flips = 0
    old_acc = new_acc = n = 0
    for rf in sorted(glob.glob(os.path.join(args.results_dir, "result_*.json"))):
        r = json.load(open(rf))
        wid = r["world_id"]
        wf = find_world_file(args.worlds_dir, wid)
        if not wf:
            print(f"  [skip] no world file for {wid}")
            continue
        world, pre = load_world_file(wf)
        if args.fresh_oracle:
            pre = {"gold": optimal_intervention(world["scm"]),
                   "battery": counterfactual_battery(world)}
        sim = SimV6(world, precomputed=pre)
        # find the final answer turn
        ar = None
        for t in r.get("turns", []):
            if t.get("answer_raw"):
                ar = t["answer_raw"]
        if not ar:
            continue
        iv, echoes = _resolve_answer_intervention(sim, ar.get("recommended_intervention_text", []))
        ans = {"recommended_intervention": iv,
               "structured": _translate_structured(sim, ar.get("structured", {})),
               "explanation": ar.get("explanation", "")}
        new = grade_fn(world, ans, sim.gold, sim.battery)
        old = r.get("grade") or {}
        n += 1
        old_acc += bool(old.get("accepted"))
        new_acc += bool(new["accepted"])
        if bool(old.get("accepted")) != new["accepted"]:
            flips += 1
            print(f"  FLIP {wid}: {old.get('accepted')} -> {new['accepted']} "
                  f"(benefit={new['benefit_recovered']:.2f}, B={new['battery_fraction']:.2f})")
        if args.write:
            r["grade"] = new
            r["regraded"] = True
            with open(rf, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, default=str)

    print(f"\n{args.results_dir}: {n} worlds | accepted {old_acc} -> {new_acc} | flips {flips} | "
          f"{'WRITTEN' if args.write else 'dry-run (use --write to persist)'}")


if __name__ == "__main__":
    main()
