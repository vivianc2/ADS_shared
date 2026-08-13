#!/usr/bin/env python3
"""Offline re-grade of a finished batch — NO new API spend.

Reads each ``result_<wid>.json`` in a results dir, reloads the world it points to,
re-resolves the stored final ``answer_raw`` through the CURRENT resolver/grader
code (so resolver + battery fixes apply retroactively), recomputes part A / part B,
and prints an old-vs-new comparison. This is how we tell a genuine reasoning result
apart from a grading/resolution artifact without re-running the (expensive) agent.

By default the resolver LLM fallback is OFF (pure lexical, fully offline). Pass
--resolver-llm to additionally use the Bedrock/Opus resolver on the answer terms
(needs AWS_BEARER_TOKEN_BEDROCK) — closer to how a live run resolves, at the cost
of a few resolver calls per world.

Usage:
    python regrade_v7.py --results-dir results_v7/mixed9_opus
    python regrade_v7.py --results-dir results_v7/mixed9_opus --write   # persist regrade_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from sim_v6 import SimV6
from run_agent_v6 import (load_world_file, _resolve_answer_intervention,
                          _resolve_answer_policy, _translate_structured,
                          build_resolver_llm)


def _regrade_one(result: Dict[str, Any], resolver_llm) -> Optional[Dict[str, Any]]:
    """Re-resolve + re-grade one stored result with current code. Returns a compact
    comparison row, or None if the world file / answer can't be recovered."""
    wf = result.get("world_file")
    if not wf or not os.path.exists(wf):
        return None
    world, pre = load_world_file(wf)
    sim = SimV6(world, resolver_llm=resolver_llm, precomputed=pre)

    # recover the agent's FINAL answer as it was emitted (free text)
    answer_raw = None
    for t in result.get("turns", []):
        if t.get("action_type") == "answer" and t.get("answer_raw"):
            answer_raw = t["answer_raw"]
    old = result.get("grade", {}) or {}
    if answer_raw is None:
        # no parsable answer turn (e.g. forced/no-answer) — carry the old grade
        return {"world_id": result.get("world_id"), "no_answer": True,
                "old_accept": bool(old.get("accepted")),
                "new_accept": bool(old.get("accepted")),
                "old_A": bool(old.get("part_a_utility_ok")), "new_A": bool(old.get("part_a_utility_ok")),
                "old_B": round(float(old.get("battery_fraction") or 0), 2),
                "new_B": round(float(old.get("battery_fraction") or 0), 2)}

    # re-resolve with current code
    iv, _echoes = _resolve_answer_intervention(sim, answer_raw.get("recommended_intervention_text", []))
    pol = _resolve_answer_policy(sim, answer_raw.get("recommended_policy_text"))
    answer = {"recommended_intervention": iv,
              "structured": _translate_structured(sim, answer_raw.get("structured", {})),
              "explanation": answer_raw.get("explanation", "")}
    if pol and not pol.get("_unresolved"):
        answer["recommended_policy"] = pol
    new = sim.grade(answer)
    return {
        "world_id": result.get("world_id"),
        "archetype": (world.get("ground_truth") or {}).get("_archetype"),
        "old_accept": bool(old.get("accepted")), "new_accept": bool(new["accepted"]),
        "old_A": bool(old.get("part_a_utility_ok")), "new_A": bool(new["part_a_utility_ok"]),
        "old_B": round(float(old.get("battery_fraction") or 0), 2),
        "new_B": round(float(new["battery_fraction"]), 2),
        "benefit": new.get("benefit_recovered"),
        "resolved_proxy": answer["structured"].get("true_mechanism_proxy"),
        "new_grade": new,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--resolver-llm", action="store_true",
                    help="also use the Bedrock/Opus resolver on answer terms (needs creds)")
    ap.add_argument("--write", action="store_true",
                    help="persist per-world regrade_<wid>.json + regrade_summary.json")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    paths = sorted(glob.glob(str(rdir / "result_*.json")))
    if not paths:
        ap.error(f"no result_*.json in {rdir}")

    # resolver LLM: reuse the same builder the runner uses (fixed strong model).
    # "bedrock" backend + not-opted-out => Bedrock Opus resolver if creds present.
    resolver_llm = None
    if args.resolver_llm:
        resolver_llm = build_resolver_llm("bedrock", None, no_resolver_llm=False)
        if resolver_llm is None:
            print("WARNING: --resolver-llm requested but no resolver LLM built "
                  "(AWS_BEARER_TOKEN_BEDROCK not set?). Falling back to lexical.",
                  flush=True)

    print(f"re-grading {len(paths)} world(s) in {rdir} "
          f"(resolver_llm={'ON — expect a few Bedrock calls per world' if resolver_llm else 'off/lexical'})",
          flush=True)

    rows = []
    for i, p in enumerate(paths, 1):
        with open(p, encoding="utf-8") as f:
            result = json.load(f)
        wid = result.get("world_id", Path(p).stem)
        print(f"  [{i}/{len(paths)}] {wid} ...", end="", flush=True)
        t0 = time.time()
        row = _regrade_one(result, resolver_llm)
        if row is None:
            print(" skip (world file not found)", flush=True)
            continue
        rows.append(row)
        if args.write and "new_grade" in row:
            with open(rdir / f"regrade_{row['world_id']}.json", "w", encoding="utf-8") as f:
                json.dump(row["new_grade"], f, indent=2, default=str)
        print(f" B {row['old_B']:.2f}->{row['new_B']:.2f}  "
              f"acc {row['old_accept']}->{row['new_accept']}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    old_acc = sum(r["old_accept"] for r in rows)
    new_acc = sum(r["new_accept"] for r in rows)
    flipped = [r for r in rows if r["new_accept"] and not r["old_accept"]]

    print(f"\n=== RE-GRADE ({rdir}) — resolver_llm={'on' if resolver_llm else 'off (lexical)'} ===")
    print(f"accepted: {old_acc}/{n}  ->  {new_acc}/{n}   (+{len(flipped)} flipped to accept)\n")
    print(f"{'world':46s} {'arch':10s} {'A':>3s} {'B(old->new)':>13s} {'acc':>10s}")
    for r in rows:
        arch = (r.get("archetype") or "")[:10]
        acc = f"{r['old_accept']!s:>5}->{r['new_accept']!s:<4}"
        aa = ("Y" if r["new_A"] else ".")
        print(f"{r['world_id'][:46]:46s} {arch:10s} {aa:>3s} {r['old_B']:>5.2f}->{r['new_B']:<5.2f} {acc:>10s}"
              + ("  <== FLIP" if (r['new_accept'] and not r['old_accept']) else ""))

    if args.write:
        summary = {"results_dir": str(rdir), "n": n, "old_accepted": old_acc,
                   "new_accepted": new_acc, "flipped_to_accept": [r["world_id"] for r in flipped],
                   "resolver_llm": bool(resolver_llm), "rows": rows}
        with open(rdir / "regrade_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nwrote regrade_summary.json (+ per-world regrade_*.json) to {rdir}/")


if __name__ == "__main__":
    main()
