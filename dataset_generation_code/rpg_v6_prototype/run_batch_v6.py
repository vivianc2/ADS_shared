#!/usr/bin/env python3
"""Run the LLM scientist over a generated batch of RPG v6 worlds.

Runs each world_*.json in a directory through run_agent_v6.run_world, writes a
per-world result JSON, and prints an aggregate summary + a per-world table that
makes the expert-vs-agent gap legible (accepted / partA utility / partB battery /
interventions run / whether it hit the turn cap).

Usage:
    python run_batch_v6.py --worlds-dir out_v6_batch \
        --backend bedrock --model us.anthropic.claude-opus-4-8 \
        --outdir results_v6/batch1 -v
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from sim_v6 import SimV6
from run_agent_v6 import build_llm, build_resolver_llm, run_world, load_world_file

logger = logging.getLogger("run_batch_v6")


def summarize_result(res: Dict[str, Any], wid: str, world: Dict[str, Any]) -> Dict[str, Any]:
    """One summary row from a result dict — whether it was just produced by
    run_world or reloaded from disk by --resume."""
    g = res.get("grade") or {}
    # artifact flag from the answer turn (if any)
    art = {}
    for t in res.get("turns", []):
        if t.get("artifact_check"):
            art = t["artifact_check"]
    return {
        "world_id": wid, "template": world.get("domain"),
        "accepted": bool(g.get("accepted")),
        "partA": bool(g.get("part_a_utility_ok")),
        "partB": round(float(g.get("battery_fraction") or 0), 2),
        "gap": round(float(g.get("utility_gap") or 0), 1),
        "interventions": res.get("interventions_run"),
        "queries": res.get("queries_used"),
        "hit_cap": bool(res.get("hit_turn_cap")),
        "artifact_suspect": bool(art.get("suspect")),
        "artifact_reasons": art.get("reasons", []),
        "wall_s": res.get("wall_seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--backend", choices=["bedrock", "nautilus", "openai", "mock"], default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; default None = use the model's preset")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="per-turn output token cap; default None = use the model's MAX output "
                         "(32768 for Qwen/gpt-oss). Set lower only to save cost.")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=60,
                    help="max productive turns; raised 32->60 so the cap does not bind")
    ap.add_argument("--no-resolver-llm", action="store_true",
                    help="disable the LLM resolver fallback (on by default)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip worlds that already have a result_<world>.json in "
                         "--outdir; summary.json is still rebuilt over all worlds")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    paths = sorted(glob.glob(os.path.join(args.worlds_dir, "world_*.json")))
    if not paths:
        ap.error(f"no world_*.json in {args.worlds_dir}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    # Resolver uses a FIXED strong model, independent of the agent model, so
    # resolution quality is constant across agent models.
    resolver_llm = build_resolver_llm(args.backend, llm, args.no_resolver_llm)

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    t_start = time.time()
    for path in paths:
        world, pre = load_world_file(path)
        wid = world["world_id"]
        result_path = outdir / f"result_{wid}.json"

        # --resume: a world that already has a readable result JSON is done.
        # Reuse it verbatim so an interrupted batch can be finished without
        # re-spending API calls, and summary.json still covers all 20 worlds.
        if args.resume and result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("=== %s: unreadable result (%s) — re-running ===", wid, e)
            else:
                logger.info("=== %s: skipping, result exists ===", wid)
                rows.append(summarize_result(res, wid, world))
                n_skipped += 1
                continue

        data_dir = str(outdir / f"{wid}_data")
        sim = SimV6(world, resolver_llm=resolver_llm, data_dir=data_dir, precomputed=pre)
        logger.info("=== %s (gold util %.1f) ===", wid, sim.gold["expected_utility"])
        t0 = time.time()
        res = run_world(sim, llm, args.max_turns, args.budget, args.verbose, args.max_new_tokens)
        res["wall_seconds"] = round(time.time() - t0, 1)
        res["model"] = args.model
        res["world_file"] = path
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        rows.append(summarize_result(res, wid, world))

    n = len(rows)
    acc = sum(r["accepted"] for r in rows)
    pa = sum(r["partA"] for r in rows)
    caps = sum(r["hit_cap"] for r in rows)
    no_iv = sum(1 for r in rows if (r["interventions"] or 0) == 0)
    suspects = [r for r in rows if r["artifact_suspect"]]
    summary = {
        "model": args.model, "n_worlds": n, "accepted": acc,
        "accuracy": round(acc / n, 3) if n else 0,
        "partA_utility_ok": pa, "avg_partB": round(sum(r["partB"] for r in rows) / n, 3) if n else 0,
        "hit_turn_cap": caps, "no_intervention_runs": no_iv,
        "artifact_suspects": len(suspects),
        "reused_results": n_skipped,
        # wall_seconds covers THIS invocation only; on a resumed batch the
        # reused worlds contribute ~0. Per-world timings live in rows[].wall_s.
        "wall_seconds": round(time.time() - t_start, 1), "rows": rows,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BATCH SUMMARY ({args.model}) ===")
    print(f"accepted {acc}/{n} (acc={summary['accuracy']}) | partA {pa}/{n} | "
          f"avg partB {summary['avg_partB']} | hit_cap {caps} | no-intervention {no_iv} | "
          f"{summary['wall_seconds']}s"
          + (f" (+{n_skipped} reused)" if n_skipped else ""))
    print(f"{'world':46s} {'acc':4s} {'A':2s} {'B':5s} {'gap':6s} {'iv':3s} {'q':3s} {'cap':4s} {'art'}")
    for r in rows:
        print(f"{r['world_id']:46s} {str(r['accepted'])[:4]:4s} "
              f"{('Y' if r['partA'] else '.'):2s} {r['partB']:<5.2f} {r['gap']:<6.1f} "
              f"{str(r['interventions']):3s} {str(r['queries']):3s} {('Y' if r['hit_cap'] else '.'):4s} "
              f"{('!' if r['artifact_suspect'] else '.')}")
    # Loud warning: a suspect run means a FAILURE might be a harness artifact, not
    # difficulty. This is the guard against reading a false 0/N as 'too hard'.
    if suspects:
        print(f"\n⚠️  {len(suspects)} world(s) flagged as ARTIFACT-SUSPECT — inspect before "
              f"interpreting these as reasoning failures:")
        for r in suspects:
            print(f"   - {r['world_id']}: {'; '.join(r['artifact_reasons'][:2])}")
    print(f"\nwrote per-world results + summary.json to {outdir}/")


if __name__ == "__main__":
    main()
