#!/usr/bin/env python3
"""Run an agent on every world in a folder; write per-world results and summary.json."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from simulator import Simulator
from agent_loop import build_llm, build_resolver_llm, run_world, load_world_file

logger = logging.getLogger("run_benchmark")


def summarize_result(res: Dict[str, Any], world: Dict[str, Any]) -> Dict[str, Any]:
    g = res.get("grade") or {}
    art = {}
    for t in res.get("turns", []):
        if t.get("artifact_check"):
            art = t["artifact_check"]
    n_length = sum(1 for t in res.get("turns", []) if t.get("finish_reason") == "length")
    return {
        "world_id": world["world_id"], "template": world.get("domain"),
        "archetype": (world.get("ground_truth") or {}).get("_archetype"),
        "accepted": bool(g.get("accepted")),
        "partA": bool(g.get("part_a_utility_ok")),
        "partB": round(float(g.get("battery_fraction") or 0), 2),
        "gap": round(float(g.get("utility_gap") or 0), 1),
        "interventions": res.get("interventions_run"),
        "queries": res.get("queries_used"),
        "hit_cap": bool(res.get("hit_turn_cap")),
        "n_length_truncated_turns": n_length,
        "artifact_suspect": bool(art.get("suspect")),
        "artifact_reasons": art.get("reasons", []),
        "wall_s": res.get("wall_seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--backend", choices=["bedrock", "openai", "vllm", "mock"], default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; default None = use the model's preset")
    ap.add_argument("--max-new-tokens", type=int, default=16000)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="number of worlds to run in parallel (threads). For a self-hosted "
                         "vLLM server, set 4-16 to exploit continuous batching — a single "
                         "GPU decoding one request at a time is why serial is slow. Ignored "
                         "for --backend mock. Match to your server's --max-num-seqs.")
    ap.add_argument("--resolver-model", default="us.anthropic.claude-opus-4-8",
                    help="Bedrock model id of the request resolver (fixed across agent models)")
    ap.add_argument("--no-resolver-llm", action="store_true",
                    help="disable the LLM resolver fallback (on by default)")
    ap.add_argument("--system-prompt-file", default=None,
                    help="replace the agent system prompt with this file's text")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip worlds that already have a result_<wid>.json in outdir "
                         "(pick up an interrupted/stopped batch without re-spending API)")
    ap.add_argument("--force", action="store_true",
                    help="re-run every world even if a result exists (opposite of --resume)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    paths = sorted(glob.glob(os.path.join(args.worlds_dir, "world_*.json*")))
    if not paths:
        ap.error(f"no world_*.json(.gz) in {args.worlds_dir}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _wid_of(p):
        try:
            return load_world_file(p)[0]["world_id"]
        except Exception:
            return Path(p).name.split(".")[0].replace("world_", "")

    todo, done_paths = [], []
    for path in paths:
        wid = _wid_of(path)
        result_file = outdir / f"result_{wid}.json"
        if args.resume and not args.force and result_file.exists():
            done_paths.append(path)
        else:
            todo.append(path)
    if done_paths:
        logger.info("resume: %d world(s) already done, %d to run", len(done_paths), len(todo))
        print(f"resume: skipping {len(done_paths)} completed world(s), running {len(todo)}")

    llm = resolver_llm = None
    if todo:
        llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
        resolver_llm = build_resolver_llm(args.backend, llm, args.no_resolver_llm, args.resolver_model)

    sysp = Path(args.system_prompt_file).read_text() if args.system_prompt_file else None

    def _run_one(path: str) -> Dict[str, Any]:
        world, pre = load_world_file(path)
        wid = world["world_id"]
        result_file = outdir / f"result_{wid}.json"
        if path in done_paths:
            try:
                with open(result_file, encoding="utf-8") as f:
                    res = json.load(f)
                return summarize_result(res, world)
            except Exception as e:
                logger.warning("resume: could not read %s (%s); re-running", result_file, e)
        data_dir = str(outdir / f"{wid}_data")
        if (world.get("ground_truth") or {}).get("_world_version") == "v10":
            import hashlib
            data_dir = str(outdir / f"data_{hashlib.sha1(wid.encode()).hexdigest()[:10]}")
        sim = Simulator(world, resolver_llm=resolver_llm, data_dir=data_dir, precomputed=pre)
        logger.info("=== %s (gold util %.1f) ===", wid, sim.gold["expected_utility"])
        t0 = time.time()
        res = run_world(sim, llm, args.max_turns, args.budget, args.verbose, system_prompt=sysp)
        res["wall_seconds"] = round(time.time() - t0, 1)
        res["model"] = args.model
        res["world_file"] = path
        res["data_dir"] = data_dir
        res["system_prompt_file"] = args.system_prompt_file
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        return summarize_result(res, world)

    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    conc = 1 if args.backend == "mock" else max(1, args.concurrency)
    if conc > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"running {len(paths)} world(s) with concurrency={conc}", flush=True)
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_run_one, p): p for p in paths}
            done_n = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done_n += 1
                print(f"  [{done_n}/{len(paths)}] {row['world_id']} "
                      f"acc={row['accepted']} A={row['partA']} B={row['partB']:.2f}", flush=True)
    else:
        for path in paths:
            rows.append(_run_one(path))

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
        "wall_seconds": round(time.time() - t_start, 1), "rows": rows,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BATCH SUMMARY ({args.model}) ===")
    print(f"accepted {acc}/{n} (acc={summary['accuracy']}) | partA {pa}/{n} | "
          f"avg partB {summary['avg_partB']} | hit_cap {caps} | no-intervention {no_iv} | "
          f"{summary['wall_seconds']}s")
    print(f"{'world':46s} {'acc':4s} {'A':2s} {'B':5s} {'gap':6s} {'iv':3s} {'q':3s} {'cap':4s} {'art'}")
    for r in rows:
        print(f"{r['world_id']:46s} {str(r['accepted'])[:4]:4s} "
              f"{('Y' if r['partA'] else '.'):2s} {r['partB']:<5.2f} {r['gap']:<6.1f} "
              f"{str(r['interventions']):3s} {str(r['queries']):3s} {('Y' if r['hit_cap'] else '.'):4s} "
              f"{('!' if r['artifact_suspect'] else '.')}")
    if suspects:
        print(f"\n⚠️  {len(suspects)} world(s) flagged as ARTIFACT-SUSPECT — inspect before "
              f"interpreting these as reasoning failures:")
        for r in suspects:
            print(f"   - {r['world_id']}: {'; '.join(r['artifact_reasons'][:2])}")
    print(f"\nwrote per-world results + summary.json to {outdir}/")


if __name__ == "__main__":
    main()
