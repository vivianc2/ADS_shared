#!/usr/bin/env python3
"""Evaluate an API model on the RL benchmark in ID SPACE (the same opaque-id catalog the
RL policy sees — resolver-free), via rollout.py. GPU-free: env is CPU, model is a Bedrock/
Nautilus API call. Gives the RL-COMPARABLE ceiling (directly comparable to run-8's step-0),
and quantifies how much the free-text (menu-free + resolver) harness deflates.

Usage:
  python id_space_eval.py --worlds-dir <dir> --backend bedrock --model us.anthropic.claude-opus-4-8 \
      --outdir <dir> --concurrency 4 [--limit N]
"""
import argparse, glob, json, os, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from engine import WorldSCM
from env import RPGEnv
from reward import RewardConfig
from rollout import rollout_episode


def build_llm(backend, model, max_new_tokens):
    if backend == "bedrock":
        from bedrock_llm import BedrockLLM
        return BedrockLLM(model_id=model, max_new_tokens=max_new_tokens)
    from openai_llm import OpenAILLM
    base = os.environ.get("NAUTILUS_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
    return OpenAILLM(model_name=model, base_url=base, api_key=os.environ.get("NAUTILUS_API_KEY"),
                     max_new_tokens=max_new_tokens)


def load_world(path):
    rec = json.load(open(path))
    world = {"world_id": rec["world_id"], "domain": rec["domain"], "scenario": rec["scenario"],
             "scm": WorldSCM.from_dict(rec["scm"]), "ground_truth": rec["ground_truth"]}
    return world, rec["oracle"]["gold"], rec["oracle"]["counterfactual_battery"], int(rec["meta"]["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--backend", default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(f for f in glob.glob(f"{args.worlds_dir}/world_*.json"))
    if args.limit:
        files = files[:args.limit]
    llm = build_llm(args.backend, args.model, args.max_new_tokens)
    def gen_fn(system, user, mnt):
        return llm.generate(system, user, max_new_tokens=mnt or args.max_new_tokens)

    def one(path):
        world, gold, battery, seed = load_world(path)
        arche = (world["ground_truth"] or {}).get("_archetype", "?")
        try:
            env = RPGEnv(world=world, gold=gold, battery=battery, catalog_seed=seed,
                         max_turns=32, budget=15, reward_cfg=RewardConfig())
            tr = rollout_episode(env, gen_fn, max_new_tokens=args.max_new_tokens)
            row = {"world_id": world["world_id"], "archetype": arche, "reward": tr.reward,
                   "part_a": tr.part_a, "part_b": tr.part_b, "accepted": tr.accepted,
                   "n_turns": tr.n_turns, "n_interv": tr.info.get("n_interventions"),
                   "benefit": (tr.info.get("grade") or {}).get("benefit_recovered")}
        except Exception as e:
            row = {"world_id": world["world_id"], "archetype": arche, "error": f"{type(e).__name__}: {e}"}
        json.dump(row, open(f"{args.outdir}/res_{world['world_id']}.json", "w"), indent=1)
        print(f"  {arche:20s} {world['world_id'][:42]:42s} r={row.get('reward')} pa={row.get('part_a')} ben={row.get('benefit')} err={row.get('error','')}")
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(one, files))

    by = defaultdict(list)
    for r in rows:
        by[r["archetype"]].append(r)
    def m(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs)/len(xs) if xs else float("nan")
    print("\n%-20s %4s %8s %8s %8s %8s" % ("archetype","n","reward","partA%","benefit","partB"))
    allr=[]
    for a in sorted(by):
        gs=[x for x in by[a] if "error" not in x]; allr+=gs
        print("%-20s %4d %8.3f %7.0f%% %8.3f %8.3f" % (a, len(by[a]), m([x["reward"] for x in gs]),
              100*m([1.0 if x["part_a"] else 0.0 for x in gs]), m([x.get("benefit") for x in gs]), m([x["part_b"] for x in gs])))
    print("%-20s %4d %8.3f %7.0f%% %8.3f %8.3f" % ("OVERALL", len(allr), m([x["reward"] for x in allr]),
          100*m([1.0 if x["part_a"] else 0.0 for x in allr]), m([x.get("benefit") for x in allr]), m([x["part_b"] for x in allr])))
    errs=[r for r in rows if "error" in r]
    if errs: print("ERRORS:", len(errs), errs[:3])


if __name__ == "__main__":
    sys.exit(main())
