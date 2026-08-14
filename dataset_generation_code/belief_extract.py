"""Post-hoc belief-graph extraction from a scientist-agent run (idea #3, approach A).

For each world, make ONE LLM call that reads the agent's ordered per-turn scratchpad
(`memory` + `reasoning`) and emits, for every ACTION turn, a structured snapshot of the
agent's current causal belief:
    {turn, cause, proxy, ruled_out[], decoys[], signs{name:+/-/0}}
Names are constrained to the world's canonical actuator/measurable vocabulary so the
snapshots can be diffed and scored against the true SCM.

Output: <results_dir>/beliefs/<wid>.json  (list of snapshots + the vocab used).

Run:  PYTHONPATH=rpg_v8 python3 belief_extract.py <results_dir> [--limit N] [--per-arch K]
                                                  [--model us.anthropic.claude-opus-4-8] [--resume]
"""
import json, glob, os, sys, argparse, re
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit
from bedrock_llm import BedrockLLM

MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"

SYS = """You are analyzing the scratchpad of a causal-discovery agent that is running experiments
to find which single control lever fixes a degraded outcome, and which observable is the true
mechanism marker. You will be given the world's controllable ACTUATORS and measurable VARIABLES
(canonical names), then the agent's notes at each turn in order.

For EACH listed turn, output the agent's CURRENT belief as of that turn, using ONLY the canonical
names given (or null / []). Return a JSON array, one object per turn, in the same order:
  {"turn": <int>,
   "cause": <actuator name it currently believes is the true fix, or null>,
   "proxy": <measurable it believes is the true mechanism marker, or null>,
   "ruled_out": [actuators it has dismissed as non-causal],
   "decoys": [variables/actuators it has flagged as confounders / red herrings / bystanders],
   "signs": {actuator: "+" | "-" | "0"}   // direction it believes increasing that actuator moves the outcome
  }
Rules: reflect ONLY what the notes support at that turn (beliefs evolve — early turns are often
null/empty). Map paraphrases to the closest canonical name; if none fits, omit it. Output ONLY the
JSON array, no prose."""


def vocab(scm):
    acts = list(scm.actuators.keys())
    meas = [nm for nm, s in scm.variables.items() if s.get("kind") == "observable" or s.get("measurable")]
    return acts, meas


def extract_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        text = text[i:j + 1]
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-arch", type=int, default=None, help="cap worlds per archetype")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    man_path = os.path.join(args.results_dir, "manifest.json") \
        if os.path.exists(os.path.join(args.results_dir, "manifest.json")) else MAN
    man = {m["world_id"]: m for m in json.load(open(man_path))}
    outdir = os.path.join(args.results_dir, "beliefs")
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.results_dir, "result_*.json")))

    # optional stratified subsetting
    if args.per_arch:
        by = defaultdict(list)
        for f in files:
            wid = json.load(open(f))["world_id"]
            by[man.get(wid, {}).get("archetype", "?")].append(f)
        files = [f for a in sorted(by) for f in by[a][:args.per_arch]]
    if args.limit:
        files = files[:args.limit]

    llm = BedrockLLM(model_id=args.model, max_new_tokens=8192, temperature=0.0)
    n_ok = 0
    for f in files:
        r = json.load(open(f))
        wid = r["world_id"]
        outp = os.path.join(outdir, f"{wid}.json")
        if args.resume and os.path.exists(outp):
            n_ok += 1
            continue
        t = man.get(wid)
        if not t:
            continue
        w = sample_world(t["seed"], skin=t["skin"], archetype=t["archetype"])
        acts, meas = vocab(w["scm"])
        turns = [tn for tn in r["turns"] if tn.get("action_type") in ("measure", "intervene", "answer")]
        notes = []
        for tn in turns:
            notes.append({"turn": int(tn.get("turn", 0)), "action": tn.get("action_type"),
                          "reasoning": (tn.get("reasoning") or "")[:600],
                          "memory": (tn.get("memory") or "")[:900]})
        user = (f"ACTUATORS (controllable levers): {acts}\n"
                f"MEASURABLE VARIABLES: {meas}\n\n"
                f"Agent notes, in order ({len(notes)} action turns):\n"
                + json.dumps(notes, indent=1))
        try:
            raw = llm.generate(SYS, user)
            snaps = extract_json_array(raw)
        except Exception as e:
            print(f"  FAIL {wid}: {type(e).__name__}: {e}")
            continue
        json.dump({"world_id": wid, "archetype": t["archetype"], "skin": t["skin"],
                   "actuators": acts, "measurables": meas, "snapshots": snaps},
                  open(outp, "w"), indent=2)
        n_ok += 1
        print(f"  ok {wid}: {len(snaps)} snapshots")
    print(f"extracted beliefs for {n_ok}/{len(files)} worlds -> {outdir}")


if __name__ == "__main__":
    main()
