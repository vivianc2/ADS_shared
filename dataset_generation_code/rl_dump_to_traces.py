"""Convert a SkyRL eval-rollout dump (Qwen, id-space) into run_batch_v6-compatible
result_<wid>.json traces, so the existing trajectory + belief pipelines work unchanged.

Reading the dumps is a pure disk read — no GPU, does not disturb the RL run.
Each rollout's output_response is a sequence of
  <reasoning>...</reasoning><action type="measure|intervene|answer">{json}</action><memory>...</memory>
Opaque ids (a3/m2) are mapped to canonical names via a catalog rebuilt with
seed = world_seed (the RL env's catalog_seed; verified against the dump's own catalog block).

Run:  PYTHONPATH=rpg_v8:rpg_rl python3 rl_dump_to_traces.py <dump.jsonl> <out_results_dir> <model_label>
"""
import json, os, re, sys
from sampler import sample_world
from generate_v7 import audit
from catalog import build_catalog
from reward import compute_reward, RewardConfig

DUMP, OUTDIR, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)

TURN_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<action type=\"(\w+)\">(\{.*?\})</action>"
    r"(?:\s*<memory>(.*?)</memory>)?", re.S)


def parse_turns(resp, cat):
    turns, qi, final_answer = [], 0, None
    for tno, mt in enumerate(TURN_RE.finditer(resp)):
        reasoning, atype, body, memory = mt.groups()
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
        t = {"turn": tno, "action_type": atype,
             "reasoning": (reasoning or "").strip(), "memory": (memory or "").strip()}
        if atype in ("measure", "intervene"):
            qi += 1
        if atype == "intervene":
            iv = {}
            for a in payload.get("actions", []) or []:
                nm = cat.actuator_name(a.get("actuator")) if isinstance(a.get("actuator"), str) else None
                if nm is not None:
                    iv[nm] = a.get("value")
            t["result"] = {"applied_intervention": iv, "experiment_id": qi}
        elif atype == "measure":
            t["result"] = {"experiment_id": qi}
        elif atype == "answer":
            final_answer = payload
            t["result"] = {}
        turns.append(t)
    return turns, qi, final_answer


def main():
    manifest = []
    n = 0
    for line in open(DUMP):
        rec = json.loads(line)
        ei = rec["env_extras"]["extra_info"]
        seed, skin, arch = ei["seed"], ei["skin"], ei["archetype"]
        w = sample_world(seed, skin=skin, archetype=arch)
        res = audit(w)
        gold, battery = res["gold"], res["battery"]
        cat = build_catalog(w, w["scm"], seed=seed)
        wid = w["world_id"]
        turns, qi, ans = parse_turns(rec["output_response"], cat)
        n_iv = sum(1 for t in turns if t["action_type"] == "intervene")
        answered = any(t["action_type"] == "answer" for t in turns)
        # grade the final id-answer via the ACTUAL RL reward path (deterministic, id-space)
        grade = {}
        if ans is not None:
            rr = compute_reward(ans, w, cat, gold, battery, RewardConfig())
            grade = rr.get("grade", {})
            grade["_reward"] = rr.get("reward")
            grade["part_a"] = rr.get("part_a")
            grade["part_b"] = rr.get("part_b")
        out = {"world_id": wid, "queries_used": qi, "interventions_run": n_iv,
               "answered": answered, "hit_turn_cap": rec.get("stop_reason") != "stop",
               "grade": grade, "turns": turns, "model": LABEL}
        json.dump(out, open(os.path.join(OUTDIR, f"result_{wid}.json"), "w"), indent=1, default=str)
        manifest.append({"world_id": wid, "seed": seed, "skin": skin, "archetype": arch,
                         "file": f"world_{wid}.json"})
        n += 1
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=1)
    print(f"{LABEL}: converted {n} rollouts -> {OUTDIR} (+ manifest.json)")


if __name__ == "__main__":
    main()
