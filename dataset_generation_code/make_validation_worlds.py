"""Reproduce a stratified per-archetype sample of the RL fast-dataset worlds.

Regenerates worlds deterministically from (seed, skin, archetype) triples dumped
from the RL parquet, so these are the *exact* worlds the RL go/no-go run trains on.

Run:
  cd ADS_shared/dataset_generation_code
  PYTHONPATH=rpg_v8:rpg_rl python3 make_validation_worlds.py
"""
import json, os, random
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit, to_record, _json_default

N_PER_ARCH = 8                     # ~8 x 9 archetypes -> ~70 worlds
OUT = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds"
TRIPLES = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast/triples.json"

os.makedirs(OUT, exist_ok=True)
trips = json.load(open(TRIPLES))
by = defaultdict(list)
for t in trips:
    by[t["archetype"]].append(t)

rng = random.Random(0)
picked = []
for a in sorted(by):               # sorted for determinism across dict orderings
    lst = list(by[a])
    rng.shuffle(lst)
    picked += lst[:N_PER_ARCH]

man = []
for t in picked:
    w = sample_world(t["seed"], skin=t["skin"], archetype=t["archetype"])
    w["ground_truth"]["_seed"] = t["seed"]
    res = audit(w)                              # embeds gold + counterfactual battery
    rec = to_record(w, res)
    fn = f"world_{w['world_id']}.json"
    json.dump(rec, open(os.path.join(OUT, fn), "w"), indent=2, default=_json_default)
    man.append({**t, "file": fn, "world_id": w["world_id"], "audit_ok": res["ok"]})

json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
counts = defaultdict(int)
for m in man:
    counts[m["archetype"]] += 1
print(f"wrote {len(man)} worlds to {OUT}; audit_ok={sum(m['audit_ok'] for m in man)}/{len(man)}")
for a in sorted(counts):
    print(f"  {a:22s} {counts[a]}")
