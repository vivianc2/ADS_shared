#!/usr/bin/env python3
"""Check reward_v10.reward_components against the camera-ready D1 rescoring (priority 0.4).

For every stored v10.2 episode scored in personal_docs/camera_ready/analysis_rpg/d1_partb/episodes_d1.csv
(12 models x core54 + Opus 5.5 on the other 486 test worlds), recompute the components from the stored result file
and the world's roles and require: battery_stored == stored grade battery_fraction; battery_core, part_b_core,
accepted_core, part_a equal to the CSV.

  ~/miniconda3/envs/ADS/bin/python test_reward_v10.py
"""
import csv, glob, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reward_v10 import reward_components, RewardCfg   # noqa: E402

REL = os.path.expanduser("~/ADS_folder/ADS_shared/rpg_benchmark")
D1 = os.path.expanduser("~/ADS_folder/personal_docs/camera_ready/analysis_rpg/d1_partb/episodes_d1.csv")


def main():
    ref = {(r["model"], r["world_id"]): r for r in csv.DictReader(open(D1))}
    files = [(f, f.split("/out/gate_v10.2/")[1].split("/")[0])
             for f in glob.glob(os.path.join(REL, "out/gate_v10.2/*/shard_*/result_*.json"))]
    files += [(f, "opus55") for f in glob.glob(os.path.join(REL, "out/opus55_v10.2_rest486/shard_*/result_*.json"))]
    n = bad = missing = 0
    for f, m in files:
        res = json.load(open(f))
        wid = res["world_id"]
        rec = json.load(gzip.open(os.path.join(REL, "data/test", f"world_{wid}.json.gz")))
        world = {"ground_truth": rec["ground_truth"]}
        c = reward_components(res, world, {})
        r = ref.get((m, wid))
        if r is None:
            missing += 1
            continue
        n += 1
        checks = [abs(c["battery_stored"] - res["grade"]["battery_fraction"]) < 1e-3,
                  abs(c["battery_core"] - float(r["battery_core"])) < 1e-4,
                  c["part_b_core"] == int(r["part_b_core"]), c["accepted_core"] == int(r["accepted_core"]),
                  c["part_a"] == int(r["part_a"]), c["accepted_release"] == int(r["accepted_release"])]
        if not all(checks):
            bad += 1
            if bad <= 5:
                print("MISMATCH", m, wid, checks, c, {k: r[k] for k in ("battery_core", "part_b_core", "accepted_core")})
    cfg = RewardCfg()
    print(f"episodes {len(files)}, matched in D1 csv {n}, not in csv {missing}, mismatches {bad}; reward cfg {cfg.as_dict()}")
    return 1 if bad or not n else 0


if __name__ == "__main__":
    raise SystemExit(main())
