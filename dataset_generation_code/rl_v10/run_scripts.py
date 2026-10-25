#!/usr/bin/env python3
"""Run the camera-ready scripted baselines on any folder of release-format world files (2026-10-23).

Imports the per-world `job` functions of personal_docs/camera_ready/analysis_rpg/baselines/{run_baselines,
cd1_causal_discovery}.py unchanged (read-only use; those files belong to the camera-ready session) and writes the
episode rows to --out (never into the camera-ready folder). Same simulator, grader and id adapter as the published
baseline numbers.

  OMP_NUM_THREADS=1 RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 ~/miniconda3/envs/ADS/bin/python run_scripts.py \
      --worlds-dir ~/rpgrl/rl_v10/data/worlds_validation_cuefree --out ~/rpgrl/rl_v10/scripts/validation_cuefree \
      [--which baselines,cd1] [--cd1-only nocd_corr,nocd_corr_screen,cd_pc_rcit_ivbase] [--workers 32]
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

BASE = os.path.expanduser("~/ADS_folder/personal_docs/camera_ready/analysis_rpg/baselines")
sys.path.insert(0, BASE)
os.environ.setdefault("RPG_PROTO", "rpg_v9")
os.environ.setdefault("RPG_SYNERGY_SOFT", "20")
import run_baselines as rb  # noqa: E402
import cd1_causal_discovery as cd1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="baselines,cd1")
    ap.add_argument("--cd1-only", default="nocd_corr,nocd_corr_screen,nocd_mi,nocd_random,cd_pc_rcit_ivbase,cd_pc_fz_screen")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.worlds_dir, "index.json")))
    core = set(open(os.path.join(rb.REL, "data/core54/worlds.txt")).read().split())
    jobs = []
    for e in idx:
        wid = e["file"][len("world_"):-len(".json.gz")]
        jobs.append((os.path.join(a.worlds_dir, e["file"]), {**e, "core54": wid in core}))
    if a.limit:
        jobs = jobs[: a.limit]
    which = set(a.which.split(","))
    if "baselines" in which:
        rows = []
        with Pool(a.workers) as p:
            for r in p.imap_unordered(rb.job, jobs, chunksize=1):
                rows.extend(r)
        rows.sort(key=lambda r: (r["policy"], int(r["budget"]), r["world_id"]))
        fn = os.path.join(a.out, "baselines_episodes.csv")
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rb.FIELDS); w.writeheader(); w.writerows(rows)
        print("wrote", fn, len(rows), flush=True)
    if "cd1" in which:
        cd1.ONLY = set(a.cd1_only.split(",")) if a.cd1_only else set()
        eps = []
        with Pool(a.workers) as p:
            for e, _g in p.imap_unordered(cd1.job, jobs, chunksize=1):
                eps.extend(e)
        eps.sort(key=lambda r: tuple(str(r[k]) for k in cd1.EP_FIELDS[:3]))
        fn = os.path.join(a.out, "cd1_episodes.csv")
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cd1.EP_FIELDS); w.writeheader(); w.writerows(eps)
        print("wrote", fn, len(eps), flush=True)


if __name__ == "__main__":
    main()
