#!/usr/bin/env python3
"""Build a SkyRL parquet for RL on RPG v10 from a folder of world files (2026-10-23).

Each row: prompt = [system, user0] exactly as the frozen benchmark loop (full history) opens the episode on that
world; extra_info.world_file = path relative to --root (the env resolves it against RLV10_DATA_ROOT, so the same
parquet works on the host and in the container); plus world_id, archetype, domain, split.

  ~/miniconda3/envs/ADS/bin/python build_dataset.py --root ~/rpgrl/rl_v10/data --worlds worlds_validation \
      --out ~/rpgrl/rl_v10/data/validation.parquet --split validation [--per-arch N]
"""
import argparse
import glob
import gzip
import json
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stepenv import BenchStepEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--worlds", required=True, help="folder under --root with world_*.json.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--per-arch", type=int, default=0, help="keep the first N worlds per structure (file order)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, args.worlds, "world_*.json*")))
    rows, cnt = [], Counter()
    for f in files:
        rec = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
        gt = rec["ground_truth"]
        arch = gt.get("_archetype")
        if args.per_arch and cnt[arch] >= args.per_arch:
            continue
        with tempfile.TemporaryDirectory() as dd:
            env = BenchStepEnv(f, dd, history="full")
            msgs = env.reset()
            env.close()
        cnt[arch] += 1
        rows.append({"data_source": "rpg_v10", "prompt": msgs, "env_class": "rpg_v10",
                     "reward_spec": {"method": "rule", "ground_truth": ""},
                     "extra_info": {"world_file": os.path.relpath(f, args.root), "world_id": rec["world_id"],
                                    "archetype": arch, "domain": rec.get("domain"), "split": args.split,
                                    "max_turns": 32, "budget": 15}})
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{args.out}: {len(rows)} rows; per structure {dict(cnt)}")


if __name__ == "__main__":
    main()
