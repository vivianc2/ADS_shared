#!/usr/bin/env python3
"""Training mix of standard and confounded (v3) worlds (2026-10-24, camera-ready session).

Why: in every standard v10 world the lever is the setting whose passive drift tracks the outcome, so a policy trained
only on standard worlds can be rewarded for following that cue. Mixing in v3 worlds (rl_v10/cuefree.py
make_confounded), where passive data name the lever only sometimes, makes the cue unreliable during training.

Split: the standard train split's rows are divided per structure by sha1(world_id) order. The first half stays
standard, and the second half is rebuilt as v3 from the same seeds. A seed appears in only one version, so the policy
never sees the same world twice.

  step 1  python make_train_mix.py split <train.parquet> <out_dir>
          writes <out_dir>/v3_rows.parquet (the rows to rebuild) and <out_dir>/standard_ids.txt
  step 2  build_cuefree.py <out_dir>/v3_rows.parquet <out_dir>/v3_dump --design v3, then
          to_release_format.py <out_dir>/v3_dump <out_dir>/worlds_v3
  step 3  python make_train_mix.py link <standard_world_dir> <out_dir>
          writes <out_dir>/worlds_mix/ (symlinks: standard half + every v3 world that passed its audits;
          v3 files are named world_v3_<id>.json.gz)
  step 4  build_dataset.py --root <data_root> --worlds <out_dir relative to data_root>/worlds_mix \
              --out <out_dir>/train_mix_v3.parquet --split train
          --root must be the env's data root (RLV10_DATA_ROOT, ~/rpgrl/rl_v10/data on the host): world_file is stored
          relative to it. (10-24: the first build used --root <out_dir>, so no row resolved; fixed by the RL session.)
"""
import collections
import glob
import hashlib
import json
import os
import sys


def split(parquet, out):
    import pandas as pd
    df = pd.read_parquet(parquet)
    infos = [e if isinstance(e, dict) else json.loads(e) for e in df["extra_info"]]
    by = collections.defaultdict(list)
    for i, e in enumerate(infos):
        wid = f"{e['skin']}_{e['archetype']}_{int(e['seed'])}"        # release-format world_id
        by[e["archetype"]].append((hashlib.sha1(wid.encode()).hexdigest(), i, wid))
    v3_idx, std_ids = [], []
    for arch, rows in sorted(by.items()):
        rows.sort()
        half = len(rows) // 2
        std_ids += [wid for _, _, wid in rows[:half]]
        v3_idx += [i for _, i, _ in rows[half:]]
    os.makedirs(out, exist_ok=True)
    df.iloc[sorted(v3_idx)].to_parquet(os.path.join(out, "v3_rows.parquet"))
    open(os.path.join(out, "standard_ids.txt"), "w").write("".join(w + "\n" for w in sorted(std_ids)))
    print(f"{len(std_ids)} standard, {len(v3_idx)} to rebuild as v3, per structure "
          f"{ {a: len(r) for a, r in sorted(by.items())} }")


def _rel(target, link_dir):
    """Relative symlink target, so the links also resolve inside the skyrl container (data root mounted elsewhere)."""
    return os.path.relpath(os.path.realpath(target), os.path.realpath(link_dir))


def link(std_dir, out):
    std = set(open(os.path.join(out, "standard_ids.txt")).read().split())
    mix = os.path.join(out, "worlds_mix")
    os.makedirs(mix, exist_ok=True)
    n = collections.Counter()
    for f in sorted(glob.glob(os.path.join(std_dir, "world_*.json.gz"))):
        wid = os.path.basename(f)[len("world_"):-len(".json.gz")]
        if wid in std:
            os.symlink(_rel(f, mix), os.path.join(mix, os.path.basename(f)))
            n["standard"] += 1
    for f in sorted(glob.glob(os.path.join(out, "worlds_v3", "world_*.json.gz"))):
        # world_v3_<id>: build_dataset globs world_*; the world_id inside the file is unchanged
        os.symlink(_rel(f, mix), os.path.join(mix, "world_v3_" + os.path.basename(f)[len("world_"):]))
        n["v3"] += 1
    missing = len(std) - n["standard"]
    if missing:
        raise SystemExit(f"{missing} standard ids not found in {std_dir}")
    print(f"{mix}: {dict(n)}")


if __name__ == "__main__":
    {"split": split, "link": link}[sys.argv[1]](*sys.argv[2:])
