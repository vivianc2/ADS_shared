#!/usr/bin/env python3
"""Manifests (file name + md5) of the RL v10 data, so a rebuilt data root can be checked against the one we trained
and evaluated on (2026-10-25). The worlds live outside git under the data root (RLV10_DATA_ROOT, ~/rpgrl/rl_v10/data);
the small evaluation splits are also copied into git (rl_v10/data/worlds/, see README).

  python manifest.py write <data_root>     # writes manifests/<set>.tsv here
  python manifest.py check <data_root>     # exit 1 on any missing or different file

World files are gzip with mtime 0, so their md5 is reproducible. Parquet md5s depend on the pyarrow version, so a
parquet difference alone is reported but not counted as a failure; its rows are checked by the env's drift guard.
"""
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "manifests")
SETS = {   # manifest name -> folder under the data root
    "train": "worlds_train",
    "train_mix_v3": "train_mix_v3/worlds_mix",
    "validation": "worlds_validation",
    "validation_confounded_v3": "worlds_validation_confounded_v3",
    "val_indomain": "worlds_val_indomain",
    "val_indomain_confounded_v3": "worlds_val_indomain_confounded_v3",
    "core54": "worlds_core54",
    "core108": "worlds_core108",
}
PARQUETS = ["train.parquet", "train_mix_v3/train_mix_v3.parquet", "validation.parquet", "core54.parquet",
            "core108.parquet", "validation_confounded_v3.parquet", "val_indomain.parquet",
            "val_indomain_confounded_v3.parquet"]


def _md5(p):
    with open(p, "rb") as f:            # follows symlinks (train_mix_v3/worlds_mix holds links)
        return hashlib.md5(f.read()).hexdigest()


def _rows(root):
    out = {}
    for name, sub in SETS.items():
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            out[name] = [(os.path.basename(f), _md5(f)) for f in sorted(glob.glob(os.path.join(d, "world_*.json.gz")))]
    out["parquets"] = [(p, _md5(os.path.join(root, p))) for p in PARQUETS if os.path.exists(os.path.join(root, p))]
    return out


def write(root):
    os.makedirs(OUT, exist_ok=True)
    for name, rows in _rows(root).items():
        with open(os.path.join(OUT, f"{name}.tsv"), "w") as f:
            f.write("".join(f"{a}\t{b}\n" for a, b in rows))
        print(f"{name}: {len(rows)}")


def check(root):
    bad = 0
    have = _rows(root)
    for fn in sorted(glob.glob(os.path.join(OUT, "*.tsv"))):
        name = os.path.basename(fn)[:-4]
        want = dict(line.rstrip("\n").split("\t") for line in open(fn) if line.strip())
        got = dict(have.get(name, []))
        miss = sorted(set(want) - set(got))
        diff = sorted(k for k in want if k in got and got[k] != want[k])
        extra = sorted(set(got) - set(want))
        hard = name != "parquets"
        bad += hard * (len(miss) + len(diff))
        print(f"{name}: {len(want)} expected, {len(miss)} missing, {len(diff)} different, {len(extra)} extra"
              + ("" if hard or not diff else "  (parquet md5 depends on pyarrow; not a failure)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    {"write": write, "check": check}[sys.argv[1]](os.path.expanduser(sys.argv[2]))
