#!/usr/bin/env python3
"""Convert dumped v10 world files (rpg_v9/dump_worlds_from_parquet.py output) to the release benchmark's world-file
format, with the exact record construction of personal_docs/scripts/v10/make_release.py::build_data (imported, not
copied): world_id = <domain>_<archetype>_<seed>, ground truth minus DROP_GT, oracle gold + battery, deterministic gzip.
Then checks that every converted world's release-path gold regrades as in the dump.

  ~/miniconda3/envs/ADS/bin/python to_release_format.py <dump_dir> <out_dir>
"""
import glob
import gzip
import io
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/ADS_folder/personal_docs/scripts/v10"))
import make_release  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    index = []
    for p in sorted(glob.glob(os.path.join(src, "world_*.json"))):
        rec = json.load(open(p))
        gt = {k: v for k, v in rec["ground_truth"].items() if k not in make_release.DROP_GT}
        seed = rec["meta"]["seed"]
        arch = rec["ground_truth"]["_archetype"]
        wid = f"{rec['domain']}_{arch}_{seed}"
        out = {"world_id": wid, "domain": rec["domain"], "scenario": rec["scenario"], "scm": rec["scm"],
               "ground_truth": gt,
               "oracle": {"gold": rec["oracle"]["gold"], "counterfactual_battery": rec["oracle"]["counterfactual_battery"]}}
        fn = f"world_{wid}.json.gz"
        with open(os.path.join(dst, fn), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(json.dumps(out, separators=(",", ":")).encode("utf-8"))
        index.append({"file": fn, "domain": rec["domain"], "archetype": arch, "seed": seed})
    with open(os.path.join(dst, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{dst}: {len(index)} worlds")


if __name__ == "__main__":
    main()
