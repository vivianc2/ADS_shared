#!/usr/bin/env python3
"""Sandbox working-directory test (2026-10-01): files written by model code must land in the episode's data dir
(where its experiment CSVs are), persist across code turns of the same episode, and never land in the caller's cwd.

Run:  cd rpg_v9 && python test_sandbox_cwd.py
"""
import os, sys, tempfile
from sandbox import run_code


def main():
    caller = tempfile.mkdtemp(prefix="caller_"); os.chdir(caller)
    d = tempfile.mkdtemp(prefix="episode_"); p = os.path.join(d, "experiment_1.csv")
    open(p, "w").write("m0,m1\n1,2\n3,4\n")
    fails = []
    out, _ = run_code('experiment_1_df.to_csv("written_by_model.csv"); print(experiment_1_df.m0.mean())',
                      {"experiment_1_csv": p})
    if "2.0" not in out: fails.append(f"code did not run: {out!r}")
    if not os.path.exists(os.path.join(d, "written_by_model.csv")): fails.append("file not in episode data dir")
    if os.path.exists(os.path.join(caller, "written_by_model.csv")): fails.append("file leaked into caller cwd")
    out2, _ = run_code('import pandas as pd; print(pd.read_csv("written_by_model.csv").shape)', {"experiment_1_csv": p})
    if "(2, 3)" not in out2: fails.append(f"file did not persist across code turns: {out2!r}")
    out3, _ = run_code('open("early.txt","w").write("x"); print("ok")', {})
    if "ok" not in out3 or os.path.exists(os.path.join(caller, "early.txt")): fails.append(f"no-CSV case leaked/failed: {out3!r}")
    print("ALL PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
