#!/usr/bin/env python3
"""Rebuild / verify RPG v9 parquets against the CURRENT generator (2026-09-24).

Why: SkyRL shows the model the parquet's stored first prompt (the only copy of the id
catalog) while the env rebuilds the world from (seed, skin, archetype) with the checked-out
code. Parquets built before commit 745c110 (2026-08-30) no longer match. See
personal_docs/results/RPG_V9_DATASET_DRIFT_2026-09-23.md.

Modes (run from dataset_generation_code/, RPG_SYNERGY_SOFT=20):
  verify   <parquet>...                 count rows whose stored prompt != env.reset()
  rebuild  <in.parquet> <out.parquet>   same rows/seeds, prompt re-rendered by current code
  balanced <out.parquet> --per_arch 30  held-out-skin test set, every archetype, seeds 43M+

    PYTHONPATH=rpg_rl:rpg_v9 RPG_SYNERGY_SOFT=20 python skyrl_rpg/rebuild_v9_sets.py verify a.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

_BASE = os.environ.get("RPG_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_BASE, os.environ.get("RPG_PROTO", "rpg_v9")), os.path.join(_BASE, "rpg_rl")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import pyarrow as pa                                  # noqa: E402
import pyarrow.parquet as pq                          # noqa: E402
from sampler import sample_world, ARCHETYPES          # noqa: E402
from generate_v7 import audit                         # noqa: E402
from splits import HELDOUT_SKINS                      # noqa: E402
from env import RPGEnv, SYSTEM_PROMPT                 # noqa: E402

SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}


def _first_obs(seed, skin, arch, max_turns=32, budget=15):
    w = sample_world(int(seed), skin=skin, archetype=arch)
    w["ground_truth"]["_seed"] = int(seed)
    res = audit(w)
    env = RPGEnv(world=w, gold=res["gold"], battery=res["battery"], catalog_seed=int(seed),
                 max_turns=int(max_turns), budget=int(budget))
    return env.reset(), bool(res["ok"])


def _row_job(row):
    e = row["extra_info"]
    obs, ok = _first_obs(e["seed"], e["skin"], e["archetype"], e.get("max_turns", 32), e.get("budget", 15))
    return obs, ok


def _norm(text):
    """Stored prompt minus an optional trailing /no_think (rpg_dataset.py RPG_NO_THINK), as the env guard does."""
    return (text or "").strip().removesuffix("/no_think").strip()


def _scenario(text):
    return (text or "").split("OUTCOME OF INTEREST")[0].strip()


def _read(path):
    return pq.read_table(path).to_pylist()


def verify(paths, workers):
    for p in paths:
        rows = _read(p)
        with Pool(workers) as pool:
            out = pool.map(_row_job, rows, chunksize=2)
        bad = Counter()
        for r, (obs, _) in zip(rows, out):
            if _norm(r["prompt"][-1]["content"]) != obs.strip():
                bad[(r["extra_info"]["skin"], r["extra_info"]["archetype"])] += 1
        print(f"{p}: {sum(bad.values())}/{len(rows)} stale prompts" + (f"  {dict(bad)}" if bad else ""))


def rebuild(src, dst, workers, force=False):
    """Re-render ONLY the user observation; keep each row's own system message and a trailing /no_think.
    Refuses rows whose SITUATION text differs from the generator's (story-ablation sets rewrite it on purpose;
    rebuilding would erase the ablation) unless force=True."""
    rows = _read(src)
    with Pool(workers) as pool:
        out = pool.map(_row_job, rows, chunksize=2)
    n_changed = n_fail = n_story = 0
    for r, (obs, ok) in zip(rows, out):
        stored = r["prompt"][-1]["content"]
        if _scenario(_norm(stored)) != _scenario(obs):
            n_story += 1
        n_changed += _norm(stored) != obs.strip()
        n_fail += not ok
        suffix = "\n/no_think" if stored.strip().endswith("/no_think") else ""
        sys_msgs = [m for m in r["prompt"] if m.get("role") == "system"] or [SYSTEM_MSG]
        r["prompt"] = [*sys_msgs, {"role": "user", "content": obs + suffix}]
    if n_story and not force:
        raise SystemExit(f"{src}: {n_story} rows have a rewritten SITUATION (story ablation?) - rebuilding would "
                         f"erase it. Re-run with --force only if that is intended.")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    print(f"{src} -> {dst}: {len(rows)} rows, {n_changed} prompts re-rendered, "
          f"{n_fail} worlds now FAIL the audit (kept; flag them)")


def _bal_job(args):
    seed, skin, arch = args
    try:
        w = sample_world(seed, skin=skin, archetype=arch)
        res = audit(w)
    except Exception:  # noqa: BLE001
        return None
    if not res["ok"]:
        return None
    env = RPGEnv(world=w, gold=res["gold"], battery=res["battery"], catalog_seed=seed,
                 max_turns=32, budget=15)
    return {"data_source": "rpg_v9", "prompt": [SYSTEM_MSG, {"role": "user", "content": env.reset()}],
            "env_class": "rpg", "reward_spec": {"method": "rule", "ground_truth": ""},
            "extra_info": {"seed": int(seed), "skin": skin, "archetype": arch, "max_turns": 32,
                           "budget": 15, "split": "heldout_balanced"}}


def balanced(dst, per_arch, seed0, workers):
    skins = sorted(HELDOUT_SKINS)
    cells = [(a, s) for a in ARCHETYPES for s in skins]
    attempts = int(per_arch * len(ARCHETYPES) / 0.5) + 64
    jobs = [(seed0 + i, cells[i % len(cells)][1], cells[i % len(cells)][0]) for i in range(attempts)]
    with Pool(workers) as pool:
        res = pool.map(_bal_job, jobs, chunksize=2)
    keep, cnt = [], Counter()
    for r in res:                              # first per_arch accepted per archetype, seed order
        if r and cnt[r["extra_info"]["archetype"]] < per_arch:
            keep.append(r)
            cnt[r["extra_info"]["archetype"]] += 1
    short = {a: per_arch - cnt[a] for a in ARCHETYPES if cnt[a] < per_arch}
    assert not short, f"not enough accepted worlds: {short}; raise attempts"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(keep), dst)
    print(f"{dst}: {len(keep)} rows; per archetype {dict(cnt)}; "
          f"skins {dict(Counter(r['extra_info']['skin'] for r in keep))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "rebuild", "balanced"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--per_arch", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=43_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="rebuild even rows with a rewritten SITUATION")
    a = ap.parse_args()
    if a.mode == "verify":
        verify(a.paths, a.workers)
    elif a.mode == "rebuild":
        rebuild(a.paths[0], a.paths[1], a.workers, a.force)
    else:
        balanced(a.paths[0], a.per_arch, a.seed0, a.workers)


if __name__ == "__main__":
    main()
