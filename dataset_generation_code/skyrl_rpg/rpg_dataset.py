#!/usr/bin/env python3
"""Build SkyRL parquet datasets of RPG v7 worlds.

Each row = one audited world. We render the world's FIRST observation (via the same
RPGEnv.reset() the env uses at train time -> identical text, guaranteed by determinism)
into the SkyRL chat `prompt`, and stash the world's identity (seed/skin/archetype) in
`extra_info` so `RPGSkyEnv` can rebuild the exact world for stepping.

TRAIN split uses only train skins/archetypes; VALIDATION uses the held-out split (reserved
skins/archetypes) — the transfer set (splits.py, decision V13). Optionally restrict TRAIN
to `--archetypes confounded_chain` for the curriculum-style easy start (the probe showed the
gradient lives there for the base model).

Run (inside the SkyRL container):
    uv run --isolated python -m examples.train.rpg.rpg_dataset \
        --output_dir /work/data/rpg --train_size 512 --val_size 64
"""

from __future__ import annotations

import argparse
import os
import sys

_BASE = os.environ.get("RPG_SRC", "/work/ADS_shared/dataset_generation_code")
for _p in (os.path.join(_BASE, "rpg_rl"), os.path.join(_BASE, os.environ.get("RPG_PROTO", "rpg_v8"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import Dataset                       # noqa: E402
from env import SYSTEM_PROMPT                       # noqa: E402
from world_stream import WorldStream                # noqa: E402

SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}


def _rows(split: str, n: int, seed0: int, archetypes, max_turns: int, budget: int):
    arch = [a.strip() for a in archetypes.split(",") if a.strip()] or None
    stream = WorldStream(split=split, seed0=seed0, archetypes=arch)
    rows = []
    for b in stream.take(n):
        env = b.make_env(max_turns=max_turns, budget=budget)
        first_obs = env.reset()                    # exactly what RPGSkyEnv will reset() to
        if os.environ.get("RPG_NO_THINK"):         # match the env's per-turn /no_think
            first_obs = first_obs + "\n/no_think"
        rows.append({
            "data_source": "rpg_v7",
            "prompt": [SYSTEM_MSG, {"role": "user", "content": first_obs}],
            "env_class": "rpg",
            # reward is computed by the env from the rebuilt world; ground_truth unused,
            # kept as a placeholder for schema compatibility.
            "reward_spec": {"method": "rule", "ground_truth": ""},
            "extra_info": {
                "seed": int(b.seed), "skin": b.skin, "archetype": b.archetype,
                "max_turns": int(max_turns), "budget": int(budget), "split": split,
            },
        })
    print(f"[{split}] built {len(rows)} rows (acceptance {stream.acceptance():.0%}); "
          f"archetypes={arch or 'ALL'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=os.path.expanduser("~/data/rpg"))
    ap.add_argument("--train_size", type=int, default=512)
    ap.add_argument("--val_size", type=int, default=64)
    ap.add_argument("--train_seed0", type=int, default=10_000_000)
    ap.add_argument("--val_seed0", type=int, default=20_000_000)
    ap.add_argument("--archetypes", default="",
                    help="comma-sep restriction for TRAIN (e.g. confounded_chain); val is full held-out")
    ap.add_argument("--max_turns", type=int, default=32)
    ap.add_argument("--budget", type=int, default=15)
    args = ap.parse_args()

    train = _rows("train", args.train_size, args.train_seed0, args.archetypes,
                  args.max_turns, args.budget)
    val = _rows("heldout", args.val_size, args.val_seed0, "", args.max_turns, args.budget)

    os.makedirs(args.output_dir, exist_ok=True)
    Dataset.from_list(train).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    Dataset.from_list(val).to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"wrote train({len(train)}) + validation({len(val)}) to {args.output_dir}")


if __name__ == "__main__":
    main()
