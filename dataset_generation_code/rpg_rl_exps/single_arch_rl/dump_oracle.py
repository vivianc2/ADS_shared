#!/usr/bin/env python3
"""Rebuild one deterministic RPG world and dump its computed oracle record."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skin", required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rpg_src = Path(os.environ.get("RPG_SRC", "/work/ADS_shared/dataset_generation_code"))
    for path in (rpg_src / "rpg_rl", rpg_src / "rpg_v9"):
        sys.path.insert(0, str(path))

    from generate_v7 import _json_default, audit, to_record  # noqa: WPS433
    from sampler import sample_world  # noqa: WPS433

    world = sample_world(args.seed, skin=args.skin, archetype=args.archetype)
    world["ground_truth"]["_seed"] = args.seed
    result = audit(world)
    record = to_record(world, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.output}: world_id={record['world_id']} audit_ok={result['ok']} "
        f"gold_utility={result['gold']['expected_utility']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
