"""Batch driver for the advanced-benchmark v3 dataset.

Default target: 300 worlds = 5 archetypes x 2 graph sizes x 30 worlds/cell.
That is large enough for per-archetype and per-size reporting while keeping
LLM-backed generation/evaluation costs manageable for a paper run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "world_gen_advanced.py"
DEFAULT_OUTDIR = SCRIPT_DIR / "all_out_bn" / "out_bn_adv_v3_300"
SEED_BASE = 3000
BACKEND = "bedrock"
MODEL = "us.anthropic.claude-opus-4-7"
N_NODES = [10, 15]
MAX_ATTEMPTS_PER_WORLD = 6

ARCHETYPES = [
    "safety_constrained",
    "mediator_structure",
    "satisficing",
    "subgroup_robust",
    "invalid_premise",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a balanced world_gen_advanced.py benchmark dataset. "
            "Defaults to 300 worlds: 60 per archetype, round-robin over "
            "10- and 15-node graphs."
        ),
    )
    ap.add_argument("--per-cell", type=int, default=30,
                    help="Worlds per (archetype, n_nodes) cell. Default: 30.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help=f"Output directory. Default: {DEFAULT_OUTDIR}")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--backend", type=str, default=BACKEND, choices=["bedrock"])
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--n-nodes", type=int, nargs="+", default=N_NODES,
                    help="Graph sizes to mix round-robin. Default: 10 15.")
    ap.add_argument("--max-attempts-per-world", type=int,
                    default=MAX_ATTEMPTS_PER_WORLD)
    ap.add_argument("--only-archetype", choices=ARCHETYPES, default=None,
                    help="Generate only one archetype, useful for debugging.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the world_gen_advanced.py command and exit.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_cell <= 0:
        raise ValueError("--per-cell must be positive")

    archetypes = [args.only_archetype] if args.only_archetype else ARCHETYPES
    total_worlds = args.per_cell * len(archetypes) * len(args.n_nodes)
    commands = []
    for arch_i, archetype in enumerate(archetypes):
        for size_i, n_nodes in enumerate(args.n_nodes):
            seed = args.seed_base + arch_i * 10_000 + size_i * 1_000
            distribution = {archetype: args.per_cell}
            commands.append([
                sys.executable, str(SCRIPT),
                "--n-nodes", str(n_nodes),
                "--seed-base", str(seed),
                "--outdir", str(args.outdir),
                "--backend", args.backend,
                "--model", args.model,
                "--max-attempts-per-world", str(args.max_attempts_per_world),
                "--distribution", json.dumps(distribution, sort_keys=True),
            ])

    print(f"Target worlds: {total_worlds}")
    print(f"Archetypes: {', '.join(archetypes)}")
    print(f"Graph sizes: {', '.join(str(n) for n in args.n_nodes)}")
    print(f"Per cell: {args.per_cell}")
    for cmd in commands:
        print("Running:", " ".join(cmd))
    if args.dry_run:
        return

    for cmd in commands:
        subprocess.run(cmd, check=True)
    print(f"\nDone. Worlds saved to {args.outdir}")


if __name__ == "__main__":
    main()
