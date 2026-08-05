"""Batch driver for the advanced-benchmark dataset.

Edit OUTDIR / SEED_BASE / BACKEND / MODEL below to change the target.
Mirrors run_many.py's structure but runs world_gen_advanced.py.
"""
import subprocess
import sys

SCRIPT = "world_gen_advanced.py"
OUTDIR = "./out_bn_advanced_4_24_n30_2"
SEED_BASE = 1000
BACKEND = "bedrock"
MODEL = "us.anthropic.claude-opus-4-6-v1"


def run(n_nodes: int, seed: int) -> None:
    cmd = [
        sys.executable, SCRIPT,
        "--n-nodes-list", str(n_nodes),
        "--n-per-size", "1",
        "--seed-base", str(seed),
        "--outdir", OUTDIR,
        "--backend", BACKEND,
        "--model", MODEL,
        # 30-node faithfulness is expensive.  Cap attempts so hard seeds
        # skip instead of stalling, and shrink the inner CPD resample loop.
        "--max-attempts-per-world", "6",
        "--cpd-max-attempts", "20",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # # 10-node worlds
    # for i in range(1, 21):
    #     run(10, SEED_BASE + i)
    # # 20-node worlds
    # for i in range(1, 21):
    #     run(20, SEED_BASE + 100 + i)
    # 30-node worlds
    for i in range(1, 21):
        run(30, SEED_BASE + 200 + i)
    print("Done: 20 worlds generated for 30 nodes.")
