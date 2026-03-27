import subprocess
import sys

SCRIPT = "world_gen_causal.py"
OUTDIR = "./out_bn_2_24"
SEED_BASE = 1000

def run(n_nodes, seed):
    cmd = [
        sys.executable, SCRIPT,
        "--n_nodes", str(n_nodes),
        "--seed", str(seed),
        "--outdir", OUTDIR,
    ]
    subprocess.run(cmd, check=True)

# 10-node worlds
for i in range(1, 21):
    run(10, SEED_BASE + i)

# 20-node worlds
for i in range(1, 21):
    run(20, SEED_BASE + 100 + i)

# 30-node worlds
for i in range(1, 21):
    run(30, SEED_BASE + 200 + i)

print("Done: 60 worlds generated.")
