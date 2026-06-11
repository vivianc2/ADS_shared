import subprocess
import sys

SCRIPT = "world_gen_xxx.py"
OUTDIR = "./out_bn_xxx"
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



# # 10-node worlds
# for i in range(1, 6):
#     run(10, SEED_BASE + i)

# # 20-node worlds
# for i in range(1, 6):
#     run(20, SEED_BASE + 100 + i)

# # 30-node worlds
# for i in range(1, 6):
#     run(30, SEED_BASE + 200 + i)

# print("Done: 15 worlds generated.")




# # 3-node worlds
# for i in range(1, 21):
#     run(3, SEED_BASE + i)

# # 4-node worlds
# for i in range(1, 21):
#     run(4, SEED_BASE + 100 + i)
