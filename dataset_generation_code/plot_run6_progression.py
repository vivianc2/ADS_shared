"""Clean apples-to-apples figure: run-6 (step 0 vs step 5) vs Opus 4.8 reference.

Only the current go/no-go run's own eval steps (0, 5) — no mixed prior/old runs.
Left: best-so-far benefit vs query (how fast each climbs toward gold). Right: final
best-so-far benefit as bars (run-6 step0, step5, Opus), with gold=1.0 reference.

Run:  conda run -n ADS-rpg python plot_run6_progression.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code"
S0 = json.load(open(f"{BASE}/results_qwen_step0/trajectories.json"))
S5 = json.load(open(f"{BASE}/results_qwen_step5/trajectories.json"))
OP = json.load(open(f"{BASE}/results_v8_validation_opus/trajectories.json"))

LIGHT, DARK, ORANGE = "#9ec5f4", "#256abf", "#eb6834"
MUTED, GRID, INK, INK2 = "#898781", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.dpi": 140})


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9); ax.set_axisbelow(True)


kmax = OP["kmax"]; xs = list(range(1, kmax + 1))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1]})

# ---- Panel A: trajectories ----
axA.axhline(1.0, ls="--", lw=2, color=MUTED)
axA.text(kmax, 1.01, "gold = 1.0 (oracle optimum)", ha="right", va="bottom", color=MUTED, fontsize=11, fontweight="bold")
series = [(S0, LIGHT, "Qwen run-6 · step 0 (base)"),
          (S5, DARK, "Qwen run-6 · step 5"),
          (OP, ORANGE, "Opus 4.8 (reference)")]
for d, col, lab in series:
    ov = d["overall"]
    axA.fill_between(xs, [p["p25"] for p in ov], [p["p75"] for p in ov], color=col, alpha=0.10, lw=0)
    axA.plot(xs, [p["mean"] for p in ov], color=col, lw=2.8, marker="o", ms=5, mec="white", mew=1.2,
             label=f"{lab}  (n={d['n_worlds']})")
    axA.annotate(f"{ov[-1]['mean']:.2f}", (kmax, ov[-1]["mean"]), textcoords="offset points",
                 xytext=(6, 0), va="center", color=col, fontsize=11, fontweight="bold")
axA.set_xlabel("query index (experiment budget = 15)")
axA.set_ylabel("best-so-far benefit recovered")
axA.set_title("How fast the agent climbs toward the optimal fix", fontweight="bold", color=INK, fontsize=13)
axA.set_ylim(-0.02, 1.10); axA.set_xlim(1, kmax); axA.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
style(axA); axA.legend(frameon=False, fontsize=11, loc="upper left")

# ---- Panel B: final benefit bars ----
labels = ["run-6\nstep 0", "run-6\nstep 5", "Opus 4.8"]
vals = [S0["overall"][-1]["mean"], S5["overall"][-1]["mean"], OP["overall"][-1]["mean"]]
cols = [LIGHT, DARK, ORANGE]
bars = axB.bar(range(3), vals, color=cols, width=0.62)
axB.axhline(1.0, ls="--", lw=2, color=MUTED); axB.text(2.4, 1.01, "gold", ha="right", color=MUTED, fontsize=10, fontweight="bold")
for i, v in enumerate(vals):
    axB.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center",
                 color=cols[i], fontsize=12, fontweight="bold")
axB.set_xticks(range(3)); axB.set_xticklabels(labels)
axB.set_ylabel("final best-so-far benefit")
axB.set_title("Final skill: run-6 flat, below Opus", fontweight="bold", color=INK, fontsize=13)
axB.set_ylim(0, 1.08); style(axB)

fig.suptitle("Run-6 (current go/no-go) vs Opus 4.8 — utility optimization on held-out worlds",
             fontweight="bold", y=1.02, fontsize=14)
fig.tight_layout()
p = f"{BASE}/results_qwen_run6_vs_opus.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
print(f"run-6 step0={vals[0]:.3f}  step5={vals[1]:.3f}  Opus={vals[2]:.3f}  (n: {S0['n_worlds']}/{S5['n_worlds']}/{OP['n_worlds']})")
