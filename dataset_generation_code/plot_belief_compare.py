"""Overlay reasoning-graph convergence for multiple models/checkpoints (matplotlib PNG).

Usage: conda run -n ADS-rpg python plot_belief_compare.py OUT.png LABEL1:scored1.json LABEL2:scored2.json ...
Panel A: graph_score vs action-turn ordinal (one line per model). Panel B: final
cause/proxy/decoy grouped bars per model.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
SERIES = [("#eb6834"), ("#9ec5f4"), ("#184f95"), ("#1baf7a")]  # opus orange, then blues
MUTED, GRID, INK, INK2 = "#898781", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 130})
models = []
for a in sys.argv[2:]:
    lab, path = a.split(":", 1)
    models.append((lab, json.load(open(path))))


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9); ax.set_axisbelow(True)


kmax = min(15, models[0][1]["kmax"]); xs = list(range(1, kmax + 1))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.4, 1]})
for i, (lab, S) in enumerate(models):
    arr = S["overall"]["graph"][:kmax]; col = SERIES[i % len(SERIES)]
    axA.fill_between(xs, [d["p25"] for d in arr], [d["p75"] for d in arr], color=col, alpha=0.08, lw=0)
    axA.plot(xs, [d["mean"] for d in arr], color=col, lw=2.6, marker="o", ms=4, mec="white", mew=1,
             label=f"{lab} (n={S['n_worlds']})")
axA.set_xlabel("action-turn ordinal (agent decisions in order)")
axA.set_ylabel("graph_score vs true SCM")
axA.set_title("Reasoning-graph convergence", fontweight="bold", color=INK)
axA.set_ylim(-0.02, 0.7); axA.set_xlim(1, kmax); style(axA); axA.legend(frameon=False, fontsize=9, loc="upper left")

comps = ["cause", "proxy", "decoy"]; x = np.arange(len(comps)); wdt = 0.8 / len(models)
for i, (lab, S) in enumerate(models):
    vals = [S["overall"][c][-1]["mean"] for c in comps]
    axB.bar(x + (i - (len(models) - 1) / 2) * wdt, vals, wdt, color=SERIES[i % len(SERIES)], label=lab)
axB.set_xticks(x); axB.set_xticklabels(["cause", "proxy", "decoy F1"])
axB.set_ylabel("final mean score"); axB.set_ylim(0, 1)
axB.set_title("Final graph accuracy by component", fontweight="bold", color=INK)
style(axB); axB.legend(frameon=False, fontsize=9)
fig.suptitle("Reasoning-graph reconstruction: Qwen (RL checkpoints) vs Opus 4.8", fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
for lab, S in models:
    print(f"  {lab}: final graph={S['overall']['graph'][-1]['mean']:.2f} cause={S['overall']['cause'][-1]['mean']:.2f} "
          f"proxy={S['overall']['proxy'][-1]['mean']:.2f} decoy={S['overall']['decoy'][-1]['mean']:.2f}")
