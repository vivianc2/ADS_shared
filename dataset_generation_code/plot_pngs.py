"""Matplotlib PNG versions of the trajectory + belief-convergence charts.

Reads trajectories.json and/or beliefs_scored.json (any number of models for the
trajectory overlay) and writes PNGs. Run in the ADS-rpg env (has matplotlib).

Run:  conda run -n ADS-rpg python plot_pngs.py <results_dir>
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = sys.argv[1] if len(sys.argv) > 1 else "results_v8_validation_opus"
# validated palette (light surface)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948",
     "muted": "#898781", "grid": "#e1e0d9", "ink": "#0b0b0b", "ink2": "#52514e"}
SERIES = [C["blue"], C["orange"], C["aqua"], C["red"]]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink2"], "text.color": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "figure.dpi": 130})


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=C["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def trajectory(models):
    kmax = models[0]["kmax"]
    xs = list(range(1, kmax + 1))
    # overall
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(1.0, ls="--", lw=2, color=C["muted"], zorder=1)
    ax.text(kmax, 1.008, "gold = 1.0", ha="right", va="bottom", color=C["muted"], fontsize=10, fontweight="bold")
    for i, m in enumerate(models):
        ov = m["overall"]; col = SERIES[i % len(SERIES)]
        ax.fill_between(xs, [d["p25"] for d in ov], [d["p75"] for d in ov], color=col, alpha=0.12, lw=0)
        ax.plot(xs, [d["mean"] for d in ov], color=col, lw=2.4, marker="o", ms=5,
                mec="white", mew=1.2, label=f"{m['model']} (n={m['n_worlds']})")
    ax.set_xlabel("query index (interventions + measurements, budget 15)")
    ax.set_ylabel("best-so-far benefit recovered")
    ax.set_title("How fast the scientist agent optimizes utility", fontweight="bold", color=C["ink"])
    ax.set_ylim(-0.02, 1.08); ax.set_xlim(1, kmax); ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    style(ax); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); p = os.path.join(RD, "trajectory_overall.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)
    # small multiples
    arches = sorted(models[0]["by_arch"])
    fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True, sharey=True)
    for a, ax in zip(arches, axs.ravel()):
        ax.axhline(1.0, ls="--", lw=1.4, color=C["muted"])
        for i, m in enumerate(models):
            c = m["by_arch"].get(a)
            if c:
                ax.plot(xs, [d["mean"] for d in c], color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=3, mec="white", mew=0.8)
        ax.set_title(a, fontsize=10.5, fontweight="bold"); ax.set_ylim(-0.02, 1.08); ax.set_xticks([1, 5, 10, 15]); style(ax)
    for ax in axs[-1]:
        ax.set_xlabel("query")
    for ax in axs[:, 0]:
        ax.set_ylabel("best-so-far benefit")
    fig.suptitle("Utility optimization by archetype (best-so-far benefit vs query)", fontweight="bold", y=0.995)
    fig.tight_layout(); p = os.path.join(RD, "trajectory_by_archetype.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


def belief(S):
    kmax = min(15, S["kmax"]); xs = list(range(1, kmax + 1))
    comps = [("graph", C["blue"], "graph_score", True), ("cause", C["aqua"], "cause correct", False),
             ("proxy", C["orange"], "proxy correct", False), ("decoy", C["red"], "decoy F1", False)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for key, col, lab, band in comps:
        arr = S["overall"][key][:kmax]
        if band:
            ax.fill_between(xs, [d["p25"] for d in arr], [d["p75"] for d in arr], color=col, alpha=0.10, lw=0)
        ax.plot(xs, [d["mean"] for d in arr], color=col, lw=2.4 if band else 1.8,
                marker="o", ms=4 if band else 3, mec="white", mew=1, label=lab, alpha=1 if band else 0.9)
    ax.set_xlabel("action-turn ordinal (agent decisions in order)")
    ax.set_ylabel("mean score vs true SCM")
    ax.set_title("Reasoning-graph convergence to the true SCM", fontweight="bold", color=C["ink"])
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(1, kmax); style(ax); ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(); p = os.path.join(RD, "belief_convergence.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)
    # per-archetype final cause/proxy/decoy grouped bars
    arches = sorted(S["by_arch"])
    import numpy as np
    x = np.arange(len(arches)); wdt = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for j, (key, col, lab) in enumerate([("cause", C["aqua"], "cause"), ("proxy", C["orange"], "proxy"), ("decoy", C["red"], "decoy F1")]):
        vals = [S["by_arch"][a][key][-1]["mean"] for a in arches]
        ax.bar(x + (j - 1) * wdt, vals, wdt, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(arches, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("final mean score"); ax.set_ylim(0, 1)
    ax.set_title("Final reasoning-graph accuracy by archetype", fontweight="bold")
    style(ax); ax.legend(frameon=False)
    fig.tight_layout(); p = os.path.join(RD, "belief_by_archetype.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    tj = os.path.join(RD, "trajectories.json")
    bs = os.path.join(RD, "beliefs_scored.json")
    if os.path.exists(tj):
        trajectory([json.load(open(tj))])
    if os.path.exists(bs):
        belief(json.load(open(bs)))
