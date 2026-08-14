"""Qwen eval rollouts vs Opus: utility-optimization, grouped by RUN provenance.

IMPORTANT: /home/ray/exports/dumped_evals keys dumps by step number, so it MIXES runs
(each run only overwrites its own eval steps). Verified by mtime:
  - run-6 (current go/no-go, Aug-13): steps 0, 5
  - prior run (Aug-12, eval every 2): steps 2, 4, 6, 8, 10
  - old run (Aug-11): step 20
So we do NOT draw one continuous training curve — points are colored/connected by run.

Run:  conda run -n ADS-rpg python plot_progression.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code"
RUN_OF = {0: "run6", 5: "run6", 2: "priorAug12", 4: "priorAug12", 6: "priorAug12",
          8: "priorAug12", 10: "priorAug12", 20: "oldAug11"}
RUN_STYLE = {"run6": ("#256abf", "-", "o", "run-6 (current go/no-go)"),
             "priorAug12": ("#9ec5f4", "--", "s", "prior run (Aug-12)"),
             "oldAug11": ("#d03b3b", ":", "X", "old run (Aug-11)")}
ORANGE, MUTED, GRID, INK, INK2 = "#eb6834", "#898781", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 130})


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9); ax.set_axisbelow(True)


steps = [s for s in RUN_OF if os.path.exists(f"{BASE}/results_qwen_step{s}/trajectories.json")]
qwen = {s: json.load(open(f"{BASE}/results_qwen_step{s}/trajectories.json")) for s in steps}
opus = json.load(open(f"{BASE}/results_v8_validation_opus/trajectories.json"))
kmax = opus["kmax"]; xs = list(range(1, kmax + 1))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.4, 1]})

# Panel A: trajectories, colored by run (alpha encodes step within run)
axA.axhline(1.0, ls="--", lw=2, color=MUTED)
axA.text(kmax, 1.008, "gold = 1.0", ha="right", va="bottom", color=MUTED, fontsize=10, fontweight="bold")
seen_runs = set()
for s in sorted(qwen):
    run = RUN_OF[s]; col, ls, mk, lab = RUN_STYLE[run]
    d = qwen[s]
    lbl = lab if run not in seen_runs else None
    seen_runs.add(run)
    axA.plot(xs, [p["mean"] for p in d["overall"]], color=col, lw=2 if run == "run6" else 1.5,
             ls=ls, alpha=0.55 + 0.45 * (list(sorted(qwen)).index(s) % 3) / 2, label=lbl)
axA.plot(xs, [p["mean"] for p in opus["overall"]], color=ORANGE, lw=2.8, marker="o", ms=4,
         mec="white", mew=1, label=f"Opus 4.8 (ref, n={opus['n_worlds']})")
axA.set_xlabel("query index (budget 15)"); axA.set_ylabel("best-so-far benefit recovered")
axA.set_title("Utility optimization (Qwen eval rollouts, by run)", fontweight="bold", color=INK)
axA.set_ylim(-0.02, 1.08); axA.set_xlim(1, kmax); axA.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
style(axA); axA.legend(frameon=False, fontsize=9, loc="upper left")

# Panel B: final benefit vs step, connected only WITHIN a run
axB.axhline(opus["overall"][-1]["mean"], ls="--", lw=2, color=ORANGE, label="Opus 4.8 (ref)")
by_run = {}
for s in sorted(qwen):
    by_run.setdefault(RUN_OF[s], []).append((s, qwen[s]["overall"][-1]["mean"]))
for run, pts in by_run.items():
    col, ls, mk, lab = RUN_STYLE[run]
    pts.sort()
    axB.plot([p[0] for p in pts], [p[1] for p in pts], color=col, ls=ls, marker=mk, ms=7,
             mec="white", mew=1.2, lw=2 if run == "run6" else 1.5, label=lab)
axB.annotate("collapsed\n(truncated,\nold run)", (20, by_run["oldAug11"][0][1]), textcoords="offset points",
             xytext=(-8, 6), ha="right", color="#d03b3b", fontsize=8.5, fontweight="bold")
axB.set_xlabel("eval step (NOT one run — see legend)"); axB.set_ylabel("final best-so-far benefit")
axB.set_title("Final optimization skill by eval step", fontweight="bold", color=INK)
axB.set_ylim(0, max(0.6, opus["overall"][-1]["mean"] + 0.08)); style(axB); axB.legend(frameon=False, fontsize=8.5)

fig.suptitle("Qwen utility optimization across eval dumps (grouped by run) vs Opus 4.8",
             fontweight="bold", y=1.02)
fig.tight_layout(); p = f"{BASE}/results_qwen_progression.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
for run, pts in by_run.items():
    print(f"  {run}: " + ", ".join(f"step{s}={v:.2f}" for s, v in sorted(pts)))
print(f"  Opus ref = {opus['overall'][-1]['mean']:.2f}")
