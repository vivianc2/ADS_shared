#!/usr/bin/env python3
"""Figure: part A on standard vs cue-free worlds, paired (validation 269 / test 539). 2026-10-23.
Inputs: ~/rpgrl/rl_v10/scripts/cuefree_vs_standard{,_test_cuefree}.csv (compare_cuefree.py) and the base-9B evals.
Output: personal_docs/rl_v10/figures/fig_rl_cuefree.{png,pdf}"""
import csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/rpgrl/rl_v10")
OUT = os.path.expanduser("~/ADS_folder/personal_docs/rl_v10/figures/fig_rl_cuefree")
BLUE, ORANGE, INK, INK2, GRID, LINE = "#2A78D6", "#EB6834", "#14213D", "#3D4A5C", "#E4E3DF", "#A9A8A3"
ROWS = [("oracle_levers@15", "told the levers"), ("blind_grid@120", "blind grid, 120 exp."),
        ("nocd_corr", "|r| ranking + tests"), ("nocd_mi", "MI ranking + tests"),
        ("cd_pc_rcit_ivbase", "PC causal discovery"), ("nocd_corr_screen", "|r| + group screen"),
        ("informed@15", "informed script"), ("nocd_random", "random level controls"), ("random@15", "random settings")]


def load(fn):
    return {r["policy"]: (float(r["standard"]), float(r["cuefree"])) for r in csv.DictReader(open(fn))}


def base9b():
    def pw(fn, split=None):
        d = collections.defaultdict(list)
        for r in csv.DictReader(open(fn)):
            if split is None or r["split"] == split:
                d[r["world_id"]].append(float(r["part_a"]))
        return {w: np.mean(v) for w, v in d.items()}
    a = pw(f"{R}/evals/base9b_full/episodes.csv", "validation"); b = pw(f"{R}/evals/base9b_full_cuefree/episodes.csv")
    ws = sorted(set(a) & set(b))
    return np.mean([a[w] for w in ws]), np.mean([b[w] for w in ws])


def main():
    panels = [("Validation (269 paired worlds)", load(f"{R}/scripts/cuefree_vs_standard.csv"), True),
              ("Test (539 paired worlds)", load(f"{R}/scripts/cuefree_vs_standard_test_cuefree.csv"), False)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, (title, d, with9b) in zip(axes, panels):
        rows = [(lab, *d[k]) for k, lab in ROWS if k in d]
        rows.append(("Qwen3.5-9B base (n = 4)", *base9b()) if with9b else ("Qwen3.5-9B base (not run on test)", np.nan, np.nan))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, s, c) in zip(y, rows):
            if np.isnan(s):
                continue
            ax.plot([s * 100, c * 100], [yi, yi], color=LINE, lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([s * 100], [yi], s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
            ax.scatter([c * 100], [yi], s=70, color=ORANGE, zorder=3, edgecolor="white", linewidth=2)
            if abs(s - c) >= 0.05:          # label only the rows that move
                ax.text(c * 100 + (-2.5 if c < s else 2.5), yi + 0.28, f"{c * 100:.0f}%", color=INK2, fontsize=9,
                        ha="right" if c < s else "left")
                ax.text(s * 100 + (2.5 if c < s else -2.5), yi + 0.28, f"{s * 100:.0f}%", color=INK2, fontsize=9,
                        ha="left" if c < s else "right")
        ax.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK)
        ax.set_xlim(-3, 100); ax.set_xlabel("fix rate (part A), %", color=INK2, fontsize=10)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="standard worlds"),
         plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="cue-free worlds (same worlds; inert settings follow the hidden cause). One dot = no change.")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
