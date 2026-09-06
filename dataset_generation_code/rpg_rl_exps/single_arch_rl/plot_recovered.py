#!/usr/bin/env python3
"""Two figures for the two tables that carry the `sarl_v1` story.

Reads only ``recovered/<run>/*.csv`` (produced by ``recover_history.py``), so the
pictures and the numbers can never disagree -- and every value in them is one the run
actually wrote down.

    figure_1_training_signal.png   the GRPO signal, all 8 optimizer steps
    figure_2_eval_heatmap.png      held-out score, 9 archetypes x 5 eval steps

    bash scripts/in_container.sh python -m single_arch_rl.plot_recovered

Design notes, so the choices are auditable rather than taste:

* **Two measures of different scale never share an axis.** The training figure is
  small multiples -- one panel per measure, each with its own y -- not a dual-axis plot.
* **Every y starts at 0.** The story is that these curves are flat; a zoomed axis would
  manufacture a trend out of noise.
* **The heatmap is the right form for 9 classes.** Past ~7 categories, more colors stop
  separating, so the form is a table-plus-chart: one sequential hue for magnitude with
  the value printed in every cell.
* **Colors are the documented palette, validated, not eyeballed.** Slot-1 blue for the
  single series; the blue 100->700 sequential ramp for the heatmap. In-cell text flips
  between ink and white by measured WCAG contrast against its own cell.
* Labels are English: the container ships no CJK font, so any Chinese in a figure would
  render as tofu.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from matplotlib.patches import Rectangle  # noqa: E402

_HERE = Path(__file__).resolve().parent

# --- palette (references/palette.md) ----------------------------------------------------
# Validated with scripts/validate_palette.py:
#   "#2a78d6" --mode light                      -> ALL CHECKS PASS
#   the sequential ramp below --ordinal --mode light -> monotone L, adjacent dL, single
#   hue (4 deg spread) all PASS. Its one FAIL, the 2:1 light-end floor, is an *ordinal*
#   rule; palette.md scopes the full 100->700 range to sequential encoding -- heatmaps
#   named explicitly -- "where the lightest step means near zero and is allowed to
#   recede toward the surface". Every cell carries its printed value and the CSV is the
#   table view, which is the relief the contrast rule asks for.
LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink": "#ffffff",
    "ink_2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": "#3987e5",
}
#: Blue sequential ramp, steps 100 -> 700 (palette.md).
RAMP_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Dark mode flips the anchor (palette.md: "flips anchor in dark"): near-zero is the
#: step that recedes toward the surface, so on the dark surface the ramp runs
#: dark -> light as the value grows. Written out rather than derived from RAMP_LIGHT --
#: a reversed() of an already-descending list silently reproduces the light order.
RAMP_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

#: The figure is authored at 1x and rendered at 2x, so a "2px" spec is 2 * 72/144 pt.
DPI = 200
PX = 72 / 100.0  # one logical px, in points, at the 100-dpi logical scale


def _srgb_to_lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (_srgb_to_lin(int(h[i : i + 2], 16) / 255) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


#: In-fill label candidates. Deliberately mode-independent: a label inside a colored
#: cell sits on the *fill*, not on the page, so the fill's luminance decides and the
#: theme's ink token has no say. Using the dark theme's ink (white) as one of only two
#: options left dark-mode midtones at 4.21:1; offering the palette's darkest ink to both
#: modes lifts that to 4.45:1 and leaves light mode unchanged at 4.50:1.
IN_FILL_INK = ("#ffffff", "#0b0b0b")


def ink_on(fill: str, theme: Dict[str, str]) -> str:
    """White or ink inside a colored fill -- whichever measurably wins on that cell.

    marks-and-anatomy.md: a label set inside a colored fill picks white or ink by the
    fill's luminance so it always clears contrast. Measure it rather than guess.
    """
    return max(IN_FILL_INK, key=lambda c: contrast(c, fill))


# --- data -------------------------------------------------------------------------------


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python -m single_arch_rl.recover_history` first")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


# --- figure 1: the training signal -------------------------------------------------------

#: (column, panel title, unit note, y-axis headroom factor)
PANELS = (
    ("reward/group_reward_mean", "Mean reward per group", "reward/group_reward_mean"),
    ("reward/group_reward_var", "Within-group variance", "reward/group_reward_var"),
    ("policy/grad_norm", "Gradient norm", "policy/grad_norm"),
)


def figure_training(rows: List[Dict[str, str]], theme: Dict[str, str], out: Path) -> None:
    steps = [int(r["global_step"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=DPI)
    fig.patch.set_facecolor(theme["page"])

    for index, (ax, (column, title, keyname)) in enumerate(zip(axes, PANELS)):
        values = [as_float(r, column) for r in rows]
        ax.set_facecolor(theme["surface"])

        # Panel 1 carries the within-group spread as a wash around the mean: the point
        # of the whole figure is that the step-to-step drift is small *relative to it*.
        if index == 0:
            spread = [math.sqrt(as_float(r, "reward/group_reward_var") or 0.0) for r in rows]
            lo = [v - s for v, s in zip(values, spread)]
            hi = [v + s for v, s in zip(values, spread)]
            ax.fill_between(steps, lo, hi, color=theme["series"], alpha=0.10, linewidth=0)
            # Direct-label the band instead of a legend box: the panel title already
            # names the line, so a two-row legend would restate it and collide with the
            # top gridline. The band is the only element that still needs naming.
            ax.annotate(
                "±1 within-group SD",
                xy=(steps[1], hi[1]),
                xytext=(4, -13),
                textcoords="offset points",
                color=theme["ink_2"],
                fontsize=8.5,
            )
            top = max(hi) * 1.20
        else:
            top = max(v for v in values if v is not None) * 1.35

        ax.grid(True, axis="y", color=theme["grid"], linewidth=1.0 * PX, zorder=0)
        ax.set_axisbelow(True)
        ax.plot(
            steps,
            values,
            color=theme["series"],
            linewidth=2 * PX,
            marker="o",
            markersize=8 * PX,
            markerfacecolor=theme["series"],
            # The surface ring, so markers stay legible where they crowd the line.
            markeredgecolor=theme["surface"],
            markeredgewidth=2 * PX,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
            clip_on=False,
        )
        # Label the endpoint only -- a number on every point goes unread.
        ax.annotate(
            f"{values[-1]:.4f}".rstrip("0") if index != 2 else f"{values[-1]:.4f}",
            xy=(steps[-1], values[-1]),
            xytext=(7, 6),
            textcoords="offset points",
            color=theme["ink_2"],
            fontsize=9,
        )

        # pad clears the metric-key line below it; the two collide at a smaller pad.
        ax.set_title(title, color=theme["ink"], fontsize=11.5, pad=26, loc="left")
        ax.text(
            0,
            1.022,
            keyname,
            transform=ax.transAxes,
            color=theme["muted"],
            fontsize=8.5,
            va="bottom",
        )
        ax.set_xlabel("optimizer step", color=theme["muted"], fontsize=9)
        # Room to the right of step 8 for the endpoint label, which would otherwise be
        # clipped by the figure edge on the last panel.
        ax.set_xlim(0.6, 9.1)
        ax.set_xticks(steps)
        ax.set_ylim(0, top)
        ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(theme["axis"])
            spine.set_linewidth(1.0 * PX)

    fig.suptitle(
        "sarl_v1 / easy — the GRPO signal never collapsed, and never moved",
        color=theme["ink"],
        fontsize=14,
        x=0.011,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.011,
        0.905,
        "8 optimizer steps · 32 worlds x 8 samples = 256 episodes each · 100% of groups "
        "had non-zero spread at every step",
        color=theme["ink_2"],
        fontsize=9.5,
        ha="left",
    )
    fig.text(
        0.011,
        0.022,
        "Recovered from logs/train_20260903_123951.log; W&B kept only steps 7-8. "
        "Group statistics are printed to 4 dp, so panels 1-2 are exact to that.",
        color=theme["muted"],
        fontsize=8,
        ha="left",
    )
    fig.subplots_adjust(left=0.048, right=0.972, top=0.735, bottom=0.16, wspace=0.22)
    fig.savefig(out, dpi=DPI, facecolor=theme["page"])
    plt.close(fig)
    print(f"  wrote {out.name}")


# --- figure 2: held-out score, archetype x step ------------------------------------------


def figure_eval(
    arch_rows: List[Dict[str, str]],
    overall_rows: List[Dict[str, str]],
    theme: Dict[str, str],
    ramp: Sequence[str],
    trained: str,
    out: Path,
) -> None:
    steps = sorted({int(r["global_step"]) for r in arch_rows})
    archetypes = sorted({r["archetype"] for r in arch_rows})
    score: Dict[Tuple[str, int], float] = {
        (r["archetype"], int(r["global_step"])): float(r["avg_score"]) for r in arch_rows
    }
    pooled = {int(r["global_step"]): float(r["eval/all/avg_score"]) for r in overall_rows}

    # Order rows by magnitude so the grid reads top-to-bottom, the way a sorted table
    # would. (The colour already encodes magnitude; the order just stops it looking
    # scattered.)
    archetypes.sort(key=lambda a: -sum(score[(a, s)] for s in steps) / len(steps))

    cmap = LinearSegmentedColormap.from_list("blue_seq", list(ramp))
    top = max(max(score.values()), max(pooled.values()))
    norm = Normalize(vmin=0.0, vmax=math.ceil(top * 20) / 20)

    n_rows = len(archetypes)
    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=DPI)
    fig.patch.set_facecolor(theme["page"])
    ax.set_facecolor(theme["surface"])

    gap = 2 * PX / 72  # the 2px surface gap, in data units scaled below
    inset = 0.028  # half the surface gap, as a fraction of a cell

    def cell(col: int, row: float, value: float) -> None:
        fill = cmap(norm(value))
        fill_hex = matplotlib.colors.to_hex(fill)
        ax.add_patch(
            Rectangle(
                (col + inset, row + inset),
                1 - 2 * inset,
                1 - 2 * inset,
                facecolor=fill_hex,
                edgecolor="none",
            )
        )
        ax.text(
            col + 0.5,
            row + 0.5,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=ink_on(fill_hex, theme),
        )

    for r, archetype in enumerate(archetypes):
        y = n_rows - 1 - r
        for c, step in enumerate(steps):
            cell(c, y + 1.0, score[(archetype, step)])

    # The pooled row is an aggregate of the nine above it, so it sits below a gap
    # rather than pretending to be a tenth archetype.
    for c, step in enumerate(steps):
        cell(c, 0.0 - 0.18, pooled[step])

    labels = []
    for archetype in archetypes:
        labels.append(f"{archetype}  ▲" if archetype == trained else archetype)

    ax.set_xlim(0, len(steps))
    ax.set_ylim(-0.30, n_rows + 1.0)
    ax.set_xticks([c + 0.5 for c in range(len(steps))])
    ax.set_xticklabels([f"step {s}" for s in steps], fontsize=10)
    ax.set_yticks([n_rows - r + 0.5 for r in range(n_rows)] + [0.32])
    ax.set_yticklabels(labels + ["ALL (90 episodes)"], fontsize=10)
    # tick_params repaints every label, so it has to run BEFORE the per-label colors
    # below -- the other order silently flattens the emphasis back to muted.
    ax.tick_params(colors=theme["muted"], length=0, labelsize=10)
    for tick, archetype in zip(ax.get_yticklabels(), archetypes + ["ALL"]):
        emphasised = archetype in (trained, "ALL")
        tick.set_color(theme["ink"] if emphasised else theme["ink_2"])
        if emphasised:
            tick.set_fontweight("bold")
    ax.xaxis.set_ticks_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.030,
        pad=0.02,
    )
    bar.set_label("avg_score", color=theme["ink_2"], fontsize=9, labelpad=8)
    bar.ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
    bar.outline.set_visible(False)

    fig.suptitle(
        "sarl_v1 / easy — held-out score by archetype, at every evaluation",
        color=theme["ink"],
        fontsize=14,
        x=0.012,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.012,
        0.918,
        "45 validation worlds x 2 samples · 10 episodes per archetype per step — "
        "cell-to-cell moves of this size are mostly sampling noise",
        color=theme["ink_2"],
        fontsize=9,
        ha="left",
    )
    fig.text(
        0.012,
        0.028,
        f"▲ {trained} is the only archetype this run trained on.  "
        "Recovered in full from exports/dumped_evals/ — independent of W&B.",
        color=theme["muted"],
        fontsize=8,
        ha="left",
    )
    fig.subplots_adjust(left=0.205, right=0.945, top=0.845, bottom=0.075)
    fig.savefig(out, dpi=DPI, facecolor=theme["page"])
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="easy")
    parser.add_argument("--dir", default=str(_HERE / "recovered"), help="recover_history output")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    base = Path(args.dir) / args.run
    theme = DARK if args.dark else LIGHT
    ramp = RAMP_DARK if args.dark else RAMP_LIGHT
    suffix = "_dark" if args.dark else ""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # DejaVu is what the container ships; the named system faces come first so
            # the same script picks up a real UI sans anywhere else.
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    print(f"reading {base}")
    figure_training(read_csv(base / "train_by_step.csv"), theme, base / f"figure_1_training_signal{suffix}.png")
    figure_eval(
        read_csv(base / "eval_by_archetype.csv"),
        read_csv(base / "eval_overall_by_step.csv"),
        theme,
        ramp,
        trained="dose_window",
        out=base / f"figure_2_eval_heatmap{suffix}.png",
    )


if __name__ == "__main__":
    main()
