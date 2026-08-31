#!/usr/bin/env python3
"""Render all required figures using only a run's canonical stats.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROMPTS = ("p1", "p2", "p3")
REWARDS = ("r1", "r2", "r3")


def _save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    fig.savefig(temporary, dpi=180, bbox_inches="tight")
    os.replace(temporary, destination)
    plt.close(fig)


def _annotated_heatmap(data, row_labels, column_labels, title, *, vmin, vmax,
                       colorbar_label, destination: Path, cmap="viridis"):
    array = np.asarray(data, dtype=float)
    width = max(6.0, 0.8 * len(column_labels) + 2.0)
    height = max(4.8, 0.55 * len(row_labels) + 2.0)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(column_labels)), labels=column_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            normalized = (array[row, column] - vmin) / max(vmax - vmin, 1e-12)
            color = "white" if normalized < 0.35 or normalized > 0.75 else "black"
            axis.text(
                column, row, f"{array[row, column]:.3f}",
                ha="center", va="center", color=color, fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    _save(fig, destination)


def _config_matrix(rows, metric: str):
    by_id = {row["config_id"]: row for row in rows}
    return [
        [float(by_id[f"{prompt}_{reward}"][metric]) for reward in REWARDS]
        for prompt in PROMPTS
    ]


def plot_all(stats_path: Path, figures_dir: Path | None = None) -> list[Path]:
    stats_path = Path(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not stats.get("completeness", {}).get("complete"):
        raise ValueError("refusing to plot an incomplete stats.json")
    figures_dir = Path(figures_dir) if figures_dir else stats_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    destination = figures_dir / "overall_avg_score_heatmap.png"
    _annotated_heatmap(
        _config_matrix(stats["overall"], "avg_score"), PROMPTS, REWARDS,
        "Overall average evaluation score", vmin=0.0, vmax=1.0,
        colorbar_label="Average score", destination=destination,
    )
    written.append(destination)

    destination = figures_dir / "overall_best_of_8_heatmap.png"
    _annotated_heatmap(
        _config_matrix(stats["overall"], "best_of_8_score"), PROMPTS, REWARDS,
        "Average per-world best-of-eight evaluation score", vmin=0.0, vmax=1.0,
        colorbar_label="Best-of-eight score", destination=destination,
    )
    written.append(destination)

    config_order = [f"{prompt}_{reward}" for prompt in PROMPTS for reward in REWARDS]
    overall = {row["config_id"]: row for row in stats["overall"]}
    x = np.arange(len(config_order))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, [overall[key]["reward_mean"] for key in config_order], color="#4472c4")
    axes[0].set_ylabel("Candidate reward mean")
    axes[0].set_ylim(-0.25, 1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        [overall[key]["within_group_reward_variance"] for key in config_order],
        color="#ed7d31",
    )
    axes[1].set_ylabel("Mean within-group variance")
    axes[1].set_ylim(0.0, 0.4)
    axes[1].set_xticks(x, labels=config_order, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Candidate reward behavior by configuration")
    fig.tight_layout()
    destination = figures_dir / "reward_summary.png"
    _save(fig, destination)
    written.append(destination)

    archetypes = []
    for row in stats["per_archetype"]:
        if row["archetype"] not in archetypes:
            archetypes.append(row["archetype"])
    by_cell = {
        (row["archetype"], row["config_id"]): row
        for row in stats["per_archetype"]
    }
    for metric, filename, title, label in (
        ("avg_score", "archetype_avg_score_heatmap.png", "Average score by archetype", "Average score"),
        ("avg_part_a", "archetype_avg_part_a_heatmap.png", "Part A by archetype", "Average Part A"),
        ("avg_part_b", "archetype_avg_part_b_heatmap.png", "Part B by archetype", "Average Part B"),
    ):
        data = [
            [float(by_cell[(archetype, config)][metric]) for config in config_order]
            for archetype in archetypes
        ]
        destination = figures_dir / filename
        _annotated_heatmap(
            data, archetypes, config_order, title, vmin=0.0, vmax=1.0,
            colorbar_label=label, destination=destination,
        )
        written.append(destination)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    written = plot_all(args.run_dir / "stats.json", args.run_dir / "figures")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
