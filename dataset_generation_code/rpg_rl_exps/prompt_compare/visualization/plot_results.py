#!/usr/bin/env python3
"""Render all required figures using only a run's canonical stats.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROMPTS = ("p1", "p2", "p3")
REWARDS = ("r1", "r2", "r3")
SCORE_COLUMNS = ("config", "avg_score", "part A", "part B", "best-of-8")
REWARD_COLUMNS = (
    "config",
    "r1", "r1 in-grp var",
    "r2", "r2 in-grp var",
    "r3", "r3 in-grp var",
)


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
    rotate_labels = len(column_labels) > len(PROMPTS)
    axis.set_xticks(
        range(len(column_labels)),
        labels=column_labels,
        rotation=45 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
    )
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


def _require_reward_invariant(values, label: str) -> float:
    numbers = [float(value) for value in values]
    reference = numbers[0]
    if any(
        not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-12)
        for value in numbers[1:]
    ):
        raise ValueError(f"{label} unexpectedly differs across reward functions")
    return reference


def _prompt_score_rows(rows):
    by_id = {row["config_id"]: row for row in rows}
    output = []
    for prompt in PROMPTS:
        prompt_rows = [by_id[f"{prompt}_{reward}"] for reward in REWARDS]
        output.append([
            prompt,
            _require_reward_invariant(
                [row["avg_score"] for row in prompt_rows], f"{prompt}.avg_score"
            ),
            _require_reward_invariant(
                [row["avg_part_a"] for row in prompt_rows], f"{prompt}.avg_part_a"
            ),
            _require_reward_invariant(
                [row["avg_part_b"] for row in prompt_rows], f"{prompt}.avg_part_b"
            ),
            _require_reward_invariant(
                [row["best_of_8_score"] for row in prompt_rows],
                f"{prompt}.best_of_8_score",
            ),
        ])
    return output


def _prompt_reward_rows(rows):
    by_id = {row["config_id"]: row for row in rows}
    output = []
    for prompt in PROMPTS:
        values = [prompt]
        for reward in REWARDS:
            row = by_id[f"{prompt}_{reward}"]
            values.extend([
                float(row["reward_mean"]),
                float(row["within_group_reward_variance"]),
            ])
        output.append(values)
    return output


def _archetype_prompt_matrix(rows, archetypes, metric: str):
    by_cell = {
        (row["archetype"], row["config_id"]): row
        for row in rows
    }
    return [
        [
            _require_reward_invariant(
                [
                    by_cell[(archetype, f"{prompt}_{reward}")][metric]
                    for reward in REWARDS
                ],
                f"{archetype}.{prompt}.{metric}",
            )
            for prompt in PROMPTS
        ]
        for archetype in archetypes
    ]


def _write_csv_sheet(destination: Path, columns, rows) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(temporary, destination)


def _render_sheet(columns, rows, title: str, destination: Path) -> None:
    width = max(8.0, 1.65 * len(columns))
    fig, axis = plt.subplots(figsize=(width, 2.7))
    axis.axis("off")
    display_rows = [
        [value if isinstance(value, str) else f"{float(value):.4f}" for value in row]
        for row in rows
    ]
    table = axis.table(
        cellText=display_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 0.78),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#4472c4")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6fa")
    axis.set_title(title, fontsize=14, weight="bold", pad=12)
    _save(fig, destination)


def plot_all(stats_path: Path, figures_dir: Path | None = None) -> list[Path]:
    stats_path = Path(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not stats.get("completeness", {}).get("complete"):
        raise ValueError("refusing to plot an incomplete stats.json")
    figures_dir = Path(figures_dir) if figures_dir else stats_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    score_rows = _prompt_score_rows(stats["overall"])
    reward_rows = _prompt_reward_rows(stats["overall"])

    destination = figures_dir / "prompt_score_summary.png"
    _render_sheet(
        SCORE_COLUMNS, score_rows, "Evaluation metrics by prompt", destination
    )
    written.append(destination)
    _write_csv_sheet(figures_dir / "prompt_score_summary.csv", SCORE_COLUMNS, score_rows)

    destination = figures_dir / "prompt_reward_summary.png"
    _render_sheet(
        REWARD_COLUMNS, reward_rows, "Reward behavior by prompt", destination
    )
    written.append(destination)
    _write_csv_sheet(figures_dir / "prompt_reward_summary.csv", REWARD_COLUMNS, reward_rows)

    archetypes = []
    for row in stats["per_archetype"]:
        if row["archetype"] not in archetypes:
            archetypes.append(row["archetype"])
    for metric, filename, title, label in (
        ("avg_score", "archetype_avg_score_heatmap.png", "Average score by archetype", "Average score"),
        ("avg_part_a", "archetype_avg_part_a_heatmap.png", "Part A by archetype", "Average Part A"),
        ("avg_part_b", "archetype_avg_part_b_heatmap.png", "Part B by archetype", "Average Part B"),
    ):
        data = _archetype_prompt_matrix(stats["per_archetype"], archetypes, metric)
        destination = figures_dir / filename
        _annotated_heatmap(
            data, archetypes, PROMPTS, title, vmin=0.0, vmax=1.0,
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
