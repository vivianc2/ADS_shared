from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from visualization.plot_results import (  # noqa: E402
    REWARD_COLUMNS,
    SCORE_COLUMNS,
    _archetype_prompt_matrix,
    _prompt_score_rows,
    plot_all,
)


class PlotResultsTests(unittest.TestCase):
    def _stats(self):
        configs = [
            f"p{prompt}_r{reward}" for prompt in range(1, 4) for reward in range(1, 4)
        ]
        archetypes = [
            "confounded_chain", "collider_selection", "hidden_subtype",
            "surrogate_trap", "instrument_only", "competing_causes",
            "synergy_pair", "dose_window", "confounded_reversal",
        ]
        prompt_values = {"p1": 0.2, "p2": 0.4, "p3": 0.6}
        return {
            "completeness": {"complete": True},
            "overall": [
                {
                    "config_id": config,
                    "prompt_id": config[:2],
                    "reward_id": config[-2:],
                    "avg_score": prompt_values[config[:2]],
                    "best_of_8_score": prompt_values[config[:2]] + 0.2,
                    "avg_part_a": prompt_values[config[:2]] + 0.1,
                    "avg_part_b": prompt_values[config[:2]] - 0.1,
                    "reward_mean": (
                        prompt_values[config[:2]] + 0.01 * int(config[-1])
                    ),
                    "within_group_reward_variance": 0.01 * int(config[-1]),
                }
                for config in configs
            ],
            "per_archetype": [
                {
                    "config_id": config,
                    "archetype": archetype,
                    "avg_score": prompt_values[config[:2]],
                    "avg_part_a": prompt_values[config[:2]] + 0.1,
                    "avg_part_b": prompt_values[config[:2]] - 0.1,
                }
                for config in configs
                for archetype in archetypes
            ],
        }

    def test_plotter_writes_prompt_only_figures_and_sheets_from_stats_only(self):
        stats = self._stats()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats_path = root / "stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            written = plot_all(stats_path)
            self.assertEqual(
                {path.name for path in written},
                {
                    "prompt_score_summary.png",
                    "prompt_reward_summary.png",
                    "archetype_avg_score_heatmap.png",
                    "archetype_avg_part_a_heatmap.png",
                    "archetype_avg_part_b_heatmap.png",
                },
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in written))

            with (root / "figures" / "prompt_score_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                score_sheet = list(csv.reader(handle))
            self.assertEqual(tuple(score_sheet[0]), SCORE_COLUMNS)
            self.assertEqual([row[0] for row in score_sheet[1:]], ["p1", "p2", "p3"])

            with (root / "figures" / "prompt_reward_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reward_sheet = list(csv.reader(handle))
            self.assertEqual(tuple(reward_sheet[0]), REWARD_COLUMNS)
            self.assertEqual([row[0] for row in reward_sheet[1:]], ["p1", "p2", "p3"])

    def test_prompt_only_views_require_score_metrics_to_match_across_rewards(self):
        stats = self._stats()
        stats["overall"][1]["avg_score"] = 0.9
        with self.assertRaisesRegex(ValueError, "differs across reward functions"):
            _prompt_score_rows(stats["overall"])

        stats = self._stats()
        archetypes = list(dict.fromkeys(
            row["archetype"] for row in stats["per_archetype"]
        ))
        stats["per_archetype"][1]["avg_part_a"] = 0.9
        with self.assertRaisesRegex(ValueError, "differs across reward functions"):
            _archetype_prompt_matrix(
                stats["per_archetype"], archetypes, "avg_part_a"
            )


if __name__ == "__main__":
    unittest.main()
