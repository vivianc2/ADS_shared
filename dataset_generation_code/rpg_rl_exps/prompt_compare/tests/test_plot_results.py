from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from visualization.plot_results import plot_all  # noqa: E402


class PlotResultsTests(unittest.TestCase):
    def test_plotter_writes_all_required_figures_from_stats_only(self):
        configs = [
            f"p{prompt}_r{reward}" for prompt in range(1, 4) for reward in range(1, 4)
        ]
        archetypes = [
            "confounded_chain", "collider_selection", "hidden_subtype",
            "surrogate_trap", "instrument_only", "competing_causes",
            "synergy_pair", "dose_window", "confounded_reversal",
        ]
        stats = {
            "completeness": {"complete": True},
            "overall": [
                {
                    "config_id": config,
                    "avg_score": 0.5,
                    "best_of_8_score": 0.75,
                    "reward_mean": 0.4,
                    "within_group_reward_variance": 0.02,
                }
                for config in configs
            ],
            "per_archetype": [
                {
                    "config_id": config,
                    "archetype": archetype,
                    "avg_score": 0.5,
                    "avg_part_a": 0.4,
                    "avg_part_b": 0.6,
                }
                for config in configs
                for archetype in archetypes
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats_path = root / "stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            written = plot_all(stats_path)
            self.assertEqual(len(written), 6)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in written))


if __name__ == "__main__":
    unittest.main()

