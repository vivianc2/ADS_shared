from __future__ import annotations

import math
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bootstrap import RPG_V9_DIR, assert_v9_imports, configure_imports  # noqa: E402

configure_imports()

import candidates  # noqa: E402
import env as env_module  # noqa: E402
from candidates import PROMPTS, reward_r2, reward_r3  # noqa: E402
from env import RPGEnv  # noqa: E402
from pipeline import (  # noqa: E402
    _transcript_hash,
    aggregate_run,
    derive_rollout_seed,
    output_path,
    prepare_run,
    run_episode,
    summarize_world_rollouts,
    validate_terminal_schema,
    validate_run,
)
from run_agent_v6 import load_world_file  # noqa: E402
from servers import SamplingSettings  # noqa: E402
from storage import atomic_write_jsonl, completed_final_record  # noqa: E402
from visualization.plot_results import plot_all  # noqa: E402


def _world_path():
    return sorted((RPG_V9_DIR / "rpg_v8_fast_worlds").glob("world_*.json"))[0]


def _fixed_reward(value):
    def reward(struct, world, cat, gold, battery, *, cfg, n_interventions):
        return {
            "reward": float(value),
            "part_a": 0.0,
            "part_b": 0.0,
            "invalid_id_fraction": 0.0,
            "accepted": False,
            "reward_error": False,
        }

    return reward


def _complete_terminal():
    return {
        "schema_version": "prompt_compare_v1",
        "record_type": "terminal",
        "run_id": "test",
        "world_id": "w",
        "archetype": "confounded_chain",
        "config_id": "p1_r1",
        "prompt_id": "p1",
        "reward_id": "r1",
        "rollout_index": 0,
        "request_seed": 1,
        "sampling": {},
        "candidate_reward": 0.0,
        "candidate_part_a": 0.0,
        "candidate_part_b": 0.0,
        "invalid_id_fraction": 0.0,
        "candidate_accepted": False,
        "candidate_reward_error": False,
        "score": 0.0,
        "part_a": 0.0,
        "part_b": 0.0,
        "evaluation_accepted": False,
        "evaluation_error": False,
        "termination_reason": "give_up",
        "intervention_count": 0,
        "experiment_count": 0,
        "turn_count": 1,
        "transcript_sha256": "0" * 64,
        "complete": True,
    }


class PromptCompareTests(unittest.TestCase):
    def test_import_resolution_is_explicitly_v9(self):
        resolved = assert_v9_imports()
        self.assertEqual(set(resolved), {"sampler", "engine", "oracle_v6"})
        self.assertTrue(all(str(RPG_V9_DIR) in source for source in resolved.values()))

    def test_prompt_and_reward_are_injected_per_instance(self):
        world, precomputed = load_world_file(str(_world_path()))
        env = RPGEnv(
            world=world,
            gold=precomputed["gold"],
            battery=precomputed["battery"],
            system_prompt="instance-only prompt",
            reward_fn=_fixed_reward(0.375),
        )
        env.reset()
        _, reward, done, _ = env.step('<action type="give_up">{}</action>')
        self.assertEqual(env.system_prompt, "instance-only prompt")
        self.assertAlmostEqual(reward, 0.375)
        self.assertTrue(done)
        self.assertEqual(env_module.SYSTEM_PROMPT, PROMPTS["p1"])

    def test_concurrent_environments_do_not_leak_prompt_or_reward(self):
        path = str(_world_path())

        def one(index):
            world, precomputed = load_world_file(path)
            env = RPGEnv(
                world=world,
                gold=precomputed["gold"],
                battery=precomputed["battery"],
                system_prompt=f"isolated-{index}",
                reward_fn=_fixed_reward(index / 10.0),
            )
            env.reset()
            _, reward, done, _ = env.step('<action type="give_up">{}</action>')
            return env.system_prompt, reward, done

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, range(8)))
        self.assertEqual(results, [(f"isolated-{i}", i / 10.0, True) for i in range(8)])
        self.assertEqual(env_module.SYSTEM_PROMPT, PROMPTS["p1"])

    def test_candidate_reward_formulas(self):
        base = {
            "reward": 99.0,
            "part_a": 0.7,
            "part_b": 0.2,
            "invalid_id_fraction": 0.4,
            "accepted": False,
        }
        with mock.patch.object(candidates, "compute_reward", return_value=dict(base)):
            self.assertAlmostEqual(reward_r2({}, None, None, None, None)["reward"], 0.6)
            self.assertAlmostEqual(reward_r3({}, None, None, None, None)["reward"], 0.1)

    def test_seed_is_stable_and_excludes_reward_id(self):
        seed = derive_rollout_seed(7_000_000, "world-x", "p2", 3)
        self.assertEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p2", 3))
        self.assertNotEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p3", 3))
        self.assertNotEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p2", 4))
        self.assertIsInstance(seed, int)
        self.assertTrue(0 <= seed < 2_147_483_647)

    def test_metric_definitions_use_population_variance_and_best_of_eight(self):
        values = [
            {
                "candidate_reward": index / 7.0,
                "score": (7 - index) / 7.0,
                "part_a": 0.25,
                "part_b": index / 7.0,
            }
            for index in range(8)
        ]
        summary = summarize_world_rollouts(values)
        expected_rewards = [index / 7.0 for index in range(8)]
        mean = sum(expected_rewards) / 8
        population_variance = sum((value - mean) ** 2 for value in expected_rewards) / 8
        self.assertEqual(summary["n_rollouts"], 8)
        self.assertAlmostEqual(summary["reward_mean"], 0.5)
        self.assertAlmostEqual(summary["reward_variance"], population_variance)
        self.assertAlmostEqual(summary["avg_score"], 0.5)
        self.assertAlmostEqual(summary["best_of_8_score"], 1.0)
        self.assertAlmostEqual(summary["avg_part_a"], 0.25)
        self.assertAlmostEqual(summary["avg_part_b"], 0.5)

    def test_terminal_schema_rejects_missing_or_nonfinite_fields(self):
        validate_terminal_schema(_complete_terminal())
        missing = _complete_terminal()
        missing.pop("part_b")
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_terminal_schema(missing)
        nonfinite = _complete_terminal()
        nonfinite["score"] = math.inf
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_terminal_schema(nonfinite)

    def test_atomic_completion_and_resume_skip_only_complete_files(self):
        world = {"archetype": "confounded_chain", "world_id": "w", "file": "world.json"}
        config = {"config_id": "p1_r1", "prompt_id": "p1", "reward_id": "r1"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = output_path(root, world, config["config_id"], 0)
            atomic_write_jsonl(path, [{"record_type": "turn", "complete": False}])
            self.assertIsNone(completed_final_record(path))

            atomic_write_jsonl(path, [_complete_terminal()])
            self.assertTrue(completed_final_record(path)["complete"])
            self.assertEqual(
                run_episode(root, {"run_id": "test"}, world, config, 0, object()),
                "skipped",
            )
            self.assertFalse(list(path.parent.glob(f".{path.name}.*.tmp")))

            atomic_write_jsonl(path, [{"record_type": "turn", "complete": False}])
            with mock.patch(
                "pipeline.load_world_file",
                side_effect=RuntimeError("incomplete rerun reached loader"),
            ):
                with self.assertRaisesRegex(RuntimeError, "incomplete rerun reached loader"):
                    run_episode(root, {"run_id": "test"}, world, config, 0, object())

    def test_smoke_manifest_persists_one_audited_world_per_archetype(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                runs_root=Path(temporary),
                run_id="unit_smoke",
                resume=False,
                smoke_test=True,
                seed=7_000_000,
                model="Qwen/Qwen3.5-9B",
                served_model_name=None,
                budget=15,
                max_turns=32,
                dtype="bfloat16",
                max_model_len=32768,
                disable_multimodal=True,
                gpus_resolved=("5", "6", "7"),
                ports_resolved=(18005, 18006, 18007),
                host="127.0.0.1",
                gpu_memory_utilization=0.8,
                request_timeout=900,
                transport_retries=3,
                world_max_attempts=1000,
            )
            sampling = SamplingSettings()
            run_dir, manifest = prepare_run(args, sampling)
            self.assertEqual(len(manifest["worlds"]), 9)
            self.assertEqual(
                {entry["archetype"] for entry in manifest["worlds"]},
                set(manifest["science"]["archetypes"]),
            )
            self.assertTrue(all((run_dir / entry["file"]).is_file() for entry in manifest["worlds"]))
            self.assertEqual(manifest["science"]["expected"]["episodes"], 648)

            for config in manifest["science"]["configurations"]:
                for world in manifest["worlds"]:
                    for rollout_index in range(8):
                        seed = derive_rollout_seed(
                            args.seed, world["world_id"], config["prompt_id"], rollout_index
                        )
                        turn = {
                            "schema_version": "prompt_compare_v1",
                            "record_type": "turn",
                            "run_id": manifest["run_id"],
                            "world_id": world["world_id"],
                            "archetype": world["archetype"],
                            "config_id": config["config_id"],
                            "prompt_id": config["prompt_id"],
                            "reward_id": config["reward_id"],
                            "rollout_index": rollout_index,
                            "turn_index": 0,
                            "request_seed": seed,
                            "observation": f"public observation for {world['world_id']}",
                            "raw_model_response": f"{config['prompt_id']} response {rollout_index}",
                            "reasoning_content": None,
                            "parsed_action_type": "give_up",
                            "finish_reason": "stop",
                            "timing": {"latency_s": 0.01},
                            "transport_attempts": 1,
                        }
                        terminal = _complete_terminal()
                        terminal.update({
                            "run_id": manifest["run_id"],
                            "world_id": world["world_id"],
                            "archetype": world["archetype"],
                            "config_id": config["config_id"],
                            "prompt_id": config["prompt_id"],
                            "reward_id": config["reward_id"],
                            "rollout_index": rollout_index,
                            "request_seed": seed,
                            "sampling": manifest["science"]["sampling"],
                            "transcript_sha256": _transcript_hash([turn]),
                        })
                        atomic_write_jsonl(
                            output_path(run_dir, world, config["config_id"], rollout_index),
                            [turn, terminal],
                        )

            stats = aggregate_run(run_dir)
            self.assertEqual(len(stats["per_world"]), 81)
            self.assertEqual(len(stats["per_archetype"]), 81)
            self.assertEqual(len(stats["overall"]), 9)
            self.assertEqual(stats["pairing"]["warning_count"], 0)
            validate_run(run_dir, require_figures=False)
            self.assertEqual(len(plot_all(run_dir / "stats.json")), 6)
            self.assertEqual(validate_run(run_dir)["episodes"], 648)

            args.resume = True
            _, resumed = prepare_run(args, sampling)
            self.assertEqual(resumed["science_fingerprint"], manifest["science_fingerprint"])


if __name__ == "__main__":
    unittest.main()
