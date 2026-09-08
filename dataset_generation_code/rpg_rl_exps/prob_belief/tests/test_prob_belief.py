from __future__ import annotations

import json
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import run_experiment as runner  # noqa: E402


class _NeverRaisedInputError(Exception):
    pass


class _NeverRaisedContextError(Exception):
    pass


class _FakeClient:
    input_length_error = _NeverRaisedInputError
    context_length_error = _NeverRaisedContextError

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def generate(self, messages, seed):
        self.calls.append((messages, seed))
        return SimpleNamespace(
            raw_text=self.raw,
            action_text=self.raw,
            reasoning_content=None,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 8},
            attempts=1,
            latency_s=0.01,
            prompt_tokens=100,
        )


class _FakeEnv:
    def __init__(self):
        self.turns = []

    def reset(self):
        return "catalog-bearing observation"

    def step(self, text):
        self.turns.append({
            "action_type": "answer",
            "answer_struct": {"actions": [{"actuator": "a0", "value": 1}]},
        })
        return "(episode complete)", 0.375, True, {
            "reward": 0.375,
            "part_a": 0.5,
            "part_b": 0.25,
            "invalid_id_fraction": 0.0,
            "accepted": False,
            "n_interventions": 1,
            "queries_used": 2,
            "turns": 1,
        }


class ProbBeliefSmokeTests(unittest.TestCase):
    @staticmethod
    def _prepare_args(root):
        return SimpleNamespace(
            source_run=runner.DEFAULT_SOURCE_RUN,
            runs_root=root,
            run_id="synthetic",
            resume=False,
            replace_empty_run=False,
            model="Qwen/Qwen3.5-9B",
            served_model_name=None,
            budget=15,
            max_turns=32,
            dtype="bfloat16",
            max_model_len=32768,
            disable_multimodal=True,
            gpus_resolved=("0", "1", "4"),
            ports_resolved=(18005, 18006, 18007),
            host="127.0.0.1",
            gpu_memory_utilization=0.45,
            request_timeout=900,
            transport_retries=3,
        )

    def test_source_is_exact_complete_dataset_and_only_p2_baseline_is_selected(self):
        source = runner._source_snapshot(runner.DEFAULT_SOURCE_RUN)
        self.assertEqual(len(source["worlds"]), 90)
        self.assertEqual(
            Counter(world["archetype"] for world in source["worlds"]),
            Counter({archetype: 10 for archetype in runner.EXPECTED_ARCHETYPES}),
        )
        self.assertEqual(len(source["p2_per_archetype"]), 9)
        self.assertTrue(
            all(row["config_id"] == "p2_r1" for row in source["p2_per_archetype"])
        )
        self.assertEqual(source["p2_overall"]["n_rollouts"], 720)

    def test_current_prompt_is_new_probability_belief_condition(self):
        source = runner._source_snapshot(runner.DEFAULT_SOURCE_RUN)
        prompt = runner.current_system_prompt()
        self.assertIn("maintain a probability belief", prompt)
        self.assertNotEqual(
            runner.sha256_bytes(prompt.encode()), source["p2_prompt_sha256"]
        )

    def test_paired_seed_matches_existing_p2_terminal(self):
        source = runner._source_snapshot(runner.DEFAULT_SOURCE_RUN)
        world = source["worlds"][0]
        old_path = (
            Path(source["source_run"])
            / "outputs"
            / world["archetype"]
            / world["world_id"]
            / "p2"
            / "rollout_00.jsonl"
        )
        terminal = runner.read_jsonl(old_path)[-1]
        self.assertEqual(
            runner.derive_rollout_seed(source["master_seed"], world["world_id"], 0),
            terminal["request_seed"],
        )

    def test_raw_response_is_preserved_in_atomic_jsonl_and_validates(self):
        source = runner._source_snapshot(runner.DEFAULT_SOURCE_RUN)
        world = source["worlds"][0]
        raw = (
            '<reasoning>done</reasoning>\n'
            '<action type="answer">{"actions":[]}</action>\n'
            '<memory>a0=0.2, a1=0.8</memory>'
        )
        manifest = {
            "run_id": "smoke",
            "science": {
                "system_prompt": runner.current_system_prompt(),
                "sampling": {"max_input_tokens": 18432},
                "environment": {"budget": 15, "max_turns": 32},
            },
            "source": {"master_seed": source["master_seed"]},
        }
        temporary = HERE / ".smoke_test_artifacts"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            destination = runner.output_path(temporary, world, 0)
            client = _FakeClient(raw)
            runner._drive_episode(
                destination,
                manifest,
                world,
                0,
                _FakeEnv(),
                client,
                lambda answer, interventions: {
                    "score": 0.375,
                    "part_a": 0.5,
                    "part_b": 0.25,
                    "accepted": False,
                    "evaluation_error": False,
                },
            )
            records = runner.read_jsonl(destination)
            self.assertEqual(records[0]["raw_model_response"], raw)
            self.assertEqual(records[-1]["score"], 0.375)
            validated = runner.validate_rollout_records(records, manifest, world, 0)
            self.assertEqual(validated["terminal"]["termination_reason"], "answer")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def test_full_cardinality_aggregation_and_csv_smoke(self):
        temporary = HERE / ".smoke_aggregate_artifacts"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            args = self._prepare_args(temporary)
            run_dir, manifest = runner.prepare_run(args, runner.SamplingConfig())
            for archetype_index, archetype in enumerate(runner.EXPECTED_ARCHETYPES):
                score = (archetype_index + 1) / 10.0
                worlds = [
                    world
                    for world in manifest["source"]["worlds"]
                    if world["archetype"] == archetype
                ]
                for world in worlds:
                    for rollout_index in range(runner.ROLLOUTS_PER_WORLD):
                        seed = runner.derive_rollout_seed(
                            manifest["source"]["master_seed"],
                            world["world_id"],
                            rollout_index,
                        )
                        observation = f"synthetic observation {world['world_id']}"
                        history = [
                            {
                                "role": "system",
                                "content": manifest["science"]["system_prompt"],
                            },
                            {"role": "user", "content": observation},
                        ]
                        turn = {
                            "schema_version": runner.SCHEMA_VERSION,
                            "record_type": "turn",
                            "run_id": manifest["run_id"],
                            "world_id": world["world_id"],
                            "archetype": archetype,
                            "prompt_id": runner.PROMPT_ID,
                            "baseline_prompt_id": runner.SEED_PAIR_PROMPT_ID,
                            "rollout_index": rollout_index,
                            "turn_index": 0,
                            "request_seed": seed,
                            "request_message_count": 2,
                            "request_messages_sha256": runner._request_messages_hash(history),
                            "request_prompt_tokens": 100,
                            "observation": observation,
                            "raw_model_response": '<action type="answer">{}</action>',
                            "reasoning_content": None,
                            "parser_input_synthesized": False,
                            "synthetic_action": False,
                            "request_error": None,
                            "parsed_action_type": "answer",
                            "finish_reason": "stop",
                            "timing": {"latency_s": 0.01},
                            "transport_attempts": 1,
                            "usage": {},
                        }
                        terminal = {
                            "schema_version": runner.SCHEMA_VERSION,
                            "record_type": "terminal",
                            "run_id": manifest["run_id"],
                            "world_id": world["world_id"],
                            "archetype": archetype,
                            "prompt_id": runner.PROMPT_ID,
                            "baseline_prompt_id": runner.SEED_PAIR_PROMPT_ID,
                            "rollout_index": rollout_index,
                            "request_seed": seed,
                            "sampling": manifest["science"]["sampling"],
                            "reward": score,
                            "invalid_id_fraction": 0.0,
                            "candidate_accepted": False,
                            "candidate_reward_error": False,
                            "score": score,
                            "part_a": score,
                            "part_b": score,
                            "evaluation_accepted": False,
                            "evaluation_error": False,
                            "termination_reason": "answer",
                            "context_limit_error": None,
                            "intervention_count": 1,
                            "experiment_count": 1,
                            "turn_count": 1,
                            "transcript_sha256": runner._transcript_hash([turn]),
                            "complete": True,
                        }
                        runner.atomic_write_jsonl(
                            runner.output_path(run_dir, world, rollout_index),
                            [turn, terminal],
                        )
            stats = runner.aggregate_run(run_dir)
            validation = runner.validate_run(run_dir)
            self.assertTrue(stats["completeness"]["complete"])
            self.assertEqual(len(stats["per_archetype"]), 9)
            self.assertEqual(stats["per_archetype"][0]["avg_score"], 0.1)
            self.assertEqual(validation["raw_jsonl_files"], 720)
            self.assertTrue((run_dir / "summaries" / "per_archetype.csv").is_file())
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def test_empty_failed_run_can_be_replaced_but_material_run_cannot(self):
        temporary = HERE / ".smoke_replace_artifacts"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            args = self._prepare_args(temporary)
            run_dir, first = runner.prepare_run(args, runner.SamplingConfig())
            args.replace_empty_run = True
            _, replacement = runner.prepare_run(args, runner.SamplingConfig())
            self.assertEqual(first["science_fingerprint"], replacement["science_fingerprint"])

            runner.atomic_write_text(run_dir / "logs" / "started.log", "started\n")
            with self.assertRaisesRegex(RuntimeError, "artifacts exist"):
                runner.prepare_run(args, runner.SamplingConfig())
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
