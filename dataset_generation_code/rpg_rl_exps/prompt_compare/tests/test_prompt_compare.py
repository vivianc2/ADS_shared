from __future__ import annotations

import copy
import io
import json
import math
import sys
import tempfile
import unittest
import urllib.error
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
from candidates import (  # noqa: E402
    PROMPTS,
    evaluate_candidate_rewards,
    reward_r2,
    reward_r3,
)
from env import RPGEnv  # noqa: E402
from pipeline import (  # noqa: E402
    SCHEMA_VERSION,
    _request_messages_hash,
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
from servers import (  # noqa: E402
    ChatTemplateTokenCounter,
    ContextLengthExceededError,
    InputLengthExceededError,
    SamplingSettings,
    VLLMClient,
)
from storage import atomic_write_jsonl, completed_final_record, read_jsonl  # noqa: E402
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
        "schema_version": SCHEMA_VERSION,
        "record_type": "terminal",
        "run_id": "test",
        "world_id": "w",
        "archetype": "confounded_chain",
        "prompt_id": "p1",
        "rollout_index": 0,
        "request_seed": 1,
        "sampling": {},
        "candidate_rewards": {"r1": 0.0, "r2": 0.0, "r3": 0.0},
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
        "context_limit_error": None,
        "intervention_count": 0,
        "experiment_count": 0,
        "turn_count": 1,
        "transcript_sha256": "0" * 64,
        "complete": True,
    }


def _complete_rollout(manifest, world, prompt, rollout_index=0):
    seed = derive_rollout_seed(
        manifest["science"]["master_seed"],
        world["world_id"],
        prompt["prompt_id"],
        rollout_index,
    )
    first_observation = "first public observation"
    first_response = '<action type="measure">{"ids":[]}</action>'
    first_request = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": first_observation},
    ]
    first_turn = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "turn",
        "run_id": manifest["run_id"],
        "world_id": world["world_id"],
        "archetype": world["archetype"],
        "prompt_id": prompt["prompt_id"],
        "rollout_index": rollout_index,
        "turn_index": 0,
        "request_seed": seed,
        "request_message_count": 2,
        "request_messages_sha256": _request_messages_hash(first_request),
        "request_prompt_tokens": 100,
        "observation": first_observation,
        "raw_model_response": first_response,
        "reasoning_content": None,
        "parser_input_synthesized": False,
        "synthetic_action": False,
        "request_error": None,
        "parsed_action_type": "measure",
        "finish_reason": "stop",
        "timing": {"latency_s": 0.01},
        "transport_attempts": 1,
        "usage": {},
    }
    second_observation = "second public observation"
    second_request = [
        *first_request,
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": second_observation},
    ]
    second_turn = {
        **first_turn,
        "turn_index": 1,
        "request_message_count": 4,
        "request_messages_sha256": _request_messages_hash(second_request),
        "request_prompt_tokens": 200,
        "observation": second_observation,
        "raw_model_response": '<action type="give_up">{}</action>',
        "parsed_action_type": "give_up",
    }
    turns = [first_turn, second_turn]
    terminal = _complete_terminal()
    terminal.update({
        "run_id": manifest["run_id"],
        "world_id": world["world_id"],
        "archetype": world["archetype"],
        "prompt_id": prompt["prompt_id"],
        "rollout_index": rollout_index,
        "request_seed": seed,
        "sampling": manifest["science"]["sampling"],
        "turn_count": len(turns),
        "transcript_sha256": _transcript_hash(turns),
    })
    return [*turns, terminal]


class PromptCompareTests(unittest.TestCase):
    def test_vllm_client_sends_the_complete_message_list_unchanged(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first observation"},
            {"role": "assistant", "content": "first action"},
            {"role": "user", "content": "second observation"},
        ]
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": '<action type="give_up">{}</action>'},
                "finish_reason": "stop",
            }],
            "usage": {},
        }).encode("utf-8")
        client = VLLMClient(
            "http://127.0.0.1:18005/v1",
            "Qwen/Qwen3.5-9B",
            SamplingSettings(),
        )
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            client.generate(messages, 123)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["top_k"], -1)
        self.assertNotIn("max_input_tokens", payload)

    def test_chat_template_counter_uses_the_request_thinking_mode(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = {"input_ids": [4, 5, 6]}
        counter = ChatTemplateTokenCounter(
            "unused",
            enable_thinking=True,
            tokenizer=tokenizer,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "observation"},
        ]
        self.assertEqual(counter(messages), 3)
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=True,
        )

    def test_client_input_limit_stops_before_http(self):
        client = VLLMClient(
            "http://127.0.0.1:18005/v1",
            "Qwen/Qwen3.5-9B",
            SamplingSettings(max_input_tokens=18432),
            token_counter=lambda _messages: 18433,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "observation"},
        ]
        with mock.patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(InputLengthExceededError) as raised:
                client.generate(messages, 123)
        self.assertEqual(raised.exception.prompt_tokens, 18433)
        self.assertEqual(raised.exception.max_input_tokens, 18432)
        urlopen.assert_not_called()

    def test_context_length_http_400_is_not_retried(self):
        client = VLLMClient(
            "http://127.0.0.1:18005/v1",
            "Qwen/Qwen3.5-9B",
            SamplingSettings(transport_retries=3),
        )
        error = urllib.error.HTTPError(
            client.url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":"maximum context length exceeded"}'),
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "observation"},
        ]
        with mock.patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(ContextLengthExceededError):
                client.generate(messages, 123)
        self.assertEqual(urlopen.call_count, 1)

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

    def test_environment_hard_stops_experiments_at_budget(self):
        world, precomputed = load_world_file(str(_world_path()))
        env = RPGEnv(
            world=world,
            gold=precomputed["gold"],
            battery=precomputed["battery"],
            budget=1,
        )
        env.reset()
        measurable_id = env.cat.measurable_ids()[0]
        action = f'<action type="measure">{{"ids":["{measurable_id}"]}}</action>'
        result = {"experiment_id": 1, "n_units": 0, "readings": {}}
        with mock.patch.object(env.sim, "measure", return_value=result) as measure:
            _, _, done, _ = env.step(action)
            self.assertFalse(done)
            self.assertEqual(env._used, 1)

            observation, reward, done, info = env.step(action)
            self.assertFalse(done)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["turn_type"], "budget_exhausted")
            self.assertIn("experiment budget exhausted", observation)
            self.assertEqual(env._used, 1)
            self.assertEqual(measure.call_count, 1)

        _, _, done, info = env.step('<action type="give_up">{}</action>')
        self.assertTrue(done)
        self.assertEqual(info["queries_used"], 1)

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
            self.assertEqual(
                evaluate_candidate_rewards(
                    {}, None, None, None, None, n_interventions=1
                ),
                {"r1": 99.0, "r2": 0.6, "r3": 0.1},
            )

    def test_seed_is_stable_for_each_prompt_rollout(self):
        seed = derive_rollout_seed(7_000_000, "world-x", "p2", 3)
        self.assertEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p2", 3))
        self.assertNotEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p3", 3))
        self.assertNotEqual(seed, derive_rollout_seed(7_000_000, "world-x", "p2", 4))
        self.assertIsInstance(seed, int)
        self.assertTrue(0 <= seed < 2_147_483_647)

    def test_metric_definitions_use_population_variance_and_best_of_eight(self):
        values = [
            {
                "candidate_rewards": {
                    "r1": 0.0,
                    "r2": index / 7.0,
                    "r3": 1.0,
                },
                "score": (7 - index) / 7.0,
                "part_a": 0.25,
                "part_b": index / 7.0,
            }
            for index in range(8)
        ]
        summary = summarize_world_rollouts(values, "r2")
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
        missing_reward = _complete_terminal()
        missing_reward["candidate_rewards"].pop("r3")
        with self.assertRaisesRegex(ValueError, "exactly r1, r2, and r3"):
            validate_terminal_schema(missing_reward)
        inconsistent_reward = _complete_terminal()
        inconsistent_reward["candidate_rewards"]["r2"] = 0.1
        with self.assertRaisesRegex(ValueError, "disagrees with its stored components"):
            validate_terminal_schema(inconsistent_reward)

    def test_atomic_completion_and_resume_skip_only_complete_files(self):
        world = {"archetype": "confounded_chain", "world_id": "w", "file": "world.json"}
        prompt = {"prompt_id": "p1", "system_prompt": "system"}
        manifest = {
            "run_id": "test",
            "science": {
                "master_seed": 7_000_000,
                "environment": {"budget": 15, "max_turns": 32},
                "sampling": {"max_input_tokens": 18432},
            },
        }
        valid_records = _complete_rollout(manifest, world, prompt)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = output_path(root, world, prompt["prompt_id"], 0)
            atomic_write_jsonl(path, [{"record_type": "turn", "complete": False}])
            self.assertIsNone(completed_final_record(path))

            atomic_write_jsonl(path, valid_records)
            self.assertTrue(completed_final_record(path)["complete"])
            self.assertEqual(
                run_episode(root, manifest, world, prompt, 0, object()),
                "skipped",
            )
            self.assertFalse(list(path.parent.glob(f".{path.name}.*.tmp")))

            corruptions = {}

            corrupted = copy.deepcopy(valid_records)
            corrupted[-1]["experiment_count"] = 16
            corruptions["over_budget"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[-1]["turn_count"] = 3
            corruptions["turn_count"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[-1]["transcript_sha256"] = "f" * 64
            corruptions["transcript_hash"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[1]["request_messages_sha256"] = "e" * 64
            corruptions["request_history_hash"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[1]["request_prompt_tokens"] = 18433
            corruptions["prompt_token_limit"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[1]["request_prompt_tokens"] = "200"
            corruptions["prompt_token_type"] = corrupted

            corrupted = copy.deepcopy(valid_records)
            corrupted[1].update({
                "synthetic_action": True,
                "request_error": "client-side input limit",
                "finish_reason": "length",
                "request_prompt_tokens": 18433,
                "transport_attempts": 0,
            })
            corrupted[-1]["transcript_sha256"] = _transcript_hash(corrupted[:-1])
            corruptions["synthetic_terminal_mismatch"] = corrupted

            for label, records in corruptions.items():
                with self.subTest(corruption=label):
                    atomic_write_jsonl(path, records)
                    with mock.patch(
                        "pipeline.load_world_file",
                        side_effect=RuntimeError(f"{label} rerun reached loader"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, f"{label} rerun"):
                            run_episode(root, manifest, world, prompt, 0, object())

            atomic_write_jsonl(path, [{"record_type": "turn", "complete": False}])
            with mock.patch(
                "pipeline.load_world_file",
                side_effect=RuntimeError("incomplete rerun reached loader"),
            ):
                with self.assertRaisesRegex(RuntimeError, "incomplete rerun reached loader"):
                    run_episode(root, manifest, world, prompt, 0, object())

    def test_completed_episode_applies_and_stores_all_three_rewards(self):
        world_path = _world_path()
        world, _ = load_world_file(str(world_path))
        world_meta = {
            "world_id": world["world_id"],
            "archetype": "confounded_chain",
            "seed": 7_000_001,
            "file": str(world_path),
        }
        manifest = {
            "run_id": "test",
            "science": {
                "master_seed": 7_000_000,
                "environment": {"budget": 15, "max_turns": 32},
                "sampling": {},
            },
        }
        response = '<action type="give_up">{}</action>'
        client = mock.Mock()
        client.generate.return_value = SimpleNamespace(
            action_text=response,
            raw_text=response,
            reasoning_content=None,
            finish_reason="stop",
            latency_s=0.01,
            attempts=1,
            usage={},
            prompt_tokens=100,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run_episode(
                    root,
                    manifest,
                    world_meta,
                    {"prompt_id": "p1", "system_prompt": PROMPTS["p1"]},
                    0,
                    client,
                ),
                "completed",
            )
            final = completed_final_record(output_path(root, world_meta, "p1", 0))
        self.assertIsNotNone(final)
        self.assertEqual(final["candidate_rewards"], {"r1": 0.0, "r2": 0.0, "r3": 0.0})
        validate_terminal_schema(final)

    def test_multiturn_episode_sends_complete_chat_history(self):
        world_path = _world_path()
        world, _ = load_world_file(str(world_path))
        world_meta = {
            "world_id": world["world_id"],
            "archetype": "confounded_chain",
            "seed": 7_000_001,
            "file": str(world_path),
        }
        manifest = {
            "run_id": "history_test",
            "science": {
                "master_seed": 7_000_000,
                "environment": {"budget": 15, "max_turns": 32},
                "sampling": {},
            },
        }
        first_response = (
            '<reasoning>inspect</reasoning>\n'
            '<action type="measure">{"ids":[]}</action>\n'
            '<memory>remember the catalog</memory>'
        )
        final_response = '<action type="give_up">{}</action>'

        def generation(text):
            return SimpleNamespace(
                action_text=text,
                raw_text=text,
                reasoning_content=None,
                finish_reason="stop",
                latency_s=0.01,
                attempts=1,
                usage={},
                prompt_tokens=100,
            )

        client = mock.Mock()
        client.generate.side_effect = [generation(first_response), generation(final_response)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run_episode(
                    root,
                    manifest,
                    world_meta,
                    {"prompt_id": "p1", "system_prompt": PROMPTS["p1"]},
                    0,
                    client,
                ),
                "completed",
            )
            records = read_jsonl(output_path(root, world_meta, "p1", 0))

        first_messages = client.generate.call_args_list[0].args[0]
        second_messages = client.generate.call_args_list[1].args[0]
        self.assertEqual([message["role"] for message in first_messages], ["system", "user"])
        self.assertEqual(
            [message["role"] for message in second_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("MEASURABLE SIGNALS (ids):", second_messages[1]["content"])
        self.assertEqual(second_messages[2]["content"], first_response)
        self.assertIn("remember the catalog", second_messages[3]["content"])
        self.assertEqual(
            [record["request_message_count"] for record in records[:-1]],
            [2, 4],
        )

    def test_context_limit_becomes_a_recorded_zero_reward_terminal(self):
        world_path = _world_path()
        world, _ = load_world_file(str(world_path))
        world_meta = {
            "world_id": world["world_id"],
            "archetype": "confounded_chain",
            "seed": 7_000_001,
            "file": str(world_path),
        }
        manifest = {
            "run_id": "context_test",
            "science": {
                "master_seed": 7_000_000,
                "environment": {"budget": 15, "max_turns": 32},
                "sampling": {},
            },
        }
        client = mock.Mock()
        client.generate.side_effect = ContextLengthExceededError(
            "HTTP 400: maximum context length exceeded"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run_episode(
                    root,
                    manifest,
                    world_meta,
                    {"prompt_id": "p1", "system_prompt": PROMPTS["p1"]},
                    0,
                    client,
                ),
                "completed",
            )
            records = read_jsonl(output_path(root, world_meta, "p1", 0))
        turn, terminal = records
        self.assertTrue(turn["synthetic_action"])
        self.assertIn("maximum context length", turn["request_error"])
        self.assertEqual(turn["parsed_action_type"], "give_up")
        self.assertEqual(terminal["termination_reason"], "context_limit")
        self.assertEqual(terminal["candidate_rewards"], {"r1": 0.0, "r2": 0.0, "r3": 0.0})
        validate_terminal_schema(terminal)

    def test_input_limit_becomes_length_terminal_without_transport(self):
        world_path = _world_path()
        world, _ = load_world_file(str(world_path))
        world_meta = {
            "world_id": world["world_id"],
            "archetype": "confounded_chain",
            "seed": 7_000_001,
            "file": str(world_path),
        }
        manifest = {
            "run_id": "input_limit_test",
            "science": {
                "master_seed": 7_000_000,
                "environment": {"budget": 15, "max_turns": 32},
                "sampling": {"max_input_tokens": 18432},
            },
        }
        client = mock.Mock()
        client.generate.side_effect = InputLengthExceededError(18433, 18432)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run_episode(
                    root,
                    manifest,
                    world_meta,
                    {"prompt_id": "p1", "system_prompt": PROMPTS["p1"]},
                    0,
                    client,
                ),
                "completed",
            )
            turn, terminal = read_jsonl(output_path(root, world_meta, "p1", 0))
        self.assertTrue(turn["synthetic_action"])
        self.assertEqual(turn["finish_reason"], "length")
        self.assertEqual(turn["request_prompt_tokens"], 18433)
        self.assertEqual(turn["transport_attempts"], 0)
        self.assertEqual(terminal["termination_reason"], "input_length")
        self.assertEqual(terminal["candidate_rewards"], {"r1": 0.0, "r2": 0.0, "r3": 0.0})
        validate_terminal_schema(terminal)

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
            self.assertEqual(manifest["science"]["expected"]["episodes"], 216)
            self.assertEqual(
                manifest["science"]["expected"]["reward_evaluations"], 648
            )

            for prompt in manifest["science"]["prompts"]:
                for world in manifest["worlds"]:
                    for rollout_index in range(8):
                        seed = derive_rollout_seed(
                            args.seed, world["world_id"], prompt["prompt_id"], rollout_index
                        )
                        observation = f"public observation for {world['world_id']}"
                        request_messages = [
                            {"role": "system", "content": prompt["system_prompt"]},
                            {"role": "user", "content": observation},
                        ]
                        turn = {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "turn",
                            "run_id": manifest["run_id"],
                            "world_id": world["world_id"],
                            "archetype": world["archetype"],
                            "prompt_id": prompt["prompt_id"],
                            "rollout_index": rollout_index,
                            "turn_index": 0,
                            "request_seed": seed,
                            "request_message_count": len(request_messages),
                            "request_messages_sha256": _request_messages_hash(request_messages),
                            "request_prompt_tokens": 100,
                            "observation": observation,
                            "raw_model_response": f"{prompt['prompt_id']} response {rollout_index}",
                            "reasoning_content": None,
                            "parser_input_synthesized": False,
                            "synthetic_action": False,
                            "request_error": None,
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
                            "prompt_id": prompt["prompt_id"],
                            "rollout_index": rollout_index,
                            "request_seed": seed,
                            "sampling": manifest["science"]["sampling"],
                            "candidate_rewards": {
                                "r1": 0.25,
                                "r2": 0.2,
                                "r3": 0.3,
                            },
                            "candidate_part_a": 0.2,
                            "candidate_part_b": 0.3,
                            "transcript_sha256": _transcript_hash([turn]),
                        })
                        atomic_write_jsonl(
                            output_path(run_dir, world, prompt["prompt_id"], rollout_index),
                            [turn, terminal],
                        )

            stats = aggregate_run(run_dir)
            self.assertEqual(len(stats["per_world"]), 81)
            self.assertEqual(len(stats["per_archetype"]), 81)
            self.assertEqual(len(stats["overall"]), 9)
            self.assertEqual(
                stats["reward_application"],
                {"mode": "post_hoc_shared_rollout", "rewards_per_terminal": 3},
            )
            reward_means = {
                row["reward_id"]: row["reward_mean"] for row in stats["overall"]
                if row["prompt_id"] == "p1"
            }
            self.assertEqual(reward_means, {"r1": 0.25, "r2": 0.2, "r3": 0.3})
            validate_run(run_dir, require_figures=False)
            self.assertEqual(len(plot_all(run_dir / "stats.json")), 5)
            summary = validate_run(run_dir)
            self.assertEqual(summary["episodes"], 216)
            self.assertEqual(summary["reward_evaluations"], 648)

            args.resume = True
            _, resumed = prepare_run(args, sampling)
            self.assertEqual(resumed["science_fingerprint"], manifest["science_fingerprint"])


if __name__ == "__main__":
    unittest.main()
