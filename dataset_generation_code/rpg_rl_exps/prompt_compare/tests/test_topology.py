from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from pipeline import run_rollouts  # noqa: E402
from run_experiment import _validate_all_args, build_parser  # noqa: E402
from servers import SamplingSettings, ServerManager, ServerSettings  # noqa: E402


class TopologyTests(unittest.TestCase):
    def test_cli_accepts_three_unique_gpus_and_default_ports(self):
        args = build_parser().parse_args([
            "all",
            "--model", "Qwen/Qwen3.5-9B",
            "--gpus", "0,1,2",
            "--seed", "7000000",
            "--run-id", "three_gpu",
        ])
        _validate_all_args(args)
        self.assertEqual(args.gpus_resolved, ("0", "1", "2"))
        self.assertEqual(args.ports_resolved, (18005, 18006, 18007))
        self.assertEqual(args.max_input_tokens, 18432)
        self.assertEqual(args.max_new_tokens, 8192)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.top_k, -1)
        self.assertTrue(args.thinking)

    def test_cli_rejects_fewer_than_three_gpus(self):
        args = build_parser().parse_args([
            "all",
            "--model", "Qwen/Qwen3.5-9B",
            "--gpus", "0,1",
            "--seed", "7000000",
            "--run-id", "bad_gpus",
        ])
        with self.assertRaisesRegex(ValueError, "exactly three unique GPU ids"):
            _validate_all_args(args)

    def test_cli_rejects_duplicate_gpu_ids(self):
        args = build_parser().parse_args([
            "all",
            "--model", "Qwen/Qwen3.5-9B",
            "--gpus", "0,1,1",
            "--seed", "7000000",
            "--run-id", "duplicate_gpus",
        ])
        with self.assertRaisesRegex(ValueError, "exactly three unique GPU ids"):
            _validate_all_args(args)

    def test_cli_rejects_input_plus_generation_beyond_model_context(self):
        args = build_parser().parse_args([
            "all",
            "--model", "Qwen/Qwen3.5-9B",
            "--gpus", "0,1,2",
            "--seed", "7000000",
            "--run-id", "bad_context_budget",
            "--max-model-len", "26000",
        ])
        with self.assertRaisesRegex(
            ValueError,
            r"max-input-tokens \+ max-new-tokens",
        ):
            _validate_all_args(args)

    def test_cli_requires_three_unique_ports(self):
        args = build_parser().parse_args([
            "all",
            "--model", "Qwen/Qwen3.5-9B",
            "--gpus", "5,6,7",
            "--ports", "18005",
            "--seed", "7000000",
            "--run-id", "bad_ports",
        ])
        with self.assertRaisesRegex(ValueError, "exactly three unique ports"):
            _validate_all_args(args)

    def test_server_manager_accepts_three_gpu_workers(self):
        settings = ServerSettings(
            model="Qwen/Qwen3.5-9B",
            served_model_name="Qwen/Qwen3.5-9B",
            host="127.0.0.1",
            ports=(18005, 18006, 18007),
            gpus=("5", "6", "7"),
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            health_timeout_s=10,
            executable="vllm",
            disable_multimodal=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manager = ServerManager(settings, Path(temporary))

            def launch(gpu, port):
                server = SimpleNamespace(gpu=gpu, port=port)
                manager.owned.append(server)
                return server

            with (
                mock.patch("servers.inspect_gpus", return_value=[]),
                mock.patch.object(manager, "_launch_one", side_effect=launch),
                mock.patch.object(manager, "_wait_healthy"),
            ):
                self.assertEqual(
                    manager.start(),
                    (
                        "http://127.0.0.1:18005/v1",
                        "http://127.0.0.1:18006/v1",
                        "http://127.0.0.1:18007/v1",
                    ),
                )

    def test_vllm_023_launch_omits_removed_request_log_flag(self):
        settings = ServerSettings(
            model="Qwen/Qwen3.5-9B",
            served_model_name="Qwen/Qwen3.5-9B",
            host="127.0.0.1",
            ports=(18005, 18006, 18007),
            gpus=("0", "1", "2"),
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            health_timeout_s=10,
            executable="vllm",
            disable_multimodal=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manager = ServerManager(settings, Path(temporary))
            process = mock.Mock()
            with (
                mock.patch("servers.shutil.which", return_value="/venv/bin/vllm"),
                mock.patch("servers._assert_port_available"),
                mock.patch("servers.subprocess.Popen", return_value=process) as popen,
            ):
                server = manager._launch_one("0", 18005)
            command = popen.call_args.args[0]
            self.assertNotIn("--disable-log-requests", command)
            self.assertEqual(command[:3], ["/venv/bin/vllm", "serve", settings.model])
            server.log_handle.close()
            manager.owned.clear()

    def test_three_workers_schedule_every_group_and_rollout(self):
        manifest = {
            "science": {
                "model": "Qwen/Qwen3.5-9B",
                "served_model_name": "Qwen/Qwen3.5-9B",
                "prompts": [
                    {"prompt_id": "p1"},
                    {"prompt_id": "p2"},
                ],
                "expected": {"rollouts_per_group": 2},
                "master_seed": 7000000,
            },
            "worlds": [
                {"world_id": "w1", "archetype": "confounded_chain"},
                {"world_id": "w2", "archetype": "dose_window"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch("pipeline.ChatTemplateTokenCounter", return_value=mock.Mock()),
                mock.patch("pipeline.run_episode", return_value="completed") as episode,
            ):
                summary = run_rollouts(
                    Path(temporary),
                    manifest,
                    (
                        "http://127.0.0.1:18005/v1",
                        "http://127.0.0.1:18006/v1",
                        "http://127.0.0.1:18007/v1",
                    ),
                    SamplingSettings(),
                )
        self.assertEqual(episode.call_count, 8)
        self.assertEqual(summary["groups"], 4)
        self.assertEqual(summary["completed"], 8)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["errors"], 0)


if __name__ == "__main__":
    unittest.main()
