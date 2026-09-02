#!/usr/bin/env python3
"""CLI for the 3 x 3 RPG system-prompt/reward comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import configure_imports

configure_imports()

from pipeline import (  # noqa: E402
    aggregate_run,
    default_runs_root,
    parse_csv_strings,
    parse_ports,
    prepare_run,
    run_rollouts,
    utc_now,
    validate_run,
)
from servers import (  # noqa: E402
    SamplingSettings,
    ServerManager,
    ServerSettings,
    inspect_gpus,
)
from storage import atomic_write_json  # noqa: E402
from visualization.plot_results import plot_all  # noqa: E402


def _sampling(args) -> SamplingSettings:
    return SamplingSettings(
        max_input_tokens=args.max_input_tokens,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        enable_thinking=args.thinking,
        request_timeout_s=args.request_timeout,
        transport_retries=args.transport_retries,
    )


def _add_all_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument(
        "--gpus", required=True,
        help="exactly three unique physical GPU ids, e.g. 0,1,2 or 5,6,7",
    )
    parser.add_argument(
        "--ports", default="18005,18006,18007",
        help="exactly three unique worker ports",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--world-max-attempts", type=int, default=1000)

    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument(
        "--max-input-tokens", type=int, default=18432,
        help="stop an episode before a rendered chat prompt exceeds this token count",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True,
        help="hold Qwen chat-template thinking mode fixed across all requests",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument(
        "--disable-multimodal", action=argparse.BooleanOptionalAction, default=True,
        help="disable unused Qwen multimodal encoder cache in each text-only worker",
    )
    parser.add_argument("--api-key", default="EMPTY", help=argparse.SUPPRESS)


def _validate_all_args(args) -> None:
    args.gpus_resolved = parse_csv_strings(args.gpus)
    args.ports_resolved = parse_ports(args.ports)
    if len(args.gpus_resolved) != 3 or len(set(args.gpus_resolved)) != 3:
        raise ValueError("--gpus must contain exactly three unique GPU ids")
    if len(args.ports_resolved) != 3 or len(set(args.ports_resolved)) != 3:
        raise ValueError("--ports must contain exactly three unique ports")
    if (
        args.budget < 1
        or args.max_turns < 1
        or args.max_input_tokens < 1
        or args.max_new_tokens < 1
        or args.max_model_len < 1
    ):
        raise ValueError(
            "budget, max-turns, max-input-tokens, max-new-tokens, and "
            "max-model-len must be positive"
        )
    if args.max_input_tokens + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            "max-input-tokens + max-new-tokens must not exceed max-model-len"
        )
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    if args.top_k != -1 and args.top_k < 1:
        raise ValueError("top-k must be -1 (disabled) or a positive integer")
    if not 0.0 <= args.min_p <= 1.0:
        raise ValueError("min-p must be in [0, 1]")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if args.transport_retries < 0 or args.world_max_attempts < 1:
        raise ValueError("retry and world-attempt settings are invalid")


def command_all(args) -> int:
    _validate_all_args(args)
    sampling = _sampling(args)
    run_dir, manifest = prepare_run(args, sampling)
    print(f"run directory: {run_dir}", flush=True)
    print(
        f"worlds persisted: {len(manifest['worlds'])}; "
        f"expected model episodes: {manifest['science']['expected']['episodes']}; "
        "expected reward evaluations: "
        f"{manifest['science']['expected']['reward_evaluations']}",
        flush=True,
    )

    inventory = inspect_gpus(args.gpus_resolved)
    launch_record = {
        "started_at": utc_now(),
        "gpus": inventory,
        "ports": list(args.ports_resolved),
        "model": args.model,
        "served_model_name": args.served_model_name or args.model,
        "server_settings": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disable_multimodal": args.disable_multimodal,
        },
    }
    launch_path = run_dir / "logs" / f"launch_{utc_now().replace(':', '').replace('+', '_')}.json"
    atomic_write_json(launch_path, launch_record)

    server_settings = ServerSettings(
        model=args.model,
        served_model_name=args.served_model_name or args.model,
        host=args.host,
        ports=args.ports_resolved,
        gpus=args.gpus_resolved,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        health_timeout_s=args.health_timeout,
        executable=args.vllm_executable,
        disable_multimodal=args.disable_multimodal,
    )
    manager = ServerManager(server_settings, run_dir / "logs")
    try:
        print("starting one-GPU fit preflight on the first worker...", flush=True)
        base_urls = manager.start()
        print(f"three vLLM workers healthy: {', '.join(base_urls)}", flush=True)
        rollout_summary = run_rollouts(
            run_dir, manifest, base_urls, sampling, api_key=args.api_key
        )
        print(f"rollout execution summary: {json.dumps(rollout_summary, sort_keys=True)}")
    finally:
        manager.stop()

    aggregate_run(run_dir, require_complete=True)
    validate_run(run_dir, require_figures=False)
    written = plot_all(run_dir / "stats.json", run_dir / "figures")
    summary = validate_run(run_dir, require_figures=True)
    print(f"figures generated: {len(written)}", flush=True)
    print(f"final completeness: {json.dumps(summary, sort_keys=True)}", flush=True)
    print(f"run directory: {run_dir}", flush=True)
    return 0


def command_aggregate(args) -> int:
    stats = aggregate_run(args.run_dir.resolve(), require_complete=not args.allow_incomplete)
    print(json.dumps(stats["completeness"], indent=2))
    return 0 if stats["completeness"]["complete"] else 2


def command_plot(args) -> int:
    paths = plot_all(args.run_dir.resolve() / "stats.json", args.run_dir.resolve() / "figures")
    for path in paths:
        print(path)
    return 0


def command_validate(args) -> int:
    summary = validate_run(args.run_dir.resolve(), require_figures=not args.no_figures)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all", help="generate, serve, run, aggregate, validate, plot")
    _add_all_arguments(all_parser)
    all_parser.set_defaults(handler=command_all)

    aggregate_parser = subparsers.add_parser("aggregate", help="regenerate stats.json")
    aggregate_parser.add_argument("--run-dir", type=Path, required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")
    aggregate_parser.set_defaults(handler=command_aggregate)

    plot_parser = subparsers.add_parser("plot", help="regenerate figures from stats.json")
    plot_parser.add_argument("--run-dir", type=Path, required=True)
    plot_parser.set_defaults(handler=command_plot)

    validate_parser = subparsers.add_parser("validate", help="validate a completed run")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument("--no-figures", action="store_true")
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
