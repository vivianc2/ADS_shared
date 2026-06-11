"""Run the static RPG scientist agent on RPG world JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator_rpg import OrchestratorRPG
from schemas_rpg import StaticRPGQuestion
from scientist_agent_rpg import ScientistAgentRPG
from world_model_rpg import StaticRPGWorldModel

logger = logging.getLogger(__name__)


def load_worlds(worlds_dir: Optional[str], world_json: Optional[str]) -> List[Tuple[str, Dict[str, Any]]]:
    if world_json:
        paths = [world_json]
    elif worlds_dir:
        paths = sorted(glob.glob(os.path.join(worlds_dir, "world_*.json")))
    else:
        raise ValueError("provide --worlds-dir or --world-json")
    worlds = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            world = json.load(f)
        if world.get("schema_version") not in ("rpg_static_v2", "rpg_static_v3"):
            logger.warning("Skipping non-static-RPG world %s schema=%s", path, world.get("schema_version"))
            continue
        if not world.get("questions"):
            logger.warning("Skipping %s: no questions", path)
            continue
        worlds.append((path, world))
    return worlds


def question_from_world(world: Dict[str, Any]) -> StaticRPGQuestion:
    q = world["questions"][0]
    return StaticRPGQuestion(
        question_type=q.get("question_type", f"rpg_{world.get('meta', {}).get('archetype', '')}"),
        question_text=q.get("question", world.get("visible", {}).get("question", "")),
        ground_truth=q.get("answer", world.get("oracle", {}).get("gold_answer", {})),
        answer_schema=q.get("answer_schema", world.get("visible", {}).get("answer_schema", "")),
        metadata={k: v for k, v in q.items() if k not in {"question_type", "question", "answer", "answer_schema"}},
    )


class MockRPGLLM:
    """Tiny local backend for smoke-testing the runner without external APIs."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        measurements = re.findall(r"^- ([A-Za-z0-9_]+):", user_prompt, flags=re.MULTILINE)
        knobs = re.findall(r"^- ([A-Za-z0-9_]+): values=\[([^\]]+)\]", user_prompt, flags=re.MULTILINE)
        if self.calls == 1:
            chosen = measurements[:3] or []
            query = {"mode": "observational_sample", "n_units": 40, "measurements": chosen}
            return (
                "<reasoning>Smoke-test query to verify the static RPG pipeline.</reasoning>\n"
                f"<action type=\"query\">{json.dumps(query)}</action>\n"
                "<scientist_memory>Tested: smoke observational query.\nKnown:\nUncertain:\nNext: answer.</scientist_memory>"
            )
        if "latent_regime_policy" in user_prompt:
            branch = measurements[0] if measurements else "Screen"
            knob_values = []
            for name, values in knobs:
                allowed = [v.strip() for v in values.split(",") if v.strip()]
                value = next((v for v in allowed if v != "off"), allowed[0] if allowed else "on")
                knob_values.append((name, value))
            above = {knob_values[0][0]: knob_values[0][1]} if knob_values else {}
            below = {knob_values[1][0]: knob_values[1][1]} if len(knob_values) > 1 else above
            answer = {
                "latent_structure": {
                    "n_regimes": 2,
                    "evidence": "Smoke-test latent-regime answer; not intended as a real scientific conclusion.",
                },
                "policy": {
                    "branch_variable": branch,
                    "branch_threshold": 50,
                    "if_above": above,
                    "if_below": below,
                },
                "hypothesis": "Smoke-test latent-regime policy.",
            }
            return (
                "<reasoning>Smoke-test latent-regime final answer.</reasoning>\n"
                f"<action type=\"answer\">{json.dumps(answer)}</action>\n"
                "<scientist_memory>Tested: latent-regime schema.\nKnown: runner executes.\nUncertain: scientific answer.\nNext:</scientist_memory>"
            )
        if "latent_cause_hypothesis" in user_prompt:
            answer = {
                "latent_hypothesis": {
                    "name": "clogged gutter or blocked downspout",
                    "description": "Leaves may be obstructing the roof drainage path and redirecting runoff into the yard.",
                    "confidence": 0.55,
                },
                "evidence": [
                    "Smoke-test answer mentions leaf fall, downspout delay, roof overflow, and flooding.",
                    "A flow test or clearing action would verify whether the hidden drainage path is obstructed.",
                ],
                "alternatives_ruled_out": [
                    "Soil compaction and slope could contribute, but they do not explain downspout delay as directly."
                ],
                "decisive_test": "Run a roof-edge flow test or clear/flush the gutters and downspout before comparing flooding.",
                "action_plan": {
                    "do_now": ["ClearRearGutters", "FlushDownspout"],
                    "avoid": ["RegradeLowCorner"],
                    "why": "Target the likely hidden roof-drainage obstruction first.",
                },
            }
            return (
                "<reasoning>Smoke-test story-hidden-cause final answer.</reasoning>\n"
                f"<action type=\"answer\">{json.dumps(answer)}</action>\n"
                "<scientist_memory>Tested: latent-cause schema.\nKnown: runner executes.\nUncertain: scientific answer quality.\nNext:</scientist_memory>"
            )
        intervention = {}
        if knobs:
            name, values = knobs[0]
            first_value = [v.strip() for v in values.split(",") if v.strip()][0]
            intervention[name] = first_value
        answer = {"intervention": intervention, "hypothesis": "Smoke-test answer; not intended as a real scientific conclusion."}
        return (
            "<reasoning>Smoke-test final answer.</reasoning>\n"
            f"<action type=\"answer\">{json.dumps(answer)}</action>\n"
            "<scientist_memory>Tested: pipeline smoke.\nKnown: runner executes.\nUncertain: scientific answer.\nNext:</scientist_memory>"
        )


def build_llm(args: argparse.Namespace) -> Any:
    if args.scientist_backend == "mock":
        return MockRPGLLM()
    if args.scientist_backend == "bedrock":
        from bedrock_llm import BedrockLLM

        return BedrockLLM(
            model_id=args.scientist_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    if args.scientist_backend == "openai":
        from world_model_causal import OpenAILLM

        return OpenAILLM(
            model_name=args.scientist_model,
            base_url=args.scientist_base_url,
            api_key=args.scientist_api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    if args.scientist_backend == "local":
        from scientist_agent_causal import ScientistLLM

        return ScientistLLM(
            model_name=args.scientist_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    raise ValueError(f"unknown backend {args.scientist_backend}")


def snapshot_world(path: str, snapshot_dir: str) -> str:
    """Copy the exact world JSON used by a run into the result bundle."""
    src = Path(path).expanduser().resolve()
    out = Path(snapshot_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dst = out / src.name
    if src != dst:
        shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static RPG scientist agent.")
    parser.add_argument("--worlds-dir", default=None)
    parser.add_argument("--world-json", default=None)
    parser.add_argument("--scientist-backend", choices=["bedrock", "openai", "local", "mock"], default="bedrock")
    parser.add_argument("--scientist-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--scientist-base-url", default=None)
    parser.add_argument("--scientist-api-key", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.run_name:
        if args.output_dir is None:
            args.output_dir = os.path.join("./results", args.run_name)
        if args.output is None:
            args.output = os.path.join(args.output_dir, f"{args.run_name}.json")
    args.output_dir = args.output_dir or "./results/rpg_static_agent"
    args.output = args.output or os.path.join(args.output_dir, "rpg_static_agent.json")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Keep our RPG logs verbose without dumping Bedrock request bodies/headers.
    # Botocore DEBUG output can include bearer Authorization headers.
    for noisy_logger in ("boto3", "botocore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    os.makedirs(args.output_dir, exist_ok=True)
    query_dir = os.path.join(args.output_dir, "rpg_query_data")
    log_dir = os.path.join(args.output_dir, "rpg_agent_logs")
    snapshot_dir = os.path.join(args.output_dir, "rpg_worlds_snapshot")
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    worlds = load_worlds(args.worlds_dir, args.world_json)
    if not worlds:
        logger.error("No static RPG worlds found.")
        sys.exit(1)
    logger.info("Loaded %d static RPG world(s)", len(worlds))

    llm = build_llm(args)
    results: List[Dict[str, Any]] = []

    for idx, (path, world) in enumerate(worlds, 1):
        logger.info("[%d/%d] %s", idx, len(worlds), Path(path).name)
        snapshot_path = snapshot_world(path, snapshot_dir)
        world_model = StaticRPGWorldModel.from_world_json(snapshot_path, output_dir=query_dir)
        question = question_from_world(world)
        visible_budget = world.get("visible", {}).get("experiment_budget", {})
        max_queries = args.max_queries or int(visible_budget.get("max_queries", 8))
        scientist = ScientistAgentRPG(llm=llm)
        orchestrator = OrchestratorRPG(
            world_model=world_model,
            scientist=scientist,
            question=question,
            max_queries=max_queries,
            max_turns=args.max_turns,
            log_dir=log_dir,
            scientist_model=args.scientist_model,
            dataset_file=snapshot_path,
        )
        exp = orchestrator.run()
        results.append({
            "world_file": snapshot_path,
            "source_world_file": path,
            "world_name": Path(path).stem,
            "topic": world.get("meta", {}).get("topic", ""),
            "archetype": world.get("meta", {}).get("archetype", ""),
            "question_id": question.metadata.get("id", 0),
            "question_text": question.question_text,
            "question_type": question.question_type,
            "ground_truth": question.ground_truth,
            "extracted_answer": exp.scientist_answer,
            "score": exp.score,
            "accepted": bool(exp.score.get("accepted")),
            "total_queries": exp.total_queries,
            "max_queries": exp.max_queries,
            "resource_usage": exp.resource_usage,
            "log_path": exp.log_path,
        })
        logger.info("Accepted=%s answer=%s", exp.score.get("accepted"), exp.scientist_answer[:160])

    payload = {
        "run_type": "rpg_static_agent",
        "scientist_backend": args.scientist_backend,
        "scientist_model": args.scientist_model,
        "n_worlds": len(worlds),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
