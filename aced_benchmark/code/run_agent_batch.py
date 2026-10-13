"""
run_agent_batch.py

Run the scientist agent on all worlds in a directory.
Output format is compatible with evaluate_zero_shot.py.

The world model (query parser) always runs locally on --world-model.
The scientist agent (reasoner) is configured separately via --scientist-*.

Usage:
    # Scientist: local Qwen (default)
    python run_agent_batch.py --worlds-dir ../dataset_generation_code/out_bn

    # Scientist: vllm (OpenAI-compatible local server)
    python run_agent_batch.py --worlds-dir ../dataset_generation_code/out_bn \
        --scientist-backend openai \
        --scientist-model Qwen/Qwen2.5-7B-Instruct \
        --scientist-base-url http://localhost:8000/v1

    # Scientist: OpenAI API
    python run_agent_batch.py --worlds-dir ../dataset_generation_code/out_bn \
        --scientist-backend openai \
        --scientist-model gpt-4o-mini

    # Evaluate results (same command as zero-shot)
    python evaluate_zero_shot.py results/agent_20250101_120000.json --details
"""

from __future__ import annotations

import argparse
import json
import logging
import glob
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from simulator import BNSimulator
from scientist_coder_agent import CoderScientistAgent
from scientist_coder_agent_new import CoderScientistAgent as CoderScientistAgentNew
from orchestrator import Orchestrator
from schemas import Question
from json_converter import JSONToBIFConverter

logger = logging.getLogger(__name__)


LIMIT_PRESETS: Dict[str, Dict[str, int]] = {
    "4_19_big": {"max_turns": 4, "max_total_samples": 12000},
    "adv_v3": {"max_turns": 4, "max_total_samples": 4000},
}


def infer_limit_preset(worlds_dir: Optional[str], world_json: Optional[str]) -> Optional[str]:
    """Infer resource-limit defaults from dataset paths."""
    joined = " ".join(p for p in (worlds_dir, world_json) if p).lower()
    if "adv_v3" in joined or "out_bn_adv" in joined:
        return "adv_v3"
    if "4_19_big" in joined or "out_bn_4_19" in joined:
        return "4_19_big"
    return None


def resolve_resource_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Resolve and validate the requested resource-limit variant."""
    if args.limit_variant == "none":
        if args.max_turns is not None or args.max_total_samples is not None or args.max_samples_per_query is not None:
            parser.error(
                "--max-turns/--max-total-samples/--max-samples-per-query require "
                "--limit-variant rounds or --limit-variant samples"
            )
        return {
            "limit_variant": "none",
            "limit_preset": None,
            "max_turns": None,
            "max_total_samples": None,
            "max_samples_per_query": None,
            "sample_accounting": args.sample_accounting,
        }

    if args.limit_variant == "rounds" and (
        args.max_total_samples is not None
        or args.max_samples_per_query is not None
    ):
        parser.error(
            "--limit-variant rounds only accepts --max-turns. Use "
            "--limit-variant samples for sample budgets."
        )

    if args.limit_variant == "samples" and args.max_turns is not None:
        parser.error(
            "--limit-variant samples only accepts sample budget knobs. Use "
            "--limit-variant rounds for --max-turns."
        )

    preset_name = args.limit_preset
    if preset_name == "auto":
        preset_name = infer_limit_preset(args.worlds_dir, args.world_json)
        if preset_name is None:
            has_explicit_limit = (
                (args.limit_variant == "rounds" and args.max_turns is not None)
                or (
                    args.limit_variant == "samples"
                    and args.max_total_samples is not None
                )
            )
            if not has_explicit_limit:
                parser.error(
                    "--limit-preset auto could not infer the dataset. Use "
                    "--limit-preset 4_19_big or --limit-preset adv_v3, or pass "
                    "explicit limits."
                )

    preset = LIMIT_PRESETS.get(preset_name, {})

    if args.limit_variant == "rounds":
        max_turns = args.max_turns if args.max_turns is not None else preset.get("max_turns")
        if max_turns is None:
            parser.error("--limit-variant rounds requires --max-turns or a known --limit-preset")
        return {
            "limit_variant": "rounds",
            "limit_preset": preset_name,
            "max_turns": max_turns,
            "max_total_samples": None,
            "max_samples_per_query": None,
            "sample_accounting": args.sample_accounting,
        }

    max_total_samples = (
        args.max_total_samples
        if args.max_total_samples is not None
        else preset.get("max_total_samples")
    )
    if max_total_samples is None:
        parser.error(
            "--limit-variant samples requires --max-total-samples or a known --limit-preset"
        )
    return {
        "limit_variant": "samples",
        "limit_preset": preset_name,
        "max_turns": None,
        "max_total_samples": max_total_samples,
        "max_samples_per_query": args.max_samples_per_query,
        "sample_accounting": args.sample_accounting,
    }


def resource_limits_compatible(expected: Dict[str, Any], actual: Optional[Dict[str, Any]]) -> bool:
    """Return whether an existing per-experiment log matches this run's limits."""
    if actual is None:
        return expected["limit_variant"] == "none"
    keys = [
        "limit_variant",
        "max_turns",
        "max_total_samples",
        "max_samples_per_query",
        "sample_accounting",
    ]
    return all(actual.get(k) == expected.get(k) for k in keys)


# ---------------------------------------------------------------------------
# World loading (mirrors run_zero_shot.py)
# ---------------------------------------------------------------------------

def load_worlds(
    worlds_dir: Optional[str] = None,
    world_json: Optional[str] = None,
) -> List:
    """Load world JSON files. Returns list of (filepath, parsed_json)."""
    results = []
    if world_json:
        paths = [world_json]
    elif worlds_dir:
        paths = sorted(glob.glob(os.path.join(worlds_dir, "world_*.json")))
    else:
        raise ValueError("Provide --worlds-dir or --world-json")

    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("questions"):
                results.append((p, data))
            else:
                logger.warning(f"Skipping {p}: no questions found")
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")
    return results


# ---------------------------------------------------------------------------
# Per-world setup
# ---------------------------------------------------------------------------

def setup_world_model(filepath: str, llm, output_dir: str, world_model_cls):
    """Create a BNSimulator + WorldModel from a world JSON file.

    The *llm* is duck-typed — any object with
    ``generate(system_prompt, user_prompt, max_new_tokens=None)`` works
    (both QwenLLM and ScientistLLM satisfy this).
    """
    converter = JSONToBIFConverter(filepath)
    errors = converter.validate()
    if errors:
        raise ValueError(f"Validation errors: {errors}")

    tmp_bif = Path(tempfile.gettempdir()) / f"_world_{Path(filepath).stem}.bif"
    converter.convert(str(tmp_bif))
    simulator = BNSimulator.from_bif(str(tmp_bif))
    config = converter.get_world_config()

    world_model = world_model_cls(simulator=simulator, llm=llm, output_dir=output_dir)
    world_model.set_variable_descriptions(config["variable_descriptions"])
    if config.get("non_intervenable_variables"):
        world_model.set_non_intervenable_variables(config["non_intervenable_variables"])
    if config.get("story"):
        world_model.set_story(config["story"])

    return world_model, config


def load_done_experiments(log_dir: str) -> Dict:
    """Scan log_dir for experiment_*.json files. Return dict keyed by
    (basename(dataset_file), question_id) -> (log_path, parsed log dict).

    Used to resume a partially-completed batch by skipping questions
    whose per-experiment log already exists.
    """
    done: Dict = {}
    if not os.path.isdir(log_dir):
        return done
    for log_path in sorted(glob.glob(os.path.join(log_dir, "experiment_*.json"))):
        try:
            with open(log_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse {log_path}: {e}")
            continue
        ds = data.get("dataset_file")
        qid = data.get("question", {}).get("metadata", {}).get("id")
        if ds is None or qid is None:
            continue
        key = (os.path.basename(ds), qid)
        # Keep the most recent log if duplicates exist (sorted ascending → last wins)
        done[key] = (log_path, data)
    return done


def result_from_prior_log(log_data: Dict, log_path: str, filepath: str,
                          world_name: str, topic: str, n_nodes: int,
                          topology: str,
                          question: Question) -> Dict[str, Any]:
    """Re-create the aggregated-results row from a previously saved experiment log."""
    turns = log_data.get("turns", []) or []
    last_turn = turns[-1] if turns else {}
    return {
        "world_file": filepath,
        "world_name": world_name,
        "topic": topic,
        "n_nodes": n_nodes,
        "topology": topology,
        "question_id": question.metadata.get("id", 0),
        "question_text": question.question_text,
        "question_type": question.question_type,
        "difficulty": question.metadata.get("difficulty", ""),
        "ground_truth": question.ground_truth,
        "raw_response": last_turn.get("raw_llm_response", ""),
        "extracted_answer": log_data.get("scientist_answer", ""),
        "total_queries": log_data.get("total_queries", 0),
        "max_queries": log_data.get("max_queries", 0),
        "resource_limits": log_data.get("resource_limits"),
        "resource_usage": log_data.get("resource_usage"),
        "resumed_from": log_path,
    }


def questions_from_config(config: dict) -> List[Question]:
    """Convert question dicts from a world config into Question objects."""
    questions = []
    for q in config.get("questions", []):
        questions.append(Question(
            question_type=q["question_type"],
            question_text=q["question_text"],
            ground_truth=q["ground_truth"],
            metadata={k: v for k, v in q.items()
                      if k not in ("question_type", "question_text", "ground_truth")},
        ))
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run scientist agent on all worlds (output compatible with evaluate_zero_shot.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--worlds-dir", default=None,
                        help="Directory containing world_*.json files")
    parser.add_argument("--world-json", default=None,
                        help="Path to a single world JSON file")

    # World model (query parser) — configurable backend
    parser.add_argument("--world-model", default="us.anthropic.claude-opus-4-7",
                        help="Model name/ID for the world-model query parser. "
                             "Interpreted per --world-model-backend: HF repo id for 'local', "
                             "API model name for 'openai', Bedrock model id for 'bedrock'.")
    parser.add_argument("--world-model-backend", choices=["local", "openai", "bedrock"], default="bedrock",
                        help="World-model parser backend: 'local' (HuggingFace), "
                             "'openai' (OpenAI-compatible API / vllm), or 'bedrock' (AWS Bedrock). "
                             "Use a stronger backend if variable names are out-of-distribution tokens "
                             "(e.g. nonsense-name datasets) — local small models can hallucinate names.")
    parser.add_argument("--world-model-base-url", default=None,
                        help="API base URL for the world-model parser (openai backend). "
                             "Falls back to OPENAI_BASE_URL env var.")
    parser.add_argument("--world-model-api-key", default=None,
                        help="API key for the world-model parser (openai backend). "
                             "Falls back to OPENAI_API_KEY env var.")

    # Scientist agent — configurable backend
    parser.add_argument("--scientist-backend", choices=["local", "openai", "bedrock", "gemini"], default="bedrock",
                        help="Scientist backend: 'local' (HuggingFace), 'openai' (OpenAI-compatible API / vllm), "
                             "'bedrock' (AWS Bedrock), or 'gemini' (Google AI Studio / Vertex)")
    parser.add_argument("--scientist-model", default=None,
                        help="Scientist model name. Defaults to --world-model when omitted.")
    parser.add_argument("--scientist-base-url", default=None,
                        help="API base URL for scientist (openai backend). For vllm: http://localhost:8000/v1. Falls back to OPENAI_BASE_URL env var")
    parser.add_argument("--scientist-api-key", default=None,
                        help="API key for scientist (openai backend). Falls back to OPENAI_API_KEY env var")

    parser.add_argument("--causal", action="store_true", default=True,
                        help="(default: always on) Use causal variants: world_model_causal + scientist_agent_causal")
    parser.add_argument("--agent-type", choices=["agent", "coder", "coder_new"], default="agent",
                        help="'agent' = ScientistAgent, 'coder' = CoderScientistAgent, "
                             "'coder_new' = modular CoderScientistAgent (INIT/CODE/ANALYSIS/DESIGN)")
    parser.add_argument("--max-queries", "-n", type=int, default=10,
                        help="Max data queries per question (default: 10)")
    parser.add_argument("--limit-variant", choices=["none", "rounds", "samples"], default="none",
                        help="Resource-limit ablation to run: none, rounds, or samples")
    parser.add_argument("--limit-preset", choices=["auto", "4_19_big", "adv_v3"], default="auto",
                        help="Dataset-aware defaults for resource limits")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Maximum outer scientist turns for --limit-variant rounds")
    parser.add_argument("--max-total-samples", type=int, default=None,
                        help="Total sample budget per question for --limit-variant samples")
    parser.add_argument("--max-samples-per-query", type=int, default=None,
                        help="Optional per-query sample budget for --limit-variant samples")
    parser.add_argument("--sample-accounting", choices=["rows", "cells"], default="rows",
                        help="Accounting unit for sample budgets")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--run-name", default=None,
                        help="Shorthand: sets --output-dir to ./results/<run-name> and "
                             "-o to ./results/<run-name>/<run-name>.json. Explicit --output-dir / -o override.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: ./results/agent_<timestamp>.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for query CSVs and per-experiment logs (default: ./results)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable auto-skip of questions already completed in --output-dir/agent_logs/")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    resource_limits = resolve_resource_limits(args, parser)

    # --run-name shorthand: derive --output-dir and -o unless explicitly set.
    if args.run_name:
        if args.output_dir is None:
            args.output_dir = f"./results/{args.run_name}"
        if args.output is None:
            args.output = os.path.join(args.output_dir, f"{args.run_name}.json")
    if args.output_dir is None:
        args.output_dir = "./results"

    # Always use causal variants (world_model_causal + scientist_agent_causal)
    from world_model_causal import WorldModel, QwenLLM, OpenAILLM
    from scientist_agent_causal import ScientistAgent, ScientistLLM

    if args.scientist_backend == "bedrock" or args.world_model_backend == "bedrock":
        from bedrock_llm import BedrockLLM
    if args.scientist_backend == "gemini":
        from gemini_llm import GeminiLLM

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------
    # Load worlds
    # ------------------------------------------------------------------
    worlds = load_worlds(worlds_dir=args.worlds_dir, world_json=args.world_json)
    if not worlds:
        logger.error("No worlds found.")
        sys.exit(1)

    total_q = sum(len(w["questions"]) for _, w in worlds)
    logger.info(f"Loaded {len(worlds)} world(s) with {total_q} total question(s)")

    # ------------------------------------------------------------------
    # Prepare output dirs (needed before resume scan)
    # ------------------------------------------------------------------
    query_data_dir = os.path.join(args.output_dir, "agent_query_data")
    log_dir = os.path.join(args.output_dir, "agent_logs")
    os.makedirs(query_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume support: scan log_dir BEFORE loading any LLM so we can skip
    # heavy model loading if everything is already done.
    # ------------------------------------------------------------------
    done_experiments: Dict = {} if args.no_resume else load_done_experiments(log_dir)
    if done_experiments:
        incompatible = [
            path for path, data in (v for v in done_experiments.values())
            if not resource_limits_compatible(resource_limits, data.get("resource_limits"))
        ]
        if incompatible:
            logger.error(
                "Resume refused: found %d completed log(s) with different "
                "resource limits in %s. Use a distinct --run-name or --no-resume.",
                len(incompatible),
                log_dir,
            )
            logger.error("First incompatible log: %s", incompatible[0])
            sys.exit(2)
        logger.info(f"Resume: found {len(done_experiments)} completed experiment log(s) in {log_dir} — will skip those questions")

    n_pending = sum(
        1
        for fp, w in worlds
        for q in (w.get("questions") or [])
        if (os.path.basename(fp), q.get("id")) not in done_experiments
    )
    logger.info(f"Pending questions to run: {n_pending}/{total_q}")

    # ------------------------------------------------------------------
    # Build world model LLM (configurable backend, used for query parsing)
    # ------------------------------------------------------------------
    if n_pending == 0:
        logger.info("Nothing to do — skipping LLM load")
        world_llm = None
        scientist_llm = None
        scientist_model = args.scientist_model or args.world_model
    else:
        logger.info(f"Loading world model LLM: {args.world_model} (backend={args.world_model_backend})")
        # Parser wants near-deterministic output; keep temperature low regardless of backend.
        if args.world_model_backend == "openai":
            world_llm = OpenAILLM(
                model_name=args.world_model,
                base_url=args.world_model_base_url,
                api_key=args.world_model_api_key,
                max_new_tokens=512,
                temperature=0.1,
            )
        elif args.world_model_backend == "bedrock":
            world_llm = BedrockLLM(
                model_id=args.world_model,
                max_new_tokens=512,
                temperature=0.1,
            )
        else:
            world_llm = QwenLLM(
                model_name=args.world_model,
                max_new_tokens=512,
                temperature=0.1,
            )

        # ------------------------------------------------------------------
        # Build scientist LLM (configurable)
        # ------------------------------------------------------------------
        scientist_model = args.scientist_model or args.world_model
        logger.info(f"Loading scientist LLM: {scientist_model} (backend={args.scientist_backend})")
        if args.scientist_backend == "openai":
            scientist_llm = OpenAILLM(
                model_name=scientist_model,
                base_url=args.scientist_base_url,
                api_key=args.scientist_api_key,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif args.scientist_backend == "bedrock":
            scientist_llm = BedrockLLM(
                model_id=scientist_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif args.scientist_backend == "gemini":
            scientist_llm = GeminiLLM(
                model_id=scientist_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        else:
            scientist_llm = ScientistLLM(
                model_name=scientist_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    n_resumed = 0
    done = 0

    for wi, (filepath, world_data) in enumerate(worlds):
        world_name = Path(filepath).stem
        topic = world_data.get("meta", {}).get("topic", "")
        n_nodes = world_data.get("meta", {}).get("n_nodes", 0)
        topology = world_data.get("meta", {}).get("topology", "")
        n_questions_in_world = len(world_data.get("questions", []) or [])
        logger.info(f"[{wi + 1}/{len(worlds)}] {world_name} ({topic}, n={n_nodes})")

        # Quick pre-check: which question ids in this world are NOT done yet?
        world_basename = os.path.basename(filepath)
        pending_ids = {
            q.get("id") for q in (world_data.get("questions") or [])
            if (world_basename, q.get("id")) not in done_experiments
        }

        if not pending_ids:
            # Re-create result rows from the prior logs so the aggregated
            # output JSON still contains every question.
            for q in world_data.get("questions") or []:
                qid = q.get("id")
                key = (world_basename, qid)
                log_path, prior = done_experiments[key]
                done += 1
                logger.info(f"  [{done}/{total_q}] q{qid}: SKIP (already done in {os.path.basename(log_path)})")
                question = Question(
                    question_type=q["question_type"],
                    question_text=q.get("question", ""),
                    ground_truth=q.get("answer", ""),
                    metadata={k: v for k, v in q.items()
                              if k not in ("question_type", "question", "answer")},
                )
                all_results.append(result_from_prior_log(
                    prior, log_path, filepath, world_name, topic, n_nodes, topology, question,
                ))
                n_resumed += 1
            logger.info(f"  All {n_questions_in_world} questions already done — skipping world setup")
            continue

        try:
            world_model, config = setup_world_model(filepath, world_llm, query_data_dir, WorldModel)
        except Exception as e:
            import traceback
            reason = f"{type(e).__name__}: {e}"
            logger.error(f"SKIPPING world {world_name}: {reason}")
            skipped.append({
                "world_file": filepath,
                "world_name": world_name,
                "topic": topic,
                "n_nodes": n_nodes,
                "topology": topology,
                "n_questions_skipped": len(pending_ids),
                "stage": "world_setup",
                "error": reason,
                "traceback": traceback.format_exc(),
            })
            done += n_questions_in_world
            continue

        questions = questions_from_config(config)

        for question in questions:
            done += 1
            q_id = question.metadata.get("id", 0)

            # Resume: if a log already exists for this (world, qid), skip and
            # re-emit the prior result row.
            key = (world_basename, q_id)
            if key in done_experiments:
                log_path, prior = done_experiments[key]
                logger.info(f"  [{done}/{total_q}] q{q_id}: SKIP (already done in {os.path.basename(log_path)})")
                all_results.append(result_from_prior_log(
                    prior, log_path, filepath, world_name, topic, n_nodes, topology, question,
                ))
                n_resumed += 1
                continue

            logger.info(f"  [{done}/{total_q}] q{q_id}: {question.question_text[:80]}...")

            try:
                # Fresh scientist per question
                if args.agent_type == "coder":
                    scientist = CoderScientistAgent(llm=scientist_llm)
                elif args.agent_type == "coder_new":
                    scientist = CoderScientistAgentNew(llm=scientist_llm)
                else:
                    scientist = ScientistAgent(llm=scientist_llm)

                orchestrator = Orchestrator(
                    world_model=world_model,
                    scientist=scientist,
                    question=question,
                    max_queries=args.max_queries,
                    log_dir=log_dir,
                    scientist_model=scientist_model,
                    world_model_name=args.world_model,
                    dataset_file=filepath,
                    limit_variant=resource_limits["limit_variant"],
                    max_turns=resource_limits["max_turns"],
                    max_total_samples=resource_limits["max_total_samples"],
                    max_samples_per_query=resource_limits["max_samples_per_query"],
                    sample_accounting=resource_limits["sample_accounting"],
                )

                exp_result = orchestrator.run()

                # Build result dict in the same format as run_zero_shot.py
                result = {
                    "world_file": filepath,
                    "world_name": world_name,
                    "topic": topic,
                    "n_nodes": n_nodes,
                    "topology": topology,
                    "question_id": q_id,
                    "question_text": question.question_text,
                    "question_type": question.question_type,
                    "difficulty": question.metadata.get("difficulty", ""),
                    "ground_truth": question.ground_truth,
                    "raw_response": (exp_result.turns[-1].raw_llm_response
                                     if exp_result.turns else ""),
                    "extracted_answer": exp_result.scientist_answer,
                    # Agent-specific extras (ignored by evaluate_zero_shot.py but useful)
                    "total_queries": exp_result.total_queries,
                    "max_queries": exp_result.max_queries,
                    "resource_limits": exp_result.resource_limits,
                    "resource_usage": exp_result.resource_usage,
                }
                all_results.append(result)
                logger.info(f"  Answer: {exp_result.scientist_answer[:100]}")

            except Exception as e:
                import traceback
                reason = f"{type(e).__name__}: {e}"
                logger.error(f"SKIPPING {world_name} q{q_id}: {reason}")
                skipped.append({
                    "world_file": filepath,
                    "world_name": world_name,
                    "topic": topic,
                    "n_nodes": n_nodes,
                    "topology": topology,
                    "question_id": q_id,
                    "question_text": question.question_text,
                    "n_questions_skipped": 1,
                    "stage": "orchestrator_run",
                    "error": reason,
                    "traceback": traceback.format_exc(),
                })

    # ------------------------------------------------------------------
    # Save — same top-level schema as run_zero_shot.py
    # ------------------------------------------------------------------
    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"agent_{timestamp}.json")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": scientist_model,
        "world_model": args.world_model,
        "scientist_backend": args.scientist_backend,
        "method": args.agent_type,
        "max_queries": args.max_queries,
        "limit_variant": resource_limits["limit_variant"],
        "resource_limits": resource_limits,
        "timestamp": datetime.now().isoformat(),
        "n_worlds": len(worlds),
        "n_questions": len(all_results),
        "n_resumed": n_resumed,
        "n_skipped": sum(s["n_questions_skipped"] for s in skipped),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Write skip report if anything was skipped
    if skipped:
        skip_path = str(Path(out_path).with_suffix("")) + "_skipped.json"
        with open(skip_path, "w") as f:
            json.dump(skipped, f, indent=2)
        n_skipped_q = sum(s["n_questions_skipped"] for s in skipped)
        logger.error(f"{len(skipped)} skip(s) ({n_skipped_q} questions). Details: {skip_path}")
        print(f"\nWARNING: {n_skipped_q} questions skipped. See {skip_path}")

    logger.info(f"Results saved to {out_path}")
    n_new = len(all_results) - n_resumed
    print(f"\nDone. {len(all_results)}/{total_q} questions answered "
          f"({n_new} new this run, {n_resumed} resumed from prior logs). Results: {out_path}")
    print(f"Evaluate with: python evaluate_zero_shot.py {out_path} --details")


if __name__ == "__main__":
    main()
