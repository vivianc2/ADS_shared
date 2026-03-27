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
from orchestrator import Orchestrator
from schemas import Question
from json_converter import JSONToBIFConverter

logger = logging.getLogger(__name__)


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

    # World model (query parser) — always runs locally
    parser.add_argument("--world-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model for the world model query parser (always local)")

    # Scientist agent — configurable backend
    parser.add_argument("--scientist-backend", choices=["local", "openai", "bedrock"], default="local",
                        help="Scientist backend: 'local' (HuggingFace), 'openai' (OpenAI-compatible API / vllm), or 'bedrock' (AWS Bedrock)")
    parser.add_argument("--scientist-model", default=None,
                        help="Scientist model name. Defaults to --world-model for local, required for openai")
    parser.add_argument("--scientist-base-url", default=None,
                        help="API base URL for scientist (openai backend). For vllm: http://localhost:8000/v1. Falls back to OPENAI_BASE_URL env var")
    parser.add_argument("--scientist-api-key", default=None,
                        help="API key for scientist (openai backend). Falls back to OPENAI_API_KEY env var")

    parser.add_argument("--causal", action="store_true", default=True,
                        help="(default: always on) Use causal variants: world_model_causal + scientist_agent_causal")
    parser.add_argument("--agent-type", choices=["agent", "coder"], default="agent",
                        help="'agent' = ScientistAgent, 'coder' = CoderScientistAgent with Python tool")
    parser.add_argument("--max-queries", "-n", type=int, default=10,
                        help="Max data queries per question (default: 10)")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: ./results/agent_<timestamp>.json)")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for query CSVs and per-experiment logs")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Always use causal variants (world_model_causal + scientist_agent_causal)
    from world_model_causal import WorldModel, QwenLLM, OpenAILLM
    from scientist_agent_causal import ScientistAgent, ScientistLLM

    if args.scientist_backend == "bedrock":
        from bedrock_llm import BedrockLLM

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
    # Build world model LLM (always local Qwen, used for query parsing)
    # ------------------------------------------------------------------
    logger.info(f"Loading world model LLM: {args.world_model} (local)")
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
    else:
        scientist_llm = ScientistLLM(
            model_name=scientist_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # ------------------------------------------------------------------
    # Prepare output dirs
    # ------------------------------------------------------------------
    query_data_dir = os.path.join(args.output_dir, "agent_query_data")
    log_dir = os.path.join(args.output_dir, "agent_logs")
    os.makedirs(query_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    done = 0

    for wi, (filepath, world_data) in enumerate(worlds):
        world_name = Path(filepath).stem
        topic = world_data.get("meta", {}).get("topic", "")
        n_nodes = world_data.get("meta", {}).get("n_nodes", 0)
        n_questions_in_world = len(world_data.get("questions", []))
        logger.info(f"[{wi + 1}/{len(worlds)}] {world_name} ({topic}, n={n_nodes})")

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
                "n_questions_skipped": n_questions_in_world,
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
            logger.info(f"  [{done}/{total_q}] q{q_id}: {question.question_text[:80]}...")

            try:
                # Fresh scientist per question
                if args.agent_type == "coder":
                    scientist = CoderScientistAgent(llm=scientist_llm)
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
                )

                exp_result = orchestrator.run()

                # Build result dict in the same format as run_zero_shot.py
                result = {
                    "world_file": filepath,
                    "world_name": world_name,
                    "topic": topic,
                    "n_nodes": n_nodes,
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
        "timestamp": datetime.now().isoformat(),
        "n_worlds": len(worlds),
        "n_questions": len(all_results),
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
    print(f"\nDone. {len(all_results)}/{total_q} questions answered. Results: {out_path}")
    print(f"Evaluate with: python evaluate_zero_shot.py {out_path} --details")


if __name__ == "__main__":
    main()
