"""
run_zero_shot_sub_prompt.py

Zero-shot baseline using a reduced version of the scientist agent's prompt.
Only the "answer" action type is available — no data queries, no code execution.
The LLM must reason about causal structure from domain knowledge alone.

Usage:
    python run_zero_shot_sub_prompt.py --worlds-dir ../dataset_generation_code/out_bn

    python run_zero_shot_sub_prompt.py --world-json ../dataset_generation_code/out_bn/world_Education_n10_seed1003.json

    python run_zero_shot_sub_prompt.py --worlds-dir ../dataset_generation_code/out_bn --backend openai --model gpt-4o

    python run_zero_shot_sub_prompt.py --worlds-dir ../dataset_generation_code/out_bn -o ./results/zero_shot_sub.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM wrappers (same as run_zero_shot.py)
# ---------------------------------------------------------------------------

class OpenAIZeroShotLLM:
    """Zero-shot LLM backed by an OpenAI-compatible API."""
    def __init__(self, model_name: str = "gpt-4o",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1):
        from openai import OpenAI
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        logger.info(f"OpenAI zero-shot LLM ready — model={model_name}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


class ZeroShotLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: Optional[str] = None,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32

        logger.info(f"Loading LLM: {model_name} on {self._device} ...")
        local_dir = snapshot_download(model_name, local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=dtype, trust_remote_code=True,
        ).to(self._device)

        self.model.eval()
        logger.info("LLM loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prompt construction — reduced scientist prompt (answer-only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a discovery scientist. Your goal is to determine relationships between variables in an unknown causal graph.

You have NO access to data — you cannot request observational or interventional samples. You must reason from domain knowledge alone.

AVAILABLE ACTIONS:
1. ANSWER - Submit your final answer:
   - For Yes/No questions: answer "Yes" or "No"
   - For listing questions: provide the specific variable names in a list

CAUSAL DISCOVERY STRATEGY:

Think like a scientist reasoning about experiments.

You have two conceptual tools (but NO data access):
1. Observational data → shows correlations
2. Interventions do(X=value) → shows causal effects

Key ideas:

- Correlation alone cannot determine causation.
- To test if X causes Y:
    → change X (using do(X=...))
    → see if Y changes

Interpretation:
- If Y changes when X is changed → X causally affects Y
- If Y does NOT change → X does NOT cause Y
- If X and Y are correlated but no effect under intervention → likely common cause

Since you have no data, you must use your domain knowledge to reason about:
- Which variables are likely causally connected based on the domain
- What the likely causal directions are
- Whether variables are independent or dependent

Before answering, ensure:
- Your conclusion is based on careful causal reasoning about the domain
- You have considered alternative causal structures

OUTPUT FORMAT (required, in this exact order):
<reasoning>[Analysis, updated understanding, hypothesis, decision rationale]</reasoning>
<action type="answer">[Your final answer]</action>
Do NOT copy the example text — write your actual analysis and actual answer."""


def build_user_prompt(world: Dict[str, Any], question: Dict[str, Any]) -> str:
    """Build the user prompt from world JSON and a question dict.

    Mirrors the structure of ScientistAgent._get_decision_user_prompt() from
    scientist_agent_causal.py, but with no query history, budget, or data
    sections (zero-shot: domain-knowledge reasoning only).
    """
    lines = []

    # =======================================================================
    # SECTION 1: GOAL (same as scientist agent)
    # =======================================================================
    lines.append("════════════════════════════════════════════════════════════════════════════════")
    lines.append(f"YOUR QUESTION: {question['question']}")
    lines.append("════════════════════════════════════════════════════════════════════════════════")
    lines.append("")

    # =======================================================================
    # SECTION 2: VARIABLES (same as scientist agent — full catalog)
    # =======================================================================
    lines.append("AVAILABLE VARIABLES (use exact names and state values for interventions):")
    for var in world["variables"]:
        states = ", ".join(var["values"])
        lines.append(f"  {var['name']}: {var['desc']} (states: {states})")
    lines.append("")

    lines.append(f"CONTEXT: {world['story']}")
    lines.append("")

    # =======================================================================
    # SECTION 3: CONSTRAINTS (non-intervenable variables, matching scientist agent)
    # =======================================================================
    if world.get("non_intervenable_variables"):
        ni_vars = world["non_intervenable_variables"]
        lines.append("INTERVENTION LIMITS — Cannot intervene on (non-manipulable variables):")
        if isinstance(ni_vars, list):
            for entry in ni_vars:
                lines.append(f"  - {entry['name']}: {entry['reason']}")
        elif isinstance(ni_vars, dict):
            for name, reason in ni_vars.items():
                lines.append(f"  - {name}: {reason}")
    else:
        lines.append("INTERVENTION LIMITS: All variables are intervenable")
    lines.append("")

    # =======================================================================
    # SECTION 4: NO DATA NOTICE (replaces query history / latest result)
    # =======================================================================
    lines.append("─── NO DATA AVAILABLE ───")
    lines.append("You have no access to observational or interventional data.")
    lines.append("You must reason from domain knowledge alone.")
    lines.append("")

    # =======================================================================
    # ASSEMBLE (same closing as scientist agent)
    # =======================================================================
    lines.append("=" * 80)
    lines.append("Now: Use your domain knowledge and causal reasoning to answer the question.")
    lines.append("Output: <reasoning> and <action type=\"answer\"> blocks (in that order).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    """Extract the content from <action type="answer">...</action> tags."""
    # Strip think blocks (Qwen-style)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Try <action type="answer">...</action>
    match = re.search(
        r'<action\s+type="answer">\s*(.*?)\s*</action>',
        response, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: try <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Last resort: last non-empty line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()


def extract_reasoning(response: str) -> str:
    """Extract the content from <reasoning>...</reasoning> tags."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: everything before the <action> tag
    match = re.search(r'<action', response, re.IGNORECASE)
    if match:
        return response[:match.start()].strip()

    return response.strip()


# ---------------------------------------------------------------------------
# World loading
# ---------------------------------------------------------------------------

def load_worlds(worlds_dir: Optional[str] = None,
                world_json: Optional[str] = None) -> List:
    """Load world JSON files. Returns list of (filepath, parsed_json) tuples."""
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
# Main runner
# ---------------------------------------------------------------------------

def run_zero_shot(worlds: List, llm) -> List[Dict[str, Any]]:
    """Run zero-shot sub-prompt baseline on all worlds/questions."""
    all_results = []
    total_questions = sum(len(w["questions"]) for _, w in worlds)
    done = 0

    for filepath, world in worlds:
        world_name = Path(filepath).stem
        for question in world["questions"]:
            done += 1
            q_id = question.get("id", 0)
            logger.info(f"[{done}/{total_questions}] {world_name} q{q_id}: {question['question'][:80]}...")

            user_prompt = build_user_prompt(world, question)
            raw_response = llm.generate(SYSTEM_PROMPT, user_prompt)
            extracted = extract_answer(raw_response)

            result = {
                "world_file": filepath,
                "world_name": world_name,
                "topic": world.get("meta", {}).get("topic", ""),
                "n_nodes": world.get("meta", {}).get("n_nodes", 0),
                "question_id": q_id,
                "question_text": question["question"],
                "question_type": question.get("question_type", ""),
                "difficulty": question.get("difficulty", ""),
                "ground_truth": question["answer"],
                "raw_response": raw_response,
                "reasoning": extract_reasoning(raw_response),
                "extracted_answer": extracted,
            }
            all_results.append(result)
            logger.info(f"  Answer: {extracted[:100]}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline using scientist sub-prompt (answer-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--worlds-dir", default=None,
                        help="Directory containing world_*.json files")
    parser.add_argument("--world-json", default=None,
                        help="Path to a single world JSON file")
    parser.add_argument("--backend", choices=["local", "openai", "bedrock"], default="local",
                        help="'local' = HuggingFace model, 'openai' = OpenAI-compatible API, 'bedrock' = AWS Bedrock")
    parser.add_argument("--model", default=None,
                        help="Model name (HuggingFace id for local, or e.g. gpt-4o for openai). "
                             "Defaults to Qwen/Qwen2.5-7B-Instruct for local.")
    parser.add_argument("--base-url", default=None,
                        help="API base URL for openai backend (falls back to OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key for openai backend (falls back to OPENAI_API_KEY)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: ./results/zero_shot_sub_<timestamp>.json)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load worlds
    worlds = load_worlds(worlds_dir=args.worlds_dir, world_json=args.world_json)
    if not worlds:
        logger.error("No worlds found.")
        sys.exit(1)

    total_q = sum(len(w["questions"]) for _, w in worlds)
    logger.info(f"Loaded {len(worlds)} world(s) with {total_q} total question(s)")

    # Load LLM
    if args.backend == "openai":
        if not args.model:
            logger.error("--model is required for openai backend (e.g. gpt-4o)")
            sys.exit(1)
        llm = OpenAIZeroShotLLM(
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "bedrock":
        if not args.model:
            logger.error("--model is required for bedrock backend")
            sys.exit(1)
        from bedrock_llm import BedrockLLM
        llm = BedrockLLM(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        model_name = args.model or "Qwen/Qwen2.5-7B-Instruct"
        llm = ZeroShotLLM(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Run
    results = run_zero_shot(worlds, llm)

    # Save — split into one file per difficulty group
    if args.output:
        base = args.output[:-5] if args.output.endswith(".json") else args.output
    else:
        os.makedirs("./results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"./results/zero_shot_sub_{timestamp}"

    Path(base).parent.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List] = defaultdict(list)
    for r in results:
        groups[r["difficulty"] or "unknown"].append(r)

    saved_paths = []
    run_ts = datetime.now().isoformat()
    for group_name in sorted(groups):
        group_results = groups[group_name]
        path = f"{base}_{group_name}.json"
        payload = {
            "run_metadata": {
                "model": args.model,
                "worlds_source": args.worlds_dir or args.world_json,
                "timestamp": run_ts,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "question_group": group_name,
                "n_worlds": len(worlds),
                "n_questions": len(group_results),
                "prompt_type": "scientist_sub_prompt",
            },
            "results": group_results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  [{group_name}] {len(group_results)} questions → {path}")
        saved_paths.append(path)

    print(f"\nDone. {len(results)} questions answered across {len(groups)} difficulty group(s):")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
