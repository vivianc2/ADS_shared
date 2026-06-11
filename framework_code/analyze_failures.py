#!/usr/bin/env python3
"""
analyze_failures.py

Analyzes failure patterns in agent experiment logs using Claude via AWS Bedrock.

Two-phase pipeline:
  Phase 1: Per-log LLM analysis — classifies each failure into categories,
            identifies which turn things went wrong and why.
  Phase 2: Per-question-type synthesis — finds patterns across sampled failures.

Usage:
    python analyze_failures.py \
        --results-dir results/opus_coder_3_4 \
        --eval-file evaluations/eval_opus_coder_3_4.json \
        --samples-per-type 5 \
        --output-dir analysis_output/opus_coder_3_4

    # Faster: skip re-analyzing, just re-synthesize from cached per-log analyses
    python analyze_failures.py ... --synthesis-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── local import ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from bedrock_llm import BedrockLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

FAILURE_CATEGORIES = [
    "format_parsing_error",      # Right reasoning, wrong XML/action format
    "reasoning_error",           # Wrong causal or statistical inference
    "code_error",                # Code written but crashed or had logic bugs
    "poor_query_strategy",       # Budget wasted; wrong variables queried
    "world_model_error",         # Simulator/NL→query translation failed
    "instruction_misunderstanding",  # Agent misread or misunderstood the question
    "insufficient_evidence",     # Ran out of budget before conclusive evidence
    "early_wrong_conclusion",    # Committed to wrong answer too soon
    "other",
]

CORRECTNESS_QUALITY = [
    "genuine",        # Sound reasoning + sufficient evidence → correct answer
    "lucky_guess",    # Correct answer but via flawed/absent reasoning or guessing
    "partial",        # Some valid reasoning but also relied on luck or incomplete evidence
    "unclear",        # Cannot determine from the log
]

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-opus-4-7"
MAX_LOG_CHARS = 12_000   # truncation limit per log sent to LLM
MAX_TURNS_DETAIL = 10    # show all turns up to this many; summarise beyond


# ══════════════════════════════════════════════════════════════════════════════
#  LOG INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def build_log_index(agent_logs_dir: Path) -> Dict[Tuple[str, int], Path]:
    """
    Scan agent_logs_dir and return a dict mapping
      (world_name, question_id) -> Path(log_file)

    world_name is derived from the dataset_file field inside each JSON.
    """
    index: Dict[Tuple[str, int], Path] = {}
    log_files = sorted(agent_logs_dir.glob("experiment_*.json"))
    log.info(f"Indexing {len(log_files)} log files in {agent_logs_dir} …")

    for lf in log_files:
        try:
            with open(lf) as f:
                data = json.load(f)
            # Extract world name from dataset_file path (stem without extension)
            ds_path = data.get("dataset_file", "")
            world_name = Path(ds_path).stem  # e.g. "world_Criminal_Justice_n10_seed1010"
            q_id = data.get("question", {}).get("metadata", {}).get("id")
            if q_id is None:
                q_id = data.get("question", {}).get("id")
            if world_name and q_id is not None:
                index[(world_name, int(q_id))] = lf
        except Exception as e:
            log.warning(f"Could not index {lf.name}: {e}")

    log.info(f"Indexed {len(index)} experiments.")
    return index


# ══════════════════════════════════════════════════════════════════════════════
#  LOG COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _trunc(s: str, n: int = 400) -> str:
    s = str(s).strip()
    return s[:n] + " …[truncated]" if len(s) > n else s


def compress_log(data: dict, max_chars: int = MAX_LOG_CHARS) -> str:
    """
    Produce a compact, human-readable text representation of a log file
    suitable for sending to an LLM for analysis.
    """
    q = data.get("question", {})
    lines = [
        "=== EXPERIMENT SUMMARY ===",
        f"Question type : {q.get('question_type', '?')}",
        f"Question      : {q.get('question_text', '?')}",
        f"Ground truth  : {data.get('ground_truth', '?')}",
        f"Agent answer  : {data.get('scientist_answer', '?')}",
        f"Correct       : {data.get('is_correct', '?')}",
        f"Queries used  : {data.get('total_queries', '?')} / {data.get('max_queries', 10)}",
        "",
    ]

    turns = data.get("turns", [])
    for t in turns:
        tn = t.get("turn_number", "?")
        tt = t.get("turn_type", "?")
        lines.append(f"--- Turn {tn} ({tt}) ---")

        reasoning = _trunc(t.get("reasoning", ""), 300)
        if reasoning:
            lines.append(f"Reasoning    : {reasoning}")

        sci_input = _trunc(t.get("scientist_input", ""), 200)
        if sci_input:
            lines.append(f"Agent request: {sci_input}")

        world_out = t.get("world_output", "")
        if world_out:
            lines.append(f"World output : {_trunc(str(world_out), 300)}")

        # Code rounds (coder agent)
        code_rounds = t.get("code_rounds", [])
        if code_rounds:
            lines.append(f"  [Code rounds: {len(code_rounds)}]")
            for i, cr in enumerate(code_rounds[:3]):  # show first 3
                code_snippet = _trunc(cr.get("code", ""), 200)
                output = _trunc(cr.get("output", ""), 200)
                error = _trunc(cr.get("error", ""), 200)
                lines.append(f"  Code round {i+1}:")
                if code_snippet:
                    lines.append(f"    code  : {code_snippet}")
                if output:
                    lines.append(f"    output: {output}")
                if error:
                    lines.append(f"    ERROR : {error}")

        # Raw action (last 300 chars is usually the <action> tag)
        raw = t.get("raw_llm_response", "")
        if raw:
            # Show tail which typically has the <action> tag
            tail = raw[-400:].strip()
            lines.append(f"Raw (tail)   : {_trunc(tail, 400)}")

        mem = _trunc(t.get("scientist_memory", ""), 200)
        if mem:
            lines.append(f"Memory       : {mem}")

        lines.append("")

    final_mem = _trunc(data.get("final_scientist_memory", ""), 400)
    if final_mem:
        lines.append(f"=== FINAL MEMORY ===\n{final_mem}\n")

    text = "\n".join(lines)

    # Hard truncation
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…[log truncated at character limit]"
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — PER-LOG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

PER_LOG_SYSTEM = """\
You are an expert in causal inference and LLM agent evaluation.
You are analyzing a failed experiment where an AI scientist agent tried to answer
a causal discovery question about a Bayesian Network by querying a simulator.

The agent had a budget of 10 queries (turns). Each turn the agent could request
observational or interventional data samples, run Python code to analyze them
(if it's a coder agent), or submit a final answer.

Respond ONLY with valid JSON matching this schema (no markdown, no prose):
{
  "primary_failure_category": "<one of the categories listed in the prompt>",
  "secondary_failure_categories": ["<optional additional categories>"],
  "failure_onset_turn": <integer — turn number where things first started going wrong, or 0 if from the start>,
  "queries_used_efficiently": <true | false>,
  "was_reasoning_sound_at_onset": <true | false — was reasoning OK before the failure turn?>,
  "failure_description": "<2-4 sentence explanation of exactly what went wrong>",
  "root_cause": "<1-2 sentences on the deeper cause — e.g. misunderstood question semantics, confused marginal vs conditional independence, etc.>",
  "could_have_been_fixed_by": "<brief suggestion — e.g. one more targeted intervention, correct XML formatting, etc.>"
}
"""

PER_LOG_USER_TEMPLATE = """\
Failure categories to choose from:
  - format_parsing_error      : Agent reached correct conclusion but used wrong XML/action format
  - reasoning_error           : Wrong causal or statistical inference (e.g. confused correlation with causation,
                                wrong d-separation reasoning, misinterpreted intervention results)
  - code_error                : Code was written but crashed, had logic bugs, or produced wrong statistics
  - poor_query_strategy       : Budget wasted on irrelevant variables or redundant queries; didn't ask the
                                right questions to answer this specific question type
  - world_model_error         : The world model (NL→query translator) failed or returned errors
  - instruction_misunderstanding : Agent misread or misunderstood what the question was asking
  - insufficient_evidence     : Ran out of query budget before collecting enough evidence for a confident answer
  - early_wrong_conclusion    : Committed to a wrong answer before finishing evidence collection
  - other                     : None of the above

Question type context:
  - causal_effect: Is there a causal path from X to Y? Use do-calculus (interventional queries).
  - direct_marginal / chain_marginal / fork_marginal / v_structure_marginal: Are X and Y marginally
    independent (no conditioning)? Test using observational marginal distributions.
  - other_conditional / chain_conditional / fork_conditional / v_structure_conditional: Are X and Y
    conditionally independent given Z? Test using conditional distributions P(X|Y,Z).
  - all_causes_of / all_effects_of: Find ALL upstream/downstream variables (not just direct neighbors).
    Typically requires systematic interventional testing across many variables.
  - direct_edge: Is there a direct causal edge from X to Y?

Here is the experiment log:

{log_text}

Analyze the failure and respond with JSON only.
"""


def analyze_single_log(
    llm: BedrockLLM,
    log_data: dict,
    log_path: Path,
    question_type: str,
) -> dict:
    """Call LLM to analyze one failing log. Returns parsed analysis dict."""
    log_text = compress_log(log_data)
    user_prompt = PER_LOG_USER_TEMPLATE.format(log_text=log_text)

    try:
        raw = llm.generate(PER_LOG_SYSTEM, user_prompt, max_new_tokens=1024)
        # Strip potential markdown code fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error for {log_path.name}: {e}\nRaw: {raw[:200]}")
        analysis = {
            "primary_failure_category": "other",
            "secondary_failure_categories": [],
            "failure_onset_turn": 0,
            "queries_used_efficiently": False,
            "was_reasoning_sound_at_onset": False,
            "failure_description": f"LLM returned non-JSON response: {raw[:300]}",
            "root_cause": "analysis failed",
            "could_have_been_fixed_by": "N/A",
        }
    except Exception as e:
        log.error(f"LLM call failed for {log_path.name}: {e}")
        analysis = {"error": str(e), "primary_failure_category": "other"}

    analysis["_log_file"] = log_path.name
    analysis["_question_type"] = question_type
    analysis["_question_text"] = log_data.get("question", {}).get("question_text", "")
    analysis["_ground_truth"] = log_data.get("ground_truth", "")
    analysis["_agent_answer"] = log_data.get("scientist_answer", "")
    analysis["_queries_used"] = log_data.get("total_queries", "?")
    analysis["_outcome"] = "failure"
    return analysis


# ══════════════════════════════════════════════════════════════════════════════
#  CORRECT-CASE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

CORRECT_LOG_SYSTEM = """\
You are an expert in causal inference and LLM agent evaluation.
You are analyzing a CORRECT experiment — the agent got the right answer.
Your job is to determine whether the agent arrived at the correct answer through
sound, principled reasoning, or whether it just got lucky (e.g. guessed, used
heuristics, skipped important steps, or reasoned incorrectly but happened to
land on the right answer).

Respond ONLY with valid JSON matching this schema (no markdown, no prose):
{
  "correctness_quality": "<one of: genuine | lucky_guess | partial | unclear>",
  "reasoning_was_sound": <true | false>,
  "evidence_was_sufficient": <true | false — did they collect enough data before answering?>,
  "strategy_was_appropriate": <true | false — did they use the right query type for this question?>,
  "queries_used_efficiently": <true | false>,
  "would_generalize": <true | false — would this approach reliably work on similar questions?>,
  "quality_description": "<2-4 sentences: what did the agent do well or not well, and why is this genuine/lucky?>",
  "lucky_elements": "<if lucky_guess or partial: what specifically was lucky or unsound?>",
  "genuine_elements": "<what aspects of the reasoning were actually correct and principled?>"
}
"""

CORRECT_LOG_USER_TEMPLATE = """\
Correctness quality categories:
  - genuine     : Sound causal/statistical reasoning + sufficient targeted evidence → correct answer.
                  The approach would reliably work on similar questions.
  - lucky_guess : Correct answer but via flawed reasoning, guessing, domain heuristics, or skipping
                  required evidence. The agent would likely fail on a harder variant.
  - partial     : Mixed — some valid reasoning steps but also relied on luck, incomplete evidence,
                  or domain shortcuts that happen to work here.
  - unclear     : The log doesn't contain enough information to judge.

Question type context:
  - causal_effect: Requires do-calculus (interventional queries). Observational data alone is insufficient.
  - direct_marginal / chain_marginal / fork_marginal / v_structure_marginal: Requires checking
    marginal distributions P(X), P(Y) — NOT conditional distributions.
  - other_conditional / chain_conditional / fork_conditional / v_structure_conditional: Requires
    conditional distributions P(X | Z), P(Y | Z). Must condition on the right variables.
  - all_causes_of / all_effects_of: Requires SYSTEMATIC testing of ALL plausible variables, not just
    a few obvious ones. Partial testing that gets lucky is not genuine.
  - direct_edge: Requires interventional evidence to distinguish direct from indirect effects.

Here is the experiment log:

{log_text}

Analyze the quality of the correct answer and respond with JSON only.
"""


def analyze_correct_log(
    llm: BedrockLLM,
    log_data: dict,
    log_path: Path,
    question_type: str,
) -> dict:
    """Call LLM to assess whether a correct answer was genuinely reasoned or lucky."""
    log_text = compress_log(log_data)
    user_prompt = CORRECT_LOG_USER_TEMPLATE.format(log_text=log_text)

    try:
        raw = llm.generate(CORRECT_LOG_SYSTEM, user_prompt, max_new_tokens=1024)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error (correct log) for {log_path.name}: {e}")
        analysis = {
            "correctness_quality": "unclear",
            "reasoning_was_sound": False,
            "evidence_was_sufficient": False,
            "strategy_was_appropriate": False,
            "queries_used_efficiently": False,
            "would_generalize": False,
            "quality_description": f"LLM returned non-JSON: {raw[:200]}",
            "lucky_elements": "",
            "genuine_elements": "",
        }
    except Exception as e:
        log.error(f"LLM call failed (correct log) for {log_path.name}: {e}")
        analysis = {"error": str(e), "correctness_quality": "unclear"}

    analysis["_log_file"] = log_path.name
    analysis["_question_type"] = question_type
    analysis["_question_text"] = log_data.get("question", {}).get("question_text", "")
    analysis["_ground_truth"] = log_data.get("ground_truth", "")
    analysis["_agent_answer"] = log_data.get("scientist_answer", "")
    analysis["_queries_used"] = log_data.get("total_queries", "?")
    analysis["_outcome"] = "correct"
    return analysis


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — PER-TYPE SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

SYNTHESIS_SYSTEM = """\
You are an expert in causal inference and LLM agent evaluation.
You are given two sets of analyses for a specific question type from a causal
discovery benchmark:
  1. FAILURE analyses — cases where the agent got the wrong answer
  2. CORRECT analyses — cases where the agent got the right answer, assessed for
     whether that was through genuine reasoning or a lucky guess

Your job is to synthesize both into a comprehensive report.

Respond ONLY with valid JSON (no markdown, no prose outside JSON):
{
  "question_type": "<string>",
  "accuracy_from_eval": <float>,
  "failures_analyzed": <int>,
  "correct_cases_analyzed": <int>,
  "dominant_failure_category": "<most common failure category>",
  "failure_category_counts": {"<category>": <count>, ...},
  "typical_failure_onset_turn": <float — average turn where things go wrong>,
  "correct_case_integrity": {
    "genuine_count": <int>,
    "lucky_guess_count": <int>,
    "partial_count": <int>,
    "unclear_count": <int>,
    "effective_accuracy": "<e.g. '3/5 correct cases appear genuinely reasoned'>"
  },
  "pattern_summary": "<3-5 sentences describing the main failure pattern AND the quality of correct answers for this question type>",
  "key_reasoning_gaps": ["<specific reasoning mistakes the agent makes for this type>", ...],
  "strategy_issues": "<how the agent mis-allocates its query budget for this question type>",
  "root_cause_hypothesis": "<1 paragraph: why does the agent specifically struggle with this question type, and what does the correct-case analysis reveal about whether successes are robust?>",
  "actionable_recommendations": ["<concrete prompt or strategy change that could fix failures AND reduce lucky guesses>", ...]
}
"""

SYNTHESIS_USER_TEMPLATE = """\
Question type: {question_type}
Accuracy on this type: {accuracy:.1%} ({correct}/{total})

=== FAILURE ANALYSES ({n_failures} samples) ===
{failure_analyses_json}

=== CORRECT CASE ANALYSES ({n_correct} samples) ===
{correct_analyses_json}

Synthesize both sets into the JSON report schema.
"""


def synthesize_type(
    llm: BedrockLLM,
    question_type: str,
    failure_analyses: List[dict],
    correct_analyses: List[dict],
    type_stats: dict,
) -> dict:
    """Call LLM to synthesize failure + correct-case patterns for one question type."""
    def _clean(lst):
        return [{k: v for k, v in a.items() if not k.startswith("_")} for a in lst]

    accuracy = type_stats.get("accuracy", 0)
    correct = type_stats.get("correct", 0)
    total = type_stats.get("total", len(failure_analyses))

    user_prompt = SYNTHESIS_USER_TEMPLATE.format(
        question_type=question_type,
        accuracy=accuracy,
        correct=correct,
        total=total,
        n_failures=len(failure_analyses),
        n_correct=len(correct_analyses),
        failure_analyses_json=json.dumps(_clean(failure_analyses), indent=2),
        correct_analyses_json=json.dumps(_clean(correct_analyses), indent=2)
            if correct_analyses else "  (no correct cases sampled for this type)",
    )

    try:
        raw = llm.generate(SYNTHESIS_SYSTEM, user_prompt, max_new_tokens=2048)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        synthesis = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in synthesis for {question_type}: {e}")
        synthesis = {
            "question_type": question_type,
            "error": f"JSON parse error: {e}",
            "raw_response": raw[:500],
        }
    except Exception as e:
        log.error(f"Synthesis LLM call failed for {question_type}: {e}")
        synthesis = {"question_type": question_type, "error": str(e)}

    synthesis["_failure_analyses"] = failure_analyses
    synthesis["_correct_analyses"] = correct_analyses
    return synthesis


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_report(results: dict, output_dir: Path) -> str:
    """Render a human-readable markdown report from synthesis results."""
    lines = ["# Agent Failure Analysis Report\n"]

    meta = results.get("meta", {})
    lines.append(f"**Results dir:** `{meta.get('results_dir', '?')}`")
    lines.append(f"**Eval file:** `{meta.get('eval_file', '?')}`")
    lines.append(f"**Model analyzed:** `{meta.get('model', '?')}`")
    lines.append(f"**Overall accuracy:** {meta.get('overall_accuracy', '?')}")
    lines.append(f"**Failing samples per type:** {meta.get('samples_per_type', '?')}")
    lines.append(f"**Correct samples per type:** {meta.get('correct_samples_per_type', '?')}")
    lines.append("")

    # Overall failure category summary across all types
    all_cats: Dict[str, int] = {}
    for s in results.get("syntheses", {}).values():
        for cat, cnt in s.get("failure_category_counts", {}).items():
            all_cats[cat] = all_cats.get(cat, 0) + cnt

    if all_cats:
        lines.append("## Overall Failure Categories\n")
        for cat, cnt in sorted(all_cats.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {cnt}")
        lines.append("")

    # Per-type breakdown
    lines.append("## By Question Type\n")
    syntheses = results.get("syntheses", {})
    # Sort by accuracy ascending (most broken first)
    sorted_types = sorted(
        syntheses.items(),
        key=lambda kv: kv[1].get("accuracy_from_eval", 1.0),
    )

    for qtype, s in sorted_types:
        if "error" in s:
            lines.append(f"### {qtype} ⚠️ (analysis error)\n")
            lines.append(f"Error: {s['error']}\n")
            continue

        acc = s.get("accuracy_from_eval", "?")
        lines.append(f"### {qtype}  (accuracy: {acc:.1%})\n")
        lines.append(f"**Dominant failure:** {s.get('dominant_failure_category', '?')}")
        lines.append(f"**Typical failure onset:** turn {s.get('typical_failure_onset_turn', '?')}")

        ci = s.get("correct_case_integrity", {})
        if ci:
            eff = ci.get("effective_accuracy", "")
            genuine = ci.get("genuine_count", "?")
            lucky = ci.get("lucky_guess_count", "?")
            partial = ci.get("partial_count", "?")
            lines.append(
                f"**Correct-case integrity:** {eff}  "
                f"(genuine={genuine}, lucky={lucky}, partial={partial})"
            )
        lines.append("")

        pattern = s.get("pattern_summary", "")
        if pattern:
            lines.append(f"**Pattern:**\n{textwrap.fill(pattern, 100)}\n")

        gaps = s.get("key_reasoning_gaps", [])
        if gaps:
            lines.append("**Reasoning gaps:**")
            for g in gaps:
                lines.append(f"- {g}")
            lines.append("")

        strategy = s.get("strategy_issues", "")
        if strategy:
            lines.append(f"**Strategy issues:** {strategy}\n")

        root = s.get("root_cause_hypothesis", "")
        if root:
            lines.append(f"**Root cause:** {textwrap.fill(root, 100)}\n")

        recs = s.get("actionable_recommendations", [])
        if recs:
            lines.append("**Recommendations:**")
            for r in recs:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze failure patterns in agent experiment logs using Claude via Bedrock."
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to results folder (e.g. results/opus_coder_3_4)",
    )
    parser.add_argument(
        "--eval-file",
        help="Path to evaluation JSON (auto-detected if omitted)",
    )
    parser.add_argument(
        "--samples-per-type", type=int, default=5,
        help="Number of failing examples to analyze per question type (default: 5)",
    )
    parser.add_argument(
        "--correct-samples-per-type", type=int, default=3,
        help="Number of correct examples to check for genuine vs lucky reasoning (default: 3)",
    )
    parser.add_argument(
        "--output-dir", default="analysis_output",
        help="Directory to write results (default: analysis_output)",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_BEDROCK_MODEL,
        help=f"Bedrock model ID (default: {DEFAULT_BEDROCK_MODEL})",
    )
    parser.add_argument(
        "--synthesis-only", action="store_true",
        help="Skip per-log analysis; load cached analyses from output-dir and re-synthesize",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--question-types",
        help="Comma-separated list of question types to analyze (default: all)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    base_dir = Path(__file__).parent
    results_dir = base_dir / args.results_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Find eval file ─────────────────────────────────────────────────────
    if args.eval_file:
        eval_path = base_dir / args.eval_file
    else:
        # Auto-detect: look for eval_*.json matching the results dir name
        dir_name = results_dir.name  # e.g. opus_coder_3_4
        candidates = list((base_dir / "evaluations").glob(f"eval_{dir_name}*.json"))
        if not candidates:
            candidates = list((base_dir / "evaluations").glob("eval_*.json"))
        if not candidates:
            log.error("No eval file found. Specify --eval-file.")
            sys.exit(1)
        eval_path = candidates[0]
        log.info(f"Auto-detected eval file: {eval_path}")

    with open(eval_path) as f:
        eval_data = json.load(f)

    evaluated = eval_data.get("evaluated", [])
    type_stats = eval_data.get("scores", {}).get("by_question_type", {})
    model_name = eval_data.get("model", "unknown")
    overall_acc = eval_data.get("scores", {}).get("overall", {}).get("accuracy", "?")

    log.info(f"Loaded {len(evaluated)} evaluated questions. Overall accuracy: {overall_acc}")

    # ── 2. Group failures by question type ────────────────────────────────────
    filter_types = None
    if args.question_types:
        filter_types = set(args.question_types.split(","))

    failures_by_type: Dict[str, List[dict]] = {}
    corrects_by_type: Dict[str, List[dict]] = {}
    for item in evaluated:
        qtype = item.get("question_type", "unknown")
        if filter_types and qtype not in filter_types:
            continue
        is_correct = item.get("eval", {}).get("correct", item.get("is_correct", True))
        if not is_correct:
            failures_by_type.setdefault(qtype, []).append(item)
        else:
            corrects_by_type.setdefault(qtype, []).append(item)

    log.info(f"Failure counts by type: { {k: len(v) for k, v in failures_by_type.items()} }")
    log.info(f"Correct counts by type: { {k: len(v) for k, v in corrects_by_type.items()} }")

    # ── 3. Build log index ────────────────────────────────────────────────────
    agent_logs_dir = results_dir / "agent_logs"
    if not agent_logs_dir.exists():
        log.error(f"agent_logs directory not found: {agent_logs_dir}")
        sys.exit(1)

    log_index = build_log_index(agent_logs_dir)

    # ── 4. Sample failures and correct cases, load logs ───────────────────────
    def _sample_logs(by_type: Dict[str, List[dict]], n: int, label: str):
        result: Dict[str, List[Tuple[dict, Path]]] = {}
        for qtype, items in by_type.items():
            sample = random.sample(items, min(n, len(items)))
            loaded = []
            for item in sample:
                world_name = item.get("world_name", "")
                q_id = item.get("question_id")
                key = (world_name, int(q_id)) if q_id is not None else None
                lf = log_index.get(key) if key else None
                if lf is None:
                    log.warning(f"Log not found for ({world_name}, {q_id})")
                    continue
                try:
                    with open(lf) as f:
                        log_data = json.load(f)
                    loaded.append((log_data, lf))
                except Exception as e:
                    log.warning(f"Could not load {lf}: {e}")
            if loaded:
                result[qtype] = loaded
                log.info(f"  {qtype}: sampled {len(loaded)} {label} logs")
        return result

    sampled_failures = _sample_logs(failures_by_type, args.samples_per_type, "failing")
    sampled_corrects = _sample_logs(corrects_by_type, args.correct_samples_per_type, "correct")

    # ── 5. Phase 1: Per-log analysis ──────────────────────────────────────────
    llm = BedrockLLM(model_id=args.model_id, max_new_tokens=1024, temperature=0.2)

    failure_cache_file = output_dir / "per_log_analyses_failures.json"
    correct_cache_file = output_dir / "per_log_analyses_correct.json"
    per_type_failure_analyses: Dict[str, List[dict]] = {}
    per_type_correct_analyses: Dict[str, List[dict]] = {}

    if args.synthesis_only and failure_cache_file.exists() and correct_cache_file.exists():
        log.info("--synthesis-only: loading cached analyses")
        with open(failure_cache_file) as f:
            per_type_failure_analyses = json.load(f)
        with open(correct_cache_file) as f:
            per_type_correct_analyses = json.load(f)
    else:
        total_logs = (
            sum(len(v) for v in sampled_failures.values())
            + sum(len(v) for v in sampled_corrects.values())
        )
        done = 0

        for qtype, logs in sampled_failures.items():
            per_type_failure_analyses[qtype] = []
            for log_data, log_path in logs:
                done += 1
                log.info(f"[{done}/{total_logs}] Failure analysis: {log_path.name} ({qtype})")
                analysis = analyze_single_log(llm, log_data, log_path, qtype)
                per_type_failure_analyses[qtype].append(analysis)

        for qtype, logs in sampled_corrects.items():
            per_type_correct_analyses[qtype] = []
            for log_data, log_path in logs:
                done += 1
                log.info(f"[{done}/{total_logs}] Correct-case analysis: {log_path.name} ({qtype})")
                analysis = analyze_correct_log(llm, log_data, log_path, qtype)
                per_type_correct_analyses[qtype].append(analysis)

        with open(failure_cache_file, "w") as f:
            json.dump(per_type_failure_analyses, f, indent=2)
        with open(correct_cache_file, "w") as f:
            json.dump(per_type_correct_analyses, f, indent=2)
        log.info(f"Saved per-log analyses to {output_dir}")

    # ── 6. Phase 2: Per-type synthesis ────────────────────────────────────────
    llm_synth = BedrockLLM(model_id=args.model_id, max_new_tokens=2048, temperature=0.2)

    all_qtypes = set(per_type_failure_analyses) | set(per_type_correct_analyses)
    syntheses: Dict[str, dict] = {}
    for qtype in all_qtypes:
        fail_analyses = per_type_failure_analyses.get(qtype, [])
        corr_analyses = per_type_correct_analyses.get(qtype, [])
        if not fail_analyses and not corr_analyses:
            continue
        log.info(
            f"Synthesizing {qtype}: {len(fail_analyses)} failures, "
            f"{len(corr_analyses)} correct cases"
        )
        stats = type_stats.get(qtype, {"total": 0, "correct": 0, "accuracy": 0})
        synthesis = synthesize_type(llm_synth, qtype, fail_analyses, corr_analyses, stats)
        syntheses[qtype] = synthesis

    # ── 7. Assemble and save results ──────────────────────────────────────────
    results = {
        "meta": {
            "results_dir": str(results_dir),
            "eval_file": str(eval_path),
            "model": model_name,
            "overall_accuracy": overall_acc,
            "samples_per_type": args.samples_per_type,
            "correct_samples_per_type": args.correct_samples_per_type,
            "bedrock_model": args.model_id,
        },
        "syntheses": syntheses,
    }

    results_file = output_dir / "failure_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved full results to {results_file}")

    report_md = render_report(results, output_dir)
    report_file = output_dir / "failure_analysis.md"
    with open(report_file, "w") as f:
        f.write(report_md)
    log.info(f"Saved markdown report to {report_file}")

    # ── 8. Print summary to console ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results dir : {results_dir}")
    print(f"Model       : {model_name}")
    print(f"Overall acc : {overall_acc}")
    print()
    print(f"{'Question Type':<30} {'Accuracy':>10}  {'Genuine':>8}  {'Lucky':>6}  Dom. Failure")
    print("-" * 80)
    for qtype in sorted(syntheses.keys(), key=lambda t: syntheses[t].get("accuracy_from_eval", 1)):
        s = syntheses[qtype]
        acc = s.get("accuracy_from_eval", "?")
        dom = s.get("dominant_failure_category", "?")
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else str(acc)
        ci = s.get("correct_case_integrity", {})
        genuine = ci.get("genuine_count", "-")
        lucky = ci.get("lucky_guess_count", "-")
        print(f"{qtype:<30} {acc_str:>10}  {str(genuine):>8}  {str(lucky):>6}  {dom}")
    print()
    print(f"Full report : {report_file}")
    print(f"JSON output : {results_file}")


if __name__ == "__main__":
    main()
