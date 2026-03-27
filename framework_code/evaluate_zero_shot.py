"""
evaluate_zero_shot.py

Evaluate zero-shot baseline results against ground truth.

Usage:
    python evaluate_zero_shot.py results/zero_shot_20250101_120000.json

    # Show per-question details
    python evaluate_zero_shot.py results/zero_shot_20250101_120000.json --details

    # Use LLM to extract answers from verbose agent responses
    python evaluate_zero_shot.py results/agent_20250101_120000.json --llm-extract
    python evaluate_zero_shot.py results/agent_20250101_120000.json --llm-extract --extract-model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-based answer extraction for verbose agent responses
# ---------------------------------------------------------------------------

_YES_NO_QUESTION_TYPES = {
    # Original question types
    "direct_edge", "is_ancestor", "marginal_independence", "d_separation",
    "chain_marginal", "chain_conditional",
    "fork_marginal", "fork_conditional",
    "v_structure_marginal", "v_structure_conditional",
    # Causal question types
    "causal_effect",
}

_LIST_QUESTION_TYPES = {
    # Original question types
    "root_nodes", "leaf_nodes", "ancestors", "descendants", "markov_blanket",
    # Causal question types
    "all_causes_of", "all_effects_of",
}


def _is_yes_no_question(ground_truth: Any, question_type: str) -> bool:
    """Determine if a question expects a Yes/No answer."""
    if question_type in _YES_NO_QUESTION_TYPES:
        return True
    if isinstance(ground_truth, bool):
        return True
    if isinstance(ground_truth, str) and ground_truth.lower() in ("yes", "no"):
        return True
    return False


def _is_list_question(ground_truth: Any, question_type: str) -> bool:
    """Determine if a question expects a list-of-variables answer."""
    if question_type in _LIST_QUESTION_TYPES:
        return True
    if isinstance(ground_truth, (list, set)):
        return True
    return False


class AnswerExtractor:
    """Uses a small LLM to extract structured answers from verbose agent responses."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32

        logger.info(f"Loading extraction LLM: {model_name} on {self._device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(self._device)
        self.model.eval()
        self._torch = torch
        logger.info("Extraction LLM loaded.")

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=self._torch.ones_like(input_ids),
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def extract(
        self,
        question_text: str,
        verbose_answer: str,
        ground_truth: Any,
        question_type: str,
        all_var_names: Optional[List[str]] = None,
    ) -> str:
        """Extract a clean answer from a verbose agent response.

        Returns:
            For Yes/No questions: "Yes" or "No"
            For list questions: comma-separated variable names
        """
        if _is_yes_no_question(ground_truth, question_type):
            return self._extract_yes_no(question_text, verbose_answer)
        elif _is_list_question(ground_truth, question_type):
            return self._extract_list(question_text, verbose_answer, all_var_names)
        else:
            # Fallback: return as-is
            return verbose_answer

    def _extract_yes_no(self, question_text: str, verbose_answer: str) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked a Yes/No question "
            "and gave a verbose response. Your job is to determine what their final answer is. "
            "Reply with EXACTLY one word: Yes or No. Nothing else."
        )
        user = (
            f"Question: {question_text}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"What is the scientist's final answer? Reply with only Yes or No."
        )
        raw = self._generate(system, user)
        # Parse — take the first word that matches yes/no
        raw_lower = raw.strip().lower()
        if raw_lower.startswith("yes"):
            return "Yes"
        elif raw_lower.startswith("no"):
            return "No"
        # Fallback: search for yes/no anywhere
        if "yes" in raw_lower and "no" not in raw_lower:
            return "Yes"
        if "no" in raw_lower and "yes" not in raw_lower:
            return "No"
        # If still ambiguous, return the raw extraction for the existing parser
        return raw

    def _extract_list(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]] = None,
    ) -> str:
        var_list_str = ""
        if all_var_names:
            var_list_str = (
                f"\n\nThe variables in this network are:\n"
                + ", ".join(all_var_names)
            )
        system = (
            "You are a precise answer extractor. A scientist was asked to list specific "
            "variables from a causal system and gave a verbose response. Your job is to "
            "extract ONLY the variable names that are part of their final answer. "
            "Reply with ONLY a comma-separated list of variable names, nothing else. "
            "Use the exact variable names as they appear in the system. "
            "If the scientist's answer is empty or unclear, reply with: NONE"
        )
        user = (
            f"Question: {question_text}\n"
            f"{var_list_str}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Extract the list of variable names from the scientist's final answer. "
            f"Reply with only the comma-separated variable names."
        )
        return self._generate(system, user)


# ---------------------------------------------------------------------------
# Answer evaluation (mirrors orchestrator._evaluate_answer logic)
# ---------------------------------------------------------------------------

def normalize_var_name(name: str) -> str:
    """Normalize a variable name for comparison: lowercase, strip whitespace/quotes."""
    return name.strip().strip("'\"").lower()


def extract_var_names_from_answer(answer: str, all_var_names: List[str]) -> Set[str]:
    """
    Extract variable names mentioned in an answer string.
    Tries exact matching first, then falls back to substring matching.
    """
    answer_lower = answer.lower()
    found = set()

    # Build normalized lookup
    norm_to_original = {}
    for v in all_var_names:
        norm_to_original[v.lower()] = v

    # Strategy 1: split by commas / "and" and try to match each token
    tokens = re.split(r'[,;\n]+|\band\b', answer)
    for token in tokens:
        token_clean = normalize_var_name(token)
        if token_clean in norm_to_original:
            found.add(norm_to_original[token_clean])

    # Strategy 2: if nothing found via splitting, try substring matching
    if not found:
        for var_lower, var_orig in norm_to_original.items():
            if var_lower in answer_lower:
                found.add(var_orig)

    return found


def evaluate_answer(answer: str, ground_truth: Any,
                    all_var_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate a single answer against ground truth.

    Returns dict with:
        correct: bool
        detail: str explanation
        answer_parsed: the parsed answer
        metrics: dict with precision/recall/f1 for set answers
    """
    answer_norm = answer.strip().lower()

    # --- Yes/No (bool or "Yes"/"No" string) ---
    if isinstance(ground_truth, bool) or (isinstance(ground_truth, str) and
                                           ground_truth.lower() in ("yes", "no")):
        gt_bool = ground_truth if isinstance(ground_truth, bool) else (ground_truth.lower() == "yes")

        positive = ["yes", "true"]
        negative = ["no", "false"]

        is_pos = any(p in answer_norm for p in positive)
        is_neg = any(n in answer_norm for n in negative)

        if is_pos and not is_neg:
            correct = gt_bool is True
        elif is_neg and not is_pos:
            correct = gt_bool is False
        else:
            # Ambiguous or both present — check which appears first
            pos_idx = min((answer_norm.find(p) for p in positive if p in answer_norm), default=999)
            neg_idx = min((answer_norm.find(n) for n in negative if n in answer_norm), default=999)
            if pos_idx < neg_idx:
                correct = gt_bool is True
            elif neg_idx < pos_idx:
                correct = gt_bool is False
            else:
                correct = False

        return {
            "correct": correct,
            "detail": f"Expected {'Yes' if gt_bool else 'No'}, got: {answer[:80]}",
            "answer_parsed": answer.strip(),
            "metrics": {},
        }

    # --- List/set of variable names ---
    if isinstance(ground_truth, (list, set)):
        gt_set = set(ground_truth)
        var_names = all_var_names or list(gt_set)
        predicted = extract_var_names_from_answer(answer, var_names)

        # Case-insensitive comparison
        gt_lower = {v.lower() for v in gt_set}
        pred_lower = {v.lower() for v in predicted}

        tp = len(gt_lower & pred_lower)
        fp = len(pred_lower - gt_lower)
        fn = len(gt_lower - pred_lower)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match = (gt_lower == pred_lower)

        return {
            "correct": exact_match,
            "detail": f"Expected {sorted(gt_set)}, got {sorted(predicted)}",
            "answer_parsed": sorted(predicted),
            "metrics": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "exact_match": exact_match,
                "tp": tp, "fp": fp, "fn": fn,
            },
        }

    # --- String ground truth ---
    if isinstance(ground_truth, str):
        correct = ground_truth.lower() in answer_norm
        return {
            "correct": correct,
            "detail": f"Expected '{ground_truth}', got: {answer[:80]}",
            "answer_parsed": answer.strip(),
            "metrics": {},
        }

    # Fallback
    correct = str(ground_truth).lower() in answer_norm
    return {
        "correct": correct,
        "detail": f"Expected {ground_truth}, got: {answer[:80]}",
        "answer_parsed": answer.strip(),
        "metrics": {},
    }


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def compute_scores(evaluated: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate scores from evaluated results."""
    scores = {}

    # Overall
    total = len(evaluated)
    correct = sum(1 for e in evaluated if e["eval"]["correct"])
    scores["overall"] = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total > 0 else 0.0,
    }

    # Aggregate set-question metrics (precision/recall/f1)
    set_evals = [e for e in evaluated if e["eval"]["metrics"]]
    if set_evals:
        avg_p = sum(e["eval"]["metrics"]["precision"] for e in set_evals) / len(set_evals)
        avg_r = sum(e["eval"]["metrics"]["recall"] for e in set_evals) / len(set_evals)
        avg_f1 = sum(e["eval"]["metrics"]["f1"] for e in set_evals) / len(set_evals)
        scores["set_questions"] = {
            "count": len(set_evals),
            "avg_precision": round(avg_p, 3),
            "avg_recall": round(avg_r, 3),
            "avg_f1": round(avg_f1, 3),
        }

    # By question_type
    by_type = defaultdict(list)
    for e in evaluated:
        by_type[e["question_type"]].append(e)
    scores["by_question_type"] = {}
    for qtype, items in sorted(by_type.items()):
        n = len(items)
        c = sum(1 for i in items if i["eval"]["correct"])
        entry = {"total": n, "correct": c, "accuracy": round(c / n, 3) if n > 0 else 0.0}
        set_items = [i for i in items if i["eval"]["metrics"]]
        if set_items:
            entry["avg_f1"] = round(sum(i["eval"]["metrics"]["f1"] for i in set_items) / len(set_items), 3)
        scores["by_question_type"][qtype] = entry

    # By difficulty
    by_diff = defaultdict(list)
    for e in evaluated:
        by_diff[e.get("difficulty", "unknown")].append(e)
    scores["by_difficulty"] = {}
    for diff, items in sorted(by_diff.items()):
        n = len(items)
        c = sum(1 for i in items if i["eval"]["correct"])
        scores["by_difficulty"][diff] = {
            "total": n, "correct": c, "accuracy": round(c / n, 3) if n > 0 else 0.0
        }

    # By n_nodes
    by_nodes = defaultdict(list)
    for e in evaluated:
        by_nodes[e.get("n_nodes", 0)].append(e)
    scores["by_n_nodes"] = {}
    for nn, items in sorted(by_nodes.items()):
        n = len(items)
        c = sum(1 for i in items if i["eval"]["correct"])
        scores["by_n_nodes"][str(nn)] = {
            "total": n, "correct": c, "accuracy": round(c / n, 3) if n > 0 else 0.0
        }

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot baseline results",
    )
    parser.add_argument("results_json", nargs="+", help="Path(s) to zero-shot results JSON (accepts multiple files)")
    parser.add_argument("--details", action="store_true",
                        help="Print per-question details")
    parser.add_argument("-o", "--output", default=None,
                        help="Save evaluation to JSON file")
    parser.add_argument("--llm-extract", action="store_true",
                        help="Use a small LLM to extract clean answers from verbose agent responses")
    parser.add_argument("--extract-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use for answer extraction (default: Qwen/Qwen2.5-7B-Instruct)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    all_data = []
    for path in args.results_json:
        with open(path) as f:
            all_data.append(json.load(f))

    # Support both old format (top-level "model") and new format ("run_metadata.model")
    def _get_model(d):
        return d.get("model") or d.get("run_metadata", {}).get("model", "?")

    results = []
    for d in all_data:
        results.extend(d["results"])
    model_str = _get_model(all_data[0])
    print(f"Model: {model_str}")
    if len(args.results_json) > 1:
        print(f"Files: {', '.join(args.results_json)}")
    print(f"Evaluating {len(results)} answers...\n")

    # Optionally load LLM for answer extraction
    extractor = None
    if args.llm_extract:
        print(f"Loading extraction LLM ({args.extract_model})...")
        extractor = AnswerExtractor(model_name=args.extract_model)
        print("Extraction LLM ready.\n")

    # For each result, get the variable names from the world file for set matching
    world_var_cache = {}

    evaluated = []
    n_extracted = 0
    for ri, r in enumerate(results):
        wf = r["world_file"]
        if wf not in world_var_cache:
            try:
                with open(wf) as f:
                    world_data = json.load(f)
                world_var_cache[wf] = [v["name"] for v in world_data["variables"]]
            except Exception:
                world_var_cache[wf] = None

        all_vars = world_var_cache.get(wf)
        answer = r["extracted_answer"]

        # LLM extraction step for verbose agent responses
        if extractor is not None:
            question_type = r.get("question_type", "")
            extracted = extractor.extract(
                question_text=r["question_text"],
                verbose_answer=answer,
                ground_truth=r["ground_truth"],
                question_type=question_type,
                all_var_names=all_vars,
            )
            if extracted != answer:
                n_extracted += 1
            answer = extracted
            if (ri + 1) % 10 == 0:
                print(f"  Extracted {ri + 1}/{len(results)}...")

        ev = evaluate_answer(answer, r["ground_truth"],
                             all_var_names=all_vars)
        entry = {
            "world_name": r["world_name"],
            "question_id": r["question_id"],
            "question_type": r.get("question_type", ""),
            "difficulty": r.get("difficulty", ""),
            "n_nodes": r.get("n_nodes", 0),
            "question_text": r["question_text"],
            "ground_truth": r["ground_truth"],
            "raw_response": r.get("raw_response", ""),
            "original_answer": r["extracted_answer"],
            "extracted_answer": answer,
            "eval": ev,
        }
        evaluated.append(entry)

    if extractor is not None:
        print(f"\nLLM extraction: {n_extracted}/{len(results)} answers were re-extracted.\n")

    # Compute scores
    scores = compute_scores(evaluated)

    # Print summary
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    ov = scores["overall"]
    print(f"  Accuracy: {ov['correct']}/{ov['total']} = {ov['accuracy']:.1%}")

    if "set_questions" in scores:
        sq = scores["set_questions"]
        print(f"\n  Set questions ({sq['count']}):")
        print(f"    Avg Precision: {sq['avg_precision']:.3f}")
        print(f"    Avg Recall:    {sq['avg_recall']:.3f}")
        print(f"    Avg F1:        {sq['avg_f1']:.3f}")

    print(f"\nBY QUESTION TYPE:")
    for qtype, s in scores["by_question_type"].items():
        line = f"  {qtype:30s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}"
        if "avg_f1" in s:
            line += f"  (avg F1: {s['avg_f1']:.3f})"
        print(line)

    print(f"\nBY DIFFICULTY:")
    for diff, s in scores["by_difficulty"].items():
        print(f"  {diff:15s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    print(f"\nBY NETWORK SIZE (n_nodes):")
    for nn, s in scores["by_n_nodes"].items():
        print(f"  n={nn:3s}  {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    # Per-question details
    if args.details:
        print(f"\n{'=' * 60}")
        print("PER-QUESTION DETAILS")
        print("=" * 60)
        for e in evaluated:
            status = "CORRECT" if e["eval"]["correct"] else "WRONG"
            print(f"\n[{status}] {e['world_name']} q{e['question_id']} ({e['question_type']}, {e['difficulty']})")
            print(f"  Q: {e['question_text'][:100]}")
            print(f"  {e['eval']['detail']}")
            if e["eval"]["metrics"]:
                m = e["eval"]["metrics"]
                print(f"  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

    # Save
    if args.output:
        out_path = args.output
        # If no directory specified, default to evaluations/ subfolder
        import os
        if os.path.dirname(out_path) == "":
            eval_dir = os.path.join(os.path.dirname(__file__), "evaluations")
            os.makedirs(eval_dir, exist_ok=True)
            out_path = os.path.join(eval_dir, out_path)

        source = args.results_json[0] if len(args.results_json) == 1 else args.results_json
        output = {
            "source": source,
            "model": model_str,
            "scores": scores,
            "evaluated": evaluated,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nEvaluation saved to {out_path}")


if __name__ == "__main__":
    main()
