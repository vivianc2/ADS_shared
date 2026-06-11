"""
evaluate_zero_shot.py

Evaluate zero-shot baseline results against ground truth.

Usage:
    python evaluate_zero_shot.py results/zero_shot_20250101_120000.json

    # Show per-question details
    python evaluate_zero_shot.py results/zero_shot_20250101_120000.json --details

    # LLM extraction is always on (Bedrock Opus 4.8 by default).
    # Override the extractor model with --extract-model.
    python evaluate_zero_shot.py results/agent_20250101_120000.json --extract-model Qwen/Qwen2.5-7B-Instruct
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


def _looks_like_bedrock_model(model_name: str) -> bool:
    """Bedrock model IDs follow 'us.<provider>.<model>-...' / contain a colon
    region tag, while HuggingFace IDs contain a '/'."""
    return "/" not in model_name


class AnswerExtractor:
    """Uses an LLM to extract structured answers from verbose agent responses.

    Backend is auto-detected from the model name: HuggingFace IDs (containing
    '/') load locally via transformers; anything else is treated as a Bedrock
    model id and routed through BedrockLLM.
    """

    def __init__(self, model_name: str = "us.anthropic.claude-opus-4-7"):
        self.model_name = model_name
        self._backend = "bedrock" if _looks_like_bedrock_model(model_name) else "hf"

        if self._backend == "bedrock":
            from bedrock_llm import BedrockLLM
            logger.info(f"Loading extraction LLM (Bedrock): {model_name}...")
            self._bedrock = BedrockLLM(
                model_id=model_name,
                temperature=0.0,
                max_new_tokens=256,
            )
            logger.info("Extraction LLM loaded.")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32

        logger.info(f"Loading extraction LLM (HF): {model_name} on {self._device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(self._device)
        self.model.eval()
        self._torch = torch
        logger.info("Extraction LLM loaded.")

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._backend == "bedrock":
            return self._bedrock.generate(system_prompt, user_prompt).strip()

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
            For advanced_*: a string shaped so downstream regex evaluators
                (_parse_var_value_tuple, _parse_rank_list, etc.) can parse.
        """
        # Advanced-benchmark dispatch: each type has its own target format
        # because downstream evaluators parse different structures.
        if question_type.startswith("advanced_"):
            return self._extract_advanced(
                question_type=question_type,
                question_text=question_text,
                verbose_answer=verbose_answer,
                all_var_names=all_var_names,
            )
        if _is_yes_no_question(ground_truth, question_type):
            return self._extract_yes_no(question_text, verbose_answer)
        elif _is_list_question(ground_truth, question_type):
            return self._extract_list(question_text, verbose_answer, all_var_names)
        else:
            # Fallback: return as-is
            return verbose_answer

    # ---------- advanced-benchmark extraction ----------

    def _extract_advanced(
        self,
        question_type: str,
        question_text: str,
        verbose_answer: str,
        all_var_names: Optional[List[str]] = None,
    ) -> str:
        handler = {
            "advanced_budget_argmin":   self._extract_var_equals_value,
            "advanced_budget_satisfy":  self._extract_var_equals_value_or_none,
            "advanced_side_effect":     self._extract_var_equals_value_or_none,
            "advanced_adjustment_set":  self._extract_adjustment_set,
            "advanced_mediator_class":  self._extract_mediator_label,
            "advanced_rank_topK":       self._extract_ranked_list,
        }.get(question_type)
        if handler is None:
            return verbose_answer
        return handler(question_text, verbose_answer, all_var_names)

    @staticmethod
    def _var_list_clause(all_var_names: Optional[List[str]]) -> str:
        if not all_var_names:
            return ""
        return (
            "\n\nThe intervenable variables in this network are:\n"
            + ", ".join(all_var_names)
        )

    def _extract_var_equals_value(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]],
    ) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked to pick "
            "ONE intervention of the form 'Variable = value' and gave a verbose "
            "response. Extract ONLY the final intervention they chose. "
            "Reply with EXACTLY one line in the format: Variable = value"
        )
        user = (
            f"Question: {question_text}{self._var_list_clause(all_var_names)}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Extract the single 'Variable = value' the scientist recommended. "
            f"Use exact variable and value names. Reply with one line only."
        )
        raw = self._generate(system, user).strip()
        if not raw:
            return verbose_answer
        return raw.splitlines()[0].strip()

    def _extract_var_equals_value_or_none(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]],
    ) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked whether "
            "some intervention 'Variable = value' satisfies a constraint, or "
            "whether no feasible intervention exists. Reply with EITHER a single "
            "line 'Variable = value' OR the single word 'none' (lowercase) if "
            "the scientist concluded no feasible intervention."
        )
        user = (
            f"Question: {question_text}{self._var_list_clause(all_var_names)}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Reply with 'Variable = value' OR 'none'. One line only."
        )
        raw = self._generate(system, user).strip()
        if not raw:
            return verbose_answer  # let downstream regex try the raw response
        first = raw.splitlines()[0].strip()
        if first.lower().startswith("none"):
            return "none"
        return first

    def _extract_adjustment_set(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]],
    ) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked for a "
            "minimal backdoor adjustment set (the variables to CONDITION ON / "
            "ADJUST FOR) for identifying a causal effect. "
            "Reply with ONE of three things, nothing else:\n"
            "  (a) 'none'           -- if no conditioning is needed (empty set)\n"
            "  (b) 'unidentifiable' -- if the effect cannot be identified by any "
            "backdoor adjustment\n"
            "  (c) a comma-separated list of variable names (e.g. 'Age, Income')\n"
            "Use exact variable names. Do NOT include the treatment or outcome."
        )
        user = (
            f"Question: {question_text}{self._var_list_clause(all_var_names)}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Extract the adjustment set. Reply with 'none', 'unidentifiable', "
            f"or a comma-separated variable list."
        )
        raw = self._generate(system, user).strip()
        if not raw:
            return verbose_answer
        lowered = raw.lower()
        if "unidentif" in lowered:
            return "unidentifiable"
        if lowered.startswith("none") or lowered == "empty" or lowered.startswith("no adjust"):
            return "none"
        return raw.splitlines()[0].strip()

    def _extract_mediator_label(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]],
    ) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked whether "
            "a variable M is a mediator between T and O. Reply with EXACTLY one "
            "of these three labels (lowercase, underscores):\n"
            "  only_through_M\n"
            "  also_direct_or_other\n"
            "  not_mediator"
        )
        user = (
            f"Question: {question_text}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Reply with one of: only_through_M, also_direct_or_other, not_mediator."
        )
        raw = self._generate(system, user).strip().lower().replace(" ", "_")
        for label in ("only_through_m", "also_direct_or_other", "not_mediator"):
            if label in raw:
                # canonical form uses capital M for only_through_M
                return "only_through_M" if label == "only_through_m" else label
        return raw  # let downstream eval fall back to its keyword heuristic

    def _extract_ranked_list(
        self, question_text: str, verbose_answer: str,
        all_var_names: Optional[List[str]],
    ) -> str:
        system = (
            "You are a precise answer extractor. A scientist was asked to rank "
            "the top-K interventions (each of the form 'Variable = value'). "
            "Extract the ranking in order. Reply with EXACTLY this format on a "
            "single line: '1. Var1 = val1; 2. Var2 = val2; 3. Var3 = val3' "
            "(semicolon-separated, most-effective first). Use exact names."
        )
        user = (
            f"Question: {question_text}{self._var_list_clause(all_var_names)}\n\n"
            f"Scientist's response:\n{verbose_answer}\n\n"
            f"Reply with the ranked list in the format specified above."
        )
        raw = self._generate(system, user).strip()
        if not raw:
            return verbose_answer
        return raw.splitlines()[0].strip()

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


# ---------------------------------------------------------------------------
# Advanced-benchmark evaluators
# ---------------------------------------------------------------------------

_ADV_NONE_TOKENS = ("none", "no intervention", "no such intervention", "not feasible",
                    "infeasible", "no feasible")
_ADV_UNIDENT_TOKENS = ("unidentifiable", "not identifiable", "cannot be identified")


def _parse_var_value_tuple(answer: str, all_vars: Optional[List[str]]) -> Optional[tuple]:
    """Extract the first 'X = v' assignment from the answer string.

    Matches `var = value`, `var: value`, `var is value`.  `var` must be a
    known variable name (case-insensitive).  The value may be multi-word
    (e.g. "Stool Test", "Screened 2-5 Years Ago") — we capture everything
    up to end-of-line, semicolon, period, or a conjunction word.  Returns
    (var_canonical, value_canonical) or None.
    """
    if not all_vars:
        return None
    var_by_lower = {v.lower(): v for v in all_vars}
    # longest-first for correct matching of substrings like Foo vs FooBar
    var_pat = "|".join(re.escape(v) for v in sorted(all_vars, key=len, reverse=True))
    # Value captures non-greedily up to a terminator (punctuation,
    # newline, conjunction, or end-of-string).  Lets multi-word state
    # names survive ("Stool Test", "Not Applicable", etc.).
    pattern = re.compile(
        rf"\b({var_pat})\b\s*(?:=|:| is | ==> | -> | to be )\s*"
        rf"['\"]?([^\n;'\"]+?)['\"]?\s*"
        rf"(?=$|[\n;.]|,\s|\s+(?:and|or|but|while|then|because)\b)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(answer):
        var = var_by_lower.get(m.group(1).lower())
        val = m.group(2).strip().rstrip(".,;:")
        if var is not None and val:
            return (var, val)
    return None


def _match_value_case_insensitive(value: str, candidate_values: List[str]) -> Optional[str]:
    vl = value.strip().strip("'\"").lower()
    for cand in candidate_values:
        if cand.lower() == vl:
            return cand
    return None


def _mentions_none(answer: str) -> bool:
    a = answer.lower()
    return any(t in a for t in _ADV_NONE_TOKENS)


def _mentions_unidentifiable(answer: str) -> bool:
    a = answer.lower()
    return any(t in a for t in _ADV_UNIDENT_TOKENS)


_DEFAULT_TIE_TOLERANCE = 0.15  # expected-state-index units


def _effect_table_lookup(effect_table: Dict[str, float], var: str, val: str) -> Optional[float]:
    """Case-insensitive lookup in a flattened 'Var=value' effect-table dict."""
    if not effect_table:
        return None
    key_lower = f"{var}={val}".lower()
    for k, v in effect_table.items():
        if k.lower() == key_lower:
            return float(v)
    return None


def _eval_budget_argmin(answer, ground_truth, all_vars,
                        question_metadata: Optional[Dict[str, Any]] = None,
                        tie_tolerance: float = _DEFAULT_TIE_TOLERANCE):
    gt_var = ground_truth.get("variable") if isinstance(ground_truth, dict) else None
    gt_val = ground_truth.get("value") if isinstance(ground_truth, dict) else None
    parsed = _parse_var_value_tuple(answer, all_vars)
    if parsed is None:
        return {"correct": False,
                "detail": f"Expected {gt_var}={gt_val}; no X=v tuple parsed from answer",
                "answer_parsed": None, "metrics": {}}
    pvar, pval = parsed
    exact = (pvar == gt_var and pval.lower() == str(gt_val).lower())

    tie_accepted = False
    pred_expected = None
    best_expected = None
    if not exact and question_metadata:
        effect_table = question_metadata.get("effect_table") or {}
        best_expected = question_metadata.get("best_expected")
        pred_expected = _effect_table_lookup(effect_table, pvar, pval)
        if (pred_expected is not None and best_expected is not None
                and abs(pred_expected - float(best_expected)) <= tie_tolerance):
            tie_accepted = True

    correct = exact or tie_accepted
    detail = f"Expected {gt_var}={gt_val}, parsed {pvar}={pval}"
    if tie_accepted:
        detail += (f" — within ε={tie_tolerance} of optimum "
                   f"(pred E={pred_expected:.4f}, best E={best_expected:.4f})")
    return {
        "correct": correct,
        "detail": detail,
        "answer_parsed": {"variable": pvar, "value": pval},
        "metrics": {
            "exact_match": exact,
            "tie_accepted": tie_accepted,
            "tie_tolerance": tie_tolerance,
            "pred_expected": pred_expected,
            "best_expected": best_expected,
        },
    }


def _eval_budget_satisfy(answer, ground_truth, all_vars):
    # ground_truth is list of {variable, value, ...}; may be empty.
    gt_list = ground_truth if isinstance(ground_truth, list) else []
    feasible = {(item["variable"], str(item["value"]).lower()) for item in gt_list}
    if not feasible:
        correct = _mentions_none(answer)
        return {
            "correct": correct,
            "detail": ("Expected 'none' (no feasible intervention) — "
                       f"{'answer mentions none' if correct else 'no match'}"),
            "answer_parsed": "none" if correct else answer.strip()[:80],
            "metrics": {},
        }
    parsed = _parse_var_value_tuple(answer, all_vars)
    if parsed is None:
        if _mentions_none(answer):
            return {"correct": False,
                    "detail": "Answered 'none' but feasible interventions exist",
                    "answer_parsed": "none", "metrics": {}}
        return {"correct": False, "detail": "no X=v tuple parsed",
                "answer_parsed": None, "metrics": {}}
    pvar, pval = parsed
    correct = (pvar, pval.lower()) in feasible
    return {
        "correct": correct,
        "detail": f"Parsed {pvar}={pval}; feasible set size = {len(feasible)}",
        "answer_parsed": {"variable": pvar, "value": pval},
        "metrics": {"feasible_set_size": len(feasible)},
    }


def _eval_side_effect(answer, ground_truth, all_vars):
    # Same structure as satisfy
    return _eval_budget_satisfy(answer, ground_truth, all_vars)


def _eval_adjustment_set(answer, ground_truth, all_vars):
    # ground_truth: "none" | "unidentifiable" | list-of-lists
    if ground_truth == "unidentifiable":
        correct = _mentions_unidentifiable(answer)
        return {"correct": correct,
                "detail": f"Expected 'unidentifiable' ({'match' if correct else 'no match'})",
                "answer_parsed": "unidentifiable" if correct else answer[:80],
                "metrics": {}}
    if ground_truth == "none":
        correct = _mentions_none(answer) and not _mentions_unidentifiable(answer)
        return {"correct": correct,
                "detail": f"Expected 'none' (empty adjustment suffices)",
                "answer_parsed": "none" if correct else answer[:80],
                "metrics": {}}
    # list-of-lists
    if not isinstance(ground_truth, list):
        return {"correct": False, "detail": f"unexpected gt {ground_truth!r}",
                "answer_parsed": answer[:80], "metrics": {}}
    gt_sets = [frozenset(s) for s in ground_truth]
    predicted = extract_var_names_from_answer(answer, all_vars or [])
    pred_set = frozenset(predicted)
    exact_match = any(pred_set == gs for gs in gt_sets)
    # Precision/recall against the closest gt set
    best = max(gt_sets, key=lambda g: len(g & pred_set) - 0.01 * len(g ^ pred_set)) if gt_sets else frozenset()
    tp = len(pred_set & best); fp = len(pred_set - best); fn = len(best - pred_set)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "correct": exact_match,
        "detail": f"Expected one of {[sorted(s) for s in gt_sets]}, got {sorted(pred_set)}",
        "answer_parsed": sorted(pred_set),
        "metrics": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                    "exact_match": exact_match},
    }


_MEDIATOR_LABELS = ("only_through_m", "also_direct_or_other", "not_mediator")


def _eval_mediator_class(answer, ground_truth, all_vars):
    a = answer.lower().replace(" ", "_")
    gt = str(ground_truth).lower()
    # Find the first label mentioned in the answer
    first_pos = {lbl: a.find(lbl) for lbl in _MEDIATOR_LABELS}
    present = {lbl: pos for lbl, pos in first_pos.items() if pos >= 0}
    if not present:
        # Fallback: keyword heuristic
        if "only" in a and "through" in a:
            pred = "only_through_m"
        elif "not" in a and ("mediat" in a or "mediator" in a):
            pred = "not_mediator"
        elif "direct" in a or "alternative" in a or "also" in a:
            pred = "also_direct_or_other"
        else:
            pred = None
    else:
        pred = min(present.items(), key=lambda kv: kv[1])[0]
    correct = (pred is not None and pred == gt.lower())
    return {"correct": correct,
            "detail": f"Expected {ground_truth}, parsed {pred}",
            "answer_parsed": pred or answer[:80],
            "metrics": {}}


def _parse_rank_list(answer: str, all_vars: Optional[List[str]]) -> List[tuple]:
    """Extract ordered [(var, value), ...] from a ranked-list answer.

    Accepts formats like '1. Var = val; 2. Var2 = val2' and handles
    multi-word values ("Stool Test") by capturing up to the next semicolon
    / item-boundary / end-of-string.
    """
    if not all_vars:
        return []
    var_by_lower = {v.lower(): v for v in all_vars}
    var_pat = "|".join(re.escape(v) for v in sorted(all_vars, key=len, reverse=True))
    # Leading number + punct, then Var = val.  Value terminates at ';',
    # newline, or the next ranking number (' 2.', ' 3)', etc.).
    pattern = re.compile(
        rf"(?:^|[\n;,])\s*\(?\d+[\.\)]?\s*({var_pat})\s*(?:=|:| is )\s*"
        rf"['\"]?([^\n;'\"]+?)['\"]?\s*"
        rf"(?=$|[\n;]|,?\s*\d+[\.\)])",
        re.IGNORECASE | re.MULTILINE,
    )
    out: List[tuple] = []
    for m in pattern.finditer(answer):
        var = var_by_lower.get(m.group(1).lower())
        val = m.group(2).strip().rstrip(".,;:")
        if var is not None and val:
            out.append((var, val))
    if not out:
        # Fallback: collect every X = val in textual order.
        pat2 = re.compile(
            rf"\b({var_pat})\b\s*(?:=|:)\s*"
            rf"['\"]?([^\n;'\"]+?)['\"]?\s*"
            rf"(?=$|[\n;.]|,\s|\s+(?:and|or|but|while|then)\b)",
            re.IGNORECASE,
        )
        for m in pat2.finditer(answer):
            var = var_by_lower.get(m.group(1).lower())
            val = m.group(2).strip().rstrip(".,;:")
            if var is not None and val:
                out.append((var, val))
    return out


def _eval_rank_topk(answer, ground_truth, all_vars,
                    question_metadata: Optional[Dict[str, Any]] = None,
                    tie_tolerance: float = _DEFAULT_TIE_TOLERANCE):
    gt_list = ground_truth if isinstance(ground_truth, list) else []
    gt_order = [(item["variable"], str(item["value"]).lower()) for item in gt_list]
    gt_expected = [item.get("expected_target") for item in gt_list]
    parsed = _parse_rank_list(answer, all_vars)
    pred_order = [(v, val.lower()) for (v, val) in parsed][: len(gt_order)]

    exact = (pred_order == gt_order)

    effect_table = (question_metadata or {}).get("effect_table") or {}

    def _pos_match(i: int) -> bool:
        if i >= len(pred_order):
            return False
        if pred_order[i] == gt_order[i]:
            return True
        if not effect_table or gt_expected[i] is None:
            return False
        pv, pval = pred_order[i]
        pe = _effect_table_lookup(effect_table, pv, pval)
        return pe is not None and abs(pe - float(gt_expected[i])) <= tie_tolerance

    # Strict: positions matched exactly
    strict_pos_correct = sum(
        1 for i, gv in enumerate(gt_order)
        if i < len(pred_order) and pred_order[i] == gv
    )
    strict_pos_acc = strict_pos_correct / max(1, len(gt_order))

    # Tie-tolerant: position counts as matched if expected_target is within ε
    tol_pos_correct = sum(1 for i in range(len(gt_order)) if _pos_match(i))
    tol_pos_acc = tol_pos_correct / max(1, len(gt_order))
    tol_exact = (len(pred_order) == len(gt_order)
                 and all(_pos_match(i) for i in range(len(gt_order))))

    set_overlap = len(set(pred_order) & set(gt_order)) / max(1, len(gt_order))

    correct = exact or tol_exact
    detail = f"Expected {gt_order}, parsed {pred_order}"
    if tol_exact and not exact:
        detail += f" — order accepted within ε={tie_tolerance}"
    return {
        "correct": correct,
        "detail": detail,
        "answer_parsed": pred_order,
        "metrics": {
            "exact_order": exact,
            "tol_exact_order": tol_exact,
            "tie_tolerance": tie_tolerance,
            "position_accuracy": round(strict_pos_acc, 3),
            "tol_position_accuracy": round(tol_pos_acc, 3),
            "set_overlap": round(set_overlap, 3),
        },
    }


_ADVANCED_EVALUATORS = {
    "advanced_budget_argmin":   _eval_budget_argmin,
    "advanced_budget_satisfy":  _eval_budget_satisfy,
    "advanced_side_effect":     _eval_side_effect,
    "advanced_adjustment_set":  _eval_adjustment_set,
    "advanced_mediator_class":  _eval_mediator_class,
    "advanced_rank_topK":       _eval_rank_topk,
}


_TIE_TOLERANT_EVALUATORS = {"advanced_budget_argmin", "advanced_rank_topK"}


def _evaluate_advanced(answer, ground_truth, question_type, all_var_names,
                       question_metadata: Optional[Dict[str, Any]] = None,
                       tie_tolerance: float = _DEFAULT_TIE_TOLERANCE):
    fn = _ADVANCED_EVALUATORS.get(question_type)
    if fn is None:
        return {"correct": False,
                "detail": f"no evaluator registered for {question_type}",
                "answer_parsed": answer[:80], "metrics": {}}
    if question_type in _TIE_TOLERANT_EVALUATORS:
        return fn(answer, ground_truth, all_var_names,
                  question_metadata=question_metadata,
                  tie_tolerance=tie_tolerance)
    return fn(answer, ground_truth, all_var_names)


def evaluate_answer(answer: str, ground_truth: Any,
                    all_var_names: List[str] = None,
                    question_type: str = "",
                    question_metadata: Optional[Dict[str, Any]] = None,
                    tie_tolerance: float = _DEFAULT_TIE_TOLERANCE) -> Dict[str, Any]:
    """
    Evaluate a single answer against ground truth.

    Returns dict with:
        correct: bool
        detail: str explanation
        answer_parsed: the parsed answer
        metrics: dict with precision/recall/f1 for set answers
    """
    # Advanced benchmark: dispatch on question_type before generic branches,
    # since their ground truths are structured (dict / list-of-dicts / labels).
    if question_type and question_type.startswith("advanced_"):
        return _evaluate_advanced(answer, ground_truth, question_type,
                                  all_var_names,
                                  question_metadata=question_metadata,
                                  tie_tolerance=tie_tolerance)

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
        exact_match = (gt_lower == pred_lower)

        # When both sets are empty, it's a perfect match (correctly predicted nothing).
        if not gt_lower and not pred_lower:
            precision = recall = f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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

# Map each question type to a higher-level category (the 3 question groups
# described in CLAUDE.md). Question types not listed fall into "other".
_QUESTION_TYPE_TO_CATEGORY = {
    # Group 1 — Causal Structure
    "causal_effect": "causal_structure",
    "all_causes_of": "causal_structure",
    "all_effects_of": "causal_structure",
    # Group 2 — Marginal Independence
    "direct_marginal": "marginal_independence",
    "chain_marginal": "marginal_independence",
    "fork_marginal": "marginal_independence",
    "v_structure_marginal": "marginal_independence",
    "other_marginal": "marginal_independence",
    # Group 3 — Conditional Independence
    "chain_conditional": "conditional_independence",
    "fork_conditional": "conditional_independence",
    "v_structure_conditional": "conditional_independence",
    "other_conditional": "conditional_independence",
    # Advanced benchmark — sequential reasoning / decision making
    "advanced_budget_argmin":   "causal_decision",
    "advanced_budget_satisfy":  "causal_decision",
    "advanced_side_effect":     "causal_decision",
    "advanced_adjustment_set":  "causal_decision",
    "advanced_mediator_class":  "causal_decision",
    "advanced_rank_topK":       "causal_decision",
}


def _structural_truth(entry: Dict) -> Optional[str]:
    """For marginal/conditional-independence questions, derive the underlying
    structural truth ('dep' or 'indep') from question-text framing + ground truth.
    Framing is "asks-independence" if the question text contains the word
    "independent"; otherwise it is "asks-dependence". Answer "Yes" under
    asks-independence means structurally independent; under asks-dependence it
    means structurally dependent (and vice versa for "No"). Returns None for
    question types outside the two independence groups.
    """
    qtype = entry.get("question_type", "") or ""
    if not (qtype.endswith("_marginal") or qtype.endswith("_conditional")):
        return None
    qtext = (entry.get("question_text") or "").lower()
    gt = entry.get("ground_truth", "")
    if isinstance(gt, bool):
        gt_yes = gt
    elif isinstance(gt, str):
        gt_yes = gt.strip().lower() == "yes"
    else:
        return None
    asks_indep = "independent" in qtext
    structurally_indep = (asks_indep == gt_yes)
    return "indep" if structurally_indep else "dep"


def _has_prf(metrics: Dict[str, Any]) -> bool:
    """True iff metrics carries precision/recall/f1 keys.

    Advanced evaluators return metric dicts with varied schemas (e.g.
    rank_topK → exact_order/position_accuracy/set_overlap, budget_satisfy →
    feasible_set_size).  Callers that want to average precision/recall/f1
    must filter on this predicate, not just on metrics being non-empty.
    """
    return bool(metrics) and all(k in metrics for k in ("precision", "recall", "f1"))


def _summarize(items: List[Dict]) -> Dict[str, Any]:
    """Build a {total, correct, accuracy, avg_f1} summary for a list of evaluated items."""
    n = len(items)
    c = sum(1 for i in items if i["eval"]["correct"])
    entry = {
        "total": n,
        "correct": c,
        "accuracy": round(c / n, 3) if n > 0 else 0.0,
    }
    set_items = [i for i in items if _has_prf(i["eval"]["metrics"])]
    if set_items:
        entry["avg_f1"] = round(
            sum(i["eval"]["metrics"]["f1"] for i in set_items) / len(set_items), 3
        )
    return entry


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

    # Aggregate set-question metrics (precision/recall/f1).  Only items whose
    # metrics dict has all three keys qualify — advanced evaluators use
    # other schemas (rank ordering, feasible-set size) and must not be
    # averaged as if they were set-valued.
    set_evals = [e for e in evaluated if _has_prf(e["eval"]["metrics"])]
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

    # Excluding set questions (i.e. only Yes/No / non-set-valued questions —
    # set_questions block above already covers the set-valued ones).  Note
    # non_set_evals here STILL includes advanced questions whose metrics
    # don't carry precision/recall/f1; they're summarized by accuracy only.
    non_set_evals = [e for e in evaluated if not _has_prf(e["eval"]["metrics"])]
    scores["excluding_set_questions"] = _summarize(non_set_evals)

    # Advanced-benchmark extra metrics: exact_order / position_accuracy /
    # set_overlap for rank_topK, plus feasible_set_size / exact_match where
    # present.  Kept in a dedicated block so they don't collide with the
    # set-question precision/recall/f1 averages above.
    adv_metric_bucket: Dict[str, List[float]] = defaultdict(list)
    for e in evaluated:
        m = e["eval"]["metrics"]
        if not m:
            continue
        for key in ("exact_order", "position_accuracy", "set_overlap",
                    "feasible_set_size", "exact_match"):
            if key in m:
                adv_metric_bucket[key].append(float(m[key]))
    if adv_metric_bucket:
        scores["advanced_metrics"] = {
            key: {"count": len(vals), "avg": round(sum(vals) / len(vals), 3)}
            for key, vals in adv_metric_bucket.items()
        }

    # By category (the 3 question groups)
    by_category = defaultdict(list)
    for e in evaluated:
        cat = _QUESTION_TYPE_TO_CATEGORY.get(e["question_type"], "other")
        by_category[cat].append(e)
    scores["by_category"] = {
        cat: _summarize(items) for cat, items in sorted(by_category.items())
    }

    # By question_type
    by_type = defaultdict(list)
    for e in evaluated:
        by_type[e["question_type"]].append(e)
    scores["by_question_type"] = {
        qtype: _summarize(items) for qtype, items in sorted(by_type.items())
    }

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

    # By topology (world-level structural motif, e.g. chain_4, v_struct_3, ...)
    by_topology = defaultdict(list)
    for e in evaluated:
        by_topology[e.get("topology") or "unknown"].append(e)
    scores["by_topology"] = {
        topo: _summarize(items) for topo, items in sorted(by_topology.items())
    }

    # By structural truth (dep vs indep) — separately for marginal and conditional.
    # Uses question-text framing, so accuracy here is about getting the
    # underlying structural relationship right, not merely the Yes/No label.
    structural_buckets: Dict[tuple, List[Dict]] = defaultdict(list)
    for e in evaluated:
        truth = _structural_truth(e)
        if truth is None:
            continue
        cat = _QUESTION_TYPE_TO_CATEGORY.get(e["question_type"], "other")
        structural_buckets[(cat, truth)].append(e)
    scores["by_structural_truth"] = {
        "marginal_independence": {
            "dep":   _summarize(structural_buckets.get(("marginal_independence", "dep"), [])),
            "indep": _summarize(structural_buckets.get(("marginal_independence", "indep"), [])),
        },
        "conditional_independence": {
            "dep":   _summarize(structural_buckets.get(("conditional_independence", "dep"), [])),
            "indep": _summarize(structural_buckets.get(("conditional_independence", "indep"), [])),
        },
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
    parser.add_argument("--extract-model", default="us.anthropic.claude-opus-4-7",
                        help="Model to use for answer extraction (default: Bedrock Opus 4.8). "
                             "HuggingFace ids (containing '/') load locally; anything else is "
                             "treated as a Bedrock model id.")
    parser.add_argument("--tie-tolerance", type=float, default=_DEFAULT_TIE_TOLERANCE,
                        help=("ε for tie-tolerant scoring on advanced_budget_argmin "
                              "and advanced_rank_topK: an answer counts as correct if "
                              "its do-effect expected target is within ε (in expected-"
                              "state-index units) of the gold's. "
                              f"Default: {_DEFAULT_TIE_TOLERANCE}. Set to 0 to require "
                              "exact (var,value) match."))

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

    # LLM extraction is always on.
    print(f"Loading extraction LLM ({args.extract_model})...")
    extractor = AnswerExtractor(model_name=args.extract_model)
    print("Extraction LLM ready.\n")

    # For each result, get the variable names from the world file for set matching
    world_var_cache = {}
    # Cache question metadata by (world_file, question_id) for tie-tolerant
    # advanced evaluators.  Stored in the world JSON, not in result rows.
    question_meta_cache: Dict[tuple, Dict[str, Any]] = {}

    evaluated = []
    n_extracted = 0
    for ri, r in enumerate(results):
        wf = r["world_file"]
        if wf not in world_var_cache:
            try:
                with open(wf) as f:
                    world_data = json.load(f)
                world_var_cache[wf] = [v["name"] for v in world_data["variables"]]
                for q in world_data.get("questions", []):
                    qid = q.get("id")
                    if qid is not None:
                        question_meta_cache[(wf, qid)] = q.get("metadata") or {}
            except Exception:
                world_var_cache[wf] = None

        all_vars = world_var_cache.get(wf)
        q_meta = question_meta_cache.get((wf, r.get("question_id")))
        answer = r["extracted_answer"]

        # Detect empty / give-up responses: the agent returned no usable answer.
        # These must always be scored as incorrect regardless of ground truth.
        _raw = r.get("raw_response", "")
        _is_empty_response = (
            not _raw.strip()
            or not answer.strip()
            or answer.startswith("GIVE_UP")
            or "Parsing failed" in answer
        )

        question_type = r.get("question_type", "")

        if _is_empty_response:
            ev = {
                "correct": False,
                "detail": "Empty/give-up response (raw_response empty or parsing failed)",
                "answer_parsed": answer,
                "metrics": {},
            }
        else:
            # LLM extraction step for verbose agent responses
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
                                 all_var_names=all_vars,
                                 question_type=r.get("question_type", ""),
                                 question_metadata=q_meta,
                                 tie_tolerance=args.tie_tolerance)
        entry = {
            "world_name": r["world_name"],
            "question_id": r["question_id"],
            "question_type": r.get("question_type", ""),
            "difficulty": r.get("difficulty", ""),
            "n_nodes": r.get("n_nodes", 0),
            "topology": r.get("topology", ""),
            "question_text": r["question_text"],
            "ground_truth": r["ground_truth"],
            "raw_response": r.get("raw_response", ""),
            "original_answer": r["extracted_answer"],
            "extracted_answer": answer,
            "eval": ev,
        }
        evaluated.append(entry)

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

    nsq = scores.get("excluding_set_questions")
    if nsq and nsq["total"] > 0:
        print(f"\n  Excluding set questions ({nsq['total']}):")
        print(f"    Accuracy: {nsq['correct']}/{nsq['total']} = {nsq['accuracy']:.1%}")

    adv_m = scores.get("advanced_metrics") or {}
    if adv_m:
        print(f"\n  Advanced-benchmark metrics (across qualifying questions):")
        for key, stats in adv_m.items():
            print(f"    {key:22s} n={stats['count']:4d}  avg={stats['avg']}")

    print(f"\nBY CATEGORY:")
    for cat, s in scores["by_category"].items():
        line = f"  {cat:30s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}"
        if "avg_f1" in s:
            line += f"  (avg F1: {s['avg_f1']:.3f})"
        print(line)

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

    by_topo = scores.get("by_topology") or {}
    # Only print if topology info was present (any non-"unknown" key).
    if by_topo and not (len(by_topo) == 1 and "unknown" in by_topo):
        print(f"\nBY TOPOLOGY:")
        for topo, s in by_topo.items():
            line = f"  {topo:30s} {s['correct']}/{s['total']} = {s['accuracy']:.1%}"
            if "avg_f1" in s:
                line += f"  (avg F1: {s['avg_f1']:.3f})"
            print(line)

    by_struct = scores.get("by_structural_truth") or {}
    if any(by_struct.get(cat, {}).get(k, {}).get("total", 0)
           for cat in by_struct for k in ("dep", "indep")):
        print(f"\nBY STRUCTURAL TRUTH (dep vs indep):")
        for cat in ("marginal_independence", "conditional_independence"):
            buckets = by_struct.get(cat, {})
            for label in ("dep", "indep"):
                s = buckets.get(label, {})
                if not s or s.get("total", 0) == 0:
                    continue
                print(f"  {cat:26s} {label:6s}  "
                      f"{s['correct']}/{s['total']} = {s['accuracy']:.1%}")

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
                if _has_prf(m):
                    print(f"  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
                else:
                    parts = [f"{k}={v}" for k, v in m.items()]
                    print(f"  Metrics: {', '.join(parts)}")

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
