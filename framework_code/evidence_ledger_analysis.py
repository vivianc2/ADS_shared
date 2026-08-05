"""evidence_ledger_analysis.py — per-turn evidence ledger for coder_new agents.

Operationalizes the paper claim:
"Models differ less in their ability to compute a causal contrast than in
their ability to maintain a stable interpretation of accumulated evidence."

Pipeline (idempotent, resumable):

  1. discover     — scan run dirs for coder_new experiment_*.json files
  2. score-final  — final-answer correctness (basic: log is_correct;
                    advanced: evaluate_advanced.evaluate_one)
  3. annotate     — Bedrock Claude annotator per (trajectory, turn)
  4. metrics      — per-trajectory CCA / ERR / EDV / PSW signals
  5. aggregate    — (model, dataset) cells + paper-ready markdown

CLI:
  python evidence_ledger_analysis.py \\
      --runs results/opus_coder_new_4_19_big \\
             results/llama_coder_new_4_19_big \\
             results/gpt4o_coder_new_4_19_big_subset \\
             results/opus_coder_new_adv_v3 \\
      --out ../analysis/evidence_ledger

  # Quick aggregate-only pass (no LLM calls):
  python evidence_ledger_analysis.py --runs ... --out ... --skip-annotate

  # Sub-stage:
  python evidence_ledger_analysis.py --runs ... --out ... --stage annotate
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("evidence_ledger")

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

ANNOTATOR_DEFAULT = "us.anthropic.claude-opus-4-6-v1"
ANNOTATOR_MAX_NEW_TOKENS = 600
ANNOTATOR_TEMPERATURE = 0.0
ANNOTATOR_RETRIES = 3

CODE_TRUNC = 2000        # chars per code body
OUTPUT_TRUNC = 2000      # chars per code output
REASONING_TRUNC = 4000   # chars of per-turn reasoning shown

VALID_SUPPORT = {"yes", "no", "unclear"}
VALID_STRENGTH = {"strong", "weak", "none"}

ANNOTATOR_FAILED = {
    "has_valid_evidence": False,
    "evidence_supports_gold": "unclear",
    "implied_answer_text": "ANNOTATOR_FAILED",
    "strength": "none",
    "reasoning_summary": "ANNOTATOR_FAILED",
}

# Question-type → estimand hint shown in the prompt
ESTIMAND_HINT = {
    "causal_effect": (
        "Compare P(target | do(source=v)) across at least two distinct "
        "values of the source variable. A non-trivial shift implies the "
        "source IS a cause; flat distributions imply it is NOT a cause."
    ),
    "all_effects_of": (
        "For each candidate downstream variable, compare its distribution "
        "under at least two distinct values of the source via do(). The "
        "implied list = candidates that show a non-trivial shift."
    ),
    "all_causes_of": (
        "For each candidate upstream variable, compare the target under "
        "at least two distinct do() values of the candidate. The implied "
        "list = candidates that move the target."
    ),
    "direct_marginal": (
        "Test marginal independence between X and Y on observational data "
        "(chi-square / Cramér's V). Independent ↔ p large and V small."
    ),
    "chain_marginal": (
        "Same as direct_marginal — observational chi-square between X and Y."
    ),
    "fork_marginal": (
        "Same as direct_marginal — observational chi-square between X and Y."
    ),
    "v_structure_marginal": (
        "Same as direct_marginal — observational chi-square between X and Y."
    ),
    "other_marginal": (
        "Same as direct_marginal — observational chi-square between X and Y."
    ),
    "chain_conditional": (
        "Conditional independence test: chi-square of X vs Y stratified on Z, "
        "or partial association controlling for Z."
    ),
    "fork_conditional": (
        "Conditional independence test: chi-square of X vs Y stratified on Z, "
        "or partial association controlling for Z."
    ),
    "v_structure_conditional": (
        "Conditional independence test: chi-square of X vs Y stratified on Z, "
        "or partial association controlling for Z."
    ),
    "other_conditional": (
        "Conditional independence test: chi-square of X vs Y stratified on Z, "
        "or partial association controlling for Z."
    ),
}

ADVANCED_ESTIMAND_HINT = {
    "confounding_reversal": (
        "Compare E[outcome | do(treatment=v)] across treatment values. The "
        "interventional contrast (NOT observational) is what the question "
        "asks about. Naming the relevant confounder is also expected."
    ),
    "mediator_structure": (
        "Compare effect of T on O with vs without intervening to fix M. The "
        "answer label encodes whether effect goes only through M, also "
        "directly, or M is not a mediator at all."
    ),
    "safety_constrained": (
        "Compute expected target AND expected side-effect under each "
        "candidate intervention. Acceptable answers are those that improve "
        "target while keeping side-effect within tolerance."
    ),
    "satisficing": (
        "Find any intervention whose interventional contrast on target "
        "exceeds the budget threshold."
    ),
    "subgroup_robust": (
        "For each candidate intervention, compute interventional contrast "
        "on target *within each subgroup*. The robust answer beats the "
        "threshold in EVERY subgroup."
    ),
    "invalid_premise": (
        "First decide whether the proposed treatment is intervenable. If "
        "not, identify an alternative intervention that does shift the "
        "target."
    ),
}


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        logger.setLevel(logging.DEBUG)
    # Silence boto3/botocore — they're extremely chatty at DEBUG level.
    for noisy in ("boto3", "botocore", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Path helpers                                                                #
# --------------------------------------------------------------------------- #

def repo_root() -> str:
    """ADS repo root, detected from this file's location."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_world_path(dataset_file: str) -> Optional[str]:
    """Find a world JSON given the orchestrator's `dataset_file` field."""
    candidates: List[str] = []
    candidates.append(dataset_file)
    if not os.path.isabs(dataset_file):
        candidates.append(os.path.join(repo_root(), dataset_file))
        candidates.append(
            os.path.join(repo_root(), "framework_code", dataset_file)
        )
    base = os.path.basename(dataset_file)
    candidates.extend(
        glob.glob(os.path.join(
            repo_root(), "dataset_generation_code", "all_out_bn",
            "**", base,
        ), recursive=True)
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def model_label_from_run_name(run_name: str) -> str:
    """Best-effort model label parsed from the run dir basename."""
    n = run_name.lower()
    if "opus" in n:
        return "Opus"
    if "gpt4o" in n or "gpt-4o" in n or "gpt4" in n:
        return "GPT-4o"
    if "llama" in n:
        return "Llama"
    if "claude" in n:
        return "Claude"
    return run_name


def dataset_label_from_world_path(world_path: Optional[str]) -> str:
    if not world_path:
        return "unknown"
    parts = os.path.normpath(world_path).split(os.sep)
    # take the directory immediately above the world file
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


# --------------------------------------------------------------------------- #
# Stage 1 — discover                                                          #
# --------------------------------------------------------------------------- #

def is_coder_new_log(exp: Dict[str, Any], run_name: str) -> bool:
    """Detect whether an experiment_*.json was produced by coder_new."""
    if "coder_new" in run_name.lower():
        return True
    for t in exp.get("turns", []):
        phases = (t.get("phase_summary") or {}).get("phases_run", [])
        if any(p in ("init", "analysis", "design") for p in phases):
            return True
        if any(p.startswith("code_round_") for p in phases):
            return True
    return False


def discover(run_dirs: List[str]) -> List[Dict[str, Any]]:
    """Walk run dirs and collect per-trajectory metadata."""
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        run_name = os.path.basename(run_dir.rstrip("/"))
        log_dir = os.path.join(run_dir, "agent_logs")
        if not os.path.isdir(log_dir):
            logger.warning(f"no agent_logs/ in {run_dir}; skipping")
            continue

        exp_files = sorted(glob.glob(os.path.join(log_dir, "experiment_*.json")))
        n_skipped = 0
        for ep in exp_files:
            try:
                with open(ep) as f:
                    exp = json.load(f)
            except Exception as e:
                logger.warning(f"could not read {ep}: {e}; skipping")
                n_skipped += 1
                continue

            if not is_coder_new_log(exp, run_name):
                logger.debug(f"not coder_new, skipping {ep}")
                n_skipped += 1
                continue

            dataset_file = exp.get("dataset_file") or ""
            world_path = resolve_world_path(dataset_file) if dataset_file else None

            qtype = (exp.get("question") or {}).get("question_type", "")
            archetype = None
            if qtype.startswith("advanced_"):
                archetype = qtype.replace("advanced_", "")

            rows.append({
                "experiment_path": os.path.abspath(ep),
                "run_name": run_name,
                "model_label": model_label_from_run_name(run_name),
                "dataset_label": dataset_label_from_world_path(world_path),
                "world_path": world_path,
                "scientist_model": exp.get("scientist_model"),
                "question_type": qtype,
                "archetype": archetype,
                "ground_truth": (exp.get("question") or {}).get("ground_truth"),
                "is_correct_log": exp.get("is_correct"),
                "n_turns": len(exp.get("turns", [])),
            })
        logger.info(
            f"{run_name}: discovered={len(exp_files) - n_skipped}, skipped={n_skipped}"
        )
    return rows


# --------------------------------------------------------------------------- #
# Stage 2 — final-answer correctness                                          #
# --------------------------------------------------------------------------- #

def _trajectory_hash(experiment_path: str) -> str:
    """Stable hash for cache key — uses file path + mtime."""
    st = os.stat(experiment_path)
    return hashlib.sha1(
        f"{experiment_path}|{int(st.st_mtime)}|{st.st_size}".encode()
    ).hexdigest()


def score_final_basic(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Use the orchestrator's already-computed is_correct flag."""
    is_correct = exp.get("is_correct")
    return {"correct": bool(is_correct), "source": "log_is_correct"}


def _answer_turn_raw_response(exp: Dict[str, Any]) -> str:
    """Return the raw_llm_response from the answer turn (the full LLM output
    including any <answer> tags).  Falls back to scientist_answer if not found.

    coder_new experiment logs store the complete LLM text in
    turns[i]["raw_llm_response"] for the turn where turn_type == "answer".
    scientist_answer at the top level is just the scientist_input excerpt,
    which is often a truncated prose fragment — not suitable for extraction."""
    turns = exp.get("turns") or []
    # prefer the last answer-type turn
    for t in reversed(turns):
        if t.get("turn_type") == "answer":
            raw = t.get("raw_llm_response") or ""
            if raw.strip():
                return raw
    # fallback: last turn's raw_llm_response
    if turns:
        raw = turns[-1].get("raw_llm_response") or ""
        if raw.strip():
            return raw
    return exp.get("scientist_answer", "")


def score_final_advanced(
    exp: Dict[str, Any], world: Dict[str, Any], extractor: Any,
) -> Dict[str, Any]:
    """Use evaluate_advanced.evaluate_one for the canonical extracted-grade.

    Uses the answer turn's raw_llm_response (full LLM output) as the text to
    extract from, NOT scientist_answer (which is a truncated prose fragment)."""
    try:
        from evaluate_advanced import (
            evaluate_one, DEFAULT_TIE_TOLERANCE, DEFAULT_SAFETY_TOLERANCE,
            DEFAULT_SUBGROUP_MIN_IMPROVEMENT,
        )
    except Exception as e:
        logger.warning(f"could not import evaluate_advanced: {e}; "
                       f"falling back to log is_correct")
        return {"correct": bool(exp.get("is_correct")),
                "source": "log_is_correct_fallback",
                "error": str(e)}

    question = exp.get("question") or {}
    qid = (question.get("metadata") or {}).get("id", 0)
    full_response = _answer_turn_raw_response(exp)
    result_row = {
        "question_type": question.get("question_type"),
        "question_id": qid,
        "question_text": question.get("question_text"),
        "extracted_answer": full_response,
        "raw_response": full_response,
    }
    try:
        out = evaluate_one(
            extractor, result_row, world,
            tie_tolerance=DEFAULT_TIE_TOLERANCE,
            safety_tolerance=DEFAULT_SAFETY_TOLERANCE,
            subgroup_min_improvement=DEFAULT_SUBGROUP_MIN_IMPROVEMENT,
        )
        return {"correct": bool(out.get("correct")),
                "source": "evaluate_advanced",
                "detail": out.get("detail", "")}
    except Exception as e:
        logger.warning(f"evaluate_advanced failed: {e}; using log is_correct")
        return {"correct": bool(exp.get("is_correct")),
                "source": "log_is_correct_fallback",
                "error": str(e)}


def _run_name_from_eval_source(eval_source: Any) -> Optional[str]:
    """Extract `run_name` from an eval file's top-level `source` field.

    Observed `source` shapes:
      - "./results/<run_name>/<run_name>.json"            (agent / coder runs)
      - "/abs/path/results/<run_name>/..."
      - "agent_logs:/abs/path/results/<run_name>"          (evaluate_from_logs.py)
      - ["./results/<file_a>.json", "./results/<file_b>.json"]   (zero-shot bundles
            multiple inputs; each entry points at a results FILE, not a run dir,
            so the per-string parse returns None and the caller falls back to
            the eval filename stem — which is the correct run identity here.)

    Returns the first run_name that can be extracted from any string in
    `eval_source`, else None.
    """
    if not eval_source:
        return None
    if isinstance(eval_source, list):
        for item in eval_source:
            run = _run_name_from_eval_source(item)
            if run is not None:
                return run
        return None
    if not isinstance(eval_source, str):
        return None
    s = eval_source.replace("\\", "/")
    # Strip a leading "kind:" prefix (e.g. "agent_logs:") if it precedes a path.
    if ":" in s:
        tail = s.rsplit(":", 1)[1]
        if tail.startswith(("/", "./", "../")):
            s = tail
    m = re.search(r"(?:^|/)results/([^/]+)", s)
    if not m:
        return None
    seg = m.group(1)
    if seg.endswith(".json"):
        return None
    return seg


def load_evaluations_index(
    paths: List[str],
) -> Dict[Tuple[str, str, Any], Dict[str, Any]]:
    """Build lookup (run_name, world_name, question_id) → {correct, source, detail}
    from eval JSON files produced by evaluate_from_logs.py / evaluate_advanced.py /
    evaluate_zero_shot.py.

    Keying includes `run_name` so that several eval files covering the same
    (world, qid) — e.g. opus vs. llama on the same dataset — do not collide.
    Each eval file's run_name is parsed from its top-level `source` field
    (which always points to results/<run_name>/...). If `source` is missing
    or unparseable, the eval filename (`eval_<run>.json`) is used as a
    best-effort fallback; if that also fails, the file is skipped with a
    warning rather than silently merging into another run.
    """
    idx: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "eval_*.json"))))
        else:
            files.extend(sorted(glob.glob(p)))
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception as e:
            logger.warning(f"unreadable eval file {fp}: {e}")
            continue
        run_name = _run_name_from_eval_source(d.get("source", ""))
        if run_name is None:
            base = os.path.basename(fp)
            if base.startswith("eval_") and base.endswith(".json"):
                run_name = base[len("eval_"):-len(".json")]
                logger.warning(
                    f"{base}: could not derive run_name from `source` field; "
                    f"falling back to filename stem '{run_name}' (this may "
                    f"mismatch the actual run dir — verify if accuracy looks off)"
                )
            else:
                logger.warning(
                    f"skipping {base}: no `source` field and filename does "
                    f"not match eval_<run>.json; cannot derive run_name"
                )
                continue
        n_added = 0
        n_collision = 0
        for ev in d.get("evaluated", []):
            wn = ev.get("world_name")
            qid = ev.get("question_id")
            if wn is None or qid is None:
                continue
            key = (run_name, wn, qid)
            if key in idx:
                n_collision += 1
            idx[key] = {
                "correct": bool(ev.get("eval", {}).get("correct", False)),
                "source": f"eval_file:{os.path.basename(fp)}",
                "detail": ev.get("eval", {}).get("detail", ""),
            }
            n_added += 1
        if n_collision:
            logger.warning(
                f"{os.path.basename(fp)}: {n_collision}/{n_added} entries "
                f"collided with earlier entries for run={run_name} (later wins). "
                f"This usually means two eval files cover the same run — "
                f"deduplicate them."
            )
    logger.info(f"loaded {len(idx)} eval entries from {len(files)} files")
    return idx


def score_final_correctness(
    discovered: List[Dict[str, Any]], cache_path: str, advanced_extractor: Any,
    evals_index: Optional[Dict[Tuple[str, str, Any], Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Returns: experiment_path → {correct: bool, source: str}.

    Priority:
      1. evals_index (built from --evaluations files) — always used when provided.
         Trajectories not found in the index get source="not_in_eval_file" and
         are NOT silently re-scored; pass additional eval files to cover them.
      2. (fallback, only when --evaluations is omitted)
         log's is_correct flag (basic) / score_final_advanced (advanced).
    """
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    n_from_eval = 0
    n_missing_eval = 0
    n_fallback = 0

    for row in discovered:
        ep = row["experiment_path"]
        key = _trajectory_hash(ep)
        if key in cache:
            continue

        if evals_index is not None:
            # eval-file mode: always read from index; never fall back to LLM scoring.
            # Read the experiment log once to get both world name and qid.
            # World name: prefer world_path (already resolved by discover); fall back
            # to dataset_file basename from the log so resolution failures don't
            # silently produce a None lookup.
            try:
                with open(ep) as f:
                    exp_quick = json.load(f)
                qid = (exp_quick.get("question") or {}).get("metadata", {}).get("id", 0)
                if row.get("world_path"):
                    wn = os.path.splitext(os.path.basename(row["world_path"]))[0]
                else:
                    df = exp_quick.get("dataset_file", "")
                    wn = os.path.splitext(os.path.basename(df))[0] if df else None
            except Exception as e:
                logger.warning(f"could not read {ep} for eval lookup: {e}")
                qid = 0
                wn = (os.path.splitext(os.path.basename(row.get("world_path") or ""))[0]
                      if row.get("world_path") else None)
            run_name = row["run_name"]
            lookup = evals_index.get((run_name, wn, qid)) if wn else None
            if lookup is not None:
                cache[key] = lookup
                n_from_eval += 1
            else:
                logger.warning(
                    f"no eval-file entry for run={run_name} world={wn} qid={qid} "
                    f"({os.path.basename(ep)}); marking as not_in_eval_file"
                )
                cache[key] = {
                    "correct": False,
                    "source": "not_in_eval_file",
                    "detail": (f"run={run_name} world={wn} qid={qid} not found "
                               f"in --evaluations index"),
                }
                n_missing_eval += 1
            continue

        # fallback path (no --evaluations supplied)
        try:
            with open(ep) as f:
                exp = json.load(f)
        except Exception as e:
            logger.warning(f"could not read {ep}: {e}")
            cache[key] = {"correct": False, "source": "read_error",
                          "error": str(e)}
            continue
        if (row["question_type"] or "").startswith("advanced_"):
            world = None
            if row["world_path"]:
                try:
                    with open(row["world_path"]) as f:
                        world = json.load(f)
                except Exception as e:
                    logger.warning(f"could not read world {row['world_path']}: {e}")
            if world is not None and advanced_extractor is not None:
                cache[key] = score_final_advanced(exp, world, advanced_extractor)
            else:
                cache[key] = {"correct": bool(exp.get("is_correct")),
                              "source": "log_is_correct_no_world"}
        else:
            cache[key] = score_final_basic(exp)
        n_fallback += 1

    if evals_index is not None:
        logger.info(
            f"final-correctness: {n_from_eval} from eval files, "
            f"{n_missing_eval} missing (marked not_in_eval_file)"
        )
        if n_missing_eval:
            logger.warning(
                f"{n_missing_eval} trajectories have no eval-file entry — "
                f"they will be excluded from accuracy metrics. "
                f"Pass additional --evaluations files to cover them."
            )
    else:
        if n_fallback:
            has_advanced = any(
                (row.get("question_type") or "").startswith("advanced_")
                for row in discovered
            )
            if has_advanced:
                logger.warning(
                    f"--evaluations not supplied; scoring {n_fallback} advanced "
                    f"trajectories via score_final_advanced (uses answer-turn "
                    f"raw_llm_response). Pass --evaluations for reliable scores."
                )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    out: Dict[str, Dict[str, Any]] = {}
    for row in discovered:
        out[row["experiment_path"]] = cache[_trajectory_hash(row["experiment_path"])]
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — annotator                                                         #
# --------------------------------------------------------------------------- #

ANNOTATOR_SYSTEM = """You are a precise annotator for an evidence-ledger analysis. You will see ONE
turn from a scientist agent's trajectory plus the original question, the
ground-truth answer, and a summary of prior turns.

Your job: decide whether THIS turn produced valid causal evidence for the
asked question, and if so, whether that evidence — taken in isolation —
points toward the ground truth.

Output ONLY a JSON object with these exact fields:
{
  "has_valid_evidence": <true|false>,
  "evidence_supports_gold": "<yes|no|unclear>",
  "implied_answer_text": "<one short sentence describing what answer this turn's evidence implies>",
  "strength": "<strong|weak|none>",
  "reasoning_summary": "<one short sentence summarizing what the turn computed>"
}

Rules:
- "has_valid_evidence" is TRUE only if THIS turn computed the CORRECT type of
  estimand for the question (interventional contrast for causal-effect
  questions, association/conditional-independence test for independence
  questions, archetype-specific estimand for advanced questions). The estimand
  must actually be COMPUTED in this turn (not just discussed). Off-target
  queries, pure narrative, and unrelated computations count as FALSE.
- "evidence_supports_gold":
   * "yes"   = this turn's evidence, read in isolation, leads to the gold answer.
   * "no"    = this turn's evidence, read in isolation, leads to a different answer.
   * "unclear" = the evidence is too weak or too ambiguous to commit either way.
- "strength":
   * "strong" = adequate sample (typically n >= 1000), clear effect/p-value,
                full comparison covered.
   * "weak"   = comparison present but small effect, small sample, or only
                covers a slice of the needed comparison.
   * "none"   = no valid comparison.
- "reasoning_summary": ONE short factual sentence about what the turn computed.
- "implied_answer_text": ONE short sentence describing the answer the
  evidence implies.

If has_valid_evidence is FALSE, set evidence_supports_gold to "unclear",
strength to "none", and implied_answer_text to "none".

Output only the JSON object, with no surrounding prose."""


def _truncate(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n…[truncated, {len(s) - n} chars omitted]"


def _intervention_str(interv: List[Dict[str, Any]]) -> str:
    if not interv:
        return ""
    parts = []
    for d in interv:
        if isinstance(d, dict):
            parts.extend([f"{k}={v}" for k, v in d.items()])
    return ",".join(parts)


def _short_query_summary(parsed: Optional[Dict[str, Any]]) -> str:
    if not parsed:
        return "(no query)"
    qt = parsed.get("query_type", "?")
    n = parsed.get("n_samples", "?")
    interv = _intervention_str(parsed.get("interventions") or [])
    if qt == "interventional" and interv:
        return f"{qt} do({interv}) n={n}"
    return f"{qt} n={n}"


def _gold_block_for_advanced(world: Optional[Dict[str, Any]]) -> str:
    """Compact summary of the advanced question's gold/roles for the prompt."""
    if not world:
        return ""
    qs = world.get("questions") or []
    if not qs:
        return ""
    md = qs[0].get("metadata") or {}
    archetype = md.get("archetype") or ""
    roles = md.get("roles") or {}
    gold = md.get("gold") or {}
    sub = md.get("sub_variant") or ""
    parts = [f"Archetype: {archetype}"]
    if sub:
        parts.append(f"Sub-variant: {sub}")
    if roles:
        roles_s = ", ".join(f"{k}={v}" for k, v in roles.items())
        parts.append(f"Roles: {roles_s}")
    hint = ADVANCED_ESTIMAND_HINT.get(archetype, "")
    if hint:
        parts.append(f"Required estimand: {hint}")
    if gold:
        gold_s = json.dumps(gold, ensure_ascii=False)
        parts.append(f"Gold metadata: {_truncate(gold_s, 600)}")
    return "\n".join(parts)


def build_annotator_user_prompt(
    exp: Dict[str, Any], world: Optional[Dict[str, Any]],
    turn_idx: int, prior_annotations: List[Dict[str, Any]],
) -> str:
    question = exp.get("question") or {}
    question_text = question.get("question_text", "")
    qtype = question.get("question_type", "")
    gt = question.get("ground_truth")
    relevant_vars = (question.get("metadata") or {}).get("relevant_variables") or []

    archetype_block = ""
    estimand_hint = ""
    if qtype.startswith("advanced_"):
        archetype_block = _gold_block_for_advanced(world)
    else:
        estimand_hint = ESTIMAND_HINT.get(qtype, "")

    turns = exp.get("turns", [])
    # build prior-turn summary lines
    prior_lines: List[str] = []
    for j, t in enumerate(turns[:turn_idx]):
        parsed = t.get("parsed_query") or {}
        ann = prior_annotations[j] if j < len(prior_annotations) else None
        ann_tag = ""
        if ann:
            ann_tag = (f" (annotator: {ann.get('strength', '?')}/"
                       f"{ann.get('evidence_supports_gold', '?')})")
        prior_lines.append(
            f"  T{j + 1}: {_short_query_summary(parsed)}{ann_tag}"
        )
    prior_block = "\n".join(prior_lines) if prior_lines else "  (none)"

    t = turns[turn_idx]
    turn_type = t.get("turn_type", "?")
    parsed = t.get("parsed_query") or {}
    sci_input = t.get("scientist_input", "") or ""
    reasoning = t.get("reasoning", "") or ""
    code_rounds = t.get("code_rounds") or []

    code_block_parts: List[str] = []
    for cr in code_rounds:
        idx = cr.get("round_num") or len(code_block_parts) + 1
        errored = "ERROR" if cr.get("errored") else "ok"
        code = _truncate(cr.get("code", ""), CODE_TRUNC)
        out = _truncate(cr.get("output", ""), OUTPUT_TRUNC)
        code_block_parts.append(
            f"--- round {idx} ({errored}) ---\nCODE:\n{code}\nOUTPUT:\n{out}"
        )
    code_block = "\n".join(code_block_parts) if code_block_parts else "(no code rounds)"

    user = (
        f"Original question:\n{question_text}\n\n"
        f"Question type: {qtype}\n"
    )
    if archetype_block:
        user += archetype_block + "\n"
    if estimand_hint:
        user += f"Required estimand: {estimand_hint}\n"
    user += f"Ground truth answer: {json.dumps(gt, ensure_ascii=False)}\n"
    if relevant_vars:
        user += f"Relevant variables: {', '.join(relevant_vars)}\n"
    user += (
        f"\nPrior turn summary (data already in hand before this turn):\n"
        f"{prior_block}\n"
        f"\nThis turn (turn {turn_idx + 1}, type={turn_type}):\n"
        f"[scientist_input]\n{_truncate(sci_input, 1500)}\n\n"
        f"[query parsed]\n{_short_query_summary(parsed)}\n\n"
        f"[reasoning]\n{_truncate(reasoning, REASONING_TRUNC)}\n\n"
        f"[code rounds]\n{code_block}\n"
    )
    return user


def _parse_first_json(text: str) -> Dict[str, Any]:
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start: i + 1])
    raise ValueError("No balanced JSON object")


def _validate_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce / sanitize annotator output."""
    out = dict(ANNOTATOR_FAILED)
    out["has_valid_evidence"] = bool(ann.get("has_valid_evidence", False))
    sup = str(ann.get("evidence_supports_gold", "unclear")).strip().lower()
    out["evidence_supports_gold"] = sup if sup in VALID_SUPPORT else "unclear"
    strn = str(ann.get("strength", "none")).strip().lower()
    out["strength"] = strn if strn in VALID_STRENGTH else "none"
    out["implied_answer_text"] = str(ann.get("implied_answer_text", "none"))[:500]
    out["reasoning_summary"] = str(ann.get("reasoning_summary", ""))[:500]
    if not out["has_valid_evidence"]:
        out["evidence_supports_gold"] = "unclear"
        out["strength"] = "none"
    return out


def annotate_turn(
    llm: Any, exp: Dict[str, Any], world: Optional[Dict[str, Any]],
    turn_idx: int, prior_annotations: List[Dict[str, Any]],
    cache_dir: str,
) -> Dict[str, Any]:
    user_prompt = build_annotator_user_prompt(
        exp, world, turn_idx, prior_annotations,
    )
    cache_key = hashlib.sha1(
        (ANNOTATOR_SYSTEM + "\n||\n" + user_prompt).encode()
    ).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return _validate_annotation(cached)

    last_err: Optional[Exception] = None
    for attempt in range(ANNOTATOR_RETRIES):
        try:
            raw = llm.generate(ANNOTATOR_SYSTEM, user_prompt)
            ann = _parse_first_json(raw)
            ann = _validate_annotation(ann)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(ann, f, indent=2)
            return ann
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    logger.warning(
        f"annotator failed for turn={turn_idx} after {ANNOTATOR_RETRIES} retries: "
        f"{last_err}"
    )
    return dict(ANNOTATOR_FAILED)


def annotate_trajectories(
    discovered: List[Dict[str, Any]], ledger_path: str, cache_dir: str,
    annotator_model: str, limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """For each trajectory: annotate every turn. Returns ledger rows."""
    # resume from previous run — but discard rows where EVERY turn failed,
    # those came from a broken annotator session and need to be redone.
    done: Dict[str, Dict[str, Any]] = {}
    n_discarded = 0
    if os.path.isfile(ledger_path):
        kept_rows: List[Dict[str, Any]] = []
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                turns = row.get("turns") or []
                all_failed = bool(turns) and all(
                    (t.get("annotation") or {}).get("reasoning_summary")
                    == "ANNOTATOR_FAILED"
                    for t in turns
                )
                if all_failed:
                    n_discarded += 1
                    continue
                done[row["experiment_path"]] = row
                kept_rows.append(row)
        if n_discarded:
            logger.info(
                f"discarded {n_discarded} ledger rows where every turn was "
                f"ANNOTATOR_FAILED; rewriting ledger.jsonl"
            )
            with open(ledger_path, "w") as f:
                for r in kept_rows:
                    f.write(json.dumps(r) + "\n")

    try:
        from bedrock_llm import BedrockLLM
    except Exception as e:
        raise SystemExit(
            f"BedrockLLM unavailable: {e}. Either install boto3 or pass "
            f"--skip-annotate."
        )

    # Fail fast on credential / region / model-id problems.
    try:
        probe = BedrockLLM(
            model_id=annotator_model, temperature=0.0, max_new_tokens=8,
        )
        probe.generate("You reply with one word.", "Say 'pong'.")
        logger.info(f"bedrock sanity check OK ({annotator_model})")
    except Exception as e:
        raise SystemExit(
            f"\n[bedrock sanity check FAILED] model={annotator_model}\n"
            f"  {type(e).__name__}: {e}\n"
            f"Common causes:\n"
            f"  - model id has a trailing ':0' (Anthropic models on this "
            f"account use the suffix-less form, e.g. "
            f"'us.anthropic.claude-opus-4-6-v1')\n"
            f"  - model not enabled in your Bedrock account "
            f"(check the model-access page)\n"
            f"  - missing AWS_DEFAULT_REGION or credentials\n"
            f"Pass --annotator-model <id> to override.\n"
        )

    llm = BedrockLLM(
        model_id=annotator_model,
        temperature=ANNOTATOR_TEMPERATURE,
        max_new_tokens=ANNOTATOR_MAX_NEW_TOKENS,
    )

    # rewrite the ledger from scratch using `done` as a base + new annotations
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

    out_rows: List[Dict[str, Any]] = []
    todo = discovered if limit is None else discovered[:limit]
    n_total = len(todo)
    for i, row in enumerate(todo, 1):
        ep = row["experiment_path"]
        if ep in done:
            out_rows.append(done[ep])
            continue
        try:
            with open(ep) as f:
                exp = json.load(f)
        except Exception as e:
            logger.warning(f"skip unreadable {ep}: {e}")
            continue
        world = None
        if row["world_path"]:
            try:
                with open(row["world_path"]) as f:
                    world = json.load(f)
            except Exception as e:
                logger.warning(f"missing world {row['world_path']}: {e}")

        turns = exp.get("turns", [])
        annotations: List[Dict[str, Any]] = []
        for j in range(len(turns)):
            ann = annotate_turn(llm, exp, world, j, annotations, cache_dir)
            annotations.append(ann)

        ledger_row = {
            **row,
            "scientist_answer": exp.get("scientist_answer", ""),
            "n_turns": len(turns),
            "turns": [
                {
                    "turn_idx": j,
                    "turn_number": (turns[j].get("turn_number") or j + 1),
                    "turn_type": turns[j].get("turn_type"),
                    "query_type": (turns[j].get("parsed_query") or {}).get("query_type"),
                    "interventions": (turns[j].get("parsed_query") or {}).get("interventions"),
                    "annotation": ann,
                }
                for j, ann in enumerate(annotations)
            ],
        }
        out_rows.append(ledger_row)
        # append to ledger immediately so partial progress is preserved
        with open(ledger_path, "a") as f:
            f.write(json.dumps(ledger_row) + "\n")

        if i % 10 == 0 or i == n_total:
            logger.info(f"annotated {i}/{n_total} trajectories")

    return out_rows


# --------------------------------------------------------------------------- #
# Stage 4 — per-trajectory metrics                                            #
# --------------------------------------------------------------------------- #

def compute_per_trajectory_metrics(
    ledger_rows: List[Dict[str, Any]],
    final_correctness: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n_excluded = 0
    for row in ledger_rows:
        ep = row["experiment_path"]
        fc = final_correctness.get(ep, {})
        if fc.get("source") == "not_in_eval_file":
            n_excluded += 1
            continue
        passed = bool(fc.get("correct", False))
        turns = row.get("turns", [])
        n_turns = len(turns)

        first_strong_correct: Optional[int] = None
        had_strong_correct = False
        had_strong_incorrect = False
        had_any_valid = False
        for t in turns:
            ann = t.get("annotation") or {}
            if ann.get("has_valid_evidence"):
                had_any_valid = True
            if (ann.get("has_valid_evidence")
                    and ann.get("strength") == "strong"
                    and ann.get("evidence_supports_gold") == "yes"):
                had_strong_correct = True
                if first_strong_correct is None:
                    first_strong_correct = t["turn_idx"]
            if (ann.get("has_valid_evidence")
                    and ann.get("strength") == "strong"
                    and ann.get("evidence_supports_gold") == "no"):
                had_strong_incorrect = True

        psw: Optional[int] = None
        if first_strong_correct is not None and n_turns > 0:
            psw = max(0, n_turns - first_strong_correct - 1)

        out.append({
            "experiment_path": ep,
            "run_name": row["run_name"],
            "model_label": row["model_label"],
            "dataset_label": row["dataset_label"],
            "question_type": row["question_type"],
            "archetype": row.get("archetype"),
            "ground_truth": row.get("ground_truth"),
            "scientist_answer": row.get("scientist_answer", ""),
            "passed": passed,
            "final_correctness_source": final_correctness.get(ep, {}).get("source"),
            "n_turns": n_turns,
            "had_any_valid_evidence": had_any_valid,
            "had_strong_correct_evidence": had_strong_correct,
            "had_strong_incorrect_evidence": had_strong_incorrect,
            "first_strong_correct_turn": first_strong_correct,
            "psw": psw,
            "walked_away_from_correct_evidence": had_strong_correct and not passed,
            "retention_consistent": (passed == had_strong_correct),
        })
    if n_excluded:
        logger.warning(
            f"excluded {n_excluded} trajectories with source=not_in_eval_file "
            f"from metrics"
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 5 — aggregate                                                          #
# --------------------------------------------------------------------------- #

def _safe_div(a: int, b: int) -> Optional[float]:
    return round(a / b, 3) if b else None


def aggregate(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_model_qtype: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        by_cell[(m["model_label"], m["dataset_label"])].append(m)
        by_model_qtype[(m["model_label"], m["question_type"] or "?")].append(m)

    def _summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        n_passed = sum(1 for m in items if m["passed"])
        n_strong_correct = sum(1 for m in items if m["had_strong_correct_evidence"])
        n_failed = n - n_passed
        n_walked = sum(
            1 for m in items
            if m["walked_away_from_correct_evidence"]
        )
        n_passed_with_strong = sum(
            1 for m in items
            if m["passed"] and m["had_strong_correct_evidence"]
        )
        psws = [m["psw"] for m in items if m["psw"] is not None]
        return {
            "n_total": n,
            "n_passed": n_passed,
            "accuracy": _safe_div(n_passed, n),
            "n_with_strong_correct_evidence": n_strong_correct,
            "CCA": _safe_div(n_strong_correct, n),
            "ERR": _safe_div(n_passed_with_strong, n_strong_correct),
            "n_failed": n_failed,
            "n_walked_away": n_walked,
            "EDV": _safe_div(n_walked, n_failed),
            "mean_psw": round(statistics.mean(psws), 2) if psws else None,
            "median_psw": (round(statistics.median(psws), 2)
                           if psws else None),
            "n_with_psw": len(psws),
        }

    cells = {
        f"{model}__{ds}": _summarize(items)
        for (model, ds), items in sorted(by_cell.items())
    }
    by_qtype = {
        f"{model}__{qt}": _summarize(items)
        for (model, qt), items in sorted(by_model_qtype.items())
    }
    return {"by_model_dataset": cells, "by_model_question_type": by_qtype}


# --------------------------------------------------------------------------- #
# Markdown writers                                                            #
# --------------------------------------------------------------------------- #

def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}".rstrip("0").rstrip(".") or "0"
    return str(x)


def render_markdown(
    aggregated: Dict[str, Any],
    metrics: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Evidence-Ledger Analysis — `coder_new` agents\n")
    lines.append(
        "**Paper claim under test.** Models differ less in their ability to "
        "compute a causal contrast (CCA) than in their ability to maintain a "
        "stable interpretation of accumulated evidence (ERR / EDV).\n\n"
        "**Definitions.** A trajectory has *strong correct evidence* if some "
        "turn computed the right estimand AND its conclusion (taken in "
        "isolation) matches the gold answer. CCA is the share with strong "
        "correct evidence. ERR is the share that also gets the FINAL answer "
        "right. EDV is, among failures, the share that *had* strong correct "
        "evidence at some prior turn. PSW = turns elapsed after the first "
        "such turn.\n"
    )

    lines.append("## Headline — by (model, dataset)\n")
    lines.append("| Model | Dataset | n | Accuracy | CCA | ERR | EDV | mean PSW | median PSW |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, s in aggregated["by_model_dataset"].items():
        model, ds = key.split("__", 1)
        lines.append(
            f"| {model} | {ds} | {s['n_total']} | {_fmt(s['accuracy'])} | "
            f"{_fmt(s['CCA'])} | {_fmt(s['ERR'])} | {_fmt(s['EDV'])} | "
            f"{_fmt(s['mean_psw'])} | {_fmt(s['median_psw'])} |"
        )
    lines.append("")

    lines.append("## By (model, question type)\n")
    lines.append("| Model | Question type | n | Accuracy | CCA | ERR | EDV | mean PSW |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, s in aggregated["by_model_question_type"].items():
        model, qt = key.split("__", 1)
        lines.append(
            f"| {model} | {qt} | {s['n_total']} | {_fmt(s['accuracy'])} | "
            f"{_fmt(s['CCA'])} | {_fmt(s['ERR'])} | {_fmt(s['EDV'])} | "
            f"{_fmt(s['mean_psw'])} |"
        )
    lines.append("")

    # qualitative case studies — worst PSW walked-away trajectory per (model, ds)
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        by_cell[(m["model_label"], m["dataset_label"])].append(m)

    lines.append("## Walk-away case studies\n")
    lines.append(
        "For each (model, dataset) cell we show one walked-away trajectory: "
        "the model had strong correct evidence and then committed to the "
        "wrong final answer. PSW is the number of turns AFTER the first "
        "sufficient one.\n"
    )
    for (model, ds), items in sorted(by_cell.items()):
        walked = [m for m in items if m["walked_away_from_correct_evidence"]]
        if not walked:
            continue
        worst = max(walked, key=lambda m: (m["psw"] or 0))
        ep_short = os.path.basename(worst["experiment_path"])
        lines.append(f"### {model} / {ds} — {ep_short}\n")
        lines.append(f"- Question type: `{worst['question_type']}`")
        lines.append(f"- Ground truth: `{worst['ground_truth']}`")
        sa = (worst.get("scientist_answer") or "").strip().replace("\n", " ")
        lines.append(f"- Final answer (truncated): `{sa[:200]}`")
        lines.append(f"- n_turns={worst['n_turns']}, "
                     f"first_strong_correct_turn={worst['first_strong_correct_turn']}, "
                     f"PSW={worst['psw']}")
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- Annotator output is cached on disk; re-runs are free.")
    lines.append("- Trajectories without strong correct evidence are excluded from "
                 "ERR and PSW (logged as `n_with_strong_correct_evidence < n_total`).")
    lines.append("- Final-correctness is read from eval JSON files in "
                 "`evaluations/for_paper/` (auto-discovered). Trajectories "
                 "without a matching eval entry are excluded from all metrics.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-turn evidence ledger for coder_new agents."
    )
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories under framework_code/results/. "
                        "Either absolute or relative to repo root.")
    p.add_argument("--out", required=True,
                   help="Output directory (e.g. analysis/evidence_ledger).")
    p.add_argument("--annotator-model", default=ANNOTATOR_DEFAULT,
                   help="Bedrock model id for the annotator.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N trajectories per run.")
    p.add_argument("--stage", default="run",
                   choices=["run", "discover", "score-final", "annotate",
                            "metrics", "aggregate"])
    p.add_argument("--skip-annotate", action="store_true",
                   help="Skip the LLM annotator stage. Aggregate uses "
                        "final-correctness only (CCA/ERR/EDV/PSW will be "
                        "trivial / null).")
    p.add_argument("--evaluations", nargs="*", default=None,
                   help="Path(s) to LLM-graded eval JSONs or directories "
                        "(from evaluate_from_logs.py / evaluate_advanced.py / "
                        "evaluate_zero_shot.py). Accepts files, dirs, or "
                        "globs. Defaults to evaluations/for_paper/ next to "
                        "this script. `passed` for each trajectory is read "
                        "from the matched eval entry (world_name + "
                        "question_id). Pass --evaluations with no args to "
                        "disable eval-file lookup entirely.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def normalize_run_dirs(run_args: List[str]) -> List[str]:
    out: List[str] = []
    for r in run_args:
        candidates = [
            r,
            os.path.abspath(r),
            os.path.join(repo_root(), r),
            os.path.join(repo_root(), "framework_code", r),
            os.path.join(repo_root(), "framework_code", "results",
                         os.path.basename(r)),
        ]
        chosen = next((c for c in candidates if os.path.isdir(c)), None)
        if chosen:
            out.append(os.path.abspath(chosen))
        else:
            logger.warning(f"run dir not found: {r}")
    return out


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    discovered_path = os.path.join(out_dir, "discovered.json")
    final_cache_path = os.path.join(out_dir, "final_correctness_cache.json")
    ledger_path = os.path.join(out_dir, "ledger.jsonl")
    annotator_cache_dir = os.path.join(out_dir, "annotator_cache")
    metrics_path = os.path.join(out_dir, "per_trajectory_metrics.json")
    aggregate_path = os.path.join(out_dir, "aggregated.json")
    md_path = os.path.join(out_dir, "aggregated_table.md")

    run_dirs = normalize_run_dirs(args.runs)
    if not run_dirs:
        sys.exit("no valid run directories")

    # Stage 1: discover
    if args.stage in ("run", "discover") or not os.path.isfile(discovered_path):
        discovered = discover(run_dirs)
        with open(discovered_path, "w") as f:
            json.dump(discovered, f, indent=2)
        logger.info(f"discovered {len(discovered)} trajectories → {discovered_path}")
        if args.stage == "discover":
            return
    else:
        with open(discovered_path) as f:
            discovered = json.load(f)
        logger.info(f"reusing {len(discovered)} discovered trajectories")

    if args.limit:
        per_run: Dict[str, int] = defaultdict(int)
        kept: List[Dict[str, Any]] = []
        for row in discovered:
            if per_run[row["run_name"]] < args.limit:
                kept.append(row)
                per_run[row["run_name"]] += 1
        discovered = kept
        logger.info(f"limited to {len(discovered)} trajectories "
                    f"({args.limit}/run)")

    # Stage 2: final-correctness
    advanced_extractor = None
    has_advanced = any(
        (row.get("question_type") or "").startswith("advanced_")
        for row in discovered
    )
    if has_advanced:
        try:
            from evaluate_advanced import BedrockExtractor
            advanced_extractor = BedrockExtractor(model_id=args.annotator_model)
            logger.info(f"advanced extractor ready: {args.annotator_model}")
        except Exception as e:
            logger.warning(f"advanced extractor unavailable ({e}); will "
                           f"fall back to log is_correct for advanced rows")

    # Resolve --evaluations: default to evaluations/for_paper/ next to this file.
    # Pass --evaluations with no arguments to disable eval-file lookup.
    eval_paths: Optional[List[str]] = args.evaluations
    if eval_paths is None:
        default_eval_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "evaluations", "for_paper"
        )
        if os.path.isdir(default_eval_dir):
            eval_paths = [default_eval_dir]
            logger.info(f"--evaluations not set; auto-discovering from "
                        f"{default_eval_dir}")
        else:
            logger.warning("--evaluations not set and evaluations/for_paper/ "
                           "not found; scoring from logs only")
    elif eval_paths == []:
        # explicit --evaluations with no args → disable
        eval_paths = None
        logger.info("--evaluations passed with no args; eval-file lookup disabled")

    evals_index: Optional[Dict[Tuple[str, str, Any], Dict[str, Any]]] = None
    if eval_paths:
        evals_index = load_evaluations_index(eval_paths)
        if evals_index:
            # Invalidate the on-disk cache so eval-file lookups take effect.
            if os.path.isfile(final_cache_path):
                logger.info(f"evaluations loaded; invalidating "
                            f"{final_cache_path}")
                os.remove(final_cache_path)

    final_correctness = score_final_correctness(
        discovered, final_cache_path, advanced_extractor, evals_index,
    )
    logger.info(f"final-correctness scores cached at {final_cache_path}")

    if args.stage == "score-final":
        return

    # Stage 3: annotate
    if args.stage in ("run", "annotate") and not args.skip_annotate:
        ledger_rows = annotate_trajectories(
            discovered, ledger_path, annotator_cache_dir,
            args.annotator_model, limit=None,
        )
        logger.info(f"ledger written to {ledger_path} "
                    f"({len(ledger_rows)} trajectories)")
    else:
        ledger_rows = []
        if os.path.isfile(ledger_path):
            with open(ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ledger_rows.append(json.loads(line))
            logger.info(f"reusing {len(ledger_rows)} ledger rows")
        elif args.skip_annotate:
            # Build no-annotation rows so the metrics+aggregate stages can run
            # with just final-correctness signals.
            for row in discovered:
                ledger_rows.append({
                    **row,
                    "scientist_answer": "",
                    "n_turns": row.get("n_turns", 0),
                    "turns": [],
                })
            logger.info("skip-annotate: built shell ledger rows "
                        "(no per-turn annotations)")

    if args.stage == "annotate":
        return

    # Stage 4: metrics
    metrics = compute_per_trajectory_metrics(ledger_rows, final_correctness)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"metrics → {metrics_path}")
    if args.stage == "metrics":
        return

    # Stage 5: aggregate
    aggregated = aggregate(metrics)
    with open(aggregate_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    md = render_markdown(aggregated, metrics)
    with open(md_path, "w") as f:
        f.write(md)
    logger.info(f"aggregated → {aggregate_path}")
    logger.info(f"markdown   → {md_path}")


if __name__ == "__main__":
    main()
