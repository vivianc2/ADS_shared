#!/usr/bin/env python3
"""Run an LLM scientist on RPG v5 SCM worlds, with full trace logging.

The scientist sees only the agent-facing catalog (story, observable names,
neutral knob catalog, budget). It issues JSON queries (observational /
interventional / sweep / clamp), reasons over compact statistical summaries,
and finally submits a structured answer that is graded by the computed grader
(utility-optimal AND counterfactual battery).

Backends reuse the tested Bedrock layer in ``framework_code/bedrock_llm.py``.
Default model is Opus 4.8. A ``mock`` backend runs the whole loop with no API
access for sanity checks.

Usage:
    python3 run_agent_v5.py --worlds-dir out_v5_trial \
        --backend bedrock --model us.anthropic.claude-opus-4-8 \
        --out results_v5/trial.json -v
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# make framework_code importable for the Bedrock backend
FRAMEWORK = Path(__file__).resolve().parents[2] / "framework_code"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from sim_v5 import SimV5           # noqa: E402
from oracle import grade_answer_record  # noqa: E402

logger = logging.getLogger("run_agent_v5")

_ACTION_RE = re.compile(r'<action\s+type="(query|answer|give_up)">\s*(.*?)\s*</action>',
                        re.DOTALL | re.IGNORECASE)
_TAG_RE = lambda tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def build_llm(backend: str, model: str, temperature: float, max_new_tokens: int):
    if backend == "mock":
        return MockScientist()
    if backend == "bedrock":
        from bedrock_llm import BedrockLLM
        return BedrockLLM(model_id=model, temperature=temperature, max_new_tokens=max_new_tokens)
    raise ValueError(f"unknown backend {backend!r}")


class MockScientist:
    """Deterministic stand-in that exercises every query mode then answers with
    the *gold* (read from the world) — used to prove the harness + grader wire
    up correctly, NOT to measure capability."""

    def __init__(self):
        self.calls = 0
        self._gold = None
        self._battery = None

    def prime(self, sim: SimV5):
        self.calls = 0
        self._gold = sim.oracle["gold_intervention"]["intervention"]
        self._battery = sim.oracle["counterfactual_battery"]
        self._clampable = sim.clampable

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        obs = re.findall(r"^- ([A-Za-z0-9_]+):", user_prompt, flags=re.MULTILINE)
        if self.calls == 1:
            q = {"mode": "observational_sample", "n_units": 200, "measurements": obs[:4]}
            return _wrap_query(q)
        if self.calls == 2 and self._gold:
            knob = list(self._gold)[0]
            q = {"mode": "sweep", "knob": knob, "measurements": obs[:2]}
            return _wrap_query(q)
        if self.calls == 3 and self._clampable:
            q = {"mode": "clamp", "node": sorted(self._clampable)[0],
                 "levels": [20, 80], "measurements": obs[:2]}
            return _wrap_query(q)
        # answer with gold + battery ground truth (harness/grader wiring check)
        ans = {"recommended_intervention": self._gold or {},
               "structured": {k: self._battery[k] for k in
                              ("true_mechanism_proxy", "confounded_decoys",
                               "knob_sign_predictions")} if self._battery else {},
               "explanation": "mock: replays gold to verify the pipeline"}
        return ("<reasoning>mock closing</reasoning>\n"
                f'<action type="answer">{json.dumps(ans)}</action>\n'
                "<scientist_memory>done</scientist_memory>")


def _wrap_query(q: Dict[str, Any]) -> str:
    return ("<reasoning>mock query</reasoning>\n"
            f'<action type="query">{json.dumps(q)}</action>\n'
            "<scientist_memory>probing</scientist_memory>")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful empirical scientist studying a partially observed process.

You are shown a short story, a list of observable measurements, and a catalog of
neutral action knobs. Some observables are noisy; some are downstream clues of an
UNOBSERVED real-world state that is never listed as a variable. Knob names are
intentionally neutral and do NOT reveal which one targets the true cause — you
must learn that from data, not from labels.

You can run experiments by requesting data samples, then submit ONE structured
answer. Available query modes:

1) observational_sample — baseline, no intervention:
   <action type="query">{"mode":"observational_sample","n_units":200,"measurements":["A","B"]}</action>
2) interventional_sample — set knobs and measure:
   <action type="query">{"mode":"interventional_sample","intervention":{"KnobA":50,"KnobB":"on"},"n_units":200,"measurements":["Outcome","ProxyX"]}</action>
3) sweep — trace a dose-response curve for one knob:
   <action type="query">{"mode":"sweep","knob":"KnobA","grid":[0,25,50,75,100],"measurements":["Outcome","ProxyX"]}</action>
4) clamp — force a clampable observable to fixed levels to break a confound:
   <action type="query">{"mode":"clamp","node":"SomeReading","levels":[20,80],"measurements":["Outcome"]}</action>

Final answer:
   <action type="answer">{
     "recommended_intervention": {"KnobA": 66},
     "structured": {
       "true_mechanism_proxy": "<the observable that is a genuine downstream clue of the hidden cause>",
       "confounded_decoys": ["<observables that correlate with the outcome but have no causal effect on it>"],
       "knob_sign_predictions": {"KnobA":"+","KnobB":"-","KnobC":"0"}
     },
     "explanation": "ordinary-language hidden cause + evidence + decisive test"
   }</action>

knob_sign_predictions: for each knob, predict the sign of its effect on the
outcome IN THE DIRECTION THAT IS BETTER. Use "+" if increasing/enabling the knob
improves the outcome, "-" if it worsens it, "0" if no real effect. A knob that
only changes the *measured* reading without changing the true state counts as
"0" (do not be fooled by a symptom-masking knob).

Good strategy:
- Start observationally; note which observables track the outcome.
- Correlation is not cause: a strong observational correlation can be a confound.
  Use clamp/interventional tests to check whether forcing a variable actually
  moves the outcome.
- Trace dose-response with sweep; the best dose may be interior (not the max).
- Distinguish a true mechanism proxy (moves when you fix the cause) from a
  symptom-masking knob (moves the reading but not the true state).
- Spend the budget: a confident answer usually needs at least one decisive
  interventional/clamp test, not just observation.

Output EXACTLY these three blocks in order:
<reasoning>Concise analysis and why the next action is appropriate.</reasoning>
<action type="query|answer|give_up">JSON query, JSON answer, or brief reason.</action>
<scientist_memory>
Tested:
Known:
Uncertain:
Next:
</scientist_memory>"""


def catalog_text(pub: Dict[str, Any]) -> str:
    lines = ["OBSERVABLE MEASUREMENTS"]
    for o in pub["observed_variables"]:
        lines.append(f"- {o['name']}: {o['description']}")
    lines.append("\nAVAILABLE ACTION KNOBS")
    for k in pub["action_variables"]:
        if k["value_type"] == "continuous":
            lines.append(f"- {k['name']}: continuous in {k['range']}")
        else:
            lines.append(f"- {k['name']}: values={k['values']}")
    lines.append(f"\nclampable measurements: {pub.get('clampable_measurements', [])}")
    lines.append(f"outcome: {pub['outcome_name']} ({pub['outcome_direction']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def summarize_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, default=str)


def run_world(sim: SimV5, llm, max_turns: int, verbose: bool) -> Dict[str, Any]:
    pub = sim.public_world()
    cat = catalog_text(pub)
    history: List[Dict[str, Any]] = []
    memory = "(empty)"
    queries_used = 0
    turns = []

    if isinstance(llm, MockScientist):
        llm.prime(sim)

    final_answer = None
    grade = None
    for turn in range(max_turns):
        remaining = sim.max_queries - queries_used
        latest = history[-1]["result"] if history else "(none yet)"
        user = f"""QUESTION
{pub['question']}

BUDGET
queries_used={queries_used}/{sim.max_queries}; {remaining} left
max_units_per_query={sim.max_units}

STORY
{pub['story']}

{cat}

ALLOWED QUERY MODES
{', '.join(pub['allowed_query_modes'])}

LATEST RESULT
{summarize_result(latest) if isinstance(latest, dict) else latest}

YOUR MEMORY
{memory}
"""
        t0 = time.time()
        raw = llm.generate(SYSTEM_PROMPT, user, max_new_tokens=2500)
        dt = round(time.time() - t0, 2)

        reasoning = _extract(raw, "reasoning")
        memory = _extract(raw, "scientist_memory") or memory
        m = _ACTION_RE.search(raw)
        turn_rec: Dict[str, Any] = {"turn": turn, "latency_s": dt, "reasoning": reasoning,
                                    "raw": raw, "memory": memory}
        if not m:
            turn_rec["error"] = "no <action> block parsed"
            turns.append(turn_rec)
            if verbose:
                logger.warning("turn %d: no action parsed", turn)
            continue

        atype, payload = m.group(1).lower(), m.group(2).strip()
        turn_rec["action_type"] = atype

        if atype == "answer":
            try:
                ans = json.loads(payload)
            except Exception as e:
                turn_rec["error"] = f"answer JSON parse failed: {e}"
                turns.append(turn_rec)
                continue
            final_answer = ans
            grade = grade_answer_record(sim, ans)
            turn_rec["answer"] = ans
            turn_rec["grade"] = grade
            turns.append(turn_rec)
            if verbose:
                logger.info("turn %d ANSWER accepted=%s gap=%.2f battery=%.2f",
                            turn, grade["accepted"], grade["utility_gap"], grade["battery_fraction"])
            break

        if atype == "give_up":
            turn_rec["give_up_reason"] = payload
            turns.append(turn_rec)
            break

        # query
        try:
            q = json.loads(payload)
            result = sim.run(q, call_idx=queries_used)
            queries_used += 1
            turn_rec["query"] = q
            turn_rec["result"] = result
            history.append({"query": q, "result": result})
            if verbose:
                logger.info("turn %d query mode=%s (used %d/%d)",
                            turn, q.get("mode"), queries_used, sim.max_queries)
        except Exception as e:
            turn_rec["query"] = payload
            turn_rec["result"] = f"QUERY ERROR: {e}"
            history.append({"query": payload, "result": f"QUERY ERROR: {e}"})
            if verbose:
                logger.warning("turn %d query error: %s", turn, e)
        turns.append(turn_rec)

    return {
        "world_id": sim.record["world_id"],
        "domain": sim.record.get("domain"),
        "queries_used": queries_used,
        "answered": final_answer is not None,
        "final_answer": final_answer,
        "grade": grade,
        "turns": turns,
    }


def _extract(text: str, tag: str) -> str:
    m = _TAG_RE(tag).search(text)
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an LLM scientist on RPG v5 worlds.")
    ap.add_argument("--worlds-dir", default=None)
    ap.add_argument("--world-json", default=None)
    ap.add_argument("--backend", choices=["bedrock", "mock"], default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.world_json:
        paths = [args.world_json]
    elif args.worlds_dir:
        paths = sorted(glob.glob(os.path.join(args.worlds_dir, "world_*.json")))
    else:
        ap.error("provide --worlds-dir or --world-json")

    llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    results = []
    t_start = time.time()
    for path in paths:
        sim = SimV5.from_json(path)
        logger.info("=== running %s (%s) ===", sim.record["world_id"], args.model)
        res = run_world(sim, llm, args.max_turns, args.verbose)
        res["world_path"] = path
        results.append(res)

    n = len(results)
    accepted = sum(1 for r in results if r.get("grade") and r["grade"]["accepted"])
    part_a = sum(1 for r in results if r.get("grade") and r["grade"]["part_a_utility_ok"])
    part_b = sum(1 for r in results if r.get("grade") and r["grade"]["part_b_battery_ok"])
    avg_q = round(sum(r["queries_used"] for r in results) / n, 2) if n else 0
    payload = {
        "model": args.model, "backend": args.backend, "n_worlds": n,
        "accepted": accepted, "accuracy": round(accepted / n, 3) if n else 0,
        "part_a_utility_ok": part_a, "part_b_battery_ok": part_b,
        "avg_queries": avg_q, "wall_seconds": round(time.time() - t_start, 1),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n{accepted}/{n} accepted (acc={payload['accuracy']}); "
          f"partA(utility)={part_a}/{n} partB(battery)={part_b}/{n}; "
          f"avg_queries={avg_q}; {payload['wall_seconds']}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
