#!/usr/bin/env python3
"""Run an LLM scientist on RPG v6 large open-scenario worlds.

The agent sees ONLY a detailed neutral prose scenario, the outcome name, and its
budget -- no variable list, no action menu. It proposes measurements and
interventions in free text; the server resolver maps them to hidden
variables/actuators, echoing its interpretation (so a resolution miss is visible
and correctable) and rejecting out-of-world requests with a plausible reason.

Actions (one per turn):
  <action type="measure">{"requests":["product titer","broth cloudiness"]}</action>
  <action type="intervene">{"actions":[{"request":"add a chelating agent","value":66},
                                        {"request":"reduce feed water flow","value":0}],
                             "measure":["product titer","broth cloudiness"]}</action>
  <action type="answer">{"recommended_intervention_text":[{"request":"add chelator","value":66}],
                          "structured":{...}, "explanation":"..."}</action>

Backends reuse framework_code/bedrock_llm.py. Default model: Opus 4.8. A mock
backend runs the loop with no API. Full per-turn trace is logged.
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

FRAMEWORK = Path(__file__).resolve().parents[2] / "framework_code"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from worlds_v6 import ALL_WORLDS_V6      # noqa: E402
from sim_v6 import SimV6                 # noqa: E402

logger = logging.getLogger("run_agent_v6")

_ACTION_RE = re.compile(r'<action\s+type="(measure|intervene|code|answer|give_up)">\s*(.*?)\s*</action>',
                        re.DOTALL | re.IGNORECASE)


def _tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _norm_req(req: str) -> str:
    """Coarse normalization so repeated rejections of the same idea collapse to
    one key (drives the 'stop asking for what isn't here' directive)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (req or "").lower())).strip()[:60]


SYSTEM_PROMPT = """You are a senior process scientist called in to diagnose a failing industrial system.

You are given a detailed situation report in plain language. You are NOT given a
list of variables or a menu of actions. Like a real scientist, you must decide
what is worth measuring and what interventions to try, and request them in your
own words. A back-office system will interpret each request, tell you how it
understood it, and either return data or explain why it can't be done (e.g. "no
assay for that", "no actuator connected", "not present on this line").

You can do four things (ONE action per turn):

1) Measure quantities (observational, or while an intervention is held). Each
   measurement returns raw per-unit rows (saved to a CSV file you can analyze
   with code) plus a quick summary (means, SDs, correlations):
   <action type="measure">{"requests":["product titer","how cloudy the broth is","dissolved oxygen"]}</action>

2) Intervene through available controls/additives, alone or in COMBINATION, and
   measure the result. Give a level/dose when relevant (a number, or words like
   low/moderate/high):
   <action type="intervene">{"actions":[{"request":"add a metal-chelating additive","value":60},
                                          {"request":"reduce the feed-water flow","value":0}],
                              "measure":["product titer","broth cloudiness"]}</action>

3) Run analysis code over the raw data you have collected. Each measure/intervene
   writes a CSV of per-unit rows, exposed to your code as a variable named
   experiment_<id>_csv holding the file path. You have pd (pandas), np (numpy),
   and stats (scipy.stats). print() what you want to see. Code does NOT cost an
   experiment from your budget -- use it to fit dose-response curves, run
   regressions, test whether a correlate is a confound, check for interactions,
   etc. Example:
   <action type="code">
import pandas as pd
from scipy import stats
df = pd.read_csv(experiment_1_csv)
print(df.describe())
print("corr:", df.corr(numeric_only=True).round(3))
slope,inter,r,p,se = stats.linregress(df["SomeSignal"], df["ProductTiter"])
print(f"slope={slope:.3f} r={r:.3f} p={p:.2e}")
   </action>

4) Submit your final conclusion:
   <action type="answer">{
     "recommended_intervention_text":[{"request":"add a chelating additive","value":66}],
     "structured":{
        "true_mechanism_proxy":"<the measurable signal that is a genuine downstream marker of the hidden cause>",
        "confounded_decoys":["<measured signals that correlate with the outcome but do NOT cause it>"],
        "actuator_sign_predictions":{"add a chelating additive":"+","reduce feed-water flow":"+","add a stabilizer":"0"}
     },
     "explanation":"plain-language hidden cause, your evidence, the decisive test, and why"
   }</action>

Notes:
- You have a limited experiment budget. You cannot try everything, so reason
  about WHERE to look from the situation report before spending a query.
- A strong correlation is not proof of cause: something that moves with the
  outcome may be a bystander driven by a common factor. Test it (e.g. hold it
  fixed and see if the outcome follows).
- Observation and code alone CANNOT establish causation or tell you what fixes
  the problem. You MUST run interventions (apply controls/additives and see what
  moves the outcome). An answer with no supporting intervention is weak.
- If a request is rejected because that measurement/actuator does not exist, do
  NOT keep asking for variants of it — that thing is not in this system. Revise
  your hypothesis to use only what IS available. Your recommended fix must use an
  actuator that actually exists here.
- Only state findings you actually observed in returned data. Do not describe
  experiments you did not run or results you did not receive.
- The best fix may be a COMBINATION, and the best dose may be in the middle
  (too much can hurt). A control that changes the *reading* but not the true
  state is a trap.
- For actuator_sign_predictions use the request phrasing you used; sign is the
  effect on the outcome in the better direction (+ improves, - worsens, 0 none).

A good loop is: design an experiment -> collect data -> analyze it with code ->
decide the next experiment. Don't just eyeball the summary means; when a number
matters (the best dose, whether a correlate is causal), compute it.

Output EXACTLY these three blocks each turn:
<reasoning>your scientific thinking and why this action</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON or code</action>
<memory>
Hypotheses:
Ruled out:
Confirmed:
Next:
</memory>"""


class MockScientistV6:
    """Deterministic stand-in: probes a couple of things then answers with gold
    + battery ground truth. Verifies the harness/resolver/grader wiring only."""

    def __init__(self):
        self.calls = 0
        self.sim = None

    def prime(self, sim: SimV6):
        self.calls = 0
        self.sim = sim

    def _alias(self, actuator_id: str) -> str:
        """First alias of an actuator (what the resolver maps back to it)."""
        return self.sim.scm.actuators[actuator_id].get("aliases", [actuator_id])[0]

    def generate(self, sysp: str, userp: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        outcome_alias = self.sim.scm.variables[self.sim.scm.outcome].get("aliases", [self.sim.scm.outcome])[0]
        proxy_alias = self.sim.scm.variables[self.sim.battery["true_mechanism_proxy"]].get("aliases", [self.sim.battery["true_mechanism_proxy"]])[0]
        if self.calls == 1:
            a = {"requests": [outcome_alias, proxy_alias]}
            return f'<reasoning>baseline</reasoning>\n<action type="measure">{json.dumps(a)}</action>\n<memory>Next: analyze</memory>'
        if self.calls == 2:
            # exercise the code tool over experiment_1_csv (smoke test path)
            code = ("import pandas as pd\n"
                    "df = pd.read_csv(experiment_1_csv)\n"
                    "print('rows', len(df), 'cols', list(df.columns))\n"
                    "print(df.corr(numeric_only=True).round(3))\n")
            return f'<reasoning>analyze raw data</reasoning>\n<action type="code">\n{code}\n</action>\n<memory>Next: test</memory>'
        # Build gold-based answer generically from the sim (works for ANY world).
        gold_iv = self.sim.gold["intervention"]
        rec = [{"request": self._alias(aid), "value": val} for aid, val in gold_iv.items()]
        bat = self.sim.battery
        if self.calls == 3 and rec:
            a = {"actions": [rec[0]], "measure": [outcome_alias, proxy_alias]}
            return f'<reasoning>test targeted lever</reasoning>\n<action type="intervene">{json.dumps(a)}</action>\n<memory>Next: answer</memory>'
        # sign predictions phrased by actuator alias, taken from the battery
        signs = {self._alias(aid): s for aid, s in bat["actuator_sign_predictions"].items()}
        decoy_aliases = [self.sim.scm.variables[d].get("aliases", [d])[0] for d in bat["confounded_decoys"]]
        ans = {"recommended_intervention_text": rec,
               "structured": {"true_mechanism_proxy": proxy_alias,
                              "confounded_decoys": decoy_aliases,
                              "actuator_sign_predictions": signs},
               "explanation": self.sim.gt["latent_plain_name"]}
        return f'<reasoning>done</reasoning>\n<action type="answer">{json.dumps(ans)}</action>\n<memory>done</memory>'


def build_llm(backend, model, temperature, max_new_tokens):
    if backend == "mock":
        return MockScientistV6()
    if backend == "bedrock":
        from bedrock_llm import BedrockLLM
        return BedrockLLM(model_id=model, temperature=temperature, max_new_tokens=max_new_tokens)
    raise ValueError(backend)


def _resolve_answer_intervention(sim: SimV6, rec_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map the agent's free-text recommended actions to actuator ids for grading."""
    iv = {}
    echoes = []
    for item in rec_text or []:
        r = sim.resolver.resolve_intervene(item.get("request", ""), item.get("value"))
        echoes.append({"request": item.get("request", ""), **r.to_dict()})
        if r.ok and r.target_id:
            iv[r.target_id] = r.value
    return iv, echoes


def run_world(sim: SimV6, llm, max_turns: int, budget: int, verbose: bool) -> Dict[str, Any]:
    pub = sim.public()
    if isinstance(llm, MockScientistV6):
        llm.prime(sim)
    history: List[str] = []
    memory = "(empty)"
    used = 0
    n_interventions = 0
    turns = []
    final = None
    grade = None
    csv_map: Dict[str, str] = {}     # experiment_<id>_csv -> path (for the code tool)
    carried_vars: Dict[str, Any] = {}  # picklable vars carried between code turns
    rejected_counts: Dict[str, int] = {}  # normalized rejected request -> times rejected
    forced_answer = False

    for turn in range(max_turns):
        remaining = budget - used
        turns_left = max_turns - turn
        latest = history[-1] if history else "(none yet)"
        files_line = ("available data files for code: " + ", ".join(sorted(csv_map))) if csv_map else \
                     "available data files for code: (none yet — run a measure/intervene first)"

        # --- directives that keep failures attributable to reasoning, not harness ---
        directives = []
        # (1) must-answer when budget or turns are nearly gone
        if remaining <= 0 or turns_left <= 2:
            directives.append(
                "STOP EXPLORING. You are out of budget/turns. Submit your best "
                "<action type=\"answer\"> NOW using the evidence you have. Do not "
                "measure or run code this turn.")
            forced_answer = True
        # (2) nudge toward experimentation if the agent keeps observing/analyzing
        if n_interventions == 0 and used >= 3:
            directives.append(
                "You have not run any INTERVENTION yet. Observation and code alone "
                "cannot establish causation or find the fix — apply a control/"
                "additive with <action type=\"intervene\"> to test a cause and to "
                "find what actually improves the outcome.")
        # (3) stop the repeated-rejection loop
        stuck = [req for req, c in rejected_counts.items() if c >= 3]
        if stuck:
            directives.append(
                "Several of your requests have been rejected repeatedly because "
                "those things are NOT part of this system (no such measurement or "
                "actuator exists): " + "; ".join(sorted(stuck)[:5]) + ". Stop "
                "asking for them and reconsider your hypothesis using only what IS "
                "available.")
        directive_block = ("\nDIRECTIVES\n" + "\n".join(f"- {d}" for d in directives) + "\n") if directives else ""

        user = f"""SITUATION REPORT
{pub['scenario']}

OUTCOME OF INTEREST: {pub['outcome_name']} ({pub['outcome_direction']})

BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{files_line}
{directive_block}
RESULT OF YOUR LAST ACTION
{latest}

YOUR MEMORY
{memory}
"""
        t0 = time.time()
        raw = llm.generate(SYSTEM_PROMPT, user, max_new_tokens=2500)
        dt = round(time.time() - t0, 2)
        memory = _tag(raw, "memory") or memory
        reasoning = _tag(raw, "reasoning")
        m = _ACTION_RE.search(raw)
        rec: Dict[str, Any] = {"turn": turn, "latency_s": dt, "reasoning": reasoning, "memory": memory, "raw": raw}
        if not m:
            rec["error"] = "no action parsed"
            turns.append(rec)
            continue
        atype, payload = m.group(1).lower(), m.group(2).strip()
        rec["action_type"] = atype
        try:
            data = json.loads(payload) if payload and payload[0] in "{[" else {}
        except Exception as e:
            rec["error"] = f"json parse: {e}"
            data = {}

        if atype == "answer":
            iv, echoes = _resolve_answer_intervention(sim, data.get("recommended_intervention_text", []))
            answer = {"recommended_intervention": iv,
                      "structured": _translate_structured(sim, data.get("structured", {})),
                      "explanation": data.get("explanation", "")}
            grade = sim.grade(answer)
            rec["answer_raw"] = data
            rec["resolved_intervention"] = iv
            rec["answer_resolution_echo"] = echoes
            rec["grade"] = grade
            rec["artifact_check"] = _artifact_check(sim, data, iv, echoes, answer["structured"], grade)
            turns.append(rec)
            if verbose:
                logger.info("turn %d ANSWER accepted=%s A=%s B=%.2f gap=%.2f",
                            turn, grade["accepted"], grade["part_a_utility_ok"],
                            grade["battery_fraction"], grade["utility_gap"])
            break

        if atype == "give_up":
            rec["give_up_reason"] = payload
            turns.append(rec)
            break

        if atype == "code":
            from sandbox import run_code
            out, new_vars = run_code(payload, csv_map, carried_vars)
            carried_vars.update(new_vars)
            rec["code"] = payload
            rec["code_output"] = out
            # code turns do NOT cost budget; feed output back as the latest result
            history.append("CODE OUTPUT:\n" + out)
            if verbose:
                logger.info("turn %d CODE (%d chars out, %d files)", turn, len(out), len(csv_map))
            turns.append(rec)
            continue

        if atype == "measure":
            result = sim.measure(data.get("requests", []))
            used += 1
            if result.get("raw_csv"):
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            for e in result["resolutions"]:
                if not e["ok"]:
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            history.append(json.dumps(result, default=str))
            if verbose:
                ok = [e["target_id"] for e in result["resolutions"] if e["ok"]]
                logger.info("turn %d MEASURE -> %s (used %d/%d)", turn, ok, used, budget)
        elif atype == "intervene":
            result = sim.intervene(data.get("actions", []), data.get("measure", []))
            used += 1
            if result.get("applied_intervention"):
                n_interventions += 1
            if result.get("raw_csv"):
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            for e in result["action_resolutions"]:
                if not e["ok"]:
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            history.append(json.dumps(result, default=str))
            if verbose:
                logger.info("turn %d INTERVENE %s (used %d/%d, interventions=%d)", turn,
                            result["applied_intervention"], used, budget, n_interventions)
        turns.append(rec)

    # loop ended without an answer (turn cap) -> synthesize a best-effort answer
    # from the last memory so the run is still gradable (no silent non-answers).
    if grade is None:
        synth = {"recommended_intervention": {}, "structured": {}, "explanation": memory}
        grade = sim.grade(synth)
        grade["forced_no_answer"] = True
        turns.append({"turn": "post", "action_type": "forced_answer",
                      "note": "hit turn cap without answering; graded empty answer from memory",
                      "grade": grade})

    return {"world_id": sim.world["world_id"], "domain": sim.world["domain"],
            "queries_used": used, "interventions_run": n_interventions,
            "answered": final is not None or grade is not None,
            "hit_turn_cap": forced_answer and grade.get("forced_no_answer", False) if grade else False,
            "grade": grade, "gold": {"intervention": sim.gold["intervention"],
                                     "expected_utility": round(sim.gold["expected_utility"], 2)},
            "turns": turns}


def _artifact_check(sim: SimV6, answer_raw: Dict[str, Any], iv: Dict[str, Any],
                    echoes: List[Dict[str, Any]], structured: Dict[str, Any],
                    grade: Dict[str, Any]) -> Dict[str, Any]:
    """Flag runs whose failure may be a HARNESS/RESOLUTION artifact rather than a
    reasoning failure — so a false 0/N can never be silently read as difficulty.

    Signals (any true => 'suspect', worth a human glance):
    - the agent proposed a recommended intervention but NONE of its actions
      resolved to an actuator (pure resolution failure);
    - a structured proxy/decoy string failed to resolve to a known variable;
    - the run FAILED but part A passed (good intervention) while part B failed —
      the classic 'correct-but-miscredited' pattern that bit v4 and batch1.
    """
    reasons = []
    rec_text = answer_raw.get("recommended_intervention_text", []) or []
    if rec_text and not iv:
        reasons.append("recommended_intervention proposed but nothing resolved to an actuator")
    unresolved = [e["request"] for e in echoes if not e.get("ok")]
    if rec_text and len(unresolved) == len(rec_text) and rec_text:
        reasons.append(f"all {len(rec_text)} recommended action(s) rejected by resolver: {unresolved[:3]}")
    st = answer_raw.get("structured", {}) or {}
    # did the named proxy fail to resolve to any known measurable?
    proxy_txt = st.get("true_mechanism_proxy")
    if proxy_txt:
        pr = sim.resolver.resolve_measure(proxy_txt)
        if not pr.ok:
            reasons.append(f"named true_mechanism_proxy did not resolve: {proxy_txt!r}")
    if (not grade.get("accepted")) and grade.get("part_a_utility_ok") and not grade.get("part_b_battery_ok"):
        reasons.append("part A (utility) passed but part B (battery) failed — verify battery credited a correct answer")
    return {"suspect": len(reasons) > 0, "reasons": reasons}


def _translate_structured(sim: SimV6, st: Dict[str, Any]) -> Dict[str, Any]:
    """The agent names the mechanism proxy / decoys by free text and predicts
    signs keyed by its own action phrasings. Map both to canonical ids so the
    grader (which knows canonical ids) can score them."""
    out = {}
    # proxy + decoys: resolve as measurements
    pm = sim.resolver.resolve_measure(st.get("true_mechanism_proxy", "")) if st.get("true_mechanism_proxy") else None
    out["true_mechanism_proxy"] = pm.target_id if (pm and pm.ok) else st.get("true_mechanism_proxy")
    decoys = []
    for d in st.get("confounded_decoys", []) or []:
        dm = sim.resolver.resolve_measure(d)
        decoys.append(dm.target_id if dm.ok else d)
    out["confounded_decoys"] = decoys
    # actuator signs: keys are free-text requests -> map to actuator ids.
    # The battery defines a sign as the effect of INCREASING the actuator's
    # setting. If the agent phrased the request with a decreasing verb
    # ("reduce/lower/cut feed flow"), its stated sign is for the decrease, so we
    # flip it back to the increase convention before scoring. This prevents a
    # correct answer ("reducing flow helps" = "+") from being marked wrong
    # against the stored increase-sign ("-").
    _DECREASE = ("reduce", "lower", "cut", "decrease", "less", "minimi", "turn down", "shut off", "stop")
    _FLIP = {"+": "-", "-": "+", "0": "0"}
    signs = {}
    for req, sign in (st.get("actuator_sign_predictions", {}) or {}).items():
        ar = sim.resolver.resolve_intervene(req, None)
        key = ar.target_id if (ar.ok and ar.target_id) else req
        rl = req.lower()
        if any(w in rl for w in _DECREASE) and sign in _FLIP:
            sign = _FLIP[sign]
        signs[key] = sign
    out["actuator_sign_predictions"] = signs
    return out


def load_world_file(path: str) -> tuple:
    """Load a generated world JSON (schema rpg_scm_v6). Returns (world_dict,
    precomputed) where precomputed carries the stored gold + battery so SimV6
    does not recompute (and stays identical to what was audited at gen time)."""
    from engine import WorldSCM
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    world = {"world_id": rec["world_id"], "domain": rec["domain"],
             "scenario": rec["scenario"], "scm": WorldSCM.from_dict(rec["scm"]),
             "ground_truth": rec["ground_truth"]}
    pre = {"gold": rec["oracle"]["gold"], "battery": rec["oracle"]["counterfactual_battery"]}
    return world, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="bioreactor_titer_loss_v6",
                    help="built-in template name (ignored if --world-file given)")
    ap.add_argument("--world-file", default=None,
                    help="path to a generated world JSON (schema rpg_scm_v6)")
    ap.add_argument("--backend", choices=["bedrock", "mock"], default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=32,
                    help="total turns; code turns count here but not against --budget")
    ap.add_argument("--no-resolver-llm", action="store_true",
                    help="disable the LLM resolver fallback (on by default for bedrock runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    # LLM resolver fallback is ON by default (bedrock only); --no-resolver-llm opts out.
    resolver_llm = llm if (args.backend == "bedrock" and not args.no_resolver_llm) else None

    # per-run data dir for raw experiment CSVs (the code tool reads these)
    data_dir = str(Path(args.out).parent / (Path(args.out).stem + "_data"))
    if args.world_file:
        world, pre = load_world_file(args.world_file)
        logger.info("=== loaded world %s from %s ===", world["world_id"], args.world_file)
        sim = SimV6(world, resolver_llm=resolver_llm, data_dir=data_dir, precomputed=pre)
    else:
        world = ALL_WORLDS_V6[args.world]()
        logger.info("=== building + calibrating world %s ===", args.world)
        sim = SimV6(world, resolver_llm=resolver_llm, data_dir=data_dir)
    logger.info("gold=%s util=%.1f | raw data -> %s", sim.gold["intervention"],
                sim.gold["expected_utility"], data_dir)

    t0 = time.time()
    res = run_world(sim, llm, args.max_turns, args.budget, args.verbose)
    res["wall_seconds"] = round(time.time() - t0, 1)
    res["model"] = args.model

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    g = res["grade"]
    if g:
        tag = " [FORCED: hit turn cap, no answer submitted]" if g.get("forced_no_answer") else ""
        print(f"\naccepted={g['accepted']} | partA(utility)={g['part_a_utility_ok']} "
              f"partB(battery)={g['battery_fraction']:.2f} | gap={g['utility_gap']:.2f} | "
              f"queries={res['queries_used']} interventions={res['interventions_run']} | "
              f"{res['wall_seconds']}s{tag}")
        print(f"gold={res['gold']['intervention']}  agent={g['recommended_intervention']}")
    else:
        print(f"\nno answer submitted; queries={res['queries_used']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
