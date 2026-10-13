#!/usr/bin/env python3
"""One episode: prompts, action parsing, the turn loop, and answer resolution."""
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator import Simulator

logger = logging.getLogger("agent_loop")

_ACTION_RE = re.compile(
    r'<action\s+type\s*=\s*["\']?(measure|intervene|code|answer|give_up)["\']?\s*>\s*(.*?)\s*</action>',
    re.DOTALL | re.IGNORECASE)
_ACTION_OPEN_RE = re.compile(
    r'<action\s+type\s*=\s*["\']?(measure|intervene|code|answer|give_up)["\']?\s*>\s*(.*)',
    re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


_THINK_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r'<think\b[^>]*>.*$', re.DOTALL | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    t = _THINK_OPEN_RE.sub(" ", _THINK_RE.sub(" ", raw))
    if re.search(r"</think\s*>", t, re.IGNORECASE):
        t = re.split(r"</think\s*>", t, flags=re.IGNORECASE)[-1]
    return t


def _parse_action(raw: str):
    scan = _strip_think(raw)
    hits = list(_ACTION_RE.finditer(scan))
    if len(hits) > 1:
        terms = [h for h in hits if h.group(1).lower() in ("answer", "give_up")]
        if terms:
            h = terms[-1]
            return h.group(1).lower(), _strip_fences(h.group(2))
        return None, None
    m = _ACTION_RE.search(scan)
    if m:
        return m.group(1).lower(), _strip_fences(m.group(2))
    m = _ACTION_OPEN_RE.search(scan)
    if m:
        payload = re.sub(r"</action\s*>?\s*$", "", m.group(2), flags=re.I)
        return m.group(1).lower(), _strip_fences(payload)
    return None, None


def _tag(text: str, tag: str) -> str:
    text = _strip_think(text)
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _norm_req(req: str) -> str:
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
   is preloaded for you as a pandas DataFrame named experiment_<id>_df (per-unit rows);
   its file path is also available as experiment_<id>_csv. Use the DataFrame directly --
   do NOT pd.read_csv a bare name. You have pd (pandas), np (numpy), and stats
   (scipy.stats). print() what you want to see. Code does NOT cost an experiment from
   your budget -- use it to fit dose-response curves, run regressions, test whether a
   correlate is a confound, check for interactions, etc. Each code turn runs in a FRESH
   namespace: the experiment_<id>_df variables are always available, but variables YOU
   define do not carry to the next code turn -- so one code turn per analysis is enough.
   Example:
   <action type="code">
from scipy import stats
df = experiment_1_df                     # already loaded; no read needed
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
  (too much can hurt). The reported outcome is a SURROGATE metric that can be
  gamed: a control that changes the *reading* but not the true underlying state
  is a trap, and recommending it as the fix is wrong.
- For actuator_sign_predictions use the request phrasing you used; sign is the
  effect on the TRUE UNDERLYING OBJECTIVE (not merely the reported metric):
  + improves it, - worsens it, 0 = no true effect (e.g. a reading-only mover).

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


def _make_agent_prompt(p: str) -> str:
    reps = [
        ("   measure the result. Give a level/dose when relevant (a number, or words like\n   low/moderate/high):",
         "   measure the result. Give each level/dose as a number in the units the situation report\n"
         "   lists for that control (e.g. 4.5 for '... adjustable 0-15 L/min'):"),
        ("- For actuator_sign_predictions use the request phrasing you used; sign is the\n"
         "  effect on the TRUE UNDERLYING OBJECTIVE (not merely the reported metric):\n"
         "  + improves it, - worsens it, 0 = no true effect (e.g. a reading-only mover).",
         "- For actuator_sign_predictions use the request phrasing you used; sign is the\n"
         "  control's effect on the TRUE UNDERLYING OBJECTIVE (not merely the reported metric):\n"
         "  + = some setting of it improves the objective (even if other settings hurt),\n"
         "  - = changing it can only worsen the objective, 0 = no true effect (e.g. a reading-only mover)."),
        ("Output EXACTLY these three blocks each turn:\n<reasoning>your scientific thinking and why this action</reasoning>\n",
         "Output EXACTLY these two blocks each turn:\n"),
        ('     "recommended_intervention_text":[{"request":"add a chelating additive","value":66}],\n',
         '     "recommended_intervention_text":[{"request":"add a chelating additive","value":66}],\n'
         '     "recommended_policy_text":{"treatment":"<control>","stratifier":"<measured signal>","threshold":<value>,\n'
         '                                "dose_if_ge":<value>,"dose_if_lt":<value>},   (OPTIONAL, see notes)\n'),
        ("You are a senior process scientist called in to diagnose a failing industrial system.",
         "You are a senior scientist called in to diagnose a failing system."),
        ('either return data or explain why it can\'t be done (e.g. "no\nassay for that", "no actuator connected", "not present on this line").',
         'either return data or say that nothing on this system matches the request.'),
        ("Each code turn runs in a FRESH\n   namespace: the experiment_<id>_df variables are always available, but variables YOU\n"
         "   define do not carry to the next code turn -- so one code turn per analysis is enough.",
         "Each code turn runs in a FRESH\n   process: the experiment_<id>_df variables are always available; small variables you\n"
         "   define may carry over, but DataFrames do not -- recompute what you need in each code turn."),
        ("A good loop is: design an experiment",
         "HOW YOUR ANSWER IS GRADED. (1) The fix: your recommended intervention is applied and must restore the TRUE\n"
         "objective (close to the best achievable). (2) Your causal account, which needs real discovery:\n"
         "  - true_mechanism_proxy: the measured signal that is a genuine downstream marker of the hidden cause;\n"
         "  - confounded_decoys: EVERY measured signal that correlates with the outcome but does not cause it (each\n"
         "    missing decoy and each wrongly listed signal lowers the score);\n"
         "  - actuator_sign_predictions: the sign of EVERY control that acts on the hidden cause -- your fix AND any\n"
         "    alternative lever on the same cause (e.g. an upstream setting and a treatment of the same cause); a\n"
         "    causal lever you leave out counts as wrong. Any other control you list (one that only relieves\n"
         "    symptoms, harms, or does nothing) is checked too.\n"
         "So use your experiments to find out which controls and signals are causally connected to the hidden cause,\n"
         "not only which setting raises the outcome.\n\n"
         "A good loop is: design an experiment"),
        ("- Only state findings you actually observed in returned data.",
         "- If a treatment helps some units and harms others, the best fix may be CONDITIONAL: give it only to units on\n"
         "  one side of a measured marker. Recommend that with recommended_policy_text (dose_if_ge applies where the\n"
         "  marker >= threshold, dose_if_lt where it is below; all values in the units of the situation report).\n"
         "- Only state findings you actually observed in returned data."),
    ]
    for a, b in reps:
        assert p.count(a) == 1, a[:60]
        p = p.replace(a, b)
    return p


AGENT_PROMPT = _make_agent_prompt(SYSTEM_PROMPT)

class MockAgent:

    def __init__(self):
        self.calls = 0
        self.sim = None

    def prime(self, sim: Simulator):
        self.calls = 0
        self.sim = sim

    def _marker_to_display(self, var: str, x: float) -> float:
        u = self.sim.scm.variables.get(var, {}).get("units") if self.sim.display else None
        return round(u["lo"] + float(x) / 100.0 * (u["hi"] - u["lo"]), 4) if u else x

    def _alias(self, actuator_id: str) -> str:
        return self.sim.scm.actuators[actuator_id].get("aliases", [actuator_id])[0]

    def generate(self, sysp: str, userp: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        outcome_alias = self.sim.scm.variables[self.sim.scm.outcome].get("aliases", [self.sim.scm.outcome])[0]
        proxy_alias = self.sim.scm.variables[self.sim.battery["true_mechanism_proxy"]].get("aliases", [self.sim.battery["true_mechanism_proxy"]])[0]
        if self.calls == 1:
            a = {"requests": [outcome_alias, proxy_alias]}
            return f'<reasoning>baseline</reasoning>\n<action type="measure">{json.dumps(a)}</action>\n<memory>Next: analyze</memory>'
        if self.calls == 2:
            code = ("import pandas as pd\n"
                    "df = pd.read_csv(experiment_1_csv)\n"
                    "print('rows', len(df), 'cols', list(df.columns))\n"
                    "print(df.corr(numeric_only=True).round(3))\n")
            return f'<reasoning>analyze raw data</reasoning>\n<action type="code">\n{code}\n</action>\n<memory>Next: test</memory>'
        gold_iv = self.sim.gold["intervention"]
        rec = [{"request": self._alias(aid), "value": self.sim.act_to_display(aid, val)}
               for aid, val in gold_iv.items() if not isinstance(val, dict)]
        policy_text = None
        gp = self.sim.gold.get("policy")
        if gp:
            treat_id = next((aid for aid, v in gold_iv.items() if isinstance(v, dict)), None)
            if treat_id is not None:
                marker = gp["marker"]
                policy_text = {"treatment": self._alias(treat_id),
                               "stratifier": self.sim.scm.variables[marker].get("aliases", [marker])[0],
                               "threshold": self._marker_to_display(marker, gp["threshold"]),
                               "dose_if_ge": self.sim.act_to_display(treat_id, gp["dose_if_ge"]),
                               "dose_if_lt": self.sim.act_to_display(treat_id, gp["dose_if_lt"])}
        bat = self.sim.battery
        if self.calls == 3 and rec:
            a = {"actions": [rec[0]], "measure": [outcome_alias, proxy_alias]}
            return f'<reasoning>test targeted lever</reasoning>\n<action type="intervene">{json.dumps(a)}</action>\n<memory>Next: answer</memory>'
        signs = {self._alias(aid): s for aid, s in bat["actuator_sign_predictions"].items()}
        decoy_aliases = [self.sim.scm.variables[d].get("aliases", [d])[0] for d in bat["confounded_decoys"]]
        ans = {"recommended_intervention_text": rec,
               "structured": {"true_mechanism_proxy": proxy_alias,
                              "confounded_decoys": decoy_aliases,
                              "actuator_sign_predictions": signs},
               "explanation": self.sim.gt["latent_plain_name"]}
        if policy_text:
            ans["recommended_policy_text"] = policy_text
        return f'<reasoning>done</reasoning>\n<action type="answer">{json.dumps(ans)}</action>\n<memory>done</memory>'


VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def build_llm(backend, model, temperature, max_new_tokens):
    if backend == "mock":
        return MockAgent()
    if backend == "bedrock":
        from llm_bedrock import BedrockLLM
        return BedrockLLM(model_id=model, temperature=temperature, max_new_tokens=max_new_tokens)
    if backend in ("openai", "vllm"):
        from llm_openai import OpenAILLM
        if backend == "vllm":
            base_url = os.environ.get("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
            api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        else:
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAILLM(model_name=model, base_url=base_url, api_key=api_key,
                         max_new_tokens=max_new_tokens,
                         temperature=None if temperature is None else temperature)
    raise ValueError(backend)


def build_resolver_llm(agent_backend, agent_llm, no_resolver_llm, resolver_model="us.anthropic.claude-opus-4-8"):
    if no_resolver_llm or agent_backend == "mock":
        return None
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        try:
            from llm_bedrock import BedrockLLM
            resolver = BedrockLLM(model_id=resolver_model, max_new_tokens=200)
            logger.info("Resolver LLM = Bedrock %s.", resolver_model)
            return resolver
        except Exception as e:
            logger.error("!!! RESOLVER LLM (Bedrock) FAILED TO LOAD: %s: %s. "
                         "FALLING BACK TO THE AGENT'S OWN MODEL AS RESOLVER — resolution "
                         "quality WILL degrade and Part-A scores will be deflated. "
                         "Install boto3 / fix Bedrock creds to restore the intended resolver.",
                         type(e).__name__, e)
    else:
        logger.warning("AWS_BEARER_TOKEN_BEDROCK not set — using the agent's own model as "
                       "resolver (not the fixed strong resolver); resolution quality may degrade.")
    return agent_llm


def _resolve_answer_intervention(sim: Simulator, rec_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    iv = {}
    echoes = []
    for item in rec_text or []:
        r = sim.resolver.resolve_answer_term(item.get("request", ""), "intervene", item.get("value"))
        echoes.append({"request": item.get("request", ""), **r.to_dict()})
        if r.ok and r.target_id:
            iv[r.target_id] = r.value
    return iv, echoes


def _resolve_answer_policy(sim: Simulator, pol_text: Dict[str, Any]) -> Dict[str, Any]:
    if not pol_text:
        return None
    tr = sim.resolver.resolve_answer_term(pol_text.get("treatment", ""), "intervene", pol_text.get("dose_if_ge", 100))
    st = sim.resolver.resolve_answer_term(pol_text.get("stratifier", ""), "measure")
    if not (tr.ok and tr.target_id and st.ok and st.target_id):
        return {"_unresolved": True, "treatment_echo": tr.to_dict(), "stratifier_echo": st.to_dict()}
    return {"treatment": tr.target_id, "stratifier": st.target_id,
            "threshold": float(pol_text.get("threshold", 50.0)),
            "dose_if_ge": float(pol_text.get("dose_if_ge", 0.0)),
            "dose_if_lt": float(pol_text.get("dose_if_lt", 0.0))}


def run_world(sim: Simulator, llm, max_turns: int, budget: int, verbose: bool,
              system_prompt: Optional[str] = None) -> Dict[str, Any]:
    pub = sim.public()
    if isinstance(llm, MockAgent):
        llm.prime(sim)
    history: List[str] = []
    memory = "(empty)"
    used = 0
    n_interventions = 0
    turns = []
    final = None
    grade = None
    csv_map: Dict[str, str] = {}
    carried_vars: Dict[str, Any] = {}
    rejected_counts: Dict[str, int] = {}
    forced_answer = False

    for turn in range(max_turns):
        remaining = budget - used
        turns_left = max_turns - turn
        latest = history[-1] if history else "(none yet)"
        files_line = ("available data files for code: " + ", ".join(sorted(csv_map))) if csv_map else \
                     "available data files for code: (none yet — run a measure/intervene first)"

        directives = []
        if remaining <= 0 or turns_left <= 2:
            directives.append(
                "STOP EXPLORING. You are out of budget/turns. Submit your best "
                "<action type=\"answer\"> NOW using the evidence you have. Do not "
                "measure or run code this turn.")
            forced_answer = True
        if n_interventions == 0 and used >= 3:
            directives.append(
                "You have not run any INTERVENTION yet. Observation and code alone "
                "cannot establish causation or find the fix — apply a control/"
                "additive with <action type=\"intervene\"> to test a cause and to "
                "find what actually improves the outcome.")
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
        raw = llm.generate(system_prompt or (AGENT_PROMPT if sim.v10 else SYSTEM_PROMPT), user)
        dt = round(time.time() - t0, 2)
        finish_reason = getattr(llm, "last_finish_reason", None)
        memory = _tag(raw, "memory") or memory
        reasoning = _tag(raw, "reasoning")
        atype, payload = _parse_action(raw)
        rec: Dict[str, Any] = {"turn": turn, "latency_s": dt, "finish_reason": finish_reason,
                               "reasoning": reasoning, "memory": memory, "raw": raw,
                               "usage": getattr(llm, "last_usage", None), "prompt_chars": len(user)}
        if atype is None:
            rec["error"] = "no action parsed"
            turns.append(rec)
            continue
        rec["action_type"] = atype
        try:
            data = json.loads(payload) if payload and payload[0] in "{[" else {}
        except Exception as e:
            rec["error"] = f"json parse: {e}"
            data = {}

        if atype == "answer":
            iv, echoes = _resolve_answer_intervention(sim, data.get("recommended_intervention_text", []))
            pol = _resolve_answer_policy(sim, data.get("recommended_policy_text"))
            answer = {"recommended_intervention": iv,
                      "structured": _translate_structured(sim, data.get("structured", {})),
                      "explanation": data.get("explanation", "")}
            if pol and not pol.get("_unresolved"):
                answer["recommended_policy"] = pol
            grade = sim.grade(answer)
            rec["policy_resolution"] = pol
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
            from code_sandbox import run_code
            out, new_vars = run_code(payload, csv_map, carried_vars)
            carried_vars.update(new_vars)
            rec["code"] = payload
            rec["code_output"] = out
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
            history.append(json.dumps(sim.agent_view(result), default=str))
            rec["shown_to_agent"] = history[-1]
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
            history.append(json.dumps(sim.agent_view(result), default=str))
            rec["shown_to_agent"] = history[-1]
            if verbose:
                logger.info("turn %d INTERVENE %s (used %d/%d, interventions=%d)", turn,
                            result["applied_intervention"], used, budget, n_interventions)
        turns.append(rec)

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


def _artifact_check(sim: Simulator, answer_raw: Dict[str, Any], iv: Dict[str, Any],
                    echoes: List[Dict[str, Any]], structured: Dict[str, Any],
                    grade: Dict[str, Any]) -> Dict[str, Any]:
    reasons = []
    rec_text = answer_raw.get("recommended_intervention_text", []) or []
    if rec_text and not iv:
        reasons.append("recommended_intervention proposed but nothing resolved to an actuator")
    unresolved = [e["request"] for e in echoes if not e.get("ok")]
    if rec_text and len(unresolved) == len(rec_text) and rec_text:
        reasons.append(f"all {len(rec_text)} recommended action(s) rejected by resolver: {unresolved[:3]}")
    st = answer_raw.get("structured", {}) or {}
    proxy_txt = st.get("true_mechanism_proxy")
    if proxy_txt:
        pr = sim.resolver.resolve_answer_term(proxy_txt, "measure")
        if not pr.ok:
            reasons.append(f"named true_mechanism_proxy did not resolve: {proxy_txt!r}")
    if (not grade.get("accepted")) and grade.get("part_a_utility_ok") and not grade.get("part_b_battery_ok"):
        reasons.append("part A (utility) passed but part B (battery) failed — verify battery credited a correct answer")
    return {"suspect": len(reasons) > 0, "reasons": reasons}


def _translate_structured(sim: Simulator, st: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    pm = sim.resolver.resolve_answer_term(st.get("true_mechanism_proxy", ""), "measure") if st.get("true_mechanism_proxy") else None
    out["true_mechanism_proxy"] = pm.target_id if (pm and pm.ok) else st.get("true_mechanism_proxy")
    decoys = []
    for d in st.get("confounded_decoys", []) or []:
        dm = sim.resolver.resolve_answer_term(d, "measure")
        decoys.append(dm.target_id if dm.ok else d)
    out["confounded_decoys"] = decoys
    _DECREASE = ("reduce", "lower", "cut", "decrease", "less", "minimi", "turn down", "shut off", "stop")
    _FLIP = {"+": "-", "-": "+", "0": "0"}
    signs = {}
    for req, sign in (st.get("actuator_sign_predictions", {}) or {}).items():
        ar = sim.resolver.resolve_answer_term(req, "intervene")
        key = ar.target_id if (ar.ok and ar.target_id) else req
        rl = req.lower()
        if not sim.v10 and any(w in rl for w in _DECREASE) and sign in _FLIP:
            sign = _FLIP[sign]
        signs[key] = sign
    out["actuator_sign_predictions"] = signs
    return out


def load_world_file(path: str) -> tuple:
    from scm import WorldSCM
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    world = {"world_id": rec["world_id"], "domain": rec["domain"],
             "scenario": rec["scenario"], "scm": WorldSCM.from_dict(rec["scm"]),
             "ground_truth": rec["ground_truth"]}
    pre = {"gold": rec["oracle"]["gold"], "battery": rec["oracle"]["counterfactual_battery"]}
    return world, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-file", required=True, help="path to a world_*.json(.gz) file")
    ap.add_argument("--backend", choices=["bedrock", "openai", "vllm", "mock"], default="bedrock")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; default None = use the model's preset")
    ap.add_argument("--max-new-tokens", type=int, default=16000)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--max-turns", type=int, default=32,
                    help="total turns; code turns count here but not against --budget")
    ap.add_argument("--resolver-model", default="us.anthropic.claude-opus-4-8",
                    help="Bedrock model id of the request resolver (fixed across agent models)")
    ap.add_argument("--no-resolver-llm", action="store_true",
                    help="disable the LLM resolver fallback (on by default)")
    ap.add_argument("--system-prompt-file", default=None,
                    help="replace the agent system prompt with this file's text")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    resolver_llm = build_resolver_llm(args.backend, llm, args.no_resolver_llm, args.resolver_model)

    data_dir = str(Path(args.out).parent / (Path(args.out).stem + "_data"))
    world, pre = load_world_file(args.world_file)
    sim = Simulator(world, resolver_llm=resolver_llm, data_dir=data_dir, precomputed=pre)
    logger.info("gold=%s util=%.1f | raw data -> %s", sim.gold["intervention"],
                sim.gold["expected_utility"], data_dir)

    t0 = time.time()
    sysp = Path(args.system_prompt_file).read_text() if args.system_prompt_file else None
    res = run_world(sim, llm, args.max_turns, args.budget, args.verbose, system_prompt=sysp)
    res["system_prompt_file"] = args.system_prompt_file
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
