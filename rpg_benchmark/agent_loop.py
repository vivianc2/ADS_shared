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

# v10.0 = the release as first published (all main runs before 2026-10-16).
# v10.1 = v10.0 + accepts tool-call syntax (<invoke name="code">...) as the action; see CHANGELOG.md.
# v10.2 = unified reply parser, feedback on unreadable replies, clearer prompt and data view, isolated sandbox.
HARNESS_VERSION = "v10.2"

def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


_THINK_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r'<think\b[^>]*>.*$', re.DOTALL | re.IGNORECASE)
# Claude models sometimes write their tool-call namespace into tags (<action>, <invoke>, <memory>),
# occasionally twice; read them as the plain tags. No other model's reply contains "antml:".
_NS_RE = re.compile(r"<(/?)\s*(?:antml:)+", re.IGNORECASE)
_TYPES = ("measure", "intervene", "code", "answer", "give_up")
_START_RE = re.compile(r"<(action|invoke|parameter|code)(?::(\w+))?\b([^<>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""(\w+)\s*=\s*["']?([\w\-]+)["']?""")
_CHAT_TOKEN = r"<\|[a-z_]+\|>"
_END_RE = {
    "action": re.compile(r"</action|<action\b|<invoke\b|<memory\b|</memory|" + _CHAT_TOKEN, re.IGNORECASE),
    "invoke": re.compile(r"</invoke|<invoke\b|<action\b|<memory\b|" + _CHAT_TOKEN, re.IGNORECASE),
    "parameter": re.compile(r"</parameter|</invoke|</action|<parameter\b|<invoke\b|<action\b|<memory\b|</memory|"
                            + _CHAT_TOKEN, re.IGNORECASE),
    "code": re.compile(r"</code|<code\b|<action\b|<invoke\b|<memory\b|</memory|" + _CHAT_TOKEN, re.IGNORECASE),
}
_PARAM_RE = re.compile(
    r'<parameter\s+name\s*=\s*["\']?([\w\-]+)["\']?[^<>]*>(.*?)'
    r'(?:</parameter>|(?=</invoke>|</action>|<parameter\b|<memory>)|$)',
    re.DOTALL | re.IGNORECASE)
_STRAY_CLOSE_RE = re.compile(r"\s*</(?:parameter|invoke|code|answer|action)\s*>?\s*$", re.IGNORECASE)
# our own forced-answer directive, when a model quotes it back
_DIRECTIVE_QUOTE_RE = re.compile(r'<action type="answer">\s*NOW', re.IGNORECASE)


def _strip_think(raw: str) -> str:
    raw = _NS_RE.sub(r"<\1", raw or "")
    t = _THINK_OPEN_RE.sub(" ", _THINK_RE.sub(" ", raw))
    if re.search(r"</think\s*>", t, re.IGNORECASE):
        t = re.split(r"</think\s*>", t, flags=re.IGNORECASE)[-1]
    return t


def _is_json(v: str) -> bool:
    try:
        json.loads(v)
        return True
    except Exception:
        return False


def _loads_lenient(s: str):
    """JSON decode that tolerates what models actually write: code fences, stray closing tags (</parameter>,
    </answer>), leaked chat tokens, trailing text or an extra closing brace, trailing commas.
    Returns (object, repaired: bool); raises ValueError if nothing readable is found."""
    s = _strip_fences(s or "")
    try:
        return json.loads(s), False
    except Exception:
        pass
    t = re.sub(r"</?(?:parameter|invoke|answer|action|code)\b[^>]*>", "", s, flags=re.IGNORECASE)
    t = _strip_fences(re.sub(_CHAT_TOKEN, "", t))
    dec = json.JSONDecoder()
    # an object first (a broken answer must not degrade to an inner list), then a top-level list
    for st in [i for i in (t.find("{"), t.find("[")) if i >= 0]:
        for cand in (t[st:], re.sub(r",\s*([}\]])", r"\1", t[st:])):
            try:
                obj, _ = dec.raw_decode(cand)
            except Exception:
                continue
            if isinstance(obj, dict) or (isinstance(obj, list) and st == t.find("[") and
                                         (t.find("{") < 0 or t.find("[") < t.find("{"))):
                return obj, True
    raise ValueError("no readable JSON object")


_LIST_FIELDS = {"measure": ("requests",), "intervene": ("actions", "measure")}


def _invoke_payload(atype: str, body: str) -> str:
    params = [(n, v.strip()) for n, v in _PARAM_RE.findall(body)]
    if not params:
        return _STRAY_CLOSE_RE.sub("", body.strip())
    if atype == "code" or (len(params) == 1 and params[0][1][:1] in "{["):
        payload = params[0][1]
        if atype != "code" and payload[:1] == "[" and _is_json(payload):   # a bare list -> the action's list field
            payload = json.dumps({"requests" if atype == "measure" else "actions": json.loads(payload)})
        return payload
    obj = {}
    for n, v in params:
        if not v or (atype in _LIST_FIELDS and n not in _LIST_FIELDS[atype]):
            continue                       # v10.2: Sonnet's placeholder <parameter name="dummy"></parameter>
        obj[n] = json.loads(v) if _is_json(v) else v
    return json.dumps(obj) if obj else ""


def _candidates(raw: str) -> List[tuple]:
    """Every action block in a reply, in either dialect: <action type="X">, tool-call <invoke name="X"> with
    <parameter> children, a bare <parameter name="X"> / <parameter name="action" type="X">, or a bare <code>
    block. Returns [(type, payload, dialect)] in order, skipping empty blocks."""
    t = _DIRECTIVE_QUOTE_RE.sub(" NOW", _strip_think(raw))
    out, pos = [], 0
    for m in _START_RE.finditer(t):
        if m.start() < pos:
            continue                       # nested inside the previous block (e.g. <parameter> inside <invoke>)
        tag = m.group(1).lower()
        attrs = {k.lower(): v.lower() for k, v in _ATTR_RE.findall(m.group(3))}
        if m.group(2) and m.group(2).lower() in _TYPES:      # Haiku's mangled form <invoke:code>
            atype = m.group(2).lower()
        elif tag == "action":
            atype = attrs.get("type")
        elif tag == "invoke":
            atype = attrs.get("name") if attrs.get("name") in _TYPES else attrs.get("type")
        elif tag == "parameter":
            atype = attrs.get("name") if attrs.get("name") in _TYPES else attrs.get("type")
            if atype is None:                                  # Haiku's <parameter="code">
                mm = re.match(r"""\s*=\s*["']?(\w+)""", m.group(3))
                atype = mm.group(1).lower() if mm else None
        else:
            atype = "code"
        if atype not in _TYPES:
            continue
        e = _END_RE[tag].search(t, m.end())
        body = t[m.end(): e.start() if e else len(t)]
        # a closing tag is consumed; an opening tag (the next block) is left for the next match
        pos = (e.end() if e.group(0).startswith("</") else e.start()) if e else len(t)
        payload = _invoke_payload(atype, body) if tag == "invoke" else _STRAY_CLOSE_RE.sub("", body.strip())
        payload = _strip_fences(payload)
        if not payload.strip() and atype != "give_up":
            continue                       # an empty block (e.g. Sonnet's empty <action type="code"></action>)
        out.append((atype, payload, "action" if tag == "action" else "invoke"))
    return out


def _valid(atype: str, payload: str) -> bool:
    if atype == "give_up":
        return True
    if atype == "code":
        try:
            compile(payload, "<agent>", "exec")
            return True
        except Exception:
            return False
    try:
        obj, _ = _loads_lenient(payload)
    except ValueError:
        return False
    if atype in _LIST_FIELDS and isinstance(obj, dict):
        # v10.2: a measure needs a "requests" list and an intervene an "actions" list; any other listed field
        # present must be a list too (a garbled '<parameter name="measure":[...]' gave measure="")
        need, opt = _LIST_FIELDS[atype][0], _LIST_FIELDS[atype][1:]
        return isinstance(obj.get(need), list) and all(isinstance(obj[k], list) for k in opt if k in obj)
    return True


def _parse_reply(raw: str, forced: bool = False):
    """v10.2 action parser. Returns (type, payload, dialect, note, error).
    One action per turn. With several blocks, the first block is executed and the agent is told the rest were
    ignored; it is skipped only for a later VALID block of the same type (code that compiles, JSON that parses).
    On a forced-answer turn the last valid answer is preferred.
    Prose that merely mentions a tag ("I will output an <action type=\"code\"> block") yields an invalid block and
    is skipped in favour of the real one."""
    cands = _candidates(raw)
    if not cands:
        return None, None, None, None, "no action block found in your reply"
    if forced:
        answers = [c for c in cands if c[0] == "answer" and _valid(c[0], c[1])]
        if answers:
            c = answers[-1]
            return c[0], c[1], c[2], ("answer taken from a reply with several actions" if len(cands) > 1 else None), None
    # The first block is executed. It is skipped only for a later block of the SAME type when it is invalid
    # (prose that merely mentions the tag, followed by the real block); a reply never jumps to a different kind
    # of action (e.g. an invalid code block followed by an invented answer runs the code, which then reports
    # its error).
    c = cands[0]
    skipped = 0
    if not _valid(c[0], c[1]):
        later = [x for x in cands[1:] if x[0] == c[0] and _valid(x[0], x[1])]
        if later:
            skipped = cands.index(later[0])
            c = later[0]
    rest = [x for x in cands if x is not c and " ".join(x[1].split()) != " ".join(c[1].split())]
    note = None
    if rest:
        kinds = ", ".join(dict.fromkeys(x[0] for x in rest))
        note = (f"your reply contained {len(rest) + 1} action blocks; only one ({c[0]}) was executed"
                + (" (an earlier, unreadable block was skipped)" if skipped else "")
                + f"; the other(s) ({kinds}) were ignored. Write exactly ONE action per turn")
    return c[0], c[1], c[2], note, None


def _tag(text: str, tag: str) -> str:
    text = _strip_think(text)
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if not m:   # v10.2: an unclosed block at the end of the reply
        m = re.search(rf"<{tag}>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
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


def _make_agent_prompt_v102(p: str) -> str:
    """v10.2: state how the environment works (memory, fresh units, baselines, limits, reply format)."""
    reps = [
        ("3) Run analysis code over the raw data you have collected. Each measure/intervene\n"
         "   is preloaded for you as a pandas DataFrame named experiment_<id>_df (per-unit rows);\n"
         "   its file path is also available as experiment_<id>_csv. Use the DataFrame directly --\n"
         "   do NOT pd.read_csv a bare name. You have pd (pandas), np (numpy), and stats\n",
         "3) Run analysis code over the raw data you have collected. Each measure/intervene\n"
         "   is preloaded for you as a pandas DataFrame named experiment_<id>_df (one row per unit;\n"
         "   also saved as the file experiment_<id>.csv). Use the DataFrame directly. You have pd (pandas),\n"
         "   np (numpy), and stats\n"),
        ("Output EXACTLY these two blocks each turn:\n",
         "HOW THIS ENVIRONMENT WORKS\n"
         "- MEMORY: each turn you are shown only the situation report, the budget and turn counters, the list of\n"
         "  your data, the result of your LAST action, and your <memory> block. Earlier results, earlier code output\n"
         "  and your earlier reasoning are NOT shown again. Write every number and finding you will need later into\n"
         "  <memory>; it is your only record between turns. (Your data stays available to code as experiment_<id>_df.)\n"
         "  You write <memory> in the same reply as your next action, before that action's result exists. So in every\n"
         "  reply, first copy the numbers you need from RESULT OF YOUR LAST ACTION into <memory>: next turn that result\n"
         "  is replaced by the new one. Re-running the same code only reproduces the same output.\n"
         "- UNITS: every experiment uses a fresh, independent batch of units, so rows of different experiments are\n"
         "  different units (unit ids are unique to each experiment). Compare experiments by their distributions,\n"
         "  never unit by unit.\n"
         "- BASELINES: a measurement without an intervention records the units as they currently run, which may be a\n"
         "  filtered subset; every intervention is applied to a fresh, representative batch. So an intervention's\n"
         "  outcome can differ from the observational baseline even when the control itself has no effect: compare\n"
         "  interventions with each other. (An intervene action with an empty \"actions\" list gives such a\n"
         "  representative batch with no control changed, i.e. an interventional baseline.)\n"
         "- LIMITS: the experiment budget and the turn limit are shown every turn; code turns use turns but not\n"
         "  experiments. When 2 turns remain you must answer.\n"
         "- FORMAT: each reply must contain exactly ONE action block, <action type=\"...\">...</action> (closed with\n"
         "  </action>), followed by ONE <memory>...</memory> block. Do not use tool-call or function-call syntax;\n"
         "  these actions are the only tools. If a reply contains several actions, only the first is executed; a\n"
         "  reply with no valid action is not executed (you are told so, and the turn is used).\n\n"
         "Output EXACTLY these two blocks each turn:\n"),
        ("<memory>\nHypotheses:\nRuled out:\nConfirmed:\nNext:\n</memory>",
         "<memory>\nHypotheses: ...\nRuled out: ...\nConfirmed: ... (your findings so far, with the numbers)\nNext: ...\n</memory>"),
        # v10.2: neutral example names. The old examples (chelating additive, feed-water flow, product titer, broth
        # cloudiness) are real controls/signals in some worlds, with a sign; models copied them into requests and
        # answers.
        ('<action type="measure">{"requests":["product titer","how cloudy the broth is","dissolved oxygen"]}</action>',
         '<action type="measure">{"requests":["<signal name>","<another signal, in your own words>"]}</action>'),
        ('<action type="intervene">{"actions":[{"request":"add a metal-chelating additive","value":60},\n'
         '                                          {"request":"reduce the feed-water flow","value":0}],\n'
         '                              "measure":["product titer","broth cloudiness"]}</action>',
         '<action type="intervene">{"actions":[{"request":"<an additive>","value":60},\n'
         '                                          {"request":"<a control>","value":0}],\n'
         '                              "measure":["<signal name>","<another signal>"]}</action>'),
        ('slope,inter,r,p,se = stats.linregress(df["SomeSignal"], df["ProductTiter"])',
         'slope,inter,r,p,se = stats.linregress(df["SignalA"], df["SignalB"])   # use the real column names'),
        ('     "recommended_intervention_text":[{"request":"add a chelating additive","value":66}],\n',
         '     "recommended_intervention_text":[{"request":"<a control or additive>","value":66}],\n'),
        ('"actuator_sign_predictions":{"add a chelating additive":"+","reduce feed-water flow":"+","add a stabilizer":"0"}',
         '"actuator_sign_predictions":{"<a control>":"+","<another control>":"-","<a third control>":"0"}'),
        ("- LIMITS: the experiment budget and the turn limit are shown every turn; code turns use turns but not\n"
         "  experiments. When 2 turns remain you must answer.",
         "- LIMITS: the experiment budget and the turn limit are shown every turn; code turns use turns but not\n"
         "  experiments. When the budget is spent you can still run code. When 2 turns remain you must answer."),
    ]
    for a, b in reps:
        assert p.count(a) == 1, a[:60]
        p = p.replace(a, b)
    return p


AGENT_PROMPT = _make_agent_prompt_v102(_make_agent_prompt(SYSTEM_PROMPT))

# v10.2 ablations (week 7). Each variant is a list of exact (old, new) edits on AGENT_PROMPT; the default prompt is
# unchanged. Selected with run_benchmark.py --prompt-variant (comma-separated) and --history full|experiments|
# experiments_add_memory.
_COMBO_NOTE = ("- The best fix may be a COMBINATION, and the best dose may be in the middle\n"
               "  (too much can hurt).")
_TRAP_SENTENCE = (" The reported outcome is a SURROGATE metric that can be\n"
                  "  gamed: a control that changes the *reading* but not the true underlying state\n"
                  "  is a trap, and recommending it as the fix is wrong.")
PROMPT_VARIANTS = {
    # R10: drop the surrogate/trap warning (the sign definition still says what a reading-only mover is)
    "no_trap_warning": [(_TRAP_SENTENCE, "")],
    # R5: say that several independent causes and subgroup-specific responses are possible (no world-specific info)
    "multi_cause": [(_COMBO_NOTE, _COMBO_NOTE + "\n"
                     "- The problem may have MORE THAN ONE independent cause, each needing its own fix at its own dose.\n"
                     "  A fix for one cause can leave the others in place, so the outcome improves only partly. After you\n"
                     "  find a fix, test whether another control adds a further improvement on top of it, and recommend\n"
                     "  all the fixes together.\n"
                     "- If a treatment's average effect is near zero but units respond differently (the spread widens or\n"
                     "  the units split into groups), look for a measured marker that separates the units that benefit\n"
                     "  from those that are harmed (see the CONDITIONAL note below).")],
    # R6b: record every setting and signal in the first batch
    "first_batch": [(_COMBO_NOTE, _COMBO_NOTE + "\n"
                     "- In your FIRST experiment, record the outcome together with EVERY signal and EVERY control setting\n"
                     "  the situation report lists (settings vary naturally across units), so you can see which settings\n"
                     "  and signals move with the outcome before you choose what to test.")],
}
_MEMORY_STATELESS = (
    "- MEMORY: each turn you are shown only the situation report, the budget and turn counters, the list of\n"
    "  your data, the result of your LAST action, and your <memory> block. Earlier results, earlier code output\n"
    "  and your earlier reasoning are NOT shown again. Write every number and finding you will need later into\n"
    "  <memory>; it is your only record between turns. (Your data stays available to code as experiment_<id>_df.)\n"
    "  You write <memory> in the same reply as your next action, before that action's result exists. So in every\n"
    "  reply, first copy the numbers you need from RESULT OF YOUR LAST ACTION into <memory>: next turn that result\n"
    "  is replaced by the new one. Re-running the same code only reproduces the same output.\n")
_MEMORY_FULL = (
    "- MEMORY: you see the whole conversation. Every earlier result, code output and reply of yours stays in your\n"
    "  context, so nothing is lost between turns. The <memory> block is still required but is only notes for\n"
    "  yourself. (Your data also stays available to code as experiment_<id>_df.) Re-running the same code only\n"
    "  reproduces the same output.\n")
# history="experiments": the prompt lists every experiment's result; there is no <memory> block
_MEMORY_EXPERIMENTS = (
    "- HISTORY: each turn you are shown the situation report, the budget and turn counters, the list of your\n"
    "  data, a HISTORICAL RECORD OF EXPERIMENTS with the result of every experiment you have run so far, and the\n"
    "  result of your LAST action. Earlier code output and your earlier replies are NOT shown again. (Your data\n"
    "  stays available to code as experiment_<id>_df.) Re-running the same code only reproduces the same output.\n")
_NO_MEMORY_EDITS = [
    (_MEMORY_STATELESS, _MEMORY_EXPERIMENTS),
    ("  </action>), followed by ONE <memory>...</memory> block. Do not use", "  </action>). Do not use"),
    ("Output EXACTLY these two blocks each turn:\n", "Output EXACTLY this one block each turn:\n"),
    ("JSON or code</action>\n<memory>\nHypotheses: ...\nRuled out: ...\n"
     "Confirmed: ... (your findings so far, with the numbers)\nNext: ...\n</memory>", "JSON or code</action>"),
]
# history="experiments_add_memory": the record of history="experiments", and the <memory> block is kept
_MEMORY_EXPERIMENTS_ADD_MEMORY = (
    "- MEMORY: each turn you are shown the situation report, the budget and turn counters, the list of your data,\n"
    "  a HISTORICAL RECORD OF EXPERIMENTS with the result of every experiment you have run so far, the result of\n"
    "  your LAST action, and your <memory> block. Earlier code output and your earlier reasoning are NOT shown\n"
    "  again. Write every number and finding you will need later into <memory>; apart from the record, it is your\n"
    "  only record between turns. (Your data stays available to code as experiment_<id>_df.) You write <memory> in\n"
    "  the same reply as your next action, before that action's result exists. So in every reply, first copy the\n"
    "  numbers you need from RESULT OF YOUR LAST ACTION into <memory>: next turn that result is replaced by the new\n"
    "  one. Re-running the same code only reproduces the same output.\n")
# histories that show the HISTORICAL RECORD OF EXPERIMENTS
_RECORD_HISTORIES = ("experiments", "experiments_add_memory")
HISTORIES = ("stateless", "full") + _RECORD_HISTORIES


def build_agent_prompt(variants: Optional[List[str]] = None, history: str = "stateless") -> str:
    p = AGENT_PROMPT
    edits = [e for v in (variants or []) for e in PROMPT_VARIANTS[v]]
    if history == "full":
        edits.append((_MEMORY_STATELESS, _MEMORY_FULL))
    elif history == "experiments":
        edits += _NO_MEMORY_EDITS
    elif history == "experiments_add_memory":
        edits.append((_MEMORY_STATELESS, _MEMORY_EXPERIMENTS_ADD_MEMORY))
    for a, b in edits:
        assert p.count(a) == 1, a[:60]
        p = p.replace(a, b)
    return p

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
    if not isinstance(pol_text, dict):
        return {"_unresolved": True, "malformed": repr(pol_text)[:200]}
    tr = sim.resolver.resolve_answer_term(pol_text.get("treatment", ""), "intervene", pol_text.get("dose_if_ge", 100))
    st = sim.resolver.resolve_answer_term(pol_text.get("stratifier", ""), "measure")
    if not (tr.ok and tr.target_id and st.ok and st.target_id):
        return {"_unresolved": True, "treatment_echo": tr.to_dict(), "stratifier_echo": st.to_dict()}
    try:
        return {"treatment": tr.target_id, "stratifier": st.target_id,
                "threshold": float(pol_text.get("threshold", 50.0)),
                "dose_if_ge": float(pol_text.get("dose_if_ge", 0.0)),
                "dose_if_lt": float(pol_text.get("dose_if_lt", 0.0))}
    except (TypeError, ValueError):     # v10.1: e.g. a threshold given as a list -> policy unresolved, not a crash
        return {"_unresolved": True, "malformed": repr(pol_text)[:200]}


_TAG_LINE_RE = re.compile(r"\s*</?[A-Za-z_][\w:.-]*\s*/?>\s*")


def _not_executed(reason: str, previous: str, format_hint: bool = True, memory: bool = True) -> str:
    """v10.2: what the agent sees after a reply that could not be executed (it used to see nothing new)."""
    return ("YOUR LAST REPLY WAS NOT EXECUTED: " + reason + ".\n"
            + ("Write exactly ONE action block per turn, closed with </action>"
               + (", followed by your <memory> block" if memory else "") + ", e.g.\n"
               "<action type=\"code\">print(experiment_1_df.describe())</action>\n" if format_hint else "")
            + "No experiment was used.\n\n"
            "THE RESULT BEFORE THAT (repeated):\n" + previous)


def _data_line(experiments: Dict[int, Dict[str, Any]], sim: Simulator) -> str:
    if not experiments:
        return "data for code: none yet (run a measure or intervene first)"
    parts = []
    for eid in sorted(experiments):
        e = experiments[eid]
        if e["mode"] == "intervene" and not e["intervention"]:
            parts.append(f"experiment_{eid}_df (baseline batch: representative units, no control changed)")
            continue
        if e["mode"] == "intervene" and e["intervention"]:
            what = "intervention: " + ", ".join(f"{sim.csv_columns.get('do_' + a, sim.act_label.get(a, a))}={v}"
                                                for a, v in e["intervention"].items())
        else:
            what = "observational"
        parts.append(f"experiment_{eid}_df ({what})")
    return "data for code: " + "; ".join(parts)


def _experiment_record_entry(eid: int, view: Dict[str, Any]) -> str:
    """One entry of the history="experiments" record: a header line + the JSON the agent sees for that experiment."""
    if view.get("mode") == "intervene":
        applied = view.get("applied_intervention") or {}
        if applied:
            head = (f"Intervene on: {', '.join(applied)} with dose: "
                    f"{', '.join(str(v) for v in applied.values())} resulting the following:")
        else:
            head = "Baseline batch (no control changed) resulting the following:"
    else:
        head = "Measure only (no intervention) resulting the following:"
    return f"Experiment {eid}:\n{head}\n{json.dumps(view, default=str)}"


def run_world(sim: Simulator, llm, max_turns: int, budget: int, verbose: bool,
              system_prompt: Optional[str] = None, history: str = "stateless",
              min_answer_turn: Optional[int] = None,
              prompt_variants: Optional[List[str]] = None) -> Dict[str, Any]:
    """history="full" (ablation R8) sends the whole conversation each turn instead of last result + memory;
    history="experiments" shows a record of every experiment's result each turn and drops the <memory> block;
    history="experiments_add_memory" shows the same record and keeps the <memory> block;
    min_answer_turn (R6) refuses an answer before that turn (unless the turn limit forces it); prompt_variants
    (R5/R6b/R10) are named edits from PROMPT_VARIANTS. The defaults reproduce the v10.2 protocol."""
    assert history in HISTORIES, history
    ablation = {"history": history, "min_answer_turn": min_answer_turn, "prompt_variants": list(prompt_variants or [])}
    if system_prompt is None and sim.v10 and (history != "stateless" or prompt_variants):
        system_prompt = build_agent_prompt(prompt_variants, history)
    conv: List[Dict[str, str]] = []  # history="full": earlier user/assistant messages
    pub = sim.public()
    if isinstance(llm, MockAgent):
        llm.prime(sim)
    latest = "(none yet)"            # what "RESULT OF YOUR LAST ACTION" shows
    last_result = "(none yet)"       # the last executed action's result (repeated after an unexecuted reply)
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
    experiment_record: List[str] = []  # history in _RECORD_HISTORIES: one entry per experiment run

    def _nx(reason: str, previous: str, format_hint: bool = True) -> str:
        return _not_executed(reason, previous, format_hint, memory=history != "experiments")

    for turn in range(max_turns):
        remaining = budget - used
        turns_left = max_turns - turn

        directives = []
        # v10.2: only the turn limit forces the answer. With the budget spent and turns left, code (free) is still
        # allowed; measure/intervene are refused without charge (v10.0/10.1 forced an immediate answer, so the
        # last experiment was never analysed).
        forced_now = turns_left <= 2
        budget_spent = remaining <= 0
        if forced_now:
            directives.append(
                "STOP EXPLORING. You are out of turns. Submit your final answer NOW (an answer action), "
                "using the evidence you have. Do not measure or run code this turn.")
            forced_answer = True
        elif budget_spent:
            directives.append(
                "Your experiment budget is spent: measure and intervene can no longer run. You can still analyse "
                "your data with code (free), and you must answer before the turn limit.")
        if n_interventions == 0 and used >= 3:
            directives.append(
                "You have not run any INTERVENTION yet. Observation and code alone "
                "cannot establish causation or find the fix — apply a control/"
                "additive with an intervene action to test a cause and to "
                "find what actually improves the outcome.")
        stuck = [req for req, c in rejected_counts.items() if c >= 3]
        if stuck:
            directives.append(
                "Several of your requests have been rejected repeatedly because "
                "those things are NOT part of this system (no such measurement or "
                "actuator exists): " + "; ".join(sorted(stuck)[:5]) + ". Stop "
                "asking for them and reconsider your hypothesis using only what IS "
                "available.")
        if min_answer_turn is not None and turn < min_answer_turn and not forced_now:
            directives.append(f"Answers are accepted only from turn {min_answer_turn} on (an earlier answer is not "
                              "executed). Keep investigating until then.")
        directive_block = ("\nDIRECTIVES\n" + "\n".join(f"- {d}" for d in directives) + "\n") if directives else ""

        if history == "full" and turn > 0:
            # the situation report, earlier results and the agent's replies are already in the conversation
            user = f"""BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{_data_line(sim.experiments, sim)}
{directive_block}
RESULT OF YOUR LAST ACTION
{latest}
"""
        elif history in _RECORD_HISTORIES:
            record = "\n\n".join(experiment_record) if experiment_record else "(no experiments yet)"
            user = f"""SITUATION REPORT
{pub['scenario']}

OUTCOME OF INTEREST: {pub['outcome_name']} ({pub['outcome_direction']})

BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{_data_line(sim.experiments, sim)}
{directive_block}
HISTORICAL RECORD OF EXPERIMENTS
{record}

RESULT OF YOUR LAST ACTION
{latest}
"""
            if history == "experiments_add_memory":
                user = user.replace("RESULT OF YOUR LAST ACTION\n", "RESULT OF YOUR LAST ACTION (shown this turn only: "
                                    "copy what you need into <memory>)\n") + f"\nYOUR MEMORY\n{memory}\n"
        else:
            user = f"""SITUATION REPORT
{pub['scenario']}

OUTCOME OF INTEREST: {pub['outcome_name']} ({pub['outcome_direction']})

BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{_data_line(sim.experiments, sim)}
{directive_block}
RESULT OF YOUR LAST ACTION (shown this turn only: copy what you need into <memory>)
{latest}

YOUR MEMORY
{memory}
"""
            if history == "full":
                user = user.replace(" (shown this turn only: copy what you need into <memory>)", "")
        t0 = time.time()
        sysp_now = system_prompt or (AGENT_PROMPT if sim.v10 else SYSTEM_PROMPT)
        try:
            if history == "full" and hasattr(llm, "generate_messages"):
                msgs = [{"role": "system", "content": sysp_now}] + [dict(m) for m in conv] \
                    + [{"role": "user", "content": user}]
                if len(msgs) > 2:
                    msgs[-2]["cache"] = True   # Bedrock/Claude: cache the conversation prefix (ignored elsewhere)
                raw = llm.generate_messages(msgs)
            else:
                raw = llm.generate(sysp_now, user)
        except RuntimeError as e:
            # a Bedrock reply with no text block (e.g. gpt-oss returning only reasoning) is an empty turn
            if "no text in Bedrock reply" not in str(e):
                raise
            logger.warning("empty agent reply (%s); counted as a turn with no action", e)
            raw = ""
        dt = round(time.time() - t0, 2)
        if history == "full":
            conv += [{"role": "user", "content": user}, {"role": "assistant", "content": raw or "(empty reply)"}]
        finish_reason = getattr(llm, "last_finish_reason", None)
        memory = _tag(raw, "memory") or memory
        reasoning = _tag(raw, "reasoning")
        atype, payload, dialect, note, perr = _parse_reply(raw, forced=forced_now or budget_spent)
        rec: Dict[str, Any] = {"turn": turn, "latency_s": dt, "finish_reason": finish_reason,
                               "reasoning": reasoning, "memory": memory, "raw": raw,
                               "model_reasoning": getattr(llm, "last_reasoning", None),
                               "usage": getattr(llm, "last_usage", None), "prompt_chars": len(user)}
        if getattr(llm, "last_served_model", None):
            rec["served_model"] = llm.last_served_model
        if getattr(llm, "last_text_blocks", None):
            rec["text_blocks"] = llm.last_text_blocks      # audit: a reply with several text blocks (last one is used)
        if dialect == "invoke":
            rec["parsed_via"] = "invoke"
        if note:
            rec["parse_note"] = note
        if atype is None:
            rec["error"] = "no action parsed"
            rec["error_detail"] = perr
            latest = _nx(perr, last_result)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        data: Any = {}
        if atype in ("measure", "intervene", "answer"):
            try:
                data, repaired = _loads_lenient(payload)
                if repaired:
                    rec["json_repaired"] = True
                if isinstance(data, list):          # a bare list: the action's list field
                    if atype == "answer":
                        raise ValueError("an answer must be a JSON object")
                    data = {"requests" if atype == "measure" else "actions": data}
            except ValueError as e:
                # v10.2: unreadable JSON is not executed (and no experiment is charged); the agent is told
                rec["error"] = f"json parse: {e}"
                rec["attempted_action_type"] = atype
                latest = _nx(f"the JSON in your {atype} action could not be read", last_result)
                rec["shown_next"] = latest
                turns.append(rec)
                continue
        if atype in _LIST_FIELDS and isinstance(data, dict) and not _valid(atype, json.dumps(data)):
            # v10.2: a measure without a "requests" list, or an intervene without an "actions" list, is not run
            # (an intervene without "actions" used to run as a charged baseline batch)
            need = _LIST_FIELDS[atype][0]
            rec["error"] = f"{atype} without a {need} list"
            rec["attempted_action_type"] = atype
            latest = _nx(f'your {atype} action needs a "{need}" list'
                                   + (' (use "actions": [] for a baseline batch with no control changed)'
                                      if atype == "intervene" else ""), last_result)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        if atype in ("measure", "intervene") and budget_spent:
            rec["error"] = "budget spent"
            rec["attempted_action_type"] = atype
            latest = _nx(f"the experiment budget is spent ({used}/{budget}), so a {atype} can no longer "
                                   "run. Use code to analyse your data, or answer", last_result, format_hint=False)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        if atype == "answer" and min_answer_turn is not None and turn < min_answer_turn and not forced_now:
            rec["error"] = "answer before min_answer_turn"
            rec["attempted_action_type"] = atype
            latest = _nx(f"answers are accepted only from turn {min_answer_turn} on (this was turn {turn}); "
                                   "keep investigating", last_result, format_hint=False)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        rec["action_type"] = atype

        if atype == "answer":
            # normalise the answer's fields so a malformed answer is graded (as what it validly says) instead
            # of crashing the batch; the original is kept in answer_raw.
            if not isinstance(data, dict):
                rec["grade_error"] = f"answer is a {type(data).__name__}, not an object"
                data = {}
            data = dict(data)
            for k, typ, empty in (("recommended_intervention_text", list, []), ("structured", dict, {}),
                                  ("recommended_policy_text", dict, None)):
                if k in data and data[k] is not None and not isinstance(data[k], typ):
                    rec["grade_error"] = f"{k} is a {type(data[k]).__name__}"
                    data[k] = empty
            data["recommended_intervention_text"] = [x for x in data.get("recommended_intervention_text") or []
                                                     if isinstance(x, dict)]
            if isinstance(data.get("structured"), dict):
                st_ = data["structured"] = dict(data["structured"])
                px = st_.get("true_mechanism_proxy")
                if px is not None and not isinstance(px, str):
                    st_["true_mechanism_proxy"] = next((x for x in px if isinstance(x, str)), "") \
                        if isinstance(px, list) else str(px)
                dc = st_.get("confounded_decoys")
                if dc is not None and not isinstance(dc, list):
                    st_["confounded_decoys"] = [dc] if isinstance(dc, str) else []
                if isinstance(st_.get("confounded_decoys"), list):
                    st_["confounded_decoys"] = [x for x in st_["confounded_decoys"] if isinstance(x, str)]
                sg = st_.get("actuator_sign_predictions")
                if sg is not None and not isinstance(sg, dict):
                    st_["actuator_sign_predictions"] = {}
            try:
                iv, echoes = _resolve_answer_intervention(sim, data.get("recommended_intervention_text", []))
                pol = _resolve_answer_policy(sim, data.get("recommended_policy_text"))
                answer = {"recommended_intervention": iv,
                          "structured": _translate_structured(sim, data.get("structured", {})),
                          "explanation": data.get("explanation", "")}
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("malformed answer (%s); graded as empty", e)
                rec["grade_error"] = f"{type(e).__name__}: {e}"
                iv, echoes, pol = {}, [], None
                answer = {"recommended_intervention": {}, "structured": {}, "explanation": str(data.get("explanation", ""))}
            if pol and not pol.get("_unresolved"):
                answer["recommended_policy"] = pol
            try:
                grade = sim.grade(answer)
            except (TypeError, ValueError) as e:
                logger.warning("answer could not be graded as given (%s); dropping malformed values", e)
                rec["grade_error"] = f"{type(e).__name__}: {e}"
                answer.pop("recommended_policy", None)
                answer["recommended_intervention"] = {
                    k: v for k, v in (answer.get("recommended_intervention") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
                try:
                    grade = sim.grade(answer)
                except (TypeError, ValueError):
                    answer["recommended_intervention"] = {}
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
            # v10.2: drop lines that are only an XML-like tag (models append invented closers such as
            # </parameter_text>); such a line is never valid Python and made the whole cell a SyntaxError
            payload = "\n".join(l for l in payload.split("\n") if not _TAG_LINE_RE.fullmatch(l))
            out, new_vars = run_code(payload, csv_map, carried_vars)
            carried_vars.update(new_vars)
            rec["code"] = payload
            rec["code_output"] = out
            latest = last_result = "CODE OUTPUT:\n" + out + (f"\n[note: {note}]" if note else "")
            if verbose:
                logger.info("turn %d CODE (%d chars out, %d files)", turn, len(out), len(csv_map))
            turns.append(rec)
            continue

        if not isinstance(data, dict):
            data = {}
        if atype == "measure":
            result = sim.measure(data.get("requests", []) if isinstance(data.get("requests", []), list) else [])
            if result.get("raw_csv"):
                used += 1
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            for e in result["resolutions"]:
                if not e["ok"] and e.get("kind") == "reject" and not e.get("candidates"):
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            if verbose:
                ok = [e["target_id"] for e in result["resolutions"] if e["ok"]]
                logger.info("turn %d MEASURE -> %s (used %d/%d)", turn, ok, used, budget)
        else:  # intervene
            acts = data.get("actions", [])
            meas = data.get("measure", [])
            result = sim.intervene(acts if isinstance(acts, list) else [], meas if isinstance(meas, list) else [])
            if result.get("raw_csv"):
                used += 1
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            if result.get("applied_intervention"):
                n_interventions += 1
            for e in result["action_resolutions"]:
                if not e["ok"] and e.get("kind") in ("reject", "signal") and not e.get("candidates"):
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            if verbose:
                logger.info("turn %d INTERVENE %s (used %d/%d, interventions=%d)", turn,
                            result["applied_intervention"], used, budget, n_interventions)
        view = sim.agent_view(result)
        if not result.get("raw_csv"):
            # v10.2: nothing could be run (every request rejected): no experiment is used, and the agent is told
            view["note"] = "Nothing was run (no request could be carried out), so no experiment was used."
            rec["not_charged"] = True
        latest = last_result = json.dumps(view, default=str) + (f"\n[note: {note}]" if note else "")
        rec["shown_to_agent"] = latest
        if history in _RECORD_HISTORIES and result.get("raw_csv"):
            # the full result goes into the record; the last-action slot points to it instead of repeating it
            experiment_record.append(_experiment_record_entry(result["experiment_id"], view))
            latest = last_result = (f"Experiment {result['experiment_id']} -- see HISTORICAL RECORD OF EXPERIMENTS "
                                    "above." + (f"\n[note: {note}]" if note else ""))
        turns.append(rec)

    if grade is None:
        synth = {"recommended_intervention": {}, "structured": {}, "explanation": memory}
        grade = sim.grade(synth)
        grade["forced_no_answer"] = True
        turns.append({"turn": "post", "action_type": "forced_answer",
                      "note": "hit turn cap without answering; graded empty answer from memory",
                      "grade": grade})

    return {"world_id": sim.world["world_id"], "domain": sim.world["domain"], "harness_version": HARNESS_VERSION,
            "ablation": ablation,
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
