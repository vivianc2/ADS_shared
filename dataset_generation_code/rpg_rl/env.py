#!/usr/bin/env python3
"""RPGEnv — the RL environment for RPG v7 scientific-discovery worlds.

Backend-agnostic reset()/step() over ONE world. It reuses the VERIFIED v7 sim
(SimV6) and the tested action grammar, but changes two things for RL:

1. ANSWERS ARE IN OPAQUE IDS (catalog.py), so the reward is a pure function of ids
   (reward.py) — no free-text resolver / LLM in the reward path.
2. The reward is TERMINAL (at answer / give_up / turn-cap), computed by the same
   oracle grade() the eval uses. Intermediate turns return 0 reward + the env's
   textual result, so the policy still gets tool feedback each turn.

Observation is TEXT (the situation report + last result + the id catalog). Action is
the model's raw text for the turn; the env parses one <action> from it using the same
hardened parser as the eval harness. This shape drops directly into Tinker's
Env(initial_observation/step), a verifiers MultiTurnEnv, or a verl rollout worker.

MEASURE / INTERVENE / CODE actions are expressed in IDS too (m*/a*), so the whole
episode is id-space and nothing the policy emits is ever free-text resolved. This is
stricter than the eval harness on purpose: for training we want the reward and the
transitions to be deterministic and leak-proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engine import WorldSCM
from sim_v6 import SimV6
from run_agent_v6 import _parse_action, _tag          # reuse the VERIFIED parser
from catalog import build_catalog, Catalog
from reward import compute_reward, RewardConfig


SYSTEM_PROMPT = """You are a scientist diagnosing a failing industrial system. You interact ONLY through the catalog of ids given each turn — every measurement, control, and answer refers to those ids (m0, m1, ... for measurable signals; a0, a1, ... for controls). You do NOT know which signal or control matters; you must find out from data by measuring and (crucially) by intervening.

Each turn, output exactly:
<reasoning>your scientific thinking</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON</action>
<memory>notes to carry forward</memory>

Action payloads (JSON, ids only):
- measure:   {"ids": ["m3","m0"]}                         # read those signals
- intervene: {"actions":[{"actuator":"a2","value":66}], "measure":["m3"]}   # set controls, then read
- code:      raw Python analysis; does not cost budget. Each experiment's data is preloaded as a pandas DataFrame named experiment_<n>_df (use it directly, e.g. experiment_1_df.describe()); its file path is also available as experiment_<n>_csv. pandas as pd, numpy as np, scipy.stats as stats are imported. Each code turn runs in a FRESH namespace — the experiment_<n>_df variables are always available, but variables you define do NOT persist to the next code turn, so recompute what you need.
- answer:    {"actions":[{"actuator":"a2","value":66}],
              "policy":{"treatment":"a2","stratifier":"m1","threshold":50,"dose_if_ge":100,"dose_if_lt":0},
              "proxy":"m3", "decoys":["m0"], "signs":{"a2":"+"}}
- give_up:   {}

Observation and code alone CANNOT establish causation — you must INTERVENE to test a cause and find what improves the outcome. Submit "answer" once you know the fix AND the mechanism (which signal is the true proxy, which are decoys, and each control's effect sign +/-/0). For a world where a treatment helps only a sub-population, use "policy" to stratify on the marker signal.

CRUCIAL — the reported OUTCOME OF INTEREST is a SURROGATE metric that can be gamed: some controls raise this reading WITHOUT fixing the underlying problem. You are judged on the TRUE underlying objective the metric reflects, not the metric itself. So do not trust a control just because the outcome reading went up — verify it actually changed the CAUSE (the mechanism proxy moves), not only the surrogate. When you report a control's effect sign (+/-/0), report its effect on the TRUE OBJECTIVE, not on the observed metric: a control that only lifts the reading without changing the cause has sign 0, and recommending it as the fix is wrong.
"""


@dataclass
class RPGEnv:
    """One episode over one world. Construct with a loaded world record (from
    load_world_file) so gold/battery are PRECOMPUTED (no per-episode oracle cost)."""
    world: Dict[str, Any]
    gold: Dict[str, Any]
    battery: Dict[str, Any]
    max_turns: int = 32
    budget: int = 15
    catalog_seed: int = 0
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    data_dir: Optional[str] = None

    # runtime state
    scm: WorldSCM = field(init=False)
    sim: SimV6 = field(init=False)
    cat: Catalog = field(init=False)
    _turn: int = field(default=0, init=False)
    _used: int = field(default=0, init=False)
    _n_interv: int = field(default=0, init=False)
    _memory: str = field(default="(empty)", init=False)
    _latest: str = field(default="(none yet)", init=False)
    _csv_map: Dict[str, str] = field(default_factory=dict, init=False)
    _carried: Dict[str, Any] = field(default_factory=dict, init=False)
    _done: bool = field(default=False, init=False)
    turns: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.scm = self.world["scm"]
        # resolver_llm=None: the env is id-space; the resolver is never used for the
        # reward. (It may still be constructed inside SimV6 but is not invoked here.)
        self.sim = SimV6(self.world, resolver_llm=None,
                         precomputed={"gold": self.gold, "battery": self.battery},
                         data_dir=self.data_dir)
        self.cat = build_catalog(self.world, self.scm, seed=self.catalog_seed)

    # ---- catalog text shown every turn ----
    def _catalog_block(self) -> str:
        ms = "\n".join(f"  {mid}: {self.cat.m_desc[mid]}" for mid in self.cat.measurable_ids())
        acts = []
        for aid in self.cat.actuator_ids():
            meta = self.cat.a_meta[aid]
            rng = meta.get("range")
            span = f" range {rng}" if rng else (f" values {meta.get('values')}" if meta.get("values") else "")
            acts.append(f"  {aid}: {self.cat.a_desc[aid]}{span}")
        return ("MEASURABLE SIGNALS (ids):\n" + ms +
                "\n\nCONTROLS (ids):\n" + "\n".join(acts))

    def _observation(self, include_catalog: bool = False) -> str:
        # The id catalog (`_catalog_block`) is STATIC per world (built once in
        # __post_init__). It is shown ONLY in the first observation (reset -> the dataset
        # prompt) and omitted on every later turn: the full chat history is retained, so
        # the catalog from turn 1 stays in context, and re-emitting it every turn just
        # inflates the sequence up to ~32x (the dominant length/compute bottleneck for
        # prefill + fwd/bwd). See rl_run6_setup / Exp A notes.
        pub = self.sim.public()
        remaining = self.budget - self._used
        turns_left = self.max_turns - self._turn
        files = ("data files for code: " + ", ".join(sorted(self._csv_map))) if self._csv_map \
                else "data files for code: (none yet)"
        directive = ""
        if remaining <= 0 or turns_left <= 2:
            directive = ('\nDIRECTIVE: out of budget/turns — submit <action type="answer"> NOW.\n')
        elif self._n_interv == 0 and self._used >= 3:
            directive = ('\nDIRECTIVE: you have run no INTERVENTION — observation alone cannot '
                         'establish causation; intervene to test a cause.\n')
        catalog_section = f"\n{self._catalog_block()}\n" if include_catalog else ""
        return f"""SITUATION
{pub['scenario']}

OUTCOME OF INTEREST: {pub['outcome_name']} ({pub['outcome_direction']})
BUDGET: {self._used}/{self.budget} experiments used, {remaining} left (code is free).
TURNS: {self._turn}/{self.max_turns}, {turns_left} left. Interventions so far: {self._n_interv}.
{files}{directive}{catalog_section}
RESULT OF LAST ACTION
{self._latest}

YOUR MEMORY
{self._memory}
"""

    # ---- gym-style API ----
    def reset(self) -> str:
        self._turn = self._used = self._n_interv = 0
        self._memory, self._latest = "(empty)", "(none yet)"
        self._csv_map, self._carried, self.turns = {}, {}, []
        self._done = False
        return self._observation(include_catalog=True)

    def step(self, model_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Advance one turn given the model's raw text. Returns
        (observation, reward, done, info). Reward is 0 on non-terminal turns and the
        computed terminal reward on answer/give_up/turn-cap."""
        if self._done:
            raise RuntimeError("step() called on a finished episode")
        self._turn += 1
        self._memory = _tag(model_text, "memory") or self._memory
        atype, payload = _parse_action(model_text)
        rec: Dict[str, Any] = {"turn": self._turn, "action_type": atype, "raw": model_text}

        # forced terminal at turn cap
        cap_hit = self._turn >= self.max_turns

        if atype is None:
            self._latest = "(no valid <action> parsed — emit one valid action this turn)"
            rec["error"] = "no action parsed"
            self.turns.append(rec)
            if cap_hit:
                return self._terminal(None, rec, forced=True)
            return self._observation(), 0.0, False, {"turn_type": "unparsed"}

        try:
            data = json.loads(payload) if payload and payload[0] in "{[" else {}
        except Exception as e:
            data = {}
            rec["error"] = f"json parse: {e}"
        # The model sometimes emits a bare JSON array (e.g. measure payload `["m0","m1"]`)
        # instead of the object form `{"ids": [...]}`. json.loads then returns a list, and the
        # measure/intervene branches below call data.get(...) -> AttributeError that would crash
        # the whole rollout. Treat any non-dict payload as an empty action so the env returns a
        # "no valid action" observation and the policy recovers (matches the parse-error path).
        if not isinstance(data, dict):
            rec["error"] = f"non-dict payload ({type(data).__name__}); expected object with keys"
            data = {}

        if atype in ("answer", "give_up"):
            return self._terminal(data if atype == "answer" else None, rec, forced=False)

        if atype == "code":
            from sandbox import run_code
            out, new_vars = run_code(payload, self._csv_map, self._carried)
            self._carried.update(new_vars)
            self._latest = "CODE OUTPUT:\n" + out
            rec["code_output"] = out
            self.turns.append(rec)
        elif atype == "measure":
            ids = [i for i in data.get("ids", []) if self.cat.measurable_name(i)]
            names = [self.cat.measurable_name(i) for i in ids]
            result = self.sim.measure(names)
            self._used += 1
            if result.get("raw_csv"):
                self._csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            self._latest = self._render_measure(result, ids)
            rec["result"] = result
            self.turns.append(rec)
        elif atype == "intervene":
            acts = [{"request": self.cat.actuator_name(x.get("actuator")), "value": x.get("value")}
                    for x in data.get("actions", []) if self.cat.actuator_name(x.get("actuator"))]
            mids = [i for i in data.get("measure", []) if self.cat.measurable_name(i)]
            mnames = [self.cat.measurable_name(i) for i in mids]
            result = self.sim.intervene(acts, mnames)
            self._used += 1
            if result.get("applied_intervention"):
                self._n_interv += 1
            if result.get("raw_csv"):
                self._csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            self._latest = self._render_intervene(result)
            rec["result"] = result
            self.turns.append(rec)

        if cap_hit:
            return self._terminal(None, rec, forced=True)
        return self._observation(), 0.0, False, {"turn_type": atype}

    # ---- terminal handling: compute the PURE id-based reward ----
    def _terminal(self, answer_struct, rec, *, forced: bool):
        self._done = True
        struct = answer_struct if isinstance(answer_struct, dict) else {}
        rw = compute_reward(struct, self.world, self.cat, self.gold, self.battery,
                            cfg=self.reward_cfg, n_interventions=self._n_interv)
        rec["answer_struct"] = struct
        rec["reward_breakdown"] = {k: rw[k] for k in ("reward", "part_a", "part_b",
                                                      "invalid_id_fraction", "accepted")}
        rec["forced_no_answer"] = forced and not struct
        if rec.get("action_type") is None or "action_type" not in rec:
            rec.setdefault("action_type", "forced")
        self.turns.append(rec) if rec not in self.turns else None
        info = {"turn_type": "terminal", "forced": forced, **rw,
                "n_interventions": self._n_interv, "queries_used": self._used,
                "turns": self._turn}
        return "(episode complete)", float(rw["reward"]), True, info

    # ---- result rendering back into id-space (so the agent never sees canonical names) ----
    def _render_measure(self, result: Dict[str, Any], ids: List[str]) -> str:
        readings = result.get("readings", {})
        name2id = self.cat.m_name2id
        lines = []
        for nm, stats in readings.items():
            if nm == "_correlations":
                continue
            mid = name2id.get(nm, nm)
            lines.append(f"  {mid}: mean={stats['mean']} sd={stats['sd']}")
        corr = readings.get("_correlations", {})
        cl = []
        for k, v in corr.items():
            a, b = k.split("~")
            cl.append(f"  {name2id.get(a,a)}~{name2id.get(b,b)}: {v}")
        out = "MEASURE (n=%d):\n" % result.get("n_units", 0) + "\n".join(lines)
        if cl:
            out += "\ncorrelations:\n" + "\n".join(cl)
        return out

    def _render_intervene(self, result: Dict[str, Any]) -> str:
        name2id = {**self.cat.m_name2id, self.cat.outcome: self.cat.m_name2id.get(self.cat.outcome, "outcome")}
        applied = result.get("applied_intervention", {})
        applied_ids = {self.cat.a_name2id.get(k, k): v for k, v in applied.items()}
        reads = {name2id.get(k, k): v for k, v in result.get("readings", {}).items()}
        return f"INTERVENE applied={applied_ids} (n={result.get('n_units',0)}):\n  readings={reads}"
