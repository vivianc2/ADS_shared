"""Scientist agent for static RPG worlds.

This is deliberately simpler than the coder agents: the model chooses JSON
sampling queries and eventually returns one structured JSON answer. The heavy
lifting is done by the simulator and by concise statistical summaries of each
returned CSV.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from schemas_rpg import StaticRPGQuestion, StaticRPGQueryResult, StaticRPGWorldInfo

logger = logging.getLogger(__name__)


_ACTION_RE = re.compile(
    r'<action\s+type="(query|answer|give_up)">\s*(.*?)\s*</action>',
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ScientistAgentRPG:
    """LLM scientist for static partially observed RPG discovery tasks."""

    llm: Any
    world_info: Optional[StaticRPGWorldInfo] = field(default=None, init=False)
    question: Optional[StaticRPGQuestion] = field(default=None, init=False)
    max_queries: int = field(default=8, init=False)

    # Number of most-recent queries whose full data summaries are re-shown to the
    # agent each turn. Older query data is NOT reshown (only a one-line status),
    # so the agent must record any numbers it still needs into scientist_memory.
    # Not annotated -> treated as a class constant, not a dataclass field.
    _RECENT_DATA_WINDOW = 3

    _query_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _queries_made: int = field(default=0, init=False)
    _scientist_memory: str = field(default="", init=False)
    _system_messages: List[str] = field(default_factory=list, init=False)

    def initialize(
        self,
        world_info: StaticRPGWorldInfo,
        question: StaticRPGQuestion,
        max_queries: int,
    ) -> None:
        self.world_info = world_info
        self.question = question
        self.max_queries = max_queries
        self._query_history = []
        self._queries_made = 0
        self._scientist_memory = ""
        self._system_messages = []
        logger.info("ScientistAgentRPG initialized for %s", world_info.world_id)

    def receive_system_message(self, message: str) -> None:
        self._system_messages.append(message)
        logger.info("RPG scientist received system message: %s", message[:160])

    def get_next_action(self) -> Dict[str, Any]:
        if self.world_info is None or self.question is None:
            raise RuntimeError("ScientistAgentRPG not initialized")
        response = self.llm.generate(
            self._system_prompt(),
            self._user_prompt(),
            max_new_tokens=2500,
        )
        logger.debug("RPG scientist raw response:\n%s", response)

        reasoning = self._extract_tag(response, "reasoning")
        self._update_memory(response)
        action = self._parse_action(response)
        action["raw_response"] = response
        action["reasoning"] = reasoning
        action["scientist_memory"] = self._scientist_memory
        logger.info("RPG scientist action=%s content=%s", action["type"], action["content"][:180])
        return action

    def receive_result(self, result: StaticRPGQueryResult) -> None:
        if result.success:
            self._queries_made += 1
        data_summary = ""
        if result.success and result.data_file:
            data_summary = self._compute_data_summary(result.data_file, result.query.intervention)
            logger.info("RPG data summary:\n%s", data_summary)
        self._query_history.append({
            "query": result.query.raw_query,
            "query_dict": result.query.to_dict(),
            "success": result.success,
            "result_xml": result.to_xml(),
            "data_file": result.data_file,
            "n_rows": result.n_rows,
            "sample_cells": result.sample_cells,
            "intervention": dict(result.query.intervention),
            "measurements": result.query.measurements,
            "data_summary": data_summary,
            "scientist_memory_snapshot": self._scientist_memory,
        })
        logger.info("RPG scientist received result success=%s rows=%s", result.success, result.n_rows)

    def _system_prompt(self) -> str:
        return """You are a careful empirical scientist studying a partially observed world.

You can request small data samples and then submit a structured discovery answer.
You are shown only public measurements and public available actions. Some
measurements are noisy; some are downstream clues of unobserved real-world
states. The true explanation may be a hidden object, process, failure mode,
diagnosis, or environmental state that is never listed as a variable.

Do not assume a variable name proves a mechanism. Treat names and story context
as clues for inventing hypotheses about hidden causes. A visible predictor is
not necessarily the cause: for example, leaf fall may be evidence for a hidden
clogged drainage path rather than the final explanation. Prefer focused tests
that distinguish competing hidden explanations. Use exact measurement names and
exact action values from the catalogs.

Available actions:
1. Query data with exactly one JSON object:
   <action type="query">{"mode":"observational_sample","n_units":200,"measurements":["A","B"]}</action>
   <action type="query">{"mode":"interventional_sample","intervention":{"Knob":"value"},"n_units":200,"measurements":["Outcome","Proxy"]}</action>
   Interventional samples may set a joint combination of public actions up to max_intervention_knobs, for example {"intervention":{"ActionA":"on","ActionB":"on"}}.
   <action type="query">{"mode":"inspect_unit","case_seed":17,"measurements":["A","B"]}</action>
2. Submit the final answer as JSON matching the answer schema:
   <action type="answer">{"intervention":{"Knob":"value"},"hypothesis":"..."}</action>
   For answer_schema=latent_cause_hypothesis:
   <action type="answer">{"latent_hypothesis":{"name":"ordinary-language hidden cause","description":"...","confidence":0.7},"evidence":["..."],"alternatives_ruled_out":["..."],"decisive_test":"...","action_plan":{"do_now":["ActionName"],"avoid":["ActionName"],"why":"..."}}</action>
   For answer_schema=conditional_policy:
   <action type="answer">{"policy":{"branch_variable":"Screen","branch_threshold":50,"if_above":{"KnobA":"on"},"if_below":{"KnobB":"on"}},"hypothesis":"..."}</action>
   For answer_schema=latent_regime_policy:
   <action type="answer">{"latent_structure":{"n_regimes":2,"evidence":"..."},"policy":{"branch_variable":"PanelA","branch_threshold":50,"if_above":{"KnobB":"on"},"if_below":{"KnobA":"on"}},"hypothesis":"..."}</action>
   For answer_schema=anomaly_identification:
   <action type="answer">{"flagged_unit_ids":[1,2],"anomaly_rule":"FeatureA > 70 AND FeatureB < 30","hypothesis":"..."}</action>
3. Give up only if the task is impossible:
   <action type="give_up">brief reason</action>

Good strategy:
- Start by identifying what hidden cause would make the story and observations cohere.
- Use observational data to form competing hypotheses, but do not treat correlation as proof.
- Ask whether an observed variable is a cause, a clue, a downstream symptom, or a confounder.
- Use targeted interventional samples as tests of a mechanism, not just as arm-ranking.
- For latent-cause questions, name an unobserved cause in ordinary language, cite evidence, rule out alternatives, and propose a decisive test.
- For conditional policies, estimate which observed screen separates response patterns.
- For latent-regime questions, ask whether one distribution is enough: look for bimodality, clusters, threshold behavior, and treatment-response reversals by a measured proxy.
- For anomaly identification, build a simple threshold rule from observed feature signatures.
- Cross-check with a second outcome/proxy when budget permits.
- Keep queries within the measurement and unit budget.

Working memory and evidence discipline (read carefully):
- Only your most recent 3 query results are shown back to you in full under
  "RECENT QUERY RESULTS". Older query data is NOT reshown; it is reduced to a
  one-line status. Therefore, copy every number you may still need later
  (means, sds, correlations, mean shifts, value counts) into <scientist_memory>
  under "Known:". Cite real figures from the data, never invent or round from
  guesswork. If you cannot see a number you need, it is in an earlier query and
  should already be in your memory — do not re-run the query to "re-see" it.
- Never repeat a query you have already run. An identical query (same mode, same
  measurements, same intervention) returns no new information and the system
  will reject it as a duplicate, wasting a turn. Each new query must measure
  something new or test a new intervention.
- For mechanism, latent-cause, and latent-regime questions, run at least one
  interventional_sample that sets a candidate action and measures the mechanism
  and outcome before you answer. Observational correlation alone is not proof of
  a hidden cause.
- Budget discipline: watch sample_units_remaining and cell_budget. When the cell
  budget is low, shrink n_units or submit your answer instead of overspending; a
  query that exceeds the remaining cell budget is rejected and wastes a turn.

Output exactly these three blocks in order:
<reasoning>Concise analysis and why the next action is appropriate.</reasoning>
<action type="query|answer|give_up">JSON query, JSON answer, or brief reason.</action>
<scientist_memory>
Tested:
Known:
Uncertain:
Next:
</scientist_memory>"""

    def _user_prompt(self) -> str:
        assert self.world_info is not None and self.question is not None
        remaining = self.max_queries - self._queries_made
        budget = self.world_info.experiment_budget
        max_cells = budget.get("max_total_samples")
        cells_used = sum(
            int(h.get("sample_cells") or 0) for h in self._query_history if h.get("success")
        )
        budget_lines = [
            f"query_budget: {self._queries_made}/{self.max_queries} successful queries used; {remaining} left",
            f"cell_budget: max_total_samples={max_cells}, "
            f"max_samples_per_query={budget.get('max_samples_per_query')}",
            f"per_query_caps: max_units_per_query={budget.get('max_units_per_query')}, "
            f"max_measurements_per_query={budget.get('max_measurements_per_query')}",
            "cell_cost_rule: approximate cells = n_units * "
            "(2 id/mode columns + number_of_measurements + number_of_intervention_knobs); "
            "late in the run, reduce n_units or answer instead of overspending.",
        ]
        if isinstance(max_cells, (int, float)) and max_cells > 0:
            cells_left = max(0, int(max_cells) - cells_used)
            budget_lines.insert(
                1, f"cell_budget_used: {cells_used}/{int(max_cells)} cells used; {cells_left} cells left"
            )
            if cells_left <= 0.15 * float(max_cells):
                budget_lines.append(
                    "CELL BUDGET LOW: keep n_units small (or submit your answer now); "
                    "a query that exceeds the remaining cell budget is rejected and wastes a turn."
                )
        if remaining <= 0:
            budget_lines.append("NO QUERIES LEFT: answer now from existing evidence.")
        if self._system_messages:
            budget_lines.append("SYSTEM MESSAGES:")
            budget_lines.extend(f"- {m}" for m in self._system_messages[-4:])

        recent_section = self._recent_results_section()
        history_section = self._history_section()
        memory = self._scientist_memory or "(empty)"

        return f"""QUESTION
{self.question.question_text}

ANSWER SCHEMA
{self.world_info.answer_schema}
max_intervention_knobs={self.world_info.max_intervention_knobs}

BUDGET
{chr(10).join(budget_lines)}

STORY
{self.world_info.story}

OBSERVED MEASUREMENTS
{self.world_info.get_measurement_catalog()}

AVAILABLE ACTIONS
{self.world_info.get_intervention_catalog()}

ALLOWED QUERY MODES
{", ".join(self.world_info.allowed_query_modes)}

RECENT QUERY RESULTS (last {self._RECENT_DATA_WINDOW}, full data; copy numbers you still need into memory)
{recent_section}

CURRENT MEMORY
{memory}

EARLIER QUERIES (data not reshown — use your memory for these)
{history_section}

Now choose the next action. If the current evidence is enough, answer with JSON."""

    def _recent_results_section(self) -> str:
        """Full data summaries for the last _RECENT_DATA_WINDOW queries.

        Earlier degeneration came from re-showing only the single most recent
        query, which forced the agent to either transcribe every number into
        memory or lose it. Showing a small rolling window keeps recent evidence
        visible across turns without unbounded prompt growth.
        """
        if not self._query_history:
            return "No data collected yet."
        window = self._query_history[-self._RECENT_DATA_WINDOW:]
        start_idx = len(self._query_history) - len(window) + 1
        blocks: List[str] = []
        for offset, item in enumerate(window):
            idx = start_idx + offset
            is_latest = item is self._query_history[-1]
            tag = "LATEST" if is_latest else f"query #{idx}"
            if not item["success"]:
                blocks.append(
                    f"[{tag}] FAILED request {item['query']}\n"
                    + (item.get("result_xml", "") or "")
                )
                continue
            lines = [
                f"[{tag}] request {item['query']}",
                f"rows={item['n_rows']} cells={item['sample_cells']}",
                item.get("data_summary", ""),
            ]
            blocks.append("\n".join(line for line in lines if line))
        return "\n\n".join(blocks)

    def _history_section(self) -> str:
        # Only the queries older than the re-shown window; the recent ones already
        # appear in full under RECENT QUERY RESULTS.
        older = self._query_history[:-self._RECENT_DATA_WINDOW]
        if not older:
            return "(none)"
        lines = []
        for idx, item in enumerate(older, 1):
            status = "OK" if item["success"] else "FAIL"
            mode = item.get("query_dict", {}).get("mode")
            intervention = item.get("intervention") or {}
            intv_text = f" do={json.dumps(intervention, sort_keys=True)}" if intervention else ""
            measurements = item.get("measurements") or []
            meas_text = f" meas={','.join(measurements)}" if measurements else ""
            lines.append(f"{idx}. {status} {mode}{intv_text}{meas_text} rows={item.get('n_rows')}")
        return "\n".join(lines)

    def _compute_data_summary(self, data_file: str, intervention: Dict[str, Any]) -> str:
        try:
            df = pd.read_csv(data_file)
        except Exception as exc:
            return f"(could not read CSV: {exc})"
        if df.empty:
            return "(empty data)"

        sections: List[str] = [f"DATAFRAME rows={len(df)} columns={list(df.columns)}"]
        if intervention:
            sections.append("Intervention: do(" + ", ".join(f"{k}={v}" for k, v in intervention.items()) + ")")

        numeric_cols = [
            c for c in df.columns
            if c not in {"unit_id", "case_seed"} and pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.columns
            if c not in {"unit_id", "case_seed"} and not pd.api.types.is_numeric_dtype(df[c])
        ]

        if numeric_cols:
            lines = ["Numeric summaries:"]
            for col in numeric_cols[:12]:
                s = df[col].dropna()
                if s.empty:
                    continue
                q10 = s.quantile(0.10)
                q25 = s.quantile(0.25)
                q75 = s.quantile(0.75)
                q90 = s.quantile(0.90)
                iqr = q75 - q25
                lines.append(
                    f"  {col}: mean={s.mean():.2f}, sd={s.std(ddof=1):.2f}, "
                    f"p10={q10:.2f}, p25={q25:.2f}, median={s.median():.2f}, "
                    f"p75={q75:.2f}, p90={q90:.2f}, iqr={iqr:.2f}"
                )
            sections.append("\n".join(lines))

            shape_lines = ["Distribution-shape cues:"]
            for col in numeric_cols[:12]:
                s = df[col].dropna()
                if len(s) < 40:
                    continue
                q10 = float(s.quantile(0.10))
                q25 = float(s.quantile(0.25))
                q50 = float(s.quantile(0.50))
                q75 = float(s.quantile(0.75))
                q90 = float(s.quantile(0.90))
                lower_gap = q50 - q25
                upper_gap = q75 - q50
                tail_gap = (q90 - q75) + (q25 - q10)
                center_gap = max(q75 - q25, 1e-9)
                cues = []
                if max(lower_gap, upper_gap) > 1.8 * max(min(lower_gap, upper_gap), 1e-9):
                    cues.append("asymmetric quartile spacing")
                if tail_gap > 1.4 * center_gap:
                    cues.append("wide tails vs center")
                if abs(float(s.mean()) - q50) > 0.25 * max(float(s.std(ddof=1)), 1e-9):
                    cues.append("mean-median separation")
                if cues:
                    shape_lines.append(f"  {col}: " + "; ".join(cues))
            if len(shape_lines) > 1:
                sections.append("\n".join(shape_lines))

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True).abs()
            pairs = []
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1:]:
                    val = corr.loc[a, b]
                    if pd.notna(val):
                        signed = df[[a, b]].corr(numeric_only=True).loc[a, b]
                        pairs.append((abs(float(val)), float(signed), a, b))
            pairs.sort(reverse=True)
            if pairs:
                lines = ["Strongest numeric correlations in this sample:"]
                for _, signed, a, b in pairs[:8]:
                    lines.append(f"  corr({a}, {b})={signed:.3f}")
                sections.append("\n".join(lines))

        if categorical_cols:
            lines = ["Categorical/value counts:"]
            for col in categorical_cols[:8]:
                counts = df[col].value_counts(normalize=True).head(8)
                text = ", ".join(f"{k}: {v:.0%}" for k, v in counts.items())
                lines.append(f"  {col}: {text}")
            sections.append("\n".join(lines))

        if self._query_history and intervention and numeric_cols:
            baseline = self._latest_observational_dataframe()
            if baseline is not None:
                common = [c for c in numeric_cols if c in baseline.columns and pd.api.types.is_numeric_dtype(baseline[c])]
                if common:
                    lines = ["Mean shifts vs most recent observational sample:"]
                    for col in common[:12]:
                        delta = df[col].mean() - baseline[col].mean()
                        lines.append(
                            f"  {col}: obs_mean={baseline[col].mean():.2f}, "
                            f"intv_mean={df[col].mean():.2f}, delta={delta:+.2f}"
                        )
                    sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _latest_observational_dataframe(self) -> Optional[pd.DataFrame]:
        for item in reversed(self._query_history):
            q = item.get("query_dict", {})
            if item.get("success") and q.get("mode") == "observational_sample" and item.get("data_file"):
                try:
                    return pd.read_csv(item["data_file"])
                except Exception:
                    return None
        return None

    def _parse_action(self, response: str) -> Dict[str, Any]:
        clean = self._strip_think_block(response)
        match = _ACTION_RE.search(clean)
        if not match:
            return {
                "type": "give_up",
                "content": "Parsing failed: no valid <action> block found.",
            }
        action_type = match.group(1).lower()
        content = match.group(2).strip()
        if action_type in {"query", "answer"}:
            try:
                self._extract_json_object(content)
            except Exception as exc:
                return {
                    "type": "give_up",
                    "content": f"Parsing failed: {action_type} content is not valid JSON: {exc}",
                }
        return {"type": action_type, "content": content}

    def _extract_json_object(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        if not (text.startswith("{") and text.endswith("}")):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("no JSON object found")
            text = match.group(0)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON content must be an object")
        return data

    def _extract_tag(self, response: str, tag: str) -> str:
        clean = self._strip_think_block(response)
        match = re.search(rf"<{tag}>(.*?)</{tag}>", clean, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _update_memory(self, response: str) -> None:
        memory = self._extract_tag(response, "scientist_memory")
        if memory:
            self._scientist_memory = memory

    def _strip_think_block(self, response: str) -> str:
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
