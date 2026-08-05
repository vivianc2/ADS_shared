# Causal Discovery Benchmark — Architecture

> Current ACED-Bench architecture update, May 6, 2026. This section reflects the
> paper-facing pipeline for `paper_polished.tex`. Older sections later in this
> file describe useful mechanics, but some names, datasets, and result paths are
> historical.

## Current Pipeline

```text
dataset_generation_code/all_out_bn/* world JSON
  -> framework_code/json_converter.py and framework_code/simulator.py
  -> framework_code/world_model_causal.py fixed parser
  -> framework_code/orchestrator.py plus scientist agents
  -> framework_code/results/<run_name>/agent_logs/*.json
  -> framework_code/evaluations/{for_paper,round,samp}/eval_*.json
  -> analysis/evidence_ledger_current/merged/per_trajectory_metrics.json
  -> framework_code/notebooks/plot_for_neurips.ipynb
  -> paper_polished.tex
```

Current public benchmark names are `ACED-Struct` and `ACED-Decision`. Legacy
code and figure filenames may still use `basic`, `advanced`, `PGM`, or older
dataset names.

## Active Artifacts

- Structural worlds: `dataset_generation_code/all_out_bn/out_bn_4_19_big*`.
  The paper uses a 30-world / 180-question structural subset.
- Decision worlds: `dataset_generation_code/all_out_bn/out_bn_adv_v3*`.
  The resource-limited runs use the balanced 60-question
  `out_bn_adv_v3_60_balance` family; unrestricted runs may still be historical
  48-row runs, 54-row changed24 hybrids, or 60-row reruns.
- For-paper evaluations:
  `framework_code/evaluations/for_paper/eval_*.json`.
- Resource-limited evaluations:
  `framework_code/evaluations/round/eval_*.json` and
  `framework_code/evaluations/samp/eval_*.json`.
- Raw logs: `framework_code/results/<run_name>/agent_logs/*.json`.
- Previous evidence ledger:
  `analysis/evidence_ledger_v2_merged/per_trajectory_metrics.json`.
- Current sharded ledger launcher:
  `framework_code/run_evidence_ledger_current_parallel.sh`, which writes
  merged outputs to `analysis/evidence_ledger_current/merged/` by default.
- Figure notebook:
  `framework_code/notebooks/plot_for_neurips.ipynb`.

## Current Components

- `json_converter.py` converts generated world JSONs into the simulator format.
- `simulator.py` exposes exact probabilistic queries and do-operator
  interventions over the hidden Bayesian network.
- `world_model_causal.py` is the fixed natural-language parser. The paper
  specifies Claude Opus-4.8 through AWS Bedrock,
  `us.anthropic.claude-opus-4-7`, temperature `0.1`, max output `512`.
- `orchestrator.py` owns the interactive loop, query budget, parser calls,
  simulator calls, and log writing.
- `scientist_agent_causal.py` is the conversational scientist-agent path.
- `scientist_coder_agent.py` is the single-prompt active-code path.
- `scientist_coder_agent_new.py` is the modular active-code path used for many
  current for-paper runs.
- `run_agent_batch.py` supports explicit resource variants:
  `--limit-variant rounds` for `round4` and `--limit-variant samples` for
  `samp5k`. Resource usage is logged per experiment in raw result JSONs.
- Evaluation JSONs in `framework_code/evaluations/for_paper/` expose a
  `scores` object with fields such as `overall`, `by_category`,
  `by_archetype`, and `by_n_nodes`.
- Evidence-ledger analysis is implemented by
  `framework_code/evidence_ledger_analysis.py`. Use
  `run_evidence_ledger_current_parallel.sh` for current large runs because it
  launches many hash shards, tracks progress, and merges shard outputs.

## Protocol Invariants

- The active query budget is 10 successful data queries. Rejected parser
  attempts do not count as successful data queries.
- Resource-limited variants are separate ablations: `round4` caps scientist
  turns at 4, while `samp5k` caps total sample rows per question at 5000.
  They should not be mixed with unrestricted logs under one run name.
- Parser failures, invalid variables, invalid states, and malformed
  interventions should be rejected before simulator execution.
- Decision scoring uses expected-state-index units: the explicit ordered state
  list in the world file, zero-based indices, and lower outcome indices are
  better unless a world says otherwise.
- Decision tie/safety tolerance is `0.05`; subgroup effect threshold is `0.15`.
- Structural validation dependency threshold is total variation epsilon `0.02`;
  CPD resampling attempts are capped at 64.
- D-separated variables are conditionally independent by graph structure.
  Exact-inference checks over sampled CPDs are sanity checks or d-connected
  dependency screens, not the source of the d-separation guarantee.
- Decision generator validation constants in the paper include: safety target
  improvement `0.20`, protected-group worsening at most `0.08`, mediator total
  effect `0.20`, residual direct effect below `0.10`, mediator-path support
  `0.18`, runner-up gap `0.20`, satisficing clean gap `0.10`, subgroup minimum
  improvement `0.10`, average-best policy at least `0.05` worse in subgroup,
  invalid-premise association `0.20`, and invalid-premise valid alternative
  effect `0.20`.

## Known Stale Areas

- `plot_for_neurips.ipynb` may have stale GPT-OSS decision overrides and old
  dataset-size labels. Trust the latest eval JSON for inline numbers until the
  notebook is updated and rerun.
- `eval_oss120_coder_new_adv_v3.json` is newer than the notebook and reports
  GPT-OSS-120B modular ACED-Decision as `45/60 = 0.750`.
- `oss120_coder_new_4_19_big` is unfinished; structural GPT-OSS values in the
  paper are placeholders.
- Older datasets such as `out_bn_3_4` and `out_bn_advanced_*` are not the
  current paper benchmark unless a task explicitly references them.
- Some scripts may still contain old model IDs or local parser assumptions. The
  paper's reported parser/annotator model is Opus-4.8 through Bedrock.

## Useful Inspection Commands

```bash
jq '.scores | {overall, by_category, by_archetype, by_n_nodes}' \
  framework_code/evaluations/for_paper/eval_opus_coder_new_adv_v3.json

jq '.scores | {overall, by_category, by_archetype, by_n_nodes}' \
  framework_code/evaluations/for_paper/eval_oss120_coder_new_adv_v3.json

cd framework_code && ./run_evidence_ledger_current_parallel.sh manifest

rg -n "PGM-Struct|PGM-Decision|guess-shot|CCA|ERR|EDV|PSW" \
  paper_polished.tex framework_code dataset_generation_code
```

---

## Overview

This system implements a **causal discovery benchmark** where an LLM "Scientist" agent
must discover properties of a hidden causal graph by requesting data samples from a
Bayesian Network simulator.

The scientist sees variable names, descriptions, and a story — but **never** the graph
structure (edges, CPDs). It must request observational or interventional data, analyze
patterns, and answer causal questions.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          run_agent_batch.py                                 │
│                        (Primary Entry Point)                               │
│                                                                            │
│  Loads worlds, builds LLMs, iterates over questions, collects results      │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            orchestrator.py                                  │
│                    (Manages the Interaction Loop)                           │
│                                                                            │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │  Initialize │────────▶│  Main Loop  │────────▶│  Evaluate   │          │
│  │   Agents    │         │  (queries)  │         │   Answer    │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
          │                       │ ▲
          ▼                       ▼ │
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ scientist_agent  │    │  world_model     │    │     schemas      │
│ _causal.py       │◄──▶│  _causal.py      │    │       .py        │
│                  │    │                  │    │                  │
│ - Reasons about  │    │ - Parses NL      │    │ - ParsedQuery    │
│   causal struct  │    │ - Validates      │    │ - QueryResult    │
│ - Decides next   │    │ - Executes       │    │ - WorldInfo      │
│   query          │    │ - Formats output │    │ - Question       │
│ - Gives answer   │    │                  │    │ - Errors         │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   simulator.py   │
                        │                  │
                        │ - Loads BIF/BN   │
                        │ - Observational  │
                        │   sampling       │
                        │ - Interventional │
                        │   sampling (do)  │
                        └──────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  pgmpy library   │
                        │  (BayesianNetwork)│
                        └──────────────────┘
```

---

## File-by-File Breakdown

### 1. `schemas.py` — Data Structures & Contracts

Defines all shared data structures used for communication between components.
This is the API contract between all modules.

**Key Classes:**

```python
@dataclass
class ParsedQuery:
    query_type: QueryType          # OBSERVATIONAL or INTERVENTIONAL
    n_samples: int                 # How many samples
    variables: Optional[List[str]] # Which variables (None = all)
    interventions: Dict[str, str]  # e.g., {"smoke": "yes"}
    raw_query: str                 # Original NL query

@dataclass
class QueryResult:
    success: bool
    query: ParsedQuery
    data_file: Optional[str]       # Path to CSV with samples
    n_rows: int
    columns: List[str]
    preview: str                   # First few rows for quick inspection
    error_message: Optional[str]
    def to_xml(self) -> str        # Format for LLM consumption

@dataclass
class WorldInfo:
    story: str                     # Context/scenario description
    variables: List[VariableInfo]  # Variable names, descriptions, states
    non_intervenable_variables: Dict[str, str]  # Variables that can't be intervened on
    def to_xml(self) -> str

@dataclass
class Question:
    question_type: str             # "causal_effect", "all_causes_of", etc.
    question_text: str             # Human-readable question
    ground_truth: Any              # Correct answer for evaluation
    metadata: Dict[str, Any]       # id, difficulty, question_group, etc.
```

**Why XML?** LLMs parse structured tags more reliably than raw text. The `to_xml()`
methods format data in `<tag>content</tag>` style for clear boundaries.

---

### 2. `simulator.py` — Bayesian Network Engine

Wraps pgmpy to provide clean sampling operations. This is the "physics engine"
of the causal world — the source of ground truth.

**Key Methods:**

```python
class BNSimulator:
    @classmethod
    def from_bif(cls, bif_path: str) -> "BNSimulator":
        """Load a Bayesian Network from a BIF file."""

    def sample_observational(self, n: int, variables=None, seed=None) -> pd.DataFrame:
        """
        Draw samples from P(V) — the joint distribution.
        Passive observation: we just watch the world.
        """

    def sample_interventional(self, interventions: Dict, n: int, ...) -> pd.DataFrame:
        """
        Draw samples from P(V | do(X=x)) — the interventional distribution.

        Implements do-calculus by:
        1. Remove all edges INTO the intervened variable(s)
        2. Set each intervened variable to a constant (delta distribution)
        3. Sample from the mutilated graph

        This simulates what happens when we FORCE a variable to a value,
        breaking its natural causes.
        """

    # Introspection (for question generation, NOT shown to scientist)
    def get_nodes(self) -> List[str]
    def get_edges(self) -> List[Tuple[str, str]]
    def get_parents(self, var) -> List[str]
    def get_children(self, var) -> List[str]
```

**The do-operator** is the key to causal inference:
- `P(Y | X=x)` = "What's Y when we OBSERVE X=x?" (correlation)
- `P(Y | do(X=x))` = "What's Y when we FORCE X=x?" (causation)

Example: Observing that people carry umbrellas (X) correlates with rain (Y),
but FORCING people to carry umbrellas doesn't cause rain!

---

### 3. `json_converter.py` — World JSON to BIF

Converts the generated world JSON files (from `dataset_generation_code/`) into
BIF format that pgmpy can load. Also extracts the world config (variable descriptions,
story, questions, non-intervenable variables) for the framework to use.

```python
class JSONToBIFConverter:
    def __init__(self, json_path: str)
    def convert(self, bif_path: str)           # Write BIF file
    def get_world_config(self) -> dict          # Extract config for framework
```

---

### 4. `world_model_causal.py` — Natural Language Query Interface

Translates natural language queries from the scientist into structured BN operations.
This is the "interpreter" between the scientist and the simulator.

**Data Flow:**

```
Scientist's NL Query (e.g., "Give me 200 samples with do(smoke=yes)")
        │
        ▼
┌───────────────────────────────────────┐
│      _parse_query() [LLM Call]        │  Uses QwenLLM/OpenAILLM/BedrockLLM
│                                       │  to parse NL → JSON inside <json> tags
│  System: "You are a query parser..."  │
│  User: "VARIABLES: ... QUERY: ..."    │
│  Output: <json>{...}</json>           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│      _validate_query() [No LLM]       │  Deterministic checks:
│                                       │  - n_samples within limits
│  - Check variables exist in BN        │  - intervention states valid
│  - Check non-intervenable variables   │  - query consistency
│  - Check consistency                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│      _execute_query() [No LLM]        │  Calls simulator.sample_*()
│                                       │  Saves results to CSV file
└───────────────────────────────────────┘
        │
        ▼
    QueryResult (XML) → returned to scientist
```

**LLM backends** (all implement `generate(system, user, max_new_tokens)`):
- `QwenLLM`: Local HuggingFace model (default: `Qwen/Qwen2.5-7B-Instruct`)
- `OpenAILLM`: OpenAI-compatible API (works with OpenAI, vLLM, etc.)
- `BedrockLLM` (from `bedrock_llm.py`): AWS Bedrock Converse API

**Key design decisions:**
1. **Single LLM call for parsing**: The LLM only does NL→JSON translation.
   All validation is deterministic (no hallucination risk).
2. **Causal-specific prompts**: Explicit do-operator semantics in the prompt
   ("sever incoming causal links"), plus expanded intervention keyword list.
3. **File-based results**: Samples saved to CSV, path included in result.

---

### 5. `scientist_agent_causal.py` — The Reasoning Agent

An LLM agent that reasons about causal structure, decides what data to request,
and formulates answers. This is the primary agent used for all experiments.

**State Machine:**

```
                    ┌─────────────────┐
                    │   INITIALIZED   │
                    │                 │
                    │ Has: WorldInfo  │
                    │      Question   │
                    │      Budget     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
              ┌────▶│   THINKING      │◀────┐
              │     │                 │     │
              │     │ get_next_action │     │
              │     └────────┬────────┘     │
              │              │              │
              │   ┌──────────┼──────────┐   │
              │   ▼          ▼          ▼   │
         ┌────────────┐ ┌────────┐ ┌───────────┐
         │   QUERY    │ │ ANSWER │ │  GIVE UP  │
         │            │ │        │ │           │
         │ Request    │ │ Submit │ │ Can't     │
         │ more data  │ │ answer │ │ determine │
         └─────┬──────┘ └────────┘ └───────────┘
               │              │          │
               │              └──────────┴──────▶ DONE
               ▼
         ┌────────────┐
         │  RECEIVE   │
         │  RESULT    │────────────────────────┘
         │            │
         │ Update     │
         │ history +  │
         │ memory     │
         └────────────┘
```

**What the scientist sees each turn:**
- World info (variables, descriptions, states — NO edges!)
- The question to answer
- Full query history (what was asked, what was returned, with statistical summaries)
- Remaining budget
- Its own scientist memory (updated each turn)

**Output format:**
```xml
<action type="query">Give me 100 samples with do(smoke=yes)</action>
```
or
```xml
<action type="answer">Yes, there is a direct causal effect from smoke to lung</action>
```

**Key features:**
- **Scientist memory**: A `<scientist_memory>` block the agent updates each turn to track
  its evolving understanding (hypotheses, evidence, ruled-out alternatives)
- **Statistical summaries**: After receiving query results, the agent automatically
  computes chi-squared tests and distribution summaries from the CSV data
- **Budget awareness**: Prompt includes remaining queries with escalating urgency warnings

**Budget tracking:** The `_queries_made` counter increments only when `receive_result()`
is called with a successful result (aligned with the orchestrator's `_query_count`).
Failed queries (parse errors, validation errors) do not consume budget.

---

### 6. `scientist_coder_agent.py` — Agent with Python Execution

Extends the standard agent with a `<action type="code">` action that executes
Python code in a sandboxed namespace. Useful for complex statistical analysis.
One LLM call per outer turn; the LLM freely interleaves `code`, `query`,
`answer`, and `give_up` actions in its response.

**Inner loop per turn:**
1. LLM generates an action (code, query, answer, or give_up)
2. If `type="code"`: execute Python, capture stdout, feed back to LLM, repeat (up to 8 rounds)
3. If `type="query"`: return to orchestrator (costs budget)
4. If `type="answer"` or `type="give_up"`: end turn

**Execution sandbox:**
- Pre-loaded: `pd` (pandas), `np` (numpy), `stats` (scipy.stats), `chi2_contingency`
- `query_files` dict and `query_N_csv` variables for accessing data CSVs
- 30-second timeout per code execution
- stdout truncated to 3000 chars

---

### 6b. `scientist_coder_agent_new.py` — Modular Coder Agent

A re-architected coder agent that splits each turn into **four specialized
LLM calls** instead of one. Same orchestrator interface (`initialize` /
`get_next_action` / `receive_result`), but the prompt-engineering surface is
broken up to give each phase a focused instruction set.

**Phase structure:**

```
Turn 1 (no data yet):
    INIT      → <hypothesis>, <verification_criteria>, <experiment_plan>
    DESIGN    → <query>, <rationale>            ──▶ orchestrator (costs budget)

Turn N (new data arrived):
    CODE      ≤5 rounds of <code> ... </code>; ends with <analysis_ready/>
    ANALYSIS  → <evidence_summary>, <confidence>, <decision>=continue|answer|give_up,
                <memory_update>, optional <hypothesis_revision>, <answer> if decision=answer
    DESIGN    (only if decision=continue) → <query>, <rationale>
```

**Differences from `scientist_coder_agent.py`:**
- 5 inner code rounds (vs. 8), 45 s timeout (vs. 30 s) to absorb the cold-start cost of a fresh subprocess.
- Explicit `<confidence>` + `<decision>` tags every analysis turn; the agent must commit to continue/answer/give_up rather than mixing actions.
- Dedicated DESIGN phase that produces only the next query, separated from analysis reasoning.
- Per-question-type hints (`_QTYPE_HINTS`, `_MIN_QUERIES_BY_QTYPE`) nudge the agent toward type-appropriate experiments.
- Same sandbox preloads as the original coder agent.

Selected via `--agent-type coder_new` in `run_agent_batch.py`.

---

### 7. `orchestrator.py` — Experiment Controller

Manages the interaction loop, enforces rules, logs everything.

```python
def run(self) -> ExperimentResult:
    # 1. Initialize
    world_info = self.world_model.get_world_info()
    self.scientist.initialize(world_info, self.question, self.max_queries)

    # 2. Interaction loop
    while True:
        action = self.scientist.get_next_action()

        if action["type"] == "query":
            if self._query_count >= self.max_queries:
                self._notify_budget_exhausted()
                continue

            result = self._process_query(action["content"])
            # Budget incremented only on success (in _process_query)
            self.scientist.receive_result(result)

        elif action["type"] == "answer":
            answer = action["content"]
            break

        elif action["type"] == "give_up":
            answer = "GIVE_UP: " + reason
            break

    # 3. Evaluate and log
    is_correct = self._evaluate_answer(answer, self.question.ground_truth)
    return ExperimentResult(...)
```

**Responsibilities:**
- **Budget enforcement**: Only successful queries consume budget; failed queries are free
- **Turn logging**: Records every query, response, reasoning, and scientist memory
- **Evaluation**: Simple substring matching (for real-time logging only — final evaluation
  uses `evaluate_zero_shot.py --llm-extract` for accurate answer extraction)
- **Persistence**: Full experiment log saved to JSON

---

### 8. `run_agent_batch.py` — Primary Entry Point

Batch runs the scientist agent on all worlds in a dataset directory.

**Two-LLM architecture:**
- **World model LLM** (`--world-model`): Always runs locally (QwenLLM). Handles
  NL→structured query parsing. Uses low temperature (0.1) for reliable parsing.
- **Scientist LLM** (`--scientist-backend`/`--scientist-model`): Configurable backend.
  Does the actual causal reasoning. Uses higher temperature (0.3) for exploration.

**Agent variants** (`--agent-type`):
- `agent` (default) → `scientist_agent_causal.ScientistAgent` (query/answer/give_up only)
- `coder` → `scientist_coder_agent.CoderScientistAgent` (adds `<code>`)
- `coder_new` → `scientist_coder_agent_new.CoderScientistAgent` (modular INIT/CODE/ANALYSIS/DESIGN)

**Flow:**
1. Load all world JSON files from `--worlds-dir`
2. Initialize world model LLM (local) and scientist LLM (configurable)
3. For each world: convert JSON→BIF, build simulator + world model
4. For each question: create a fresh scientist agent of the requested type, run orchestrator, collect result
5. Save all results to `results/agent_<timestamp>.json` (the run-level metadata field `method` records the agent type)

Always imports from `world_model_causal` and `scientist_agent_causal` (the `--causal` flag
is kept for backward compatibility but is always on).

---

### 9. Zero-Shot Baselines

**`run_zero_shot.py`**: Presents the world description (story, variables, question)
directly to an LLM and demands an answer — no data queries allowed. Splits results
by difficulty group into separate JSON files.

**`run_zero_shot_sub_prompt.py`**: Uses the same scientist-like prompt structure
(with causal reasoning instructions) but without data access. Tests whether the
prompt engineering alone (vs. the data) drives performance.

Both support `--backend {local, openai, bedrock}`.

---

### 10. `evaluate_zero_shot.py` — Evaluation

Evaluates results from both zero-shot and agent runs. Dispatches by
`question_type`:

- **Yes/No** types → boolean substring match.
- **List** types (`all_causes_of`, `all_effects_of`, `markov_blanket`, …) → set match with precision/recall/F1.
- **Advanced types** → dedicated evaluators per question type (see below).

**Advanced-question evaluators (one per type):**

| Type | Evaluator | What it checks |
|------|-----------|----------------|
| `advanced_budget_argmin` | `_eval_budget_argmin` | exact `(var, value)` OR within ε of `metadata.best_expected` (looked up in `metadata.effect_table`) |
| `advanced_rank_topK` | `_eval_rank_topk` | strict order match OR every position within ε of the gold's `expected_target` at that rank |
| `advanced_budget_satisfy` | `_eval_budget_satisfy` | predicted `(var, value)` ∈ feasible set; "none" if feasible is empty |
| `advanced_side_effect` | `_eval_side_effect` | same as satisfy, against the side-effect-aware feasible set |
| `advanced_adjustment_set` | `_eval_adjustment_set` | exact-set match against any minimal backdoor adjustment set; reports prec/rec/F1 against closest |
| `advanced_mediator_class` | `_eval_mediator_class` | label match: `only_through_M` / `also_direct_or_other` / `not_mediator` |

**Tie tolerance** (argmin + rank_topK): controlled by `--tie-tolerance` (ε,
default 0.15 in expected-state-index units, set to 0 for strict). The
evaluator caches each world's `questions` list and looks up `metadata` by
`(world_file, question_id)`; existing result files don't need to carry the
metadata. Per-row metrics expose both strict matches (`exact_match`,
`exact_order`) and tie-accepted matches (`tie_accepted`,
`tol_exact_order`) so plotting can compare regimes.

**Key flags:**
- `--llm-extract` — use a local LLM to re-extract the answer from verbose agent reasoning. Required for `agent_*` / `coder_*` results; optional for zero-shot.
- `--extract-model` — extraction LLM (default `Qwen/Qwen2.5-7B-Instruct`).
- `--tie-tolerance` — ε for advanced argmin/rank_topK (default 0.15).

Outputs per-question-type accuracy breakdowns to `evaluations/eval_*.json`.

---

### 11. `analyze_failures.py` — Failure Analysis

Uses AWS Bedrock Claude to analyze individual experiment failures. Reads per-experiment
JSON logs and categorizes failures into types:
- `format_parsing_error`: Agent output couldn't be parsed
- `reasoning_error`: Agent reasoned incorrectly from the data
- `code_error`: Code execution failures (coder agent)
- `poor_query_strategy`: Inefficient use of query budget
- etc.

Produces per-log analyses and a synthesis report.

---

## Communication Formats

### Scientist → World Model (Natural Language)
```
"Give me 200 samples where we intervene to set smoke to yes,
 and I want to see the lung and dysp variables"
```

### World Model → Scientist (XML)
```xml
<query_result>
  <success>true</success>
  <query_type>interventional</query_type>
  <interventions>do(smoke=yes)</interventions>
  <n_samples>200</n_samples>
  <columns>smoke, lung, dysp</columns>
  <data_file>/results/query_data/query_0001.csv</data_file>
  <preview>
smoke,lung,dysp
yes,no,yes
yes,yes,yes
...
  </preview>
</query_result>
```

### Scientist Actions (XML)
```xml
<action type="query">
Give me 150 observational samples of all variables
</action>

<action type="answer">
Yes, there is a direct causal effect from smoke to lung.
Evidence: When I intervened on smoke, the distribution of lung changed
significantly, and they share no common parent that could explain this.
</action>

<action type="code">
df = pd.read_csv(query_1_csv)
print(df.groupby("smoke")["lung"].value_counts(normalize=True))
</action>
```

---

## Dataset Generation

Located in `dataset_generation_code/`. There are **two question regimes**.

### Basic regime — `world_gen_causal.py` (driver: `run_many.py`)

1. **Graph construction**: Random DAG with configurable nodes (10/20/30), target edges
   (~1.5x nodes), max 3 parents per node. Ensures connectivity.
2. **Variable generation**: LLM (Qwen) generates domain-appropriate variable
   names, descriptions, and categorical values for the chosen topic.
3. **Story generation**: LLM creates a narrative context.
4. **CPD construction**: Logistic CPDs with random weights; root nodes use Dirichlet prior.
   The CPD loop iterates until d-separation claims hold within `faithfulness_eps`.
5. **Non-intervenable variables**: LLM identifies variables that shouldn't be manipulable
   (e.g., "Age" in a social-science study).
6. **Question generation**: Answer-first stratified generation — 3 groups × 2 questions, Yes/No balanced.

Question groups:
- Group 1 (easy): 1 `causal_effect` (Yes/No) + 1 `all_causes_of` / `all_effects_of` (list)
- Group 2 (medium): 2 marginal independence questions (1 Yes + 1 No)
- Group 3 (medium): 2 conditional independence questions (1 Yes + 1 No)

Independence questions are post-hoc classified by structural motif
(`chain`, `fork`, `v_structure`, `direct`, `other`) for analysis only.

Primary basic dataset: **`all_out_bn/out_bn_3_4/`** — 60 worlds, 360 questions.
Audit tool: `check_faithfulness.py`.

### Advanced regime — `world_gen_advanced.py` (drivers: `run_many_advanced*.py`)

Builds 10/20/30-node worlds with **exactly 6 advanced "Causal Decision"
questions per world** (one per type). Reuses topology + LLM machinery from the
basic regime, but adds a per-variable `preferred_low: true|false|null` flag
(LLM + token heuristic) so "reduce target" questions have unambiguous gold.

| Type | Generator | Gold answer |
|------|-----------|-------------|
| `advanced_budget_argmin` | `_gen_budget_argmin` | argmin do(X=v) of expected target index |
| `advanced_budget_satisfy` | `_gen_budget_satisfy` | feasible set or "none"; threshold balanced across `satisfy_empty / few / many` buckets |
| `advanced_side_effect` | `_gen_side_effect` | feasible set with `target_improvement ≥ τ_t` and `side_change ≤ τ_s` |
| `advanced_adjustment_set` | `_gen_adjustment_set` | minimal backdoor sets, "none", or "unidentifiable" |
| `advanced_mediator_class` | `_gen_mediator_class` | one of `only_through_M / also_direct_or_other / not_mediator` |
| `advanced_rank_topK` | `_gen_rank_topk` | top-K interventions ordered by expected target |

Supporting helpers live in **`advanced_utils.py`**:
`compute_effect_table`, `expected_value_under_do`,
`get_minimal_backdoor_adjustment_sets`, `is_valid_backdoor_set`,
`classify_mediator`, `rebuild_from_world`, `intervenable_var_names`,
`scoring_for_target`.

**Generation-time robustness guards** (added 2026-04-26):
- `_pick_target_for_reduce(..., min_top_gap=_MIN_TOP_GAP)` — for argmin and rank_topK targets, the rank-1 vs rank-2 expected-do-effect gap must be ≥ 0.15 (in expected-state-index units). Targets failing this filter are skipped, preventing golds that are statistically indistinguishable from the runner-up.
- `_make_threshold_robust(threshold, values)` — applied in budget_satisfy and side_effect; shifts the chosen threshold to the midpoint of the bracketing gap in the improvements distribution if any value lies within `_THRESHOLD_GUARD_EPS = 0.01`. Stops tiny CPD perturbations from flipping the answer-bucket.

The matching eval-time mechanism is `--tie-tolerance` in
`evaluate_zero_shot.py` (default ε = 0.15).

Drivers (`bedrock` backend, model `us.anthropic.claude-opus-4-7`):
- `run_many_advanced.py` — full 60 worlds (20 × {10, 20, 30})
- `run_many_advanced_10.py` / `_20.py` / `_30.py` — per-size variants. The 30-node variant tightens `--max-attempts-per-world` and `--cpd-max-attempts` because faithfulness is expensive at that scale.

Audit tool: `audit_advanced.py` re-derives every gold answer from the
stored CPDs and reports drift; mirror of `check_faithfulness.py` for the
basic regime.

Advanced datasets: `all_out_bn/out_bn_advanced_4_24/`,
`all_out_bn/out_bn_advanced_4_24_n10_long/`.

---

## Analysis Notebooks

Under `framework_code/notebooks/`. They consume `evaluations/eval_*.json`
files written by `evaluate_zero_shot.py`.

- **`examine_results_many_together.ipynb`** — canonical multi-run
  comparator. Define `RUN_LABELS` / `RUN_FILES` at the top, then plots:
  (1) overall accuracy bar, (2) per-group accuracy, (3) drill-down by
  question type within each group, (4) by difficulty + graph size,
  (5) Yes/No answer-bias check, (6) set/list precision/recall/F1, (7)
  per-group accuracy by node count.
- Per-dataset snapshot notebooks: `examine_results_3_4.ipynb`,
  `examine_results_zero_shot.ipynb`, `examine_results_big.ipynb`,
  `examine_results_small.ipynb`, `examine_results_causal.ipynb`, etc.
- Demo notebooks: `demo_asia.ipynb`, `demo_json.ipynb`,
  `demo_scientist_agent.ipynb`.

The notebooks read `evaluated[]` rows; for advanced runs each row's
`eval.metrics` already exposes both strict and tie-accepted match flags,
so a plotting cell can switch regimes without re-running the evaluator.
