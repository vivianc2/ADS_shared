# Causal Discovery Benchmark — Architecture

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

**Flow:**
1. Load all world JSON files from `--worlds-dir`
2. Initialize world model LLM (local) and scientist LLM (configurable)
3. For each world: convert JSON→BIF, build simulator + world model
4. For each question: create fresh scientist agent, run orchestrator, collect result
5. Save all results to `results/agent_<timestamp>.json`

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

Evaluates results from both zero-shot and agent runs.

**Key flag: `--llm-extract`**
- Agent responses contain verbose reasoning that makes simple regex extraction unreliable
- `--llm-extract` uses a local LLM to re-extract the actual answer from the raw response
  between `<answer>` tags
- This is the authoritative evaluation method for agent results

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

Located in `dataset_generation_code/`.

**`run_many.py`** calls **`world_gen_causal.py`** to generate worlds:

1. **Graph construction**: Random DAG with configurable nodes (10/20/30), target edges
   (1.5x nodes), max 3 parents per node. Ensures connectivity.
2. **Variable generation**: Uses LLM (Qwen) to generate domain-appropriate variable
   names, descriptions, and categorical values for a given topic.
3. **Story generation**: LLM creates a narrative context for the variables.
4. **CPD construction**: Logistic CPDs with random weights; root nodes use Dirichlet prior.
5. **Non-intervenable variables**: LLM identifies variables that shouldn't be manipulable
   (e.g., "Age" in a social science study).
6. **Question generation** (`world_gen_causal.py`): Answer-first stratified generation —
   3 groups x 2 questions = 6 questions per world, with guaranteed Yes/No balance.

**Question groups:**
- **Group 1 (easy)**: 1 causal_effect (Yes/No) + 1 list (all_causes_of or all_effects_of)
- **Group 2 (medium)**: 2 marginal independence questions (1 Yes + 1 No)
- **Group 3 (medium)**: 2 conditional independence questions (1 Yes + 1 No)

Independence questions are post-hoc classified by structural motif (chain, fork,
v-structure, direct, other) for analysis purposes only.

**Primary dataset: `all_out_bn/out_bn_3_4/`** — 60 worlds, 360 questions.
