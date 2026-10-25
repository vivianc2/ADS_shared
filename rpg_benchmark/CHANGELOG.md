# Changelog

Every result file records the harness version it ran with (`harness_version` in `result_*.json` and `summary.json`).
Files without that field come from v10.0.

## v10.2 sandbox hardening (2026-10-24)

Found by the RL session's red-team test (`dataset_generation_code/rl_v10/test_sandbox_isolation.py`): agent code
could open any file through libc via ctypes, below the audit hook (the world files hold the gold), and could read the
process environment. No stored episode tried either: 0 of 8,357 result files contain `import ctypes`, `gzip.open`, an
`open(...json.gz)`, `os.environ` or `/proc/self`. `code_sandbox.py` now:
- clears the environment to a small allowlist (`_ENV_KEEP`) and empties `posix.environ`;
- refuses ctypes lookups of libc / Python C-API entry points that open files, run programs, open sockets or read the
  environment (`_DENY_SYM`; numpy and sklearn's BLAS lookups are unaffected);
- refuses `/proc/<pid>/{environ,mem,pagemap}` and UDP sends and binds.

Checks:
- `tests/test_sandbox.py`: 21/21 (5 new cases).
- The red-team set: 0 of 28 must-block attacks leak (`import agent_loop` still works, as before; it holds no
  world data).
- Replay of every stored v10.2 episode with code turns through the old and the new sandbox (1,132 episodes: 12
  models × core54 + Opus on the other 486; deterministic resolver; `PYTHONHASHSEED=0`, because agent code iterates
  over sets): grades identical 1132/1132, prompts and turn records identical 1128/1132.
  - **2 of the 4:** an output-cut message caused by the long file path of the test copy of the sandbox (the release
    file prints `<sandbox>`).
  - **1:** identical on rerun.
  - **1:** a memory address printed in an agent's error message.

  Script and rows: `personal_docs/camera_ready/analysis_rpg/sandbox/`.

One visible difference: Python errors show the sandbox's own line number (`File "<sandbox>", line N, in _worker`),
which moved because lines were added above it. `harness_version` stays `v10.2`, as with the ablation switches.

## v10.2 ablation switches (2026-10-20)

Opt-in switches for the week-7 ablations. The defaults reproduce v10.2 exactly (the default prompt is byte-identical,
checked with `build_agent_prompt([], "stateless") == AGENT_PROMPT`), so `harness_version` stays `v10.2`. Every result
records the setting in `ablation` (`result_*.json`; `summary.json` also records budget and turn limit).

- `--history full` (R8): the whole conversation is sent each turn (system, then the first full prompt, then each reply
  and a short per-turn message with counters, data list, directives and the last result). The situation report and
  memory block are not repeated after turn 0. The prompt's MEMORY paragraph says so (`_MEMORY_FULL`). For Claude on
  Bedrock a cache point is placed after the conversation prefix (`cache` flag → Converse `cachePoint`); the OpenAI
  wrapper drops the flag.
- `--min-answer-turn N` (R6): an answer before turn N is not executed and the agent is told; a directive says when
  answers open. The turn limit still forces an answer.
- `--prompt-variant a,b` (R5 `multi_cause`, R6b `first_batch`, R10 `no_trap_warning`): exact edits on the default
  prompt, defined in `agent_loop.PROMPT_VARIANTS`.
- `out/launch_model.sh`: `EXTRA_ARGS` is passed to `run_benchmark.py` and written to `run_info.txt`.
- Audit field `text_blocks` (Bedrock): when a reply has several text blocks, all of them are kept in the turn record;
  the last one is still the one used (behaviour unchanged). Added after ACED found that GPT-6's extra blocks there
  continue the episode with invented tool output. RPG check (out/week7/blocks_*, 6 core54 worlds each):
  - **Frequency:** multi-block in 49/186 turns for GPT-6.1 Sol and 33/186 for GPT-6 Astra; Haiku 5.5 0/154.
  - **Content:** every block is a complete reply (one action plus memory), i.e. a revision of the same turn, not a
    continuation. No block contains a result the agent did not receive. In the one block that mentions results, the
    memory says "Last code output not received in context; do not assert unreturned results".
  - **Agreement:** first and last block have the same action type in 35/49 (Sol) and 21/33 (Astra), and the identical
    action in 20 and 16. No turn switches to an answer only in the last block.
  - **Conclusion:** the last-block rule is kept for RPG.

## v10.2 (2026-10-17)

Fixes from the full trace review of 12 runs (`personal_docs/results/v10/TRACE_REVIEW_2026-10-17.md`). The worlds
are unchanged. The default prompt, what the agent is shown, the reply parser and the sandbox change, so v10.2
results are not comparable with v10.0/v10.1 results; all reported models are rerun on v10.2.

**Prompt** (built-in; copy in `prompts/agent_prompt_v10_2.txt`): a new "HOW THIS ENVIRONMENT WORKS" section.
- **Memory:** each turn shows only the situation report, counters, data list, the last result and the memory block,
  so memory is the only record between turns.
- **Units:** every experiment is a fresh, independent batch; unit ids are unique per experiment; compare by
  distributions, not unit by unit.
- **Baselines:** a measurement without an intervention records the units as they currently run (possibly a filtered
  subset), while interventions use fresh representative batches; compare interventions with each other.
- **Limits:** experiment budget and turn limit; you must answer when 2 turns remain.
- **Format:** exactly one `<action>...</action>` block plus one `<memory>` block; no tool-call syntax; if several
  actions are written, only the first is executed.
- **Other text changes:**
  - the code section names `experiment_<id>_df` and the file `experiment_<id>.csv`;
  - the memory template asks for findings with numbers;
  - the forced-answer directive no longer contains a literal `<action ...>` tag (models quoted it back).

**Reply parser** (`agent_loop._parse_reply`). One parser for both dialects:
- `<action type=..>`;
- tool-call `<invoke name=..>` with `<parameter>` children, and Haiku's mangled `<invoke:code>` form;
- a bare `<parameter name=..>`;
- a bare `<code>` block;
- any number of `antml:` prefixes.

Rules:
- Empty blocks are skipped, so an empty `<action type="code"></action>` no longer beats a real `<invoke>`.
- An action ends at `</action>`, at the next block, at `<memory>` or at a leaked chat token (`<|message|>`).
- With several blocks, the first block is executed and the agent is told how many blocks there were and that the
  rest were ignored. It is skipped only for a later **valid** block of the same type (code that compiles, JSON that
  parses), so prose that merely mentions a tag is skipped, but a broken code block is never swapped for a later
  answer.
- On a forced-answer turn the last valid answer block is preferred.
- JSON is read leniently: code fences, stray `</parameter>`/`</answer>`, an extra closing brace, trailing text,
  trailing commas.
- An unclosed `<memory>` at the end of a reply is read.
- Replay of all ~190k stored replies: every difference from v10.1 was checked by category
  (`personal_docs/scripts/v10/replay_parser.py`).

**Feedback:**
- A reply with no executable action now produces "YOUR LAST REPLY WAS NOT EXECUTED: <reason> ..." plus the previous
  result. It used to repeat the old result silently.
- A measure/intervene whose JSON cannot be read is not executed and does not use an experiment.

**What the agent is shown:**
- unit ids are `e<exp>_u<nnn>` (unique per experiment);
- each result names its DataFrame and file and lists its columns;
- the turn header lists `experiment_<id>_df (observational | intervention: SetX=v)`;
- control columns in the CSVs are named `Set` + the control's name in CamelCase (e.g. `SetUpperCutoffVoltage`),
  uniformly for every control, in the style of the signal columns. v10.0/10.1 used `do_upper_cutoff_voltage`; in
  gate round 1, 44% of models' KeyErrors were a control written in CamelCase;
- `pd.read_csv('experiment_1_df.csv')` also reads the right file;
- a control named in a measure request is reported as "a control, not a measured signal; it was at <level>";
- the same control twice in one intervention: the first setting is applied and the repeat is rejected with a message.

**Sandbox** (`code_sandbox.py`):
- Each code turn runs in a fresh temporary folder holding only copies of this episode's CSVs, removed afterwards.
- A Python audit hook refuses file and directory access outside that folder and the Python installation, and blocks
  subprocesses and sockets.
- `pd.read_csv('experiment_1_csv')` and `pd.read_csv('experiment_1.csv')` both work.
- Pandas prints wide tables in full.
- The output cap is raised from 6,000 to 10,000 characters, with an explicit "OUTPUT CUT" message.
- Tests: `tests/test_sandbox.py`.

**Grading:**
- Sign predictions for controls that do not exist in the world are not scored (they used to count as correct "0"s).
- **A conditional policy on an additive or scaling control is now graded.** Policies were implemented only for
  "set" controls; a valid policy on e.g. a salt-bath dose crashed the grader. In v10.0 that ended the batch; in v10.1
  the policy was dropped and only the plain dose graded.
  - Each unit now gets dose_if_ge or dose_if_lt by its stratifier, with per-unit effect and side effect (`scm.py`).
  - Checks: a policy giving everyone the same dose grades identically to that dose; all 54 core gold answers regrade
    to their stored gold utility.

**Resolver:**
- Identical requests within a world are resolved once (cache), which cuts resolver calls by about 60%.
- The resolver model's reply is read as its first complete JSON object; a trailing second object or text used to
  force the lexical fallback.
- New rule: a measure request that names an actuator returns that actuator, so its level can be reported.

**Backends:**
- OpenAI-compatible:
  - token usage and the model's reasoning (`reasoning_content`) are saved per turn;
  - an empty reply is retried up to twice;
  - sampling presets for Gemini (temperature 1.0, top_p 0.95) and Gemma (temperature 1.0, top_p 0.95, top_k 64).
- Bedrock: the model's reasoning text is saved when the API returns it.
- Every turn records `model_reasoning`, `parse_note`, `json_repaired`, `error_detail` and `shown_next` (what the agent
  saw after an unexecuted reply).

**Gate round 2 fixes** (from reading the real round-2 traces of 8 models; `personal_docs/results/v10/GATE_V10_2_2026-10-17.md`):
- Parser: a block also ends at `</memory`; `<parameter="code">` is read; lenient JSON decodes the first object (a
  list only when it comes first); a list payload is wrapped as requests/actions; an answer given as a list is an
  error, not executed.
- A turn's `action_type` is recorded only after its JSON is read (`attempted_action_type` otherwise).
- A measure/intervene that produced no data (every request rejected) does not use an experiment, and the agent is
  told "Nothing was run ..., so no experiment was used."
- A rejected intervention on a measured signal says "'X' is a measured signal, not a control; it cannot be set
  directly."
- An intervene with an empty `actions` list returns a representative batch with no control changed (an
  interventional baseline); the prompt says so, and the data list labels it "baseline batch".
- Python tracebacks no longer show server paths (`<python>/`, `<sandbox>`, `<analysis folder>`).
- `served_model` (the model id the API reports) is saved per turn.
- An intervention result names each control by its CSV column: `applied_intervention` is keyed by the column name
  (`{"SetZincTherapy": 25}`), and each action's interpretation says "apply 'zinc therapy' at 25 (column
  SetZincTherapy)". It used to show only the label ("zinc therapy"). Models then wrote it in CamelCase without
  `Set` (`df['ZincTherapy']`), a KeyError in 0.5–5% of code turns and up to 40% of episodes in gate round 3.
- Resolver fallback: when the resolver LLM call fails (Bedrock `content_filtered`, seen for "fogging rate" requests),
  the lexical fallback could not read a control written as its CSV column name (`SetFoggingRate`): 0 of 1,338
  core54 column names resolved. It now looks the column name up exactly, so 1,338 / 1,338 resolve to the right
  control (method `"column"`). Normal resolution (LLM succeeds) is unchanged. This came in during gate round 3, so the
  round-3 stage-1 runs use the earlier fallback; the affected turns are counted in the gate doc.
- Checks after these fixes: sandbox tests 15/15, mock 270/270, replay of the stored replies (every new category
  inspected), and 3-world real probes (Haiku 5.5, Qwen 3.5 9B).

**Gate round 3 fixes** (trace review of ~220 round-3 episodes over 6 models; details in the gate doc):
- **Budget spent no longer forces the answer.** v10.0–10.1 forced an answer as soon as the 15 experiments were used,
  so a model never analysed its last experiment: in all 38 round-3 episodes that used the full budget, the answer
  came on the next turn, once with 15 turns left. Now only the turn limit forces the answer (2 turns left). With the
  budget spent, a directive says measure/intervene can no longer run and code is still free. A measure/intervene
  then is refused without charge, and the turn records `error: "budget spent"`. The prompt's LIMITS line says so.
  The same refusal also stops a 16th experiment on a forced turn.
- **Unit ids are integers again** (`experiment * 1000 + row`, e.g. 3000–3399; still unique per experiment). The
  v10.2 string ids (`e3_u000`) broke `df.mean()`, `(df - df.mean()) / df.std()` and `np.asarray(df)`: 75 code turns
  in 55 round-3 episodes.
- **Sandbox: a carried variable that is reassigned is saved again.** The carried names were counted as "provided by
  us", so `best = 5`, then `best = 7`, then `print(best)` printed 5. Test added (`tests/test_sandbox.py`, 17 checks).
- **A request that runs nothing uses no experiment number**, and its result no longer shows `experiment_id` /
  `n_units` (ids had gaps; one model then read a non-existent `experiment_13_df`).
- **Prompt examples are neutral placeholders** (`<an additive>`, `<signal name>`, `SignalA`). The old examples
  ("chelating additive", "feed-water flow", "product titer", "broth cloudiness") are real controls or signals in
  some worlds, the example signs included; models copied them into requests and answers.
- **Code:** lines that are only an XML-like tag (`</parameter_text>`, invented by Haiku) are dropped before running.
- **Resolver:** a Bedrock reply cut by the content filter (partial text such as `{"target":"`) counts as a failed
  call, so the fallback runs instead of an empty mapping. Bedrock turns record `finish_reason: "content_filtered"`.
- **Directive:** repeated attempts to set a measured signal now count toward "stop asking for things that are not
  part of this system".
- Not changed (documented in the gate doc):
  - some signal column names differ from the situation report's wording (24%; the result view names each
    measured column);
  - an optional policy with a valid stratifier sets the dose of its treatment.

**Stage-2 review fixes** (2026-10-17, after the stage-2 runs; a replay of every stored reply shows that only
Sonnet 5.5's parsing changes: 13 turns. Every other model's replies parse identically, so their runs stand, and
Sonnet is rerun):
- **Sonnet's placeholder tool call ran as a charged baseline.** `<invoke name="intervene"><parameter
  name="dummy"></parameter></invoke>` became `{"dummy":""}`, counted as valid, and ran as a baseline batch (no
  "actions"). It cost an experiment and dropped the real action later in the same reply: 10 turns in 9 of 54 Sonnet
  episodes.
  - Empty or unknown invoke parameters are now dropped, so the block is empty and skipped.
  - A measure must have a "requests" list and an intervene an "actions" list, and other list fields must be lists.
  - A lone measure/intervene without its list is refused without charge, with a message. Only an explicit
    `"actions": []` is a baseline.
- `<invoke name="action" type="code">` is read as code (3 Sonnet turns were "no action").
- **Answer terms:** the full text is tried first when it is exactly a name, alias or control column.
  "chlorine-ammonia ratio" (an alias) had been split at the hyphen and mapped through "chlorine" to another control;
  1 of 4,008 sign keys, no score change.
- **Sandbox:** when output is cut at 10,000 characters, a Python error after it is kept. It never happened in the
  runs.
- Checked, not changed:
  - Values are rounded to 2 decimals. Regrading the 2 affected answers at full precision changes no grade.
  - A stop sequence for Sonnet's `</invoke>` repetition loop would lose the memory block in 106 of 143 affected
    replies (Sonnet often writes `</invoke>` twice, then its memory), so it is not added.
  - Reading `<parameter name="memory">` would also change 5 Haiku turns, so it is not added either.

**Data:** `data/core54/`: 54 test worlds, 6 per causal structure and 5–6 per domain, chosen deterministically; the
paper's main set.

## v10.1 (2026-10-16)

Changes to how replies are read. The worlds, simulator, grader and default prompt are unchanged.

- **Tool-call syntax is accepted as the action** (`agent_loop.py`, `_parse_invoke`).
  - Some models write the action as `<invoke name="code"><parameter name="code">...</parameter></invoke>` instead of
    `<action type="code">...</action>`. Claude Sonnet 5.5 did so in 83 of 93 replies in a pilot.
  - v10.0 rejected these replies as "no action parsed".
  - The new path runs only when a reply contains no `<action` text at all, so every reply v10.0 could parse is parsed
    exactly as before.
  - Variants handled, all seen in Sonnet 5.5 replies:
    - **Namespace prefix:** tags carrying Anthropic's tool-call namespace prefix (`antml:`, as in `<invoke`,
      `<action`, `<memory`) are read as the plain tags. This applies to every reply, but no reply from any other
      model contains `antml:`.
    - **Bare blocks:** a `<parameter name="code">` block without the `<invoke>` wrapper, or a non-empty `<code>` block,
      is read as a code action.
    - **Repeated blocks:** when one action type is repeated several times (Sonnet's runaway repetition, often with
      truncated copies), the first block is used.
  - Different action types in one reply are still rejected, as v10.0 does for several `<action>` blocks. The
    `<action>` rule itself is unchanged.
  - Turns read this way carry `"parsed_via": "invoke"`.
  - On the 2026-10-16 Sonnet smoke test (90 replies), v10.0 parsed 23 and v10.1 parses all 90.
  - Regression: on 31,760 stored replies from other models, the action and the memory read by v10.1 are identical to
    v10.0.
- **Stop sequences for Claude models on Bedrock** (`llm_bedrock.py`, `DEGENERATE_STOPS`).
  - The two stop strings are `</invoke>\n<parameter` and `</invoke>\n\n<parameter`.
  - Sonnet 5.5 sometimes repeats `<parameter name="action">x</parameter></invoke>` after its first block until it hits
    the output cap (up to 686 times in one reply).
  - Stopping at the first repeat keeps the first block; the memory block is lost only in a reply that degenerates.
  - Opus 4.8 and 5.5 accept the parameter; it does not apply to non-Claude models.
- **Bedrock replies: only the last text block is used** (`llm_bedrock.py`).
  - GPT-6 Astra and GPT-6.1 Sol (and Claude Haiku 5.5) return interleaved reasoning and text blocks: a draft
    answer, more reasoning, then the final answer.
  - v10.0 joined all text blocks, so the reply held two actions and was rejected. That affected 5,333 of 16,740
    turns (32%) in the GPT-6 Astra test run and 5,812 of 16,738 (35%) for GPT-6.1 Sol: every rejected turn in both
    runs held two `<action>` blocks.
  - Haiku 5.5: 246 two-action replies.
  - Single-text-block replies (Opus, Sonnet, Llama, the resolver) are unchanged.
- **An agent reply with no text is an empty turn** (`agent_loop.py`).
  - Bedrock's gpt-oss-120b sometimes returns only a reasoning block. v10.0 raised an error that ended the whole
    batch process, so the worlds in flight were restarted from scratch.
  - Such a reply now counts as one turn with no action (the OpenAI-compatible backend already behaved this way).
  - The resolver still falls back to lexical matching on such errors.
- **A malformed answer no longer crashes the batch** (`agent_loop.py`).
  - Example: gpt-oss gave a dose as a JSON object, and grading raised a `TypeError` that ended the process; the
    world was then re-run from scratch.
  - Non-numeric values (and an unusable policy) are now dropped and the rest is graded, with `grade_error` recorded
    on the turn.
  - A policy whose threshold or doses are not numbers (e.g. a list) is now unresolved.
  - The answer's fields are normalised before use:
    - a non-list recommendation becomes `[]`, and non-object items are dropped;
    - a non-object `structured` becomes `{}`;
    - a non-object policy is dropped;
    - a list proxy becomes its first string;
    - non-string decoys are dropped;
    - non-object sign predictions become `{}`.
  - The original answer is kept in `answer_raw`, and `grade_error` says what was dropped. Well-formed answers are
    unchanged (mock regression 270 / 270).
- **No `temperature` for Sonnet 5.x and GPT-6 on Bedrock** (`llm_bedrock.py`). These models reject the field. Before,
  this mattered only when `--temperature` was passed explicitly.
- **`harness_version` and `system_prompt_file` are recorded** in `summary.json`, and `harness_version` in each result.
- **`prompts/` added:**
  - `agent_prompt_v10.txt` is the built-in v10 agent prompt, byte-identical; it is used when no
    `--system-prompt-file` is given.
  - `agent_prompt_v10_memnote.txt` is the same prompt plus one paragraph before the output format. The paragraph
    states that each turn shows only the last action's result and the memory block, so memory is the agent's only
    record. It is used only by the memory-note robustness check; the benchmark's main protocol stays
    `agent_prompt_v10.txt`.

## v10.0 (2026-10-13, ADS_shared b0203a5)

The first release, used for all main runs up to 2026-10-16: Opus 5.5, Gemini 3.1 Pro, GPT-6.1 Sol, GPT-6 Astra,
GPT-OSS, Qwen, Llamas and DeepSeek-V4-Pro.
