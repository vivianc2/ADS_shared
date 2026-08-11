# Undergrad task brief: auto-curriculum + reward-signal analysis

**Vocabulary reminder (we went through this today — just so the terms below are unambiguous):**
- **Policy** `π_θ` — the model; `θ` are the (LoRA) weights we train.
- **Reward** — the oracle's score in `[0,1]` for a *finished* attempt (`0.5·PartA + 0.5·PartB`);
  pure code, no human, no judge.
- **Group** — for one world we roll the model out **G times**; an attempt's **advantage** is its
  reward minus the group's mean reward ("how much better than typical").
- **GRPO** — push the probability of above-average attempts up and below-average ones down; the
  group mean is the baseline, so there's no separate value network.
- **Dynamic sampling** — if every attempt in a group gets the *same* reward, all advantages are 0
  → no gradient; such groups are dropped as useless.
- **Entropy** — how spread-out the model's choices are; if it *collapses*, the model has stopped
  exploring (a red flag), so we watch it.

---

## 0. The 60-second version

We're RL-training a language model to act like a scientist: it runs experiments in a
simulated broken system and must discover the hidden cause + the fix. **Your job is not to
train the model — it's to decide WHICH practice worlds it trains on, and in what order, so
training actually makes progress.** That's a *curriculum*. You'll (1) map two different things
— where the **reward can differentiate** answer quality (from the CPU **oracle**, fully on your
own) and where the **current model actually produces reward spread** (from rollout logs the
mentor dumps) — then (2) build a component that steers world selection toward the cells that are
informative *for the model as it is right now*. This is a real, citable piece of the research
(the "auto-curriculum" idea from RLVE), and it's built so it can't break the training code.

---

## 1. What the whole project is (context)

- We procedurally generate **worlds**: a simulated failing system (bioreactor, datacenter,
  crop field…) with a **hidden cause**, knobs the agent can turn (**actuators**) and signals
  it can read (**measurables**).
- An **agent** (an LLM) interacts turn by turn — measure / intervene / run code / answer.
  The answer has two parts: **Part A** the fix, **Part B** the mechanism (true proxy, decoys,
  effect signs).
- A trustworthy **oracle** grades the answer → a **reward** in [0,1]
  (`0.5·PartA + 0.5·PartB`). Pure code, no human, no LLM judge ("RL with a **verifiable
  reward**"). **This oracle runs on CPU in milliseconds — it's your main tool.**
- We use **GRPO** to improve the model (see the vocabulary reminder above).
- **Thesis:** training on many procedurally-generated causal worlds teaches *generalizable*
  scientific reasoning — the model should improve on **held-out** world families.
- **Where we are:** the environment, reward, and training stack (**SkyRL**, on the mentor's
  8×L40S box) are built, and the first training run is starting. Your workstream runs in parallel.

---

## 2. Why your task exists (the problem, from first principles)

GRPO learns from the **spread of rewards within a group**. For each world the model is rolled
out **G times** (a group); an attempt's learning signal is how far its reward is from the group
average (its **advantage**, `A_i = r_i − mean`), and the update is `−A_i · ∇log π_θ`. Key fact:

> **If all G attempts on a world get the same reward, every advantage is 0 → no gradient, no
> learning, wasted compute.** (DAPO calls dropping these "dynamic sampling".)

So the quantity that matters is the **within-group reward variance under the current policy**,
`Var_{τ∼π_θ}[R(τ, w)]`. Read that subscript carefully: it depends on **π_θ — the model as it is
right now** — not on the world alone. This splits into two things you must NOT conflate:

- **Potential spread** — a property of the *world + reward*. How much the oracle *can*
  differentiate answer qualities here: does Part B create intermediate rungs, or is the world
  effectively pass/fail? Measurable on CPU with no model (Milestone 1A). It's a **necessary
  condition** — if the reward is near-binary on a world, no policy can get a graded signal from
  it — but it is **not** the learning signal.
- **Realized spread** — a property of the *world + reward + current model*. Whether the model
  *actually* produces attempts whose rewards differ. A world with huge potential spread yields
  **zero** signal if the model always answers equally badly (`[0,0,…,0]`); a world whose reward
  only ranges 0.3→0.7 can be gold if the model's attempts land across it. This is the real GRPO
  signal, and measuring it needs model rollouts (Milestone 1B).

The useful "**Goldilocks band**" is therefore `world + reward + current policy` — **not** an
intrinsic property of the world. And because π_θ changes every step, the band is a **moving
target**: a cell that's flat-zero for the base model can become informative once the policy
improves, and an easy cell saturates (everyone ≈ 1) and stops teaching. That non-stationarity is
exactly why the curriculum needs a *feedback* hook (`update_...`, §3) and an *exploration floor*
(§3); a fixed, set-once difficulty ordering fails.

Our own probe already shows the split biting at the base model: the *only* archetype with
**realized** reward spread was `confounded_chain`; `collider_selection` and `hidden_subtype`
were flat at 0 — even though the oracle assigns them plenty of *potential* spread. So the
evidence for "**start on chain, unlock the rest as the model earns it**" comes from the
**rollout probe (realized)**, not from the oracle ladder (potential). Keeping that attribution
straight is the whole point of your analysis.

---

## 3. What you will build (three milestones — all CPU on your side)

### Milestone 1 — Map the signal: potential (1A) then realized (1B) — START WITH 1A

**1A — Reward-landscape audit (CPU, oracle-only, fully yours; start here).** This measures
**potential** spread — where the *reward function* can differentiate answer quality — and
explicitly **not** where the current model can learn (that's 1B). For many worlds, score a
**ladder of synthetic answer qualities** with `rpg_rl/reward.py` (CPU): `empty`, `wrong-knob`,
`fix-only` (Part A only), `+proxy`, `+decoys`, `full-gold`, `random-ids`. Record per world the
reward at each rung and the achievable spread (max−min, std). Aggregate by **archetype /
difficulty feature (`sign_flip`, `interior_dose`, `two_cause`, `symptom_trap`) / skin / depth**.

The questions 1A legitimately answers: does this world type expose *intermediate* reward levels,
or is it effectively binary despite our "continuous" reward? Does Part B add real grading
resolution? Do some features *compress* the landscape so nearly all answer qualities score the
same? A cell that's near-binary in 1A can never give a graded GRPO signal — a real red flag to
surface. What 1A **cannot** tell you: whether Qwen currently produces those partial answers.

*(Reuse the ladder in `test_reward_integrity.py` and the gold-id builder in `test_env_reward.py`
— they already build these answers in id-space.)*

**Deliverable (1A):** report + JSONL/parquet — per (archetype, feature[, skin, depth]) cell, the
achievable reward spread and where the rungs bunch up.

**1B — Realized-signal analysis (CPU, from the mentor's rollout logs; this is what drives the
curriculum).** As soon as the mentor dumps per-world reward distributions from **actual model
rollouts** (JSONL — you analyze the file, you never run the model), compute per cell: `μ_c`
(mean reward), `σ²_c` (reward variance), the **fraction of groups with `σ_group > 0`** (the DAPO
"keep" fraction — the most direct signal metric), and the **saturation split** (share of groups
sitting at ~0 vs at ~max). Then lay 1A and 1B **side by side** — the gap between potential and
realized is the diagnosis:
- both high → **learnable now** (train here);
- potential high, realized ≈ 0 → **not yet** (too hard for the current model — hold, revisit);
- potential ≈ 0 → **dead reward** (the reward itself can't teach here — escalate to the mentor).

**Success (Milestone 1):** we can point at the map and say, *with the right justification*, which
cells are informative now, which are one improvement away, and which have a reward problem.
"Start on `confounded_chain`" must be justified by **1B (realized)**, not by 1A.

### Milestone 2 — Build the curriculum (CPU module)
Training runs in **SkyRL**, which reads a **dataset (parquet) of worlds**, so the curriculum
operates at the **dataset level** (a per-round batch, not a live per-step loop). Two functions:

```python
# curriculum.py  (pure CPU; no torch, no model)
def select_worlds(n, curriculum_state, split="train") -> list[WorldSpec]:
    """Pick n worlds (seed/skin/archetype/features) biased toward cells that are informative
    FOR THE CURRENT MODEL, respecting the train/heldout split (never emit held-out families)."""

def update_curriculum_state(curriculum_state, reward_logs) -> curriculum_state:
    """Fold in per-world/-cell rewards from the last SkyRL run: update per-(archetype, feature)
    estimates of realized informativeness so the NEXT round samples the still-informative cells
    and unlocks ones the model has grown into (RLVE-style adaptation)."""
```
Note the name — **`curriculum_state`, not `difficulty`**: a world is not intrinsically
"difficulty 0.73"; its usefulness is *relative to the current policy* and moves as the model
learns (§2). You're tracking realized informativeness, not an intrinsic constant.

**Two hard requirements (not optional):**
1. **Exploration floor.** Never let any train cell's probability reach 0. Mix
   `p(c) = (1−α)·p_uniform(c) + α·p_adaptive(c)` with a real floor. Reason: `hidden_subtype`
   starts flat-zero; a pure "chase the informative cells" rule gives it probability 0 forever,
   the policy improves elsewhere, and the curriculum **never re-discovers** that hidden_subtype
   became learnable. The floor is what lets a moving frontier (§2) be tracked.
2. **Split safety.** Only TRAIN skins/archetypes; held-out families are never selected
   (`rpg_rl/splits.py`). Assert it, like `test_stream_parallel.py` does.

**Candidate policies to compare (this is the research part — none is the known answer):**
(a) **uniform** (baseline); (b) **hand ramp** — start chain + few features, unlock harder cells
as easy-cell realized-signal fades (i.e. they've been learned); (c) **adaptive informativeness**
— up-weight cells with a recent high non-degenerate-group fraction; (d) **hybrid** — ramp +
adaptive + the floor. **Do not** just set `p(c) ∝ σ²_c`: a pathological cell can have huge
variance without teaching anything transferable, and pure variance-chasing fights the
exploration floor. Variance is *one input*, not the objective. (Our reward also has an oracle
"accepted" flag — Part A ≥ 0.90 and Part B ≥ 0.8 — so a per-cell *accepted-rate* is a legitimate
**saturation** indicator; use it to detect "too easy," not as the primary driver.)

**Success (CPU-testable, no training):** given synthetic `reward_logs`, `update_curriculum_state`
demonstrably shifts `select_worlds` toward the currently-informative cells and away from
saturated/flat ones **while keeping every cell above the floor**, and it **never** emits a
held-out world.

### Milestone 3 — Show it helps (GPU; MENTOR-RUN)
The mentor builds a training parquet from your `select_worlds()`, runs it in SkyRL against a
**uniform-selection control**, and returns the reward logs (which feed back into 1B/2). You need
no GPU. Two tiers of success criteria — and the second is the one that counts:
- **Mechanism diagnostics (necessary, not sufficient):** vs uniform, the curriculum run has a
  higher fraction of non-degenerate groups and a steeper reward / Part-B slope, without entropy
  collapsing. **Caveat:** a curriculum can win on these *just by feeding easy worlds* — so these
  alone do not prove it helped.
- **The real objective — held-out transfer.** Under **matched compute** (same starting
  checkpoint, same rollout budget, same GRPO hyperparameters), does the curriculum produce a
  larger improvement on a **fixed held-out eval set** — `ΔR_heldout`, reported **Part A and Part
  B separately**, eventually over a few seeds? That transfer number is the project's headline
  (§1); it's what makes "the curriculum helped" a result rather than a training-curve artifact.

---

## 4. Why this is safe to hand you (and how not to break things)

Your component connects through **one thin seam**: it emits *world specs* and consumes *reward
logs*. It never touches how the model is trained. So Milestones 1–2 are fully yours and fully
CPU; a bug can at most feed a bad *mix* of worlds — it can't corrupt the reward, env, or trainer.

**House rules:** don't edit the science code (`engine.py`, `oracle_v6.py`, `sampler.py`,
`skins.py`, `reward.py`, `env.py`, `world_stream.py`, `splits.py`) — build alongside it. Never
put credentials in a file. Don't commit large dumps — but you have 100+ GB locally, so **cache
freely on your own disk** (see §5) and share only summaries/reports.

---

## 5. How to run things (CPU-only)

You need only the **science env** — numpy/pandas/scipy, **no torch/vLLM**. On your machine:

```bash
# one-time env (conda or venv), python 3.12 + these only:
pip install "numpy>=2.1" "pandas>=2.2" "scipy>=1.13"
# get the code (rpg_rl + rpg_v7_prototype); run everything from rpg_rl with:
export PYTHONPATH=../rpg_v7_prototype

# Milestone 1: generate worlds + score answer ladders (pure CPU, no model)
python world_stream.py --split train --n 200        # sanity: acceptance + distribution
# (write your ladder-scoring script using reward.py + the ladder in test_reward_integrity.py)

# Milestone 2: your module, unit-tested with SYNTHETIC reward logs (no training)
python your_test_curriculum.py
```

**Use your 100+ GB:** cache a **large audited world bank** (worlds are small JSON — you can
store tens of thousands) plus their oracle reward-ladders, so Milestone-1 analysis and
Milestone-2 experiments are instant and reproducible. World generation + audit is the only
slowish part (CPU, ms–seconds per world); caching it once pays off.

**Key files to read (all in `rpg_rl/` unless noted):** `world_stream.py` (world specs +
splits), `reward.py` (how reward is computed — your main tool), `splits.py` (train/heldout —
never leak), `catalog.py` (id-space), `test_reward_integrity.py` + `test_env_reward.py`
(ready-made answer-ladder + gold-answer builders to copy).

---

## 6. Deliverables checklist

1. **Signal-map report** (markdown + JSONL/parquet) — **1A** oracle *potential* spread by
   archetype/feature/skin/depth, and (once logs arrive) **1B** *realized* per-cell statistics
   (`μ_c`, `σ²_c`, non-degenerate-group fraction, saturation split) laid side by side. *(M1)*
2. **`curriculum.py`** (`select_worlds` + `update_curriculum_state`) with an **exploration floor**
   and **split safety**, + a **unit test** proving it adapts on synthetic logs, keeps every cell
   above the floor, and never leaks held-out worlds. *(Milestone 2)*
3. **A/B result** (with mentor) — mechanism diagnostics **and** matched-compute **held-out**
   improvement (`ΔR_heldout`, Part A / Part B separately) vs uniform. *(Milestone 3)*
4. A running log of what you tried/found (a markdown file).

---

## 7. Optional deeper reading

Not required to start — everything you need is in §0–§6 and the code files listed in §5. If you
want the academic framing once you're underway:
- **RLVE** (arXiv 2511.07317) — RL over procedurally-generated verifiable environments with an
  adaptive per-environment curriculum. This is the closest published version of your task.
- **DAPO** (arXiv 2503.14476) — the source of "dynamic sampling" (dropping zero-variance groups),
  the failure mode your curriculum is designed to reduce.
