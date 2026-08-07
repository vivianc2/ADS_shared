# RL for scientific-reasoning worlds (RPG v7): field survey, plan, and compute estimate

**Date:** 2026-08-06 · **Literature refresh:** 2026-08-06 (arXiv + Semantic Scholar sweep)
**Author:** research pass for the RPG v7 line (ADS_collab_clean)
**Scope:** How to turn the v7 world generator into an RL training environment; what the
field does for similar setups; how our project is positioned against the 2025–2026
literature; compute needed to RL-tune a ~27–32B open model in our scenario; how many
worlds/traces and how long; model recommendation.

> **Sourcing note (updated).** The first draft was written with no web access. This
> revision adds a literature sweep run against the **arXiv Atom API** and the
> **Semantic Scholar Graph API** (plus direct fetches of framework READMEs and model
> cards). Foundational 2025 papers cited here (DAPO, Dr.GRPO, GSPO, GiGPO, the
> elicitation-vs-teaching debate, the entropy-mechanism work, Search-R1/ReTool/ToRL/
> rStar2-Agent, AReaL, RLVE, NewtonBench, BoxingGym) are **high-confidence** — real,
> established, widely-cited. **Caveat on 2026-dated IDs:** the APIs in this environment
> return a forward-dated feed, and several `26xx.xxxxx` IDs are reported *as returned*
> and flagged **[verify-id]** in the reference list — the finding is real but the exact
> identifier should be confirmed before formal citation. Compute figures remain
> *engineering estimates* (arithmetic + public reference runs), shown with assumptions
> so you can re-derive and replace them with pilot measurements. Full reference list
> with confidence flags is in the Appendix.

---

## 0. TL;DR

- **Our project sits in genuinely unoccupied whitespace, and the sweep confirms it.**
  Two 2025–2026 lines converge on our exact recipe from opposite sides: **RLVE**
  (arXiv 2511.07317) does procedural-environment + verifiable-reward RL for *symbolic*
  reasoning with an adaptive per-environment auto-curriculum, and **NewtonBench /
  PhysGym / BoxingGym** do interactive, intervention-based *scientific discovery* but
  as **eval-only harnesses on frozen models**. Nobody is *RL-training* an agent on
  procedurally generated **causal-discovery** worlds with a verifiable oracle. That
  intersection is the contribution claim. See §7.
- Our setup is a textbook fit for **RLVR** (RL with verifiable rewards): the oracle
  computes the reward, so there is no reward model and no LLM judge in the loop. This
  is the single biggest thing going for us — cheap, low-variance, and the exact regime
  (math/code/agentic-with-checkable-answers) where RL has worked in 2024–2026.
- It is a **multi-turn, tool-using agent** RL problem (measure / intervene / code /
  answer), not single-turn — harder than single-turn math-GRPO but now well-supported.
  The 2025–2026 frontier for credit assignment here is **hierarchical
  (episode + step) advantage** (**GiGPO**, arXiv 2505.10978) rather than a single
  terminal reward; adopt it as the upgrade path from plain GRPO.
- **Don't start from vanilla GRPO — start from the DAPO recipe.** The field converged
  on four fixes (Clip-Higher, dynamic sampling to drop zero-advantage groups,
  token-level loss, overlong-reward shaping; arXiv 2503.14476), plus **Dr.GRPO's**
  unbiased normalization (2503.20783) for length control and **GSPO's** sequence-level
  importance sampling (2507.18071) which is specifically what stabilizes **MoE** RL.
- **Recommended stack:** `verl` remains the safe default (most active, broadest algo
  set, confirmed multi-turn agent support, scales past 30B). New in 2026 and worth
  evaluating: **AReaL 2.0** (fully-async, best agentic-integration story) and
  **SkyRL-Agent**, which trained **Qwen3-32B → SA-SWE-32B at 39.4% on SWE-Bench
  Verified** — the closest public reference run to our scale and shape. Prototype the
  environment contract in **verifiers** (`MultiTurnEnv` + `Rubric`); its abstractions
  map 1:1 to ours.
- **Model:** a **~30B MoE (≈3B active), thinking** model is the sweet spot for rollout
  economy. **Qwen3-30B-A3B** (30.5B/3.3B active, Apache-2.0) is the default; **Qwen3-32B
  dense** is the higher-ceiling, MoE-instability-free alternative; **gpt-oss-20b**
  (21B/3.6B active, Apache-2.0) is a new low-active-param option; **Qwen3-8B** (or even
  SmolLM3-3B) is the debug loop. **If you train an MoE, budget for the documented
  train/inference router-mismatch instability** (§3.3) — use GSPO and a router-replay /
  router-aware IS technique from the start.
- **Data:** you do **not** need a huge fixed dataset. Generation is on-demand and cheap,
  so the real budget is **rollout tokens**, not worlds. Plan **~2k–5k distinct audited
  worlds** (breadth) × on-the-fly reuse, ~**8–16 rollouts per world per step**, ~**1–3k
  steps** → order **1–5M graded traces** over a full run.
- **Compute:** a **single 8×H100 (80GB) node** is enough to *start* and to do a
  meaningful pilot on a ~30B-A3B model; a **serious run** wants **2–4 nodes**
  (16–32 H100) for **~1–3 weeks**. Rough envelope: **~5k–30k H100-GPU-hours** for a full
  A3B run, dominated by rollout generation. Dense 32B ≈ 3–5× that.
- **Biggest risk is not compute — it's reward hacking.** Every oracle bug is a reward
  the policy will exploit (the `two_cause` bug we fixed was exactly that). The audit
  suite is your reward-model integrity layer and must run on **every** training world.
  The 2025–2026 literature is now full of concrete failure demonstrations
  ("One Token to Fool LLM-as-a-Judge"; "specification hacking" against formal checkers)
  and mitigations — see §4.

---

## 1. What the field looks like for setups like ours

### 1.1 The paradigm: RLVR / RL-with-verifiable-rewards, and where it's moved since R1

Since DeepSeek-R1 (early 2025) the dominant recipe for "make a model reason better" is
**GRPO-family RL on tasks with a programmatic reward**:

- **Group Relative Policy Optimization (GRPO):** for each prompt, sample a *group* of G
  completions, score each with a reward function, set each completion's advantage to its
  group-normalized reward `(r_i − mean)/std`. No value/critic network (unlike PPO),
  which halves memory and is why it dominates open reasoning RL.
- **Verifiable reward:** computed by code — a math checker, unit tests, a compiler, or
  (in our case) the SCM oracle. No learned reward model → no reward-model drift, no
  judge cost, low variance. This is the regime where RL reliably improves reasoning;
  where it's shakiest is exactly where the reward is a learned/LLM judge.

**Our environment is squarely in the strong regime.** `oracle_v6.grade()` already
returns `benefit_recovered` (part A, fraction of achievable utility, ∈[0,1], 0.90 bar)
and `battery_fraction` (part B, mechanism/proxy/decoy correctness, 0.8 bar) — a dense,
checkable, low-variance reward with no judge. Most groups spend enormous effort building
a verifier; we already have one, plus an audit suite that proves it's sound.

**What's changed since the first draft — don't run vanilla GRPO.** The 2025–2026
literature converged on a standard, better-behaved recipe. Adopt it:

- **DAPO** (arXiv 2503.14476, Mar 2025, the reference open scale-up system): four fixes
  that are now table stakes — **Clip-Higher** (decoupled clip ranges to preserve
  exploration), **dynamic sampling** (drop prompt groups whose rollouts are all-correct
  or all-wrong so advantages aren't zero — directly kills our "zero-variance group"
  failure mode), **token-level policy-gradient loss**, and **overlong-reward shaping**.
- **Dr.GRPO** (2503.20783): removes GRPO's length and std-dev normalization terms, which
  otherwise inflate response length without accuracy gains. Use it if you see creeping
  length. A 2026 unifying result (**"GRPO, Dr.GRPO, and DAPO are three operations on one
  number"**, 2607.00152 [verify-id]) shows these three differ only in how they use the
  group std-dev — pick the tradeoff deliberately; a companion impossibility result
  (2607.23364 [verify-id]) proves you cannot be both gradient-unbiased and
  length-invariant with outcome rewards alone.
- **GSPO — Group Sequence Policy Optimization** (2507.18071, Qwen team): moves the
  importance-sampling ratio from **token level to sequence level**, matching the unit of
  the reward. Reported to be more stable than GRPO and, critically, **to stabilize MoE
  RL** — credited in the Qwen3 models. This is the most relevant refinement if you train
  a 30B-A3B MoE (§3.3).
- **"Tricks or Traps? A Deep Dive into RL for LLM Reasoning"** (2508.08221): an
  evidence-based ablation arguing a *minimalist* two-technique combination can match full
  GRPO/DAPO — a useful filter against cargo-culting every trick.

### 1.2 Calibrate expectations: does RLVR teach or just elicit?

A major 2025–2026 debate directly bears on what a training run can buy us, and the
report should state its position on it:

- **"Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"**
  (2504.13837): using pass@k at large k, argues RLVR does **not** create fundamentally
  new reasoning — it **sharpens/samples more efficiently within the base model's
  support**, and can even shrink the reachable solution set; distillation is what
  expands capability.
- **"Spurious Rewards: Rethinking Training Signals in RLVR"** (2506.10947): RLVR can
  improve benchmarks even with random/incorrect rewards, attributed to GRPO clipping
  bias amplifying pre-existing pretraining behaviors — and the effect is highly
  model-dependent (strong on Qwen, weak elsewhere). A caution that your reward may not
  be doing what you think.
- Follow-ups reinforce the "sharpening" view (verifier-induced support reshaping,
  2608.00220 [verify-id]; "BODHI" on constricted continuation space, 2608.02867
  [verify-id]) and give a mechanism (clipping bias reduces entropy and activates
  memorization circuits, 2512.16912 / 2601.11061 [verify-id]).

**Implication for us.** Two things. (1) Validate the verifier and report **pass@k**, not
just pass@1, so we can tell sharpening from genuine gain. (2) Our strongest defense
against the "it only elicits" critique is *breadth of fresh structure*: if RL merely
re-weights base-model behaviors, then held-out-skin / held-out-archetype transfer is the
number that distinguishes "learned to reason about hidden causes" from "sharpened a
memorized ritual." This is exactly what the v7 structural sampler is for, and it's why
the transfer eval (§6) is the headline result. It also means a **strong base model
matters more than a long RL run** — invest in model choice (§3).

### 1.3 Where our problem sits: multi-turn agentic RL, and its credit-assignment frontier

Most published reasoning-RL is **single-turn**: one long CoT, one answer, one reward.
Ours is **multi-turn agentic**: the policy runs a *loop* — `measure` → `intervene` →
`code` → `answer` — with environment feedback (assay readings, resolver echoes) between
turns, reward at the end. This is the "LLM-as-agent RL" frontier. Key facts:

- It works but is **noisier and more expensive** than single-turn: rollouts are long,
  credit assignment is over a trajectory, and the generation:train compute ratio is even
  more lopsided toward generation.
- **Tool-integrated RL is now well-mapped.** The foundational quartet — **Search-R1**
  (2503.09516), **ToRL** (2503.23383), **ReTool** (2504.11536), and **rStar2-Agent**
  (2508.20722) — establish that outcome-reward RL teaches a base model *when and how* to
  call tools (search / code execution). rStar2-Agent reached frontier math with a 14B
  model and only **~510 RL steps**, and introduced **GRPO-RoC (Resample-on-Correct)**:
  oversample rollouts, keep the correct-trajectory ones, to stabilize learning against
  **noisy tool feedback**. RoC transfers directly to our `code`/`intervene` loop.
- **Credit assignment has moved below the trajectory.** The 2025–2026 thread is
  hierarchical, critic-free advantage:
  - **GiGPO** (2505.10978, NeurIPS 2025) — the anchor. Two-level advantage: an
    episode-level group (like GRPO) *plus* a step-level group that reuses repeated /
    anchor states across rollouts for fine-grained credit. Reports >12% over GRPO on
    ALFWorld, ~9% on WebShop. **Build the trainer around this**: our
    measure/intervene/code/answer loop provides natural turn boundaries and repeated
    observations that GiGPO's step-grouping exploits.
  - **Milestone / turn-level variants** (BEACON milestone partitioning, 2605.06078
    [verify-id]; turn-level credit TCPO/TL-GRPO; ECHO's *posterior-sensitive* turn
    rewards for info-seeking agents, 2606.29745 [verify-id]) densify the signal. **ECHO
    is notably on-point** — it rewards turns by how much they reduce posterior
    uncertainty, which is precisely what a hidden-cause-discovery agent should be doing.
  - **Caution on dense credit:** "When Denser Credit Is Not Enough" (ECPO, 2606.05885
    [verify-id]) shows naive step-level credit is *unreliable under few rollouts* and
    must be calibrated. Relevant since our group size G is modest.
- **Async rollout + inference-server separation** (vLLM/SGLang serving rollouts while the
  trainer does gradient steps) is the standard efficiency pattern and is what makes agent
  RL affordable. **AReaL** (2505.24298) is the canonical fully-async substrate;
  **SkyRL-Agent** (2511.16108) is the closest public reference to our setting — it trains
  long-horizon multi-turn tool agents and produced **SA-SWE-32B (from Qwen3-32B) at 39.4%
  Pass@1 on SWE-Bench Verified** using a verl backend.

### 1.4 The frameworks (2026 state, grounded from READMEs)

| Framework | Latest activity | Algorithms | Multi-turn / agent | Scale | Notes for us |
|---|---|---|---|---|---|
| **verl** (verl-project org) | active, 2026/07 | PPO, **GRPO**, **GSPO**, **DAPO**, Dr.GRPO, PRIME, RLOO, REINFORCE++, Clip/KL-Cov | **Yes** — multi-turn tool-calling rollout (SGLang), verl-agent, ReTool/Search-R1 recipes | up to 671B / Qwen3-235B | **Still the safe default.** Widest algo coverage, mature multi-turn, Qwen3 examples. |
| **AReaL / AReaL 2.0** (Ant + Tsinghua) | 2026/07 (2.0 refactor) | GRPO, GSPO, DAPO, IcePop, KPop… | **Yes** — SWE/search/tool-use; OpenAI Agents SDK + CAMEL integrations | 7B/32B, 235B MoE | **Top new entrant.** Fully-async by design (~2.77× speedup reported); best agentic-integration story. Evaluate for long-horizon. |
| **SkyRL / SkyRL-Agent** (Berkeley + Anyscale) | 2026/02 | GRPO-family | **Yes** — explicitly long-horizon multi-turn agents; verl backend | 32B demonstrated (SWE-Bench) | **Closest reference stack to us.** Async + in-flight weight updates; Tinker API. |
| **NeMo-RL** (NVIDIA) | v0.6 2026/04 | GRPO, GSPO, DAPO, DPO, distillation | **Yes** — multi-turn w/ tool use, games | Qwen2.5-32B on 32 nodes; Nemotron Nano-30B-A3B | NVIDIA-stack reproducibility; strong 32B GRPO recipes. Apache-2.0. |
| **ROLL** (Alibaba Taotian) | OSDI'26, 2026/06 | 20+ incl. StarPO (trajectory) + **GiGPO** (step-wise) | **Yes** — games, dialogue, tool use | Qwen3-MoE 235B, thousand-GPU | Megatron 5D parallelism; ships GiGPO natively. Apache-2.0. |
| **OpenRLHF** | v0.10 2026/04 | PPO, GRPO, Dr.GRPO, RLOO, REINFORCE++-baseline | **Yes** — single/multi-turn, OpenAI-compatible agent server | 70B+ (Molt → 100s B) | Clean agent API; good if you like Ray. |
| **slime** (THUDM) | behind GLM-4.5→5.2 | GRPO-family | **Yes** — multi-agent, search/RAG, SWE sandboxes | frontier (GLM/DeepSeek) | Megatron + SGLang only; production pedigree. |
| **prime-rl + verifiers** (Prime Intellect) | 2025+ | GRPO (AIPO off-policy loss) | **Yes** — agentic (Wordle, tool-calling, web) | 1T+ MoE targets; lists Qwen3-30B-A3B | **Best abstractions for us.** verifiers' Environment/Rubric/Parser map 1:1 to sim/oracle/resolver; prime-rl is the trainer. Prototype the contract here. |
| **TRL** (HuggingFace) | active | **GRPO**, DPO, KTO, SFT, Reward | **Yes (new)** — multi-env agentic RL, env-owned rewards (Harbor/OpenEnv) | DeepSpeed + PEFT/LoRA; no frontier claim | Easiest to read/hack; great for the small-model debug loop. |

**Reward-function contract is the same everywhere:** a function taking `prompts`,
`completions`, plus dataset columns as kwargs, returning `list[float]`; multiple reward
functions summed with weights — exactly our part-A + part-B decomposition.

### 1.5 Failure modes the field has hit (updated with citations)

1. **Reward hacking / spec gaming.** The policy optimizes the *reward*, not your intent;
   any oracle bug becomes an exploited shortcut. The literature now has vivid
   demonstrations: **"One Token to Fool LLM-as-a-Judge"** (2507.08794) shows a lone
   symbol or a generic "Let's solve this" opener fools generative verifiers;
   **specification hacking** against formal Dafny/Lean checkers (2605.30914 [verify-id])
   is the direct analog of exploiting a programmatic-oracle gap. *Mitigation:* full audit
   suite on every training world; keep part B (counterfactual battery) so
   right-utility/wrong-mechanism is penalized; adversarially spot-check high-reward
   trajectories; **test that trivial/degenerate answers fail the grader before any RL
   run** (see §4).
2. **Length / format hacking.** GRPO inflates output length or spams tool calls if it
   correlates with reward. *Mitigation:* per-turn budget + `productive turn` accounting;
   Dr.GRPO's unbiased normalization; a small length/again cost.
3. **Reward collapse / zero-variance groups.** If every rollout in a group gets the same
   reward, advantage = 0 → no gradient. *Mitigation:* **dense** reward (we have both
   `benefit_recovered` and `battery_fraction` ∈ [0,1]) so groups spread out; **DAPO
   dynamic sampling** to drop dead groups; curriculum so worlds are neither trivial nor
   impossible.
4. **KL blowup / entropy collapse** — the *dominant* stability failure in 2025–2026.
   **"The Entropy Mechanism of RL for Reasoning LMs"** (2505.22617) derives entropy
   dynamics from logit–advantage covariance and proposes **Clip-Cov / KL-Cov** to prevent
   collapse; asymmetric clipping (clip-low raises entropy, clip-high lowers it,
   2509.26114 [verify-id]) and quantile-baseline advantage (QAE, 2509.22611 [verify-id])
   are cheaper knobs. Over-trained SFT init *predicts* collapse (2606.18487 [verify-id]) —
   don't over-SFT before RL. **Monitor policy entropy as your primary health metric.**
5. **Train/inference mismatch.** vLLM-generated tokens vs trainer logprobs differ;
   uncorrected, this biases the gradient. Keep **truncated importance sampling** on
   (TRL/verl default). For MoE this is worse — see §3.3.
6. **Multi-turn credit dilution.** Long trajectories with a single terminal reward learn
   slowly. *Mitigation:* keep trajectories short (tight budget); adopt **GiGPO** step-level
   advantage rather than hand-shaping (§1.3, §4).
7. **Overfitting to a motif / memorization.** If worlds are too similar the policy learns
   the *ritual*, not reasoning — and per §1.2 this is precisely what critics say RLVR
   mostly does. *This is why the v7 structural sampler exists.* Non-leaking names + fresh
   graphs per episode + multiple archetypes force generalization; held-out
   skins/archetypes are the eval that proves it.
8. **Evaluation contamination.** Never RL on worlds you evaluate on. Hold out whole
   **skins** and whole **archetypes** for the transfer eval.

---

## 2. How our environment maps onto the RL stack

Good news: v6/v7 already implement ~90% of an RL environment. The mapping:

| RL concept | Our existing piece |
|---|---|
| Environment / episode | `SimV6` over one generated world (`run_agent_v6.run_world` is the rollout loop) |
| Action space | free-text `measure` / `intervene` / `code` / `answer` / `give_up`, resolved by `resolver.py` |
| State / observation | scenario prose + assay readings + resolver echoes + code stdout |
| Transition | `SimV6.measure()` / `intervene()` (with selection/policy-engine support) |
| **Reward (verifiable)** | `oracle_v6.grade()` → `benefit_recovered` (A, 0.90 bar) + `battery_fraction` (B, 0.8 bar) |
| Reward integrity | the 5 audits + artifact-check (`_artifact_check`) |
| Dataset row | one world JSON (`world_*.json`) = a "prompt" with `info` = stored gold/battery |
| Curriculum knob | archetype × features × depth in the sampler |
| **Turn / step boundary** (new) | each `measure`/`intervene`/`code`/`answer` turn = a GiGPO step; repeated observations = anchor states |

**What needs building for RL (the gap):**

1. **A thin environment adapter** exposing the rollout loop to the trainer's interface.
   In verifiers it's a `MultiTurnEnv` subclass (`env_response`, `is_completed`) + a
   `Rubric` whose reward functions call `sim.grade()`; in verl it's a multi-turn rollout
   worker + reward fn. A wrapper, not a rewrite — `run_world` already *is* the loop.
2. **Advantage / reward-shaping decision** (see §4): terminal-only vs GiGPO step-level.
3. **Batched, async rollouts.** Today `run_batch_v6` runs worlds serially for eval. For RL
   we need N parallel rollouts per step. The trainer handles this; we mostly need the env
   process-safe and fast to construct (it is — gold is precomputed and stored in the world
   JSON, so no per-episode oracle recompute). Prefer an **async** trainer (AReaL/SkyRL
   style) because our tool round-trips make rollouts long and variable-length.
4. **A world *stream*, not a fixed file set.** Generate worlds on demand (seeded) so we get
   fresh structure every step; cache the audited ones. The generator is fast and emits
   self-contained records. This is exactly RLVE's "adaptive verifiable environments" model
   (§7).
5. **The resolver in the loop.** The resolver's LLM fallback is a second model call per
   ambiguous action. For training throughput, prefer the **lexical resolver** (hardened,
   incl. short scientific names) and reserve the LLM fallback for eval — else resolver cost
   dominates rollout time.

---

## 3. Model choice for a ~27–32B RL run

### 3.1 The user's model: Qwen3.6-27B

A current-generation thinking-capable model in the class that RL reliably improves, and
the code already has presets for it (`openai_llm.py`, temp 1.0 / top_p 0.95). Fine
default if inertia matters. **Flag from the sweep:** a `Qwen/Qwen3.6-27B` card was
retrievable (27–28B, dense, thinking on by default, Apache-2.0, and — notably — a much
larger native context, ~256K) but it was **single-source and could not be
cross-confirmed** against the Qwen blog; a sibling probe returned 401. Confirm dense-vs-
MoE and the active-param count before budgeting, since that determines rollout cost
(§5.2). If it is dense ~27B, budget closer to the 32B-dense envelope.

### 3.2 If choosing fresh — the trade the field actually makes

The decisive factor for **RL cost** is not total params but **active params during
rollout generation**, because generation is 70–90% of RL compute (§5). That strongly
favors a **MoE with few active params** — *if* you pay the MoE-RL stability tax (§3.3).

| Candidate | Total / active | Why | Watch-outs |
|---|---|---|---|
| **Qwen3-30B-A3B** (thinking) | 30.5B / **3.3B active** (128 experts, 8 active) | **Best rollout economy.** Generates like a ~3–4B model, reasons like ~30B. Apache-2.0, hybrid `enable_thinking`. | MoE RL router instability (§3.3) — use GSPO + router replay from day 1. |
| **Qwen3-32B dense** (thinking) | 32.8B / 32.8B | Highest single-model ceiling; simplest to train (dense, well-trodden); **no router drift**. Apache-2.0, 32K→131K ctx. Reference for SkyRL SWE run. | ~10× the rollout FLOPs of A3B → materially more expensive per trace. |
| **gpt-oss-20b** (new) | 21B / **3.6B active** MoE | **Cheapest credible rollouts.** Configurable reasoning effort (low/med/high), Apache-2.0, comfortably single-node. | Newer/less-trodden for RL than Qwen3; context length not stated on card. Confirm before committing. |
| **Qwen3.6-27B** (user's) | ~27B (dense per single-source card) | Newest, already wired up. | Confirm dense vs MoE + active-param count (§3.1). |
| **Magistral-Small-2509** | 24B dense | Solid dense alternative, Apache-2.0, `[THINK]` tokens, SFT+RL base. | Context degrades past ~40K. |
| **7–8B (Qwen3-8B) / SmolLM3-3B** | 8B / 3B dense | **The debug loop.** Iterate env/reward/harness for hours, not days. Both Apache-2.0, thinking toggle. | Not the final artifact; lower ceiling. |

Out of single-node scope (awareness only): DeepSeek-R1 / R1-0528 (671B-class, MIT),
GLM-4.6 (357B), Kimi-K2 (1T, not a reasoning model), MiniMax-M2 (230B/10B-active),
Llama-4-Scout (109B, non-Apache license). The DeepSeek-R1-Distill-Qwen-7B/14B/32B
checkpoints (Apache-2.0 bases) are the relevant DeepSeek-lineage option at our scale.

### 3.3 MoE-RL stability (new — read before choosing an MoE)

A genuinely active 2025–2026 subfield confirms MoE-with-few-active-params needs care
during RL. The dominant documented failure is a **train/inference router mismatch**: the
expert-routing decision differs between the training forward pass and the vLLM/SGLang
rollout, "even leading to catastrophic RL training collapse" (2510.11370 [verify-id]).
Related diagnoses: **router drift / staleness** (PR2 predictive routing replay,
2606.00395 [verify-id]), and fixes via **router-aware importance-sampling correction**
(2510.23027 [verify-id]) and **Routing Replay** (2512.01374 [verify-id]). **Ring-lite**
(2506.14731) — a 16.8B/2.75B-active MoE reasoning model — explicitly names "undocumented
challenges in MoE RL training" and stabilizes with C3PO. **Practical upshot:** if you
train Qwen3-30B-A3B or gpt-oss-20b with RL, adopt **GSPO** (sequence-level IS, designed
for this) plus a router-replay / router-aware-IS technique *from the start*, not
reactively. If you'd rather avoid the whole failure class, **Qwen3-32B dense** trades
rollout cost for far simpler RL dynamics.

### 3.4 Recommendation

Do env/reward/curriculum bring-up on a **7–8B** model (fast, cheap, catches 90% of
harness bugs). Run real training on a **30B-A3B thinking MoE** for rollout economy
(with §3.3 mitigations); keep **32B dense** as the "if A3B plateaus or router instability
bites, spend more / de-risk" option, and **gpt-oss-20b** as a low-active-param
alternative. Either way use the **thinking/reasoning** variant — RL on a model that
already emits long CoT is far more sample-efficient than inducing reasoning from a
non-thinking base, and it maximizes the base-model support that §1.2 says RLVR sharpens.
Staying in the Qwen3 family keeps your presets, tokenizer, and thinking-mode plumbing.

---

## 4. Reward design (this is where the project wins or loses)

We already have the pieces; the design choices:

**Base reward (recommend terminal, dense):**
```
r = w_A · benefit_recovered            # part A: fraction of achievable utility, ∈[0,1] (clip <0 → 0)
  + w_B · battery_fraction             # part B: mechanism/proxy/decoy correctness, ∈[0,1]
  − c_len · (tokens or turns over budget)   # light anti-length/anti-spam
  − c_hack · artifact_suspect                # penalize unresolvable/hallucinated answers
```
Start `w_A = w_B = 0.5`. This is a per-trajectory scalar — exactly the `list[float]` that
GRPO/TRL/verl want, and it's **dense** (not binary accept), which keeps groups from
collapsing to zero variance.

**Why keep part B.** Part A alone is hackable: an agent can stumble onto the right utility
without understanding the mechanism (the "acts right, explains wrong" pattern seen across
every v6 batch). Part B (counterfactual battery) rewards naming the true proxy, rejecting
the decoy, and getting actuator signs right. Together they reward *understanding*, which
is the training target. This is our built-in answer to the "reward hacking" literature:
part B is a second, orthogonal verifiable check that a part-A shortcut can't satisfy.

**Process / step-level reward — the updated recommendation.** The first draft said "use
sparingly." The 2025–2026 credit-assignment work refines this into a concrete, *lower-
risk* option than hand-shaped process rewards:

- **Prefer structural step-credit (GiGPO, 2505.10978) over hand-authored process
  rewards.** GiGPO derives step-level advantage from *group statistics over repeated
  states* — it does not require us to encode "the right experiment," so it densifies the
  signal without teaching a fixed ritual (the overfitting trap). Our loop's repeated
  observations (same assay reached via different action orders) are exactly what it needs.
- **If you do add an explicit process term, keep it minimal and posterior-based.** ECHO
  (2606.29745 [verify-id]) rewards a turn by how much it *reduces posterior uncertainty
  about the hidden cause* — a principled, non-ritual signal well-matched to
  hidden-cause-discovery. At most add one small term: "did the trajectory gather
  identifying evidence before answering" (e.g., ≥1 intervention on a causally-relevant
  lever).
- **Heed the dense-reward warnings.** "The Dark Room in the Reward Channel" (2607.21273
  [verify-id]) found dense per-step rewards can *collapse* training under standard
  std-normalization — "the delivery channel, not the content, decides." "When Denser
  Credit Is Not Enough" (2606.05885 [verify-id]) shows dense credit is noisy under few
  rollouts. So: densify via **GiGPO's group structure**, not via a large hand-tuned
  process bonus, and calibrate.
- **Copy rStar2-Agent's Resample-on-Correct (GRPO-RoC, 2508.20722)** for the noisy
  `code`/`intervene` feedback: oversample rollouts, keep correct-trajectory ones. Cheap
  stability without a learned PRM.

**Curriculum.** Difficulty is already parameterized. Start easy (confounded_chain,
depth-2, single feature), ramp to hard (collider_selection / hidden_subtype, depth-4,
`two_cause + sign_flip`). Make it **adaptive** (raise difficulty as batch pass-rate
climbs) — this is precisely RLVE's per-environment adaptive difficulty (§7), and Chart-RL
(2603.06958 [verify-id]) finds task *difficulty*, not data quantity, drives transfer.
Cheap because generation is on-demand.

**Reward integrity = your #1 job.** Concrete checklist, in priority order:
1. **Master-key test first.** Before any RL, feed the grader trivial/degenerate answers
   (empty, generic opener, all-decoy) and confirm they score ~0 ("One Token to Fool",
   2507.08794). Cheapest highest-value check.
2. **Full audit suite (5 audits + artifact-check) on every world that enters training.**
   This is the reward-model integrity layer; a silent oracle bug is found and amplified
   by the policy far faster than by human review (the `two_cause` bug is the cautionary
   tale).
3. **Adversarial oracle tests.** For the programmatic checker, add negative test cases
   the way the formal-verification-hacking work (2605.30914 [verify-id]) recommends —
   assume the policy will find any spec gap.
4. **Periodically audit top-reward trajectories** for spec gaming.
5. **Report pass@k, not just pass@1** (§1.2), so sharpening is distinguishable from gain.

---

## 5. Compute, data, and time (the numbers you asked for)

> **Estimates from first-principles arithmetic + public reference runs**, not benchmarked
> on our exact setup. Assumptions shown so you can re-derive and replace them after the
> first pilot. Note: the sweep could **not** retrieve reliable published GPU-hour/step
> figures for 30B-class reasoning RL (APIs rate-limited on those queries), so the numbers
> below remain arithmetic-derived. The one solid *anchor* data point is **rStar2-Agent:
> frontier math at 14B in ~510 RL steps** — evidence that with a good verifier, step
> counts are modest.

### 5.1 The arithmetic that dominates everything: rollout tokens

RL cost ≈ **generation cost** (rollouts) + training cost (gradient steps); for
reasoning/agent RL the split is typically **~70–90% generation**. So:

```
total generated tokens ≈ steps × prompts_per_step × G × tokens_per_trace
```

Per-trace token count is **large** — multi-turn with thinking: long CoT per turn
(hundreds–thousands of tokens) × ~5–15 turns + environment text fed back each turn.
Estimate **~8k–20k tokens per completed trace** (~**12k** midpoint).

**A concrete "serious run" scenario:**
| Quantity | Value | Note |
|---|---|---|
| Optimization steps | 2,000 | rStar2 hit frontier in ~510; 1–3k is a full run |
| Prompts (worlds) per step | 64 | distinct worlds sampled per step |
| Group size G | 8 | rollouts per world (GRPO group) |
| Traces per step | 512 | 64 × 8 |
| **Total traces** | **~1.0M** | 2,000 × 512 |
| Tokens per trace | ~12k | multi-turn + thinking |
| **Total generated tokens** | **~12.3B** | the real cost driver |

Pilot (500 × 32 × 8) ≈ **128k traces ≈ ~1.5B tokens**. Larger run (3k × 128 × 16) ≈
**6M traces ≈ ~70B tokens**. So: **order 1–5M traces** for a full run, **~100k–200k**
for a first real pilot.

### 5.2 Turning tokens into GPU-hours

Rollout throughput on vLLM/SGLang for a **~3B-active MoE (Qwen3-30B-A3B)** on an H100 is
on the order of **~2,000–5,000 generated tok/s per GPU** at RL batch sizes (very
setup-dependent; dense 32B is ~5–10× slower per token). Using a deliberately conservative
**~2,000 tok/s/GPU**:

- Full run generation: 12.3B ÷ 2,000 ≈ **~1.7M GPU-seconds ≈ ~475 GPU-hours** for
  generation on A3B; add training + overhead (generation is ~80%) → **~600 GPU-hours** of
  *pure compute*.
- Real runs are far below 100% utilized (rollout stragglers, tool round-trips through our
  sim/resolver, sync barriers). Applying a realistic **3–8× wall-clock inflation** →
  **~2,000–5,000 H100-GPU-hours** for a full A3B run. **Async trainers (AReaL/SkyRL) exist
  precisely to shrink this inflation factor** — the ~2.77× async speedup AReaL reports is
  the difference between the low and high end here.
- **Dense 32B:** multiply generation by ~5–10× → **~15,000–40,000 GPU-hours**. Single
  biggest reason to prefer the A3B MoE.

**Envelope to quote:** a full RPG-v7 RL run on a **30B-A3B** model is roughly
**5k–30k H100-GPU-hours** (wide because agent-RL utilization varies); a **32B dense**
run is **~3–5× that**.

### 5.3 Hardware and wall-clock

| Setup | GPUs | Good for | Wall-clock |
|---|---|---|---|
| **Debug loop** | 1–2× H100 (or A100 80GB) | 7–8B model, env/reward/harness bring-up, tiny runs | hours |
| **Pilot** (recommended first) | **1 node = 8× H100 80GB** | 30B-A3B, ~500 steps, prove learning curve moves | **~3–7 days** |
| **Serious run** | **2–4 nodes = 16–32× H100** | 30B-A3B, 1–3k steps, held-out eval | **~1–3 weeks** |
| Dense 32B serious run | 4–8 nodes | higher ceiling | 3–6 weeks |

Standard topology (all frameworks): dedicate **1–2 GPUs (or a node) to the vLLM/SGLang
rollout server**, the rest to the FSDP/Megatron trainer, run rollouts **async** so
generation and training overlap. On a single 8×H100 node a common split is 2 serving +
6 training (colocate mode also works for a first pilot). A **30B-A3B model fits
comfortably** in RL on one 8×H100 node (weights + optimizer + KV cache + vLLM); a **32B
dense** full-parameter RL is tight on one node → 2+ nodes, or LoRA-RL on one node
(verl/TRL support it, at some ceiling cost).

### 5.4 How many worlds and traces

- **Worlds (breadth):** you do **not** need millions. Value is *structural diversity*;
  beyond a few thousand distinct graphs the marginal world teaches little. Target
  **~2k–5k distinct audited worlds** in the pool (10 skins × 3 archetypes × feature combos
  × seeds gives this easily), **generated on demand** and cached. Hold out **≥2 skins and
  ≥1 archetype entirely** for the transfer eval.
- **Traces (the actual cost):** **~100k–200k** for a first real pilot, **~1–5M** for a full
  run (§5.1). Worlds are reused across steps and across the group, so trace count ≫ world
  count — that's fine (reliability within a structure + breadth across structures).
- **Rule of thumb:** scale **traces** to your compute budget; scale **worlds** to the
  diversity you need for generalization. Decoupled.

### 5.5 Cost intuition (if renting)

At rough market rates (~$2–3/H100-hr on-demand, less reserved): a **pilot** on one 8×H100
node for ~5 days ≈ **~1,000 GPU-hours ≈ $2–3k**. A **full A3B run** at ~5k–30k GPU-hours
≈ **~$10k–90k**. A **32B dense** full run ≈ **$50k–300k**. The spread is real and
dominated by (a) dense vs MoE and (b) how well you overlap rollout and training. **This
is why the pilot exists: measure our actual tok/s and utilization before the big run.**

---

## 6. Recommended path

1. **Environment adapter (1–2 weeks eng).** Wrap `run_world` as a `verifiers.MultiTurnEnv`
   (fastest to prototype the contract) *and/or* a verl multi-turn rollout worker. Reward =
   `sim.grade()` → `w_A·benefit + w_B·battery − penalties`. Lexical resolver in-loop; LLM
   resolver only at eval.
2. **World stream + adaptive curriculum.** On-demand seeded generation with the full audit
   gate; difficulty schedule over archetype/features/depth; **adaptive per-environment
   difficulty** (RLVE-style, §7); hold out skins + archetypes.
3. **Debug loop on 7–8B.** Get a *moving learning curve* on a tiny run. Verify: reward
   goes up, **entropy stays bounded** (primary health metric), groups have non-zero
   variance (use DAPO dynamic sampling), no obvious hacking in top-reward traces, and the
   master-key test passes.
4. **Pilot on 30B-A3B, 1 node, ~500 steps** with the **DAPO recipe + GSPO** (and MoE
   router-replay if MoE). **Measure real tok/s, trace length, and utilization** — replace
   §5 estimates with numbers. Confirm the learning curve *and* that held-out transfer +
   **pass@k** improve, not just pass@1.
5. **Add GiGPO step-level advantage** if terminal-reward credit is too slow (§4); consider
   an async trainer (AReaL/SkyRL) to cut wall-clock.
6. **Scale to 2–4 nodes for the full run** only after the pilot's curve and eval look
   right. Keep the audit suite on every world; periodically audit top-reward trajectories.
7. **Eval throughout** on held-out skins/archetypes with `run_batch_v6 + analyze_results`
   (per-archetype accept/partA/partB, artifact flags). **Also evaluate on ≥1 external
   benchmark** (NewtonBench / BoxingGym / PhysGym / Extended Corr2Cause, §7) to validate
   the internal oracle and pre-empt the "your benchmark is bespoke" reviewer. That transfer
   number is the headline result: did *scientific reasoning* generalize, or did the model
   memorize a motif?

---

## 6.5 The 3-week plan: Tinker, and a time-boxed schedule (new)

**Constraint: 3 weeks, one person, end-to-end.** This changes the dominant risk from
*compute* to *infra bring-up*. A verl/SkyRL run on a raw GPU cluster costs 1–2 weeks just
to stand up FSDP + vLLM + async rollouts + checkpointing before a single learning curve
moves — which would eat the whole budget. So the top recommendation for a 3-week timeline
is a **managed training API**, and the sweep found that **Tinker (Thinking Machines)**
fits our situation almost exactly.

### 6.5.1 Why Tinker for this deadline

- **Our model is a first-class citizen.** Tinker's live model list includes
  **`Qwen/Qwen3.6-27B`** (the user's model — dense, medium, hybrid-thinking, 64K ctx) and
  **`Qwen/Qwen3.6-35B-A3B`** (MoE, ~3B active — the current replacement for the
  now-retired Qwen3-30B-A3B I recommended in §3, which Tinker retired 2026-06-12). Also
  available at our scale: `Qwen3.5-9B`, `Qwen3.5-4B`, `Qwen3-8B`, `gpt-oss-20b/120b`,
  `Nemotron-3-Nano-30B-A3B`.
- **Its RL abstraction is our loop.** Tinker's `Env` interface is
  `initial_observation()` + `step(action) → StepResult(reward, episode_done,
  next_observation)`, grouped into `EnvGroupBuilder.compute_group_rewards()`. That is a
  near-literal wrapper around `run_agent_v6.run_world` + `oracle_v6.grade()` — the
  environment adapter (§2, item 1) becomes an afternoon, not a week. Multi-turn, tool-use
  (`AgentToolMessageEnv`, `build_agent_tool_env`), and group-based GRPO advantage
  (reward centering across a group) are built in; the loss is configurable (`cispo`, PPO).
- **You write the loop locally; Tinker runs the GPUs.** The five-step loop (sample batch
  → rollout → `compute_advantages` (GRPO) → `forward_backward` + `optim_step` → eval/
  checkpoint) runs on your laptop; the forward/backward/sample primitives execute
  remotely. No cluster, no FSDP, no vLLM ops.
- **It's LoRA-first.** Per §1.2, RLVR mostly *sharpens/elicits* existing base-model
  behavior rather than teaching new capability — which is precisely the regime where
  **LoRA-RL loses little** relative to full fine-tuning. LoRA also sidesteps the MoE
  router-instability tax (§3.3) being your problem to solve. Good fit.
- **Cost is per-token, not per-GPU-hour** ("MoE priced by active params"), so the §5
  token arithmetic converts *directly* to a dollar figure with no utilization guesswork —
  the single biggest source of spread in the §5 estimate disappears.

**Tradeoffs to accept:** you don't control the kernels/parallelism; there's no
full-parameter frontier run; and you're betting on a hosted service's availability and
model list for your critical path. For a 3-week research signal, that trade is right. Keep
verl/SkyRL as the "if this becomes a real training effort" scale-out path (§1.4).

### 6.5.2 Model recommendation for 3 weeks

**Primary: `Qwen3.6-35B-A3B` (MoE, ~3B active) on Tinker with LoRA-RL.** Reasoning ceiling
of a ~35B model, generation cost of a ~3B model (Tinker prices it in the "Medium" tier,
roughly on par with `Qwen3.5-9B` and far below the `Qwen3.6-27B` dense model per token).
It's smart enough for meaningful signal on collider/subtype worlds, and cheap enough to
iterate. **Debug on `Qwen3-8B` or `Qwen3.5-4B`** (cheapest tiers) to shake out the
env/reward/harness. **Keep `Qwen3.6-27B` (dense) as the comparison point** since you
already have presets and it's the model you know — but expect ~3–4× the per-token cost of
the A3B, so use it for the final confirmatory run, not the iteration loop.

> If Tinker access isn't available: fall back to **Qwen3.6-27B / Qwen3.5-9B on a single
> 8×H100 node with verl + LoRA**, and budget the first week for infra. This is the
> higher-risk path for a 3-week deadline.

### 6.5.3 Time-boxed schedule (3 weeks)

| Days | Milestone | Deliverable / exit criterion |
|---|---|---|
| **1–3** | **Env adapter + reward** | `RPGEnv(Env)` wrapping `run_world`; `compute_group_rewards` → `w_A·benefit + w_B·battery − penalties` from `sim.grade()`. **Master-key test passes** (trivial answers score ~0, §4). Lexical resolver in-loop. |
| **4–5** | **Debug loop on 8B/4B** | A *moving learning curve* on ~50 worlds, ~50–100 steps. Verify: reward ↑, **entropy bounded**, groups non-zero variance (DAPO dynamic sampling), no obvious hacking in top traces. This is where 90% of harness bugs die. |
| **6–8** | **World stream + curriculum** | On-demand seeded generation behind the full audit gate; adaptive difficulty over archetype/features/depth; **hold out ≥2 skins + ≥1 archetype**. Confirm mock replays partA≈1.0. |
| **9–14** | **Main run on 35B-A3B** | GRPO/CISPO + DAPO fixes. Target **~500–1,000 steps × 32–64 worlds × G=8**. **Measure real trace length + per-step cost first** (days 9–10) then commit. Checkpoint + eval every ~50 steps on held-out set (pass@1 *and* pass@k). |
| **15–18** | **Analyze + iterate** | Read the transfer curve. If flat: check for hacking, raise/lower difficulty, adjust `w_A/w_B`, try GiGPO step-credit (§4). One or two re-runs from a good checkpoint. |
| **19–21** | **External eval + write-up** | Evaluate best checkpoint on **≥1 external benchmark** (BoxingGym or NewtonBench, §7) + the internal held-out set. Write results. **The transfer number is the headline.** |

**Buffer built in:** the schedule assumes days 9–14 may need a restart; if the debug loop
(days 4–5) reveals a reward bug, it's caught before any expensive run. If everything goes
smoothly, days 15–21 become a second, larger run (more steps or the 27B-dense confirmatory
run).

### 6.5.4 How many iterations / what's a reasonable estimate

- **Steps:** anchor on **rStar2-Agent's ~510 RL steps to frontier math at 14B**
  (2508.20722) — with a good verifier, step counts are *modest*. Plan **~500–1,000
  optimization steps** for the main run; a **first pilot at ~200–300 steps** is enough to
  see whether the curve moves. Do *not* plan for multi-thousand-step runs in 3 weeks.
- **Traces:** at 64 worlds/step × G=8 → 512 traces/step → **~250k–500k traces** for the
  main run; **~50k–150k** for the pilot. (This is the §5.1 pilot regime, not the full-run
  regime — appropriate for the deadline.)
- **Tokens & cost (Tinker, per-token):** at ~12k tokens/trace, the main run is
  **~3–6B tokens** total (sample + train). On the A3B "Medium" tier this is a **low-
  thousands-of-dollars** run, not tens of thousands — the per-token pricing and few
  active params are what make a 3-week budget realistic. **Measure your real trace length
  on days 9–10 before committing** — it's the biggest single lever (§8).
- **Worlds:** **~1k–2k distinct audited worlds** is plenty for a 3-week run (breadth
  matters more than count; §5.4). Generate on demand, cache the audited ones.
- **Realistic outcome to promise:** a **moving, held-out learning curve** and a **transfer
  number on ≥1 external benchmark** — i.e., *evidence the approach works*, not a
  fully-tuned frontier model. That is the right scope for 3 weeks and is exactly the
  result the positioning in §7 needs.

### 6.5.5 What we still need to figure out along the way (decisions the plan forces)

1. **Reward weighting `w_A`/`w_B` and penalty magnitudes** — start 0.5/0.5; the debug loop
   tells you if part B is too easy/hard to move. (§4)
2. **Terminal-only vs GiGPO step-credit** — start terminal-only (simplest); add step-credit
   only if credit dilution stalls learning. Tinker's `Transition` list supports per-step
   rewards if needed. (§1.3, §4)
3. **Trace length reality** — measure on days 9–10; it sets the whole budget. (§8)
4. **Resolver in-loop cost** — lexical-only in training; confirm latency doesn't dominate.
5. **Curriculum schedule** — fixed ramp vs adaptive-to-pass-rate (RLVE-style, §7). Start
   fixed; go adaptive if easy worlds saturate.
6. **LoRA rank** — start with Tinker's default; raise only if the curve plateaus below the
   27B-dense reference.
7. **pass@k protocol** — decide k and sampling temp up front so sharpening-vs-gain is
   measurable from the first eval (§1.2).
8. **Which external benchmark** — BoxingGym (experimental design, closest reward shape) vs
   NewtonBench (interactive discovery). Pick one on day 1 and wire its adapter early.

---

## 7. Positioning: how this project relates to the 2025–2026 literature (new)

This section was absent from the first draft and is the most important addition. The
sweep shows the project is **timely and only partially occupied** — and, critically, that
there is one **theoretical counter-argument it must engage**.

### 7.1 The whitespace: two literatures that meet exactly here

- **Procedural-environment + verifiable-reward RL** — but for *symbolic* reasoning:
  - **RLVE** (2511.07317, Nov 2025) — the closest methodological sibling. RLVE-Gym is
    400 verifiable environments that *procedurally generate problems*, give
    algorithmically verifiable rewards, and *adapt per-environment difficulty to the
    policy* (an auto-curriculum). Its finding — environment-scaling improves generalizable
    reasoning — is our thesis in a different domain. **Must-cite; frame RPG as "RLVE for
    interventional causal discovery."**
  - **Reasoning Core** (2509.18083 and updates) — procedural symbolic-reasoning RL
    environments with continuous difficulty control and infinite novel instances;
    anti-memorization by construction. Comparable on the procedural-worlds axis.
- **Interactive, intervention-based scientific discovery** — but *eval-only, on frozen
  models*:
  - **NewtonBench** (2510.07172, Oct 2025) — **closest benchmark to us.** 324 physics
    tasks with counterfactual law shifts; agents interactively probe simulated systems to
    uncover hidden laws. Our "discover the hidden cause via interventions" in one sentence.
  - **PhysGym** (2507.15550) — interactive physics discovery with *controllable agent
    prior knowledge*; agents probe and form hypotheses.
  - **BoxingGym** (2501.01540, Jan 2025) — 10 probabilistic-model environments; agents run
    experiments and revise theories, scored by *expected information gain*. Direct
    experimental-design comparable.
  - **BixBench** (2503.00096) — multi-step exploratory comp-bio reasoning; frontier models
    only ~17%, i.e. the capability is far from saturated.

**Our intersection — RL-*training* an agent on procedurally generated *causal-discovery*
worlds with a verifiable oracle — appears genuinely unoccupied.** RLVE trains but on
symbolic tasks; NewtonBench/PhysGym/BoxingGym are causal/interventional but evaluate
frozen models. That intersection is the paper's contribution claim.

### 7.2 The counter-argument to engage head-on

- **"Why LLMs Fail at Causal Discovery and How Interventional Agents Escape" / A-CBO**
  (2605.27567, May 2026 [verify-id]) — **the single most important paper for our framing.**
  It argues via a "kernel obstruction theorem" that SFT, DPO, and ICL *fundamentally
  cannot* do causal discovery from observational data, and proposes using a **frozen LLM
  as an interventional oracle inside an external Bayesian loop** (A-CBO) rather than
  training the weights. It ships **Extended Corr2Cause** (24 vars, 18K samples).
  **We must address this directly:** does RL escape the obstruction because it optimizes
  over *intervention actions/decisions* (arguably outside the obstructed observational-
  prediction space), or do we hit the same wall? Either answer is a strong
  related-work/discussion section — and if RL *does* escape it, that is itself a result.

### 7.3 Techniques to borrow, and framing support

- **Environment generation / auto-curriculum:** the UED / POET descendants — Imagined
  Autocurricula (2509.13341), regret-based generation (2601.14957 [verify-id]), PACE
  (2605.01358 [verify-id]) — plus RLVE's adaptive difficulty. **None applies UED to
  causal/scientific-discovery worlds** — a citable gap and a source of world-generator
  techniques.
- **Turn-level RL for discovery specifically:** SciDisco, "Scaling Scientific Discovery
  Environments for Turn-Level Agentic RL" (2607.28990 [verify-id]) — process-verifiable
  scientific-discovery environments with explicit turn-level credit; read closely for
  environment design. ECHO's posterior-sensitive turn rewards (2606.29745 [verify-id]) —
  the info-gathering credit signal (§4).
- **Framing support:** EurekAgent's "environment engineering is all you need for
  autonomous scientific discovery" (2606.13662 [verify-id]) directly backs our premise
  that procedurally generated worlds + oracle rewards are the key lever; the LLM-world-
  model training cluster (2606.27483, 2606.25421 [verify-id]) shows the field trending
  toward agent + explicit world model, our sim/world-model architecture.
- **Adjacent causal-LLM work** (mostly observational/prompting, useful to show our
  interventional+RL framing is distinct): "Causal Discovery in the Era of Agents" survey
  (2606.23608 [verify-id]); "From Gameplay Traces to Game Mechanics" (2602.00190
  [verify-id], conceptually close — inferring hidden mechanics from interaction).

### 7.4 Concrete asks for the paper

- **Cite as direct comparables:** RLVE (2511.07317), NewtonBench (2510.07172), BoxingGym
  (2501.01540), PhysGym (2507.15550), A-CBO/Extended-Corr2Cause (2605.27567 [verify-id]),
  BixBench (2503.00096).
- **Evaluate on ≥1 external benchmark** (NewtonBench or BoxingGym) to externally validate
  the internal RPG oracle.
- **Write a discussion paragraph answering A-CBO's obstruction claim.**

---

## 8. Open questions / to confirm before the big run

- **Qwen3.6-27B architecture:** dense or MoE, active-param count, and whether the ~256K
  context on the single-source card is real (§3.1). Sets rollout cost (§5.2).
- **Cluster access:** what GPUs and how many (Nautilus / other)? Plan assumes H100 80GB;
  A100 80GB works at ~1.5–2× wall-clock.
- **MoE vs dense decision:** if going MoE, commit to GSPO + a router-replay/router-aware-IS
  technique up front (§3.3); if the instability risk isn't worth it, go 32B dense.
- **Resolver in-loop cost:** benchmark lexical-only resolver rollout latency; the LLM
  fallback roughly doubles per-turn model calls if ever needed in training.
- **Trace length reality:** the ~12k per-trace estimate is the biggest lever on the whole
  budget — measure it in the pilot first thing.
- **Framework pick:** verl (scale + agent maturity, safe default) vs AReaL 2.0 / SkyRL
  (async, closest to our shape) vs verifiers+prime-rl (cleanest abstractions). Suggest
  prototyping the env in verifiers, then training in verl or SkyRL-Agent.
- **Verify the [verify-id] citations** before formal use (see Appendix).

---

## Appendix: sources consulted

### A. Frameworks & model cards (fetched READMEs / cards this session)
- **verl** (verl-project org) — algorithms, multi-turn/agent recipes, scale, Qwen3 examples.
- **AReaL / AReaL 2.0** (arXiv **2505.24298**) — fully-async RL substrate; 2.0 microservice refactor.
- **SkyRL / SkyRL-Agent** (arXiv **2511.16108**) — long-horizon multi-turn agent training; Qwen3-32B → SA-SWE-32B, 39.4% SWE-Bench Verified.
- **NeMo-RL**, **ROLL** (OSDI'26; ships GiGPO), **OpenRLHF** (v0.10), **slime** (THUDM/GLM), **prime-rl + verifiers** (Prime Intellect), **TRL** (HuggingFace) — 2026 state per §1.4.
- **Model cards:** Qwen3-30B-A3B (30.5B/3.3B active, Apache-2.0, thinking), Qwen3-32B (32.8B dense), Qwen3-8B, Qwen3-235B-A22B; gpt-oss-20b (21B/3.6B) & 120b; DeepSeek-R1 / R1-0528 (671B-class, MIT) + R1-Distill-Qwen-7B/14B/32B; Magistral-Small-2509 (24B); GLM-4.6; Kimi-K2; MiniMax-M2; Llama-4-Scout; SmolLM3-3B; Nemotron-Super-49B. `Qwen3.6-27B` card **single-source/unverified** (§3.1).

### B. High-confidence papers (established 2025, real IDs)
- **DAPO** 2503.14476 · **Dr.GRPO** 2503.20783 · **GSPO** 2507.18071 · **Tricks or Traps** 2508.08221 — algorithm recipe (§1.1).
- **Does RL Incentivize Reasoning Beyond the Base Model** 2504.13837 · **Spurious Rewards** 2506.10947 — elicitation debate (§1.2).
- **The Entropy Mechanism of RL** 2505.22617 (Clip-Cov/KL-Cov) — entropy collapse (§1.5).
- **GiGPO** 2505.10978 (NeurIPS 2025) — hierarchical credit (§1.3, §4).
- **Search-R1** 2503.09516 · **ToRL** 2503.23383 · **ReTool** 2504.11536 · **rStar2-Agent** 2508.20722 (GRPO-RoC) — tool-integrated RL (§1.3, §4).
- **AReaL** 2505.24298 · **AgentGym** 2406.04151 — agentic-RL infra (§1.3).
- **One Token to Fool LLM-as-a-Judge** 2507.08794 — verifier hacking (§1.5, §4).
- **RLVE** 2511.07317 · **Reasoning Core** 2509.18083 — procedural verifiable-reward RL (§7.1).
- **NewtonBench** 2510.07172 · **PhysGym** 2507.15550 · **BoxingGym** 2501.01540 · **BixBench** 2503.00096 — interactive-discovery benchmarks (§7.1).
- **Imagined Autocurricula** 2509.13341 — UED / learned-world-model curricula (§7.3).

### C. Findings to cite but **[verify-id]** (real findings; IDs from a forward-dated feed — confirm exact identifier before formal citation)
- Algorithm/entropy: three-operations-on-one-number 2607.00152 · unbiased-vs-length-invariant impossibility 2607.23364 · asymmetric-clip entropy 2509.26114 · QAE 2509.22611 · EDGE-GRPO 2507.21848 · OPEFO 2605.11491 · SFT-overtraining→collapse 2606.18487 · spurious-reward mechanism 2512.16912 / 2601.11061 · support-reshaping 2608.00220 · BODHI 2608.02867.
- Multi-turn credit: BEACON 2605.06078 · ECHO 2606.29745 · ECPO "denser credit not enough" 2606.05885 · RSPO 2607.04713 · SciDisco 2607.28990 · TurnSight 2608.04007.
- Reward hacking / rubrics: specification hacking (formal verify) 2605.30914 · RRM rubric rewards 2510.07774 · step-wise rubrics 2605.17291 · online rubrics 2510.07284 · EvoRubrics 2606.23038 · precision-over-diversity 2601.04954 · dark-room reward channel 2607.21273 · PASS 2606.29296.
- MoE-RL stability: align train/inference routers 2510.11370 · router-aware IS 2510.23027 · PR2 routing replay 2606.00395 · routing-replay practices 2512.01374 · Ring-lite 2506.14731 (real) · MoE-GRPO 2603.24984.
- Causal / scientific discovery: **A-CBO / Why LLMs Fail at Causal Discovery** 2605.27567 · Causal Discovery in the Era of Agents (survey) 2606.23608 · Gameplay-Traces→Game-Mechanics 2602.00190 · EurekAgent 2606.13662 · Graph-Native RL for hypothesis generation 2607.00924 · amortised BED for LLMs 2607.03426 · LLM-world-model cluster 2606.27483 / 2606.25421.
- Environment design / curriculum: PACE 2605.01358 · regret-based UED 2601.14957 · Chart-RL (difficulty>quantity) 2603.06958 · LEACL 2607.23515.

### D. Domain knowledge
- GRPO/DAPO/GSPO mechanics, agentic-RL failure modes, and the compute arithmetic
  (grounded against the above; compute figures remain engineering estimates pending the pilot).
