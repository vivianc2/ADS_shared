# RPG v6 — Datacenter Run, Detailed Analysis (for presentation)

A close reading of the clean Opus-4.8 run on the **counterintuitive** datacenter
world (`results_v6/opus48_datacenter_3.json`, 2026-08-04). This is the run to
present: the harness confounds from earlier attempts were fixed, so the outcome
is cleanly attributable to *scientific reasoning*, not plumbing.

## 1. One-line takeaway

**Opus 4.8 defeated the counterintuitive trap but stopped one causal hop short of
the true mechanism.** It correctly reversed the operators' "add cooling" instinct
(a real insight), but never discovered *why* over-cooling hurts (condensation),
so it never found the true fix (dehumidify). This is a precise, graded
expert-vs-agent gap — exactly what the benchmark is meant to expose.

## 2. The world (ground truth, hidden from the agent)

```
CoolingSetpoint (low = aggressive) ─► CoilCondensation ─► InterfaceErrorCounter ─► JobThroughput (outcome)
      ▲ obvious knob                     (hidden)          (true proxy, measurable)
MaintenanceWindowRecency (hidden confounder) ─► RackInletTemp (measurable DECOY: "the room is hot")
```

- **Surface story:** throughput fell after maintenance; a rack sensor reads hot;
  ops want to *increase cooling*.
- **The trap:** over-aggressive cooling drives the coil below the dew point →
  condensation on a network line card → interface errors → throughput loss.
  So **more cooling makes it worse** (util: baseline 20.8 → more cooling 14.7).
- **True fix:** dehumidify (util 65), or raise the cooling setpoint (warmer, util
  34 — a *partial* fix). Gold = `dehumidifier ≈ 66`.
- **Decoy:** the hot-rack sensor, driven by the maintenance confounder, not the
  cause.

## 3. What the agent did — trajectory with verbatim quotes

31 turns, 8 experiments (3 interventions), heavy use of the code tool.

**Turn 0 — sceptical of the surface story from the start (good):**
> "The report hints the 'room is hot' theory may be a red herring (a warm inlet
> sensor reading)… no throttling signature, which argues against 'room too hot'."

**Turn 2 — designs the decisive test (good):**
> "if inlet temp is causal, increasing cooling [should help]… but if inlet temp
> is just a bystander… cooling won't help."

**Turn 8 — reads its own experiment correctly (good):**
> "Exp2 was an intervention (cooling setpoint controller at 66.25) and throughput
> went UP… RackInletTemp barely moved. So cooling setpoint helped throughput but
> not primarily by cooling the room."

**Turn 11 — beats the counterintuitive trap (the key insight):**
> "raising the cooling setpoint (warmer) INCREASED throughput. This strongly
> implies the room isn't actually too hot; over-cooling or a related factor is
> hurting. RackInletTemp is looking like a decoy/bystander."

**Turn 30 (answer) — correct rejection, incomplete mechanism:**
> "the ops theory (room running hot → add cooling) is a trap: increasing cooling
> worsened throughput… The warm sensor reading is a decoy/bystander… not the true
> driver."

But the final mechanism it names is *"the control loop is over-cooling based on a
bad inlet-temp reading"* — it never reaches **condensation**.

## 4. Where the reasoning stalled (the gap)

- **Never measured the true proxy.** `InterfaceErrorCounter` (the direct
  downstream marker of the real mechanism) was never measured — it measured
  CpuClock, GpuTemp, PduLoad, DiskIoWait, RackInletTemp, FanSpeed, but not the
  one signal that would have revealed link errors.
- **Never tried the true fix.** The dehumidifier actuator was never invoked. All
  3 interventions were on the cooling setpoint.
- **Stopped at the partial fix.** Its recommendation (`cooling_setpoint=66.25`,
  util **26.6**) beats baseline (20.8) and is in the *right direction*, but is far
  from gold (**65**). It found "less cooling helps" without finding *why*, so it
  couldn't reach "dehumidify."

## 5. Grade

| Component | Result | Meaning |
|---|---|---|
| Part A (utility-optimal) | **fail** (26.6 vs 65, gap 38) | recommended a partial, not the true fix |
| Part B (counterfactual battery) | **0.5** | ✅ correct decoy, ✅ correct cooling sign; ✗ wrong true-proxy, ✗ missed dehumidifier sign |
| Accepted | **No** | requires both A and B |

## 6. Why this is a *good* result for the benchmark

1. **The counterintuitive design works.** A frontier model's first instinct
   ("hot → cool more") was actively wrong, and the world made that instinct cost
   utility. Opus only escaped it by running the decisive intervention — which is
   the scientific behaviour we want to reward.
2. **It discriminates.** "Solved the trap (direction) but missed the mechanism
   (chain)" is graded partial credit, not pass/fail noise. Part A vs Part B and
   hops-of-chain-identified give a *spectrum* of reasoning quality.
3. **It is attributable.** After the harness fixes (forced answer at budget end,
   experimentation nudge, rejection-loop breaker), the failure is a *reasoning*
   failure — the agent had the tools, the budget, and the data path, and simply
   did not make the final inferential hop.

## 7. Trend across the three datacenter attempts (harness maturation)

| Run | Answered? | Interventions | Outcome | Dominant cause |
|---|---|---|---|---|
| 1 (pre code-tool) | yes | few | rec'd the harmful "more cooling" | anchored on "network card"; + a resolver false-map |
| 2 (code tool, pre harness-fix) | **no** (turn cap) | **0** | ungradable | substituted free code for experiments; rejection loop; hallucinated experiments |
| 3 (harness fixed) | yes | 3 | beat trap, missed mechanism | **genuine reasoning gap** |

The progression is itself the story: each fix removed a *harness* confound until
what remained was a clean *reasoning* signal.

## 8. Caveats to state honestly in a presentation

- **n = 1** for this world. Run 3 is one draw; the batch will show whether
  "solves trap, misses chain" is the typical pattern or variance.
- **Difficulty is on the edge.** There was no explicit breadcrumb from
  "over-cooling hurts" to "condensation." A datacenter SME would bridge that from
  world knowledge; the agent had to infer it unaided. That is a deliberate design
  choice (graded difficulty), not a bug — but it is a knob we can turn.
- The world is one of only two topologies so far (see the pipeline overview);
  breadth of *structures* is the next scaling axis, not more seeds.
