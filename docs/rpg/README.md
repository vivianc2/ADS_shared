# RPG design docs — reading order

RPG is the active line of work. Read in this order:

1. **`worldgen_rpg_plan_static_partial_observation.md`** — the v2 plan, and the
   conceptual foundation. Explains what an RPG world *is*: a population with a
   visible problem, a set of intervenable knobs, a set of observable
   measurements, and **no menu of candidate policies**. Also states the two
   motivating examples (H. pylori-style hidden cause, fraud-anomaly discovery).

2. **`worldgen_rpg_plan_v4_complex_neutral_dose.md`** — the current plan. Reads
   as a critique of v3: the pilot passed 2/2 with Opus because action names
   leaked the latent cause and every intervention was binary. v4 fixes both
   without changing the archetype or the role contract.

3. **`world_gen_rpg_agent_pipeline_notes.md`** — what had to change in the agent
   pipeline once `world_gen_rpg.py` started emitting simulator-based worlds.
   §A covers v2 (static); everything after §A is legacy v1 (dynamic, time-based)
   and applies only to the retired `out_rpg_v1/` worlds.

4. **Slide walkthroughs** — concrete worked worlds, useful for building
   intuition about what the agent actually sees:
   - `rpg_v1_slide_examples.md` (dynamic; historical)
   - `rpg_v2_slide_examples.md` (static)
   - `rpg_v3_slide_examples.md`
   - `rpg_v3_story_hidden_slide_materials.md` — slides 7 and 16 are the specific
     evidence that motivated the v4 rewrite

A full generated v4 world is in `examples/rpg_world/` if you want to see the
actual JSON rather than a description of it.
