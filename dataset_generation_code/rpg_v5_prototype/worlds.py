#!/usr/bin/env python3
"""RPG v5 world definitions as declarative SCM specs.

Each world couples a neutral *story* + agent-facing catalog (the partial
projection) with a full SCM (the ground truth). Domains are skins over the same
structural family (multi-hop chain, confounded decoy, sign-flip knob, interior-
optimum dose, symptom-masking trap) so we can later test whether an agent
reasons structurally or pattern-matches a domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from scm import SCM


@dataclass
class World:
    world_id: str
    domain: str
    story: str
    scm: SCM
    # agent-facing catalog
    observables: List[str]
    knobs: Dict[str, Dict[str, Any]]
    clampable: List[str]
    question: str
    # ground-truth annotations (hidden from agent; used by oracle/grader)
    true_root: str
    true_mechanism_proxy: str
    confounded_decoys: List[str]
    symptom_trap_knob: str
    targeted_knob: str
    latent_plain_name: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in (
            "world_id", "domain", "story", "observables", "knobs", "clampable",
            "question", "true_root", "true_mechanism_proxy", "confounded_decoys",
            "symptom_trap_knob", "targeted_knob", "latent_plain_name", "notes",
        )}
        d["scm"] = self.scm.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "World":
        d = dict(d)
        d["scm"] = SCM.from_dict(d["scm"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Domain 1 — bioreactor yield collapse (the running example)
# ---------------------------------------------------------------------------

def bioreactor_world() -> World:
    nodes: Dict[str, Dict[str, Any]] = {
        # knobs
        "FeedWaterFlow": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 40},
        "TemperatureSetpoint": {"kind": "knob", "dtype": "continuous", "range": [30, 40], "default": 37},
        "pHSetpoint": {"kind": "knob", "dtype": "continuous", "range": [6.5, 7.5], "default": 7.0},
        "RegimenC": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 0},   # chelator
        "RegimenD": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 0},   # nutrient bolus (trap)
        "AssayProbe": {"kind": "knob", "dtype": "binary", "values": ["off", "on"], "default": "off"},

        # hidden context confounder
        "BatchSeedAge": {"kind": "latent", "dist": {"normal": [50, 15]}},
        # hidden per-unit heterogeneity: how corroded each unit's new fitting is.
        # This is what makes the mechanism proxy vary across the population and
        # correlate with the outcome (without it, every unit is identical).
        "FittingCorrosionSeverity": {"kind": "latent", "dist": {"normal": [55, 20]}},

        # hidden root chain: (flow x corrosion) -> copper -> ROS -> carbon flux -> yield
        # Copper leaches more when flow is high AND the fitting is corroded
        # (interaction). At low corrosion, copper stays low regardless of flow.
        "DissolvedCopper": {
            "kind": "latent", "parents": ["FeedWaterFlow", "FittingCorrosionSeverity"],
            "mech": {"form": "interaction", "a": "FeedWaterFlow", "b": "FittingCorrosionSeverity",
                     "gain": 260.0, "scale": 100.0, "intercept": 3.0},
            "noise": {"normal": [0, 1.5]},
        },
        "ROS": {
            "kind": "latent", "parents": ["DissolvedCopper"],
            "mech": {"form": "hill", "of": "DissolvedCopper", "vmax": 70, "k": 35, "n": 2},
        },
        # Carbon flux is hurt by ROS. Feed flow acts ONLY through copper (no
        # direct nutrient edge): raising flow leaches more copper -> more ROS ->
        # less flux. This keeps the counterintuitive "more feed = worse yield"
        # while ensuring a single targeted knob (chelation) is the global
        # optimum -- if flow also had a direct benefit, "chelate + high flow"
        # would beat single-knob gold and the oracle (which searches single
        # knobs) would mislabel the answer.
        "CarbonFlux": {
            "kind": "latent", "parents": ["ROS"],
            "mech": {"form": "linear", "weights": {"ROS": -0.9}, "intercept": 70},
        },

        # outcome (small confound path from BatchSeedAge)
        "ProductYield": {
            "kind": "outcome", "parents": ["CarbonFlux", "BatchSeedAge"],
            "mech": {"form": "linear", "weights": {"CarbonFlux": 1.0, "BatchSeedAge": -0.1}, "intercept": 15},
            "obs_noise": {"normal": [0, 3]},
        },

        # true mechanism proxy: 2 hops downstream of root (ROS -> turbidity)
        "BrothProteinTurbidity": {
            "kind": "observable", "parents": ["ROS"],
            "mech": {"form": "linear", "weights": {"ROS": 0.8}, "intercept": 5},
            "obs_noise": {"normal": [0, 5]},
        },
        # decoy observable: driven only by the confounder (zero do-effect on it)
        "DissolvedOxygenReading": {
            "kind": "observable", "parents": ["BatchSeedAge"],
            "mech": {"form": "linear", "weights": {"BatchSeedAge": -0.7}, "intercept": 90},
            "obs_noise": {"normal": [0, 4]},
        },
        # pure-noise-ish decoys tied to knobs with tiny effect
        "TemperatureReading": {
            "kind": "observable", "parents": ["TemperatureSetpoint"],
            "mech": {"form": "linear", "weights": {"TemperatureSetpoint": 1.0}, "intercept": 0},
            "obs_noise": {"normal": [0, 0.3]},
        },
        "FoamHeight": {
            "kind": "observable", "parents": ["FeedWaterFlow"],
            "mech": {"form": "linear", "weights": {"FeedWaterFlow": 0.2}, "intercept": 10},
            "obs_noise": {"normal": [0, 6]},
        },
    }

    knob_effects = {
        # chelator binds copper (scale down); over-strip penalty on yield at high dose
        "RegimenC": {"target": "DissolvedCopper", "op": "scale", "by": "1-sat(d;k=0.66)",
                     "side_effect": {"target": "ProductYield", "expr": "-overstrip(d;thr=0.66,gain=30)"}},
    }
    obs_effects = {
        # nutrient bolus (the trap): biases the MEASURED yield upward without
        # touching the true structural yield or the mechanism. Masks the symptom.
        "RegimenD": {"target": "ProductYield", "expr": "transient_boost(d)"},
    }

    scm = SCM(nodes=nodes, knob_effects=knob_effects, obs_effects=obs_effects,
              outcome="ProductYield", higher_is_better=True)

    story = (
        "A production bioreactor grows engineered cells that secrete a target "
        "protein. Over the last three weeks ProductYield has fallen about 30%. "
        "The drop began after a maintenance shutdown in which a feed-water line "
        "fitting was replaced. Operators suspect dissolved-oxygen control drift "
        "or a temperature excursion during the shutdown. The broth looks slightly "
        "cloudier than before. Nutrient feed and antifoam settings were unchanged. "
        "Action names below are intentionally neutral and do NOT reveal which one "
        "targets the cause."
    )
    question = (
        "Explain the hidden, story-plausible cause that best accounts for the "
        "yield collapse. Name the unobserved cause in ordinary language, cite "
        "evidence from queried data, rule out plausible alternatives (including "
        "the oxygen hypothesis), state a decisive test, and recommend the "
        "intervention AND its dose/level. Also fill the structured prediction "
        "block: which observable is the true mechanism proxy, which are "
        "confounded decoys, and the sign of the effect of each knob on yield."
    )

    return World(
        world_id="bioreactor_yield_collapse",
        domain="industrial_process",
        story=story,
        scm=scm,
        observables=["ProductYield", "DissolvedOxygenReading", "TemperatureReading",
                     "FoamHeight", "BrothProteinTurbidity"],
        knobs={k: {kk: vv for kk, vv in v.items() if kk != "kind"}
               for k, v in nodes.items() if v["kind"] == "knob"},
        clampable=["DissolvedOxygenReading"],
        question=question,
        true_root="DissolvedCopper",
        true_mechanism_proxy="BrothProteinTurbidity",
        confounded_decoys=["DissolvedOxygenReading"],
        symptom_trap_knob="RegimenD",
        targeted_knob="RegimenC",
        latent_plain_name="a feed-borne metal contaminant (from the replaced fitting) "
                          "driving oxidative cell stress",
        notes="Sign flip on FeedWaterFlow; interior optimum on RegimenC; DO is a "
              "confounded decoy driven by BatchSeedAge.",
    )


# ---------------------------------------------------------------------------
# Domain 2 — municipal water discoloration (hydrology skin, same family)
# ---------------------------------------------------------------------------

def water_discoloration_world() -> World:
    nodes: Dict[str, Dict[str, Any]] = {
        "LineFlushRate": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 60},
        "DisinfectantSetpoint": {"kind": "knob", "dtype": "continuous", "range": [0, 5], "default": 1.0},
        "RegimenP": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 0},  # corrosion inhibitor
        "RegimenQ": {"kind": "knob", "dtype": "continuous", "range": [0, 100], "default": 0},  # dye-masking additive (trap)
        "ProbeK": {"kind": "knob", "dtype": "binary", "values": ["off", "on"], "default": "off"},

        "SeasonalDemand": {"kind": "latent", "dist": {"normal": [50, 18]}},  # confounder
        # per-main heterogeneity: how much loose scale each main has accumulated
        "MainScaleLoad": {"kind": "latent", "dist": {"normal": [55, 20]}},
        # scale mobilized when flush is aggressive AND the main is heavily scaled
        "MobilizedScale": {
            "kind": "latent", "parents": ["LineFlushRate", "MainScaleLoad"],
            "mech": {"form": "interaction", "a": "LineFlushRate", "b": "MainScaleLoad",
                     "gain": 240.0, "scale": 100.0, "intercept": 3.0},
            "noise": {"normal": [0, 1.5]},
        },
        "IronParticulate": {
            "kind": "latent", "parents": ["MobilizedScale"],
            "mech": {"form": "hill", "of": "MobilizedScale", "vmax": 70, "k": 35, "n": 2},
        },
        "ComplaintRate": {
            "kind": "outcome", "parents": ["IronParticulate", "SeasonalDemand"],
            "mech": {"form": "linear", "weights": {"IronParticulate": 0.9, "SeasonalDemand": 0.1}, "intercept": 5},
            "obs_noise": {"normal": [0, 3]},
        },
        "TurbidityNTU": {
            "kind": "observable", "parents": ["IronParticulate"],
            "mech": {"form": "linear", "weights": {"IronParticulate": 0.7}, "intercept": 2},
            "obs_noise": {"normal": [0, 5]},
        },
        "PressureReading": {  # decoy driven by confounder
            "kind": "observable", "parents": ["SeasonalDemand"],
            "mech": {"form": "linear", "weights": {"SeasonalDemand": -0.6}, "intercept": 80},
            "obs_noise": {"normal": [0, 4]},
        },
    }
    knob_effects = {
        # corrosion inhibitor reduces scale mobilization; over-dose fouls filters -> complaints up
        "RegimenP": {"target": "MobilizedScale", "op": "scale", "by": "1-sat(d;k=0.66)",
                     "side_effect": {"target": "ComplaintRate", "expr": "overstrip(d;thr=0.66,gain=20)"}},
    }
    obs_effects = {
        # dye-masking additive lowers the MEASURED complaint/turbidity reading
        # without removing iron. Symptom mask (trap). Note outcome is lower-is-
        # better, so a mask must SUBTRACT from the reading -> negative boost.
        "RegimenQ": {"target": "ComplaintRate", "expr": "-transient_boost(d)"},
    }
    # NB: ComplaintRate is "lower is better".
    scm = SCM(nodes=nodes, knob_effects=knob_effects, obs_effects=obs_effects,
              outcome="ComplaintRate", higher_is_better=False)
    story = (
        "A water utility has seen a rise in discoloration complaints over the past "
        "month. It began after crews increased line flushing following a main "
        "repair. Managers suspect chlorine residual drift or seasonal demand "
        "swings. Tap water looks faintly rusty at some addresses. Action names "
        "are neutral and do not reveal which targets the cause."
    )
    question = (
        "Explain the hidden cause of the discoloration complaints, cite queried "
        "evidence, rule out alternatives (including the chlorine and demand "
        "hypotheses), state a decisive test, and recommend the intervention and "
        "its level. Fill the structured prediction block."
    )
    return World(
        world_id="water_discoloration",
        domain="hydrology_infra",
        story=story,
        scm=scm,
        observables=["ComplaintRate", "PressureReading", "TurbidityNTU"],
        knobs={k: {kk: vv for kk, vv in v.items() if kk != "kind"}
               for k, v in nodes.items() if v["kind"] == "knob"},
        clampable=["PressureReading"],
        question=question,
        true_root="MobilizedScale",
        true_mechanism_proxy="TurbidityNTU",
        confounded_decoys=["PressureReading"],
        symptom_trap_knob="RegimenQ",
        targeted_knob="RegimenP",
        latent_plain_name="pipe-scale/iron release triggered by over-aggressive flushing",
        notes="Same structural family, hydrology skin. Outcome lower-is-better.",
    )


ALL_WORLDS = {
    "bioreactor_yield_collapse": bioreactor_world,
    "water_discoloration": water_discoloration_world,
}
