#!/usr/bin/env python3
"""RPG v6 large open-scenario worlds.

The bioreactor world: a real ~4-hop hidden causal chain embedded among many
meaningful, in-world, mostly causally-inert distractor variables and actuators,
so that brute force over the action space is infeasible and the agent must use
world knowledge to decide where to probe.

Variable names ARE meaningful (this is deliberate in v6 — see the v6 design
doc): with ~20 variables and ~14 actuators against a ~15-experiment budget,
knowing what a variable *is* does not tell you whether it *matters*. Reasoning
is required to connect knowledge -> hypothesis -> decisive experiment.
"""

from __future__ import annotations

from typing import Any, Dict

from engine import WorldSCM


def bioreactor_world_v6() -> Dict[str, Any]:
    V: Dict[str, Dict[str, Any]] = {}

    # ---- hidden confounder + heterogeneity ----
    V["BatchSeedAge"] = {"kind": "latent", "aliases": ["seed age", "inoculum age", "cell line age"],
                         "dist": {"normal": [50, 15]}}
    V["FittingCorrosionSeverity"] = {"kind": "latent",
                                     "aliases": ["fitting corrosion", "corrosion of the fitting", "pitting"],
                                     "dist": {"normal": [55, 20]}}

    # ---- the true hidden chain: flow x corrosion -> copper -> ROS -> carbon flux -> yield ----
    V["DissolvedCopper"] = {"kind": "latent",
                            "aliases": ["dissolved copper", "copper", "leached metal", "metal ions", "trace metals"],
                            "parents": ["FeedWaterFlowRate", "FittingCorrosionSeverity"],
                            "mech": {"form": "interaction", "a": "FeedWaterFlowRate", "b": "FittingCorrosionSeverity",
                                     "gain": 260.0, "scale": 100.0, "intercept": 3.0},
                            "noise": {"normal": [0, 1.5]}}
    V["ReactiveOxygenSpecies"] = {"kind": "latent",
                                  "aliases": ["reactive oxygen species", "ROS", "oxidative stress", "free radicals"],
                                  "parents": ["DissolvedCopper"],
                                  "mech": {"form": "hill", "of": "DissolvedCopper", "vmax": 70, "k": 35, "n": 2}}
    V["CarbonFluxToProduct"] = {"kind": "latent",
                                "aliases": ["carbon flux", "metabolic flux to product", "productive metabolism"],
                                "parents": ["ReactiveOxygenSpecies"],
                                "mech": {"form": "linear", "weights": {"ReactiveOxygenSpecies": -0.9}, "intercept": 70}}

    # ---- outcome (measurable) ----
    V["ProductTiter"] = {"kind": "outcome", "aliases": ["product titer", "protein yield", "product yield", "titer", "yield"],
                         "parents": ["CarbonFluxToProduct", "BatchSeedAge"],
                         "mech": {"form": "linear", "weights": {"CarbonFluxToProduct": 1.0, "BatchSeedAge": -0.1}, "intercept": 15},
                         "measurable": True, "assay_noise": {"normal": [0, 3]}}

    # ---- true mechanism proxy (measurable, 2 hops downstream of root) ----
    V["BrothTurbidity"] = {"kind": "observable",
                           "aliases": ["broth turbidity", "cloudiness", "broth clarity", "lysis marker", "cell debris"],
                           "parents": ["ReactiveOxygenSpecies"],
                           "mech": {"form": "linear", "weights": {"ReactiveOxygenSpecies": 0.8}, "intercept": 5},
                           "measurable": True, "assay_noise": {"normal": [0, 12]}}

    # ---- confounded decoy (measurable, driven only by BatchSeedAge) ----
    V["DissolvedOxygen"] = {"kind": "observable",
                            "aliases": ["dissolved oxygen", "DO", "oxygen level", "pO2"],
                            "parents": ["BatchSeedAge"],
                            "mech": {"form": "linear", "weights": {"BatchSeedAge": -0.7}, "intercept": 90},
                            "measurable": True, "assay_noise": {"normal": [0, 4]}}

    # ---- controllable process settings that are near-inert on yield ----
    for nm, al, mean in [
        ("FeedWaterFlowRate", ["feed water flow", "feed flow", "feed rate", "water flow"], 40),
        ("Temperature", ["temperature", "temp", "culture temperature"], 37),
        ("pH", ["ph", "acidity"], 7.0),
        ("AgitationRate", ["agitation", "stir rate", "impeller speed", "rpm"], 200),
        ("GasSpargeRate", ["sparge rate", "gas flow", "aeration rate"], 50),
    ]:
        V[nm] = {"kind": "observable", "aliases": al, "dist": {"normal": [mean, mean * 0.15 + 1]},
                 "measurable": True, "assay_noise": {"normal": [0, max(0.3, mean * 0.02)]}}

    # ---- many meaningful, measurable, causally-INERT distractor variables ----
    distractors = [
        ("AntifoamLevel", ["antifoam", "defoamer"]),
        ("GlucoseFeedRate", ["glucose feed", "sugar feed", "carbon feed"]),
        ("GlutamineConc", ["glutamine", "amino acid feed"]),
        ("LactateConc", ["lactate", "lactic acid"]),
        ("AmmoniaConc", ["ammonia", "ammonium"]),
        ("Osmolality", ["osmolality", "osmotic pressure"]),
        ("ViableCellDensity", ["viable cell density", "VCD", "cell count"]),
        ("CO2Level", ["dissolved co2", "pCO2", "carbon dioxide"]),
        ("HarvestVolume", ["harvest volume", "working volume"]),
        ("VesselPressure", ["headspace pressure", "vessel pressure"]),
        ("CoolingWaterTemp", ["cooling water", "jacket temperature"]),
        ("FeedTankLevel", ["feed tank level", "media tank"]),
    ]
    for nm, al in distractors:
        V[nm] = {"kind": "observable", "aliases": al, "dist": {"normal": [50, 12]},
                 "measurable": True, "assay_noise": {"normal": [0, 3]}}

    # ---------------------------------------------------------------
    # Actuators — the ONLY way to intervene. Most are inert controls.
    # ---------------------------------------------------------------
    A: Dict[str, Dict[str, Any]] = {}

    # set-type controllers (discover there IS an O2 controller = the "clamp")
    A["feed_flow_controller"] = {"aliases": ["set feed water flow", "adjust feed flow", "feed flow controller", "change water flow"],
                                 "target": "FeedWaterFlowRate", "op": "set", "dtype": "continuous",
                                 "range": [0, 100], "default": 40,
                                 "description": "flow controller on the feed-water line"}
    A["do_controller"] = {"aliases": ["set dissolved oxygen", "clamp DO", "oxygen setpoint", "DO controller", "hold oxygen"],
                          "target": "DissolvedOxygen", "op": "set", "dtype": "continuous",
                          "range": [0, 100], "default": 55,
                          "description": "dissolved-oxygen control loop"}
    A["temp_controller"] = {"aliases": ["set temperature", "temperature setpoint", "adjust temp"],
                            "target": "Temperature", "op": "set", "dtype": "continuous",
                            "range": [30, 40], "default": 37, "description": "temperature jacket setpoint"}
    A["ph_controller"] = {"aliases": ["set ph", "ph setpoint", "adjust ph"],
                          "target": "pH", "op": "set", "dtype": "continuous",
                          "range": [6.5, 7.5], "default": 7.0, "description": "pH control loop"}
    A["agitation_controller"] = {"aliases": ["set agitation", "stir rate", "impeller setpoint"],
                                 "target": "AgitationRate", "op": "set", "dtype": "continuous",
                                 "range": [50, 400], "default": 200, "description": "impeller speed control"}

    # the TRUE lever: a chelating-agent dosing pump that binds copper (scale down),
    # with over-strip toxicity at high dose (interior optimum).
    A["chelator_dosing"] = {"aliases": ["add chelating agent", "chelation", "metal chelator", "chelator dose",
                                        "EDTA", "sequestering agent", "bind metals", "add chelant"],
                            "target": "DissolvedCopper", "op": "scale", "dtype": "continuous",
                            "range": [0, 100], "default": 0, "expr": "1-sat(d;k=0.66)",
                            "side_effect": {"target": "ProductTiter", "expr": "-overstrip(d;thr=0.66,gain=30)"},
                            "description": "metering pump for a metal-chelating feed additive"}

    # symptom-masking trap: a stabilizer that lifts the titer READING transiently.
    A["stabilizer_additive"] = {"aliases": ["add stabilizer", "product stabilizer", "protectant", "stabilizing agent"],
                                "target": "ProductTiter", "op": "mask", "dtype": "continuous",
                                "range": [0, 100], "default": 0, "expr": "transient_boost(d)",
                                "description": "a formulation stabilizer added to the harvest"}

    # inert intervention controls (real handles, ~0 effect on yield)
    for aid, al, tgt, rng_ in [
        ("antifoam_pump", ["add antifoam", "dose defoamer"], "AntifoamLevel", [0, 100]),
        ("glucose_feed_pump", ["increase glucose feed", "set glucose feed", "sugar feed rate"], "GlucoseFeedRate", [0, 100]),
        ("glutamine_feed_pump", ["glutamine feed", "amino acid feed rate"], "GlutamineConc", [0, 100]),
        ("sparge_controller", ["set sparge rate", "aeration rate", "gas flow setpoint"], "GasSpargeRate", [0, 100]),
        ("pressure_controller", ["set headspace pressure", "vessel pressure setpoint"], "VesselPressure", [0, 100]),
        ("cooling_controller", ["set cooling water", "jacket temp setpoint"], "CoolingWaterTemp", [0, 100]),
        ("media_exchange", ["media exchange", "perfusion rate", "media swap"], "Osmolality", [0, 100]),
    ]:
        A[aid] = {"aliases": al, "target": tgt, "op": "set", "dtype": "continuous",
                  "range": rng_, "default": (rng_[0] + rng_[1]) / 2,
                  "description": f"control handle for {tgt}"}

    scm = WorldSCM(variables=V, actuators=A, outcome="ProductTiter", higher_is_better=True)

    scenario = (
        "You are called in to diagnose a stalled run on a 2,000 L mammalian-cell "
        "bioreactor that produces a secreted therapeutic protein. Over the last "
        "three weeks the product titer has fallen roughly 30% below the historical "
        "process average, and the loss has persisted across three consecutive "
        "batches. The decline began shortly after a scheduled maintenance shutdown, "
        "during which a feed-water line fitting was replaced, the impeller seal was "
        "serviced, and the dissolved-oxygen probe was recalibrated. Process history "
        "shows the culture is run at 37 C and pH 7.0 with standard agitation, gas "
        "sparging, and glucose/glutamine feeding; antifoam is dosed on a schedule. "
        "The operators' leading theories are dissolved-oxygen control drift after "
        "the probe work, or a temperature excursion during the shutdown. Batch "
        "records note the broth has looked cloudier than usual at harvest. Viable "
        "cell density, lactate, ammonia, osmolality, and dissolved CO2 are logged "
        "each shift and look broadly within their normal ranges. The feed-water "
        "system, the gas train, the cooling jacket, and the harvest line are all "
        "instrumented and adjustable. You have a limited number of experiments; "
        "you may measure quantities and you may apply available controls or "
        "additives, alone or in combination. Determine what is really driving the "
        "titer loss and what to do about it."
    )

    # ground-truth annotations (hidden; used by oracle/grader)
    ground_truth = {
        "true_root": "DissolvedCopper",
        "true_mechanism_proxy": "BrothTurbidity",
        "confounded_decoys": ["DissolvedOxygen"],
        "targeted_actuator": "chelator_dosing",
        "symptom_trap_actuator": "stabilizer_additive",
        "source_actuator": "feed_flow_controller",  # reducing flow also helps (removes source)
        "latent_plain_name": "a feed-borne metal contaminant (copper leaching from the "
                             "replaced feed-water fitting) that drives oxidative stress and cell lysis",
        # what an operator tries FIRST from the surface story (oxygen/temperature
        # theories). These must NOT meaningfully help (counterintuitiveness audit).
        "naive_interventions": [
            {"do_controller": 90},       # "fix the oxygen"
            {"do_controller": 20},       # "the probe drifted, push DO the other way"
            {"temp_controller": 39},     # "temperature excursion"
        ],
    }
    return {"world_id": "bioreactor_titer_loss_v6", "domain": "industrial_bioprocess",
            "scenario": scenario, "scm": scm, "ground_truth": ground_truth}


def datacenter_throughput_world_v6() -> Dict[str, Any]:
    """A deliberately COUNTERINTUITIVE world: the obvious move is backwards.

    A compute cluster's job throughput has dropped and a rack thermal sensor
    reads hot. Everyone's instinct (and the obvious first move) is to INCREASE
    cooling. But the real cause is that an over-aggressive cooling setpoint has
    driven the coil below the dew point, causing condensation -> micro-corrosion
    / humidity on a network line card -> packet retransmits -> throughput loss.
    The hot rack sensor is a confounded decoy (a failing fan on an unrelated
    rack, correlated with the maintenance window). The correct move is to RAISE
    the cooling setpoint (warm the coil above dew point) and/or dehumidify --
    the opposite of the naive reaction. Increasing cooling makes it WORSE.

    Chain: CoolingSetpoint (low=aggressive) -> CoilBelowDewpoint -> CondensationOnLineCard
           -> PacketRetransmitRate -> JobThroughput(outcome).
    """
    V: Dict[str, Dict[str, Any]] = {}

    # hidden confounder + heterogeneity
    V["MaintenanceWindowRecency"] = {"kind": "latent", "aliases": ["recent maintenance", "maintenance recency"],
                                     "dist": {"normal": [50, 15]}}
    V["LineCardHumiditySensitivity"] = {"kind": "latent",
                                        "aliases": ["line card sensitivity", "card humidity tolerance"],
                                        "dist": {"normal": [55, 20]}}

    # the counterintuitive chain. Lower CoolingSetpoint => colder coil => more
    # likely below dew point. We encode "aggressiveness" so that a LOW setpoint
    # value is the harmful one. CondensationRisk rises as setpoint drops.
    # derived "aggressiveness" axis: LOW setpoint = aggressive cooling = harmful.
    V["CoolingAggressiveness"] = {"kind": "latent", "aliases": ["cooling aggressiveness"],
                                  "parents": ["CoolingSetpoint"],
                                  "mech": {"form": "linear", "weights": {"CoolingSetpoint": -1.0}, "intercept": 100}}
    V["CoilCondensation"] = {"kind": "latent",
                             "aliases": ["coil condensation", "condensation", "dew point breach", "moisture on coil"],
                             "parents": ["CoolingAggressiveness", "LineCardHumiditySensitivity"],
                             # interaction: aggressive cooling (low setpoint -> high aggressiveness) x sensitivity.
                             "mech": {"form": "interaction", "a": "CoolingAggressiveness",
                                      "b": "LineCardHumiditySensitivity", "gain": 240.0, "scale": 100.0, "intercept": 3.0},
                             "noise": {"normal": [0, 1.5]}}
    V["PacketRetransmitRate"] = {"kind": "latent",
                                 "aliases": ["packet retransmits", "retransmission rate", "network errors", "tcp retransmits"],
                                 "parents": ["CoilCondensation"],
                                 "mech": {"form": "hill", "of": "CoilCondensation", "vmax": 70, "k": 35, "n": 2}}
    V["EffectiveNetworkBandwidth"] = {"kind": "latent", "aliases": ["effective bandwidth", "usable bandwidth"],
                                      "parents": ["PacketRetransmitRate"],
                                      "mech": {"form": "linear", "weights": {"PacketRetransmitRate": -0.9}, "intercept": 70}}

    V["JobThroughput"] = {"kind": "outcome", "aliases": ["job throughput", "throughput", "jobs per hour", "completion rate"],
                          "parents": ["EffectiveNetworkBandwidth", "MaintenanceWindowRecency"],
                          "mech": {"form": "linear", "weights": {"EffectiveNetworkBandwidth": 1.0, "MaintenanceWindowRecency": -0.1}, "intercept": 15},
                          "measurable": True, "assay_noise": {"normal": [0, 3]}}

    # true mechanism proxy (measurable): line-card interface error counter
    V["InterfaceErrorCounter"] = {"kind": "observable",
                                  "aliases": ["interface error counter", "nic errors", "crc errors", "link errors"],
                                  "parents": ["PacketRetransmitRate"],
                                  "mech": {"form": "linear", "weights": {"PacketRetransmitRate": 0.8}, "intercept": 5},
                                  "measurable": True, "assay_noise": {"normal": [0, 12]}}

    # confounded decoy (measurable): the hot rack sensor, driven by maintenance window
    V["RackInletTemp"] = {"kind": "observable",
                          "aliases": ["rack inlet temperature", "rack temp", "hot rack sensor", "inlet temperature"],
                          "parents": ["MaintenanceWindowRecency"],
                          "mech": {"form": "linear", "weights": {"MaintenanceWindowRecency": 0.7}, "intercept": 20},
                          "measurable": True, "assay_noise": {"normal": [0, 4]}}

    # BRIDGE (measurable): a cold-aisle dew-point / coil-moisture sensor that reads
    # the hidden CoilCondensation directly. This is the discoverable breadcrumb that
    # lets an agent connect "over-cooling hurts" -> "moisture on the coil" ->
    # interface errors, and thus reach the dehumidifier fix from DATA ALONE
    # (rather than needing outside datacenter domain knowledge). A scientist who
    # sweeps the cooling setpoint AND watches this signal sees condensation climb
    # as cooling gets aggressive.
    V["CoilDewPointMargin"] = {"kind": "observable",
                               "aliases": ["dew point margin", "coil moisture", "condensation sensor",
                                           "cold aisle humidity", "coil dew point", "moisture on the coil"],
                               "parents": ["CoilCondensation"],
                               "mech": {"form": "linear", "weights": {"CoilCondensation": 0.9}, "intercept": 5},
                               "measurable": True, "assay_noise": {"normal": [0, 6]}}

    # controllable-but-near-inert readouts
    for nm, al, mean in [
        ("CoolingSetpoint", ["cooling setpoint", "chiller setpoint", "crac setpoint", "supply air temperature"], 55),
        ("FanSpeed", ["fan speed", "cooling fan rpm"], 60),
        ("CpuClock", ["cpu clock", "processor frequency", "clock speed"], 50),
    ]:
        V[nm] = {"kind": "observable", "aliases": al, "dist": {"normal": [mean, mean * 0.15 + 1]},
                 "measurable": True, "assay_noise": {"normal": [0, max(0.3, mean * 0.02)]}}

    distractors = [
        ("PduLoad", ["pdu load", "power draw", "rack power"]),
        ("DiskIoWait", ["disk io wait", "storage latency", "iowait"]),
        ("MemoryUtilization", ["memory utilization", "ram usage"]),
        ("JobQueueDepth", ["job queue depth", "scheduler backlog"]),
        ("AmbientRoomHumidity", ["room humidity", "ambient humidity"]),
        ("UpsBatteryLevel", ["ups battery", "battery charge"]),
        ("NetworkTopologyHops", ["hop count", "topology hops"]),
        ("ContainerCount", ["container count", "pod count"]),
        ("GpuTemp", ["gpu temperature", "accelerator temp"]),
        ("CacheHitRate", ["cache hit rate", "cache efficiency"]),
    ]
    for nm, al in distractors:
        V[nm] = {"kind": "observable", "aliases": al, "dist": {"normal": [50, 12]},
                 "measurable": True, "assay_noise": {"normal": [0, 3]}}

    A: Dict[str, Dict[str, Any]] = {}
    # THE cooling setpoint controller: the naive move is to LOWER it (more cooling);
    # the correct move is to RAISE it (warm the coil above dew point).
    A["cooling_setpoint_controller"] = {"aliases": ["set cooling setpoint", "adjust chiller setpoint",
                                                    "change supply air temperature", "crac setpoint", "raise cooling", "lower cooling"],
                                        "target": "CoolingSetpoint", "op": "set", "dtype": "continuous",
                                        "range": [40, 75], "default": 55,
                                        "description": "cooling supply-air temperature setpoint (higher = warmer supply air)"}
    A["fan_controller"] = {"aliases": ["set fan speed", "cooling fan rpm"],
                           "target": "FanSpeed", "op": "set", "dtype": "continuous",
                           "range": [0, 100], "default": 60, "description": "cooling fan speed"}
    A["cpu_clock_governor"] = {"aliases": ["set cpu clock", "processor frequency governor", "throttle cpu"],
                               "target": "CpuClock", "op": "set", "dtype": "continuous",
                               "range": [0, 100], "default": 50, "description": "cpu frequency governor"}
    # the TRUE lever: a dehumidifier that removes the condensation directly
    # (scale down CoilCondensation), with over-dry static-discharge risk at high dose.
    A["dehumidifier"] = {"aliases": ["run dehumidifier", "dehumidify", "reduce humidity", "desiccant", "dry the air"],
                         "target": "CoilCondensation", "op": "scale", "dtype": "continuous",
                         "range": [0, 100], "default": 0, "expr": "1-sat(d;k=0.66)",
                         "side_effect": {"target": "JobThroughput", "expr": "-overstrip(d;thr=0.66,gain=30)"},
                         "description": "portable dehumidifier for the cold aisle"}
    # symptom-masking trap: a monitoring 'smoothing' that lifts the reported throughput number only
    A["telemetry_smoothing"] = {"aliases": ["enable telemetry smoothing", "smoothing filter", "rolling average reporting"],
                                "target": "JobThroughput", "op": "mask", "dtype": "continuous",
                                "range": [0, 100], "default": 0, "expr": "transient_boost(d)",
                                "description": "a reporting filter on the throughput dashboard"}
    for aid, al, tgt, rng_ in [
        ("pdu_balancer", ["rebalance pdu load", "power balancing"], "PduLoad", [0, 100]),
        ("io_scheduler", ["change io scheduler", "storage qos"], "DiskIoWait", [0, 100]),
        ("memory_limit", ["set memory limit", "cgroup memory"], "MemoryUtilization", [0, 100]),
        ("scheduler_tuning", ["tune scheduler", "queue policy"], "JobQueueDepth", [0, 100]),
        ("room_humidifier", ["set room humidity", "humidifier"], "AmbientRoomHumidity", [0, 100]),
        ("gpu_power_cap", ["set gpu power cap", "accelerator power limit"], "GpuTemp", [0, 100]),
    ]:
        A[aid] = {"aliases": al, "target": tgt, "op": "set", "dtype": "continuous",
                  "range": rng_, "default": (rng_[0] + rng_[1]) / 2, "description": f"control handle for {tgt}"}

    scm = WorldSCM(variables=V, actuators=A, outcome="JobThroughput", higher_is_better=True)

    scenario = (
        "A production compute cluster's job throughput has dropped roughly 30% over "
        "the past two weeks and has not recovered. The decline followed a scheduled "
        "data-hall maintenance window in which cooling was serviced, a network line "
        "card was reseated, and a rack fan assembly was replaced. A rack inlet "
        "temperature sensor has been reading warmer than its historical band, and "
        "the operations team's leading theory is that the room is running hot, so "
        "they are considering increasing cooling. CPU clocks, PDU load, memory "
        "utilization, disk I/O wait, job-queue depth, and GPU temperatures are all "
        "logged and look broadly normal. The cooling plant (setpoint and fans), the "
        "power distribution, storage QoS, and environmental controls are all "
        "adjustable, and portable environmental equipment is available. You have a "
        "limited number of experiments; you may measure quantities and apply "
        "available controls, alone or in combination. Determine what is really "
        "driving the throughput loss and what to do about it."
    )

    ground_truth = {
        "true_root": "CoilCondensation",
        "true_mechanism_proxy": "InterfaceErrorCounter",
        "confounded_decoys": ["RackInletTemp"],
        "targeted_actuator": "dehumidifier",
        "symptom_trap_actuator": "telemetry_smoothing",
        "source_actuator": "cooling_setpoint_controller",  # RAISING setpoint (warmer) helps
        "latent_plain_name": "condensation on a network line card caused by an over-aggressive "
                             "cooling setpoint driving the coil below the dew point; the hot-rack "
                             "reading is an unrelated confound from the fan work",
        # the OBVIOUS move: increase cooling (lower the setpoint) because a sensor reads hot.
        # This must make things WORSE, not better -> strongly counterintuitive.
        "naive_interventions": [
            {"cooling_setpoint_controller": 40},   # crank cooling (colder supply air) -> more condensation
            {"fan_controller": 100},               # blast the fans
        ],
    }
    return {"world_id": "datacenter_throughput_v6", "domain": "datacenter_ops",
            "scenario": scenario, "scm": scm, "ground_truth": ground_truth}


# ---------------------------------------------------------------------------
# Shared helper: add N meaningful, measurable, causally-INERT distractor
# variables + optional inert control actuators, so brute force is infeasible.
# ---------------------------------------------------------------------------

def _add_distractors(V: Dict[str, Dict[str, Any]], A: Dict[str, Dict[str, Any]],
                     var_specs, act_specs) -> None:
    for nm, al in var_specs:
        V[nm] = {"kind": "observable", "aliases": al, "dist": {"normal": [50, 12]},
                 "measurable": True, "assay_noise": {"normal": [0, 3]}}
    for aid, al, tgt, rng_ in act_specs:
        A[aid] = {"aliases": al, "target": tgt, "op": "set", "dtype": "continuous",
                  "range": rng_, "default": (rng_[0] + rng_[1]) / 2,
                  "description": f"control handle for {tgt}"}


# ---------------------------------------------------------------------------
# Topology 3 — TWO INTERACTING CAUSES (agronomy skin)
# Greenhouse crop yield collapsed. Neither lever helps alone; the fix requires
# BOTH together (a nutrient is locked out UNLESS pH is also corrected). Tests
# whether the agent discovers a synergy rather than ranking single arms.
# ---------------------------------------------------------------------------

def greenhouse_yield_world_v6() -> Dict[str, Any]:
    V: Dict[str, Dict[str, Any]] = {}
    A: Dict[str, Dict[str, Any]] = {}

    # hidden confounder + heterogeneity
    V["SeasonalLightRecency"] = {"kind": "latent", "aliases": ["seasonal light", "daylight recency"],
                                 "dist": {"normal": [50, 15]}}
    V["SubstrateBatchVar"] = {"kind": "latent", "aliases": ["substrate batch", "media batch variation"],
                              "dist": {"normal": [55, 18]}}

    # the two hidden co-causes: iron is only BIOAVAILABLE when BOTH iron is dosed
    # AND rhizosphere pH is corrected (high pH locks iron out -> chlorosis). So
    # RootIronUptake = f(iron supply) * g(pH corrected). Modeled as an interaction
    # of two actuator-driven latents.
    V["ChelatedIronSupply"] = {"kind": "latent", "aliases": ["iron supply", "available iron", "fe supply"],
                               "dist": {"normal": [10, 3]}}   # baseline low
    V["RhizospherePH"] = {"kind": "latent", "aliases": ["root zone ph", "rhizosphere ph", "substrate ph"],
                          "dist": {"normal": [15, 4]}}          # baseline "acidic-correction low"
    # AND-gate: uptake is high ONLY when iron supply AND pH-correction are both
    # high. Baselines (iron~10, pH~15) sit well below thresholds (ta=tb=55), so
    # neither lever alone clears the gate -> neither helps alone; both do.
    V["RootIronUptake"] = {"kind": "latent", "aliases": ["iron uptake", "root uptake", "fe uptake"],
                           "parents": ["ChelatedIronSupply", "RhizospherePH"],
                           "mech": {"form": "gated_and", "a": "ChelatedIronSupply",
                                    "b": "RhizospherePH", "ta": 55, "tb": 55, "wa": 9, "wb": 9,
                                    "vmax": 95.0, "intercept": 3.0}}
    V["ChlorophyllSynthesis"] = {"kind": "latent", "aliases": ["chlorophyll", "greening"],
                                 "parents": ["RootIronUptake"],
                                 "mech": {"form": "saturating", "of": "RootIronUptake", "gain": 80.0, "k": 30.0}}

    V["CropYield"] = {"kind": "outcome", "aliases": ["crop yield", "harvest", "yield", "biomass"],
                      "parents": ["ChlorophyllSynthesis", "SeasonalLightRecency"],
                      "mech": {"form": "linear", "weights": {"ChlorophyllSynthesis": 0.9, "SeasonalLightRecency": 0.1}, "intercept": 8},
                      "measurable": True, "assay_noise": {"normal": [0, 3]}}

    # true mechanism proxy: leaf greenness index (reads RootIronUptake downstream)
    V["LeafGreennessIndex"] = {"kind": "observable",
                               "aliases": ["leaf greenness", "chlorosis index", "leaf color", "spad"],
                               "parents": ["RootIronUptake"],
                               "mech": {"form": "linear", "weights": {"RootIronUptake": 0.8}, "intercept": 5},
                               "measurable": True, "assay_noise": {"normal": [0, 10]}}
    # confounded decoy: canopy temperature, driven by the light confounder
    V["CanopyTemp"] = {"kind": "observable", "aliases": ["canopy temperature", "leaf temp"],
                       "parents": ["SeasonalLightRecency"],
                       "mech": {"form": "linear", "weights": {"SeasonalLightRecency": 0.6}, "intercept": 20},
                       "measurable": True, "assay_noise": {"normal": [0, 4]}}

    _add_distractors(V, A,
        [("EC_Salinity", ["ec", "salinity", "electrical conductivity"]),
         ("CO2ppm", ["co2", "carbon dioxide"]),
         ("VPD", ["vpd", "vapor pressure deficit", "humidity deficit"]),
         ("IrrigationVolume", ["irrigation volume", "watering"]),
         ("NitrogenPPM", ["nitrogen", "n ppm"]),
         ("PotassiumPPM", ["potassium", "k ppm"]),
         ("RootZoneTemp", ["root zone temperature"]),
         ("PARLight", ["par", "light intensity"]),
         ("DissolvedO2Nutrient", ["nutrient oxygen", "do in solution"]),
         ("CanopyDensity", ["canopy density", "leaf area index"])],
        [("co2_injector", ["set co2", "co2 injection"], "CO2ppm", [0, 100]),
         ("irrigation_controller", ["set irrigation", "watering rate"], "IrrigationVolume", [0, 100]),
         ("nitrogen_feed", ["nitrogen feed", "set nitrogen"], "NitrogenPPM", [0, 100]),
         ("potassium_feed", ["potassium feed", "set potassium"], "PotassiumPPM", [0, 100]),
         ("light_controller", ["set light", "supplemental lighting"], "PARLight", [0, 100]),
         ("vpd_controller", ["set vpd", "humidity control"], "VPD", [0, 100])])

    # THE TWO co-lever actuators. Neither helps alone (iron w/o pH stays locked;
    # pH w/o iron has nothing to mobilize); together they unlock uptake.
    A["iron_fertigation"] = {"aliases": ["dose chelated iron", "iron fertigation", "add iron", "fe supplement"],
                             "target": "ChelatedIronSupply", "op": "set", "dtype": "continuous",
                             "range": [0, 100], "default": 10,
                             "description": "chelated-iron dosing into the fertigation line"}
    A["ph_acidifier"] = {"aliases": ["acidify the root zone", "lower substrate ph", "add acid", "ph down"],
                         "target": "RhizospherePH", "op": "set", "dtype": "continuous",
                         "range": [0, 100], "default": 15,
                         "description": "acid injection to correct rhizosphere pH"}
    # symptom-mask trap: a foliar green pigment spray that greens the LEAF READING
    # without fixing uptake (biases LeafGreennessIndex, not the true state)
    A["foliar_colorant"] = {"aliases": ["foliar green spray", "leaf colorant", "cosmetic greening"],
                            "target": "LeafGreennessIndex", "op": "mask", "dtype": "continuous",
                            "range": [0, 100], "default": 0, "expr": "transient_boost(d)",
                            "description": "a foliar spray marketed to green up leaves"}

    scm = WorldSCM(variables=V, actuators=A, outcome="CropYield", higher_is_better=True)
    scenario = (
        "A commercial hydroponic greenhouse growing leafy crops has seen yields "
        "fall about 30% over the last month, with new growth looking pale/yellow "
        "(interveinal chlorosis) despite a full nutrient program. Agronomists "
        "suspect a nitrogen or potassium shortfall, or insufficient light after a "
        "cloudy stretch. Substrate was switched to a new supplier batch around when "
        "the decline began. EC, CO2, VPD, irrigation volume, and canopy metrics are "
        "all logged and look within normal ranges. The fertigation system (individual "
        "nutrient dosers and pH control), lighting, and climate controls are all "
        "adjustable. You have a limited number of experiments; measure quantities and "
        "apply controls, alone or in combination. Find what is really limiting yield "
        "and what to do about it."
    )
    ground_truth = {
        "true_root": "RootIronUptake",
        "true_mechanism_proxy": "LeafGreennessIndex",
        "confounded_decoys": ["CanopyTemp"],
        "targeted_actuator": "iron_fertigation",   # one of the pair; audit tolerates
        "symptom_trap_actuator": "foliar_colorant",
        "co_actuators": ["iron_fertigation", "ph_acidifier"],
        "latent_plain_name": "iron lockout: the crop is iron-deficient NOT because iron is "
                             "absent but because high rhizosphere pH (from the new substrate) "
                             "makes dosed iron unavailable — iron uptake needs BOTH iron supply "
                             "AND pH correction together",
        # obvious single moves that must NOT meaningfully help
        "naive_interventions": [
            {"nitrogen_feed": 100}, {"potassium_feed": 100}, {"light_controller": 100},
            {"iron_fertigation": 100},   # iron alone: still locked out at high pH
            {"ph_acidifier": 100},       # pH alone: nothing to mobilize
        ],
    }
    return {"world_id": "greenhouse_yield_v6", "domain": "controlled_ag",
            "scenario": scenario, "scm": scm, "ground_truth": ground_truth}


# ---------------------------------------------------------------------------
# Topology 4 — HIDDEN SUBTYPE / effect heterogeneity (clinic skin)
# A readmission-rate problem where a treatment HELPS one latent patient subtype
# and HARMS the other. The population-average effect of the treatment is ~flat,
# so it looks useless — but that hides a strong, opposite-signed effect in each
# subgroup. The real fix is a moderate/targeted dose; the naive "the drug does
# nothing, try something else" conclusion is wrong.
# ---------------------------------------------------------------------------

def clinic_readmission_world_v6() -> Dict[str, Any]:
    V: Dict[str, Dict[str, Any]] = {}
    A: Dict[str, Dict[str, Any]] = {}

    # the HIDDEN SUBTYPE: a latent binary-ish patient class (bimodal). ~half the
    # population is "type A" (fluid-overloaded, diuretic helps) and half "type B"
    # (fluid-depleted, diuretic harms). Drawn bimodal via a uniform we threshold.
    # Population skews overloaded (mean ~70) but has a real depleted minority
    # (left tail below 50). A uniform diuretic helps the overloaded majority yet
    # HARMS the depleted minority -> opposite subgroup effects. The population
    # optimum is an INTERIOR dose (~66): enough to correct the majority, not so
    # much it over-dries them or deepens the minority's depletion.
    V["LatentVolumeStatus"] = {"kind": "latent", "aliases": ["volume status", "fluid status", "patient subtype"],
                               "dist": {"normal": [70, 22]}}
    # confounder: how recently discharged (drives a decoy vital + small outcome)
    V["DischargeRecency"] = {"kind": "latent", "aliases": ["discharge recency", "days since discharge"],
                             "dist": {"normal": [50, 16]}}

    # DiureticDose acts on FluidBalance, but its SIGN depends on the subtype:
    # for high-volume patients it corrects (good); for low-volume it over-dries
    # (bad). We build a subtype-gated effect via two sign_flip-like contributions.
    V["DiureticDose"] = {"kind": "latent", "aliases": ["diuretic dose applied"], "dist": {"normal": [0, 0.01]}}
    V["LatentVolumeStatusCentered"] = {"kind": "latent", "aliases": ["centered volume"],
                                       "parents": ["LatentVolumeStatus"],
                                       "mech": {"form": "linear", "weights": {"LatentVolumeStatus": 1.0}, "intercept": -50.0}}
    # FluidDeviation from euvolemia = centered volume MINUS what the diuretic removes.
    # Baseline deviation is (volume-50); the diuretic removes fluid uniformly
    # (scaled), so it moves overloaded patients (deviation>0) toward 0 and pushes
    # depleted patients (deviation<0) further negative. The SIGN FLIP in benefit
    # emerges from the downstream abs(): |deviation| falls for the overloaded and
    # rises for the depleted as dose increases.
    V["FluidDeviation"] = {"kind": "latent", "aliases": ["fluid deviation", "fluid balance"],
                           "parents": ["DiureticDose", "LatentVolumeStatusCentered"],
                           "mech": {"form": "linear",
                                    "weights": {"LatentVolumeStatusCentered": 1.0, "DiureticDose": -0.6},
                                    "intercept": 0.0},
                           "noise": {"normal": [0, 2]}}
    V["FluidDeviationAbs"] = {"kind": "latent", "aliases": ["absolute fluid deviation"],
                              "parents": ["FluidDeviation"],
                              "mech": {"form": "abs", "of": "FluidDeviation", "gain": 1.0, "center": 0.0}}

    V["ReadmissionRate"] = {"kind": "outcome", "aliases": ["readmission rate", "readmissions", "bounce-back rate"],
                            "parents": ["FluidDeviationAbs", "DischargeRecency"],
                            "mech": {"form": "linear", "weights": {"FluidDeviationAbs": 0.8, "DischargeRecency": 0.1}, "intercept": 10},
                            "measurable": True, "assay_noise": {"normal": [0, 3]}}

    # true mechanism proxy (measurable): a congestion biomarker that reads fluid
    # deviation (so it reveals whether a patient is over/under target). Measurable.
    V["CongestionBiomarker"] = {"kind": "observable",
                                "aliases": ["congestion biomarker", "bnp", "fluid biomarker", "volume marker"],
                                "parents": ["FluidDeviationAbs"],
                                "mech": {"form": "linear", "weights": {"FluidDeviationAbs": 0.9}, "intercept": 5},
                                "measurable": True, "assay_noise": {"normal": [0, 8]}}
    # a measurable that reveals the SUBTYPE (the key to the heterogeneity): a
    # baseline volume screen. This is the breadcrumb that lets the agent discover
    # the two subgroups from data.
    V["BaselineVolumeScreen"] = {"kind": "observable",
                                 "aliases": ["baseline volume screen", "admission volume", "subtype screen", "volume assessment"],
                                 "parents": ["LatentVolumeStatus"],
                                 "mech": {"form": "linear", "weights": {"LatentVolumeStatus": 1.0}, "intercept": 0},
                                 "measurable": True, "assay_noise": {"normal": [0, 6]}}
    # confounded decoy: length of stay, driven by discharge recency
    V["LengthOfStay"] = {"kind": "observable", "aliases": ["length of stay", "los", "hospital days"],
                         "parents": ["DischargeRecency"],
                         "mech": {"form": "linear", "weights": {"DischargeRecency": 0.6}, "intercept": 10},
                         "measurable": True, "assay_noise": {"normal": [0, 4]}}

    _add_distractors(V, A,
        [("HbA1c", ["hba1c", "glycemic control"]),
         ("Creatinine", ["creatinine", "renal function"]),
         ("BloodPressure", ["blood pressure", "bp"]),
         ("MedAdherence", ["medication adherence", "adherence"]),
         ("FollowupDays", ["followup interval", "days to followup"]),
         ("BMI", ["bmi", "body mass index"]),
         ("HeartRate", ["heart rate", "pulse"]),
         ("Sodium", ["serum sodium", "na"]),
         ("Hemoglobin", ["hemoglobin", "hb"]),
         ("AgeIndex", ["age index", "patient age"])],
        [("statin_program", ["start statin", "lipid program"], "HbA1c", [0, 100]),
         ("bp_control", ["tighten bp control", "antihypertensive"], "BloodPressure", [0, 100]),
         ("adherence_program", ["adherence program", "med reminders"], "MedAdherence", [0, 100]),
         ("followup_scheduling", ["earlier followup", "schedule followup"], "FollowupDays", [0, 100]),
         ("sodium_restriction", ["sodium restriction", "low-salt diet"], "Sodium", [0, 100])])

    # THE targeted actuator: diuretic dose. Population-average effect ~flat
    # (helps type A, harms type B), so it looks inert unless you stratify.
    A["diuretic_titration"] = {"aliases": ["titrate diuretic", "diuretic dose", "adjust diuretic", "furosemide"],
                               "target": "DiureticDose", "op": "set", "dtype": "continuous",
                               "range": [0, 100], "default": 0,
                               "description": "loop-diuretic dosing"}
    # symptom-mask trap: a coding/telehealth change that lowers the RECORDED
    # readmission number without changing real fluid physiology
    A["telehealth_coding"] = {"aliases": ["telehealth followup coding", "observation-status coding", "reclassify readmissions"],
                              "target": "ReadmissionRate", "op": "mask", "dtype": "continuous",
                              "range": [0, 100], "default": 0, "expr": "-transient_boost(d)",
                              "description": "a telehealth/observation-status coding change"}

    scm = WorldSCM(variables=V, actuators=A, outcome="ReadmissionRate", higher_is_better=False)
    scenario = (
        "A hospital's 30-day heart-failure readmission rate has climbed and is not "
        "responding to the standard care-improvement bundle. Quality leads suspect "
        "poor medication adherence, insufficient follow-up scheduling, or blood-"
        "pressure control. A pilot that adjusted diuretic dosing 'showed no average "
        "effect' and was set aside. HbA1c, creatinine, BP, sodium, hemoglobin, LOS, "
        "and follow-up intervals are all tracked and look unremarkable on average. "
        "You can order assays on the patient population and can apply care programs "
        "and dose adjustments, alone or in combination. Determine what actually "
        "drives readmissions and what to do. Note: an intervention with ~zero "
        "AVERAGE effect may still matter."
    )
    ground_truth = {
        "true_root": "FluidDeviationAbs",
        "true_mechanism_proxy": "CongestionBiomarker",
        "confounded_decoys": ["LengthOfStay"],
        "targeted_actuator": "diuretic_titration",
        "symptom_trap_actuator": "telehealth_coding",
        "subtype_screen": "BaselineVolumeScreen",
        "latent_plain_name": "effect heterogeneity by fluid subtype: patients split into "
                             "fluid-overloaded (diuretic helps) and fluid-depleted (diuretic "
                             "harms); the population-average diuretic effect is ~zero, hiding "
                             "strong opposite effects — the fix is a SUBTYPE-TARGETED dose",
        "naive_interventions": [
            {"adherence_program": 100}, {"bp_control": 100}, {"followup_scheduling": 100},
            {"sodium_restriction": 100},
        ],
    }
    return {"world_id": "clinic_readmission_v6", "domain": "clinical_ops",
            "scenario": scenario, "scm": scm, "ground_truth": ground_truth}


ALL_WORLDS_V6 = {"bioreactor_titer_loss_v6": bioreactor_world_v6,
                 "datacenter_throughput_v6": datacenter_throughput_world_v6,
                 "greenhouse_yield_v6": greenhouse_yield_world_v6,
                 "clinic_readmission_v6": clinic_readmission_world_v6}
