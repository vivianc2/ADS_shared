#!/usr/bin/env python3
"""Domain skins for RPG v7 structural sampling.

A skin supplies meaningful, non-leaking names for the roles the sampler needs,
plus a scenario template. The sampler draws a skin, then draws names from its
banks. Names are meaningful (so world knowledge is usable) but which named thing
is the true cause is not implied by the name — that must be learned from data.

Each skin provides pools keyed by ROLE so the same abstract graph can be dressed
in different domains. Pools are deliberately larger than any single world needs.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _mk(name, *aliases):
    return {"name": name, "aliases": list(aliases)}


SKINS: Dict[str, Dict[str, Any]] = {
    "bioprocess": {
        "outcome": _mk("ProductTiter", "product titer", "protein yield", "titer", "yield"),
        "root_cause_pool": [
            _mk("DissolvedMetal", "dissolved metal", "leached metal", "trace metals", "metal ions"),
            _mk("EndotoxinLoad", "endotoxin load", "endotoxin", "pyrogen level"),
            _mk("ShearStressLoad", "shear stress", "mechanical shear", "hydrodynamic stress"),
        ],
        "mediator_pool": [
            _mk("OxidativeStress", "oxidative stress", "reactive oxygen species", "ROS"),
            _mk("MembraneDamage", "membrane damage", "cell envelope stress"),
            _mk("MetabolicFluxLoss", "metabolic flux loss", "carbon flux drop"),
            _mk("ProteaseActivity", "protease activity", "proteolysis"),
            _mk("MisfoldingRate", "misfolding rate", "aggregation propensity"),
        ],
        "proxy_pool": [
            _mk("BrothTurbidity", "broth turbidity", "cloudiness", "lysis marker"),
            _mk("LDHRelease", "ldh release", "cell lysis marker", "membrane leakage"),
            _mk("FragmentedProteinPct", "fragmented protein", "clipping percent"),
        ],
        "confounder_pool": [
            _mk("BatchSeedAge", "seed age", "inoculum age", "cell line age"),
            _mk("HarvestDayOffset", "harvest day", "days in culture"),
        ],
        "decoy_pool": [
            _mk("DissolvedOxygen", "dissolved oxygen", "DO", "pO2"),
            _mk("ViableCellDensity", "viable cell density", "VCD", "cell count"),
        ],
        "source_knob_pool": [
            _mk("FeedWaterFlowRate", "feed water flow", "feed flow", "water flow"),
            _mk("MediaExchangeRate", "media exchange rate", "perfusion rate"),
        ],
        "fix_actuator_pool": [
            _mk("ChelatorDosing", "chelating agent", "metal chelator", "sequestering agent", "chelation"),
            _mk("AntioxidantDosing", "antioxidant additive", "radical scavenger"),
            _mk("EndotoxinScrubber", "endotoxin scrubber", "polymyxin filter"),
        ],
        "trap_actuator_pool": [
            _mk("StabilizerAdditive", "protein stabilizer", "formulation stabilizer", "protectant"),
            _mk("AssayCalibrationShift", "assay recalibration", "reporting calibration"),
        ],
        "inert_var_pool": [
            _mk("AntifoamLevel", "antifoam", "defoamer"),
            _mk("GlucoseFeedRate", "glucose feed", "sugar feed"),
            _mk("GlutamineConc", "glutamine", "amino acid feed"),
            _mk("LactateConc", "lactate", "lactic acid"),
            _mk("AmmoniaConc", "ammonia", "ammonium"),
            _mk("Osmolality", "osmolality", "osmotic pressure"),
            _mk("CO2Level", "dissolved co2", "pCO2"),
            _mk("VesselPressure", "headspace pressure", "vessel pressure"),
            _mk("AgitationRate", "agitation", "impeller speed", "rpm"),
            _mk("GasSpargeRate", "sparge rate", "aeration rate"),
            _mk("CoolingWaterTemp", "cooling water", "jacket temperature"),
            _mk("Temperature", "temperature", "culture temperature"),
            _mk("pH", "ph", "acidity"),
        ],
        "scenario": (
            "A {vessel} producing {product} has seen {outcome_phrase} fall about "
            "{drop}% over the last {period}, and the loss has persisted across "
            "several batches. The decline began shortly after {trigger_event}. The "
            "operators' leading theories are {naive_theories}. Batch records note "
            "{surface_clue}. Routine panels look broadly within range. The "
            "{controllable_systems} are all instrumented and adjustable. You have a "
            "limited number of experiments; you may measure quantities and apply "
            "available controls or additives, alone or in combination. Determine "
            "what is really driving the loss and what to do about it."
        ),
        "fills": {
            "vessel": "2,000 L mammalian-cell bioreactor",
            "product": "a secreted therapeutic protein",
            "outcome_phrase": "product titer",
            "drop": "30", "period": "three weeks",
            "trigger_event": "a maintenance shutdown in which a feed-water line fitting was replaced and probes were recalibrated",
            "naive_theories": "dissolved-oxygen control drift or a temperature excursion",
            "surface_clue": "the broth has looked cloudier than usual at harvest",
            "controllable_systems": "feed-water system, gas train, cooling jacket, and dosing pumps",
        },
    },
    "datacenter": {
        "outcome": _mk("JobThroughput", "job throughput", "throughput", "jobs per hour"),
        "root_cause_pool": [
            _mk("CoilCondensation", "coil condensation", "condensation", "moisture on the coil"),
            _mk("MicroArcing", "micro-arcing", "electrical arcing", "connector arcing"),
        ],
        "mediator_pool": [
            _mk("InterfaceErrorRate", "interface errors", "link errors", "crc errors"),
            _mk("PacketRetransmit", "packet retransmits", "retransmission rate"),
            _mk("EffectiveBandwidth", "effective bandwidth", "usable bandwidth"),
        ],
        "proxy_pool": [
            _mk("DewPointMargin", "dew point margin", "coil moisture", "condensation sensor"),
            _mk("NicErrorCounter", "nic error counter", "interface error counter"),
        ],
        "confounder_pool": [
            _mk("MaintenanceRecency", "recent maintenance", "maintenance recency"),
            _mk("TenantMixShift", "tenant mix", "workload mix shift"),
        ],
        "decoy_pool": [
            _mk("RackInletTemp", "rack inlet temperature", "hot rack sensor", "rack temp"),
            _mk("PduLoad", "pdu load", "rack power draw"),
        ],
        "source_knob_pool": [
            _mk("CoolingAggressiveness", "cooling aggressiveness", "how hard cooling runs"),
        ],
        "fix_actuator_pool": [
            _mk("Dehumidifier", "dehumidifier", "dehumidify", "reduce humidity", "desiccant"),
            _mk("ConnectorReseat", "connector reseat program", "reseat connectors"),
        ],
        "trap_actuator_pool": [
            _mk("TelemetrySmoothing", "telemetry smoothing", "rolling-average reporting"),
        ],
        "inert_var_pool": [
            _mk("CpuClock", "cpu clock", "processor frequency"),
            _mk("GpuTemp", "gpu temperature", "accelerator temp"),
            _mk("DiskIoWait", "disk io wait", "storage latency"),
            _mk("MemoryUtilization", "memory utilization", "ram usage"),
            _mk("JobQueueDepth", "job queue depth", "scheduler backlog"),
            _mk("AmbientRoomHumidity", "room humidity", "ambient humidity"),
            _mk("UpsBatteryLevel", "ups battery", "battery charge"),
            _mk("CacheHitRate", "cache hit rate", "cache efficiency"),
            _mk("ContainerCount", "container count", "pod count"),
            _mk("FanSpeed", "fan speed", "cooling fan rpm"),
        ],
        "scenario": (
            "A production compute cluster's {outcome_phrase} has dropped about "
            "{drop}% over the past {period} and has not recovered. The decline "
            "followed {trigger_event}. {surface_clue}, and the operations team's "
            "leading theory is {naive_theories}. Standard host metrics look broadly "
            "normal. The {controllable_systems} are all adjustable, and portable "
            "environmental equipment is available. You have a limited number of "
            "experiments; you may measure quantities and apply available controls, "
            "alone or in combination. Determine what is really driving the loss and "
            "what to do about it."
        ),
        "fills": {
            "outcome_phrase": "job throughput", "drop": "30", "period": "two weeks",
            "trigger_event": "a data-hall maintenance window in which cooling was serviced and a line card was reseated",
            "naive_theories": "that the room is running hot, so they are considering increasing cooling",
            "surface_clue": "A rack inlet-temperature sensor has been reading warmer than its historical band",
            "controllable_systems": "cooling plant, power distribution, storage QoS, and environmental controls",
        },
    },
    "watertreatment": {
        "outcome": _mk("WaterClarityScore", "water clarity score", "clarity index", "clearness rating"),
        "root_cause_pool": [
            _mk("MobilizedScale", "mobilized pipe scale", "scale release", "iron release"),
            _mk("BiofilmSloughing", "biofilm sloughing", "biofilm detachment"),
        ],
        "mediator_pool": [
            _mk("IronParticulate", "iron particulate", "particulate load"),
            _mk("TurbidityLoad", "turbidity load", "suspended solids"),
        ],
        "proxy_pool": [
            _mk("TurbidityNTU", "turbidity ntu", "water turbidity", "cloudiness"),
            _mk("MetalColorimetry", "metal colorimetry", "iron colorimetric reading"),
        ],
        "confounder_pool": [
            _mk("SeasonalDemand", "seasonal demand", "demand swing"),
            _mk("SourceBlendShift", "source blend", "source water mix"),
        ],
        "decoy_pool": [
            _mk("PressureReading", "line pressure", "system pressure"),
            _mk("ResidualChlorine", "residual chlorine", "disinfectant residual"),
        ],
        "source_knob_pool": [
            _mk("LineFlushRate", "line flush rate", "flushing intensity"),
        ],
        "fix_actuator_pool": [
            _mk("CorrosionInhibitor", "corrosion inhibitor", "orthophosphate dosing"),
            _mk("PhStabilizer", "ph stabilizer", "alkalinity adjustment"),
        ],
        "trap_actuator_pool": [
            _mk("MaskingAdditive", "dye-masking additive", "color-masking agent"),
        ],
        "inert_var_pool": [
            _mk("FluorideLevel", "fluoride level", "fluoridation"),
            _mk("WaterTemp", "water temperature", "supply temperature"),
            _mk("ContactTime", "contact time", "detention time"),
            _mk("PumpCycles", "pump cycles", "pump starts"),
            _mk("TankLevel", "tank level", "reservoir level"),
            _mk("HardnessCaCO3", "hardness", "calcium carbonate"),
            _mk("UVDose", "uv dose", "uv disinfection"),
            _mk("FilterHeadloss", "filter headloss", "filter pressure drop"),
            _mk("BackwashRate", "backwash rate", "filter backwash"),
        ],
        "scenario": (
            "A water utility's {outcome_phrase} has dropped about {drop}% over the "
            "past {period} (more discoloration reaching customers). It began after "
            "{trigger_event}. {surface_clue}. Routine water-quality panels look "
            "broadly within range. The {controllable_systems} are adjustable, and "
            "dosing equipment is available. You have a limited number of "
            "experiments; you may measure quantities and apply available controls "
            "or additives, alone or in combination. Determine what is really "
            "driving the drop and what to do about it."
        ),
        "fills": {
            "outcome_phrase": "water clarity score", "drop": "40", "period": "month",
            "trigger_event": "crews increased line flushing following a main repair",
            "naive_theories": "chlorine residual drift or seasonal demand swings",
            "surface_clue": "Tap water looks faintly rusty at some addresses",
            "controllable_systems": "flushing program, disinfection, and chemical dosing",
        },
    },
    "agronomy": {
        "outcome": _mk("CropYield", "crop yield", "marketable yield", "harvest yield", "yield"),
        "root_cause_pool": [
            _mk("MicronutrientLockout", "micronutrient lockout", "trace-nutrient lockout", "nutrient unavailability"),
            _mk("RootZoneSalinity", "root-zone salinity", "salt accumulation", "salinity stress"),
            _mk("SoilbornePathogen", "soilborne pathogen", "root pathogen load", "root rot pressure"),
        ],
        "mediator_pool": [
            _mk("ChlorophyllSynthesis", "chlorophyll synthesis", "greening capacity"),
            _mk("PhotosyntheticEfficiency", "photosynthetic efficiency", "carbon assimilation"),
            _mk("StomatalConductance", "stomatal conductance", "gas exchange"),
            _mk("SugarTranslocation", "sugar translocation", "photosynthate transport"),
            _mk("RootVigor", "root vigor", "root mass index"),
        ],
        "proxy_pool": [
            _mk("LeafGreenness", "leaf greenness", "spad reading", "chlorophyll index"),
            _mk("TissueNutrientAssay", "tissue nutrient assay", "leaf tissue test"),
        ],
        "confounder_pool": [
            _mk("SeasonalLight", "seasonal light", "solar radiation", "daylight hours"),
            _mk("PlantingCohort", "planting cohort", "transplant batch", "crop age"),
        ],
        "decoy_pool": [
            _mk("SoilMoisture", "soil moisture", "root-zone moisture"),
            _mk("CanopyTemp", "canopy temperature", "leaf surface temp"),
        ],
        "source_knob_pool": [
            _mk("IrrigationRate", "irrigation rate", "watering rate", "water delivery"),
            _mk("FertigationRate", "fertigation rate", "nutrient feed rate"),
        ],
        "fix_actuator_pool": [
            _mk("ChelatedMicronutrient", "chelated micronutrient", "chelated iron-zinc", "micronutrient chelate"),
            _mk("SoilAcidifier", "soil acidifier", "ph correction", "elemental sulfur"),
            _mk("BiofungicideDrench", "biofungicide drench", "root drench"),
        ],
        "trap_actuator_pool": [
            _mk("FoliarGreeningSpray", "foliar greening spray", "cosmetic greening agent"),
            _mk("HarvestGradingShift", "harvest grading shift", "grading recalibration"),
        ],
        "inert_var_pool": [
            _mk("SoilPhosphorus", "soil phosphorus", "phosphate level"),
            _mk("SoilPotassium", "soil potassium", "potash level"),
            _mk("RowSpacing", "row spacing", "plant density"),
            _mk("MulchDepth", "mulch depth", "ground cover"),
            _mk("WindExposure", "wind exposure", "shelter index"),
            _mk("CO2Enrichment", "co2 enrichment", "carbon dioxide dosing"),
            _mk("PruningIntensity", "pruning intensity", "canopy pruning"),
            _mk("PollinatorActivity", "pollinator activity", "bee visits"),
            _mk("SoilOrganicMatter", "soil organic matter", "humus content"),
            _mk("DrainageRate", "drainage rate", "field drainage"),
            _mk("SeedTreatment", "seed treatment", "seed coating"),
            _mk("GrowthRegulator", "growth regulator", "plant hormone spray"),
            _mk("NightTemperature", "night temperature", "nocturnal temp"),
        ],
        "scenario": (
            "A commercial {operation} growing {crop} has seen {outcome_phrase} fall "
            "about {drop}% over {period}, and the shortfall has repeated across "
            "successive plantings. The decline set in after {trigger_event}. The "
            "agronomists' leading theories are {naive_theories}. {surface_clue}. "
            "Routine soil and tissue panels read broadly within range. The "
            "{controllable_systems} are all adjustable, and dosing and amendment "
            "equipment is available. You have a limited number of experiments; you "
            "may measure quantities and apply available controls or amendments, "
            "alone or in combination. Determine what is really driving the loss and "
            "what to do about it."
        ),
        "fills": {
            "operation": "greenhouse tomato operation", "crop": "vine tomatoes",
            "outcome_phrase": "marketable yield", "drop": "30", "period": "two growing cycles",
            "trigger_event": "a switch to a new irrigation-water source following a well upgrade",
            "naive_theories": "under-watering or a nitrogen shortfall, so they are considering more water and feed",
            "surface_clue": "New leaves have looked paler than usual between the veins",
            "controllable_systems": "irrigation, fertigation, and amendment dosing",
        },
    },
    "clinical": {
        "outcome": _mk("RecoveryScore", "recovery score", "recovery index", "functional recovery"),
        "root_cause_pool": [
            _mk("OccultInfection", "occult infection", "smoldering infection", "hidden infection"),
            _mk("DrugInteraction", "drug interaction", "medication interaction", "polypharmacy effect"),
            _mk("ElectrolyteImbalance", "electrolyte imbalance", "electrolyte derangement"),
        ],
        "mediator_pool": [
            _mk("SystemicInflammation", "systemic inflammation", "inflammatory cascade"),
            _mk("TissueOxygenDelivery", "tissue oxygen delivery", "oxygen delivery"),
            _mk("OrganPerfusion", "organ perfusion", "end-organ perfusion"),
            _mk("MetabolicStress", "metabolic stress", "catabolic load"),
            _mk("ImmuneExhaustion", "immune exhaustion", "immune depletion"),
        ],
        "proxy_pool": [
            _mk("CReactiveProtein", "c-reactive protein", "crp", "inflammatory marker"),
            _mk("SerumLactate", "serum lactate", "lactate level"),
        ],
        "confounder_pool": [
            _mk("PatientAge", "patient age", "age band"),
            _mk("ComorbidityIndex", "comorbidity index", "baseline frailty"),
        ],
        "decoy_pool": [
            _mk("BodyTemperature", "body temperature", "core temp", "fever reading"),
            _mk("HeartRate", "heart rate", "pulse rate"),
        ],
        "source_knob_pool": [
            _mk("FluidInfusionRate", "fluid infusion rate", "iv fluid rate", "hydration rate"),
            _mk("SupplementalOxygen", "supplemental oxygen", "oxygen flow"),
        ],
        "fix_actuator_pool": [
            _mk("TargetedAntimicrobial", "targeted antimicrobial", "culture-directed antibiotic"),
            _mk("Deprescribe", "deprescribing", "medication withdrawal", "stop offending drug"),
            _mk("ElectrolyteRepletion", "electrolyte repletion", "electrolyte correction"),
        ],
        "trap_actuator_pool": [
            _mk("Antipyretic", "antipyretic", "fever suppressant", "temperature-lowering drug"),
            _mk("ScoreReweighting", "score reweighting", "assessment recalibration"),
        ],
        "inert_var_pool": [
            _mk("BloodPressure", "blood pressure", "arterial pressure"),
            _mk("RespiratoryRate", "respiratory rate", "breathing rate"),
            _mk("UrineOutput", "urine output", "diuresis"),
            _mk("PainScore", "pain score", "reported pain"),
            _mk("MobilizationLevel", "mobilization level", "ambulation"),
            _mk("SleepHours", "sleep hours", "rest duration"),
            _mk("CaloricIntake", "caloric intake", "nutrition intake"),
            _mk("VisitorFrequency", "visitor frequency", "family visits"),
            _mk("RoomOccupancy", "room occupancy", "ward crowding"),
            _mk("PhysioTherapyDose", "physiotherapy dose", "rehab intensity"),
            _mk("SedationLevel", "sedation level", "sedative dose"),
            _mk("BloodGlucose", "blood glucose", "serum glucose"),
            _mk("AmbientNoise", "ambient noise", "ward noise level"),
        ],
        "scenario": (
            "On a {ward}, the average {outcome_phrase} for {population} has slipped "
            "about {drop}% over {period}, and the pattern has held across successive "
            "admissions. It emerged after {trigger_event}. The care team's leading "
            "theories are {naive_theories}. {surface_clue}. Routine labs and vitals "
            "read broadly within range. The {controllable_systems} are all "
            "adjustable within protocol. You have a limited number of experiments; "
            "you may measure quantities and apply available interventions, alone or "
            "in combination. Determine what is really driving the decline and what "
            "to do about it."
        ),
        "fills": {
            "ward": "post-surgical step-down unit", "population": "recovering patients",
            "outcome_phrase": "functional recovery score", "drop": "25", "period": "the past quarter",
            "trigger_event": "a change to the unit's standing medication and fluid protocol",
            "naive_theories": "that patients are dehydrated or under-oxygenated, so they favor more fluids and oxygen",
            "surface_clue": "Several patients have shown low-grade temperature readings on and off",
            "controllable_systems": "fluid and oxygen orders, medication list, and supportive care",
        },
    },
    "semiconductor": {
        "outcome": _mk("WaferYield", "wafer yield", "die yield", "functional yield", "yield"),
        "root_cause_pool": [
            _mk("MetalContamination", "metallic contamination", "trace-metal contamination", "metal ion contamination"),
            _mk("ParticleContamination", "particle contamination", "particulate defects"),
            _mk("PlasmaEtchDrift", "plasma etch drift", "etch process drift"),
        ],
        "mediator_pool": [
            _mk("GateOxideDefects", "gate-oxide defects", "oxide integrity loss"),
            _mk("JunctionLeakage", "junction leakage", "leakage paths"),
            _mk("InterconnectResistance", "interconnect resistance", "line resistance rise"),
            _mk("ThresholdShift", "threshold-voltage shift", "vt shift"),
            _mk("DielectricBreakdown", "dielectric breakdown", "insulation failure"),
        ],
        "proxy_pool": [
            _mk("DefectDensity", "defect density", "inline defect count", "defect map density"),
            _mk("LeakageCurrent", "leakage current", "off-state current"),
        ],
        "confounder_pool": [
            _mk("ChamberAge", "chamber age", "tool hours since service"),
            _mk("WaferLotSource", "wafer lot source", "substrate supplier"),
        ],
        "decoy_pool": [
            _mk("ChamberPressure", "chamber pressure", "process pressure"),
            _mk("StageTemperature", "stage temperature", "chuck temperature"),
        ],
        "source_knob_pool": [
            _mk("EtchPower", "etch power", "rf power", "plasma power"),
            _mk("DepositionRate", "deposition rate", "film growth rate"),
        ],
        "fix_actuator_pool": [
            _mk("PrecleanStep", "preclean step", "wet clean", "contaminant removal clean"),
            _mk("GetteringAnneal", "gettering anneal", "impurity gettering"),
            _mk("FilterUpgrade", "filtration upgrade", "particle filter swap"),
        ],
        "trap_actuator_pool": [
            _mk("TestBinReclassify", "test-bin reclassification", "yield rebinning"),
            _mk("ProbeCardRecal", "probe-card recalibration", "tester recalibration"),
        ],
        "inert_var_pool": [
            _mk("SpinSpeed", "spin speed", "coater rpm"),
            _mk("BakeTemperature", "bake temperature", "soft-bake temp"),
            _mk("ExposureDose", "exposure dose", "litho dose"),
            _mk("GasFlowArgon", "argon flow", "carrier gas flow"),
            _mk("HumidityFab", "fab humidity", "cleanroom humidity"),
            _mk("VibrationLevel", "vibration level", "tool vibration"),
            _mk("SlurryConcentration", "slurry concentration", "cmp slurry"),
            _mk("PadConditioning", "pad conditioning", "cmp pad wear"),
            _mk("RinseTime", "rinse time", "dhf rinse"),
            _mk("HandlerThroughput", "handler throughput", "tool cadence"),
            _mk("ChamberFlowUniformity", "chamber flow uniformity", "gas distribution"),
            _mk("BacksideCooling", "backside cooling", "helium cooling"),
            _mk("ScannerFocus", "scanner focus", "focus offset"),
        ],
        "scenario": (
            "A high-volume {fab} making {product} has seen {outcome_phrase} drop "
            "about {drop}% over {period}, holding low across many lots. The dip "
            "began after {trigger_event}. Process engineers' leading theories are "
            "{naive_theories}. {surface_clue}. Standard tool logs read broadly "
            "within spec. The {controllable_systems} are all adjustable. You have a "
            "limited number of experiments; you may measure quantities and apply "
            "available process controls, alone or in combination. Determine what is "
            "really driving the loss and what to do about it."
        ),
        "fills": {
            "fab": "300 mm logic fab", "product": "a mature logic device",
            "outcome_phrase": "functional die yield", "drop": "20", "period": "the past six weeks",
            "trigger_event": "a scheduled preventive-maintenance window on the etch and deposition tools",
            "naive_theories": "a pressure or temperature setpoint drift, so they favor retuning those setpoints",
            "surface_clue": "Inline inspection has flagged more defects than usual on a mid-line layer",
            "controllable_systems": "etch, deposition, clean, and filtration modules",
        },
    },
    "aquaculture": {
        "outcome": _mk("HarvestBiomass", "harvest biomass", "grow-out biomass", "final biomass", "biomass"),
        "root_cause_pool": [
            _mk("GillParasite", "gill parasite load", "ectoparasite burden", "parasite infestation"),
            _mk("DissolvedToxicant", "dissolved toxicant", "waterborne toxin", "trace toxicant"),
            _mk("ChronicHypoxia", "chronic hypoxia", "recurring oxygen sag"),
        ],
        "mediator_pool": [
            _mk("GillFunction", "gill function", "respiratory surface health"),
            _mk("FeedConversion", "feed conversion", "feed efficiency"),
            _mk("StressResponse", "stress response", "cortisol axis activation"),
            _mk("GrowthHormoneAxis", "growth-hormone axis", "somatic growth signaling"),
            _mk("ImmuneCompetence", "immune competence", "disease resistance"),
        ],
        "proxy_pool": [
            _mk("GillHistologyScore", "gill histology score", "gill damage index"),
            _mk("PlasmaCortisol", "plasma cortisol", "stress hormone level"),
        ],
        "confounder_pool": [
            _mk("SourceWaterSeason", "source-water season", "intake seasonality"),
            _mk("StockGenetics", "stock genetics", "broodstock line"),
        ],
        "decoy_pool": [
            _mk("TankTurbidity", "tank turbidity", "water cloudiness"),
            _mk("SurfaceActivity", "surface activity", "swimming activity"),
        ],
        "source_knob_pool": [
            _mk("FeedRate", "feed rate", "feeding intensity", "ration size"),
            _mk("StockingDensity", "stocking density", "fish density"),
        ],
        "fix_actuator_pool": [
            _mk("AntiparasiticBath", "antiparasitic bath", "parasite treatment bath"),
            _mk("CarbonFiltration", "activated-carbon filtration", "toxicant adsorption filter"),
            _mk("Aeration", "aeration upgrade", "oxygen injection"),
        ],
        "trap_actuator_pool": [
            _mk("AppetiteStimulant", "appetite stimulant", "feed attractant"),
            _mk("GradingReclassify", "grading reclassification", "size-grading shift"),
        ],
        "inert_var_pool": [
            _mk("WaterpH", "water ph", "tank ph"),
            _mk("Salinity", "salinity", "salt concentration"),
            _mk("WaterTempAqua", "water temperature", "tank temperature"),
            _mk("AmmoniaLevel", "ammonia level", "total ammonia"),
            _mk("NitriteLevel", "nitrite level", "nitrite concentration"),
            _mk("FlowThroughRate", "flow-through rate", "water exchange"),
            _mk("PhotoperiodHours", "photoperiod", "light hours"),
            _mk("PelletSize", "pellet size", "feed particle size"),
            _mk("CalciumHardness", "calcium hardness", "water hardness"),
            _mk("TankDepth", "tank depth", "water column depth"),
            _mk("BiofilterLoad", "biofilter load", "filter biomass"),
            _mk("UVSterilizer", "uv sterilizer", "uv treatment"),
            _mk("FeedProteinPct", "feed protein", "protein content"),
        ],
        "scenario": (
            "A {farm} raising {species} has seen {outcome_phrase} at harvest fall "
            "about {drop}% over {period}, repeating across grow-out cohorts. The "
            "shortfall began after {trigger_event}. The farm team's leading "
            "theories are {naive_theories}. {surface_clue}. Routine water-quality "
            "panels read broadly within range. The {controllable_systems} are all "
            "adjustable, and treatment equipment is available. You have a limited "
            "number of experiments; you may measure quantities and apply available "
            "controls or treatments, alone or in combination. Determine what is "
            "really driving the loss and what to do about it."
        ),
        "fills": {
            "farm": "recirculating aquaculture facility", "species": "Atlantic salmon",
            "outcome_phrase": "harvest biomass", "drop": "30", "period": "the last two cohorts",
            "trigger_event": "a change in intake water after a neighboring discharge permit was revised",
            "naive_theories": "underfeeding or overcrowding, so they favor more feed and lower density",
            "surface_clue": "Fish have been seen flaring their gills and crowding the inflow",
            "controllable_systems": "feeding, stocking, filtration, and aeration systems",
        },
    },
    "battery": {
        "outcome": _mk("CapacityRetention", "capacity retention", "cycle-life retention", "capacity health"),
        "root_cause_pool": [
            _mk("MoistureIngress", "moisture ingress", "trace-water contamination", "humidity contamination"),
            _mk("ElectrolyteContaminant", "electrolyte contaminant", "trace-metal in electrolyte"),
            _mk("DendriteNucleation", "dendrite nucleation", "lithium dendrite seeding"),
        ],
        "mediator_pool": [
            _mk("SEIInstability", "sei instability", "passivation-layer breakdown"),
            _mk("LithiumPlating", "lithium plating", "metallic-lithium deposition"),
            _mk("ImpedanceGrowth", "impedance growth", "internal resistance rise"),
            _mk("ActiveMaterialLoss", "active-material loss", "capacity fade mechanism"),
            _mk("GasEvolution", "gas evolution", "cell gassing"),
        ],
        "proxy_pool": [
            _mk("CoulombicEfficiency", "coulombic efficiency", "charge efficiency"),
            _mk("ImpedanceSpectrum", "impedance spectrum", "eis resistance"),
        ],
        "confounder_pool": [
            _mk("CellVintage", "cell vintage", "build date"),
            _mk("SeparatorSupplier", "separator supplier", "component lot"),
        ],
        "decoy_pool": [
            _mk("SurfaceTemperature", "surface temperature", "cell skin temp"),
            _mk("PackVoltage", "pack voltage", "terminal voltage"),
        ],
        "source_knob_pool": [
            _mk("ChargeRate", "charge rate", "charging c-rate", "fast-charge current"),
            _mk("FormationCurrent", "formation current", "initial cycling current"),
        ],
        "fix_actuator_pool": [
            _mk("DryRoomControl", "dry-room control", "desiccant drying", "moisture control"),
            _mk("ElectrolyteAdditive", "electrolyte additive", "film-forming additive"),
            _mk("SlowFormation", "slow-formation protocol", "gentle formation cycling"),
        ],
        "trap_actuator_pool": [
            _mk("CapacityRecal", "capacity recalibration", "gauge recalibration"),
            _mk("VoltageWindowShift", "voltage-window shift", "test-window adjustment"),
        ],
        "inert_var_pool": [
            _mk("CoatingThickness", "coating thickness", "electrode loading"),
            _mk("CalendarAge", "calendar age", "shelf time"),
            _mk("PressStackForce", "stack pressure", "cell compression"),
            _mk("TabWeldQuality", "tab-weld quality", "current-collector weld"),
            _mk("AmbientHumidityBatt", "ambient humidity", "room humidity"),
            _mk("DischargeDepth", "depth of discharge", "dod window"),
            _mk("RestPeriod", "rest period", "relaxation time"),
            _mk("ElectrolyteFillVolume", "electrolyte fill volume", "wetting amount"),
            _mk("CoolingPlateTemp", "cooling-plate temperature", "thermal-plate temp"),
            _mk("CrimpPressure", "crimp pressure", "seal force"),
            _mk("AnodeExcessRatio", "anode excess ratio", "n/p ratio"),
            _mk("CyclingTemperature", "cycling temperature", "test-chamber temp"),
            _mk("VentPressure", "vent pressure", "burst-disc setpoint"),
        ],
        "scenario": (
            "A {plant} producing {product} has seen {outcome_phrase} degrade about "
            "{drop}% faster than spec over {period}, consistently across production "
            "lots. The problem appeared after {trigger_event}. Cell engineers' "
            "leading theories are {naive_theories}. {surface_clue}. Standard "
            "line data read broadly within spec. The {controllable_systems} are all "
            "adjustable. You have a limited number of experiments; you may measure "
            "quantities and apply available process controls, alone or in "
            "combination. Determine what is really driving the accelerated fade and "
            "what to do about it."
        ),
        "fills": {
            "plant": "lithium-ion cell plant", "product": "automotive pouch cells",
            "outcome_phrase": "capacity retention", "drop": "25", "period": "the past two months",
            "trigger_event": "a facilities change to the dry-room HVAC and an electrolyte-supplier switch",
            "naive_theories": "that charging is too aggressive or cells run too hot, so they favor slower charging and more cooling",
            "surface_clue": "Early-life coulombic-efficiency readings have looked slightly low",
            "controllable_systems": "charging profile, formation, environmental, and additive systems",
        },
    },
    "catalysis": {
        "outcome": _mk("ConversionEfficiency", "conversion efficiency", "reactor conversion", "process conversion"),
        "root_cause_pool": [
            _mk("CatalystPoisoning", "catalyst poisoning", "active-site poisoning", "poison adsorption"),
            _mk("HotSpotFormation", "hot-spot formation", "thermal runaway zones"),
            _mk("FeedstockImpurity", "feedstock impurity", "trace-impurity slip"),
        ],
        "mediator_pool": [
            _mk("ActiveSiteBlocking", "active-site blocking", "site coverage loss"),
            _mk("CatalystSintering", "catalyst sintering", "metal-particle coarsening"),
            _mk("SelectivityLoss", "selectivity loss", "byproduct routing"),
            _mk("MassTransferLimit", "mass-transfer limitation", "diffusion limitation"),
            _mk("CokeDeposition", "coke deposition", "carbon fouling"),
        ],
        "proxy_pool": [
            _mk("CatalystSurfaceArea", "catalyst surface area", "active-site density"),
            _mk("EffluentByproduct", "effluent byproduct", "byproduct fraction"),
        ],
        "confounder_pool": [
            _mk("CatalystBatchAge", "catalyst batch age", "charge lifetime"),
            _mk("AmbientLoad", "ambient plant load", "upstream throughput"),
        ],
        "decoy_pool": [
            _mk("ReactorPressure", "reactor pressure", "system pressure"),
            _mk("JacketTemperature", "jacket temperature", "cooling-jacket temp"),
        ],
        "source_knob_pool": [
            _mk("FeedFlowRate", "feed flow rate", "feed throughput", "space velocity"),
            _mk("InletTemperature", "inlet temperature", "feed preheat"),
        ],
        "fix_actuator_pool": [
            _mk("GuardBedAdsorbent", "guard-bed adsorbent", "poison-scavenger bed", "sacrificial adsorbent"),
            _mk("RegenerationCycle", "regeneration cycle", "in-situ catalyst regeneration"),
            _mk("FeedPurification", "feed purification", "impurity polishing"),
        ],
        "trap_actuator_pool": [
            _mk("AnalyzerSpanAdjust", "analyzer span adjustment", "gc recalibration"),
            _mk("ReportingBasisShift", "reporting-basis shift", "conversion-basis change"),
        ],
        "inert_var_pool": [
            _mk("RecycleRatio", "recycle ratio", "recycle fraction"),
            _mk("CoolantFlow", "coolant flow", "utility flow"),
            _mk("BedHeight", "bed height", "catalyst-bed depth"),
            _mk("ParticleSizeCat", "catalyst particle size", "pellet size"),
            _mk("InertDiluent", "inert diluent", "diluent fraction"),
            _mk("PurgeRate", "purge rate", "vent rate"),
            _mk("SteamToCarbon", "steam-to-carbon ratio", "steam ratio"),
            _mk("RecycleGasPurity", "recycle-gas purity", "loop purity"),
            _mk("PreheaterDuty", "preheater duty", "heat input"),
            _mk("PressureDropBed", "bed pressure drop", "delta-p"),
            _mk("FeedMoisture", "feed moisture", "water content"),
            _mk("AgitationCat", "agitation rate", "mixing intensity"),
            _mk("QuenchFlow", "quench flow", "quench injection"),
        ],
        "scenario": (
            "A {plant} running {process} has seen {outcome_phrase} fall about "
            "{drop}% over {period}, staying low run after run. The decline followed "
            "{trigger_event}. The process team's leading theories are "
            "{naive_theories}. {surface_clue}. Routine unit data read broadly "
            "within range. The {controllable_systems} are all adjustable, and "
            "polishing and regeneration options are available. You have a limited "
            "number of experiments; you may measure quantities and apply available "
            "controls or treatments, alone or in combination. Determine what is "
            "really driving the loss and what to do about it."
        ),
        "fills": {
            "plant": "continuous catalytic reactor unit", "process": "a fixed-bed hydrogenation",
            "outcome_phrase": "reactor conversion", "drop": "20", "period": "the past month",
            "trigger_event": "a switch to a new feedstock supplier after a routine catalyst reload",
            "naive_theories": "that the reactor is running too cool or at the wrong pressure, so they favor retuning temperature and pressure",
            "surface_clue": "Effluent analysis has shown a slightly elevated byproduct fraction",
            "controllable_systems": "feed, temperature, purification, and regeneration systems",
        },
    },
    "fermentation": {
        "outcome": _mk("EthanolYield", "ethanol yield", "fermentation yield", "product yield", "yield"),
        "root_cause_pool": [
            _mk("WildYeastContamination", "wild-yeast contamination", "contaminant strain", "off-strain load"),
            _mk("OrganicAcidInhibition", "organic-acid inhibition", "acetic-acid stress", "weak-acid inhibition"),
            _mk("TraceMetalDeficiency", "trace-metal deficiency", "micronutrient shortfall"),
        ],
        "mediator_pool": [
            _mk("YeastViability", "yeast viability", "cell vitality"),
            _mk("GlycolyticFlux", "glycolytic flux", "sugar-uptake flux"),
            _mk("MembraneIntegrity", "membrane integrity", "cell-membrane health"),
            _mk("RedoxBalance", "redox balance", "nad ratio"),
            _mk("BuddingRate", "budding rate", "proliferation rate"),
        ],
        "proxy_pool": [
            _mk("ViabilityStain", "viability stain", "dead-cell fraction", "methylene-blue count"),
            _mk("OrganicAcidTiter", "organic-acid titer", "acetate concentration"),
        ],
        "confounder_pool": [
            _mk("PitchBatch", "pitch batch", "yeast pitch lot"),
            _mk("FermenterHistory", "fermenter history", "vessel service age"),
        ],
        "decoy_pool": [
            _mk("BrothFoaming", "broth foaming", "foam level"),
            _mk("CO2OffgasRate", "co2 offgas rate", "gas-evolution rate"),
        ],
        "source_knob_pool": [
            _mk("SugarFeedRate", "sugar feed rate", "substrate feed", "wort feed rate"),
            _mk("FermentTemp", "fermentation temperature", "brew temperature"),
        ],
        "fix_actuator_pool": [
            _mk("AntimicrobialHopExtract", "antimicrobial hop extract", "contaminant-selective inhibitor"),
            _mk("pHBuffering", "ph buffering", "acid neutralization"),
            _mk("MicronutrientSpike", "micronutrient spike", "trace-metal supplement"),
        ],
        "trap_actuator_pool": [
            _mk("Antifoam", "antifoam addition", "defoamer dosing"),
            _mk("RefractometerRecal", "refractometer recalibration", "gravity-meter recalibration"),
        ],
        "inert_var_pool": [
            _mk("DissolvedO2Ferm", "dissolved oxygen", "aeration level"),
            _mk("AgitationFerm", "agitation rate", "stir speed"),
            _mk("VesselPressureFerm", "vessel pressure", "headspace pressure"),
            _mk("NitrogenSource", "nitrogen source", "yan level"),
            _mk("InoculationVolume", "inoculation volume", "pitch volume"),
            _mk("GravityInitial", "original gravity", "starting sugar"),
            _mk("MashTemp", "mash temperature", "conversion temp"),
            _mk("WaterHardnessFerm", "water hardness", "brewing-liquor minerals"),
            _mk("YeastGeneration", "yeast generation", "repitch count"),
            _mk("TankGeometry", "tank geometry", "vessel aspect ratio"),
            _mk("CarbonationLevel", "carbonation level", "dissolved co2"),
            _mk("ClarifierDose", "clarifier dose", "fining agent"),
            _mk("ChillRate", "chill rate", "crash-cool rate"),
        ],
        "scenario": (
            "A {plant} running {process} has seen {outcome_phrase} fall about "
            "{drop}% over {period}, repeating batch after batch. The drop began "
            "after {trigger_event}. The production team's leading theories are "
            "{naive_theories}. {surface_clue}. Routine process panels read broadly "
            "within range. The {controllable_systems} are all adjustable, and "
            "dosing and treatment options are available. You have a limited number "
            "of experiments; you may measure quantities and apply available "
            "controls or additives, alone or in combination. Determine what is "
            "really driving the loss and what to do about it."
        ),
        "fills": {
            "plant": "industrial fermentation plant", "process": "a batch ethanol fermentation",
            "outcome_phrase": "fermentation yield", "drop": "25", "period": "the past three weeks",
            "trigger_event": "a cleaning-protocol change and a switch to a new yeast-propagation batch",
            "naive_theories": "that fermentation runs too cool or is underfed, so they favor warmer temperatures and more sugar feed",
            "surface_clue": "Batches have smelled faintly sour at the end of fermentation",
            "controllable_systems": "feed, temperature, dosing, and treatment systems",
        },
    },
}


def skin_names() -> List[str]:
    return list(SKINS)
