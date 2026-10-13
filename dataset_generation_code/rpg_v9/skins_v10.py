#!/usr/bin/env python3
"""Extra name pools for v10 worlds (sampler version="v10"), kept OUT of skins.py so v9 / v9.1 worlds are untouched.

v10 worlds have 22-28 controls against a 15-experiment budget (personal_docs plans/V10_WORLDGEN_PLAN_2026-10-12.md),
so each skin needs many more plausible, role-neutral names than skins.py provides:
  extra_process_pool  : measurable process variables that get a "set the level" control. Their role in a world
                        (inert / small harm / ...) is assigned by the sampler; nothing in the name may reveal it.
  extra_additive_pool : additives / treatments / interventions that get a "dose" control acting on a hidden node
                        (inert / symptomatic relief / small harm). Names must read as plausible remedies, so a remedy
                        name does not single out the true fix.
Every name is unique within its skin and collides with no name in skins.py (checked by test_skins_v10()).
"""
from __future__ import annotations

from typing import Dict, List


def _p(name: str, *aliases: str) -> Dict[str, object]:
    return {"name": name, "aliases": list(aliases)}


EXTRA: Dict[str, Dict[str, List[Dict[str, object]]]] = {
    "bioprocess": {
        "extra_process_pool": [
            _p("PermeateFlux", "permeate flux", "filtrate draw"), _p("BleedRate", "bleed rate", "cell bleed"),
            _p("HarvestPumpRate", "harvest pump rate", "clarification flow"), _p("ImpellerTipSpeed", "impeller tip speed", "tip speed"),
            _p("BaseAdditionRate", "base addition rate", "alkali addition"), _p("TraceElementFeed", "trace element feed", "trace feed"),
            _p("VitaminFeed", "vitamin feed", "vitamin supplement rate"), _p("CopperSulfateFeed", "copper sulfate feed", "copper feed"),
            _p("JacketFlowRate", "jacket flow rate", "jacket coolant flow"), _p("InoculumDensity", "inoculum density", "seeding density"),
            _p("FeedBolusSize", "feed bolus size", "bolus volume"), _p("OverlayAirFlow", "overlay air flow", "headspace sweep"),
            _p("PoloxamerLevel", "poloxamer level", "shear protectant level"), _p("HydrolysateFeed", "hydrolysate feed", "peptone feed"),
            _p("PhosphateFeed", "phosphate feed", "phosphate addition"), _p("TemperatureShiftDay", "temperature shift day", "shift timing"),
            _p("CellRetentionRate", "cell retention rate", "retention device setting"), _p("SamplingFrequency", "sampling frequency", "sample draw rate"),
        ],
        "extra_additive_pool": [
            _p("ButyrateSpike", "butyrate spike", "productivity enhancer"), _p("ZincSupplement", "zinc supplement", "zinc addition"),
            _p("LipidConcentrate", "lipid concentrate", "lipid supplement"), _p("InsulinSupplement", "insulin supplement", "growth factor addition"),
            _p("CysteineSpike", "cysteine spike", "thiol supplement"), _p("ManganeseSpike", "manganese spike", "glycosylation additive"),
            _p("DextranSulfate", "dextran sulfate", "anti-clumping agent"), _p("HypothermiaProtectant", "cold-shift protectant", "thermal protectant"),
            _p("UridineFeed", "uridine feed", "nucleoside supplement"), _p("GalactoseSpike", "galactose spike", "sugar supplement"),
            _p("SeleniumSupplement", "selenium supplement", "selenite addition"), _p("ProteaseInhibitorCocktail", "protease inhibitor cocktail", "protease inhibitor"),
        ],
    },
    "datacenter": {
        "extra_process_pool": [
            _p("ChilledWaterSetpoint", "chilled water setpoint", "chiller setpoint"), _p("HotAisleContainment", "hot aisle containment", "containment level"),
            _p("RaisedFloorPressure", "raised floor pressure", "underfloor pressure"), _p("PowerCapLimit", "power cap", "node power limit"),
            _p("SchedulerBackfill", "scheduler backfill", "backfill aggressiveness"), _p("NetworkMtu", "network MTU", "frame size"),
            _p("TcpWindowScale", "TCP window size", "receive window"), _p("StorageReplication", "storage replication factor", "replica count"),
            _p("CompressionLevel", "compression level", "data compression"), _p("FirmwareRevision", "firmware revision", "firmware level"),
            _p("LinkAggregation", "link aggregation", "bonded links"), _p("RebalanceRate", "rebalance rate", "data rebalancing"),
            _p("SwapPolicy", "swap policy", "swappiness"), _p("NumaPinning", "NUMA pinning", "core affinity"),
            _p("PrefetchDepth", "prefetch depth", "read-ahead"), _p("LogVerbosity", "log verbosity", "logging level"),
            _p("HeartbeatInterval", "heartbeat interval", "health-check interval"), _p("CheckpointInterval", "checkpoint interval", "job checkpointing"),
        ],
        "extra_additive_pool": [
            _p("KernelPatch", "kernel patch", "kernel update"), _p("DriverRollback", "driver rollback", "driver downgrade"),
            _p("CableSwap", "cable swap", "patch-cable replacement"), _p("ThermalPasteRework", "thermal paste rework", "heatsink reseat"),
            _p("MemoryScrub", "memory scrub", "ECC scrubbing"), _p("QueueRebalance", "queue rebalance", "job queue reshuffle"),
            _p("FilterReplacement", "air filter replacement", "filter change"), _p("BmcReset", "BMC reset", "management controller reset"),
            _p("SpanningTreeTune", "spanning-tree tuning", "switch topology tune"), _p("OpticsClean", "optics cleaning", "fiber end-face clean"),
            _p("ConfigRollback", "config rollback", "configuration revert"), _p("BurstCredits", "burst credits", "I/O burst allowance"),
        ],
    },
    "watertreatment": {
        "extra_process_pool": [
            _p("CoagulantDose", "coagulant dose", "alum feed"), _p("SettlingTime", "settling time", "clarifier detention"),
            _p("ReservoirDraw", "reservoir draw", "storage drawdown"), _p("BoosterPumpSpeed", "booster pump speed", "booster setting"),
            _p("ValveExercising", "valve exercising", "valve turn program"), _p("DeadEndBleed", "dead-end bleed", "hydrant bleed"),
            _p("ChloramineRatio", "chlorine-ammonia ratio", "chloramine ratio"), _p("SludgeBlanket", "sludge blanket level", "blanket depth"),
            _p("RawWaterIntakeDepth", "intake depth", "raw water draw depth"), _p("RechlorinationDose", "booster chlorination", "rechlorination"),
            _p("OzoneDose", "ozone dose", "pre-ozonation"), _p("MixingEnergy", "rapid-mix energy", "mixing intensity"),
            _p("FilterRunTime", "filter run time", "run length"), _p("CarbonContactTime", "carbon contact time", "GAC contact"),
            _p("StorageTurnover", "storage turnover", "tank cycling"), _p("NightFlowRate", "night flow rate", "minimum night flow"),
            _p("SampleTapFlush", "sample tap flush", "sampling flush"), _p("DistributionPressureZone", "pressure zone setpoint", "zone pressure"),
        ],
        "extra_additive_pool": [
            _p("PermanganateFeed", "permanganate feed", "pre-oxidant"), _p("PowderedCarbon", "powdered activated carbon", "PAC dose"),
            _p("SodaAsh", "soda ash", "carbonate feed"), _p("CarbonDioxideFeed", "carbon dioxide feed", "recarbonation"),
            _p("Polyphosphate", "polyphosphate", "sequestering phosphate"), _p("SodiumSilicate", "sodium silicate", "silicate inhibitor"),
            _p("FilterAid", "filter aid polymer", "filter conditioner"), _p("Dechlorinator", "dechlorination agent", "bisulfite feed"),
            _p("BiocideShock", "biocide shock", "shock disinfection"), _p("AmmoniaFeed", "ammonia feed", "chloramination feed"),
            _p("LimeSlurry", "lime slurry", "lime addition"), _p("IonExchangeRegen", "ion-exchange regeneration", "resin regeneration"),
        ],
    },
    "agronomy": {
        "extra_process_pool": [
            _p("VentOpening", "vent opening", "roof vents"), _p("ShadeScreen", "shade screen", "shading"),
            _p("FoggingRate", "fogging rate", "misting"), _p("SupplementalLighting", "supplemental lighting", "grow lights"),
            _p("HeatingPipeTemp", "heating pipe temperature", "pipe rail heat"), _p("SubstrateEC", "substrate EC", "feed conductivity"),
            _p("DrainPercentage", "drain percentage", "leachate fraction"), _p("DripperFlow", "dripper flow", "emitter output"),
            _p("IrrigationFrequency", "irrigation frequency", "watering pulses"), _p("StemDensity", "stem density", "heads per square metre"),
            _p("LeafPruning", "leaf removal", "deleafing"), _p("TrussPruning", "truss pruning", "fruit thinning"),
            _p("CalciumNitrateFeed", "calcium nitrate", "calcium feed"), _p("MagnesiumFeed", "magnesium feed", "Epsom salt feed"),
            _p("BumblebeeHives", "bumblebee hives", "hive count"), _p("DayNightDifferential", "day-night difference", "DIF setting"),
            _p("RootZoneTemp", "root zone temperature", "substrate heating"), _p("AirCirculationFans", "circulation fans", "horizontal air flow"),
        ],
        "extra_additive_pool": [
            _p("SeaweedExtract", "seaweed extract", "kelp biostimulant"), _p("HumicAcid", "humic acid", "humate drench"),
            _p("SiliconSupplement", "silicon supplement", "potassium silicate"), _p("MycorrhizalInoculant", "mycorrhizal inoculant", "root inoculant"),
            _p("AminoAcidFoliar", "amino acid foliar", "foliar amino spray"), _p("BoronFoliar", "boron foliar", "boron spray"),
            _p("WettingAgentAg", "soil wetting agent", "surfactant drench"), _p("SulfurDust", "sulfur dust", "sulfur application"),
            _p("NeemOil", "neem oil", "botanical oil spray"), _p("CopperFungicide", "copper fungicide", "copper spray"),
            _p("SalicylicAcidSpray", "salicylic acid spray", "resistance inducer"), _p("MolassesDrench", "molasses drench", "carbon drench"),
        ],
    },
    "clinical": {
        "extra_process_pool": [
            _p("HeadOfBedAngle", "head-of-bed angle", "bed elevation"), _p("TurningFrequency", "turning frequency", "repositioning schedule"),
            _p("IncentiveSpirometry", "incentive spirometry", "breathing exercises"), _p("OralFluidTarget", "oral fluid target", "drink target"),
            _p("ProteinSupplementRate", "protein supplement", "nutrition supplement"), _p("CompressionStockings", "compression stockings", "leg compression"),
            _p("LightingSchedule", "lighting schedule", "day-night lighting"), _p("NurseCheckFrequency", "nursing check frequency", "observation frequency"),
            _p("VitalSignsInterval", "vital signs interval", "obs interval"), _p("WoundDressingChange", "dressing change frequency", "wound care"),
            _p("CatheterDays", "catheter duration", "line days"), _p("FootPumpUse", "foot pump", "sequential compression"),
            _p("OxygenWeanRate", "oxygen wean rate", "oxygen taper"), _p("FluidBalanceTarget", "fluid balance target", "net balance goal"),
            _p("MealTiming", "meal timing", "feeding schedule"), _p("SittingOutTime", "time out of bed", "chair time"),
            _p("RoomTemperature", "room temperature", "ward temperature"), _p("EducationSessions", "patient education", "teaching sessions"),
        ],
        "extra_additive_pool": [
            _p("ProtonPumpInhibitor", "proton pump inhibitor", "acid suppression"), _p("Anticoagulant", "anticoagulant prophylaxis", "clot prevention"),
            _p("Laxative", "laxative", "bowel regimen"), _p("Antiemetic", "antiemetic", "anti-nausea medication"),
            _p("VitaminD", "vitamin D", "cholecalciferol"), _p("IronSupplement", "iron supplement", "oral iron"),
            _p("Probiotic", "probiotic", "gut flora supplement"), _p("ZincTherapy", "zinc therapy", "zinc tablets"),
            _p("Melatonin", "melatonin", "sleep aid"), _p("Bronchodilator", "bronchodilator", "inhaler"),
            _p("Diuretic", "diuretic", "water tablet"), _p("Corticosteroid", "corticosteroid", "steroid course"),
        ],
    },
    "semiconductor": {
        "extra_process_pool": [
            _p("ChamberSeasoning", "chamber seasoning", "seasoning wafers"), _p("WaferCoolDelay", "cool-down delay", "wafer cool time"),
            _p("ResistThickness", "resist thickness", "photoresist coat"), _p("DevelopTime", "develop time", "resist develop"),
            _p("HmdsPrime", "HMDS prime", "adhesion prime"), _p("EndpointOverEtch", "over-etch", "endpoint over-etch"),
            _p("RfMatchSetting", "RF match setting", "matching network"), _p("ElectrostaticChuckVoltage", "chuck voltage", "ESC clamp"),
            _p("PurgeGasFlow", "purge gas flow", "nitrogen purge"), _p("LoadLockPumpTime", "load-lock pump time", "pumpdown time"),
            _p("ImplantTilt", "implant tilt", "tilt angle"), _p("AnnealRampRate", "anneal ramp rate", "ramp rate"),
            _p("CmpDownforce", "CMP downforce", "polish pressure"), _p("PlatenSpeed", "platen speed", "polish speed"),
            _p("DiWaterResistivity", "DI water resistivity", "rinse water quality"), _p("StockerDwell", "stocker dwell time", "queue time"),
            _p("OverlayCorrection", "overlay correction", "alignment offset"), _p("MetrologySampling", "metrology sampling rate", "inspection sampling"),
        ],
        "extra_additive_pool": [
            _p("ChelatingRinse", "chelating rinse", "metal-scavenging rinse"), _p("HfDip", "HF dip", "oxide strip"),
            _p("OzoneWaterClean", "ozonated water clean", "ozone rinse"), _p("ScClean1", "SC-1 clean", "ammonia peroxide clean"),
            _p("PlasmaAsh", "plasma ash", "resist strip"), _p("ForminggasAnneal", "forming-gas anneal", "hydrogen anneal"),
            _p("SurfactantRinse", "surfactant rinse", "wetting rinse"), _p("ChamberWetClean", "chamber wet clean", "preventive clean"),
            _p("DummyWaferRun", "dummy wafer run", "conditioning wafers"), _p("AntistaticIonizer", "ionizer bar", "static neutralizer"),
            _p("CorrosionInhibitorBath", "corrosion inhibitor bath", "BTA rinse"), _p("PostCmpBrush", "post-CMP brush", "brush scrub"),
        ],
    },
    "aquaculture": {
        "extra_process_pool": [
            _p("OxygenConeFlow", "oxygen cone flow", "oxygenation"), _p("DegassingRate", "degassing rate", "CO2 stripping"),
            _p("FeedingFrequency", "feeding frequency", "meals per day"), _p("FeedSpread", "feed spread", "feeder distribution"),
            _p("LightIntensity", "light intensity", "tank lighting"), _p("CurrentSpeed", "water current", "swimming current"),
            _p("GradingFrequency", "grading frequency", "size grading"), _p("MakeupWaterRate", "make-up water", "new water addition"),
            _p("SludgeRemoval", "sludge removal", "drum filter backwash"), _p("ProteinSkimmer", "protein skimmer", "foam fractionation"),
            _p("OzoneInjection", "ozone injection", "ozonation"), _p("AlkalinityDosing", "alkalinity dosing", "bicarbonate dosing"),
            _p("TankCleaningInterval", "tank cleaning interval", "wall scrubbing"), _p("NetChangeInterval", "net change interval", "screen change"),
            _p("HandlingEvents", "handling events", "crowding events"), _p("SmoltTransferSize", "transfer size", "smolt size at transfer"),
            _p("WaterExchangeTurnover", "tank turnover", "exchange rate"), _p("NoiseLevelTank", "tank noise", "pump vibration"),
        ],
        "extra_additive_pool": [
            _p("SaltBath", "salt bath", "freshwater-salt dip"), _p("HydrogenPeroxideBath", "hydrogen peroxide bath", "peroxide treatment"),
            _p("VitaminCSupplement", "vitamin C supplement", "ascorbate feed"), _p("Immunostimulant", "immunostimulant", "beta-glucan feed"),
            _p("ProbioticFeed", "probiotic feed", "gut probiotic"), _p("AstaxanthinFeed", "astaxanthin feed", "pigment feed"),
            _p("FormalinBath", "formalin bath", "formalin treatment"), _p("NucleotideFeed", "nucleotide feed", "nucleotide supplement"),
            _p("ChelatorDose", "water conditioner", "metal binder"), _p("EnzymeAdditive", "digestive enzyme additive", "feed enzyme"),
            _p("OmegaOilTopdress", "fish-oil top dressing", "omega-3 topdress"), _p("BacteriophageTreatment", "bacteriophage treatment", "phage therapy"),
        ],
    },
    "battery": {
        "extra_process_pool": [
            _p("MixingTime", "slurry mixing time", "mix time"), _p("SlurryViscosity", "slurry viscosity", "slurry rheology"),
            _p("DryingOvenTemp", "drying oven temperature", "electrode drying"), _p("CalenderGap", "calender gap", "roll gap"),
            _p("NotchingSpeed", "notching speed", "die cut speed"), _p("StackAlignment", "stack alignment", "electrode registration"),
            _p("DegassingVacuum", "degassing vacuum", "vacuum level"), _p("SealTemperature", "seal temperature", "pouch seal temp"),
            _p("AgingTemperature", "aging temperature", "soak temperature"), _p("AgingDuration", "aging duration", "soak time"),
            _p("FormationTemperature", "formation temperature", "formation chamber temp"), _p("UpperCutoffVoltage", "upper cutoff voltage", "charge cutoff"),
            _p("BinderContent", "binder content", "PVDF fraction"), _p("ConductiveCarbon", "conductive carbon", "carbon black fraction"),
            _p("ElectrodePorosity", "electrode porosity", "porosity target"), _p("CurrentCollectorThickness", "foil thickness", "collector foil"),
            _p("TapeApplication", "insulation tape", "edge tape"), _p("BurrInspection", "burr inspection rate", "edge inspection"),
        ],
        "extra_additive_pool": [
            _p("VinyleneCarbonate", "vinylene carbonate", "VC additive"), _p("FluoroethyleneCarbonate", "FEC additive", "fluorinated additive"),
            _p("LiPO2F2", "difluorophosphate salt", "LiPO2F2"), _p("SultoneAdditive", "sultone additive", "propane sultone"),
            _p("ScavengerAdditive", "acid scavenger", "HF scavenger"), _p("BoronAdditive", "borate additive", "LiBOB"),
            _p("SurfaceCoatingAlumina", "alumina coating", "cathode coating"), _p("PrelithiationAgent", "prelithiation agent", "lithium compensation"),
            _p("FlameRetardant", "flame retardant", "phosphate retardant"), _p("RedoxShuttle", "redox shuttle", "overcharge additive"),
            _p("NitrileAdditive", "nitrile additive", "succinonitrile"), _p("SilaneTreatment", "silane treatment", "separator coating"),
        ],
    },
    "catalysis": {
        "extra_process_pool": [
            _p("HydrogenPartialPressure", "hydrogen partial pressure", "H2 pressure"), _p("MakeupHydrogenRate", "makeup hydrogen", "fresh hydrogen rate"),
            _p("SulfidingLevel", "sulfiding level", "presulfiding"), _p("QuenchGasRate", "quench gas rate", "interbed quench"),
            _p("FlareLoad", "flare load", "relief flow"), _p("ReboilerDuty", "reboiler duty", "column heat"),
            _p("RefluxRatio", "reflux ratio", "column reflux"), _p("SeparatorLevel", "separator level", "drum level"),
            _p("WashWaterRate", "wash water rate", "water wash"), _p("AmineCirculation", "amine circulation", "scrubber circulation"),
            _p("CompressorSpeed", "compressor speed", "recycle compressor"), _p("FeedPreheatBypass", "preheat bypass", "exchanger bypass"),
            _p("StripperSteam", "stripper steam", "stripping steam"), _p("ProductDrawRate", "product draw", "draw rate"),
            _p("BedTemperatureSpread", "bed temperature spread", "radial spread"), _p("FlushOilRate", "flush oil rate", "flushing oil"),
            _p("CatalystLoadingDensity", "loading density", "dense loading"), _p("PressureControlValve", "back-pressure setpoint", "pressure valve"),
        ],
        "extra_additive_pool": [
            _p("DmdsInjection", "DMDS injection", "sulfiding agent"), _p("AntifoulantDose", "antifoulant", "fouling inhibitor"),
            _p("CorrosionInhibitorCat", "corrosion inhibitor", "filming amine"), _p("ChlorideGuard", "chloride guard", "chloride trap"),
            _p("AntioxidantInhibitor", "oxidation inhibitor", "gum inhibitor"), _p("DemulsifierDose", "demulsifier", "emulsion breaker"),
            _p("NeutralizerDose", "neutralizing amine", "overhead neutralizer"), _p("SiliconTrap", "silicon trap", "silica guard"),
            _p("ArsenicTrap", "arsenic trap", "arsenic guard"), _p("WaxDispersant", "wax dispersant", "paraffin inhibitor"),
            _p("MetalDeactivator", "metal deactivator", "metal passivator"), _p("OxygenScavenger", "oxygen scavenger", "deaerating agent"),
        ],
    },
    "fermentation": {
        "extra_process_pool": [
            _p("PitchRate", "pitch rate", "yeast pitch"), _p("AerationAtPitch", "aeration at pitch", "wort oxygenation"),
            _p("RecirculationRate", "recirculation rate", "pump-over"), _p("CoolingJacketFlow", "cooling jacket flow", "jacket chill"),
            _p("MashpH", "mash pH", "mash acidity"), _p("LiquefactionTemp", "liquefaction temperature", "cook temperature"),
            _p("BackSetRatio", "backset ratio", "stillage recycle"), _p("SolidsLoading", "solids loading", "dry solids"),
            _p("GlucoamylaseRate", "glucoamylase dose", "saccharifying enzyme"), _p("UreaFeed", "urea feed", "nitrogen feed"),
            _p("PropagationTime", "propagation time", "prop duration"), _p("PropagatorTemp", "propagator temperature", "prop temperature"),
            _p("FillTime", "fill time", "fermenter fill"), _p("DropTiming", "drop timing", "transfer timing"),
            _p("CipCausticStrength", "caustic strength", "CIP caustic"), _p("BeerWellLevel", "beer well level", "beer well"),
            _p("FlashCoolerTemp", "flash cooler temperature", "mash cooler"), _p("ScrubberWaterFlow", "scrubber water", "CO2 scrubber flow"),
        ],
        "extra_additive_pool": [
            _p("ZincSulfate", "zinc sulfate", "yeast zinc"), _p("YeastNutrientBlend", "yeast nutrient", "nutrient blend"),
            _p("Penicillin", "penicillin", "bacterial control antibiotic"), _p("ChlorineDioxide", "chlorine dioxide", "ClO2 dose"),
            _p("ProteaseEnzyme", "protease enzyme", "FAN enzyme"), _p("CellulaseEnzyme", "cellulase", "fiber enzyme"),
            _p("MagnesiumSulfate", "magnesium sulfate", "magnesium addition"), _p("VirginiamycinDose", "virginiamycin", "antibiotic dose"),
            _p("PhytaseEnzyme", "phytase", "phytate enzyme"), _p("ErgosterolSupplement", "ergosterol supplement", "sterol supplement"),
            _p("PotassiumMetabisulfite", "metabisulfite", "sulfite dose"), _p("HopAcidsBoost", "hop acids", "beta-acid dose"),
        ],
    },
}


def extra_pools(skin: str) -> Dict[str, List[Dict[str, object]]]:
    return EXTRA[skin]


# ---------------------------------------------------------------------------------------------------------------------
# Units (2026-10-12, v10 rebuild). The SCM stays on an internal 0-100 scale for every control; the agent sees and answers
# in physical units through a linear map (display = lo + internal/100 * (hi - lo)). Only names listed here can become a
# control in a v10 world: a quantity with no sensible unit / physical handle (heart rate, body temperature, pain score,
# "deprescribing", a firmware revision) is never offered as something to set.
#   LEVEL_UNITS[skin][name] = (unit, lo, hi)  -- a process setting that is always running at some level (measurable)
#   DOSE_UNITS[skin][name]  = (unit, hi)      -- something added / applied; 0 = not given (target is latent)
# ---------------------------------------------------------------------------------------------------------------------
LEVEL_UNITS: Dict[str, Dict[str, tuple]] = {
    "bioprocess": {
        "FeedWaterFlowRate": ("L/h", 0, 200), "MediaExchangeRate": ("vessel volumes/day", 0, 3),
        "PermeateFlux": ("L/m²/h", 0, 40), "BleedRate": ("%/day", 0, 30), "HarvestPumpRate": ("L/h", 0, 150),
        "ImpellerTipSpeed": ("m/s", 0, 3), "BaseAdditionRate": ("mL/h", 0, 500), "TraceElementFeed": ("mL/L/day", 0, 10),
        "VitaminFeed": ("mL/L/day", 0, 10), "CopperSulfateFeed": ("µM", 0, 20), "JacketFlowRate": ("L/min", 0, 120),
        "InoculumDensity": ("10⁶ cells/mL", 0, 2), "FeedBolusSize": ("% of working volume", 0, 10),
        "OverlayAirFlow": ("L/min", 0, 20), "PoloxamerLevel": ("g/L", 0, 3), "HydrolysateFeed": ("g/L/day", 0, 10),
        "PhosphateFeed": ("mM/day", 0, 5), "TemperatureShiftDay": ("culture day", 0, 14),
        "CellRetentionRate": ("% of cells retained", 0, 100), "SamplingFrequency": ("samples/day", 0, 8),
    },
    "datacenter": {
        "CoolingAggressiveness": ("% of cooling capacity", 0, 100), "ChilledWaterSetpoint": ("°C", 5, 15),
        "HotAisleContainment": ("% of aisle sealed", 0, 100), "RaisedFloorPressure": ("Pa", 0, 40),
        "PowerCapLimit": ("kW per rack", 5, 40), "SchedulerBackfill": ("jobs backfilled/h", 0, 200),
        "NetworkMtu": ("bytes", 1500, 9000), "TcpWindowScale": ("MB", 0, 64), "StorageReplication": ("replicas", 1, 5),
        "CompressionLevel": ("compression level", 0, 9), "LinkAggregation": ("bonded links", 1, 8),
        "RebalanceRate": ("MB/s", 0, 500), "SwapPolicy": ("swappiness", 0, 100), "NumaPinning": ("% of jobs pinned", 0, 100),
        "PrefetchDepth": ("KB", 0, 4096), "LogVerbosity": ("log level", 0, 5), "HeartbeatInterval": ("s", 1, 60),
        "CheckpointInterval": ("min", 5, 240),
    },
    "watertreatment": {
        "LineFlushRate": ("m³/h", 0, 120), "CoagulantDose": ("mg/L", 0, 60), "SettlingTime": ("h", 0.5, 6),
        "ReservoirDraw": ("ML/day", 0, 40), "BoosterPumpSpeed": ("rpm", 600, 3000), "ValveExercising": ("valves/week", 0, 80),
        "DeadEndBleed": ("m³/day", 0, 50), "ChloramineRatio": ("Cl₂:NH₃ mass ratio", 3, 6), "SludgeBlanket": ("m", 0, 3),
        "RawWaterIntakeDepth": ("m", 1, 15), "RechlorinationDose": ("mg/L", 0, 2), "OzoneDose": ("mg/L", 0, 5),
        "MixingEnergy": ("1/s velocity gradient", 100, 1000), "FilterRunTime": ("h", 12, 96),
        "CarbonContactTime": ("min", 0, 30), "StorageTurnover": ("%/day", 0, 100), "NightFlowRate": ("L/s", 0, 50),
        "SampleTapFlush": ("min", 0, 20), "DistributionPressureZone": ("kPa", 200, 700),
    },
    "agronomy": {
        "IrrigationRate": ("mm/day", 0, 12), "FertigationRate": ("L/m²/day", 0, 6), "VentOpening": ("% open", 0, 100),
        "ShadeScreen": ("% closed", 0, 100), "FoggingRate": ("g/m²/h", 0, 200), "SupplementalLighting": ("µmol/m²/s", 0, 250),
        "HeatingPipeTemp": ("°C", 20, 80), "SubstrateEC": ("mS/cm", 1, 6), "DrainPercentage": ("%", 0, 50),
        "DripperFlow": ("L/h", 0.5, 4), "IrrigationFrequency": ("pulses/day", 0, 30), "StemDensity": ("stems/m²", 2, 5),
        "LeafPruning": ("leaves removed/week", 0, 6), "TrussPruning": ("fruits per truss", 3, 8),
        "CalciumNitrateFeed": ("mmol/L", 0, 10), "MagnesiumFeed": ("mmol/L", 0, 4), "BumblebeeHives": ("hives/ha", 0, 20),
        "DayNightDifferential": ("°C", -4, 10), "RootZoneTemp": ("°C", 14, 26), "AirCirculationFans": ("% fan speed", 0, 100),
    },
    "clinical": {
        "FluidInfusionRate": ("mL/h", 0, 250), "SupplementalOxygen": ("L/min", 0, 15), "HeadOfBedAngle": ("°", 0, 60),
        "TurningFrequency": ("turns/day", 0, 12), "IncentiveSpirometry": ("sessions/day", 0, 12),
        "OralFluidTarget": ("L/day", 0.5, 3), "ProteinSupplementRate": ("g/day", 0, 100), "CompressionStockings": ("mmHg", 0, 40),
        "LightingSchedule": ("h of bright light/day", 0, 16), "NurseCheckFrequency": ("checks/day", 2, 24),
        "VitalSignsInterval": ("h", 1, 12), "WoundDressingChange": ("changes/week", 0, 14), "CatheterDays": ("days", 0, 14),
        "FootPumpUse": ("h/day", 0, 24), "OxygenWeanRate": ("L/min per day", 0, 3), "FluidBalanceTarget": ("mL/day", -1500, 1500),
        "MealTiming": ("meals/day", 1, 6), "SittingOutTime": ("h/day", 0, 10), "RoomTemperature": ("°C", 18, 26),
        "EducationSessions": ("sessions/week", 0, 7),
    },
    "semiconductor": {
        "EtchPower": ("W", 100, 2000), "DepositionRate": ("nm/min", 0, 200), "ChamberSeasoning": ("seasoning wafers", 0, 25),
        "WaferCoolDelay": ("s", 0, 120), "ResistThickness": ("nm", 300, 2000), "DevelopTime": ("s", 20, 120),
        "HmdsPrime": ("s", 0, 90), "EndpointOverEtch": ("%", 0, 60), "RfMatchSetting": ("pF", 50, 500),
        "ElectrostaticChuckVoltage": ("V", 0, 2000), "PurgeGasFlow": ("sccm", 0, 5000), "LoadLockPumpTime": ("s", 10, 180),
        "ImplantTilt": ("°", 0, 10), "AnnealRampRate": ("°C/s", 1, 100), "CmpDownforce": ("psi", 1, 7),
        "PlatenSpeed": ("rpm", 30, 120), "DiWaterResistivity": ("MΩ·cm", 10, 18.2), "StockerDwell": ("h", 0, 48),
        "OverlayCorrection": ("nm", 0, 20), "MetrologySampling": ("% of wafers", 0, 100),
    },
    "aquaculture": {
        "FeedRate": ("% body weight/day", 0, 3), "StockingDensity": ("kg/m³", 5, 60), "OxygenConeFlow": ("L/min", 0, 50),
        "DegassingRate": ("% of flow", 0, 100), "FeedingFrequency": ("meals/day", 1, 12), "FeedSpread": ("m radius", 1, 10),
        "LightIntensity": ("lux", 0, 1000), "CurrentSpeed": ("body lengths/s", 0, 2), "GradingFrequency": ("gradings/month", 0, 4),
        "MakeupWaterRate": ("%/day", 0, 30), "SludgeRemoval": ("backwashes/h", 0, 12), "ProteinSkimmer": ("% of flow", 0, 50),
        "OzoneInjection": ("g O₃/kg feed", 0, 25), "AlkalinityDosing": ("kg NaHCO₃/day", 0, 20),
        "TankCleaningInterval": ("days", 1, 30), "NetChangeInterval": ("days", 7, 90), "HandlingEvents": ("events/month", 0, 8),
        "SmoltTransferSize": ("g", 50, 200), "WaterExchangeTurnover": ("exchanges/h", 0.5, 4),
    },
    "battery": {
        "ChargeRate": ("C", 0.1, 3), "FormationCurrent": ("C", 0.02, 1), "MixingTime": ("min", 30, 240),
        "SlurryViscosity": ("Pa·s", 1, 10), "DryingOvenTemp": ("°C", 60, 150), "CalenderGap": ("µm", 50, 200),
        "NotchingSpeed": ("strokes/min", 60, 400), "StackAlignment": ("µm tolerance", 50, 500),
        "DegassingVacuum": ("kPa absolute", 1, 50), "SealTemperature": ("°C", 160, 220), "AgingTemperature": ("°C", 20, 60),
        "AgingDuration": ("days", 1, 21), "FormationTemperature": ("°C", 20, 55), "UpperCutoffVoltage": ("V", 4.0, 4.4),
        "BinderContent": ("wt%", 1, 6), "ConductiveCarbon": ("wt%", 0.5, 5), "ElectrodePorosity": ("%", 20, 45),
        "CurrentCollectorThickness": ("µm", 6, 20), "TapeApplication": ("mm tape width", 0, 10),
        "BurrInspection": ("% of cells inspected", 0, 100),
    },
    "catalysis": {
        "FeedFlowRate": ("t/h", 10, 120), "InletTemperature": ("°C", 280, 420), "HydrogenPartialPressure": ("bar", 20, 120),
        "MakeupHydrogenRate": ("kNm³/h", 0, 30), "SulfidingLevel": ("ppm S", 0, 500), "QuenchGasRate": ("kNm³/h", 0, 20),
        "FlareLoad": ("t/h", 0, 10), "ReboilerDuty": ("MW", 0, 15), "RefluxRatio": ("reflux ratio", 0.5, 5),
        "SeparatorLevel": ("% of drum", 10, 90), "WashWaterRate": ("m³/h", 0, 15), "AmineCirculation": ("m³/h", 0, 200),
        "CompressorSpeed": ("rpm", 6000, 12000), "FeedPreheatBypass": ("% bypassed", 0, 50),
        "StripperSteam": ("kg/m³ of feed", 0, 60), "ProductDrawRate": ("m³/h", 0, 100), "FlushOilRate": ("m³/h", 0, 20),
        "CatalystLoadingDensity": ("kg/m³", 600, 900), "PressureControlValve": ("bar", 20, 150),
    },
    "fermentation": {
        "SugarFeedRate": ("g/L/h", 0, 10), "FermentTemp": ("°C", 25, 38), "PitchRate": ("10⁶ cells/mL", 1, 30),
        "AerationAtPitch": ("ppm O₂", 0, 15), "RecirculationRate": ("tank volumes/day", 0, 6),
        "CoolingJacketFlow": ("m³/h", 0, 60), "MashpH": ("pH", 4.5, 6.5), "LiquefactionTemp": ("°C", 80, 95),
        "BackSetRatio": ("%", 0, 50), "SolidsLoading": ("% dry solids", 25, 38), "GlucoamylaseRate": ("kg/t grain", 0, 1),
        "UreaFeed": ("ppm N", 0, 800), "PropagationTime": ("h", 6, 16), "PropagatorTemp": ("°C", 28, 36),
        "FillTime": ("h", 6, 20), "DropTiming": ("h after fill", 40, 70), "CipCausticStrength": ("% NaOH", 0.5, 4),
        "BeerWellLevel": ("%", 10, 90), "FlashCoolerTemp": ("°C", 28, 40), "ScrubberWaterFlow": ("m³/h", 0, 20),
    },
}

DOSE_UNITS: Dict[str, Dict[str, tuple]] = {
    "bioprocess": {
        "ChelatorDosing": ("µM", 200), "AntioxidantDosing": ("mM", 10), "EndotoxinScrubber": ("passes/day", 6),
        "StabilizerAdditive": ("g/L", 5), "CarrierProtein": ("g/L", 10), "ButyrateSpike": ("mM", 5),
        "ZincSupplement": ("µM", 100), "LipidConcentrate": ("mL/L", 10), "InsulinSupplement": ("mg/L", 20),
        "CysteineSpike": ("mM", 8), "ManganeseSpike": ("µM", 50), "DextranSulfate": ("mg/L", 200),
        "HypothermiaProtectant": ("mM", 20), "UridineFeed": ("mM", 10), "GalactoseSpike": ("g/L", 20),
        "SeleniumSupplement": ("µg/L", 50), "ProteaseInhibitorCocktail": ("mL/L", 5),
    },
    "datacenter": {
        "Dehumidifier": ("L/h moisture removed", 20), "ConnectorReseat": ("connectors reseated/week", 200),
        "PriorityBoost": ("% of jobs boosted", 100), "KernelPatch": ("% of nodes patched", 100),
        "DriverRollback": ("% of nodes rolled back", 100), "CableSwap": ("cables replaced/week", 100),
        "ThermalPasteRework": ("servers reworked/week", 50), "MemoryScrub": ("scrub passes/day", 24),
        "QueueRebalance": ("rebalances/day", 48), "FilterReplacement": ("filters changed/month", 40),
        "BmcReset": ("resets/week", 20), "SpanningTreeTune": ("% of switches tuned", 100),
        "OpticsClean": ("ports cleaned/week", 400), "ConfigRollback": ("% of nodes reverted", 100),
        "BurstCredits": ("I/O credits/s", 1000),
    },
    "watertreatment": {
        "CorrosionInhibitor": ("mg/L as P", 3), "PhStabilizer": ("mg/L", 40), "ClarifyingPolymer": ("mg/L", 2),
        "PermanganateFeed": ("mg/L", 4), "PowderedCarbon": ("mg/L", 50), "SodaAsh": ("mg/L", 60),
        "CarbonDioxideFeed": ("mg/L", 30), "Polyphosphate": ("mg/L", 4), "SodiumSilicate": ("mg/L", 25),
        "FilterAid": ("mg/L", 0.5), "Dechlorinator": ("mg/L", 10), "BiocideShock": ("mg/L", 50),
        "AmmoniaFeed": ("mg/L", 1.5), "LimeSlurry": ("mg/L", 80), "IonExchangeRegen": ("regenerations/week", 14),
    },
    "agronomy": {
        "ChelatedMicronutrient": ("g/m²", 5), "SoilAcidifier": ("kg S/ha", 500), "BiofungicideDrench": ("mL/m²", 10),
        "FoliarPigment": ("mL/L of spray", 5), "PostharvestWax": ("mL/kg fruit", 2), "SeaweedExtract": ("mL/L", 5),
        "HumicAcid": ("kg/ha", 30), "SiliconSupplement": ("mmol/L", 2), "MycorrhizalInoculant": ("g/plant", 10),
        "AminoAcidFoliar": ("mL/L", 4), "BoronFoliar": ("g/L", 1), "WettingAgentAg": ("mL/m²", 3),
        "SulfurDust": ("kg/ha", 30), "NeemOil": ("mL/L", 10), "CopperFungicide": ("g/L", 3),
        "SalicylicAcidSpray": ("mM", 2), "MolassesDrench": ("mL/m²", 20),
    },
    "clinical": {
        "TargetedAntimicrobial": ("mg/kg/day", 30), "ElectrolyteRepletion": ("mmol/day", 120),
        "Antipyretic": ("mg/day", 4000), "Analgesic": ("mg morphine-equivalent/day", 60),
        "ProtonPumpInhibitor": ("mg/day", 80), "Anticoagulant": ("mg/day", 80), "Laxative": ("g/day", 34),
        "Antiemetic": ("mg/day", 24), "VitaminD": ("IU/day", 4000), "IronSupplement": ("mg/day", 200),
        "Probiotic": ("billion CFU/day", 50), "ZincTherapy": ("mg/day", 50), "Melatonin": ("mg/night", 10),
        "Bronchodilator": ("puffs/day", 16), "Diuretic": ("mg/day", 160), "Corticosteroid": ("mg/day", 60),
    },
    "semiconductor": {
        "PrecleanStep": ("min", 10), "GetteringAnneal": ("min at 1000 °C", 60), "EdgeExclusionTrim": ("mm", 5),
        "MegasonicRinse": ("min", 10), "ChelatingRinse": ("min", 10), "HfDip": ("s", 60), "OzoneWaterClean": ("min", 10),
        "ScClean1": ("min", 10), "PlasmaAsh": ("s", 120), "ForminggasAnneal": ("min", 60), "SurfactantRinse": ("min", 5),
        "ChamberWetClean": ("cleans/week", 7), "DummyWaferRun": ("wafers/lot", 10), "AntistaticIonizer": ("ionizer bars", 8),
        "CorrosionInhibitorBath": ("mM BTA", 10), "PostCmpBrush": ("s", 120),
    },
    "aquaculture": {
        "AntiparasiticBath": ("mg/L", 50), "CarbonFiltration": ("% of flow through carbon", 100),
        "Aeration": ("kW blower power", 30), "AppetiteStimulant": ("g/kg feed", 5), "SkinConditioner": ("mL/m³", 10),
        "SaltBath": ("g/L", 30), "HydrogenPeroxideBath": ("mg/L", 1500), "VitaminCSupplement": ("mg/kg feed", 1000),
        "Immunostimulant": ("g/kg feed", 2), "ProbioticFeed": ("g/kg feed", 5), "AstaxanthinFeed": ("mg/kg feed", 80),
        "FormalinBath": ("mg/L", 250), "NucleotideFeed": ("g/kg feed", 5), "ChelatorDose": ("mL/m³", 20),
        "EnzymeAdditive": ("g/kg feed", 2), "OmegaOilTopdress": ("% of feed", 5), "BacteriophageTreatment": ("10⁶ PFU/mL", 100),
    },
    "battery": {
        "DryRoomControl": ("kW desiccant drying", 50), "ElectrolyteAdditive": ("wt%", 3), "SlowFormation": ("h at low current", 48),
        "WettingAgent": ("wt%", 2), "ConditioningCycle": ("conditioning cycles", 5), "VinyleneCarbonate": ("wt%", 3),
        "FluoroethyleneCarbonate": ("wt%", 10), "LiPO2F2": ("wt%", 2), "SultoneAdditive": ("wt%", 3),
        "ScavengerAdditive": ("wt%", 1), "BoronAdditive": ("wt%", 2), "SurfaceCoatingAlumina": ("nm coating", 20),
        "PrelithiationAgent": ("wt%", 5), "FlameRetardant": ("wt%", 15), "RedoxShuttle": ("wt%", 5),
        "NitrileAdditive": ("wt%", 3), "SilaneTreatment": ("g/m²", 2),
    },
    "catalysis": {
        "GuardBedAdsorbent": ("% of feed through guard bed", 100), "RegenerationCycle": ("regenerations/month", 4),
        "FeedPurification": ("% of feed polished", 100), "PromoterSpike": ("ppm", 200), "DiluentCoFeed": ("% of feed", 30),
        "DmdsInjection": ("kg/h", 200), "AntifoulantDose": ("ppm", 100), "CorrosionInhibitorCat": ("ppm", 50),
        "ChlorideGuard": ("% of feed treated", 100), "AntioxidantInhibitor": ("ppm", 50), "DemulsifierDose": ("ppm", 50),
        "NeutralizerDose": ("ppm", 100), "SiliconTrap": ("% of feed treated", 100), "ArsenicTrap": ("% of feed treated", 100),
        "WaxDispersant": ("ppm", 500), "MetalDeactivator": ("ppm", 20), "OxygenScavenger": ("ppm", 50),
    },
    "fermentation": {
        "AntimicrobialHopExtract": ("ppm", 20), "pHBuffering": ("g/hL", 50), "MicronutrientSpike": ("ppm", 50),
        "Antifoam": ("ppm", 200), "ClarifyingFinings": ("g/hL", 10), "ZincSulfate": ("ppm Zn", 1),
        "YeastNutrientBlend": ("g/hL", 50), "Penicillin": ("ppm", 5), "ChlorineDioxide": ("ppm", 50),
        "ProteaseEnzyme": ("kg/t", 0.5), "CellulaseEnzyme": ("kg/t", 1), "MagnesiumSulfate": ("g/hL", 20),
        "VirginiamycinDose": ("ppm", 3), "PhytaseEnzyme": ("kg/t", 0.5), "ErgosterolSupplement": ("mg/L", 50),
        "PotassiumMetabisulfite": ("ppm SO₂", 150), "HopAcidsBoost": ("ppm", 30),
    },
}

# Mechanism-marker polarity. In v10 the marker is drawn to match the direction it will move: in a world where the
# hidden cause HURTS the outcome the marker rises with the problem, so it must be a "bad" marker (higher = worse); in
# an AND-gate (synergy) world it rises as things improve, so it must be a "good" marker. v9 drew the name blind to
# this (8 of 18 checked v10 worlds: serum lactate rising as patients recover). GOOD = higher is healthier.
GOOD_MARKERS = {"DewPointMargin", "LeafGreenness", "TissueNutrientAssay", "CoulombicEfficiency", "CatalystSurfaceArea",
                "ViabilityStain", "FiltrationIntegrityIndex", "TissuePerfusionIndex", "FilmUniformityIndex",
                "GillPerfusionIndex", "ViableCellDensity"}
# extra BAD markers (higher = worse), so every skin has >= 3 (proxy + competing cause-2 marker + spare)
EXTRA_BAD_MARKERS: Dict[str, List[Dict[str, object]]] = {
    "bioprocess": [_p("AggregatePct", "aggregate percent", "high-molecular-weight species"),
                   _p("HostCellProtein", "host cell protein", "HCP level")],
    "datacenter": [_p("CorrectableEccErrors", "correctable ECC errors", "memory error rate"),
                   _p("ThermalThrottleEvents", "thermal throttle events", "throttling count"),
                   _p("PacketDropRate", "packet drop rate", "dropped packets")],
    "watertreatment": [_p("HeterotrophicPlateCount", "heterotrophic plate count", "HPC bacteria"),
                       _p("DissolvedIron", "dissolved iron", "iron residual")],
    "agronomy": [_p("LeafNecrosisScore", "leaf necrosis score", "leaf scorch"),
                 _p("RootBrowningIndex", "root browning index", "root discoloration"),
                 _p("StressEthylene", "ethylene emission", "stress ethylene")],
    "clinical": [_p("Procalcitonin", "procalcitonin", "PCT level"), _p("WhiteCellCount", "white cell count", "leukocyte count")],
    "semiconductor": [_p("ParticleAdders", "particle adders", "wafer particle count"),
                      _p("SheetResistanceDrift", "sheet resistance drift", "Rs drift")],
    "aquaculture": [_p("GillMucusScore", "gill mucus score", "gill mucus"), _p("FinErosionIndex", "fin erosion index", "fin damage")],
    "battery": [_p("SelfDischargeRate", "self-discharge rate", "voltage decay"), _p("CellSwellingIndex", "cell swelling index", "pouch swelling"),
                _p("LithiumPlatingIndex", "lithium plating index", "plating signature")],
    "catalysis": [_p("CokeOnCatalyst", "carbon on catalyst", "coke content"), _p("SulfurSlip", "sulfur slip", "product sulfur"),
                  _p("DeactivationRate", "deactivation rate", "activity decline rate")],
    "fermentation": [_p("GlycerolTiter", "glycerol titer", "glycerol level"), _p("LacticAcidTiter", "lactic acid titer", "lactic acid level"),
                     _p("AceticAcidTiter", "acetic acid titer", "acetic acid level")],
}

# extra GOOD markers (higher = healthier), for AND-gate worlds where the marker rises as the outcome recovers
EXTRA_GOOD_MARKERS: Dict[str, List[Dict[str, object]]] = {
    "bioprocess": [_p("SpecificProductivity", "specific productivity", "qP"),
                   _p("CellViabilityPct", "cell viability", "viability percent")],
    "datacenter": [_p("LinkMarginDb", "optical link margin", "link budget margin"),
                   _p("ThermalHeadroom", "thermal headroom", "temperature margin")],
    "watertreatment": [_p("DisinfectantStability", "disinfectant stability index", "residual stability"),
                       _p("PipeWallPassivation", "pipe passivation index", "scale stability index")],
    "agronomy": [_p("RootActivityIndex", "root activity index", "root respiration index")],
    "clinical": [_p("GripStrength", "grip strength", "muscle strength score"),
                 _p("AppetiteScore", "appetite score", "oral intake score")],
    "semiconductor": [_p("CarrierLifetime", "minority carrier lifetime", "carrier lifetime"),
                      _p("GateOxideIntegrity", "gate oxide integrity", "GOI pass rate")],
    "aquaculture": [_p("FeedConversionEfficiency", "feed conversion efficiency", "feed utilization"),
                    _p("HemoglobinLevel", "blood hemoglobin", "hemoglobin level")],
    "battery": [_p("EnergyEfficiency", "energy efficiency", "round-trip efficiency"),
                _p("ActiveLithiumInventory", "active lithium inventory", "lithium inventory")],
    "catalysis": [_p("CatalystActivityIndex", "catalyst activity index", "relative activity"),
                  _p("MetalDispersion", "metal dispersion", "active metal dispersion")],
    "fermentation": [_p("BuddingIndex", "budding index", "yeast budding rate"),
                     _p("YeastVitality", "yeast vitality", "acidification power")],
}
GOOD_MARKERS = GOOD_MARKERS | {e["name"] for v in EXTRA_GOOD_MARKERS.values() for e in v}

# Units of MEASURED signals (outcome, inert signals, decoys, markers). Same linear display map as controls
# (display = lo + internal/100 * (hi - lo)); internal values of these signals sit mostly in 0-100. "__outcome__" is the
# skin's outcome; its map keeps >= 0.4 display units per internal point so effect sizes stay readable.
MEASURE_UNITS: Dict[str, Dict[str, tuple]] = {
    "bioprocess": {
        "__outcome__": ("mg/L", 2500, 5000), "AntifoamLevel": ("ppm", 0, 200), "GlucoseFeedRate": ("g/L/day", 0, 10),
        "GlutamineConc": ("mM", 0, 8), "LactateConc": ("g/L", 0, 4), "AmmoniaConc": ("mM", 0, 12),
        "Osmolality": ("mOsm/kg", 260, 420), "CO2Level": ("mmHg", 20, 160), "VesselPressure": ("mbar", 0, 300),
        "AgitationRate": ("rpm", 50, 250), "GasSpargeRate": ("vvm", 0, 0.1), "CoolingWaterTemp": ("°C", 4, 20),
        "Temperature": ("°C", 33, 38), "pH": ("pH", 6.6, 7.4), "DissolvedOxygen": ("% air saturation", 0, 80),
        "ViableCellDensity": ("10⁶ cells/mL", 0, 30), "BrothTurbidity": ("NTU", 0, 400), "LDHRelease": ("U/L", 0, 1500),
        "FragmentedProteinPct": ("%", 0, 10), "SpecificProductivity": ("pg/cell/day", 0, 60),
        "CellViabilityPct": ("%", 50, 100), "AggregatePct": ("%", 0, 8), "HostCellProtein": ("ppm", 0, 2000),
    },
    "datacenter": {
        "__outcome__": ("jobs/h", 1000, 2000), "CpuClock": ("GHz", 2.0, 3.8), "GpuTemp": ("°C", 40, 90),
        "DiskIoWait": ("%", 0, 30), "MemoryUtilization": ("%", 0, 100), "JobQueueDepth": ("jobs", 0, 500),
        "AmbientRoomHumidity": ("% RH", 20, 70), "UpsBatteryLevel": ("%", 50, 100), "CacheHitRate": ("%", 60, 100),
        "ContainerCount": ("containers", 0, 2000), "FanSpeed": ("rpm", 2000, 12000), "RackInletTemp": ("°C", 15, 35),
        "PduLoad": ("kW", 0, 20), "DewPointMargin": ("°C", 0, 15), "NicErrorCounter": ("errors/min", 0, 200),
        "CorrectableEccErrors": ("errors/h", 0, 100), "ThermalThrottleEvents": ("events/h", 0, 50),
        "PacketDropRate": ("per 1000 packets", 0, 10), "LinkMarginDb": ("dB", 0, 10), "ThermalHeadroom": ("°C", 0, 30),
    },
    "watertreatment": {
        "__outcome__": ("clarity points", 40, 140), "FluorideLevel": ("mg/L", 0, 1.5), "WaterTemp": ("°C", 5, 25),
        "ContactTime": ("min", 0, 120), "PumpCycles": ("cycles/h", 0, 12), "TankLevel": ("% full", 0, 100),
        "HardnessCaCO3": ("mg/L as CaCO₃", 0, 300), "UVDose": ("mJ/cm²", 0, 100), "FilterHeadloss": ("m", 0, 3),
        "BackwashRate": ("m/h", 0, 60), "PressureReading": ("kPa", 200, 600), "ResidualChlorine": ("mg/L", 0, 2),
        "FiltrationIntegrityIndex": ("% integrity", 80, 100), "TurbidityNTU": ("NTU", 0, 5),
        "MetalColorimetry": ("mg/L iron + manganese", 0, 1), "HeterotrophicPlateCount": ("CFU/mL", 0, 500),
        "DissolvedIron": ("mg/L", 0, 1), "DisinfectantStability": ("% residual retained", 50, 100),
        "PipeWallPassivation": ("% passivated", 0, 100),
    },
    "agronomy": {
        "__outcome__": ("g/m²/week", 800, 1600), "SoilPhosphorus": ("mg/kg", 0, 120), "SoilPotassium": ("mg/kg", 50, 400),
        "RowSpacing": ("cm", 40, 160), "MulchDepth": ("cm", 0, 15), "WindExposure": ("m/s", 0, 5),
        "CO2Enrichment": ("ppm", 400, 1200), "PruningIntensity": ("% foliage removed", 0, 40),
        "PollinatorActivity": ("visits/flower/day", 0, 10), "SoilOrganicMatter": ("%", 0, 10), "DrainageRate": ("mm/h", 0, 50),
        "SeedTreatment": ("g/kg seed", 0, 5), "GrowthRegulator": ("ppm", 0, 100), "NightTemperature": ("°C", 10, 22),
        "SoilMoisture": ("% volumetric", 10, 60), "CanopyTemp": ("°C", 15, 35), "LeafGreenness": ("SPAD", 20, 60),
        "TissueNutrientAssay": ("% of sufficiency", 40, 140), "LeafNecrosisScore": ("% leaf area", 0, 40),
        "RootBrowningIndex": ("% roots browned", 0, 60), "StressEthylene": ("nL/g/h", 0, 20),
        "RootActivityIndex": ("% of healthy", 40, 140),
    },
    "clinical": {
        "__outcome__": ("points", 40, 140), "BloodPressure": ("mmHg systolic", 90, 170), "RespiratoryRate": ("breaths/min", 10, 30),
        "UrineOutput": ("mL/h", 0, 120), "PainScore": ("on a 0-10 scale", 0, 10), "MobilizationLevel": ("steps/day", 0, 6000),
        "SleepHours": ("h/night", 0, 10), "CaloricIntake": ("kcal/day", 800, 2800), "VisitorFrequency": ("visits/day", 0, 6),
        "RoomOccupancy": ("patients per bay", 1, 6), "PhysioTherapyDose": ("min/day", 0, 90),
        "SedationLevel": ("on a 0-5 scale", 0, 5), "BloodGlucose": ("mmol/L", 4, 14), "AmbientNoise": ("dB", 30, 70),
        "BodyTemperature": ("°C", 35.5, 39.5), "HeartRate": ("bpm", 50, 120), "TissuePerfusionIndex": ("perfusion index", 0, 10),
        "CReactiveProtein": ("mg/L", 0, 200), "SerumLactate": ("mmol/L", 0.5, 6), "Procalcitonin": ("ng/mL", 0, 5),
        "WhiteCellCount": ("×10⁹/L", 3, 20), "GripStrength": ("kg", 5, 40), "AppetiteScore": ("% of meals eaten", 0, 100),
    },
    "semiconductor": {
        "__outcome__": ("good die per wafer", 400, 600), "SpinSpeed": ("rpm", 1000, 5000), "BakeTemperature": ("°C", 90, 130),
        "ExposureDose": ("mJ/cm²", 10, 40), "GasFlowArgon": ("sccm", 0, 200), "HumidityFab": ("% RH", 35, 50),
        "VibrationLevel": ("µm/s", 0, 50), "SlurryConcentration": ("wt%", 0, 15), "PadConditioning": ("% duty", 0, 100),
        "RinseTime": ("s", 0, 120), "HandlerThroughput": ("wafers/h", 50, 250),
        "ChamberFlowUniformity": ("% non-uniformity", 0, 10), "BacksideCooling": ("Torr He", 0, 20),
        "ScannerFocus": ("nm focus offset", -100, 100), "ChamberPressure": ("mTorr", 0, 200), "StageTemperature": ("°C", 15, 80),
        "FilmUniformityIndex": ("% uniformity", 90, 100), "DefectDensity": ("defects/cm²", 0, 1), "LeakageCurrent": ("nA", 0, 100),
        "ParticleAdders": ("particles/wafer", 0, 200), "SheetResistanceDrift": ("% drift", 0, 10),
        "CarrierLifetime": ("µs", 0, 2000), "GateOxideIntegrity": ("% pass", 80, 100),
    },
    "aquaculture": {
        "__outcome__": ("kg per tank", 500, 1500), "WaterpH": ("pH", 6.5, 8.5), "Salinity": ("ppt", 0, 35),
        "WaterTempAqua": ("°C", 6, 18), "AmmoniaLevel": ("mg/L TAN", 0, 3), "NitriteLevel": ("mg/L", 0, 1),
        "FlowThroughRate": ("L/s", 0, 50), "PhotoperiodHours": ("h light/day", 8, 24), "PelletSize": ("mm", 1, 12),
        "CalciumHardness": ("mg/L", 0, 200), "TankDepth": ("m", 1, 4), "BiofilterLoad": ("g TAN/m²/day", 0, 2),
        "UVSterilizer": ("mJ/cm²", 0, 80), "FeedProteinPct": ("% protein", 30, 55), "TankTurbidity": ("NTU", 0, 20),
        "SurfaceActivity": ("% of fish at surface", 0, 60), "GillPerfusionIndex": ("% of healthy", 40, 140),
        "GillHistologyScore": ("lesion score 0-5", 0, 5), "PlasmaCortisol": ("ng/mL", 0, 200),
        "GillMucusScore": ("score 0-5", 0, 5), "FinErosionIndex": ("% of fins eroded", 0, 60),
        "FeedConversionEfficiency": ("kg gain per kg feed", 0.4, 1.4), "HemoglobinLevel": ("g/dL", 4, 12),
    },
    "battery": {
        "__outcome__": ("mAh retained", 3000, 4000), "CoatingThickness": ("µm", 40, 120), "CalendarAge": ("days", 0, 365),
        "PressStackForce": ("kPa", 0, 300), "TabWeldQuality": ("N pull strength", 10, 60),
        "AmbientHumidityBatt": ("ppm H₂O", 0, 300), "DischargeDepth": ("%", 20, 100), "RestPeriod": ("h", 0, 48),
        "ElectrolyteFillVolume": ("g", 3, 7), "CoolingPlateTemp": ("°C", 15, 40), "CrimpPressure": ("kN", 5, 25),
        "AnodeExcessRatio": ("N/P ratio", 1.0, 1.3), "CyclingTemperature": ("°C", 15, 45), "VentPressure": ("kPa", 0, 50),
        "SurfaceTemperature": ("°C", 20, 50), "PackVoltage": ("V", 3.4, 4.2), "CoulombicEfficiency": ("%", 99.0, 100.0),
        "ImpedanceSpectrum": ("mΩ", 10, 60), "SelfDischargeRate": ("mV/day", 0, 5),
        "CellSwellingIndex": ("% thickness growth", 0, 10), "LithiumPlatingIndex": ("µm plated", 0, 5),
        "EnergyEfficiency": ("%", 85, 98), "ActiveLithiumInventory": ("% of initial", 80, 100),
    },
    "catalysis": {
        "__outcome__": ("% conversion", 60, 100), "RecycleRatio": ("recycle ratio", 0, 5), "CoolantFlow": ("m³/h", 0, 100),
        "BedHeight": ("m", 2, 12), "ParticleSizeCat": ("mm", 1, 5), "InertDiluent": ("% of bed", 0, 40),
        "PurgeRate": ("kNm³/h", 0, 5), "SteamToCarbon": ("mol/mol", 1.5, 4), "RecycleGasPurity": ("% H₂", 70, 95),
        "PreheaterDuty": ("MW", 0, 20), "PressureDropBed": ("bar", 0, 5), "FeedMoisture": ("ppm", 0, 500),
        "AgitationCat": ("rpm", 0, 600), "QuenchFlow": ("kNm³/h", 0, 20), "ReactorPressure": ("bar", 20, 120),
        "JacketTemperature": ("°C", 250, 400), "CatalystSurfaceArea": ("m²/g", 100, 250), "EffluentByproduct": ("wt%", 0, 5),
        "CokeOnCatalyst": ("wt% carbon", 0, 15), "SulfurSlip": ("ppm S", 0, 50), "DeactivationRate": ("°C/month", 0, 10),
        "CatalystActivityIndex": ("% of fresh", 40, 110), "MetalDispersion": ("%", 10, 60),
    },
    "fermentation": {
        "__outcome__": ("L per tonne of grain", 360, 440), "DissolvedO2Ferm": ("ppm", 0, 8), "AgitationFerm": ("rpm", 0, 200),
        "VesselPressureFerm": ("bar", 0, 2), "NitrogenSource": ("ppm FAN", 50, 400), "InoculationVolume": ("% v/v", 0, 10),
        "GravityInitial": ("°Plato", 10, 30), "MashTemp": ("°C", 60, 90), "WaterHardnessFerm": ("mg/L as CaCO₃", 0, 300),
        "YeastGeneration": ("generations", 0, 10), "TankGeometry": ("height:diameter", 1, 4), "CarbonationLevel": ("vol CO₂", 0, 3),
        "ClarifierDose": ("g/hL", 0, 10), "ChillRate": ("°C/h", 0, 10), "BrothFoaming": ("cm of foam", 0, 60),
        "CO2OffgasRate": ("kg/h", 0, 500), "ViabilityStain": ("% viable", 40, 100), "OrganicAcidTiter": ("g/L", 0, 5),
        "GlycerolTiter": ("g/L", 0, 15), "LacticAcidTiter": ("g/L", 0, 3), "AceticAcidTiter": ("g/L", 0, 2),
        "BuddingIndex": ("% budding", 0, 40), "YeastVitality": ("ΔpH acidification", 0, 2),
    },
}

# Story pieces (v10 per-world situation report; sampler._build_story_v10). The facility and the trigger are drawn per
# world; neither names a control or a measured signal, so the report never points at a cause.
STORY: Dict[str, Dict[str, List[str]]] = {
    "bioprocess": {
        "facility": ["a 2,000 L mammalian-cell bioreactor producing a secreted therapeutic protein",
                     "a perfusion bioreactor train making a monoclonal antibody",
                     "a fed-batch cell-culture suite producing a recombinant enzyme"],
        "trigger": ["a maintenance shutdown in which a line fitting was replaced and probes were recalibrated",
                    "a switch to a new lot of basal medium", "a change of raw-material supplier for several feeds",
                    "the commissioning of a new single-use assembly"],
        "unit": ["batches", "production runs", "campaigns"]},
    "datacenter": {
        "facility": ["a GPU training cluster in a raised-floor data hall", "a multi-tenant compute cluster serving batch jobs",
                     "a high-performance computing hall running scientific workloads"],
        "trigger": ["a maintenance window in which cooling was serviced and hardware was reseated",
                    "a firmware and driver roll-out", "a change in the tenant workload mix", "a power-distribution upgrade"],
        "unit": ["job windows", "scheduling cycles", "workload batches"]},
    "watertreatment": {
        "facility": ["a municipal drinking-water treatment plant and its distribution network",
                     "a regional water utility serving several pressure zones", "a small-town treatment works and reservoir system"],
        "trigger": ["a main repair and the flushing that followed", "a change in the raw-water source blend",
                    "a seasonal demand shift", "the replacement of a filter bed"],
        "unit": ["sampling rounds", "pressure zones", "reporting periods"]},
    "agronomy": {
        "facility": ["a soil-grown greenhouse producing vine tomatoes", "a soil-bed glasshouse growing truss tomatoes",
                     "a commercial greenhouse growing cherry tomatoes in soil beds"],
        "trigger": ["a switch to a new irrigation-water source", "a change of substrate supplier",
                    "a new climate-control software release", "the replacement of the fertigation unit"],
        "unit": ["harvests", "growing cycles", "picks"]},
    "clinical": {
        "facility": ["a post-surgical step-down unit", "an orthopaedic recovery ward", "a general medical ward for older patients"],
        "trigger": ["a change to the unit's standing protocols", "a staffing and rota change",
                    "a new electronic order set", "a move to a refurbished ward"],
        "unit": ["admissions", "discharge cohorts", "patient cohorts"]},
    "semiconductor": {
        "facility": ["a 300 mm logic fab running a mature device", "a 200 mm power-device fab", "a memory fab running a legacy node"],
        "trigger": ["a preventive-maintenance window on several tools", "a chemical-supplier change",
                    "a recipe revision", "the qualification of a refurbished chamber"],
        "unit": ["lots", "wafer starts", "production shifts"]},
    "aquaculture": {
        "facility": ["a recirculating salmon smolt farm", "a land-based salmon grow-out facility",
                     "a flow-through salmon hatchery"],
        "trigger": ["a change in intake water", "a new feed formulation", "the arrival of a new stock cohort",
                    "a filtration-system upgrade"],
        "unit": ["cohorts", "tanks", "grow-out cycles"]},
    "battery": {
        "facility": ["a lithium-ion pouch-cell plant", "a cylindrical-cell production line", "a prismatic-cell pilot line"],
        "trigger": ["an HVAC change in the dry room", "an electrolyte-supplier switch",
                    "a new formation-cycler software release", "a change of separator lot"],
        "unit": ["production lots", "cell batches", "test batches"]},
    "catalysis": {
        "facility": ["a fixed-bed diesel hydrotreater", "a naphtha hydrotreating unit", "a fixed-bed gas-oil hydrotreater"],
        "trigger": ["a switch to a new feedstock supplier", "a routine catalyst reload", "a turnaround and restart",
                    "a change in upstream unit operations"],
        "unit": ["operating runs", "feed batches", "reactor runs"]},
    "fermentation": {
        "facility": ["a fuel-ethanol fermentation plant", "a dry-mill grain ethanol plant", "a grain-ethanol distillery's fermentation hall"],
        "trigger": ["a cleaning-protocol change", "a switch to a new yeast-propagation batch", "a new grain supplier",
                    "a cooling-system repair"],
        "unit": ["fermentations", "tanks", "batches"]},
}


def test_skins_v10() -> int:
    """Every skin has >= 18 process + >= 12 additive names; no name or alias repeats within a skin (including
    skins.py names / aliases). Returns the number of problems (0 = ok)."""
    from skins import SKINS
    bad = 0
    for skin, S in SKINS.items():
        if skin not in EXTRA:
            print(f"missing skin {skin}"); bad += 1; continue
        ex = EXTRA[skin]
        if len(ex["extra_process_pool"]) < 18 or len(ex["extra_additive_pool"]) < 12:
            print(f"{skin}: pools too small"); bad += 1
        names, aliases = {}, {}
        for pool, entries in list(S.items()) + list(ex.items()):
            if not pool.endswith("_pool"):
                continue
            for e in entries:
                n = e["name"]
                # skins.py itself reuses a few names across its own pools (e.g. a proxy that is also an inert var);
                # only collisions that involve a v10 name are errors here.
                if n in names and (pool.startswith("extra_") or names[n].startswith("extra_")):
                    print(f"{skin}: name {n} in {names[n]} and {pool}"); bad += 1
                names.setdefault(n, pool)
                for a in e["aliases"]:
                    al = a.lower()
                    if al in aliases and aliases[al] != n and (pool.startswith("extra_") or names.get(aliases[al], "").startswith("extra_")):
                        print(f"{skin}: alias '{a}' of {n} also used by {aliases[al]}"); bad += 1
                    aliases.setdefault(al, n)
    return bad


if __name__ == "__main__":
    b = test_skins_v10()
    print("ALL PASS" if b == 0 else f"{b} PROBLEM(S)")
    raise SystemExit(1 if b else 0)
